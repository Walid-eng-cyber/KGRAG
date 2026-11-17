"""Detailed graph extraction: read the whole document, look again, then check every fact.

The fast pipeline (corpus_graph, INGEST_MODE=fast) asks a model for triples one ~512-token chunk at a time.
This one spends more model time per document to get a more complete and more exact graph:

  1. extract  the whole document (or a large window of it) in view -> entities, and facts that each carry the
              quote, copied from the document, that states them
  2. glean    "here is what was found: what did you miss?" -> more entities and facts
  3. verify   every fact against its quote: the quote must be found in the document (checked in code), and the
              model must confirm the quote states the fact; anything else is dropped

Each kept fact is anchored to the chunk that holds its quote, so a graph answer cites the passage it came from.
Every call returns JSON; a reply that isn't JSON is asked for once more, then skipped (the pass yields nothing).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import config

_SAFE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")
VERIFY_BATCH = 25          # facts checked per verify call
EXTRACT_TOKENS = 4096      # output cap for the extract and glean calls
VERIFY_TOKENS = 1024


@dataclass
class Fact:
    source: str
    relation: str
    target: str
    quote: str
    chunk: str | None = None      # the chunk holding the quote (None: the quote isn't in the document)


@dataclass
class Extraction:
    entities: dict[str, str] = field(default_factory=dict)       # name -> type
    facts: list[Fact] = field(default_factory=list)
    stats: dict = field(default_factory=lambda: {
        "windows": 0, "extracted": 0, "gleaned": 0, "quote_not_found": 0, "rejected_by_check": 0,
        "kept": 0, "invalid": 0, "bad_json": 0, "calls": 0, "seconds": {"extract": 0.0, "glean": 0.0, "verify": 0.0}})


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def norm(text: str) -> str:
    """Lower case, straight quotes, single spaces: for matching a quote against the document."""
    t = text.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    return _WS.sub(" ", t).strip()


def find_chunk(quote: str, chunks: list[tuple[str, str]]) -> str | None:
    """The id of the chunk that contains `quote`: verbatim (ignoring case and spacing), else the chunk sharing
    at least 80% of the quote's words in order-free overlap (a model may drop a comma or a word)."""
    q = norm(quote).strip(" .\"'")
    if len(q) < 3:
        return None
    for cid, text in chunks:
        if q in norm(text):
            return cid
    words = _WORD.findall(q)
    if len(words) < 3:
        return None
    best, best_score = None, 0.0
    for cid, text in chunks:
        have = set(_WORD.findall(norm(text)))
        score = sum(w in have for w in words) / len(words)
        if score > best_score:
            best, best_score = cid, score
    return best if best_score >= 0.8 else None


def windows(chunks: list[tuple[str, str]], max_chars: int) -> list[list[tuple[str, str]]]:
    """Consecutive chunks grouped so each group's text stays under max_chars (a chunk is never split).
    A short document is one window: the model reads all of it at once."""
    out: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    size = 0
    for c in chunks:
        if cur and size + len(c[1]) > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c[1]) + 2
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def schema_text(schema: dict) -> str:
    rels = "\n".join(f"  {s} -[{r}]-> {t}" for s, r, t in schema["relations"])
    return f"ENTITY TYPES: {', '.join(schema['entities'])}\nALLOWED FACTS (source type -[RELATION]-> target type):\n{rels}"


JSON_SHAPE = ('{"entities":[{"name":"...","type":"..."}],'
              '"facts":[{"source":"...","relation":"...","target":"...","quote":"..."}]}')


def extract_system(schema: dict, hint: str) -> str:
    return (
        f"You build a knowledge graph from one document. {hint}\n\n{schema_text(schema)}\n\n"
        "Rules:\n"
        "- Extract every entity of the types above and every allowed fact that the document states explicitly. "
        "Be thorough: list every skill, role, organization, place, date-bearing item, amount and clause it names.\n"
        "- Name entities by their full name as written in the document (e.g. the person's full name, never "
        "'he', 'she', 'the candidate' or 'the company'). Use the same name every time you mention an entity.\n"
        "- Each fact's source and target must be entities in your list, with the types the relation allows.\n"
        "- Each fact needs a quote: a short passage copied word for word from the document that states it.\n"
        "- Don't infer, don't guess, don't add anything the document doesn't say.\n"
        f"Reply ONLY with JSON: {JSON_SHAPE}")


def glean_system(schema: dict, hint: str) -> str:
    return (extract_system(schema, hint) +
            "\n\nYou are given what was already extracted from this document. Read the document again and reply "
            "with ONLY what is missing: entities and facts the document states that are not in the list yet. "
            'If nothing is missing, reply {"entities":[],"facts":[]}.')


VERIFY_SYSTEM = (
    "You check facts extracted from a document. A fact is supported only if the document states it: the quote "
    "shows where, and the quote (read in the document) must say what the fact says, about the same entity. "
    "Reject a fact that is implied, guessed, about someone else, or has the wrong direction or relation.\n"
    'Reply ONLY with JSON: {"supported":[numbers of the supported facts]}')


def _listing(ext: Extraction) -> str:
    ents = "; ".join(f"{n} ({t})" for n, t in ext.entities.items())
    facts = "\n".join(f"- {f.source} -[{f.relation}]-> {f.target}" for f in ext.facts)
    return f"ENTITIES ALREADY FOUND: {ents or '(none)'}\nFACTS ALREADY FOUND:\n{facts or '(none)'}"


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def _ask_json(client, model: str, system: str, user: str, max_tokens: int, ext: Extraction) -> dict:
    """One JSON call (asked twice if the first reply isn't JSON). {} when no JSON came back."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    options = {"temperature": 0, "num_ctx": config.EXTRACT_NUM_CTX, "num_predict": max_tokens}
    for _ in range(2):
        ext.stats["calls"] += 1
        resp = client.chat(model=model, messages=messages, format="json", options=options)
        content = resp["message"]["content"] or ""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        messages = messages + [{"role": "assistant", "content": content[:2000]},
                               {"role": "user", "content": "That was not valid JSON. Reply with the JSON object only."}]
    ext.stats["bad_json"] += 1
    return {}


def _add(raw: dict, schema: dict, ext: Extraction, chunks: list[tuple[str, str]], counter: str) -> None:
    """Merge one model reply into `ext`, keeping only schema-valid entities and facts."""
    types = set(schema["entities"])
    allowed = {tuple(r) for r in schema["relations"]}
    by_lower = {n.lower(): n for n in ext.entities}

    for e in raw.get("entities") or []:
        if not isinstance(e, dict):
            continue
        name = _WS.sub(" ", str(e.get("name", ""))).strip()
        typ = str(e.get("type", "")).strip().upper()
        if not name or len(name) > 120 or typ not in types or not _SAFE.match(typ):
            ext.stats["invalid"] += 1
            continue
        if name.lower() not in by_lower:
            ext.entities[name] = typ
            by_lower[name.lower()] = name

    seen = {(f.source, f.relation, f.target) for f in ext.facts}
    for f in raw.get("facts") or []:
        if not isinstance(f, dict):
            continue
        src = by_lower.get(_WS.sub(" ", str(f.get("source", ""))).strip().lower())
        dst = by_lower.get(_WS.sub(" ", str(f.get("target", ""))).strip().lower())
        rel = str(f.get("relation", "")).strip().upper().replace(" ", "_")
        quote = _WS.sub(" ", str(f.get("quote", ""))).strip()
        if not src or not dst or src == dst or not _SAFE.match(rel) or not quote:
            ext.stats["invalid"] += 1
            continue
        ts, tt = ext.entities[src], ext.entities[dst]
        if (ts, rel, tt) not in allowed:
            if (tt, rel, ts) in allowed:          # the model wrote it backwards
                src, dst = dst, src
            else:
                ext.stats["invalid"] += 1
                continue
        if (src, rel, dst) in seen:
            continue
        seen.add((src, rel, dst))
        ext.facts.append(Fact(src, rel, dst, quote, find_chunk(quote, chunks)))
        ext.stats[counter] += 1


def _verify(client, model: str, title: str, window_text: str, facts: list[Fact], ext: Extraction) -> list[Fact]:
    """The facts the model confirms against the document (in batches). A failed check keeps the batch: the
    quotes were already found in the document, so an unreadable reply shouldn't erase them."""
    kept: list[Fact] = []
    for i in range(0, len(facts), VERIFY_BATCH):
        batch = facts[i:i + VERIFY_BATCH]
        listing = "\n".join(f"{n}. {f.source} -[{f.relation}]-> {f.target}  | quote: \u201c{f.quote}\u201d"
                            for n, f in enumerate(batch, 1))
        raw = _ask_json(client, model, VERIFY_SYSTEM,
                        f"Document \u201c{title}\u201d:\n\n{window_text}\n\nFACTS TO CHECK:\n{listing}", VERIFY_TOKENS, ext)
        ok = raw.get("supported") if isinstance(raw, dict) else None
        if not isinstance(ok, list):
            kept += batch
            continue
        nums = {int(n) for n in ok if isinstance(n, (int, float)) or (isinstance(n, str) and n.strip().isdigit())}
        for n, f in enumerate(batch, 1):
            if n in nums:
                kept.append(f)
            else:
                ext.stats["rejected_by_check"] += 1
    return kept


def extract_document(title: str, chunks: list[tuple[str, str]], schema: dict, hint: str,
                     client=None, model: str | None = None, progress=None) -> Extraction:
    """Run extract -> glean -> verify over a document's chunks ([(chunk_id, text)], in order)."""
    if client is None:
        from src.llm import extract_client
        client = extract_client("extract")
    model = model or config.EXTRACT_MODEL
    ext = Extraction()
    groups = windows(chunks, config.DETAILED_WINDOW_CHARS)
    ext.stats["windows"] = len(groups)

    for w, group in enumerate(groups, 1):
        text = "\n\n".join(t for _, t in group)
        doc = f"Document \u201c{title}\u201d" + (f" (part {w} of {len(groups)})" if len(groups) > 1 else "") + f":\n\n{text}"
        start = len(ext.facts)

        def step(name: str, pass_no: int) -> None:
            if progress:
                part = f", part {w}/{len(groups)}" if len(groups) > 1 else ""
                progress(f"Detailed extraction: pass {pass_no}/3 ({name}{part})", None)

        step("reading the whole document", 1)
        t0 = time.perf_counter()
        client.stage = "extract"
        _add(_ask_json(client, model, extract_system(schema, hint), doc, EXTRACT_TOKENS, ext), schema, ext, chunks, "extracted")
        ext.stats["seconds"]["extract"] += time.perf_counter() - t0

        step("what did we miss?", 2)
        t0 = time.perf_counter()
        client.stage = "glean"
        _add(_ask_json(client, model, glean_system(schema, hint), f"{doc}\n\n{_listing(ext)}", EXTRACT_TOKENS, ext),
             schema, ext, chunks, "gleaned")
        ext.stats["seconds"]["glean"] += time.perf_counter() - t0

        step("checking every fact", 3)
        t0 = time.perf_counter()
        client.stage = "verify"
        new = ext.facts[start:]
        found = [f for f in new if f.chunk]
        ext.stats["quote_not_found"] += len(new) - len(found)
        ext.facts[start:] = _verify(client, model, title, text, found, ext) if found else []
        ext.stats["seconds"]["verify"] += time.perf_counter() - t0

    ext.stats["kept"] = len(ext.facts)
    ext.stats["seconds"] = {k: round(v, 1) for k, v in ext.stats["seconds"].items()}
    return ext


def per_chunk(ext: Extraction, chunks: list[tuple[str, str]]) -> dict[str, tuple[list[dict], list[dict]]]:
    """The extraction in the shape the graph writer takes: chunk id -> (entities, relations).

    A fact is written under the chunk holding its quote, with its two entities. An entity is also linked to
    every chunk that names it, so text search and the graph meet on the same passages. An entity no fact uses
    and no chunk names is linked to the first chunk (it was still stated in the document)."""
    out: dict[str, tuple[list[dict], list[dict]]] = {cid: ([], []) for cid, _ in chunks}
    placed: dict[str, set[str]] = {cid: set() for cid, _ in chunks}

    def place(name: str, cid: str) -> None:
        if name not in placed[cid]:
            placed[cid].add(name)
            out[cid][0].append({"name": name, "label": ext.entities[name]})

    for f in ext.facts:
        if f.chunk not in out:
            continue
        place(f.source, f.chunk)
        place(f.target, f.chunk)
        out[f.chunk][1].append({"source": f.source, "target": f.target, "type": f.relation})
    for name in ext.entities:
        n = norm(name)
        hits = [cid for cid, text in chunks if n and n in norm(text)]
        for cid in hits:
            place(name, cid)
        if not hits and not any(name in placed[c] for c in placed) and chunks:
            place(name, chunks[0][0])
    return out
