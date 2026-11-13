"""Corpus-isolated graph + vector storage for uploaded documents.

Unlike the 10-K pipeline (LlamaIndex's Neo4j store, which merges entities by name
across everything), extracted triples are written here with an explicit corpus:

  (:DocChunk:Chunk {id, corpus, doc_id, ticker=doc_id, title, text})
  (:DocEntity:<TYPE> {corpus, name})        unique per (corpus, name)
  (chunk)-[:MENTIONS]->(entity)
  (entity)-[:<RELATION> {source_chunk, corpus}]->(entity)

so the same name in two corpora is two different nodes, and every query filters
on `corpus`. Vectors live in pgvector `chunk_embeddings` with a `corpus_id` column.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import date
from typing import Literal

import config

CHUNK_SIZE, CHUNK_OVERLAP, BATCH = 512, 64, 2
Progress = Callable[[str, float | None], None]
_SAFE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")

SCHEMA_CYPHER = [
    "CREATE CONSTRAINT doc_entity_key IF NOT EXISTS FOR (e:DocEntity) REQUIRE (e.corpus, e.name) IS UNIQUE",
    "CREATE CONSTRAINT doc_chunk_id IF NOT EXISTS FOR (c:DocChunk) REQUIRE c.id IS UNIQUE",
    "CREATE INDEX doc_entity_corpus IF NOT EXISTS FOR (e:DocEntity) ON (e.corpus)",
    "CREATE INDEX doc_chunk_corpus IF NOT EXISTS FOR (c:DocChunk) ON (c.corpus)",
]

_WRITE_CHUNK = """
MERGE (ch:DocChunk {id: $id})
SET ch:Chunk, ch.corpus = $corpus, ch.doc_id = $doc_id, ch.ticker = $doc_id, ch.title = $title, ch.text = $text
"""
_WRITE_ENTITIES = """
UNWIND $rows AS r
MERGE (e:DocEntity {corpus: $corpus, name: r.name})
WITH e, r
CALL apoc.create.addLabels(e, [r.label]) YIELD node
WITH node, r
MATCH (ch:DocChunk {id: $chunk})
MERGE (ch)-[:MENTIONS]->(node)
"""
_WRITE_RELATIONS = """
UNWIND $rows AS r
MATCH (s:DocEntity {corpus: $corpus, name: r.source}), (t:DocEntity {corpus: $corpus, name: r.target})
CALL apoc.merge.relationship(s, r.type, {}, {corpus: $corpus, source_chunk: $chunk}, t, {}) YIELD rel
RETURN count(rel) AS n
"""


def _driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
                                notifications_min_severity="OFF")


def ensure_schema() -> None:
    d = _driver()
    try:
        with d.session() as s:
            for q in SCHEMA_CYPHER:
                s.run(q).consume()
    finally:
        d.close()


def _chunk_id(i: int, doc) -> str:
    return hashlib.sha1(f"{doc.doc_id}:{i}".encode()).hexdigest()


def build_extractor(llm, schema: dict):
    from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
    return SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=Literal[tuple(schema["entities"])],          # type: ignore[valid-type]
        possible_relations=Literal[tuple({r[1] for r in schema["relations"]})],  # type: ignore[valid-type]
        kg_validation_schema=[tuple(r) for r in schema["relations"]],
        strict=True, num_workers=1)


def triples_from(node) -> tuple[list[dict], list[dict]]:
    """Entities and relations the extractor attached to a chunk, filtered to safe type names."""
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
    ents, rels = {}, []
    for n in node.metadata.get(KG_NODES_KEY, []) or []:
        name, label = str(getattr(n, "name", "")).strip(), str(getattr(n, "label", "")).upper()
        if name and _SAFE.match(label):
            ents[name] = {"name": name, "label": label}
    for r in node.metadata.get(KG_RELATIONS_KEY, []) or []:
        typ = str(r.label).upper()
        if _SAFE.match(typ) and r.source_id in ents and r.target_id in ents:
            rels.append({"source": r.source_id, "target": r.target_id, "type": typ})
    return list(ents.values()), rels


def ingest_text(corpus: dict, doc_id: str, title: str, text: str, progress: Progress) -> dict:
    """Extract a corpus-scoped graph from `text`, then embed its chunks into pgvector."""
    from llama_index.core import Document, Settings
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
    from llama_index.core.node_parser import SentenceSplitter

    from src.corpus_schema import extraction_hint
    from src.ingest import build_llm

    schema = corpus["schema"]
    text = text[: config.MAX_FILING_CHARS]
    from llama_index.core.schema import TextNode

    from src.chunking import chunk_texts, strip_boilerplate
    doc = Document(doc_id=doc_id, text=text, metadata={"title": title})
    # Cut where the layout changes (sections, entries), not every N tokens; pieces that are still
    # too long for the extractor go through the sentence splitter.
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    pieces = []
    for part in chunk_texts(strip_boilerplate(text)):
        pieces += splitter.split_text(part) if len(part) > CHUNK_SIZE * 4 else [part]
    chunks = [TextNode(id_=_chunk_id(i, doc), text=t, metadata={"title": title}) for i, t in enumerate(pieces)]
    hint = extraction_hint(schema, corpus["name"], corpus.get("description", ""))
    # Layout chunks are short: a project's chunk doesn't repeat the person's name at the top of the CV.
    # Tell the extractor which document (and whose) each chunk comes from; the stored text stays as is.
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")[:100]
    hint += (f"\nThis passage is part of the document “{title}”, which begins with: “{first_line}”. "
             "Facts in it are about that document's subject unless the passage names someone else.")
    total = len(chunks)
    eta = max(1, round(total * config.SECONDS_PER_CHUNK / 60))
    progress(f"Extracting graph: 0/{total} chunks (about {eta} min)", 0.0)
    ensure_schema()

    d = _driver()
    entities = relations = 0
    try:
        for i in range(0, total, BATCH):
            batch = chunks[i:i + BATCH]
            # The extractor sees the hint, but the stored/embedded chunk text never contains it.
            prompts = [c.model_copy(update={"text": f"{hint}\n\n{c.text}"}) for c in batch]
            for c in prompts:
                c.metadata.pop(KG_NODES_KEY, None)
                c.metadata.pop(KG_RELATIONS_KEY, None)
            llm = build_llm()   # fresh client per batch: each extractor call runs in a new event loop
            Settings.llm = llm
            extracted = build_extractor(llm, schema)(prompts)
            with d.session() as s:
                for chunk, out in zip(batch, extracted):
                    ents, rels = triples_from(out)
                    s.run(_WRITE_CHUNK, id=chunk.node_id, corpus=corpus["id"], doc_id=doc_id, title=title,
                          text=chunk.text).consume()
                    if ents:
                        s.run(_WRITE_ENTITIES, rows=ents, corpus=corpus["id"], chunk=chunk.node_id).consume()
                    if rels:
                        s.run(_WRITE_RELATIONS, rows=rels, corpus=corpus["id"], chunk=chunk.node_id).consume()
                    entities, relations = entities + len(ents), relations + len(rels)
            done = min(i + BATCH, total)
            progress(f"Extracting graph: {done}/{total} chunks", 0.85 * done / total)
    finally:
        d.close()

    progress("Cleaning up entities (types, duplicates)", 0.88)
    resolve_corpus(corpus["id"], schema)
    progress("Embedding chunks for vector search", 0.9)
    embed_chunks(corpus["id"], doc_id, title, [(c.node_id, c.text) for c in chunks])
    progress("Reading tables", 0.96)
    from src import tables
    n_tables = tables.ingest(corpus["id"], doc_id, title, text)
    return {"chunks": total, "entities": entities, "relations": relations, "tables": n_tables}


# ---------------------------------------------------------------------------
# pgvector
# ---------------------------------------------------------------------------

def ensure_vector_table(conn) -> None:
    from src.vectorize import DDL
    with conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute("ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS corpus_id text")
        cur.execute("CREATE INDEX IF NOT EXISTS chunk_embeddings_corpus ON chunk_embeddings (corpus_id)")
    conn.commit()


def embed_chunks(corpus_id: str, doc_id: str, title: str, rows: list[tuple[str, str]]) -> int:
    from src.embeddings import embed_texts
    from src.vectorize import _vec, pg_connect

    conn = pg_connect()
    try:
        ensure_vector_table(conn)
        vectors = embed_texts([t for _, t in rows]) if rows else []
        with conn.cursor() as cur:
            for (cid, txt), vec in zip(rows, vectors):
                cur.execute("""
                    INSERT INTO chunk_embeddings (chunk_id, doc_id, section_path, filing_date, entity_ids,
                                                  text, embedding, corpus_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET doc_id=EXCLUDED.doc_id, section_path=EXCLUDED.section_path,
                        filing_date=EXCLUDED.filing_date, text=EXCLUDED.text, embedding=EXCLUDED.embedding,
                        corpus_id=EXCLUDED.corpus_id
                """, (cid, doc_id, title, date.today(), [], txt, _vec(vec), corpus_id))
        conn.commit()
        return len(rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def remove_document(corpus_id: str, doc_id: str) -> None:
    """Delete a document's chunks, and entities no other document in the corpus mentions."""
    from src.vectorize import pg_connect

    d = _driver()
    try:
        with d.session() as s:
            s.run("""
                MATCH (ch:DocChunk {corpus: $c, doc_id: $doc})-[:MENTIONS]->(e:DocEntity {corpus: $c})
                WHERE NOT EXISTS { MATCH (o:DocChunk {corpus: $c})-[:MENTIONS]->(e) WHERE o.doc_id <> $doc }
                WITH DISTINCT e DETACH DELETE e
            """, c=corpus_id, doc=doc_id).consume()
            s.run("MATCH (ch:DocChunk {corpus: $c, doc_id: $doc}) DETACH DELETE ch", c=corpus_id, doc=doc_id).consume()
    finally:
        d.close()
    conn = pg_connect()
    try:
        ensure_vector_table(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunk_embeddings WHERE corpus_id = %s AND doc_id = %s", (corpus_id, doc_id))
        conn.commit()
    finally:
        conn.close()
    from src import tables
    tables.delete(corpus_id, doc_id)


def remove_corpus(corpus_id: str) -> None:
    from src.vectorize import pg_connect

    d = _driver()
    try:
        with d.session() as s:
            s.run("MATCH (n) WHERE (n:DocChunk OR n:DocEntity) AND n.corpus = $c DETACH DELETE n", c=corpus_id).consume()
    finally:
        d.close()
    conn = pg_connect()
    try:
        ensure_vector_table(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunk_embeddings WHERE corpus_id = %s", (corpus_id,))
        conn.commit()
    finally:
        conn.close()
    from src import tables
    tables.delete(corpus_id)


# ---------------------------------------------------------------------------
# Profile: what's in a corpus (drives the UI's suggestions and legend)
# ---------------------------------------------------------------------------

def profile(corpus_id: str) -> dict:
    d = _driver()
    try:
        with d.session() as s:
            types = [dict(r) for r in s.run("""
                MATCH (e:DocEntity {corpus: $c})
                UNWIND [l IN labels(e) WHERE l <> 'DocEntity'] AS type
                RETURN type, count(*) AS n ORDER BY n DESC""", c=corpus_id)]
            top = [dict(r) for r in s.run("""
                MATCH (e:DocEntity {corpus: $c})
                OPTIONAL MATCH (e)-[r]-(:DocEntity {corpus: $c})
                WITH e, count(r) AS degree
                RETURN e.name AS name, [l IN labels(e) WHERE l <> 'DocEntity'][0] AS type, degree
                ORDER BY degree DESC, name LIMIT 12""", c=corpus_id)]
            rels = [dict(r) for r in s.run("""
                MATCH (:DocEntity {corpus: $c})-[r]->(:DocEntity {corpus: $c})
                RETURN type(r) AS relation, count(*) AS n ORDER BY n DESC LIMIT 12""", c=corpus_id)]
    finally:
        d.close()
    return {"types": types, "top_entities": top, "relations": rels}


def suggestions(prof: dict, kind: str) -> list[dict]:
    """Example questions built from what was actually extracted. No LLM involved."""
    top = [e for e in prof["top_entities"] if e["degree"] > 0] or prof["top_entities"]
    out = []
    if top:
        a = top[0]
        out.append({"label": "Graph · connections", "path": "GRAPH", "question": f"What is {a['name']} connected to?"})
    if len(top) > 1:
        out.append({"label": "Graph · path", "path": "GRAPH",
                    "question": f"How is {top[0]['name']} related to {top[1]['name']}?"})
    for t in prof["types"][:2]:
        out.append({"label": "Graph · list", "path": "GRAPH",
                    "question": f"Which {t['type'].lower().replace('_', ' ')} entries are in these documents?"})
    if top:
        out.append({"label": "Vector · explanation", "path": "VECTOR",
                    "question": f"What do the documents say about {top[-1]['name']}?"})
    out.append({"label": "Vector · summary", "path": "VECTOR", "question": "Summarize the main points of these documents."})
    return out[:6]


# ---------------------------------------------------------------------------
# Graph querying (parameterized templates; the model never writes Cypher)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict] = {
    "entity_connections": {
        "cypher": "MATCH (e:DocEntity {corpus:$corpus, name:$name})-[r]-(o:DocEntity {corpus:$corpus}) "
                  "RETURN e.name AS entity, [l IN labels(e) WHERE l <> 'DocEntity'][0] AS _e_type, "
                  "type(r) AS relation, startNode(r) = e AS outgoing, o.name AS other, "
                  "[l IN labels(o) WHERE l <> 'DocEntity'][0] AS _o_type, r.source_chunk AS _src LIMIT 40",
        "entity_slots": ["name"],
        "desc": "everything an entity is connected to (who/what is X, what does X have/do)",
    },
    "entities_by_relation": {
        "cypher": "MATCH (e:DocEntity {corpus:$corpus, name:$name})-[r]-(o:DocEntity {corpus:$corpus}) "
                  "WHERE type(r) = $relation "
                  "RETURN e.name AS entity, [l IN labels(e) WHERE l <> 'DocEntity'][0] AS _e_type, "
                  "type(r) AS relation, startNode(r) = e AS outgoing, o.name AS other, "
                  "[l IN labels(o) WHERE l <> 'DocEntity'][0] AS _o_type, r.source_chunk AS _src LIMIT 40",
        "entity_slots": ["name"], "value_slot": "relation",
        "desc": "one specific relation of an entity (value = a RELATION name from the list below)",
    },
    "entities_linked_of_type": {
        "cypher": "MATCH (e:DocEntity {corpus:$corpus, name:$name})-[r]-(o:DocEntity {corpus:$corpus}) "
                  "WHERE $type IN labels(o) AND type(r) <> 'MENTIONS' "
                  "RETURN e.name AS entity, [l IN labels(e) WHERE l <> 'DocEntity'][0] AS _e_type, o.name AS name, "
                  "$type AS type, collect(DISTINCT type(r)) AS relations ORDER BY name LIMIT 100",
        "entity_slots": ["name"], "value_slot": "type",
        "desc": "LIST or COUNT the entities of one type linked to an entity "
                "(how many projects did X work on; which courses did X take; where did X study)",
    },
    "entities_of_type": {
        "cypher": "MATCH (e:DocEntity {corpus:$corpus}) WHERE $type IN labels(e) "
                  "RETURN e.name AS name, $type AS type ORDER BY name LIMIT 50",
        "value_slot": "type",
        "desc": "list all entities of one type (value = an ENTITY TYPE from the list below)",
    },
    "count_of_type": {
        "cypher": "MATCH (e:DocEntity {corpus:$corpus}) WHERE $type IN labels(e) "
                  "WITH e ORDER BY e.name RETURN $type AS type, count(e) AS count, collect(e.name) AS names",
        "value_slot": "type",
        "desc": "how many entities of one type there are (value = an ENTITY TYPE)",
    },
    "path_between": {
        "cypher": "MATCH (a:DocEntity {corpus:$corpus, name:$a}), (b:DocEntity {corpus:$corpus, name:$b}), "
                  "p = shortestPath((a)-[*..4]-(b)) "
                  "WHERE all(n IN nodes(p) WHERE n.corpus = $corpus) "
                  "RETURN [n IN nodes(p) | n.name] AS path, [r IN relationships(p) | type(r)] AS relations LIMIT 3",
        "entity_slots": ["a", "b"],
        "desc": "how two entities are connected (path between them)",
    },
}


# Words people use for schema types (English + German, since documents are often German).
TYPE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "PROJECT": ("project", "projekt"),
    "INSTITUTION": ("university", "universit", "hochschule", "college", "school", "schule", "institution", "studied"),
    "ORGANIZATION": ("company", "companies", "employer", "organization", "organisation", "firm", "unternehmen", "arbeitgeber", "worked at", "work at"),
    "ROLE": ("role", "job", "position", "stelle", "rolle"),
    "SKILL": ("skill", "kenntnis", "fähigkeit", "technolog", "tool", "programming"),
    "COURSE": ("course", "kurs", "modul", "module", "lecture", "vorlesung"),
    "DEGREE": ("degree", "abschluss", "bachelor", "master"),
    "CERTIFICATION": ("certific", "zertifik", "certificate"),
    "LANGUAGE": ("language", "sprache"),
    "LOCATION": ("city", "cities", "country", "countries", "location", "place", "stadt", "land", "where"),
    "PERSON": ("person", "people", "who"),
}


def guess_type(question: str, allowed: list[str]) -> str | None:
    """The schema type a question asks about, from its words. None if unclear."""
    q = question.lower()
    hits = []
    for t in allowed:
        words = (t.lower().replace("_", " "),) + TYPE_SYNONYMS.get(t, ())
        pos = min((q.find(w) for w in words if w in q), default=-1)
        if pos >= 0 and t != "PERSON":
            hits.append((pos, t))
    return min(hits)[1] if hits else None


def guess_relation(question: str, allowed: list[str]) -> str | None:
    q = question.lower()
    for rel in allowed:
        words = [w for w in rel.lower().split("_") if len(w) > 2]
        if words and all(w in q for w in words):
            return rel
    return None


def mentioned_people(question: str, people: list[str]) -> list[str]:
    """PERSON entities the question names (full name, or a first/last name that fits only one person),
    for when the small planner forgets to extract them."""
    words = set(re.findall(r"[\w-]+", question.lower()))
    q = norm_name(question)
    full = [n for n in people if len(n) >= 3 and f" {norm_name(n)} " in f" {q} "]
    if full:
        return full[:1]
    parts = [n for n in people if any(len(w) >= 3 and w in words for w in re.findall(r"[\w-]+", n.lower()))]
    return parts if len(parts) == 1 else []


def _resolve(session, corpus_id: str, surface: str) -> str | None:
    rec = session.run("""
        MATCH (n:DocEntity {corpus: $c})
        WITH n, toLower(n.name) AS nm, toLower($q) AS q
        WITH n, CASE WHEN nm = q THEN 3 WHEN nm CONTAINS q AND size(q) >= 3 THEN 2
                     WHEN q CONTAINS nm AND size(nm) >= 4 THEN 1 ELSE 0 END AS score
        WHERE score > 0
        RETURN n.name AS name ORDER BY score DESC, CASE WHEN score = 1 THEN -size(n.name) ELSE size(n.name) END
        LIMIT 1""", c=corpus_id, q=surface).single()
    return rec["name"] if rec else None


def plan(question: str, schema: dict) -> dict:
    import json

    from src.llm import chat_client
    docs = "\n".join(
        f"- {k} ({'entities: ' + str(t['entity_slots']) if 'entity_slots' in t else ''}"
        f"{', ' if 'entity_slots' in t and 'value_slot' in t else ''}"
        f"{'value: <' + t['value_slot'] + '>' if 'value_slot' in t else ''}) — {t['desc']}"
        for k, t in TEMPLATES.items())
    system = ("Convert the question into a graph-query PLAN. Pick exactly one query_type and extract the "
              "entity name(s) and/or value it needs. NEVER write Cypher.\n"
              f"Query types:\n{docs}\n"
              f"ENTITY TYPES: {', '.join(schema['entities'])}\n"
              f"RELATIONS: {', '.join(sorted({r[1] for r in schema['relations']}))}\n"
              'Reply ONLY JSON: {"query_type":"...","entities":["..."],"value":"..."}.')
    client = chat_client("graph_plan")
    resp = client.chat(model=config.ROUTER_MODEL, format="json",
                       messages=[{"role": "system", "content": system}, {"role": "user", "content": question}])
    return json.loads(resp["message"]["content"])


def run_query(question: str, corpus: dict) -> dict:
    import difflib
    try:
        p = plan(question, corpus["schema"])
    except Exception as e:  # noqa: BLE001
        return {"source": "corpus", "query_type": None, "cypher": None, "rows": [], "error": f"plan failed: {str(e)[:80]}"}
    qtype = str(p.get("query_type", "")).strip()
    if qtype not in TEMPLATES:
        close = difflib.get_close_matches(qtype, TEMPLATES.keys(), n=1, cutoff=0.7)
        if not close:
            return {"source": "corpus", "query_type": qtype, "cypher": None, "rows": [], "error": "no template for this query type"}
        qtype = close[0]
    # A corpus-wide count/list that names an entity really asks about that entity's links:
    # "how many projects did Walid work on" is not "how many projects exist".
    if qtype in ("count_of_type", "entities_of_type") and not (p.get("entities") or []):
        try:
            d = _driver()
            with d.session() as s:
                people = [r["n"] for r in s.run("MATCH (e:DocEntity:PERSON {corpus:$c}) RETURN e.name AS n", c=corpus["id"])]
            d.close()
            p["entities"] = mentioned_people(question, people)
        except Exception:  # noqa: BLE001
            pass
    if qtype in ("count_of_type", "entities_of_type") and (p.get("entities") or []):
        qtype = "entities_linked_of_type"
    # "How many skills does X have" picked as a relation lookup: count the relation's object type
    # exactly instead of letting the answer model count (duplicate) rows.
    if qtype == "entities_by_relation" and re.match(r"\s*(how many|wie viele)\b", question, re.I):
        rels = {r[1]: r[2] for r in corpus["schema"]["relations"]}
        rel = str(p.get("value", "")).strip().upper().replace(" ", "_")
        rel = rel if rel in rels else guess_relation(question, sorted(rels))
        target = rels.get(rel) if rel else None
        if target:
            qtype, p["value"] = "entities_linked_of_type", target
    tmpl = TEMPLATES[qtype]
    params: dict = {"corpus": corpus["id"]}
    if "value_slot" in tmpl:
        allowed = corpus["schema"]["entities"] if tmpl["value_slot"] == "type" else \
            sorted({r[1] for r in corpus["schema"]["relations"]})
        value = str(p.get("value", "")).strip().upper().replace(" ", "_")
        match = value if value in allowed else (difflib.get_close_matches(value, allowed, n=1, cutoff=0.6) or [None])[0]
        if match is None:
            # The small planner often leaves the slot empty or puts a relation in a type slot:
            # read it from the question instead (deterministic, schema-bound).
            match = (guess_type(question, allowed) if tmpl["value_slot"] == "type"
                     else guess_relation(question, allowed))
        if match is None:
            return {"source": "corpus", "query_type": qtype, "cypher": None, "rows": [], "error": f"unknown {tmpl['value_slot']} {value!r}"}
        params[tmpl["value_slot"]] = match
    d = _driver()
    try:
        with d.session() as s:
            if "entity_slots" in tmpl:
                resolved = [r for r in (_resolve(s, corpus["id"], str(x)) for x in (p.get("entities") or [])) if r]
                resolved = list(dict.fromkeys(resolved))
                if len(resolved) < len(tmpl["entity_slots"]):
                    return {"source": "corpus", "query_type": qtype, "cypher": None, "rows": [],
                            "error": f"could not resolve {p.get('entities')} in this corpus"}
                params.update(dict(zip(tmpl["entity_slots"], resolved)))
            rows = [dict(r) for r in s.run(tmpl["cypher"], **params)][:50]
            if qtype == "entities_linked_of_type" and len(resolved) > 1:
                # "Walid's projects that used Docker": keep the items linked to every named entity
                for other in resolved[1:]:
                    linked = {r["name"] for r in s.run(tmpl["cypher"], **{**params, "name": other})}
                    rows = [r for r in rows if r["name"] in linked]
                params["also_linked_to"] = resolved[1:]
        shown = {k: v for k, v in params.items() if k != "corpus"}
        return {"source": "corpus", "query_type": qtype, "cypher": tmpl["cypher"], "params": shown, "rows": rows}
    except Exception as e:  # noqa: BLE001
        return {"source": "corpus", "query_type": qtype, "cypher": tmpl["cypher"], "rows": [], "error": str(e)[:120]}
    finally:
        d.close()


def verbalize(g: dict) -> list[str] | None:
    """Statements for corpus templates; None for other query types."""
    if g.get("source") != "corpus":
        return None
    qt, rows, params = g.get("query_type"), g.get("rows") or [], g.get("params", {})
    if qt in ("entity_connections", "entities_by_relation"):
        out = []
        for r in rows:
            rel = r["relation"].lower().replace("_", " ")
            out.append(f"{r['entity']} {rel} {r['other']}." if r["outgoing"] else f"{r['other']} {rel} {r['entity']}.")
        return out
    if qt == "entities_linked_of_type":
        if not rows:
            return []
        kind = params.get("type", "").lower().replace("_", " ")
        names = "; ".join(r["name"] for r in rows)
        also = params.get("also_linked_to") or []
        both = f" that are also linked to {' and '.join(also)}" if also else ""
        return [f"{rows[0]['entity']} is linked to exactly {len(rows)} {kind} entit{'y' if len(rows) == 1 else 'ies'}"
                f"{both} in this corpus: {names}."]
    if qt == "entities_of_type":
        return [f"The {params.get('type', '').lower().replace('_', ' ')} entities in this corpus are: "
                f"{', '.join(r['name'] for r in rows)}."] if rows else []
    if qt == "count_of_type":
        if not rows:
            return []
        kind, names = params.get("type", "").lower().replace("_", " "), rows[0].get("names") or []
        listed = f": {'; '.join(names)}" if names else ""
        return [f"This corpus has exactly {rows[0]['count']} {kind} entities{listed}."]
    if qt == "path_between":
        out = []
        for r in rows:
            path, rels = r.get("path", []), r.get("relations", [])
            out.append("; ".join(f"{path[i]} {rels[i].lower().replace('_', ' ')} {path[i + 1]}"
                                 for i in range(len(rels))) + ".")
        return out
    return None


def edges(g: dict) -> list[tuple[str, str, str, int, str, str]] | None:
    """(source, relation, target, row, source_type, target_type) for the graph view; None if not a corpus query."""
    if g.get("source") != "corpus":
        return None
    qt, rows = g.get("query_type"), g.get("rows") or []
    if qt in ("entity_connections", "entities_by_relation"):
        out = []
        for i, r in enumerate(rows):
            a, at, b, bt = r["entity"], r.get("_e_type") or "CONCEPT", r["other"], r.get("_o_type") or "CONCEPT"
            out.append((a, r["relation"], b, i, at, bt) if r["outgoing"] else (b, r["relation"], a, i, bt, at))
        return out
    if qt == "path_between":
        out = []
        for i, r in enumerate(rows):
            p, rels = r.get("path", []), r.get("relations", [])
            out += [(p[j], rels[j], p[j + 1], i, "CONCEPT", "CONCEPT") for j in range(len(rels))]
        return out
    if qt == "entities_linked_of_type":
        return [(r["entity"], (r.get("relations") or ["RELATED_TO"])[0], r["name"], 0,
                 r.get("_e_type") or "PERSON", r.get("type") or "CONCEPT") for r in rows]
    if qt in ("entities_of_type", "count_of_type"):
        return []
    return None


# ---------------------------------------------------------------------------
# Entity resolution within a corpus (runs after every ingest)
# ---------------------------------------------------------------------------

_SEP = re.compile(r"^\s*(?:[–—-]|\(|,|:|\bfor\b|\bfür\b|\bat\b|\bbei\b)", re.I)


def norm_name(name: str) -> str:
    """Comparison key: case, punctuation and spacing don't distinguish entities."""
    s = re.sub(r"[^\w]+", " ", name.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _display_rank(name: str) -> tuple:
    """Prefer a properly cased, longer name as the canonical one."""
    mixed = name != name.upper() and name != name.lower()
    return (mixed, len(name))


def plan_merges(nodes: list[dict]) -> list[tuple[str, list[str]]]:
    """Which nodes to merge: [(keep_id, [dup_ids])]. Pure function, unit-tested.

    nodes: [{id, name, type}] for one corpus, after each node has a single type.
    Rules (same type only):
      1. same name up to case/punctuation/spacing;
      2. one name + a separator + more ("NBA-Vorhersage – ML-Projekt", "General Treasury (TGR)");
      3. PERSON: a single-word name equal to the first name of exactly one full name.
    """
    parent = {n["id"]: n["id"] for n in nodes}
    by_id = {n["id"]: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_type: dict[str, list[dict]] = {}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)
    for typ, group in by_type.items():
        keys: dict[str, str] = {}
        for n in group:                                   # rule 1
            k = norm_name(n["name"])
            if k in keys:
                union(keys[k], n["id"])
            else:
                keys[k] = n["id"]
        for a in group:                                   # rule 2
            na = a["name"].strip()
            if len(norm_name(na)) < 6:
                continue
            for b in group:
                nb = b["name"].strip()
                if a is b or len(nb) <= len(na) or not nb.lower().startswith(na.lower()):
                    continue
                if _SEP.match(nb[len(na):]):
                    union(a["id"], b["id"])
        if typ == "PERSON":                               # rule 3
            full = [n for n in group if len(n["name"].split()) >= 2]
            for n in group:
                if len(n["name"].split()) == 1:
                    first = n["name"].lower()
                    cands = {find(f["id"]) for f in full if f["name"].split()[0].lower() == first}
                    if len(cands) == 1:
                        union(cands.pop(), n["id"])

    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n["id"]), []).append(n["id"])
    out = []
    for members in groups.values():
        if len(members) > 1:
            keep = max(members, key=lambda i: _display_rank(by_id[i]["name"]))
            out.append((keep, [m for m in members if m != keep]))
    return out


def pick_type(candidates: list[str], incident: list[tuple[str, str, str]], schema: dict) -> str:
    """One type for a node the extractor tagged with several. Votes come from the
    node's relations: each (direction, RELATION, other_type) that the schema allows
    for a candidate type supports it. Pure function, unit-tested."""
    triples = {tuple(t) for t in schema["relations"]}
    best, best_score = candidates[0], -1
    for cand in candidates:
        score = 0
        for direction, rel, other in incident:
            if (direction == "out" and (cand, rel, other) in triples) or \
               (direction == "in" and (other, rel, cand) in triples):
                score += 1
        if score > best_score:
            best, best_score = cand, score
    return best


def resolve_corpus(corpus_id: str, schema: dict) -> dict:
    """Clean a corpus's graph: one type per entity, schema-valid relations only,
    and duplicate entities merged (aliases kept)."""
    types = set(schema["entities"])
    d = _driver()
    stats = {"retyped": 0, "invalid_relations": 0, "merged": 0}
    try:
        with d.session() as s:
            # 1. one type per entity
            multi = [dict(r) for r in s.run("""
                MATCH (e:DocEntity {corpus: $c})
                WITH e, [l IN labels(e) WHERE l <> 'DocEntity'] AS ts WHERE size(ts) > 1
                OPTIONAL MATCH (e)-[r]-(o:DocEntity {corpus: $c}) WHERE type(r) <> 'MENTIONS'
                RETURN elementId(e) AS id, ts,
                       collect(CASE WHEN r IS NULL THEN null ELSE
                         [CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END, type(r),
                          [l IN labels(o) WHERE l <> 'DocEntity'][0]] END) AS incident""", c=corpus_id)]
            for m in multi:
                cands = [t for t in m["ts"] if t in types] or m["ts"]
                keep = pick_type(cands, [tuple(x) for x in m["incident"] if x], schema)
                drop = [t for t in m["ts"] if t != keep]
                s.run("MATCH (e) WHERE elementId(e) = $id CALL apoc.create.removeLabels(e, $drop) YIELD node RETURN 1",
                      id=m["id"], drop=drop).consume()
                stats["retyped"] += 1
            # 2. relations the schema doesn't allow (e.g. a person WORKED_ON a university)
            stats["invalid_relations"] = s.run("""
                MATCH (a:DocEntity {corpus: $c})-[r]->(b:DocEntity {corpus: $c}) WHERE type(r) <> 'MENTIONS'
                WITH r, [l IN labels(a) WHERE l <> 'DocEntity'][0] AS ta, type(r) AS rel,
                     [l IN labels(b) WHERE l <> 'DocEntity'][0] AS tb
                WHERE NOT [ta, rel, tb] IN $triples
                DELETE r RETURN count(*) AS n""", c=corpus_id, triples=[list(t) for t in schema["relations"]]).single()["n"]
            # 3. merge duplicates
            nodes = [dict(r) for r in s.run("""
                MATCH (e:DocEntity {corpus: $c})
                RETURN elementId(e) AS id, e.name AS name, [l IN labels(e) WHERE l <> 'DocEntity'][0] AS type""",
                c=corpus_id)]
            for keep, dups in plan_merges(nodes):
                s.run("""
                    MATCH (k) WHERE elementId(k) = $keep
                    MATCH (d) WHERE elementId(d) IN $dups
                    WITH k, collect(d) AS ds, [x IN collect(d) | x.name] + coalesce(k.aliases, []) AS aliases
                    CALL apoc.refactor.mergeNodes([k] + ds, {properties: 'discard', mergeRels: true}) YIELD node
                    SET node.aliases = apoc.coll.toSet(aliases)
                    RETURN 1""", keep=keep, dups=dups).consume()
                stats["merged"] += len(dups)
            s.run("MATCH (e:DocEntity {corpus: $c})-[r]->(e) WHERE type(r) <> 'MENTIONS' DELETE r", c=corpus_id).consume()
    finally:
        d.close()
    return stats


if __name__ == "__main__":
    import argparse
    import json
    import os

    from server.corpora import CorpusStore
    ap = argparse.ArgumentParser(description="Clean up a corpus's graph (types, invalid relations, duplicates).")
    ap.add_argument("corpus_id")
    a = ap.parse_args()
    c = CorpusStore(os.getenv("DOCUMENTS_DB", str(config.DATA_DIR / "documents.db"))).get(a.corpus_id)
    if not c:
        raise SystemExit("No such corpus.")
    print(json.dumps(resolve_corpus(c["id"], c["schema"])))
