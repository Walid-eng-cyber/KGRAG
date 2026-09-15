"""Phase 4 — merge graph + vector into one grounded, cited answer.

The router (Phase 3) returns two shapes of evidence:
  - graph path -> rows / paths (triples)
  - vector path -> passages (chunk text)

Raw triples like (Apple)-[HAS_SUBSIDIARY]->(Beats) make an LLM write awkward
text, so we VERBALIZE graph results into plain-English statements *before* they
reach the prompt. Then we assemble both kinds of evidence as numbered sources
and ask the model to answer using only those, citing [S1], [S2], ... — grounded
and traceable back to the graph / the source chunk.
"""
from __future__ import annotations

import re

import ollama

import config
from src.router import route

# Relationship -> readable phrase, for verbalizing paths.
REL_PHRASE = {
    "HAS_SUBSIDIARY": "has subsidiary",
    "COMPETES_WITH": "competes with",
    "PARTNERS_WITH": "partners with",
    "ACQUIRED": "acquired",
    "SUPPLIES_TO": "supplies to",
    "HAS_EXECUTIVE": "has executive",
    "HAS_DIRECTOR": "has director",
    "OFFERS_PRODUCT": "offers",
    "OPERATES_SEGMENT": "operates segment",
    "FACES_RISK": "faces the risk",
    "AUDITED_BY": "is audited by",
    "HEADQUARTERED_IN": "is headquartered in",
    "OPERATES_IN": "operates in",
    "REGULATED_BY": "is regulated by",
    "LISTED_ON": "is listed on",
}


def verbalize_graph(g: dict) -> list[str]:
    """Turn graph rows/paths into readable statements (query-type aware)."""
    if not g or g.get("error") or not g.get("rows"):
        return []
    qt = g.get("query_type")
    rows = g["rows"]
    params = g.get("params", {})
    name = params.get("name") or (params.get("names") or [None])[0]

    def col(key: str) -> str:
        return ", ".join(str(r[key]) for r in rows if r.get(key) is not None)

    if qt == "subsidiaries_of":
        return [f"{name}'s subsidiaries include: {col('subsidiary')}."]
    if qt == "count_subsidiaries":
        return [f"{name} has {rows[0]['count']} subsidiaries recorded in the graph."]
    if qt == "risks_of":
        return [f"{name} discloses these risks: {col('risk')}."]
    if qt == "executives_of":
        return [f"{name}'s executives include: {col('executive')}."]
    if qt == "competitors_of":
        return [f"{name}'s competitors include: {col('competitor')}."]
    if qt == "regulators_of":
        return [f"{name} is regulated by: {col('regulator')}."]
    if qt == "companies_facing_risk":
        return [f"Companies that face '{params.get('risk','')}' risk: {col('company')}."]
    if qt == "compare_subsidiary_counts":
        return [f"{r['company']} has {r['count']} subsidiaries." for r in rows]
    if qt == "neighbors":
        return [f"{name} {REL_PHRASE.get(r['relation'], r['relation'].lower())} "
                f"{r['connected_to']}." for r in rows]
    if qt == "path_between":
        out = []
        for r in rows:
            path, rels = r.get("path", []), r.get("relations", [])
            hops = [f"{path[i]} {REL_PHRASE.get(rels[i], rels[i].lower())} {path[i+1]}"
                    for i in range(len(rels))]
            out.append("; ".join(hops) + ".")
        return out
    return [f"{qt}: {rows}"]  # generic fallback


# query_type -> (relation label, center is the $name entity) for graph viz.
def graph_triples(g: dict) -> list[tuple[str, str, str]]:
    """Structured (source, relation, target) triples for visualizing a graph
    answer (distinct from verbalize_graph, which makes prose)."""
    if not g or g.get("error") or not g.get("rows"):
        return []
    qt, rows, params = g.get("query_type"), g["rows"], g.get("params", {})
    name = params.get("name")
    simple = {"subsidiaries_of": ("HAS_SUBSIDIARY", "subsidiary"),
              "risks_of": ("FACES_RISK", "risk"),
              "executives_of": ("HAS_EXECUTIVE", "executive"),
              "competitors_of": ("COMPETES_WITH", "competitor"),
              "regulators_of": ("REGULATED_BY", "regulator")}
    if qt in simple:
        rel, col = simple[qt]
        return [(name, rel, str(r[col])) for r in rows if r.get(col)]
    if qt == "companies_facing_risk":
        return [(str(r["company"]), "FACES_RISK", params.get("risk", "risk")) for r in rows]
    if qt == "neighbors":
        return [(name, str(r["relation"]), str(r["connected_to"])) for r in rows]
    if qt == "path_between":
        out = []
        for r in rows:
            path, rels = r.get("path", []), r.get("relations", [])
            out += [(str(path[i]), str(rels[i]), str(path[i + 1])) for i in range(len(rels))]
        return out
    return []


def format_passages(vs: list[dict], limit: int = 4) -> list[tuple[str, str]]:
    """(citation, passage-text) tuples from vector hits."""
    out = []
    for r in vs[:limit]:
        cite = (f"{r['doc_id']} · {r['section_path']} · {r['filing_date']} "
                f"(chunk {r['chunk_id'][:8]})")
        text = re.sub(r"\s+", " ", r["text"]).strip()[:400]
        out.append((cite, text))
    return out


def _dedup_passages(vs: list[dict], graph_src_chunks: set[str]) -> list[dict]:
    """Cross-set + intra-set dedup: drop a passage that is already the source
    chunk of a graph-derived fact (cross-set), or a repeat chunk (intra-set)."""
    out, seen = [], set()
    for r in vs:
        cid = r["chunk_id"]
        if cid in graph_src_chunks or cid in seen:
            continue
        seen.add(cid)
        out.append(r)
    return out


_CITE = re.compile(r"\[(G\d+|P\d+)\]")


def validate_citations(text: str, valid_tags: set[str]) -> tuple[list[str], list[str]]:
    """Return (invalid_citations, uncited_claims).

    invalid: cited labels that were NOT in the retrieved sources (invented).
    uncited: substantive claims (sentences/bullets) with no citation at all.
    """
    cited = set(_CITE.findall(text))
    invalid = sorted(cited - valid_tags)

    uncited = []
    for line in re.split(r"(?<=[.!?])\s+|\n+", text):
        claim = line.strip("•*-  ").strip()
        # a "claim" is a real sentence, not a header/label line
        if len(claim.split()) >= 5 and not claim.endswith(":") and not _CITE.search(claim):
            uncited.append(claim[:70])
    return invalid, uncited


def _corrective(valid_tags: set[str], invalid: list[str], uncited: list[str]) -> str:
    msg = ["Revise the answer. Rules: every claim must carry a citation, and you "
           f"may ONLY cite these labels: {sorted(valid_tags)}."]
    if invalid:
        msg.append(f"You cited labels that do not exist: {invalid}. Remove them.")
    if uncited:
        msg.append(f"These claims have no citation: {uncited}. Add a valid one or drop them.")
    return " ".join(msg)


def answer(question: str) -> dict:
    routed = route(question)
    res = routed["result"]
    graph_result = res.get("graph", {}) if "graph" in res else {}

    # 1. Graph-derived facts (verbalized) + the source chunks behind them.
    graph_stmts = list(dict.fromkeys(verbalize_graph(graph_result)))  # dedup identical
    graph_src_chunks = {r["_src"] for r in graph_result.get("rows", []) if r.get("_src")}

    # 2. Passages, deduped across both sets (a passage that IS a graph fact's
    #    source is redundant) and within the set.
    vhits = _dedup_passages(res.get("vector", []), graph_src_chunks) if "vector" in res else []
    passages = format_passages(vhits)

    if not graph_stmts and not passages:
        return {"question": question, "path_used": routed["path_used"],
                "answer": "I don't have enough information in the sources to answer.",
                "graph_statements": graph_stmts, "sources": [], "context": ""}

    # 3. Assemble context with EXPLICIT labels separating the two kinds.
    blocks, sources = [], []
    if graph_stmts:
        lines = []
        for i, s in enumerate(graph_stmts):
            tag = f"G{i+1}"
            lines.append(f"[{tag}] {s}")
            sources.append({"tag": tag, "kind": "graph-derived fact", "text": s})
        blocks.append("GRAPH-DERIVED FACTS (from the knowledge graph):\n" + "\n".join(lines))
    if passages:
        lines = []
        for i, (cite, text) in enumerate(passages):
            tag = f"P{i+1}"
            lines.append(f"[{tag}] ({cite}) {text}")
            sources.append({"tag": tag, "kind": "retrieved passage", "cite": cite, "text": text})
        blocks.append("RETRIEVED PASSAGES (from vector search):\n" + "\n".join(lines))

    context = "\n\n".join(blocks)
    valid_tags = {s["tag"] for s in sources}
    system = (
        "You answer questions about SEC 10-K filings using ONLY the sources "
        "below. There are two kinds: GRAPH-DERIVED FACTS (structured, labeled "
        "[G1], [G2], ...) and RETRIEVED PASSAGES (text, labeled [P1], [P2], ...). "
        "Prefer graph-derived facts for counts and relationships and passages for "
        "explanations. EVERY claim MUST end with a citation, and you may ONLY "
        "cite labels that appear below — never invent a citation. If the sources "
        "do not contain the answer, say you don't have enough information. Be concise."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}\n\n{context}"}]

    # Generate, then validate every citation resolves to a retrieved source.
    # Reject and regenerate on invalid/missing citations (cheap: only on failure).
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    text, invalid, uncited, attempts = "", [], [], 0
    for attempts in range(1, config.ANSWER_MAX_RETRIES + 2):
        text = client.chat(model=config.ANSWER_MODEL, messages=messages)["message"]["content"].strip()
        invalid, uncited = validate_citations(text, valid_tags)
        if not invalid and not uncited:
            break
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": _corrective(valid_tags, invalid, uncited)}]

    # Fail-safe: if a hallucinated citation survived all retries, strip it so no
    # invented source is ever presented to the user.
    if invalid:
        text = re.sub(r"\s*\[(?:" + "|".join(re.escape(t) for t in invalid) + r")\]", "", text)

    return {"question": question,
            "classified": routed["classified"], "confidence": routed["confidence"],
            "path_used": routed["path_used"],
            "answer": text, "graph_statements": graph_stmts, "sources": sources,
            "graph_result": graph_result, "vector_hits": vhits, "context": context,
            "attempts": attempts, "citations_valid": not invalid,
            "uncited_claims": uncited}


if __name__ == "__main__":
    for q in ["How many subsidiaries does Microsoft have?",
              "What does Apple say about supply chain risk?"]:
        r = answer(q)
        print(f"\n{'='*70}\nQ: {q}\n   path: {r['path_used']}")
        if r["graph_statements"]:
            print("   verbalized graph:")
            for s in r["graph_statements"]:
                print(f"     - {s}")
        print(f"\nANSWER:\n{r['answer']}")
        print(f"\nsources: {len(r['sources'])}")
