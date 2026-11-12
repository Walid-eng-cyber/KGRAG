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
    doc = Document(doc_id=doc_id, text=text, metadata={"title": title})
    chunks = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                              id_func=_chunk_id).get_nodes_from_documents([doc])
    hint = extraction_hint(schema, corpus["name"], corpus.get("description", ""))
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

    progress("Embedding chunks for vector search", 0.9)
    embed_chunks(corpus["id"], doc_id, title, [(c.node_id, c.text) for c in chunks])
    return {"chunks": total, "entities": entities, "relations": relations}


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
    "entities_of_type": {
        "cypher": "MATCH (e:DocEntity {corpus:$corpus}) WHERE $type IN labels(e) "
                  "RETURN e.name AS name, $type AS type ORDER BY name LIMIT 50",
        "value_slot": "type",
        "desc": "list all entities of one type (value = an ENTITY TYPE from the list below)",
    },
    "count_of_type": {
        "cypher": "MATCH (e:DocEntity {corpus:$corpus}) WHERE $type IN labels(e) RETURN $type AS type, count(e) AS count",
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

    import ollama
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
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
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
    tmpl = TEMPLATES[qtype]
    params: dict = {"corpus": corpus["id"]}
    if "value_slot" in tmpl:
        allowed = corpus["schema"]["entities"] if tmpl["value_slot"] == "type" else \
            sorted({r[1] for r in corpus["schema"]["relations"]})
        value = str(p.get("value", "")).strip().upper().replace(" ", "_")
        match = value if value in allowed else (difflib.get_close_matches(value, allowed, n=1, cutoff=0.6) or [None])[0]
        if match is None:
            return {"source": "corpus", "query_type": qtype, "cypher": None, "rows": [], "error": f"unknown {tmpl['value_slot']} {value!r}"}
        params[tmpl["value_slot"]] = match
    d = _driver()
    try:
        with d.session() as s:
            if "entity_slots" in tmpl:
                resolved = [r for r in (_resolve(s, corpus["id"], str(x)) for x in (p.get("entities") or [])) if r]
                if len(resolved) < len(tmpl["entity_slots"]):
                    return {"source": "corpus", "query_type": qtype, "cypher": None, "rows": [],
                            "error": f"could not resolve {p.get('entities')} in this corpus"}
                params.update(dict(zip(tmpl["entity_slots"], resolved)))
            rows = [dict(r) for r in s.run(tmpl["cypher"], **params)][:50]
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
    if qt == "entities_of_type":
        return [f"The {params.get('type', '').lower().replace('_', ' ')} entities in this corpus are: "
                f"{', '.join(r['name'] for r in rows)}."] if rows else []
    if qt == "count_of_type":
        return [f"This corpus has {rows[0]['count']} {params.get('type', '').lower().replace('_', ' ')} entities."] if rows else []
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
    if qt in ("entities_of_type", "count_of_type"):
        return []
    return None
