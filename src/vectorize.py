"""Phase 2 — build the vector index alongside the graph.

Embeds the SAME chunks that Phase 1 wrote to Neo4j into a pgvector table, tagged
with the metadata that makes a vector hit graph-aware:

  chunk_id      the join key — identical to the Neo4j :Chunk id
  doc_id        which filing (ticker)
  section_path  which 10-K section the chunk came from (Item 1 / Item 1A)
  filing_date   report/period date of the filing
  entity_ids[]  the graph entities this chunk MENTIONS
  text          the chunk text (for retrieval / citation)
  embedding     bge-m3 vector(1024)

The chunk_id is the bridge: vector search finds relevant chunks, and their
entity_ids / chunk_id let you jump straight into the Neo4j graph.

Usage:
  python -m src.vectorize                         # build/refresh the index
  python -m src.vectorize --query "supply chain risk"   # build + demo search
"""
from __future__ import annotations

import argparse
import re
from datetime import date

import psycopg2
from pgvector.psycopg2 import register_vector

import config
from src.edgar import ITEM_TITLES, fetch_10k_text, latest_filing_date, narrative_sections
from src.embeddings import embed_one, embed_texts

from neo4j import GraphDatabase

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id     text PRIMARY KEY,
    doc_id       text,
    section_path text,
    filing_date  date,
    entity_ids   text[],
    text         text,
    embedding    vector({config.VECTOR_DIM})
);
CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
"""

UPSERT = """
INSERT INTO chunk_embeddings
    (chunk_id, doc_id, section_path, filing_date, entity_ids, text, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id=EXCLUDED.doc_id, section_path=EXCLUDED.section_path,
    filing_date=EXCLUDED.filing_date, entity_ids=EXCLUDED.entity_ids,
    text=EXCLUDED.text, embedding=EXCLUDED.embedding;
"""


def _vec(v: list[float]) -> str:
    """pgvector text literal, e.g. [0.1,0.2,...] — robust across drivers."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def pg_connect():
    conn = psycopg2.connect(
        host=config.PG_HOST, port=config.PG_PORT, dbname=config.PG_DB,
        user=config.PG_USER, password=config.PG_PASSWORD,
    )
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def section_for(chunk_text: str, sections: dict[str, str]) -> str:
    """Which 10-K section a chunk came from, by matching a mid-chunk signature
    against each section's text."""
    body = _norm(chunk_text)
    sig = body[200:280] if len(body) > 320 else body[:80]
    for item, sec in sections.items():
        if sig and sig in _norm(sec):
            return f"10-K / {ITEM_TITLES.get(item, 'Item ' + item)}"
    return "10-K / other"


def tickers_in_graph(driver) -> list[str]:
    with driver.session() as s:
        return [r["t"] for r in s.run(
            "MATCH (ch:Chunk) WHERE ch.ticker IS NOT NULL RETURN DISTINCT ch.ticker AS t"
        )]


def chunks_for(driver, ticker: str) -> list[dict]:
    q = """
    MATCH (ch:Chunk {ticker: $t})
    OPTIONAL MATCH (ch)-[:MENTIONS]->(e)
    WITH ch, collect(e.id) AS ents
    RETURN ch.id AS chunk_id, ch.text AS text, ents AS entity_ids
    """
    with driver.session() as s:
        return [dict(r) for r in s.run(q, t=ticker)]


def build() -> None:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )
    conn = pg_connect()
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()

    total = 0
    for ticker in tickers_in_graph(driver):
        _, filing_text = fetch_10k_text(ticker)
        sections = narrative_sections(filing_text)
        fdate = latest_filing_date(ticker)
        fdate = date.fromisoformat(fdate) if fdate else None

        rows = chunks_for(driver, ticker)
        if not rows:
            continue
        vectors = embed_texts([r["text"] for r in rows])

        with conn.cursor() as cur:
            for r, vec in zip(rows, vectors):
                cur.execute(UPSERT, (
                    r["chunk_id"], ticker, section_for(r["text"], sections),
                    fdate, r["entity_ids"], r["text"], _vec(vec),
                ))
        conn.commit()
        total += len(rows)
        print(f"  {ticker}: embedded {len(rows)} chunks  (date {fdate})")

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunk_embeddings;")
        n = cur.fetchone()[0]
    print(f"\nVector index built: {total} chunks embedded, {n} rows in pgvector.")
    conn.close()
    driver.close()


_COLS = ["chunk_id", "doc_id", "section_path", "filing_date", "text", "entity_ids", "similarity"]
# Words that carry no meaning for keyword search (English, German, French).
_KW_STOP = set("""a an the and or of to in on at for with by from is are was were be been has have had do does did
what which who whom whose when where why how many much this that these those it its his her their them they he she
i you we my your our me us not no yes can could would should will about into over than then there here also all any
der die das ein eine einer eines und oder von zu im in am an auf für mit bei aus ist sind war waren hat haben wie
was wer wo wann welche welcher welches viele sein seine ihr ihre er sie es nicht kein keine auch le la les un une
des et ou de du au aux pour avec dans sur est sont il elle quel quelle combien""".split())


def _kw_terms(query: str) -> list[str]:
    """The question's meaningful words, plus their synonyms in other languages (thesis -> bachelorarbeit)."""
    from src.tools.dates import SYNONYMS
    words = [re.sub(r"[^\w]", "", w) for w in re.findall(r"[\w-]+", query.lower())]
    words = [w for w in words if len(w) >= 3 and w not in _KW_STOP]
    out = []
    for w in words:
        out += [w] + [x for x in re.split(r"[\s.]+", SYNONYMS.get(w, "")) if len(x) >= 3]
    return list(dict.fromkeys(out))[:16]


def _kw_query(query: str) -> str | None:
    """An OR-query for Postgres full-text search ('simple' config: no stemming, so it behaves the same in
    every language; prefix matching covers plurals and compounds: bachelor:* matches Bachelorarbeit)."""
    terms = _kw_terms(query)
    return " | ".join(f"{w}:*" for w in terms) or None


def bm25(query: str, docs: list[dict], n_total: int, df: dict[str, int], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """BM25 scores of docs for the query terms (prefix match, like the SQL query). A word found in
    most chunks (the person's own name, "note") counts for almost nothing; a rare one ("Neo4Py") counts a lot."""
    import math
    terms = _kw_terms(query)
    toks = [re.findall(r"\w+", d["text"].lower()) for d in docs]
    avg = (sum(map(len, toks)) / len(toks)) if toks else 1.0
    out = []
    for t in toks:
        sc = 0.0
        for term in terms:
            tf = sum(1 for w in t if w.startswith(term))
            if not tf:
                continue
            n = df.get(term, 0)
            idf = math.log(1 + (n_total - n + 0.5) / (n + 0.5))
            sc += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg))
        out.append(sc)
    return out


def _ensure_fts(cur) -> None:
    cur.execute("ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS corpus_id text")
    cur.execute("CREATE INDEX IF NOT EXISTS chunk_embeddings_fts ON chunk_embeddings "
                "USING gin (to_tsvector('simple', text))")


def retrieve(query: str, k: int = 5, corpus_id: str | None = None, hybrid: bool = True) -> list[dict]:
    """Top-k chunks for a question. With a corpus, only that corpus's chunks are searched; without one,
    only chunks that belong to no corpus (the legacy 10-K index).

    hybrid=True also runs a keyword search and merges both rankings (reciprocal rank fusion): vectors
    find paraphrases, keywords find exact names, codes and numbers ("Neo4Py", "Cadi Ayyad", "PNr 1010")
    that embeddings blur. hybrid=False is plain vector search (used for the baseline)."""
    conn = pg_connect()
    qv = _vec(embed_one(query))
    with conn.cursor() as cur:
        _ensure_fts(cur)
        conn.commit()
        cur.execute(
            """
            SELECT chunk_id, doc_id, section_path, filing_date, text, entity_ids,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunk_embeddings
            WHERE corpus_id IS NOT DISTINCT FROM %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (qv, corpus_id, qv, k * 3 if hybrid else k),
        )
        vec_rows = [dict(zip(_COLS, r)) for r in cur.fetchall()]
        kw_rows: list[dict] = []
        tsq = _kw_query(query) if hybrid else None
        if tsq:
            cur.execute(
                """
                SELECT chunk_id, doc_id, section_path, filing_date, text, entity_ids,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunk_embeddings
                WHERE corpus_id IS NOT DISTINCT FROM %s
                  AND to_tsvector('simple', text) @@ to_tsquery('simple', %s)
                ORDER BY ts_rank_cd(to_tsvector('simple', text), to_tsquery('simple', %s)) DESC
                LIMIT 300;
                """,
                (qv, corpus_id, tsq, tsq),
            )
            cands = [dict(zip(_COLS, r)) for r in cur.fetchall()]
            if cands:
                cur.execute("SELECT count(*) FROM chunk_embeddings WHERE corpus_id IS NOT DISTINCT FROM %s", (corpus_id,))
                n_total = cur.fetchone()[0]
                df = {}
                for term in _kw_terms(query):
                    cur.execute("SELECT count(*) FROM chunk_embeddings WHERE corpus_id IS NOT DISTINCT FROM %s "
                                "AND to_tsvector('simple', text) @@ to_tsquery('simple', %s)", (corpus_id, f"{term}:*"))
                    df[term] = cur.fetchone()[0]
                scores = bm25(query, cands, n_total, df)
                kw_rows = [r for sc, r in sorted(zip(scores, cands), key=lambda t: -t[0]) if sc > 0][:k * 3]
    conn.close()
    if not hybrid:
        return vec_rows
    fused = fuse([vec_rows, kw_rows], k)
    # The best exact-word matches always get a place: a chunk that is the only one naming "Neo4Py"
    # must not lose to chunks that are merely similar in meaning.
    for r in kw_rows[:2]:
        if r["chunk_id"] not in {x["chunk_id"] for x in fused}:
            fused = fused[:-1] if len(fused) >= k else fused
            fused.append(r)
    return fused


def fuse(rankings: list[list[dict]], k: int, c: int = 60) -> list[dict]:
    """Reciprocal rank fusion: a chunk ranked high by either search ranks high overall."""
    score: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranking in rankings:
        for rank, r in enumerate(ranking):
            score[r["chunk_id"]] = score.get(r["chunk_id"], 0.0) + 1.0 / (c + rank + 1)
            rows.setdefault(r["chunk_id"], r)
    return [rows[cid] for cid in sorted(score, key=lambda x: -score[x])[:k]]


def chunks_matching(corpus_id: str, pattern: str, limit: int = 40) -> list[dict]:
    """All chunks of one corpus whose text matches a POSIX regex (case-insensitive).
    Used by the answer tools, which need every dated passage, not just the top-k similar ones."""
    conn = pg_connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, doc_id, section_path, filing_date, text, entity_ids, 0.0 AS similarity
            FROM chunk_embeddings WHERE corpus_id = %s AND text ~* %s
            ORDER BY doc_id, chunk_id LIMIT %s;
            """,
            (corpus_id, pattern, limit),
        )
        cols = ["chunk_id", "doc_id", "section_path", "filing_date", "text", "entity_ids", "similarity"]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def search(query: str, k: int = 5) -> None:
    print(f"\nTop {k} chunks for: {query!r}\n" + "-" * 60)
    for r in retrieve(query, k):
        print(f"[{r['similarity']:.3f}] {r['doc_id']} · {r['section_path']} · {r['filing_date']}")
        print(f"   {_norm(r['text'][:140])}…")
        print(f"   entities: {', '.join(r['entity_ids'][:6])}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the pgvector index from graph chunks.")
    p.add_argument("--query", help="run a demo similarity search after building")
    args = p.parse_args()
    build()
    if args.query:
        search(args.query)
