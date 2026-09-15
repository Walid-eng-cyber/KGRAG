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


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Vector top-k as structured rows (the vector retrieval path)."""
    conn = pg_connect()
    qv = _vec(embed_one(query))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, doc_id, section_path, filing_date, text, entity_ids,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunk_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (qv, qv, k),
        )
        cols = ["chunk_id", "doc_id", "section_path", "filing_date",
                "text", "entity_ids", "similarity"]
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
