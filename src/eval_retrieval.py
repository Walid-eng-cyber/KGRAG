"""Measure retrieval quality BEFORE building anything on top of it.

Two numbers matter:
  recall@k          did the embedding rank the known-relevant chunk(s) in top-k?
                    (a property of the embedding model + data)
  ann_recall@k      did HNSW return the same top-k as an exact brute-force scan?
                    (a property of the index / ef_search — the ANN error)

We sweep pgvector's `hnsw.ef_search` and report both, plus latency, so the
ef_search / recall / speed tradeoff is explicit. A labeled query set (below)
maps each query to the chunk id(s) that actually answer it.

Usage:
  python -m src.eval_retrieval
"""
from __future__ import annotations

import time

import psycopg2

import config
from src.embeddings import embed_texts

# (query, [chunk_id prefixes that are genuinely relevant]) — hand-labeled by
# reading the 27 chunks. Prefixes are unique within the corpus.
LABELED: list[tuple[str, list[str]]] = [
    ("Apple supply chain concentration and geopolitical risk", ["15b03466", "6f67ca15"]),
    ("Apple uses custom components from its suppliers", ["f5a695f0"]),
    ("Apple TV media streaming device", ["7379d7d6"]),
    ("Apple macroeconomic conditions affecting demand", ["af7b0eeb"]),
    ("Microsoft 365 Commercial revenue", ["ce5004a1"]),
    ("competitors offering free applications and services", ["8bdcf3c8"]),
    ("switching costs and platform lock-in network effects", ["8363dc18", "12631ced"]),
    ("NVIDIA data center GPU platform for AI", ["35213344", "19a054f7"]),
    ("NVIDIA Grace data center CPU", ["ff79a53a"]),
    ("competitors operating their own fabrication facilities", ["697eaee4"]),
    ("cybersecurity breaches and data protection incidents", ["e25e57b8"]),
    ("additional corporate tax liabilities", ["249bf4db"]),
]

EF_VALUES = [10, 20, 40, 80, 160]
K = 5


def connect():
    return psycopg2.connect(
        host=config.PG_HOST, port=config.PG_PORT, dbname=config.PG_DB,
        user=config.PG_USER, password=config.PG_PASSWORD,
    )


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def resolve_gold(cur, prefixes: list[str]) -> set[str]:
    ids = set()
    for p in prefixes:
        cur.execute("SELECT chunk_id FROM chunk_embeddings WHERE chunk_id LIKE %s", (p + "%",))
        ids.update(r[0] for r in cur.fetchall())
    return ids


def topk(cur, qvec: str, k: int, exact: bool) -> list[str]:
    # exact = force a sequential (brute-force) scan for ground truth;
    # otherwise use the HNSW index.
    cur.execute(f"SET enable_seqscan = {'on' if exact else 'off'};")
    cur.execute(f"SET enable_indexscan = {'off' if exact else 'on'};")
    cur.execute(
        "SELECT chunk_id FROM chunk_embeddings ORDER BY embedding <=> %s::vector LIMIT %s;",
        (qvec, k),
    )
    return [r[0] for r in cur.fetchall()]


def main() -> None:
    conn = connect()
    cur = conn.cursor()

    queries = [q for q, _ in LABELED]
    qvecs = [_vec(v) for v in embed_texts(queries)]
    golds = [resolve_gold(cur, gp) for _, gp in LABELED]
    missing = [q for (q, _), g in zip(LABELED, golds) if not g]
    if missing:
        print("WARNING: no chunk matched labels for:", missing)

    # Exact ground truth (brute force) per query — the recall reference.
    exact_topk = [topk(cur, qv, K, exact=True) for qv in qvecs]

    print(f"Labeled queries: {len(LABELED)}   k={K}   corpus=27 chunks\n")
    print(f"{'ef_search':>9}{'recall@'+str(K):>10}{'ann_recall@'+str(K):>14}{'avg ms':>9}")
    print("-" * 42)

    for ef in EF_VALUES:
        cur.execute("SET hnsw.ef_search = %s;", (ef,))
        hits = gold_total = ann_overlap = 0
        gold_count = 0
        t0 = time.perf_counter()
        for qv, gold, exact in zip(qvecs, golds, exact_topk):
            res = topk(cur, qv, K, exact=False)
            rset = set(res)
            hits += len(rset & gold)
            gold_total += len(gold)
            ann_overlap += len(rset & set(exact))
        dt_ms = (time.perf_counter() - t0) / len(qvecs) * 1000
        recall = hits / gold_total if gold_total else 0.0
        ann_recall = ann_overlap / (len(qvecs) * K)
        print(f"{ef:>9}{recall:>10.3f}{ann_recall:>14.3f}{dt_ms:>9.2f}")

    # Also show recall@1 at the default-ish ef for a stricter view.
    cur.execute("SET hnsw.ef_search = 40;")
    r1 = 0
    for qv, gold in zip(qvecs, golds):
        top1 = topk(cur, qv, 1, exact=False)
        r1 += 1 if set(top1) & gold else 0
    print(f"\nrecall@1 (ef_search=40): {r1}/{len(LABELED)} = {r1/len(LABELED):.3f}")
    conn.close()


if __name__ == "__main__":
    main()
