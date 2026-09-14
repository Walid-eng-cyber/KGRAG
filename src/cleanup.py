"""Post-extraction cleanup of the knowledge graph.

A free local model extracts good *shapes* but noisy *values*: duplicate nodes
('Apple' vs 'Apple Inc.'), tickers mislabeled as companies ('AAPL'), and junk
('A0'). This pass fixes that with deterministic rules (no LLM, no cost):

  1. Delete junk entities (tickers, digit-noise, too-short names).
  2. Merge duplicate entities that share a canonical name (Apple == Apple Inc.).
  3. Remove self-loops created by merging.

Run after ingest:
  python -m src.cleanup
"""
from __future__ import annotations

import re
from collections import defaultdict

from neo4j import GraphDatabase

import config

# Tickers we ingest — these show up mislabeled as COMPANY and should be dropped.
KNOWN_TICKERS = {
    "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "AMZN", "META", "TSLA", "A0",
}

# Legal suffixes / group words stripped when computing a canonical name.
_LEGAL = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|llc|ltd|limited|plc|"
    r"lp|holdings|group|the)\b"
)


def canonical(name: str) -> str:
    """Normalize a name so 'Apple Inc.' and 'Apple' collapse to the same key."""
    n = name.lower()
    n = re.sub(r"[.,'\"]", "", n)
    n = _LEGAL.sub(" ", n)
    n = re.sub(r"[^a-z0-9]+", " ", n).strip()
    return n


def is_junk(name: str, label: str) -> bool:
    n = (name or "").strip()
    if len(n) < 2:
        return True
    if re.search(r"\d", n) and len(n) <= 4:          # 'A0'
        return True
    if label == "COMPANY" and n.upper() in KNOWN_TICKERS:  # 'AAPL'
        return True
    return False


def _entities(session):
    """All non-Chunk entity nodes with their domain label + name."""
    q = """
    MATCH (n) WHERE NOT n:Chunk
    WITH n, [l IN labels(n) WHERE NOT l STARTS WITH '__'] AS labels
    WHERE size(labels) > 0 AND n.name IS NOT NULL
    RETURN elementId(n) AS eid, n.name AS name, labels[0] AS label
    """
    return [dict(r) for r in session.run(q)]


def run() -> None:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )
    with driver.session() as session:
        rows = _entities(session)
        print(f"Entities before cleanup: {len(rows)}")

        # --- 1. Delete junk ---
        junk = [r for r in rows if is_junk(r["name"], r["label"])]
        if junk:
            session.run(
                "MATCH (n) WHERE elementId(n) IN $ids DETACH DELETE n",
                ids=[r["eid"] for r in junk],
            )
        print(f"  deleted {len(junk)} junk entities: {[r['name'] for r in junk]}")

        # --- 2. Merge duplicates by (label, canonical name) ---
        survivors = [r for r in rows if r not in junk]
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in survivors:
            groups[(r["label"], canonical(r["name"]))].append(r)

        merged = 0
        for (label, _key), members in groups.items():
            if len(members) < 2:
                continue
            # Keep the most descriptive name (longest) as the survivor.
            members.sort(key=lambda r: len(r["name"]), reverse=True)
            survivor, others = members[0], members[1:]
            session.run(
                """
                MATCH (s) WHERE elementId(s) = $sid
                MATCH (o) WHERE elementId(o) IN $oids
                WITH s, collect(o) AS os
                CALL apoc.refactor.mergeNodes([s] + os,
                    {properties: 'discard', mergeRels: true}) YIELD node
                RETURN node
                """,
                sid=survivor["eid"],
                oids=[o["eid"] for o in others],
            )
            merged += len(others)
            print(f"  merged {[o['name'] for o in others]} -> {survivor['name']!r}")

        # --- 3. Remove self-loops introduced by merging ---
        session.run("MATCH (n)-[r]->(n) DELETE r")

        after = session.run(
            "MATCH (n) WHERE NOT n:Chunk RETURN count(n) AS n"
        ).single()["n"]
        print(f"\nMerged away {merged} duplicate nodes.")
        print(f"Entities after cleanup: {after}")
    driver.close()


if __name__ == "__main__":
    run()
