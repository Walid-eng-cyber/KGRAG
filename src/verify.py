"""Sanity-check the graph that Phase 1 built.

Run after ingest:
  python -m src.verify
"""
from __future__ import annotations

from neo4j import GraphDatabase

import config

QUERIES: list[tuple[str, str]] = [
    ("Total nodes", "MATCH (n) RETURN count(n) AS n"),
    ("Total relationships", "MATCH ()-[r]->() RETURN count(r) AS n"),
    (
        "Node counts by type",
        """
        MATCH (n)
        UNWIND labels(n) AS label
        WITH label WHERE NOT label STARTS WITH '__'
        RETURN label, count(*) AS n ORDER BY n DESC
        """,
    ),
    (
        "Relationship counts by type",
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC",
    ),
    (
        "Sample: companies and their competitors",
        """
        MATCH (a)-[r:COMPETES_WITH]->(b)
        RETURN a.name AS company, b.name AS competitor LIMIT 15
        """,
    ),
    (
        "Sample: risks faced (multi-hop entry point)",
        """
        MATCH (c)-[:FACES_RISK]->(risk)
        RETURN c.name AS company, risk.name AS risk LIMIT 15
        """,
    ),
]


def run() -> None:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )
    with driver.session() as session:
        for title, cypher in QUERIES:
            print(f"\n=== {title} ===")
            for record in session.run(cypher):
                print("  ", dict(record))
    driver.close()


if __name__ == "__main__":
    run()
