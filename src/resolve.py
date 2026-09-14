"""Entity resolution — the hard part that most pipelines skip.

`Acme Corp`, `Acme Corporation`, and `ACME` are three surface forms of ONE
real-world company. If they stay as three nodes, every query fragments and the
graph lies. This module collapses them into a single node and remembers every
surface form as an alias list.

Two-stage matching (per entity type — we never merge a PERSON into a COMPANY):
  1. Normalize + exact match  — cheap, high precision ('Acme Corp' == 'ACME'
     after stripping legal suffixes / case / punctuation).
  2. Embedding similarity      — catches variants normalization misses. Names
     are embedded with a local model (bge-m3) and any pair whose cosine
     similarity clears a tuned threshold is merged.

Matches are grouped transitively (A~B, B~C => one node) via union-find, the
fullest name is kept, all relationships are rewired (APOC), and every surface
form is stored on the survivor's `aliases` property.

Usage:
  python -m src.resolve --dry-run      # preview matches + scores (tune threshold)
  python -m src.resolve                # apply
  python -m src.resolve --threshold 0.93
"""
from __future__ import annotations

import argparse
import math

import ollama
from neo4j import GraphDatabase

import config
from src.cleanup import canonical


# --------------------------- union-find ---------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self):
        out: dict = {}
        for i in self.parent:
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def embed(names: list[str]) -> list[list[float]]:
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    return client.embed(model=config.EMBED_MODEL, input=names)["embeddings"]


def _entities_by_label(session) -> dict[str, list[dict]]:
    q = """
    MATCH (n) WHERE NOT n:Chunk
    WITH n, [l IN labels(n) WHERE NOT l STARTS WITH '__'] AS labels
    WHERE size(labels) > 0 AND n.name IS NOT NULL
    RETURN elementId(n) AS eid, n.name AS name, labels[0] AS label
    """
    groups: dict[str, list[dict]] = {}
    for r in session.run(q):
        groups.setdefault(r["label"], []).append({"eid": r["eid"], "name": r["name"]})
    return groups


def resolve(threshold: float, dry_run: bool) -> None:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )
    with driver.session() as session:
        by_label = _entities_by_label(session)
        total_merged = 0

        for label, members in by_label.items():
            if len(members) < 2:
                continue
            uf = UnionFind([m["eid"] for m in members])

            # Stage 1: exact match on normalized name.
            canon: dict[str, str] = {}  # canonical -> first eid seen
            for m in members:
                key = canonical(m["name"])
                if key in canon:
                    uf.union(canon[key], m["eid"])
                else:
                    canon[key] = m["eid"]

            # Stage 2: embedding similarity on the remaining distinct names.
            names = [m["name"] for m in members]
            vecs = embed(names)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if uf.find(members[i]["eid"]) == uf.find(members[j]["eid"]):
                        continue  # already matched in stage 1
                    score = cosine(vecs[i], vecs[j])
                    if score >= threshold:
                        uf.union(members[i]["eid"], members[j]["eid"])
                        if dry_run:
                            print(
                                f"  [{label}] MATCH {score:.3f}: "
                                f"{names[i]!r} ~ {names[j]!r}"
                            )
                    elif dry_run and score >= threshold - 0.08:
                        print(
                            f"  [{label}] near  {score:.3f}: "
                            f"{names[i]!r} ? {names[j]!r}  (below {threshold})"
                        )

            by_eid = {m["eid"]: m for m in members}
            for group in uf.groups():
                if len(group) < 2:
                    continue
                names_in = [by_eid[e]["name"] for e in group]
                survivor_eid = max(group, key=lambda e: len(by_eid[e]["name"]))
                survivor_name = by_eid[survivor_eid]["name"]
                aliases = sorted({n for n in names_in})
                total_merged += len(group) - 1

                if dry_run:
                    print(f"  => {survivor_name!r}  aliases={aliases}")
                    continue

                others = [e for e in group if e != survivor_eid]
                session.run(
                    """
                    MATCH (s) WHERE elementId(s) = $sid
                    MATCH (o) WHERE elementId(o) IN $oids
                    WITH s, collect(o) AS os
                    CALL apoc.refactor.mergeNodes([s] + os,
                        {properties: 'discard', mergeRels: true}) YIELD node
                    SET node.aliases = $aliases, node.name = $name
                    RETURN node
                    """,
                    sid=survivor_eid, oids=others,
                    aliases=aliases, name=survivor_name,
                )

        if not dry_run:
            session.run("MATCH (n)-[r]->(n) DELETE r")  # drop self-loops from merges
        verb = "Would merge" if dry_run else "Merged"
        print(f"\n{verb} {total_merged} duplicate node(s) (threshold {threshold}).")
    driver.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Resolve duplicate entities.")
    p.add_argument("--dry-run", action="store_true", help="preview, don't modify")
    p.add_argument("--threshold", type=float, default=config.RESOLVE_THRESHOLD)
    args = p.parse_args()
    resolve(args.threshold, args.dry_run)
