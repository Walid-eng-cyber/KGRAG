"""Admin CLI for corpora.

    python -m server.corpus_admin list
    python -m server.corpus_admin create-sec --owner you@gmail.com [--name "SEC insider trading"]
    python -m server.corpus_admin adopt-documents --owner you@gmail.com --name "Walid" --preset career

`create-sec` registers the loaded SEC Form 3/4/5 data as a private corpus and tags
its nodes. `adopt-documents` moves documents uploaded before corpora existed into a
new private corpus, removes their old unscoped graph data, and marks them for
re-ingestion (press Ingest in the UI afterwards).
"""
from __future__ import annotations

import argparse
import os

import config
from server.corpora import CorpusStore
from server.doc_store import DocumentStore

SEC_SCHEMA = {"preset": "sec_insiders", "entities": ["COMPANY", "PERSON"],
              "relations": [["PERSON", "INSIDER_OF", "COMPANY"], ["PERSON", "TRADED", "COMPANY"]]}


def _stores() -> tuple[CorpusStore, DocumentStore]:
    db = os.getenv("DOCUMENTS_DB", str(config.DATA_DIR / "documents.db"))
    return CorpusStore(db), DocumentStore(db, os.getenv("UPLOAD_DIR", str(config.DATA_DIR / "uploads")))


def create_sec(owner: str | None, name: str) -> dict:
    from neo4j import GraphDatabase
    corpora, _ = _stores()
    existing = [c for c in corpora.list_for(owner) if c["kind"] == "sec_insiders"]
    c = existing[0] if existing else corpora.create(
        name, "Insider transactions from SEC Forms 3/4/5 (directors, officers, 10% owners).",
        "sec_insiders", SEC_SCHEMA, owner)
    d = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
                             notifications_min_severity="OFF")
    try:
        with d.session() as s:
            n = s.run("MATCH (n:SecEntity) SET n.corpus = $c RETURN count(n) AS n", c=c["id"]).single()["n"]
    finally:
        d.close()
    print(f"SEC corpus {c['id']} ({c['name']}) owned by {owner or 'local user'}: {n:,} nodes attached.")
    return c


def adopt_documents(owner: str | None, name: str, preset: str, description: str) -> dict:
    from src import corpus_schema
    from src.documents import remove_document as remove_legacy
    corpora, store = _stores()
    loose = [d for d in store.list(None) if d["status"] not in ("queued", "ingesting")]
    if not loose:
        print("No documents outside a corpus.")
    c = corpora.create(name, description, "documents", corpus_schema.from_preset(preset), owner)
    for d in loose:
        if d["status"] == "ready":
            remove_legacy(d["id"])          # old unscoped chunks + entities only it mentioned
        store.update(d["id"], corpus_id=c["id"], status="uploaded", progress=None, chunks=None,
                     message="Moved into a corpus. Ingest to extract it with the corpus schema.")
        print(f"  moved {d['title']!r}")
    print(f"Corpus {c['id']} ({c['name']}, preset {preset}) owned by {owner or 'local user'}: {len(loose)} document(s).")
    return c


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m server.corpus_admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sec = sub.add_parser("create-sec")
    sec.add_argument("--owner")
    sec.add_argument("--name", default="SEC insider trading")
    ad = sub.add_parser("adopt-documents")
    ad.add_argument("--owner")
    ad.add_argument("--name", required=True)
    ad.add_argument("--preset", default="general")
    ad.add_argument("--description", default="")
    a = ap.parse_args()
    if a.cmd == "list":
        _stores()   # make sure both tables exist
        import sqlite3
        db = sqlite3.connect(os.getenv("DOCUMENTS_DB", str(config.DATA_DIR / "documents.db")))
        for cid, name, kind, owner in db.execute("SELECT id, name, kind, owner FROM corpora ORDER BY created_at"):
            n = db.execute("SELECT count(*) FROM documents WHERE corpus_id = ?", (cid,)).fetchone()[0]
            print(f"{cid}  {kind:<13} {name:<30} owner={owner or 'local'}  documents={n}")
        loose = db.execute("SELECT count(*) FROM documents WHERE corpus_id IS NULL").fetchone()[0]
        print(f"documents outside any corpus: {loose}")
    elif a.cmd == "create-sec":
        create_sec(a.owner, a.name)
    else:
        adopt_documents(a.owner, a.name, a.preset, a.description)


if __name__ == "__main__":
    main()
