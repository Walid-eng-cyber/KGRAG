"""Corpora: private collections of documents, each with its own profile and schema.

Every corpus belongs to one user and is visible only to them (on a local setup
with sign-in off there is a single implicit user, `owner = NULL`).

kind:
  documents     uploaded files, extracted with the corpus's own schema
  sec_insiders  the structured SEC Form 3/4/5 data set (loaded by src/form4.py)
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("documents", "sec_insiders")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL CHECK (kind IN ('documents','sec_insiders')),
    schema_json TEXT NOT NULL,
    owner       TEXT,
    created_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CorpusStore:
    def __init__(self, db_path: str | Path) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["schema"] = json.loads(d.pop("schema_json"))
        return d

    def get(self, corpus_id: str) -> dict | None:
        r = self._db.execute("SELECT * FROM corpora WHERE id = ?", (corpus_id,)).fetchone()
        return self._row(r) if r else None

    def list_for(self, owner: str | None) -> list[dict]:
        """Corpora this user can see: their own (always private)."""
        if owner is None:
            rows = self._db.execute("SELECT * FROM corpora WHERE owner IS NULL ORDER BY created_at").fetchall()
        else:
            rows = self._db.execute("SELECT * FROM corpora WHERE owner = ? ORDER BY created_at", (owner,)).fetchall()
        return [self._row(r) for r in rows]

    def visible(self, corpus_id: str, owner: str | None) -> dict | None:
        c = self.get(corpus_id)
        return c if c and c["owner"] == owner else None

    def create(self, name: str, description: str, kind: str, schema: dict, owner: str | None,
               corpus_id: str | None = None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        cid = corpus_id or "c_" + secrets.token_hex(5)
        with self._lock:
            self._db.execute("INSERT INTO corpora (id, name, description, kind, schema_json, owner, created_at) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (cid, name.strip()[:80], description.strip()[:500], kind, json.dumps(schema), owner, _now()))
            self._db.commit()
        return self.get(cid)  # type: ignore[return-value]

    def update(self, corpus_id: str, **fields) -> dict | None:
        allowed = {}
        if "name" in fields:
            allowed["name"] = str(fields["name"]).strip()[:80]
        if "description" in fields:
            allowed["description"] = str(fields["description"]).strip()[:500]
        if "schema" in fields:
            allowed["schema_json"] = json.dumps(fields["schema"])
        if allowed:
            cols = ", ".join(f"{k} = ?" for k in allowed)
            with self._lock:
                self._db.execute(f"UPDATE corpora SET {cols} WHERE id = ?", (*allowed.values(), corpus_id))
                self._db.commit()
        return self.get(corpus_id)

    def delete(self, corpus_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM corpora WHERE id = ?", (corpus_id,))
            self._db.commit()
