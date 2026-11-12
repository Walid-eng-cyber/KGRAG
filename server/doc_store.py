"""Uploaded documents: registry (SQLite), file storage, and a one-at-a-time ingest worker.

Statuses:  uploaded -> queued -> ingesting -> ready
                                          \\-> failed (error kept; can be re-queued)
Files live in data/uploads/<id><ext> under server-generated names, never the
uploader's filename. Ingestion runs on a single background thread because the
local models handle one extraction at a time.
"""
from __future__ import annotations

import queue
import secrets
import sqlite3
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    title        TEXT NOT NULL,
    kind         TEXT NOT NULL,
    ext          TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    chars        INTEGER,
    content_hash TEXT,
    status       TEXT NOT NULL CHECK (status IN ('uploaded','queued','ingesting','ready','failed')),
    progress     REAL,
    message      TEXT,
    chunks       INTEGER,
    uploaded_by  TEXT,
    uploaded_at  TEXT NOT NULL,
    ingested_at  TEXT
);
"""
_COLS = ("id", "filename", "title", "kind", "size_bytes", "chars", "status", "progress", "message",
         "chunks", "uploaded_by", "uploaded_at", "ingested_at")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentStore:
    def __init__(self, db_path: str | Path, upload_dir: str | Path, recover: bool = False) -> None:
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()
        if recover:
            self.mark_interrupted()

    def mark_interrupted(self) -> None:
        """Only for server startup: jobs that were queued or running when the previous
        process stopped can't still be running, so mark them for re-queueing.
        (Never call this while a server is running: it would orphan its live jobs.)"""
        with self._lock:
            self._db.execute("UPDATE documents SET status='failed', message='Interrupted by a server restart. "
                             "Ingest again to retry.' WHERE status IN ('queued','ingesting')")
            self._db.commit()

    # ---- reads ----
    def get(self, doc_id: str) -> dict | None:
        r = self._db.execute(f"SELECT {', '.join(_COLS)}, ext, content_hash FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(r) if r else None

    def list(self) -> list[dict]:
        rows = self._db.execute(f"SELECT {', '.join(_COLS)} FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [dict(r) for r in rows]

    def by_hash(self, h: str) -> dict | None:
        r = self._db.execute("SELECT id FROM documents WHERE content_hash=?", (h,)).fetchone()
        return self.get(r["id"]) if r else None

    def path(self, doc: dict) -> Path:
        return self.upload_dir / f"{doc['id']}{doc['ext']}"

    # ---- writes ----
    def add(self, filename: str, title: str, kind: str, ext: str, data: bytes, uploaded_by: str | None) -> dict:
        doc_id = "DOC-" + secrets.token_hex(5)
        (self.upload_dir / f"{doc_id}{ext}").write_bytes(data)
        with self._lock:
            self._db.execute(
                "INSERT INTO documents (id, filename, title, kind, ext, size_bytes, status, uploaded_by, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'uploaded', ?, ?)",
                (doc_id, filename, title, kind, ext, len(data), uploaded_by, _now()))
            self._db.commit()
        return self.get(doc_id)  # type: ignore[return-value]

    def update(self, doc_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE documents SET {cols} WHERE id=?", (*fields.values(), doc_id))
            self._db.commit()

    def delete(self, doc_id: str) -> None:
        doc = self.get(doc_id)
        if doc:
            self.path(doc).unlink(missing_ok=True)
        with self._lock:
            self._db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            self._db.commit()


class IngestWorker:
    """One background thread that ingests queued documents in order."""

    def __init__(self, store: DocumentStore, extract, ingest) -> None:
        self.store, self._extract, self._ingest = store, extract, ingest
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    def enqueue(self, doc_id: str) -> None:
        self.store.update(doc_id, status="queued", progress=0.0, message="Waiting to start")
        self._q.put(doc_id)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="kgrag-ingest", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                doc_id = self._q.get(timeout=5)
            except queue.Empty:
                return
            self.process(doc_id)

    def process(self, doc_id: str) -> None:
        doc = self.store.get(doc_id)
        if not doc or doc["status"] != "queued":
            return
        self.store.update(doc_id, status="ingesting", progress=0.0, message="Reading the file")
        try:
            text = self._extract(self.store.path(doc), doc["kind"])
            dup = self.store.by_hash(_hash(text))
            if dup and dup["id"] != doc_id and dup["status"] == "ready":
                raise ValueError(f"Same content as “{dup['title']}”, which is already ingested.")
            self.store.update(doc_id, chars=len(text), content_hash=_hash(text))

            def progress(msg: str, frac: float | None = None) -> None:
                self.store.update(doc_id, message=msg, **({"progress": round(frac, 3)} if frac is not None else {}))

            result = self._ingest(doc_id, doc["title"], text, progress)
            self.store.update(doc_id, status="ready", progress=1.0, chunks=result.get("chunks"),
                              message=f"Ready: {result.get('chunks', 0)} chunks in the graph and vector index",
                              ingested_at=_now())
        except Exception as e:  # noqa: BLE001 - surface a short reason; details go to the log
            traceback.print_exc()
            reason = str(e) if isinstance(e, ValueError) else (
                "Ingestion failed: a backend service (Neo4j, Postgres or Ollama) isn't available, "
                "or the model couldn't process this file. The details are in the server log.")
            self.store.update(doc_id, status="failed", message=reason[:300])


def _hash(text: str) -> str:
    from src.documents import content_hash
    return content_hash(text)
