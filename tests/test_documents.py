"""Uploaded documents: validation, permissions, the ingest worker and text extraction.
The LLM/graph ingestion step is stubbed, so no Neo4j, Postgres or Ollama is needed."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server import auth
from server.doc_store import DocumentStore, IngestWorker
from src import documents as docs

TEXT = "Acme Corp is a manufacturer of widgets. " * 20


@pytest.fixture
def store(tmp_path, monkeypatch):
    calls = []

    def fake_ingest(doc_id, title, text, progress):
        progress("Extracting graph: 1/1 chunks", 0.5)
        calls.append((doc_id, title, len(text)))
        if "explode" in title:
            raise RuntimeError("bolt://secret-host refused")
        return {"chunks": 3}

    s = DocumentStore(tmp_path / "documents.db", tmp_path / "uploads")
    w = IngestWorker(s, docs.extract_text, fake_ingest)
    monkeypatch.setattr(main, "doc_store", s)
    monkeypatch.setattr(main, "ingest_worker", w)
    monkeypatch.setattr(docs, "remove_document", lambda doc_id: calls.append(("removed", doc_id)))
    s.calls = calls
    return s


def wait_for(store, doc_id, statuses=("ready", "failed"), timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        d = store.get(doc_id)
        if d["status"] in statuses:
            return d
        time.sleep(0.05)
    raise AssertionError(f"{doc_id} stuck in {store.get(doc_id)['status']}")


def upload(c, name="report.txt", data=TEXT.encode(), title=""):
    return c.post("/api/documents", files={"file": (name, data)}, data={"title": title})


def test_upload_then_ingest_reaches_ready(store):
    c = TestClient(main.app)   # auth off (conftest): local single-user mode
    doc = upload(c, title="Acme annual report").json()["document"]
    assert doc["status"] == "uploaded" and doc["title"] == "Acme annual report" and doc["kind"] == "text"
    assert store.path(store.get(doc["id"])).exists() and "report.txt" not in str(store.path(store.get(doc["id"])))
    assert c.post("/api/documents/ingest").json()["queued"] == [doc["id"]]
    done = wait_for(store, doc["id"])
    assert done["status"] == "ready" and done["chunks"] == 3 and done["progress"] == 1.0
    assert store.calls == [(doc["id"], "Acme annual report", len(TEXT.strip()))]
    assert [d["id"] for d in c.get("/api/documents").json()["documents"]] == [doc["id"]]


def test_failed_ingest_hides_internals_and_can_be_retried(store):
    c = TestClient(main.app)
    doc = upload(c, title="explode").json()["document"]
    c.post("/api/documents/ingest")
    failed = wait_for(store, doc["id"])
    assert failed["status"] == "failed" and "secret-host" not in failed["message"]
    assert c.post("/api/documents/ingest").json()["queued"] == [doc["id"]]  # failed docs are re-queued


@pytest.mark.parametrize("name,data,code", [
    ("malware.exe", b"MZ" + b"x" * 300, 415),
    ("fake.pdf", b"not a pdf at all" * 20, 415),
    ("empty.txt", b"", 400),
])
def test_upload_rejects_bad_files(store, name, data, code):
    assert upload(TestClient(main.app), name, data).status_code == code


def test_upload_size_limit(store, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 1)
    assert upload(TestClient(main.app), "big.txt", b"a" * (1024 * 1024 + 10)).status_code == 413


def test_too_little_text_fails_with_a_clear_reason(store):
    c = TestClient(main.app)
    doc = upload(c, "tiny.txt", b"hello world").json()["document"]
    c.post("/api/documents/ingest")
    assert "Too little text" in wait_for(store, doc["id"])["message"]


def test_duplicate_content_is_refused(store):
    c = TestClient(main.app)
    first = upload(c, "a.txt").json()["document"]
    c.post("/api/documents/ingest")
    wait_for(store, first["id"])
    second = upload(c, "b.txt").json()["document"]
    c.post("/api/documents/ingest")
    assert "already ingested" in wait_for(store, second["id"])["message"]


def test_delete_removes_graph_data_for_ready_documents(store):
    c = TestClient(main.app)
    doc = upload(c).json()["document"]
    c.post("/api/documents/ingest")
    wait_for(store, doc["id"])
    assert c.delete(f"/api/documents/{doc['id']}").json() == {"ok": True}
    assert ("removed", doc["id"]) in store.calls and store.get(doc["id"]) is None
    assert c.delete(f"/api/documents/{doc['id']}").status_code == 404


def test_restart_marks_running_jobs_as_interrupted(tmp_path):
    s = DocumentStore(tmp_path / "d.db", tmp_path / "u")
    doc = s.add("x.txt", "x", "text", ".txt", TEXT.encode(), None)
    s.update(doc["id"], status="ingesting")
    reader = DocumentStore(tmp_path / "d.db", tmp_path / "u")          # e.g. a script reading the list
    assert reader.get(doc["id"])["status"] == "ingesting"               # must not touch live jobs
    again = DocumentStore(tmp_path / "d.db", tmp_path / "u", recover=True)   # server startup
    assert again.get(doc["id"])["status"] == "failed" and "restart" in again.get(doc["id"])["message"]


# ---------- permissions with accounts enabled ----------

@pytest.fixture
def accounts(tmp_path, monkeypatch):
    env = {"GOOGLE_CLIENT_ID": "test", "ADMIN_EMAILS": "boss@gmail.com", "SESSION_SECRET": "x" * 40,
           "COOKIE_SECURE": "0", "USERS_DB": str(tmp_path / "users.db")}
    monkeypatch.setattr(auth, "settings", auth.Settings(env))
    claims = {"boss": {"email": "boss@gmail.com", "email_verified": True, "name": "Boss"},
              "user": {"email": "user@gmail.com", "email_verified": True, "name": "User"}}
    monkeypatch.setattr(auth, "verify_google_token", lambda cred: claims[cred.split(":")[1].strip("-")])

    def client(who):
        c = TestClient(main.app)
        if who == "user":
            auth.settings.store.create("user@gmail.com", "User", None, admin=False, updates_opt_in=False)
            auth.settings.store.set_status("user@gmail.com", "approved", by="test")
        c.post("/api/auth/google", json={"credential": f"token:{who}" + "-" * 20})
        return c
    return client


def test_only_admins_upload_ingest_and_delete(store, accounts):
    anon = TestClient(main.app)
    assert anon.get("/api/documents").status_code == 401
    user = accounts("user")
    assert user.get("/api/documents").status_code == 200          # everyone signed in can see the list
    assert upload(user).status_code == 403
    assert user.post("/api/documents/ingest").status_code == 403
    boss = accounts("boss")
    doc = upload(boss).json()["document"]
    assert doc["uploaded_by"] == "boss@gmail.com"
    assert user.delete(f"/api/documents/{doc['id']}").status_code == 403


# ---------- text extraction ----------

def test_extract_text_kinds(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\n" + TEXT)
    (tmp_path / "b.html").write_text(f"<html><body><script>x()</script><p>{TEXT}</p></body></html>")
    assert docs.extract_text(tmp_path / "a.md", docs.kind_for("a.md")).startswith("# Title")
    html = docs.extract_text(tmp_path / "b.html", docs.kind_for("b.html"))
    assert "Acme Corp" in html and "x()" not in html
    with pytest.raises(docs.DocumentError, match="Unsupported"):
        docs.kind_for("x.docx")


def test_extract_pdf(tmp_path):
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    p = tmp_path / "blank.pdf"
    with open(p, "wb") as fh:
        w.write(fh)
    with pytest.raises(docs.DocumentError, match="Too little text"):   # image-only / blank PDFs
        docs.extract_text(p, "pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7 garbage" * 10)
    with pytest.raises(docs.DocumentError):
        docs.extract_text(bad, "pdf")


def test_extraction_instructions_never_reach_the_index():
    body = "Zentrix depends on a few semiconductor suppliers."
    first = f"{docs.DOC_HINT}\n\nDocument: Report\n\n{body}"
    assert docs.strip_hint(first, "Report") == body
    assert docs.strip_hint(body, "Report") == body


def test_graph_source_passages_are_kept():
    from src.answer import _dedup_passages
    hits = [{"chunk_id": "a"}, {"chunk_id": "a"}, {"chunk_id": "b"}]
    assert [h["chunk_id"] for h in _dedup_passages(hits, {"a"})] == ["a", "b"]
