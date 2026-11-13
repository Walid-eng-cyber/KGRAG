"""Documents inside corpora: validation, privacy, the ingest worker and text extraction.
The LLM/graph ingestion step is stubbed, so no Neo4j, Postgres or Ollama is needed."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server import auth
from server.corpora import CorpusStore
from server.doc_store import DocumentStore, IngestWorker
from src import documents as docs

TEXT = "Acme Corp is a manufacturer of widgets. " * 20


@pytest.fixture
def store(tmp_path, monkeypatch):
    calls = []

    def fake_ingest(doc, text, progress):
        progress("Extracting graph: 1/1 chunks", 0.5)
        corpus = main.corpus_store.get(doc["corpus_id"])
        calls.append((doc["id"], corpus["id"], doc["title"], len(text)))
        if "explode" in doc["title"]:
            raise RuntimeError("bolt://secret-host refused")
        return {"chunks": 3, "entities": 5, "relations": 2}

    s = DocumentStore(tmp_path / "documents.db", tmp_path / "uploads")
    monkeypatch.setattr(main, "corpus_store", CorpusStore(tmp_path / "documents.db"))
    monkeypatch.setattr(main, "doc_store", s)
    monkeypatch.setattr(main, "ingest_worker", IngestWorker(s, docs.extract_text, fake_ingest))
    monkeypatch.setattr(main.corpus_graph, "remove_document", lambda c, d: calls.append(("removed", c, d)))
    monkeypatch.setattr(main.corpus_graph, "remove_corpus", lambda c: calls.append(("removed-corpus", c)))
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


def new_corpus(c, name="Acme", **body):
    r = c.post("/api/corpora", json={"name": name, **body})
    assert r.status_code == 200, r.text
    return r.json()["corpus"]["id"]


def upload(c, cid, name="report.txt", data=TEXT.encode(), title=""):
    return c.post(f"/api/corpora/{cid}/documents", files={"file": (name, data)}, data={"title": title})


def test_upload_then_ingest_reaches_ready(store):
    c = TestClient(main.app)   # auth off (conftest): local single-user mode
    cid = new_corpus(c)
    doc = upload(c, cid, title="Acme annual report").json()["document"]
    assert doc["status"] == "uploaded" and doc["corpus_id"] == cid and doc["kind"] == "text"
    assert "report.txt" not in str(store.path(store.get(doc["id"])))
    assert c.post(f"/api/corpora/{cid}/ingest").json()["queued"] == [doc["id"]]
    done = wait_for(store, doc["id"])
    assert done["status"] == "ready" and done["chunks"] == 3 and "5 entities" in done["message"]
    assert store.calls == [(doc["id"], cid, "Acme annual report", len(TEXT.strip()))]
    assert [d["id"] for d in c.get(f"/api/corpora/{cid}/documents").json()["documents"]] == [doc["id"]]


def test_documents_stay_in_their_corpus(store):
    c = TestClient(main.app)
    a, b = new_corpus(c, "A"), new_corpus(c, "B")
    upload(c, a, "a.txt")
    assert len(c.get(f"/api/corpora/{a}/documents").json()["documents"]) == 1
    assert c.get(f"/api/corpora/{b}/documents").json()["documents"] == []
    assert c.post(f"/api/corpora/{b}/ingest").json()["queued"] == []
    corpora = {x["name"]: x for x in c.get("/api/corpora").json()["corpora"]}
    assert (corpora["A"]["documents"], corpora["B"]["documents"]) == (1, 0)


def test_failed_ingest_hides_internals_and_can_be_retried(store):
    c = TestClient(main.app)
    cid = new_corpus(c)
    doc = upload(c, cid, title="explode").json()["document"]
    c.post(f"/api/corpora/{cid}/ingest")
    failed = wait_for(store, doc["id"])
    assert failed["status"] == "failed" and "secret-host" not in failed["message"]
    assert c.post(f"/api/corpora/{cid}/ingest").json()["queued"] == [doc["id"]]


@pytest.mark.parametrize("name,data,code", [
    ("malware.exe", b"MZ" + b"x" * 300, 415),
    ("fake.pdf", b"not a pdf at all" * 20, 415),
    ("empty.txt", b"", 400),
])
def test_upload_rejects_bad_files(store, name, data, code):
    c = TestClient(main.app)
    assert upload(c, new_corpus(c), name, data).status_code == code


def test_upload_size_limit(store, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 1)
    c = TestClient(main.app)
    assert upload(c, new_corpus(c), "big.txt", b"a" * (1024 * 1024 + 10)).status_code == 413


def test_too_little_text_fails_with_a_clear_reason(store):
    c = TestClient(main.app)
    cid = new_corpus(c)
    doc = upload(c, cid, "tiny.txt", b"hello world").json()["document"]
    c.post(f"/api/corpora/{cid}/ingest")
    assert "Too little text" in wait_for(store, doc["id"])["message"]


def test_duplicate_content_is_refused_within_a_corpus_only(store):
    c = TestClient(main.app)
    a, b = new_corpus(c, "A"), new_corpus(c, "B")
    first = upload(c, a, "a.txt").json()["document"]
    c.post(f"/api/corpora/{a}/ingest")
    wait_for(store, first["id"])
    second = upload(c, a, "b.txt").json()["document"]
    c.post(f"/api/corpora/{a}/ingest")
    assert "already ingested" in wait_for(store, second["id"])["message"]
    other = upload(c, b, "c.txt").json()["document"]          # same text, different corpus: fine
    c.post(f"/api/corpora/{b}/ingest")
    assert wait_for(store, other["id"])["status"] == "ready"


def test_delete_document_and_corpus(store):
    c = TestClient(main.app)
    cid = new_corpus(c)
    doc = upload(c, cid).json()["document"]
    c.post(f"/api/corpora/{cid}/ingest")
    wait_for(store, doc["id"])
    assert c.delete(f"/api/documents/{doc['id']}").json() == {"ok": True}
    assert ("removed", cid, doc["id"]) in store.calls and store.get(doc["id"]) is None
    upload(c, cid)
    assert c.delete(f"/api/corpora/{cid}").json() == {"ok": True}
    assert ("removed-corpus", cid) in store.calls
    assert c.get(f"/api/corpora/{cid}/documents").status_code == 404 and store.list(cid) == []


def test_restart_recovery_runs_only_when_asked(tmp_path):
    s = DocumentStore(tmp_path / "d.db", tmp_path / "u")
    doc = s.add("x.txt", "x", "text", ".txt", TEXT.encode(), None, "c_1")
    s.update(doc["id"], status="ingesting")
    reader = DocumentStore(tmp_path / "d.db", tmp_path / "u")          # e.g. a script or a test import
    assert reader.get(doc["id"])["status"] == "ingesting"               # must not touch live jobs
    DocumentStore(tmp_path / "d.db", tmp_path / "u", recover=True)      # server startup
    assert "restart" in reader.get(doc["id"])["message"]


# ---------- privacy with accounts enabled ----------

@pytest.fixture
def accounts(tmp_path, monkeypatch):
    env = {"GOOGLE_CLIENT_ID": "test", "ADMIN_EMAILS": "boss@gmail.com", "SESSION_SECRET": "x" * 40,
           "COOKIE_SECURE": "0", "USERS_DB": str(tmp_path / "users.db")}
    monkeypatch.setattr(auth, "settings", auth.Settings(env))
    claims = {w: {"email": f"{w}@gmail.com", "email_verified": True, "name": w.title()} for w in ("boss", "ann", "bob")}
    monkeypatch.setattr(auth, "verify_google_token", lambda cred: claims[cred.split(":")[1].strip("-")])

    def client(who):
        c = TestClient(main.app)
        if who != "boss" and not auth.settings.store.get(f"{who}@gmail.com"):
            auth.settings.store.create(f"{who}@gmail.com", who, None, admin=False, updates_opt_in=False)
            auth.settings.store.set_status(f"{who}@gmail.com", "approved", by="test")
        assert c.post("/api/auth/google", json={"credential": f"token:{who}" + "-" * 20}).status_code == 200
        return c
    return client


def test_corpora_are_private_to_their_owner(store, accounts):
    assert TestClient(main.app).get("/api/corpora").status_code == 401
    ann, bob = accounts("ann"), accounts("bob")
    cid = new_corpus(ann, "Ann's notes")
    doc = upload(ann, cid).json()["document"]
    assert [c["name"] for c in ann.get("/api/corpora").json()["corpora"]] == ["Ann's notes"]
    assert bob.get("/api/corpora").json()["corpora"] == []
    for r in (bob.get(f"/api/corpora/{cid}/documents"), upload(bob, cid), bob.post(f"/api/corpora/{cid}/ingest"),
              bob.delete(f"/api/documents/{doc['id']}"), bob.delete(f"/api/corpora/{cid}"),
              bob.post("/api/ask", json={"question": "anything here?", "corpus_id": cid})):
        assert r.status_code == 404           # indistinguishable from "doesn't exist"
    assert accounts("boss").get("/api/corpora").json()["corpora"] == []   # admins don't see others' corpora either


def test_a_document_can_be_read_by_its_owner_and_nobody_else(store, accounts):
    ann, bob = accounts("ann"), accounts("bob")
    cid = new_corpus(ann, "Ann's notes")
    doc = upload(ann, cid, name="cv.txt", data=b"Elodie Marchand, data engineer in Lyon. " * 8, title="Elodie CV").json()["document"]
    r = ann.get(f"/api/documents/{doc['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["document"]["title"] == "Elodie CV" and body["document"]["filename"] == "cv.txt"
    assert body["text"].startswith("Elodie Marchand, data engineer in Lyon.") and body["truncated"] is False
    assert bob.get(f"/api/documents/{doc['id']}").status_code == 404          # someone else's
    assert ann.get("/api/documents/DOC-nope").status_code == 404              # missing looks the same
    assert TestClient(main.app).get(f"/api/documents/{doc['id']}").status_code == 401


def test_a_long_document_is_cut_for_the_viewer_and_says_so(store, accounts, monkeypatch):
    monkeypatch.setattr(main, "MAX_VIEW_CHARS", 50)
    ann = accounts("ann")
    cid = new_corpus(ann)
    doc = upload(ann, cid, data=b"abcdefghij " * 20).json()["document"]
    body = ann.get(f"/api/documents/{doc['id']}").json()
    assert len(body["text"]) == 50 and body["truncated"] is True and body["chars"] > 50


def test_signed_in_questions_must_name_a_corpus(store, accounts):
    r = accounts("ann").post("/api/ask", json={"question": "What is in my documents?"})
    assert r.status_code == 400 and "corpus" in r.json()["detail"]


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
    with pytest.raises(docs.DocumentError, match="Too little text"):
        docs.extract_text(p, "pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7 garbage" * 10)
    with pytest.raises(docs.DocumentError):
        docs.extract_text(bad, "pdf")


def test_graph_source_passages_are_kept():
    from src.answer import _dedup_passages
    hits = [{"chunk_id": "a"}, {"chunk_id": "a"}, {"chunk_id": "b"}]
    assert [h["chunk_id"] for h in _dedup_passages(hits, {"a"})] == ["a", "b"]
