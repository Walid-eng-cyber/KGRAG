"""KGRAG API.

Run (development, with the React dev server on :5173):
    uvicorn server.main:app --reload --port 8000

Run (single process, serving the built UI from web/dist):
    cd web && npm run build && cd .. && uvicorn server.main:app --port 8000

Set KGRAG_DEMO=1 to serve labelled sample answers without Neo4j/Postgres/Ollama.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, Field

import config  # loads .env before auth settings are read
from server import auth, demo
from server.benchmark_view import benchmark_summary
from server.graph_view import build_graph, cited_tags
from server.health import health
from server.security import SecurityHeaders, check_public_config, cors_origins, docs_enabled
from server.corpora import CorpusStore
from server.doc_store import DocumentStore, IngestWorker
from server.users import CONSENT_TEXT
from src import corpus_graph, corpus_schema
from src import documents as docs

DEMO = os.getenv("KGRAG_DEMO", "").lower() in {"1", "true", "yes"}
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

log = logging.getLogger("kgrag.api")

check_public_config(auth.settings.enabled, auth.settings.cookie_secure, DEMO,
                    config.NEO4J_PASSWORD, config.PG_PASSWORD)

@asynccontextmanager
async def lifespan(_app):
    # Only a real server start may mark jobs from a previous process as interrupted;
    # merely importing this module (tests, scripts) must never touch live jobs.
    doc_store.mark_interrupted()
    yield


_DOCS = docs_enabled()
app = FastAPI(title="KGRAG API", version="1.3.0-dev", lifespan=lifespan,
              docs_url="/docs" if _DOCS else None, redoc_url=None,
              openapi_url="/openapi.json" if _DOCS else None)
app.add_middleware(SecurityHeaders, hsts=auth.settings.cookie_secure)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    corpus_id: str | None = Field(default=None, max_length=40)


class CorpusBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    preset: str | None = Field(default=None, max_length=20)
    entities: list[str] | None = None
    relations: list[list[str]] | None = None


class CorpusPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    entities: list[str] | None = None
    relations: list[list[str]] | None = None


class GoogleLogin(BaseModel):
    credential: str = Field(min_length=20, max_length=4096)


class GoogleSignup(GoogleLogin):
    updates_opt_in: bool = False


class UpdatesBody(BaseModel):
    opt_in: bool


class StatusBody(BaseModel):
    status: Literal["approved", "rejected", "disabled", "pending"]


def _path_key(path_used: str | None) -> str:
    return (path_used or "NONE").split(" ")[0].upper()


def shape_answer(raw: dict, latency_s: float, is_demo: bool = False) -> dict:
    """answer() output -> the JSON contract the React UI renders."""
    sources = []
    for s in raw.get("sources", []):
        k = s.get("kind", "")
        kind = "graph" if k.startswith("graph") else "computed" if k.startswith("computed") else "passage"
        sources.append({"tag": s["tag"], "kind": kind,
                        "text": s.get("text", ""), "cite": s.get("cite")})

    # P1..Pn are the first deduped vector hits, in order (see src.answer.format_passages).
    passages = []
    for i, h in enumerate(raw.get("vector_hits") or []):
        tag = f"P{i + 1}"
        passages.append({"tag": tag if any(s["tag"] == tag for s in sources) else None,
                         "chunk_id": h.get("chunk_id"), "doc_id": h.get("doc_id"),
                         "section": h.get("section_path"), "filing_date": str(h.get("filing_date", "")),
                         "similarity": float(h.get("similarity", 0.0)),
                         "text": " ".join(str(h.get("text", "")).split())[:900]})

    text = raw.get("answer", "")
    return {
        "question": raw.get("question"),
        "path": _path_key(raw.get("path_used")),
        "path_label": raw.get("path_used"),
        "classified": raw.get("classified"),
        "confidence": raw.get("confidence"),
        "answer": text,
        "cited": cited_tags(text),
        "citations_valid": raw.get("citations_valid", True),
        "attempts": raw.get("attempts", 0),
        "uncited_claims": raw.get("uncited_claims", []),
        "sources": sources,
        "graph": build_graph(raw.get("graph_result")),
        "passages": passages,
        "latency_s": round(latency_s, 2),
        "demo": is_demo,
    }


@app.get("/api/meta")
def meta() -> dict:
    return {"demo": DEMO, "examples": demo.EXAMPLES if DEMO else [],
            "tickers": ["AAPL", "MSFT", "NVDA"],
            "auth": {"enabled": auth.settings.enabled, "client_id": auth.settings.client_id or None,
                     "consent_text": CONSENT_TEXT},
            "models": {"router": config.ROUTER_MODEL, "answer": config.ANSWER_MODEL,
                       "embed": config.EMBED_MODEL}}


@app.post("/api/auth/signup")
def auth_signup(body: GoogleSignup, response: Response) -> dict:
    """Create an account from a Google sign-in. Pending until an admin approves it."""
    user = auth.signup(body.credential, body.updates_opt_in)
    if user["status"] == "approved":  # ADMIN_EMAILS
        auth.start_session(response, user)
    return {"user": auth.public(user)}


@app.post("/api/auth/google")
def auth_google(body: GoogleLogin, response: Response) -> dict:
    user = auth.login(body.credential)
    auth.start_session(response, user)
    return {"user": auth.public(user)}


@app.get("/api/auth/me")
def auth_me(user: dict | None = Depends(auth.require_user)) -> dict:
    return {"user": auth.public(user) if user else None}


@app.post("/api/auth/logout")
def auth_logout(response: Response) -> dict:
    auth.end_session(response)
    return {"ok": True}


@app.put("/api/me/updates")
def set_updates(body: UpdatesBody, user: dict = Depends(auth.require_user)) -> dict:
    if user is None:
        raise HTTPException(status_code=404, detail="Accounts are not enabled on this server.")
    return {"user": auth.public(auth.settings.store.set_updates(user["email"], body.opt_in))}


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>KGRAG</title><style>body{{font-family:Georgia,serif;
background:#faf9f5;color:#141413;display:grid;place-items:center;min-height:100vh;margin:0;padding:16px}}
main{{max-width:440px}}h1{{font-weight:400}}p{{font-family:system-ui,sans-serif;color:#4c5a65}}</style></head>
<body><main><h1>{title}</h1><p>{body}</p></main></body></html>"""


@app.get("/api/unsubscribe", include_in_schema=False)
def unsubscribe(token: str = "") -> HTMLResponse:
    """One-click opt-out from the link in update emails. Needs no sign-in."""
    if not auth.settings.enabled or not token or auth.settings.store.unsubscribe(token) is None:
        return HTMLResponse(_PAGE.format(title="Link not recognised",
                                         body="This unsubscribe link is invalid or has expired."), status_code=404)
    return HTMLResponse(_PAGE.format(title="You're unsubscribed",
                                     body="You won't get KGRAG update emails any more. You can turn them back on "
                                          "from your account in the app."))


@app.get("/api/admin/users")
def admin_users(status: str | None = None, _: dict = Depends(auth.require_admin)) -> dict:
    users = auth.settings.store.list(status)
    return {"users": [{**auth.public(u), "created_at": u["created_at"], "decided_at": u["decided_at"],
                       "decided_by": u["decided_by"]} for u in users]}


@app.post("/api/admin/users/{email}/status")
def admin_set_status(email: str, body: StatusBody, admin: dict = Depends(auth.require_admin)) -> dict:
    if email.lower() == admin["email"]:
        raise HTTPException(status_code=400, detail="You can't change your own account's status.")
    user = auth.settings.store.set_status(email, body.status, by=admin["email"])
    if user is None:
        raise HTTPException(status_code=404, detail="No such account.")
    return {"user": auth.public(user)}


@app.get("/api/health", dependencies=[Depends(auth.require_user)])
def get_health() -> dict:
    if DEMO:
        down = {"ok": False, "detail": "demo mode"}
        return {"ok": True, "demo": True,
                "services": {"neo4j": down, "postgres": down, "ollama": down},
                "models": meta()["models"]}
    return {**health(), "demo": False}


@app.post("/api/ask", dependencies=[Depends(auth.rate_limit)])
def ask(req: AskRequest, user: dict | None = Depends(auth.require_user)) -> dict:
    t0 = perf_counter()
    if DEMO:
        return shape_answer(demo.demo_answer(req.question), perf_counter() - t0, is_demo=True)
    corpus = None
    if req.corpus_id:
        corpus = _own_corpus(req.corpus_id, user)
    elif auth.settings.enabled:
        # With accounts on, every question must name one of the asker's corpora:
        # the unscoped legacy graph could otherwise reach other users' entities.
        raise HTTPException(status_code=400, detail="Choose a corpus to ask in.")
    try:
        from src.answer import answer
        raw = answer(req.question.strip(), corpus)
    except Exception as e:  # noqa: BLE001 - log the details, show the user a generic message
        log.exception("answer() failed")
        raise HTTPException(status_code=503,
                            detail="KGRAG couldn't answer right now: a backend service (Neo4j, Postgres or Ollama) "
                                   "isn't available. The details are in the server log.") from e
    return shape_answer(raw, perf_counter() - t0)


@app.get("/api/benchmark", dependencies=[Depends(auth.require_user)])
def get_benchmark() -> dict:
    return benchmark_summary()


# ---------------------------------------------------------------------------
# Corpora (private per user) and the documents in them
# ---------------------------------------------------------------------------

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
_DB = os.getenv("DOCUMENTS_DB", str(config.DATA_DIR / "documents.db"))
corpus_store = CorpusStore(_DB)
doc_store = DocumentStore(_DB, os.getenv("UPLOAD_DIR", str(config.DATA_DIR / "uploads")))


def _ingest(doc: dict, text: str, progress) -> dict:
    corpus = corpus_store.get(doc["corpus_id"]) if doc.get("corpus_id") else None
    if corpus is None:
        raise ValueError("This document isn't in a corpus. Move it into one, then ingest again.")
    return corpus_graph.ingest_text(corpus, doc["id"], doc["title"], text, progress)


ingest_worker = IngestWorker(doc_store, docs.extract_text, _ingest)


def _owner(user: dict | None) -> str | None:
    return user["email"] if user else None


def _own_corpus(corpus_id: str, user: dict | None) -> dict:
    c = corpus_store.visible(corpus_id, _owner(user))
    if c is None:   # private: someone else's corpus is indistinguishable from a missing one
        raise HTTPException(status_code=404, detail="No such corpus.")
    return c


def _corpus_view(c: dict) -> dict:
    docs_in = doc_store.list(c["id"])
    return {**c, "documents": len(docs_in), "ready": sum(d["status"] == "ready" for d in docs_in)}


@app.get("/api/corpora")
def list_corpora(user: dict | None = Depends(auth.require_user)) -> dict:
    return {"corpora": [_corpus_view(c) for c in corpus_store.list_for(_owner(user))],
            "presets": {k: {"label": v["label"], "description": v["description"],
                            **corpus_schema.from_preset(k)} for k, v in corpus_schema.PRESETS.items()}}


@app.post("/api/corpora")
def create_corpus(body: CorpusBody, user: dict | None = Depends(auth.require_user)) -> dict:
    try:
        if body.entities is not None or body.relations is not None:
            schema = {"preset": "custom", **corpus_schema.validate(body.entities or [], body.relations or [])}
        else:
            schema = corpus_schema.from_preset(body.preset or "general")
    except corpus_schema.SchemaError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    c = corpus_store.create(body.name, body.description, "documents", schema, _owner(user))
    return {"corpus": _corpus_view(c)}


@app.patch("/api/corpora/{corpus_id}")
def update_corpus(corpus_id: str, body: CorpusPatch, user: dict | None = Depends(auth.require_user)) -> dict:
    c = _own_corpus(corpus_id, user)
    fields: dict = {}
    if body.name is not None:
        fields["name"] = body.name
    if body.description is not None:
        fields["description"] = body.description
    if body.entities is not None or body.relations is not None:
        if c["kind"] != "documents":
            raise HTTPException(status_code=400, detail="This corpus's schema is fixed by its data set.")
        try:
            fields["schema"] = {"preset": "custom", **corpus_schema.validate(
                body.entities if body.entities is not None else c["schema"]["entities"],
                body.relations if body.relations is not None else c["schema"]["relations"])}
        except corpus_schema.SchemaError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    return {"corpus": _corpus_view(corpus_store.update(corpus_id, **fields))}


@app.delete("/api/corpora/{corpus_id}")
def delete_corpus(corpus_id: str, user: dict | None = Depends(auth.require_user)) -> dict:
    c = _own_corpus(corpus_id, user)
    if c["kind"] != "documents":
        raise HTTPException(status_code=400, detail="Data-set corpora are removed by an administrator.")
    if any(d["status"] in ("queued", "ingesting") for d in doc_store.list(corpus_id)):
        raise HTTPException(status_code=409, detail="Wait for ingestion to finish before deleting this corpus.")
    try:
        corpus_graph.remove_corpus(corpus_id)
    except Exception as e:  # noqa: BLE001
        log.exception("removing corpus data failed")
        raise HTTPException(status_code=503, detail="Couldn't remove its data. The details are in the server log.") from e
    for d in doc_store.list(corpus_id):
        doc_store.delete(d["id"])
    corpus_store.delete(corpus_id)
    return {"ok": True}


@app.get("/api/corpora/{corpus_id}/profile")
def corpus_profile(corpus_id: str, user: dict | None = Depends(auth.require_user)) -> dict:
    c = _own_corpus(corpus_id, user)
    if c["kind"] == "sec_insiders":
        return {"corpus": _corpus_view(c), "profile": None, "examples": demo.LIVE_EXAMPLES}
    try:
        prof = corpus_graph.profile(corpus_id)
    except Exception:  # noqa: BLE001 - Neo4j down: the UI still works, just without suggestions
        log.exception("profile failed")
        prof = {"types": [], "top_entities": [], "relations": []}
    return {"corpus": _corpus_view(c), "profile": prof, "examples": corpus_graph.suggestions(prof, c["kind"])}


@app.get("/api/corpora/{corpus_id}/documents")
def list_documents(corpus_id: str, user: dict | None = Depends(auth.require_user)) -> dict:
    _own_corpus(corpus_id, user)
    return {"documents": doc_store.list(corpus_id), "max_upload_mb": MAX_UPLOAD_MB,
            "accepted": sorted(docs.SUPPORTED)}


@app.post("/api/corpora/{corpus_id}/documents")
async def upload_document(corpus_id: str, file: UploadFile = File(...), title: str = Form(""),
                          user: dict | None = Depends(auth.require_user)) -> dict:
    c = _own_corpus(corpus_id, user)
    if c["kind"] != "documents":
        raise HTTPException(status_code=400, detail="This corpus is a fixed data set; create a document corpus to upload files.")
    name = Path(file.filename or "document").name[:200]
    try:
        kind = docs.kind_for(name)
    except docs.DocumentError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Files can be at most {MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if kind == "pdf" and not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="This file isn't a valid PDF.")
    clean_title = (title.strip() or Path(name).stem)[:200]
    doc = doc_store.add(name, clean_title, kind, Path(name).suffix.lower(), data, _owner(user), corpus_id)
    return {"document": doc_store.get(doc["id"])}


@app.post("/api/corpora/{corpus_id}/ingest")
def ingest_documents(corpus_id: str, user: dict | None = Depends(auth.require_user)) -> dict:
    """Queue every document in the corpus that isn't ingested yet (uploaded or failed)."""
    _own_corpus(corpus_id, user)
    queued = [d["id"] for d in doc_store.list(corpus_id) if d["status"] in ("uploaded", "failed")]
    for doc_id in queued:
        ingest_worker.enqueue(doc_id)
    return {"queued": queued}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, user: dict | None = Depends(auth.require_user)) -> dict:
    doc = doc_store.get(doc_id)
    if doc is None or not doc.get("corpus_id") or corpus_store.visible(doc["corpus_id"], _owner(user)) is None:
        raise HTTPException(status_code=404, detail="No such document.")
    if doc["status"] in ("queued", "ingesting"):
        raise HTTPException(status_code=409, detail="Wait for ingestion to finish before removing it.")
    if doc["status"] == "ready":
        try:
            corpus_graph.remove_document(doc["corpus_id"], doc_id)
        except Exception as e:  # noqa: BLE001
            log.exception("removing document data failed")
            raise HTTPException(status_code=503, detail="Couldn't remove its data from the graph. "
                                                        "The details are in the server log.") from e
    doc_store.delete(doc_id)
    return {"ok": True}


# Serve the built React app (web/dist) when it exists; the dev server proxies /api instead.
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        target = (WEB_DIST / full_path).resolve()
        if full_path and target.is_file() and WEB_DIST in target.parents:
            return FileResponse(target)
        return FileResponse(WEB_DIST / "index.html")
