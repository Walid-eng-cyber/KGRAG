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
from pathlib import Path
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Response
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
from server.users import CONSENT_TEXT

DEMO = os.getenv("KGRAG_DEMO", "").lower() in {"1", "true", "yes"}
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

log = logging.getLogger("kgrag.api")

check_public_config(auth.settings.enabled, auth.settings.cookie_secure, DEMO,
                    config.NEO4J_PASSWORD, config.PG_PASSWORD)

_DOCS = docs_enabled()
app = FastAPI(title="KGRAG API", version="1.3.0-dev",
              docs_url="/docs" if _DOCS else None, redoc_url=None,
              openapi_url="/openapi.json" if _DOCS else None)
app.add_middleware(SecurityHeaders, hsts=auth.settings.cookie_secure)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


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
        graph = s.get("kind", "").startswith("graph")
        sources.append({"tag": s["tag"], "kind": "graph" if graph else "passage",
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
    corpus = ([{"label": "AAPL", "detail": "10-K"}, {"label": "MSFT", "detail": "10-K"}, {"label": "NVDA", "detail": "10-K"}]
              if DEMO else [{"label": "SEC Form 4", "detail": "insider trades"}])
    return {"demo": DEMO, "examples": demo.EXAMPLES if DEMO else demo.LIVE_EXAMPLES,
            "tickers": ["AAPL", "MSFT", "NVDA"], "corpus": corpus,
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


@app.post("/api/ask", dependencies=[Depends(auth.require_user), Depends(auth.rate_limit)])
def ask(req: AskRequest) -> dict:
    t0 = perf_counter()
    if DEMO:
        return shape_answer(demo.demo_answer(req.question), perf_counter() - t0, is_demo=True)
    try:
        from src.answer import answer
        raw = answer(req.question.strip())
    except Exception as e:  # noqa: BLE001 - log the details, show the user a generic message
        log.exception("answer() failed")
        raise HTTPException(status_code=503,
                            detail="KGRAG couldn't answer right now: a backend service (Neo4j, Postgres or Ollama) "
                                   "isn't available. The details are in the server log.") from e
    return shape_answer(raw, perf_counter() - t0)


@app.get("/api/benchmark", dependencies=[Depends(auth.require_user)])
def get_benchmark() -> dict:
    return benchmark_summary()


# Serve the built React app (web/dist) when it exists; the dev server proxies /api instead.
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        target = (WEB_DIST / full_path).resolve()
        if full_path and target.is_file() and WEB_DIST in target.parents:
            return FileResponse(target)
        return FileResponse(WEB_DIST / "index.html")
