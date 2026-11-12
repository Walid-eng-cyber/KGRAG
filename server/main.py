"""KGRAG API.

Run (development, with the React dev server on :5173):
    uvicorn server.main:app --reload --port 8000

Run (single process, serving the built UI from web/dist):
    cd web && npm run build && cd .. && uvicorn server.main:app --port 8000

Set KGRAG_DEMO=1 to serve labelled sample answers without Neo4j/Postgres/Ollama.
"""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from server import demo
from server.benchmark_view import benchmark_summary
from server.graph_view import build_graph, cited_tags
from server.health import health

DEMO = os.getenv("KGRAG_DEMO", "").lower() in {"1", "true", "yes"}
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

app = FastAPI(title="KGRAG API", version="1.2.0-dev")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


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
    return {"demo": DEMO, "examples": demo.EXAMPLES, "tickers": ["AAPL", "MSFT", "NVDA"],
            "models": {"router": config.ROUTER_MODEL, "answer": config.ANSWER_MODEL,
                       "embed": config.EMBED_MODEL}}


@app.get("/api/health")
def get_health() -> dict:
    if DEMO:
        down = {"ok": False, "detail": "demo mode"}
        return {"ok": True, "demo": True,
                "services": {"neo4j": down, "postgres": down, "ollama": down},
                "models": meta()["models"]}
    return {**health(), "demo": False}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    t0 = perf_counter()
    if DEMO:
        return shape_answer(demo.demo_answer(req.question), perf_counter() - t0, is_demo=True)
    try:
        from src.answer import answer
        raw = answer(req.question.strip())
    except Exception as e:  # noqa: BLE001 - surface any backend failure to the UI
        raise HTTPException(status_code=503,
                            detail=f"Backend error ({type(e).__name__}): {e}. "
                                   "Are Neo4j, Postgres and Ollama running?") from e
    return shape_answer(raw, perf_counter() - t0)


@app.get("/api/benchmark")
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
