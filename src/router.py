"""Phase 3 — a cheap question router.

A small local model (few-shot) classifies each question into a retrieval path
and returns an enum + confidence. When confidence is low (the model is unsure,
or a cheap rule heuristic disagrees with it), we DON'T guess — we run both the
vector and graph paths and merge the results.

  VECTOR  semantic lookup      "what does Apple say about supply chain risk"
  GRAPH   structural / counts  "how many subsidiaries does Microsoft have"
  HYBRID  both, merged         "what risks do Apple's suppliers face"

The classifier is intentionally cheap: one small-model call returning JSON.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from time import perf_counter
from typing import get_args

import ollama

import config
from src import routerlog
from src.graph_query import run_graph_query
from src.schema import Entities, Relations
from src.vectorize import retrieve as vector_retrieve

ENTITY_LABELS = list(get_args(Entities))
REL_TYPES = list(get_args(Relations))


class Path(str, Enum):
    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    HYBRID = "HYBRID"


# --- 1. The cheap classifier -------------------------------------------------

SYSTEM = (
    "You route a question to a retrieval path over an SEC-filing knowledge base.\n"
    "Choose GRAPH for:\n"
    "  - connection questions ('how is X connected to Y')\n"
    "  - multi-hop chains ('which risks do X's suppliers face')\n"
    "  - comparisons across entities ('compare X and Y')\n"
    "  - aggregations over relationships ('how many', 'which companies', 'count').\n"
    "Choose VECTOR for:\n"
    "  - definitions ('what is X')\n"
    "  - policy lookups ('what is X's policy on Y')\n"
    "  - single-fact questions ('who is X's CEO').\n"
    "Choose HYBRID only when a question needs a semantic entry point AND graph "
    "expansion and neither GRAPH nor VECTOR clearly fits.\n"
    'Reply ONLY with JSON: {"path":"VECTOR|GRAPH|HYBRID","confidence":0.0-1.0}.'
)

FEWSHOT = [
    # GRAPH — connection, multi-hop, comparison, aggregation
    ("How is Apple connected to its contract manufacturers?", '{"path":"GRAPH","confidence":0.9}'),
    ("Which risks do Apple's suppliers face?", '{"path":"GRAPH","confidence":0.9}'),
    ("Compare the number of subsidiaries of Apple and Microsoft.", '{"path":"GRAPH","confidence":0.92}'),
    ("How many companies are regulated by the SEC?", '{"path":"GRAPH","confidence":0.95}'),
    # VECTOR — definition, policy, single fact
    ("What is a reportable business segment?", '{"path":"VECTOR","confidence":0.9}'),
    ("What is Apple's policy on data privacy?", '{"path":"VECTOR","confidence":0.88}'),
    ("Who is Apple's CEO?", '{"path":"VECTOR","confidence":0.9}'),
]

# GRAPH: connections, multi-hop chains, comparisons, aggregations over relationships.
_GRAPH = re.compile(
    r"\b(how many|number of|count|total|most|fewest|list all|which companies|who are"
    r"|regulated by|connected to|related to|linked to|relationship between|path from"
    r"|compare|versus|vs\.?|difference between|more .* than"
    r"|suppliers? of|competitors? of|subsidiaries? of|executives? of)\b"
    r"|\b\w+'s (suppliers?|competitors?|subsidiaries?|executives?)\b", re.I)
# VECTOR: definitions, policy lookups, single-fact questions.
_VECTOR = re.compile(
    r"\b(what is|what are|define|definition of|meaning of|policy on|policy for"
    r"|policy regarding|approach to|who is|when did|where is)\b", re.I)


def _heuristic(q: str) -> Path | None:
    """A cheap rule guess used only to cross-check the model.
    GRAPH is checked first: a possessive multi-hop like "Apple's suppliers"
    should win over a leading "what/who" phrasing."""
    if _GRAPH.search(q):
        return Path.GRAPH
    if _VECTOR.search(q):
        return Path.VECTOR
    return None


def classify(question: str) -> tuple[Path, float]:
    messages = [{"role": "system", "content": SYSTEM}]
    for q, a in FEWSHOT:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": question})

    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    try:
        resp = client.chat(model=config.ROUTER_MODEL, messages=messages, format="json")
        data = json.loads(resp["message"]["content"])
        path = Path(str(data["path"]).upper())
        conf = float(data.get("confidence", 0.5))
    except Exception:
        # Model failed: defer to the rule if it has a clear answer, else run both.
        h = _heuristic(question)
        return (h, config.ROUTER_CONFIDENCE_THRESHOLD) if h else (Path.HYBRID, 0.0)

    # Cross-check with the rule heuristic: disagreement => treat as unsure.
    h = _heuristic(question)
    if h is not None and h != path:
        conf = min(conf, config.ROUTER_CONFIDENCE_THRESHOLD - 0.01)
    return path, conf


# --- 2. The retrieval paths --------------------------------------------------

def vector_path(question: str, k: int = 5) -> list[dict]:
    return vector_retrieve(question, k)


def graph_path(question: str) -> dict:
    # Safe path: entity resolution + parameterized template. The model never
    # emits Cypher (see src/graph_query.py).
    return run_graph_query(question)


# --- 3. The router -----------------------------------------------------------

def _outcome(result: dict) -> dict:
    """Compact, reconstructable summary of what each path returned — no bulky
    chunk text, just the signals Phase 5 needs."""
    o: dict = {}
    if "vector" in result:
        v = result["vector"]
        o["vector"] = {
            "count": len(v),
            "top": None if not v else {
                "chunk_id": v[0]["chunk_id"], "doc_id": v[0]["doc_id"],
                "section": v[0]["section_path"],
                "similarity": round(v[0]["similarity"], 3)},
        }
    if "graph" in result:
        g = result["graph"]
        o["graph"] = {"query_type": g.get("query_type"),
                      "resolved": g.get("resolved"),
                      "rows": len(g.get("rows", [])),
                      "error": g.get("error")}
    return o


def route(question: str) -> dict:
    t0 = perf_counter()
    path, conf = classify(question)
    low = conf < config.ROUTER_CONFIDENCE_THRESHOLD

    if path == Path.HYBRID or low:
        chosen = "HYBRID (both + merge)" + (" [low-confidence fallback]" if low else "")
        result = {"vector": vector_path(question), "graph": graph_path(question)}
    elif path == Path.GRAPH:
        chosen, result = "GRAPH", {"graph": graph_path(question)}
    else:
        chosen, result = "VECTOR", {"vector": vector_path(question)}

    latency_ms = round((perf_counter() - t0) * 1000, 1)

    # Log the decision — this is Phase 5's raw material and can't be rebuilt later.
    routerlog.log({
        "question": question,
        "classified": path.value,
        "confidence": round(conf, 2),
        "low_confidence": low,
        "path_used": chosen,
        "router_model": config.ROUTER_MODEL,
        "latency_ms": latency_ms,
        "outcome": _outcome(result),
    })

    return {"question": question, "classified": path.value,
            "confidence": round(conf, 2), "path_used": chosen,
            "latency_ms": latency_ms, "result": result}


def _brief(r: dict) -> None:
    print(f"\nQ: {r['question']}")
    print(f"   -> {r['classified']} (conf {r['confidence']}) -> {r['path_used']}")
    if "vector" in r["result"]:
        v = r["result"]["vector"]
        print(f"   vector: {len(v)} chunks; top = {v[0]['doc_id']}/{v[0]['section_path']}"
              if v else "   vector: (none)")
    if "graph" in r["result"]:
        g = r["result"]["graph"]
        print(f"   graph: {len(g['rows'])} rows  cypher={g['cypher']}")


if __name__ == "__main__":
    demo = [
        "What does Apple say about supply chain risk?",
        "How many subsidiaries does Microsoft have?",
        "What risks do Apple's suppliers face?",
        "Tell me about NVIDIA's data center business.",
    ]
    for q in demo:
        _brief(route(q))
