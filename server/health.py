"""Reachability of the local stack (Neo4j, Postgres/pgvector, Ollama models)."""
from __future__ import annotations

import config


def _neo4j() -> dict:
    try:
        from neo4j import GraphDatabase
        with GraphDatabase.driver(config.NEO4J_URI,
                                  auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
                                  connection_timeout=2) as d:
            d.verify_connectivity()
        return {"ok": True, "detail": config.NEO4J_URI}
    except Exception as e:  # noqa: BLE001 - report any failure as "down"
        return {"ok": False, "detail": type(e).__name__}


def _postgres() -> dict:
    try:
        import psycopg2
        conn = psycopg2.connect(host=config.PG_HOST, port=config.PG_PORT, dbname=config.PG_DB,
                                user=config.PG_USER, password=config.PG_PASSWORD,
                                connect_timeout=2)
        conn.close()
        return {"ok": True, "detail": f"{config.PG_HOST}:{config.PG_PORT}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": type(e).__name__}


def _ollama() -> dict:
    # With a hosted chat provider, Ollama only has to serve the embedding model.
    needed = {config.EMBED_MODEL} if config.LLM_PROVIDER != "ollama" else \
        {config.ROUTER_MODEL, config.ANSWER_MODEL, config.EMBED_MODEL}
    try:
        import ollama
        listed = ollama.Client(host=config.OLLAMA_BASE_URL, timeout=2).list()
        have = {m.get("model") or m.get("name") for m in listed.get("models", [])}
        have |= {h.split(":")[0] for h in have if h}
        missing = sorted(m for m in needed if m not in have and m.split(":")[0] not in have)
        return {"ok": not missing, "detail": "missing: " + ", ".join(missing) if missing else "models ready"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": type(e).__name__}


def _hosted_llm() -> dict:
    """Config check only — no paid call on every health poll."""
    if not config.LLM_API_KEY:
        return {"ok": False, "detail": "LLM_API_KEY is empty"}
    return {"ok": True, "detail": f"{config.LLM_PROVIDER} @ {config.LLM_BASE_URL}"}


def health() -> dict:
    services = {"neo4j": _neo4j(), "postgres": _postgres(), "ollama": _ollama()}
    if config.LLM_PROVIDER != "ollama":
        services["llm"] = _hosted_llm()
    return {"ok": all(s["ok"] for s in services.values()), "services": services,
            "models": {"router": config.ROUTER_MODEL, "answer": config.ANSWER_MODEL,
                       "embed": config.EMBED_MODEL}}
