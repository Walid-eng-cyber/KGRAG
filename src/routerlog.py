"""Append-only log of every routing decision.

This data cannot be reconstructed after the fact — which path a question took,
the classifier's confidence, whether the fallback fired, and what each path
returned are only knowable at decision time. Phase 5 (learning from real usage:
tuning the threshold, spotting misroutes, building a labeled set from live
traffic) depends on it, so we capture it now.

Format: JSON Lines (one decision per line) at data/router_log.jsonl — easy to
append safely and to load for analysis later. Logging never raises; a logging
failure must not break routing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from config import DATA_DIR

LOG_PATH = DATA_DIR / "router_log.jsonl"


def log(record: dict) -> None:
    try:
        row = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass  # telemetry must never break the request path


def load() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(ln) for ln in LOG_PATH.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
