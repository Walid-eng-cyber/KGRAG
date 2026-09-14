"""Document-hash cache: never pay to extract the same content twice.

Extraction (the LLM step) is the expensive part — in time on a local model, in
dollars on a hosted one. We key it on a hash of the *document content*, so a
filing that hasn't changed is skipped entirely on re-runs. A manifest records
which document hashes have already been ingested into the graph.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from config import DATA_DIR

MANIFEST = DATA_DIR / "ingest_manifest.json"


def doc_hash(text: str) -> str:
    """Stable content fingerprint. Same text -> same hash -> cache hit."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def is_cached(h: str) -> bool:
    return h in load_manifest()


def record(h: str, meta: dict) -> None:
    m = load_manifest()
    m[h] = {**meta, "ingested_at": datetime.now(timezone.utc).isoformat()}
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")
