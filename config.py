"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Ollama (local, open-source models — no API cost) ---
# Models must be pulled first, e.g.:  ollama pull llama3.1:8b ; ollama pull bge-m3
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding model used for entity resolution (fuzzy duplicate detection).
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
# Cosine-similarity threshold above which two entity names are treated as the
# same real-world thing. Tune per corpus with `python -m src.resolve --dry-run`.
RESOLVE_THRESHOLD = float(os.getenv("RESOLVE_THRESHOLD", "0.90"))

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "kgrag-password")

# --- EDGAR ---
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "KGRAG Research example@example.com")
MAX_FILING_CHARS = int(os.getenv("MAX_FILING_CHARS", "60000"))

# --- Cost / budget model (for the pre-flight estimator) ---
# Price per 1M tokens. Local Ollama is free -> 0. Set these to your provider's
# rates (e.g. a hosted model) to see real dollar estimates before a full run.
PRICE_PER_1M_INPUT = float(os.getenv("PRICE_PER_1M_INPUT", "0.0"))
PRICE_PER_1M_OUTPUT = float(os.getenv("PRICE_PER_1M_OUTPUT", "0.0"))
# Hard ceiling: budget.py refuses to green-light a run estimated above this.
BUDGET_LIMIT_USD = float(os.getenv("BUDGET_LIMIT_USD", "10.0"))
# Measured wall-clock per chunk on this machine (for the local time estimate).
SECONDS_PER_CHUNK = float(os.getenv("SECONDS_PER_CHUNK", "20.0"))
