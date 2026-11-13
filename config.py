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

# --- Chat-model provider (src/llm.py) ---
# "ollama" = local (default). "openai" = any OpenAI-compatible API, e.g. Moonshot's
# Kimi K3.1 Pro: LLM_BASE_URL=https://api.moonshot.ai/v1, model id kimi-k3.1-pro.
# Embeddings (EMBED_MODEL) always stay on Ollama.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.moonshot.ai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "180"))
# Model used for graph extraction (ingest). Defaults to OLLAMA_MODEL for local runs.
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", OLLAMA_MODEL)

# Embedding model used for entity resolution (fuzzy duplicate detection).
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
# Cosine-similarity threshold above which two entity names are treated as the
# same real-world thing. Tune per corpus with `python -m src.resolve --dry-run`.
RESOLVE_THRESHOLD = float(os.getenv("RESOLVE_THRESHOLD", "0.90"))

# --- Neo4j (the graph) ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")  # set in .env; no built-in default

# --- Postgres + pgvector (the vector index, Phase 2) ---
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5434"))
PG_DB = os.getenv("PG_DB", "kgrag")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")  # set in .env; no built-in default
# bge-m3 produces 1024-dim embeddings.
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))

# --- Router (Phase 3) ---
# A small, cheap model classifies the question into a retrieval path.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama3.2")
# Below this confidence, fall back to running BOTH paths and merging.
ROUTER_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.6"))

# --- Answer generation (Phase 4) ---
# A more capable model synthesizes the grounded, cited answer.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "llama3.1:8b")
# How many times to reject-and-regenerate an answer with a bad/missing citation.
ANSWER_MAX_RETRIES = int(os.getenv("ANSWER_MAX_RETRIES", "2"))

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

# --- Client billing (server/usage.py) ---
# Every LLM call is priced as if it ran on this hosted model, whatever actually ran it
# (local Ollama included), so usage shows what an external provider would cost.
# Empty = price each call at its own model's rate (local models then cost 0).
BILLING_PRICE_AS = os.getenv("BILLING_PRICE_AS", "claude-sonnet-5-5").strip()
# Provider list prices are in USD; client credit and reports are in EUR.
BILLING_CURRENCY = "EUR"
USD_TO_EUR = float(os.getenv("USD_TO_EUR", "0.86"))
# Usage credit each new client account gets per calendar month.
DEFAULT_MONTHLY_CREDIT_EUR = float(os.getenv("DEFAULT_MONTHLY_CREDIT_EUR", "2500"))
# Measured wall-clock per chunk on this machine (for the local time estimate).
SECONDS_PER_CHUNK = float(os.getenv("SECONDS_PER_CHUNK", "20.0"))
EXTRACTION_TIMEOUT_S = int(os.getenv("EXTRACTION_TIMEOUT_S", "240"))  # per model call while extracting
