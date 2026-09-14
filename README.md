# KGRAG — Knowledge Graph RAG for Enterprise Data (SEC filings)

Phase 1: **extract entities and relationships from SEC 10-K filings into a Neo4j
knowledge graph**, using a **local open-source LLM (Llama 3.1 via Ollama)** for
schema-constrained extraction via LlamaIndex. Fully free — nothing calls a paid
API.

```
EDGAR 10-K  ->  clean text  ->  chunk  ->  Claude extraction (fixed ontology)
            ->  triples  ->  Neo4j property graph  ->  verify with Cypher
```

## Why a fixed ontology?
`src/schema.py` defines the only allowed entity types (COMPANY, PERSON, PRODUCT,
SEGMENT, RISK, AUDITOR, LOCATION, GOVERNMENT_AGENCY, STOCK_EXCHANGE) and
relationship types (HAS_SUBSIDIARY, COMPETES_WITH, HAS_EXECUTIVE, FACES_RISK,
AUDITED_BY, ...). The extractor runs in `strict` mode, so the LLM can only emit
valid triples. This is what separates a clean, queryable graph from noise.

## Prerequisites
- Python 3.10+
- Docker (for local Neo4j) — or your own Neo4j instance
- [Ollama](https://ollama.com) (free, runs the LLM locally)

## Setup

1. **Install Ollama and pull the models**
   ```bash
   ollama pull llama3.1:8b
   ollama pull bge-m3
   ```
   `llama3.1:8b` does extraction; `bge-m3` does embeddings for entity
   resolution. Ollama serves both at `http://localhost:11434`. `qwen2.5:7b` is a
   good extraction alternative that often follows strict schemas better.

2. **Config**
   ```bash
   cp .env.example .env
   ```
   Edit `.env`: set a real `SEC_USER_AGENT` (SEC requires a name + contact
   email, e.g. `Jane Doe jane@acme.com`). No API key needed.

3. **Install Python deps**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Neo4j**
   ```bash
   docker compose up -d
   ```
   Browser UI at http://localhost:7474 (neo4j / kgrag-password).

## Run Phase 1

**Always budget first.** This estimates tokens, chunks, time, and dollar cost
per document *without* calling the model, marks already-cached documents, and
refuses to green-light a run above `BUDGET_LIMIT_USD`:

```bash
python -m src.budget --tickers AAPL MSFT NVDA
```

Then extract. Documents whose content hash is already ingested are **skipped**
(no re-extraction cost); use `--force` to re-run them:

```bash
python -m src.ingest --tickers AAPL MSFT NVDA
```

Clean up junk + exact duplicates:

```bash
python -m src.cleanup
```

Resolve fuzzy duplicates (`Acme Corp` == `ACME`) and attach alias lists. Preview
scores first to tune the threshold, then apply:

```bash
python -m src.resolve --dry-run
```

```bash
python -m src.resolve
```

Then inspect the graph:

```bash
python -m src.verify
```

## Phase 2 — build the vector index (pgvector)

Embeds the same chunks into Postgres/pgvector alongside the graph, tagged with
`doc_id`, `section_path`, `filing_date`, and the `entity_ids` each chunk mentions
(the `chunk_id` is the join key back to Neo4j). Start Postgres (bundled in
`docker compose`), then build:

```bash
docker compose up -d postgres
python -m src.vectorize --query "supply chain and manufacturing risk"
```

`--query` runs a demo similarity search after building. Re-running is idempotent
(upsert by `chunk_id`).

Or explore visually in the Neo4j Browser (http://localhost:7474):

```cypher
MATCH (n) RETURN n LIMIT 100
```

## Using a paid model safely (Claude, hosted APIs)
We run local/free, but if you swap in a paid model the cost guardrails protect
you automatically — you do **not** have to remember to run the budget script:

- Known paid models are **auto-priced** (`src/pricing.py`) even if you never set
  `PRICE_PER_1M_*`. A `$0` estimate on a paid model is treated as unsafe.
- `src.ingest` runs a **built-in budget gate**: on any non-free model it prints
  the estimate, **blocks** if it exceeds `BUDGET_LIMIT_USD` (a hard wall), and
  otherwise **refuses to spend without `--yes`**:
  ```bash
  python -m src.ingest --tickers AAPL --yes   # authorize the estimated spend
  ```
- Local Ollama passes the gate silently (cost `$0`), so free runs have no
  friction.

## Tuning knobs
- `OLLAMA_MODEL` — defaults to `llama3.1:8b`. Bigger models (e.g. `qwen2.5:14b`)
  extract more accurately but run slower; smaller ones are faster.
- `MAX_FILING_CHARS` — caps how much of each 10-K is sent for extraction
  (default 12k chars) to keep local runs fast. Raise once it works.
- `RESOLVE_THRESHOLD` — cosine cutoff (default `0.90`) for merging entities.
  Higher = stricter (fewer merges); use `--dry-run` to see scores and tune it.
- `PRICE_PER_1M_INPUT` / `PRICE_PER_1M_OUTPUT` — set to your provider's rates to
  get real dollar estimates from `src.budget` (default `0` for local Ollama).
- `BUDGET_LIMIT_USD` — hard ceiling; `src.budget` blocks runs estimated above it.
- Downloaded filing text is cached under `data/`, and extraction results are
  cached by **document hash** (`data/ingest_manifest.json`) — re-running never
  re-downloads or re-extracts unchanged filings.

## A note on local models
A local 7–8B model is free but less accurate at strict structured extraction
than a hosted frontier model. Expect to miss some triples and occasionally see a
malformed one. The `strict` schema (`src/schema.py`) filters out invalid shapes,
which helps a lot. If quality is poor, try `qwen2.5:7b`/`14b` or lower
`MAX_FILING_CHARS` so each chunk is easier to reason over.

## Project layout
```
config.py            # env-driven configuration
docker-compose.yml   # local Neo4j + APOC
src/schema.py        # the SEC ontology (entity/relation/validation schema)
src/edgar.py         # download 10-Ks from EDGAR, HTML -> clean text
src/ingest.py        # extraction -> Neo4j (the main Phase 1 entry point)
src/verify.py        # Cypher sanity checks on the built graph
```

## Next (Phase 2, later)
Turn on node embeddings (`embed_kg_nodes=True`) and add a retriever that
combines vector search + Text2Cypher over this graph to answer natural-language
questions.
