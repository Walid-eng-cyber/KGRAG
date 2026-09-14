# Phase 1 — Extraction: from SEC filings to a knowledge graph

**Status: complete and verified.** This document is the full, detailed reference
for Phase 1 of KGRAG: taking raw SEC 10-K filings and turning them into a clean,
deduplicated, provenance-tracked knowledge graph in Neo4j — entirely with free,
local, open-source components, and with cost guardrails that make the pipeline
safe to run on a paid model too.

---

## Table of contents
1. [What Phase 1 delivers](#1-what-phase-1-delivers)
2. [The pipeline at a glance](#2-the-pipeline-at-a-glance)
3. [Technology stack](#3-technology-stack)
4. [Prerequisites & setup](#4-prerequisites--setup)
5. [Configuration reference](#5-configuration-reference)
6. [Step 1 — Data collection (`edgar.py`)](#6-step-1--data-collection-edgarpy)
7. [Step 2 — Budgeting & cost safety (`budget.py`, `pricing.py`)](#7-step-2--budgeting--cost-safety-budgetpy-pricingpy)
8. [Step 3 — Extraction (`ingest.py`, `schema.py`)](#8-step-3--extraction-ingestpy-schemapy)
9. [Step 4 — Cleanup (`cleanup.py`)](#9-step-4--cleanup-cleanuppy)
10. [Step 5 — Entity resolution (`resolve.py`)](#10-step-5--entity-resolution-resolvepy)
11. [Step 6 — Verification (`verify.py`)](#11-step-6--verification-verifypy)
12. [The graph data model](#12-the-graph-data-model)
13. [Running Phase 1 end to end](#13-running-phase-1-end-to-end)
14. [Results we measured](#14-results-we-measured)
15. [Design decisions & rationale](#15-design-decisions--rationale)
16. [Known limitations](#16-known-limitations)
17. [Troubleshooting](#17-troubleshooting)
18. [File reference](#18-file-reference)
19. [What Phase 2 will add](#19-what-phase-2-will-add)

---

## 1. What Phase 1 delivers

The problem: standard vector RAG retrieves isolated text chunks and cannot
answer relationship or multi-hop questions ("which companies face supply-chain
risk?", "who are the executives of firms regulated by the SEC?"). Phase 1 builds
the structure those questions need — a **knowledge graph** of companies, people,
risks, regulators, locations and their relationships, extracted from the
authoritative source (SEC filings).

Concretely, Phase 1 produces:
- A **Neo4j property graph** of typed entities and relationships.
- **Provenance** — every edge is traceable to the exact source sentence.
- **Clean identities** — junk removed, duplicates merged, aliases recorded.
- **Reproducibility** — idempotent re-ingestion and content-hash caching.
- **Cost safety** — a pre-flight budget gate for anyone using a paid model.

Everything runs locally for **$0** on open-source models.

---

## 2. The pipeline at a glance

```
   ┌─────────┐   ┌────────┐   ┌─────────┐   ┌────────┐   ┌─────────┐   ┌─────────┐
   │ collect │──▶│ budget │──▶│ extract │──▶│ cleanup│──▶│ resolve │──▶│ verify  │
   │ EDGAR   │   │ gate   │   │ LLM+KG  │   │ junk/  │   │ dedupe/ │   │ Cypher  │
   │ 10-K    │   │ cost   │   │ →Neo4j  │   │ dupes  │   │ aliases │   │ checks  │
   └─────────┘   └────────┘   └─────────┘   └────────┘   └─────────┘   └─────────┘
   edgar.py      budget.py     ingest.py     cleanup.py   resolve.py    verify.py
                 pricing.py    schema.py
                 cache.py      cache.py
```

---

## 3. Technology stack

| Layer | Technology | Role | Cost |
|---|---|---|---|
| Data source | SEC EDGAR REST API | Public 10-K filings | Free |
| Extraction LLM | Ollama + Llama 3.1 8B | Entity/relationship extraction | Free (local) |
| Embeddings | Ollama + `bge-m3` | Entity-resolution similarity | Free (local) |
| Orchestration | LlamaIndex (`PropertyGraphIndex`, `SchemaLLMPathExtractor`) | Chunking, schema-constrained extraction, graph upsert | Free (OSS) |
| Graph database | Neo4j 5 Community + APOC (Docker) | Stores the graph | Free |
| Language | Python 3.10+ | Pipeline code | Free |

---

## 4. Prerequisites & setup

**Prerequisites**
- Python 3.10+
- Docker (for local Neo4j) — or a reachable Neo4j instance
- [Ollama](https://ollama.com) (runs the models locally)

**Setup**

1. Pull the models:
   ```bash
   ollama pull llama3.1:8b
   ollama pull bge-m3
   ```
2. Configure:
   ```bash
   cp .env.example .env
   ```
   Set `SEC_USER_AGENT` to your real name + email — **EDGAR blocks requests
   without a descriptive User-Agent**. No API key is needed for the local stack.
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Neo4j:
   ```bash
   docker compose up -d
   ```
   Browser UI: http://localhost:7474 · credentials `neo4j` / `kgrag-password`
   · Bolt: `bolt://localhost:7687`.

---

## 5. Configuration reference

All configuration is environment-driven (`.env`), loaded by `config.py`.

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_MODEL` | Extraction model | `llama3.1:8b` |
| `OLLAMA_BASE_URL` | Ollama endpoint | `http://localhost:11434` |
| `EMBED_MODEL` | Embedding model for entity resolution | `bge-m3` |
| `RESOLVE_THRESHOLD` | Cosine cutoff to merge entities | `0.90` |
| `NEO4J_URI` / `_USERNAME` / `_PASSWORD` | Graph DB connection | `bolt://localhost:7687` / `neo4j` / `kgrag-password` |
| `SEC_USER_AGENT` | Identity sent to EDGAR (required) | — |
| `MAX_FILING_CHARS` | Per-filing extraction cap | `20000` |
| `PRICE_PER_1M_INPUT` / `_OUTPUT` | Token prices for the budget estimator | `0.0` |
| `BUDGET_LIMIT_USD` | Hard ceiling the budget gate refuses to cross | `10.0` |
| `SECONDS_PER_CHUNK` | Measured local latency, for time estimates | `20.0` |

---

## 6. Step 1 — Data collection (`edgar.py`)

**Goal:** fetch each company's latest 10-K and reduce it to clean, relevant
prose.

### 6.1 Ticker → CIK
EDGAR identifies companies by a permanent **CIK** (Central Index Key), not by
ticker. We download SEC's `company_tickers.json` once (cached), then map e.g.
`AAPL` → CIK `320193`.

### 6.2 Find the latest 10-K
We call the submissions API:
```
https://data.sec.gov/submissions/CIK0000320193.json
```
It lists every filing the company has made; we take the first `form == "10-K"`
and its primary document name.

### 6.3 Download the filing
We construct the archive URL, e.g.:
```
https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm
```

### 6.4 Extract the narrative sections (the important part)
A 10-K's raw document begins with a large block of **inline XBRL** (machine-
readable financial tagging), not prose. Naively taking the "first N characters"
feeds the model tag soup — which is exactly why an early run produced junk
entities like `A0` (from `aapl:A0.000Notesdue2025Member`) and `AAPL`.

`extract_narrative()` fixes this by parsing the 10-K's **Item structure**:
- It finds every `Item N.` header via regex.
- For each target Item it keeps the **longest** matching slice — the real
  section, not the shorter table-of-contents line.
- It targets **Item 1 (Business)** and **Item 1A (Risk Factors)** — the
  relationship-rich prose (subsidiaries, products, segments, competitors,
  risks). Each section gets an equal share of `MAX_FILING_CHARS`.
- If parsing fails, it falls back to a raw slice.

After this change, extraction reads real prose ("The Company designs,
manufactures and markets smartphones…") instead of XBRL tags.

### 6.5 Caching
Cleaned narrative text is cached to `data/<TICKER>_10k.txt`, so re-runs never
re-hit EDGAR. We also send a polite `User-Agent` and sleep 0.2 s between
requests to respect EDGAR's ~10 req/sec limit.

---

## 7. Step 2 — Budgeting & cost safety (`budget.py`, `pricing.py`)

**Goal:** never spend money (or hours) by surprise. This is the guardrail that
matters the instant someone swaps the free local model for a paid one.

### 7.1 Pre-flight estimator (`budget.py`)
Run **before** a full-corpus extraction. Without calling the model, it estimates
per document: token count, chunk count, wall-clock time, and dollar cost. It
also marks documents already ingested (via the hash cache) as `CACHED (skip)`,
and **refuses to green-light** a run whose estimated cost exceeds
`BUDGET_LIMIT_USD`.

Cost model (per document): tokens ≈ chars/4; chunks ≈ tokens / (512−64); input ≈
tokens + chunks × 600 (prompt overhead); output ≈ chunks × 300; cost = tokens ÷
1e6 × price.

### 7.2 Model price presets (`pricing.py`)
Known paid models (Claude Opus/Sonnet/Haiku/Fable) are **auto-priced** even if
the user never sets `PRICE_PER_1M_*`. A `$0.00` estimate on a paid model is
treated as unsafe, not free. Priority: explicit env override → preset → free.

### 7.3 Built-in budget gate (in `ingest.py`)
The estimator is not just an optional script — `ingest.py` runs a `_budget_gate`
before extraction:
- **Free/local model** → passes silently ($0, no friction).
- **Paid model** → prints the estimate; **blocks** if over `BUDGET_LIMIT_USD`
  (a hard wall `--yes` cannot override); otherwise **requires `--yes`** to
  authorize the spend.
- The gate runs **before the model is even constructed**, so it costs nothing.

---

## 8. Step 3 — Extraction (`ingest.py`, `schema.py`)

**Goal:** turn narrative text into typed graph triples.

### 8.1 The ontology (`schema.py`) — the quality lever
Extraction is constrained to a fixed schema. This is the single biggest driver
of graph quality: without it, an LLM invents arbitrary entities/edges and the
graph becomes unqueryable.

- **Entity types (9):** `COMPANY`, `PERSON`, `PRODUCT`, `SEGMENT`, `RISK`,
  `AUDITOR`, `LOCATION`, `GOVERNMENT_AGENCY`, `STOCK_EXCHANGE`.
- **Relationship types (15):** `HAS_SUBSIDIARY`, `COMPETES_WITH`,
  `PARTNERS_WITH`, `ACQUIRED`, `SUPPLIES_TO`, `HAS_EXECUTIVE`, `HAS_DIRECTOR`,
  `OFFERS_PRODUCT`, `OPERATES_SEGMENT`, `FACES_RISK`, `AUDITED_BY`,
  `HEADQUARTERED_IN`, `OPERATES_IN`, `REGULATED_BY`, `LISTED_ON`.
- **Validation schema:** a whitelist of allowed `(subject, relation, object)`
  shapes, e.g. `(COMPANY)-[HAS_EXECUTIVE]->(PERSON)`. With `strict=True`, any
  triple not on the list is rejected.

### 8.2 Chunking
Text is split into **512-token chunks** (64 overlap). Small chunks keep each
local-LLM call fast and reliable. Each chunk becomes a `Chunk` node storing the
original text.

### 8.3 Schema-constrained extraction
Each chunk is passed through Llama 3.1 via LlamaIndex's
`SchemaLLMPathExtractor` (strict mode, `num_workers=1` for a single local
model). It emits only valid triples.

### 8.4 Idempotent, provenance-carrying writes
- **MERGE, not CREATE:** LlamaIndex's `Neo4jPropertyGraphStore` upserts nodes
  (by id) and edges (by source+type+target) with APOC `MERGE`, so re-runs don't
  duplicate.
- **Deterministic chunk ids:** each `Document` gets a stable `doc_id` (the
  ticker) and each chunk id is `sha1(doc_id:index)`. Combined with MERGE, this
  makes re-ingesting the same filing update in place instead of spawning
  duplicate chunks and orphaned provenance.
- **Citable edges:** every relationship carries `triplet_source_id`, which
  resolves to the `Chunk` whose text justified it.

### 8.5 Hash cache (`cache.py`)
Before extracting, `ingest.py` computes a SHA-256 of each document's content. If
the hash is already in `data/ingest_manifest.json`, the document is **skipped**
(extraction already paid for). `--force` overrides. This means re-running a
100-filing corpus after adding 3 new ones costs only the 3 new ones.

---

## 9. Step 4 — Cleanup (`cleanup.py`)

**Goal:** deterministic, free repair of the noise a small local model produces.
No LLM calls.

1. **Junk removal** — drops tickers (`AAPL`), digit-noise (`A0`, `10-K`), and
   too-short names.
2. **Exact-canonical merge** — strips legal suffixes (`Inc.`, `Corp`, `LLC`…),
   case and punctuation, so `Apple` and `Apple Inc.` collapse to one node
   (APOC merge, keeping the fullest name and rewiring relationships).
3. **Self-loop removal** — deletes edges a node points at itself post-merge.

---

## 10. Step 5 — Entity resolution (`resolve.py`)

**Goal:** the hard part most pipelines skip — collapse different surface forms of
the same real-world entity into one node. `Acme Corp`, `Acme Corporation`, and
`ACME` are one company.

Two stages, run **per entity type** (a `PERSON` is never merged into a
`COMPANY`):
1. **Normalize + exact match** — cheap, high precision.
2. **Embedding similarity** — remaining distinct names are embedded with
   `bge-m3`; any pair whose **cosine similarity** ≥ `RESOLVE_THRESHOLD` is
   matched.

Matches are grouped transitively (union-find); the fullest name survives,
relationships are rewired (APOC), and **every surface form is stored on the
survivor's `aliases` property**.

**Tuning:** `python -m src.resolve --dry-run` prints every match and near-match
with its score, so the threshold can be set empirically — merging duplicates
without fusing genuinely different entities.

---

## 11. Step 6 — Verification (`verify.py`)

Runs a set of Cypher sanity checks: total node/relationship counts, counts by
type, and sample queries (competitors, risks). Use it — or the Neo4j Browser —
to confirm the graph after each run.

---

## 12. The graph data model

**Two layers:**
- **Entity nodes** (`COMPANY`, `PERSON`, `RISK`, …) carry `name`, `id` (= name),
  and metadata (`source`, `ticker`, `company`, `triplet_source_id`, and after
  resolution, `aliases`).
- **`Chunk` nodes** hold the original source text.

**Two kinds of edge:**
- **`MENTIONS`** (`Chunk → Entity`) — *provenance*: this passage mentions this
  thing.
- **Domain edges** (`HAS_EXECUTIVE`, `FACES_RISK`, `HAS_SUBSIDIARY`, …)
  (`Entity → Entity`) — *knowledge*: the real relationship. Each carries
  `triplet_source_id` pointing back to the justifying chunk.

**Guarantees:**
- Every knowledge edge is **citable** to a source sentence.
- Ingestion is **idempotent** (MERGE + deterministic chunk ids).

Example citation (real): `HAS_SUBSIDIARY → "Component Manufacturers"` is
justified by the chunk *"The Company uses some custom components that are not
commonly used by its competitors…"*.

---

## 13. Running Phase 1 end to end

```bash
# 0. Neo4j up
docker compose up -d

# 1. Preview cost/time (auto-prices paid models; blocks over budget)
python -m src.budget --tickers AAPL MSFT NVDA

# 2. Extract (gate requires --yes on a paid model; skips cached docs)
python -m src.ingest --tickers AAPL MSFT NVDA

# 3. Clean junk + exact duplicates
python -m src.cleanup

# 4. Resolve fuzzy duplicates + attach aliases (tune first)
python -m src.resolve --dry-run
python -m src.resolve

# 5. Verify
python -m src.verify
```

Explore visually at http://localhost:7474:
```cypher
MATCH (c:COMPANY)-[:FACES_RISK]->(r) RETURN c, r
```

---

## 14. Results we measured

Runs on the real latest 10-Ks for Apple, Microsoft, and NVIDIA.

**Data-quality fix impact (Apple):**
| | XBRL slice (before) | Narrative sections (after) |
|---|---|---|
| Business terms in text | "iPhone" only | competitor ×8, "risk factor", "adversely" ×10, … |

**Scale-up (3 companies, narrative extraction):**
- After ingest: **89 nodes, 133 relationships**.
- After cleanup: **54 entities** (junk removed, duplicates merged).
- Relationship types present: `MENTIONS` 64, `HAS_SUBSIDIARY` 29, `FACES_RISK`
  10, `OPERATES_IN` 4, `SUPPLIES_TO` 2, `HAS_EXECUTIVE` 2, `REGULATED_BY` 2,
  `HAS_DIRECTOR` 1.
- Entity types: `COMPANY` 33, `RISK` 10, `LOCATION` 4, `PERSON` 3,
  `GOVERNMENT_AGENCY` 2.
- Example knowledge: Apple → risks *Global Competition, Supply Chain
  Disruptions, Currency Fluctuations*; Apple → regulator *SEC*; subsidiaries
  Apple 12 / Microsoft 7 / NVIDIA 4.

**Entity-resolution demo:** `Acme Corp` / `Acme Corporation` / `ACME` collapsed
into one node with `aliases=[ACME, Acme Corp, Acme Corporation]`; the distractor
`Acme Restaurants` scored **0.823** (< 0.90) and was correctly kept separate.

**Budget demo:** local run estimated 36 chunks, ~12 min, **$0.00**. Simulated at
hosted rates ($3/$15 per 1M) the same run estimated **$0.27** and was **BLOCKED**
against a $0.10 ceiling.

---

## 15. Design decisions & rationale

- **Local/open-source first** — zero cost, full privacy, and a strong enterprise
  story. The LLM is swappable if higher accuracy is later worth a price.
- **Schema-first extraction** — a fixed ontology + strict validation is what
  separates a queryable graph from noise.
- **Target narrative sections, not raw bytes** — the XBRL header is not prose;
  parsing Items 1/1A is the single biggest quality win at fixed model cost.
- **Deterministic cleanup before LLM cleverness** — rules are free, predictable,
  and catch most noise; embeddings handle only the genuinely fuzzy cases.
- **Provenance on every edge** — enterprise use (audit, compliance) needs
  traceable, citable answers.
- **Idempotency + hash caching** — pipelines get re-run; they must not duplicate
  or re-spend.
- **Cost gate keyed off model, not a flag** — safety is automatic the moment a
  paid model is configured.

---

## 16. Known limitations

- **Local-model accuracy.** A 7–8B model is free but less precise than a hosted
  frontier model. Some triples are missed; a few are malformed. The strict
  schema filters invalid *shapes*, but not wrong *values*.
- **Value-level errors survive.** Observed examples: `MICROSOFT 365 COMMERCIAL`
  tagged as a `COMPANY` (it's a product/segment); `LIINKEDIN` (a typo of
  LinkedIn); a regulator captured as the generic string `"Government Agency"`.
  Fixing this class needs a stronger extractor or an LLM verification pass.
- **Missed relationship types.** In the current run no `COMPETES_WITH` or
  `OFFERS_PRODUCT` edges were extracted despite the prose containing them — the
  8B model simply missed them. A larger model (`qwen2.5:14b`) or a second,
  targeted pass would likely recover these.
- **Filing coverage is capped.** `MAX_FILING_CHARS` truncates each section to
  keep local runs fast; full coverage means larger caps and longer runs.
- **Extraction is slightly non-deterministic.** Re-running may add/drop a triple
  (a model property). Stable ids prevent duplication; set model temperature to 0
  for reproducible extraction.

---

## 17. Troubleshooting

- **EDGAR 403 / empty responses** — set a real `SEC_USER_AGENT` (name + email);
  EDGAR blocks generic/empty agents.
- **Extraction produces junk like `A0`, ticker names** — you're feeding the XBRL
  header, not prose. Ensure the narrative extractor is used (it is, by default);
  check the `[edgar] … extracted narrative sections` log line.
- **Runs are slow** — expected on a local model (~20 s/chunk). Lower
  `MAX_FILING_CHARS`, use fewer tickers, or a smaller model; long runs can be
  backgrounded.
- **Neo4j connection errors** — ensure Docker is running and the container is up
  (`docker compose up -d`); credentials are `neo4j` / `kgrag-password`.
- **Docker Desktop won't start / daemon unreachable (Windows)** — zombie Docker
  processes can block a clean start. Kill all `*docker*` processes, run
  `wsl --shutdown`, then start a single Docker Desktop and wait for the daemon.
- **Paid-model run "HELD" or "BLOCKED"** — that's the budget gate working. Review
  with `python -m src.budget`, raise `BUDGET_LIMIT_USD` or trim scope, and pass
  `--yes` to authorize.

---

## 18. File reference

```
KGRAG/
├── config.py            # env-driven configuration
├── docker-compose.yml   # local Neo4j 5 + APOC
├── requirements.txt
├── .env.example
├── README.md            # quick start
├── ARCHITECTURE.md      # architecture overview
├── PHASE1.md            # this document
├── data/                # cached filings, logs, ingest_manifest.json (git-ignored)
└── src/
    ├── schema.py        # SEC ontology (entities, relations, validation)
    ├── edgar.py         # EDGAR download + narrative-section extraction
    ├── pricing.py       # per-model price presets
    ├── cache.py         # document-hash cache / manifest
    ├── budget.py        # pre-flight cost/time estimator + ceiling
    ├── ingest.py        # extraction pipeline → Neo4j (entry point)
    ├── cleanup.py       # deterministic junk removal + exact-dupe merge
    ├── resolve.py       # entity resolution (embedding dedupe + aliases)
    └── verify.py        # Cypher sanity checks
```

---

## 19. What Phase 2 will add

Phase 1 built the graph. Phase 2 makes it answer questions:
- **Vector search** over the graph using local `bge-m3` embeddings.
- **Text2Cypher** — translate natural-language questions into Cypher queries the
  graph runs directly (for multi-hop and aggregation questions).
- **Grounded answers with citations** — every answer traced back to source
  chunks via the `triplet_source_id` / `MENTIONS` provenance already in place.

All still local and free.
