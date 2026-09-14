"""Phase 1: extract entities & relationships from 10-Ks into Neo4j.

Pipeline (per ticker):
  EDGAR 10-K text  ->  LlamaIndex Document
                   ->  chunked into nodes
                   ->  SchemaLLMPathExtractor (Claude, schema-constrained)
                   ->  triples written to the Neo4j property graph

Run:
  python -m src.ingest --tickers AAPL MSFT NVDA
"""
from __future__ import annotations

import argparse
import hashlib

from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama

import config
from src.budget import estimate
from src.cache import doc_hash, is_cached, record
from src.edgar import fetch_10k_text
from src.pricing import resolve_prices
from src.schema import EXTRACTION_HINT, Entities, Relations, VALIDATION_SCHEMA

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]


def _stable_chunk_id(i: int, doc: Document) -> str:
    """Deterministic chunk id: same document + position -> same id every run.

    Chunk nodes (and the source-chunk id on every edge) then MERGE instead of
    duplicating, so re-running ingestion is idempotent."""
    return hashlib.sha1(f"{doc.doc_id}:{i}".encode()).hexdigest()


def build_llm() -> Ollama:
    return Ollama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        # Local models are slower than a hosted API — give each call room.
        request_timeout=900.0,
        context_window=4096,
    )


def build_graph_store() -> Neo4jPropertyGraphStore:
    return Neo4jPropertyGraphStore(
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URI,
    )


def build_extractor(llm: Ollama) -> SchemaLLMPathExtractor:
    return SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=Entities,
        possible_relations=Relations,
        kg_validation_schema=VALIDATION_SCHEMA,
        # strict=True rejects any triple whose (subject, relation, object) types
        # are not in VALIDATION_SCHEMA -> a clean, consistent graph.
        strict=True,
        # One local model instance — avoid flooding it with concurrent calls.
        num_workers=1,
    )


def load_documents(tickers: list[str], force: bool = False) -> list[Document]:
    """Build Documents, skipping any whose content hash is already ingested
    (cache hit) unless force=True."""
    docs: list[Document] = []
    for ticker in tickers:
        name, text = fetch_10k_text(ticker)
        h = doc_hash(text)
        if not force and is_cached(h):
            print(f"  [cache] {ticker}: unchanged — skipping extraction ({h})")
            continue
        docs.append(
            Document(
                # Stable doc_id makes chunk ids (below) deterministic across runs.
                doc_id=ticker.upper(),
                text=f"{EXTRACTION_HINT}\n\nCompany: {name} ({ticker})\n\n{text}",
                metadata={
                    "ticker": ticker.upper(), "company": name,
                    "source": "10-K", "content_hash": h,
                },
            )
        )
    return docs


def _budget_gate(documents: list[Document], assume_yes: bool) -> None:
    """Stop a paid run from spending without a check. Free/local models pass
    silently; any non-zero estimate must clear the ceiling AND be confirmed."""
    price_in, price_out = resolve_prices(config.OLLAMA_MODEL)
    total = sum(estimate(d.text, price_in, price_out)["cost"] for d in documents)
    if total == 0.0:
        return  # local / free model — no financial risk

    print(f"\n  PAID MODEL: {config.OLLAMA_MODEL} "
          f"(${price_in}/${price_out} per 1M tokens)")
    print(f"  Estimated cost for {len(documents)} document(s): ${total:.2f}")
    print(f"  Budget ceiling (BUDGET_LIMIT_USD): ${config.BUDGET_LIMIT_USD:.2f}")

    if total > config.BUDGET_LIMIT_USD:
        raise SystemExit(
            f"  BLOCKED: ${total:.2f} exceeds the ${config.BUDGET_LIMIT_USD:.2f} "
            f"ceiling. Trim the corpus, lower MAX_FILING_CHARS, or raise "
            f"BUDGET_LIMIT_USD.")
    if not assume_yes:
        raise SystemExit(
            "  HELD: paid extraction not confirmed. Review with "
            "`python -m src.budget`, then re-run with --yes to authorize.")
    print("  Confirmed (--yes). Proceeding.\n")


def main(tickers: list[str], force: bool = False, assume_yes: bool = False) -> None:
    llm = build_llm()
    Settings.llm = llm  # ensure LlamaIndex never falls back to OpenAI

    print(f"Extraction model (Ollama): {config.OLLAMA_MODEL}")
    print(f"Neo4j: {config.NEO4J_URI}")
    print(f"Tickers: {', '.join(tickers)}\n")

    documents = load_documents(tickers, force=force)
    if not documents:
        print("\nNothing to extract — all documents are cached. Use --force to "
              "re-extract. (Run `python -m src.budget` to preview cost first.)")
        return

    _budget_gate(documents, assume_yes)  # refuse to spend on a paid model unchecked

    graph_store = build_graph_store()
    extractor = build_extractor(llm)

    print("\nExtracting triples and writing to Neo4j (this calls the LLM)...")
    PropertyGraphIndex.from_documents(
        documents,
        llm=llm,
        kg_extractors=[extractor],
        property_graph_store=graph_store,
        # Phase 1 is about structure, not semantic search yet, so skip embeddings.
        # (Phase 2 will turn these on for hybrid vector + graph retrieval.)
        embed_kg_nodes=False,
        # Smaller chunks = faster, more reliable calls on a local model.
        # id_func gives each chunk a deterministic id -> idempotent re-ingest.
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=64, id_func=_stable_chunk_id)
        ],
        show_progress=True,
    )

    # Record what we extracted so the next run treats it as cached.
    for doc in documents:
        record(doc.metadata["content_hash"],
               {"ticker": doc.metadata["ticker"], "company": doc.metadata["company"]})
    print("\nDone. Open http://localhost:7474 and run:  MATCH (n) RETURN n LIMIT 100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SEC 10-K graph into Neo4j.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--force", action="store_true",
                        help="re-extract even if the document hash is cached")
    parser.add_argument("--yes", action="store_true",
                        help="authorize spend on a paid model (see src.budget)")
    args = parser.parse_args()
    main([t.upper() for t in args.tickers], force=args.force, assume_yes=args.yes)
