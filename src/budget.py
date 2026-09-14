"""Pre-flight cost/time budget — run this BEFORE a full-corpus extraction.

This is the guardrail that stops an unattended run from quietly spending hours
(local) or hundreds of dollars (hosted). It estimates, per document, the tokens,
chunks, wall-clock time, and dollar cost — WITHOUT calling the model — flags
which documents are already cached (cost already paid), and refuses to
green-light a run whose estimate exceeds BUDGET_LIMIT_USD.

Usage:
  python -m src.budget --tickers AAPL MSFT NVDA
"""
from __future__ import annotations

import argparse
import math

import config
from src.cache import doc_hash, is_cached
from src.edgar import fetch_10k_text
from src.pricing import resolve_prices

# Rough per-chunk prompt overhead (schema + instructions) and output size.
PROMPT_OVERHEAD_TOKENS = 600
OUTPUT_TOKENS_PER_CHUNK = 300
CHARS_PER_TOKEN = 4
EFFECTIVE_CHUNK_TOKENS = 512 - 64  # chunk_size - overlap


def estimate(text: str, price_in: float = 0.0, price_out: float = 0.0) -> dict:
    tokens = math.ceil(len(text) / CHARS_PER_TOKEN)
    chunks = max(1, math.ceil(tokens / EFFECTIVE_CHUNK_TOKENS))
    input_tokens = tokens + chunks * PROMPT_OVERHEAD_TOKENS
    output_tokens = chunks * OUTPUT_TOKENS_PER_CHUNK
    cost = (
        input_tokens / 1_000_000 * price_in
        + output_tokens / 1_000_000 * price_out
    )
    seconds = chunks * config.SECONDS_PER_CHUNK
    return {
        "tokens": tokens, "chunks": chunks,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cost": cost, "seconds": seconds,
    }


def main(tickers: list[str]) -> None:
    price_in, price_out = resolve_prices(config.OLLAMA_MODEL)
    priced = price_in or price_out
    mode = (f"{config.OLLAMA_MODEL} @ ${price_in}/${price_out} per 1M"
            if priced else f"local Ollama ({config.OLLAMA_MODEL}), $0")
    print(f"Budget estimate — model: {mode}\n")
    print(f"{'ticker':<8}{'chunks':>7}{'in tok':>10}{'out tok':>9}"
          f"{'time':>8}{'cost':>9}  status")
    print("-" * 62)

    total_cost = total_secs = new_cost = 0.0
    total_chunks = 0
    for ticker in tickers:
        _, text = fetch_10k_text(ticker)
        e = estimate(text, price_in, price_out)
        cached = is_cached(doc_hash(text))
        status = "CACHED (skip)" if cached else "will extract"
        if not cached:
            new_cost += e["cost"]
        total_cost += e["cost"]
        total_secs += e["seconds"]
        total_chunks += e["chunks"]
        print(f"{ticker:<8}{e['chunks']:>7}{e['input_tokens']:>10,}"
              f"{e['output_tokens']:>9,}{e['seconds']/60:>6.1f}m"
              f"{'$'+format(e['cost'],'.2f'):>9}  {status}")

    print("-" * 62)
    print(f"{'TOTAL':<8}{total_chunks:>7}{'':>10}{'':>9}"
          f"{total_secs/60:>6.1f}m{'$'+format(total_cost,'.2f'):>9}")
    print(f"\nNew work only (uncached): ${new_cost:.2f}")
    print(f"Budget ceiling: ${config.BUDGET_LIMIT_USD:.2f}")

    if new_cost > config.BUDGET_LIMIT_USD:
        print(f"\n  BLOCKED: estimate ${new_cost:.2f} exceeds the "
              f"${config.BUDGET_LIMIT_USD:.2f} ceiling. Raise BUDGET_LIMIT_USD, "
              f"trim the corpus, or lower MAX_FILING_CHARS before running.")
        raise SystemExit(1)
    print("\n  OK to proceed:  python -m src.ingest --tickers "
          + " ".join(tickers))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Estimate extraction cost/time.")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    args = p.parse_args()
    main([t.upper() for t in args.tickers])
