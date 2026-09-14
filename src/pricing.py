"""Known model prices, so a paid model is estimated correctly even if the user
never sets PRICE_PER_1M_* by hand.

We're local/free, but anyone who swaps `OLLAMA_MODEL` (or rewires the LLM) for a
hosted model should get a real dollar estimate automatically — a `$0.00`
estimate on a paid model is worse than none, because it looks safe.

Prices are USD per 1M tokens (input, output). Update as providers change rates.
"""
from __future__ import annotations

import config

# Anthropic list prices (per 1M tokens). Extend with any provider you use.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}


def resolve_prices(model: str) -> tuple[float, float]:
    """Return (input, output) price per 1M tokens for a model.

    Priority: explicit PRICE_PER_1M_* env override > known-model preset > free.
    Ollama / any unknown local model resolves to (0, 0).
    """
    if config.PRICE_PER_1M_INPUT or config.PRICE_PER_1M_OUTPUT:
        return config.PRICE_PER_1M_INPUT, config.PRICE_PER_1M_OUTPUT
    name = (model or "").lower()
    for key, prices in PRICING.items():
        if key in name:
            return prices
    return 0.0, 0.0


def is_free(model: str) -> bool:
    return resolve_prices(model) == (0.0, 0.0)
