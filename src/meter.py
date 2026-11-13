"""Per-request metering: what one question (or one ingest) cost in tokens, dollars and time.

The API opens a meter around each request; every LLM call made through
`src.llm.chat_client()` and every retrieval step wrapped in `span()` adds itself
to the meter of the request it belongs to (a ContextVar, so concurrent requests
never mix). Outside a meter — CLI runs, evals, tests — recording is a no-op.

Only measurements are kept here: stage names, models, token counts, durations.
Never prompt or answer text; that belongs to the customer, not to our monitoring.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter

from src.pricing import billing_prices, usd_to_eur

CHARS_PER_TOKEN = 4  # fallback when a backend doesn't report token counts


@dataclass
class LLMCall:
    stage: str
    model: str
    tokens_in: int
    tokens_out: int
    ms: float
    estimated: bool = False

    @property
    def priced_as(self) -> str:
        return billing_prices(self.model)[0]

    @property
    def cost_usd(self) -> float:
        """List price of this call in USD, at the billing model's rates (see BILLING_PRICE_AS)."""
        _, p_in, p_out = billing_prices(self.model)
        return self.tokens_in / 1_000_000 * p_in + self.tokens_out / 1_000_000 * p_out

    @property
    def cost_eur(self) -> float:
        return usd_to_eur(self.cost_usd)


@dataclass
class Meter:
    calls: list[LLMCall] = field(default_factory=list)
    spans: dict[str, float] = field(default_factory=dict)  # non-LLM stage -> ms

    @property
    def tokens_in(self) -> int:
        return sum(c.tokens_in for c in self.calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.tokens_out for c in self.calls)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def cost_eur(self) -> float:
        return sum(c.cost_eur for c in self.calls)

    @property
    def requests(self) -> int:
        return len(self.calls)

    @property
    def priced_as(self) -> str | None:
        return self.calls[0].priced_as if self.calls else None

    @property
    def estimated(self) -> bool:
        return any(c.estimated for c in self.calls)

    def stages(self) -> dict[str, dict]:
        """{stage: {calls, tokens_in, tokens_out, cost_eur, ms}} — LLM stages and timed spans together."""
        out: dict[str, dict] = {}
        for c in self.calls:
            s = out.setdefault(c.stage, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_eur": 0.0, "ms": 0.0})
            s["calls"] += 1
            s["tokens_in"] += c.tokens_in
            s["tokens_out"] += c.tokens_out
            s["cost_eur"] += c.cost_eur
            s["ms"] += c.ms
        for name, ms in self.spans.items():
            s = out.setdefault(name, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_eur": 0.0, "ms": 0.0})
            s["ms"] += ms
        for s in out.values():
            s["cost_eur"] = round(s["cost_eur"], 6)
            s["ms"] = round(s["ms"], 1)
        return out


_current: ContextVar[Meter | None] = ContextVar("kgrag_meter", default=None)


@contextmanager
def meter() -> Iterator[Meter]:
    """Collect every LLM call and span made inside this block."""
    m = Meter()
    token = _current.set(m)
    try:
        yield m
    finally:
        _current.reset(token)


def record(stage: str, model: str, tokens_in: int, tokens_out: int, ms: float, estimated: bool = False) -> None:
    m = _current.get()
    if m is not None:
        m.calls.append(LLMCall(stage, model or "", max(0, int(tokens_in)), max(0, int(tokens_out)),
                               ms, estimated))


@contextmanager
def span(stage: str) -> Iterator[None]:
    """Time a non-LLM step (vector search, graph query) into the current meter."""
    m = _current.get()
    t0 = perf_counter()
    try:
        yield
    finally:
        if m is not None:
            m.spans[stage] = m.spans.get(stage, 0.0) + (perf_counter() - t0) * 1000


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN)
