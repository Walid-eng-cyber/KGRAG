"""One place that decides which LLM backend answers chat calls.

LLM_PROVIDER=ollama (default) -> the local Ollama server, as before.
LLM_PROVIDER=openai           -> any OpenAI-compatible Chat Completions API
                                 (Moonshot's Kimi K3.1 Pro, OpenRouter, ...).

Embeddings stay on Ollama (bge-m3) either way: the graph and the vector index
were built with them, and swapping the chat model must not invalidate either.
"""
from __future__ import annotations

import time

import requests

import config


def is_hosted() -> bool:
    return config.LLM_PROVIDER != "ollama"


class OpenAICompatClient:
    """Speaks Chat Completions but answers in Ollama's shape
    (`resp["message"]["content"]`), so call sites don't change."""

    def __init__(self, base_url: str, api_key: str, timeout: float):
        if not api_key:
            raise RuntimeError("LLM_PROVIDER is hosted but LLM_API_KEY is empty — set it in .env.")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout = timeout

    def chat(self, model: str, messages: list[dict], format: str | None = None,  # noqa: A002 (Ollama's name)
             options: dict | None = None, **_) -> dict:
        body: dict = {"model": model, "messages": messages}
        if format == "json":
            body["response_format"] = {"type": "json_object"}
        if options and "temperature" in options:
            body["temperature"] = options["temperature"]
        if options and "num_predict" in options:
            body["max_tokens"] = options["num_predict"]

        for attempt in range(4):  # rate limits and provider hiccups are routine on hosted APIs
            r = requests.post(self.url, headers=self.headers, json=body, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(float(r.headers.get("retry-after") or 2 ** attempt))
                continue
            if not r.ok:
                raise RuntimeError(f"{model} via {self.url}: HTTP {r.status_code} {r.text[:300]}")
            break
        payload = r.json()
        msg = payload["choices"][0]["message"]
        usage = payload.get("usage") or {}
        # Same field names Ollama uses, so the meter reads both backends one way.
        return {"message": {"role": "assistant", "content": msg.get("content") or ""},
                "prompt_eval_count": usage.get("prompt_tokens"),
                "eval_count": usage.get("completion_tokens")}


def _field(resp, key: str):
    try:
        return resp[key]          # dict, or Ollama's subscriptable response model
    except (KeyError, TypeError, AttributeError):
        return getattr(resp, key, None)


class MeteredClient:
    """Wraps any chat client and reports each call's tokens and duration to the
    request's meter (src/meter.py). `stage` names the step for cost breakdowns;
    a caller may change it between calls (e.g. "answer" then "retry")."""

    def __init__(self, inner, stage: str):
        self.inner = inner
        self.stage = stage

    def chat(self, model: str, messages: list[dict], **kw):
        from src import meter
        t0 = time.perf_counter()
        resp = self.inner.chat(model=model, messages=messages, **kw)
        ms = (time.perf_counter() - t0) * 1000
        tin, tout = _field(resp, "prompt_eval_count"), _field(resp, "eval_count")
        estimated = tin is None or tout is None
        if estimated:  # backend didn't say: estimate from characters rather than record zero
            content = (_field(resp, "message") or {})
            content = _field(content, "content") or ""
            tin = meter.estimate_tokens("".join(str(m.get("content", "")) for m in messages))
            tout = meter.estimate_tokens(content)
        meter.record(self.stage, model, tin, tout, ms, estimated)
        return resp

    def __getattr__(self, name):
        return getattr(self.inner, name)


def chat_client(stage: str = "other"):
    """A client with Ollama's `.chat(model=, messages=, format=)` interface, metered."""
    if is_hosted():
        inner = OpenAICompatClient(config.LLM_BASE_URL, config.LLM_API_KEY, float(config.LLM_TIMEOUT_S))
    else:
        import ollama
        inner = ollama.Client(host=config.OLLAMA_BASE_URL)
    return MeteredClient(inner, stage)
