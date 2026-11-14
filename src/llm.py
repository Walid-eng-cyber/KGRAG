"""One place that decides which LLM backend answers chat calls.

LLM_PROVIDER=ollama (default) -> the local Ollama server, as before.
LLM_PROVIDER=openai           -> any OpenAI-compatible Chat Completions API
                                 (Moonshot's Kimi K3.1 Pro, OpenRouter, ...).

Embeddings stay on Ollama (bge-m3) either way: the graph and the vector index
were built with them, and swapping the chat model must not invalidate either.
"""
from __future__ import annotations

import json
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

    @staticmethod
    def _body(model: str, messages: list[dict], format: str | None, options: dict | None) -> dict:  # noqa: A002
        body: dict = {"model": model, "messages": messages}
        if format == "json":
            body["response_format"] = {"type": "json_object"}
        if options and "temperature" in options:
            body["temperature"] = options["temperature"]
        if options and "num_predict" in options:
            body["max_tokens"] = options["num_predict"]
        return body

    def _post(self, model: str, body: dict, stream: bool = False):
        for attempt in range(4):  # rate limits and provider hiccups are routine on hosted APIs
            r = requests.post(self.url, headers=self.headers, json=body, timeout=self.timeout,
                              **({"stream": True} if stream else {}))
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                delay = float(r.headers.get("retry-after") or 2 ** attempt)
                if stream:
                    r.close()   # a streamed response holds its connection open until closed
                time.sleep(delay)
                continue
            if not r.ok:
                raise RuntimeError(f"{model} via {self.url}: HTTP {r.status_code} {r.text[:300]}")
            return r
        raise RuntimeError("unreachable")  # pragma: no cover - the loop returns or raises

    def chat(self, model: str, messages: list[dict], format: str | None = None,  # noqa: A002 (Ollama's name)
             options: dict | None = None, **_) -> dict:
        r = self._post(model, self._body(model, messages, format, options))
        payload = r.json()
        msg = payload["choices"][0]["message"]
        usage = payload.get("usage") or {}
        # Same field names Ollama uses, so the meter reads both backends one way.
        return {"message": {"role": "assistant", "content": msg.get("content") or ""},
                "prompt_eval_count": usage.get("prompt_tokens"),
                "eval_count": usage.get("completion_tokens")}

    def chat_stream(self, model: str, messages: list[dict], format: str | None = None,  # noqa: A002
                    options: dict | None = None, **_):
        """Yield Ollama-shaped chunks as the provider streams them: {"message": {"content": piece}}, with the
        token counts on the last one (the provider reports usage when asked with include_usage)."""
        body = self._body(model, messages, format, options)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        r = self._post(model, body, stream=True)
        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage") or {}
                choices = obj.get("choices") or []
                piece = ((choices[0].get("delta") or {}).get("content") or "") if choices else ""
                yield {"message": {"role": "assistant", "content": piece},
                       "prompt_eval_count": usage.get("prompt_tokens"), "eval_count": usage.get("completion_tokens")}
        finally:
            r.close()


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

    def chat_stream(self, model: str, messages: list[dict], **kw):
        """Yield the answer as text pieces while it is being generated; the call is metered when the stream
        ends, even if the caller stops reading early."""
        from src import meter
        t0 = time.perf_counter()
        stream = (self.inner.chat_stream(model=model, messages=messages, **kw) if hasattr(self.inner, "chat_stream")
                  else self.inner.chat(model=model, messages=messages, stream=True, **kw))
        pieces: list[str] = []
        tin = tout = None
        try:
            for chunk in stream:
                n_in, n_out = _field(chunk, "prompt_eval_count"), _field(chunk, "eval_count")
                tin, tout = n_in if n_in is not None else tin, n_out if n_out is not None else tout
                piece = _field(_field(chunk, "message") or {}, "content") or ""
                if piece:
                    pieces.append(piece)
                    yield piece
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
            estimated = tin is None or tout is None
            if estimated:   # the backend didn't report counts: estimate from characters rather than record zero
                tin = meter.estimate_tokens("".join(str(m.get("content", "")) for m in messages))
                tout = meter.estimate_tokens("".join(pieces))
            meter.record(self.stage, model, tin, tout, (time.perf_counter() - t0) * 1000, estimated)

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
