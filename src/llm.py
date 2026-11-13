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
        msg = r.json()["choices"][0]["message"]
        return {"message": {"role": "assistant", "content": msg.get("content") or ""}}


def chat_client():
    """A client with Ollama's `.chat(model=, messages=, format=)` interface."""
    if is_hosted():
        return OpenAICompatClient(config.LLM_BASE_URL, config.LLM_API_KEY, float(config.LLM_TIMEOUT_S))
    import ollama
    return ollama.Client(host=config.OLLAMA_BASE_URL)
