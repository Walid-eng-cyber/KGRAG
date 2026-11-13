"""The hosted-provider adapter: Ollama's call shape in, Chat Completions on the wire."""
import pytest

import config
from src import llm


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}
        self.ok, self.text = status < 400, str(payload)

    def json(self):
        return self._payload


def _ok(content):
    return _Resp(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_default_provider_is_local_ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    assert not llm.is_hosted()
    assert type(llm.chat_client().inner).__module__.startswith("ollama")


def test_hosted_client_speaks_chat_completions_and_returns_ollama_shape(monkeypatch):
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, body=json)
        return _ok('{"path": "graph"}')

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://api.moonshot.ai/v1/")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm.requests, "post", fake_post)

    resp = llm.chat_client().chat(model="kimi-k3.1-pro", format="json",
                                  messages=[{"role": "user", "content": "route this, answer in json"}])

    assert resp["message"]["content"] == '{"path": "graph"}'
    assert sent["url"] == "https://api.moonshot.ai/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["body"]["model"] == "kimi-k3.1-pro"
    assert sent["body"]["response_format"] == {"type": "json_object"}


def test_hosted_client_retries_rate_limits(monkeypatch):
    replies = iter([_Resp(429, headers={"retry-after": "0"}), _ok("hi")])
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: next(replies))
    client = llm.OpenAICompatClient("https://x/v1", "k", 5)
    assert client.chat(model="m", messages=[])["message"]["content"] == "hi"


def test_hosted_client_without_key_fails_loudly():
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        llm.OpenAICompatClient("https://x/v1", "", 5)


def test_kimi_is_priced_so_the_budget_gate_engages(monkeypatch):
    from src.pricing import resolve_prices
    monkeypatch.setattr(config, "PRICE_PER_1M_INPUT", 0.0)
    monkeypatch.setattr(config, "PRICE_PER_1M_OUTPUT", 0.0)
    assert resolve_prices("kimi-k3.1-pro") == (2.0, 10.0)
    assert resolve_prices("kimi-k3.1-pro-highspeed") == (4.0, 20.0)
