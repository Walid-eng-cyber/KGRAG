"""Streaming answers: the model clients, the answer step and the /api/ask/stream endpoint."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import config
import server.main as main
from server.usage import UsageStore
from src import llm, meter


@pytest.fixture(autouse=True)
def _sonnet_billing(monkeypatch):
    monkeypatch.setattr(config, "BILLING_PRICE_AS", "claude-sonnet-5-5")
    monkeypatch.setattr(config, "USD_TO_EUR", 0.5)


# ---- the hosted client reads the provider's stream -------------------------------------------------------------

class _SSE:
    status_code, ok, headers, text = 200, True, {}, ""

    def __init__(self, lines):
        self.lines, self.closed = lines, False

    def iter_lines(self, decode_unicode=True):
        yield from self.lines

    def close(self):
        self.closed = True


def _sse(*pieces, usage=(120, 8)):
    chunks = [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}" for p in pieces]
    chunks.append(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': usage[0], 'completion_tokens': usage[1]}})}")
    return ["", ": keep-alive", *chunks, "data: [DONE]"]


def _hosted(monkeypatch, stream):
    sent = {}

    def fake_post(url, headers, json, timeout, **kw):
        sent.update(body=json, kw=kw)
        return stream

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    return sent


def test_hosted_client_streams_text_pieces_and_meters_the_provider_usage(monkeypatch):
    stream = _SSE(_sse("Elodie ", "knows ", "Python."))
    sent = _hosted(monkeypatch, stream)
    with meter.meter() as m:
        pieces = list(llm.chat_client("answer").chat_stream(model="kimi-k3.1-pro", messages=[{"role": "user", "content": "q"}]))
    assert pieces == ["Elodie ", "knows ", "Python."]
    assert sent["body"]["stream"] is True and sent["body"]["stream_options"] == {"include_usage": True}
    assert sent["kw"].get("stream") is True and stream.closed
    (c,) = m.calls
    assert (c.stage, c.tokens_in, c.tokens_out, c.estimated) == ("answer", 120, 8, False)
    assert m.cost_eur == pytest.approx((120 / 1e6 * 2.0 + 8 / 1e6 * 10.0) * 0.5)   # priced as Sonnet


def test_a_stream_without_usage_is_estimated_not_recorded_as_zero(monkeypatch):
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "x" * 400}}]}), "data: [DONE]"]
    _hosted(monkeypatch, _SSE(lines))
    with meter.meter() as m:
        list(llm.chat_client("answer").chat_stream(model="kimi-k3.1-pro", messages=[{"role": "user", "content": "y" * 800}]))
    (c,) = m.calls
    assert (c.tokens_in, c.tokens_out, c.estimated) == (200, 100, True)


def test_stopping_a_stream_early_still_records_the_call_and_closes_the_connection(monkeypatch):
    stream = _SSE(_sse("one ", "two ", "three"))
    _hosted(monkeypatch, stream)
    with meter.meter() as m:
        gen = llm.chat_client("answer").chat_stream(model="kimi-k3.1-pro", messages=[{"role": "user", "content": "q"}])
        assert next(gen) == "one "
        gen.close()
    assert stream.closed and len(m.calls) == 1


def test_a_local_ollama_stream_is_read_the_same_way():
    class FakeOllama:
        def chat(self, model, messages, stream=False, **kw):
            assert stream is True
            yield {"message": {"content": "Hello "}}
            yield {"message": {"content": "there"}, "prompt_eval_count": 50, "eval_count": 4}

    with meter.meter() as m:
        pieces = list(llm.MeteredClient(FakeOllama(), "answer").chat_stream(model="llama3.1:8b", messages=[]))
    assert pieces == ["Hello ", "there"]
    assert [(c.tokens_in, c.tokens_out, c.estimated) for c in m.calls] == [(50, 4, False)]


# ---- the answer step streams, and starts over when the citation check fails -------------------------------------

HIT = {"chunk_id": "c1", "doc_id": "DOC-1", "section_path": "CV", "filing_date": "2026-01-01",
       "text": "Elodie Marchand knows Python and Spark. " * 4, "similarity": 0.9}


class _ScriptedClient:
    def __init__(self, attempts):
        self.attempts, self.stage, self.calls = list(attempts), "answer", 0

    def chat_stream(self, model, messages, **kw):
        text = self.attempts[self.calls]
        self.calls += 1
        for word in text.split(" "):
            yield word + " "

    def chat(self, model, messages, **kw):
        text = self.attempts[self.calls]
        self.calls += 1
        return {"message": {"content": text}}


def _answer_with(monkeypatch, attempts):
    import src.answer as answer_mod
    client = _ScriptedClient(attempts)
    monkeypatch.setattr(answer_mod, "route", lambda q, corpus=None: {
        "question": q, "classified": "vector", "confidence": 0.9, "path_used": "VECTOR",
        "result": {"vector": [dict(HIT)]}})
    monkeypatch.setattr(answer_mod, "chat_client", lambda stage="other": client)
    return answer_mod, client


def test_answer_streams_its_text_and_announces_the_route(monkeypatch):
    answer_mod, _ = _answer_with(monkeypatch, ["Elodie knows Python [P1]."])
    tokens, events = [], []
    raw = answer_mod.answer("Who knows Python?", on_token=tokens.append, on_event=lambda n, d: events.append((n, d)))
    assert "".join(tokens).strip() == raw["answer"] == "Elodie knows Python [P1]."
    assert events == [("route", {"path_used": "VECTOR", "classified": "vector", "confidence": 0.9})]
    assert raw["attempts"] == 1 and raw["citations_valid"]


def test_a_failed_citation_check_resets_the_stream_and_writes_the_answer_again(monkeypatch):
    answer_mod, client = _answer_with(monkeypatch, ["Elodie knows Python [P9].", "Elodie knows Python [P1]."])
    tokens, events = [], []
    raw = answer_mod.answer("Who knows Python?", on_token=tokens.append, on_event=lambda n, d: events.append(n))
    assert events == ["route", "reset"]                       # the invented [P9] sent the model back
    assert client.calls == 2 and raw["attempts"] == 2
    assert raw["answer"] == "Elodie knows Python [P1]." and raw["citations_valid"]


def test_without_callbacks_the_answer_is_unchanged(monkeypatch):
    answer_mod, client = _answer_with(monkeypatch, ["Elodie knows Python [P1]."])
    raw = answer_mod.answer("Who knows Python?")
    assert raw["answer"] == "Elodie knows Python [P1]." and client.calls == 1


# ---- the endpoint -------------------------------------------------------------------------------------------------

def _events(response):
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_demo_answers_stream_word_by_word_and_end_with_the_full_result(monkeypatch):
    monkeypatch.setattr(main, "DEMO", True)
    ev = _events(TestClient(main.app).post("/api/ask/stream", json={"question": "How many subsidiaries does Apple have?"}))
    assert [e["type"] for e in ev][-1] == "done" and {e["type"] for e in ev[:-1]} == {"token"}
    assert len([e for e in ev if e["type"] == "token"]) > 3
    assert "".join(e["text"] for e in ev if e["type"] == "token").strip() == ev[-1]["result"]["answer"].strip()
    assert ev[-1]["result"]["demo"] is True


@pytest.fixture
def usage(monkeypatch, tmp_path):
    s = UsageStore(tmp_path / "usage.db")
    monkeypatch.setattr(main, "usage_store", s)
    monkeypatch.setattr(main, "DEMO", False)
    return s


def test_a_live_stream_sends_route_tokens_then_the_validated_answer_and_records_usage(usage, monkeypatch):
    import src.answer

    def fake_answer(question, corpus=None, today=None, on_token=None, on_event=None):
        on_event("route", {"path_used": "GRAPH", "classified": "graph", "confidence": 0.95})
        meter.record("router", "llama3.2", 800, 20, 100.0)
        for piece in ("We have ", "4 candidates ", "[G1]."):
            on_token(piece)
        meter.record("answer", "llama3.1:8b", 500, 12, 900.0)
        return {"question": question, "classified": "graph", "confidence": 0.95, "path_used": "GRAPH",
                "answer": "We have 4 candidates [G1].", "sources": [{"tag": "G1", "kind": "graph-derived fact", "text": "t"}],
                "vector_hits": [], "graph_result": {}, "attempts": 1, "citations_valid": True}

    monkeypatch.setattr(src.answer, "answer", fake_answer)
    ev = _events(TestClient(main.app).post("/api/ask/stream", json={"question": "How many candidates do we have?"}))
    assert [e["type"] for e in ev] == ["route", "token", "token", "token", "done"]
    assert ev[0]["path"] == "GRAPH" and ev[0]["confidence"] == 0.95
    assert ev[-1]["result"]["answer"] == "We have 4 candidates [G1]." and ev[-1]["result"]["path"] == "GRAPH"
    (row,) = usage._events(None, __import__("datetime").datetime(2000, 1, 1, tzinfo=__import__("datetime").timezone.utc))
    assert row["kind"] == "ask" and row["outcome"] == "answered" and row["requests"] == 2 and row["cost_eur"] > 0


def test_a_failing_answer_ends_the_stream_with_a_generic_error_and_is_recorded(usage, monkeypatch):
    import src.answer

    def boom(question, corpus=None, today=None, on_token=None, on_event=None):
        meter.record("router", "llama3.2", 800, 20, 100.0)
        raise ConnectionError("bolt://secret-host:7687 refused")

    monkeypatch.setattr(src.answer, "answer", boom)
    ev = _events(TestClient(main.app).post("/api/ask/stream", json={"question": "anything at all?"}))
    assert [e["type"] for e in ev] == ["error"] and "secret-host" not in json.dumps(ev)
    (row,) = usage._events(None, __import__("datetime").datetime(2000, 1, 1, tzinfo=__import__("datetime").timezone.utc))
    assert row["outcome"] == "error" and row["tokens_in"] == 800     # the router call still cost money


def test_a_bad_corpus_is_an_ordinary_http_error_not_a_stream(usage):
    r = TestClient(main.app).post("/api/ask/stream", json={"question": "who is there?", "corpus_id": "c_nope"})
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
