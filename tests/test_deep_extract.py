"""Detailed extraction (read whole -> what did we miss -> check each fact) and the graph rebuild script."""
from __future__ import annotations

import json

import pytest

import config
from src import corpus_graph, deep_extract, rebuild_graph

SCHEMA = {"entities": ["PERSON", "ORGANIZATION", "SKILL", "LOCATION"],
          "relations": [["PERSON", "WORKED_AT", "ORGANIZATION"], ["PERSON", "HAS_SKILL", "SKILL"],
                        ["PERSON", "LIVES_IN", "LOCATION"], ["ORGANIZATION", "LOCATED_IN", "LOCATION"]]}
CHUNKS = [("c1", "Elodie Marchand lives in Lyon. Since 2021 she has worked at Brightwater as a data engineer."),
          ("c2", "Skills: Python, Spark and SQL. Elodie led the migration of Brightwater's pipelines.")]


class Scripted:
    """Replies by stage (extract / glean / verify), in order; records every prompt."""

    def __init__(self, **replies):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.stage, self.calls = "", []

    def chat(self, model, messages, format=None, options=None):  # noqa: A002
        self.calls.append((self.stage, messages, options))
        reply = self.replies[self.stage].pop(0)
        return {"message": {"content": reply if isinstance(reply, str) else json.dumps(reply)}}


EXTRACT = {"entities": [{"name": "Elodie Marchand", "type": "PERSON"}, {"name": "Brightwater", "type": "ORGANIZATION"},
                        {"name": "Lyon", "type": "LOCATION"}, {"name": "Python", "type": "SKILL"},
                        {"name": "the candidate", "type": "CANDIDATE"}],                       # not a schema type
           "facts": [{"source": "Elodie Marchand", "relation": "WORKED_AT", "target": "Brightwater",
                      "quote": "Since 2021 she has worked at Brightwater"},
                     {"source": "Lyon", "relation": "lives in", "target": "Elodie Marchand",  # backwards, spaced
                      "quote": "Elodie Marchand lives in Lyon."},
                     {"source": "Elodie Marchand", "relation": "HAS_SKILL", "target": "Python",
                      "quote": "Skills: Python, Spark and SQL"},
                     {"source": "Elodie Marchand", "relation": "HAS_SKILL", "target": "Rust",  # Rust isn't an entity
                      "quote": "Rust"}]}
GLEAN = {"entities": [{"name": "Spark", "type": "SKILL"}, {"name": "brightwater", "type": "ORGANIZATION"}],
         "facts": [{"source": "Elodie Marchand", "relation": "HAS_SKILL", "target": "Spark",
                    "quote": "Skills: Python, Spark and SQL."},
                   {"source": "Brightwater", "relation": "LOCATED_IN", "target": "Lyon",
                    "quote": "Brightwater is headquartered in Lyon"}]}     # not in the document
VERIFY = {"supported": [1, 2, 4]}                                           # rejects fact 3 (Python)


def test_a_document_is_read_whole_then_gleaned_then_every_fact_checked(monkeypatch):
    monkeypatch.setattr(config, "EXTRACT_NUM_CTX", 8192)
    client = Scripted(extract=[EXTRACT], glean=[GLEAN], verify=[VERIFY])
    ext = deep_extract.extract_document("elodie marchand", CHUNKS, SCHEMA, "HINT", client=client, model="m")

    assert [s for s, _, _ in client.calls] == ["extract", "glean", "verify"]
    first_user = client.calls[0][1][1]["content"]
    assert CHUNKS[0][1] in first_user and CHUNKS[1][1] in first_user          # the whole document at once
    assert "HINT" in client.calls[0][1][0]["content"] and "-[WORKED_AT]->" in client.calls[0][1][0]["content"]
    assert "Elodie Marchand -[WORKED_AT]-> Brightwater" in client.calls[1][1][1]["content"]   # glean sees the list
    assert client.calls[0][2] == {"temperature": 0, "num_ctx": 8192, "num_predict": deep_extract.EXTRACT_TOKENS}

    kept = {(f.source, f.relation, f.target, f.chunk) for f in ext.facts}
    assert kept == {("Elodie Marchand", "WORKED_AT", "Brightwater", "c1"),
                    ("Elodie Marchand", "LIVES_IN", "Lyon", "c1"),             # turned the right way round
                    ("Elodie Marchand", "HAS_SKILL", "Spark", "c2")}
    assert ext.entities == {"Elodie Marchand": "PERSON", "Brightwater": "ORGANIZATION", "Lyon": "LOCATION",
                            "Python": "SKILL", "Spark": "SKILL"}               # "brightwater" is the same entity
    s = ext.stats
    assert (s["extracted"], s["gleaned"], s["quote_not_found"], s["rejected_by_check"], s["kept"]) == (3, 2, 1, 1, 3)
    assert s["invalid"] == 2 and s["calls"] == 3


def test_a_reply_that_is_not_json_is_asked_for_again_then_given_up():
    client = Scripted(extract=["here you go: {broken", EXTRACT], glean=["nope", "still nope"],
                      verify=[{"supported": [1, 2, 3]}])
    ext = deep_extract.extract_document("t", CHUNKS, SCHEMA, "", client=client, model="m")
    assert ext.stats["bad_json"] == 1 and ext.stats["calls"] == 5 and ext.stats["kept"] == 3
    assert "not valid JSON" in client.calls[1][1][-1]["content"]


def test_an_unreadable_check_keeps_facts_whose_quotes_were_found():
    client = Scripted(extract=[EXTRACT], glean=[{"entities": [], "facts": []}], verify=["x", "y"])
    ext = deep_extract.extract_document("t", CHUNKS, SCHEMA, "", client=client, model="m")
    assert ext.stats["kept"] == 3 and ext.stats["rejected_by_check"] == 0


def test_long_documents_are_read_in_windows_of_whole_chunks(monkeypatch):
    monkeypatch.setattr(config, "DETAILED_WINDOW_CHARS", 100)
    assert [[c for c, _ in w] for w in deep_extract.windows(CHUNKS, 100)] == [["c1"], ["c2"]]
    assert [[c for c, _ in w] for w in deep_extract.windows(CHUNKS, 10_000)] == [["c1", "c2"]]
    client = Scripted(extract=[EXTRACT, {"entities": [], "facts": []}], glean=[GLEAN, {}],
                      verify=[{"supported": [1, 2, 3, 4]}])
    ext = deep_extract.extract_document("t", CHUNKS, SCHEMA, "", client=client, model="m")
    assert ext.stats["windows"] == 2 and "(part 1 of 2)" in client.calls[0][1][1]["content"]


@pytest.mark.parametrize("quote,chunk", [
    ("since 2021 SHE has   worked at Brightwater", "c1"),       # case and spacing don't matter
    ("Skills: Python, Spark, and SQL", "c2"),                   # a comma more: word overlap still finds it
    ("Elodie speaks fluent Portuguese", None),                  # not in the document
    ("ok", None),
])
def test_quotes_are_found_in_their_chunk(quote, chunk):
    assert deep_extract.find_chunk(quote, CHUNKS) == chunk


def test_facts_are_written_under_the_chunk_that_states_them():
    ext = deep_extract.Extraction(entities={"Elodie Marchand": "PERSON", "Brightwater": "ORGANIZATION",
                                            "Spark": "SKILL", "Remote": "LOCATION"},
                                  facts=[deep_extract.Fact("Elodie Marchand", "HAS_SKILL", "Spark", "q", "c2")])
    out = deep_extract.per_chunk(ext, CHUNKS)
    names = {cid: {e["name"] for e in ents} for cid, (ents, _) in out.items()}
    assert out["c2"][1] == [{"source": "Elodie Marchand", "target": "Spark", "type": "HAS_SKILL"}]
    assert names["c1"] == {"Elodie Marchand", "Brightwater", "Remote"}   # named there; Remote: named nowhere
    assert names["c2"] == {"Elodie Marchand", "Spark", "Brightwater"}


# ---- ingest_text: the old graph is replaced only after a successful extraction -----------------------------------

CORPUS = {"id": "c_test", "name": "Candidates", "description": "", "schema": SCHEMA, "kind": "documents"}


@pytest.fixture
def no_db(monkeypatch):
    calls = []
    monkeypatch.setattr(corpus_graph, "remove_document", lambda c, d: calls.append("remove"))
    monkeypatch.setattr(corpus_graph, "_write_graph", lambda *a: calls.append("write") or (3, 2))
    monkeypatch.setattr(corpus_graph, "resolve_corpus", lambda *a: calls.append("resolve"))
    monkeypatch.setattr(corpus_graph, "embed_chunks", lambda *a: calls.append("embed"))
    import src.tables
    monkeypatch.setattr(src.tables, "ingest", lambda *a: 0)
    return calls


def test_a_rebuild_replaces_the_graph_after_extraction(no_db, monkeypatch):
    monkeypatch.setattr(corpus_graph, "_extract_detailed",
                        lambda *a: (no_db.append("extract") or {}, {"kept": 2}))
    out = corpus_graph.ingest_text(CORPUS, "DOC-1", "elodie", CHUNKS[0][1], lambda *a: None, mode="detailed", replace=True)
    assert no_db == ["extract", "remove", "write", "resolve", "embed"]
    assert out["mode"] == "detailed" and out["detail"] == {"kept": 2} and out["relations"] == 2


def test_a_failed_extraction_keeps_the_old_graph(no_db, monkeypatch):
    def boom(*a):
        raise TimeoutError("the PC went to sleep")
    monkeypatch.setattr(corpus_graph, "_extract_detailed", boom)
    with pytest.raises(TimeoutError):
        corpus_graph.ingest_text(CORPUS, "DOC-1", "elodie", CHUNKS[0][1], lambda *a: None, mode="detailed", replace=True)
    assert no_db == []


def test_the_default_mode_comes_from_the_setting(no_db, monkeypatch):
    monkeypatch.setattr(config, "INGEST_MODE", "fast")
    monkeypatch.setattr(corpus_graph, "_extract_fast", lambda *a: {})
    assert corpus_graph.ingest_text(CORPUS, "D", "t", "text here", lambda *a: None)["mode"] == "fast"
    with pytest.raises(ValueError):
        corpus_graph.ingest_text(CORPUS, "D", "t", "text", lambda *a: None, mode="thorough")


# ---- the rebuild script -------------------------------------------------------------------------------------------

class _Resp:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def test_the_extraction_server_check_says_how_to_fix_it(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "EXTRACT_OLLAMA_URL", "http://192.168.1.50:11434")
    monkeypatch.setattr(config, "EXTRACT_MODEL", "gemma4:26b")
    monkeypatch.setattr(rebuild_graph.requests, "get", lambda url, timeout: _Resp({"models": [{"name": "llama3.1:8b"}]}))
    with pytest.raises(SystemExit, match="ollama pull gemma4:26b"):
        rebuild_graph.check_extraction_server()
    monkeypatch.setattr(rebuild_graph.requests, "get", lambda url, timeout: _Resp({"models": [{"name": "gemma4:26b"}]}))
    assert rebuild_graph.check_extraction_server() == "gemma4:26b on http://192.168.1.50:11434"

    def down(url, timeout):
        raise rebuild_graph.requests.ConnectionError("refused")
    monkeypatch.setattr(rebuild_graph.requests, "get", down)
    with pytest.raises(SystemExit, match="OLLAMA_HOST=0.0.0.0"):
        rebuild_graph.check_extraction_server()


def test_resuming_reuses_the_report_and_skips_finished_documents(tmp_path):
    path = tmp_path / "r.json"
    r = rebuild_graph.Report(path, {"mode": "detailed"})
    r.data["documents"] = {"A": {"ok": True}, "B": {"ok": False, "error": "x"}}
    r.save()
    again = rebuild_graph.Report(path, {}, json.loads(path.read_text()))
    assert again.done_ok() == {"A"}
    assert rebuild_graph._resume_hint(["--all", "--resume", "old.json", "--mode", "detailed"], path) == \
        f"python -m src.rebuild_graph --all --mode detailed --resume {path}"


def test_the_finish_time_follows_the_pace_so_far():
    from datetime import datetime
    eta = rebuild_graph._eta([60.0, 60.0], 2000, 10_000, 10)          # 60 s per 1,000 chars -> 600 s left
    assert 590 <= (eta - datetime.now()).total_seconds() <= 610
    eta = rebuild_graph._eta([60.0], 1000, 100, 10)                    # tiny documents still cost ~3 calls each
    assert (eta - datetime.now()).total_seconds() >= 350
