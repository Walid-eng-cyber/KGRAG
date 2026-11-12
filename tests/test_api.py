"""API tests: no Neo4j, Postgres or Ollama needed (answer() is stubbed)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server.benchmark_view import benchmark_summary
from server.graph_view import build_graph, cited_tags


@pytest.fixture
def client():
    return TestClient(main.app)


# ---------- graph_view ----------

def test_radial_graph_types_nodes_from_the_ontology():
    g = build_graph({"query_type": "risks_of", "params": {"name": "Apple"},
                     "rows": [{"risk": "Export controls"}, {"risk": "Supply chain disruption"}]})
    assert g["layout"] == "radial"
    types = {n["label"]: (n["type"], n["role"]) for n in g["nodes"]}
    assert types["Apple"] == ("COMPANY", "anchor")
    assert types["Export controls"] == ("RISK", "target")
    assert {e["tag"] for e in g["edges"]} == {"G1"}  # one verbalized statement covers all rows


def test_bipartite_when_several_anchors():
    g = build_graph({"query_type": "companies_facing_risk", "params": {"risk": "Export controls"},
                     "rows": [{"company": "Apple"}, {"company": "NVIDIA"}]})
    assert g["layout"] == "bipartite"
    assert sum(n["role"] == "anchor" for n in g["nodes"]) == 2


def test_path_edges_are_tagged_per_path_row():
    g = build_graph({"query_type": "path_between", "params": {"a": "A", "b": "B"},
                     "rows": [{"path": ["A", "X", "B"], "relations": ["FACES_RISK", "FACES_RISK"]},
                              {"path": ["A", "Nasdaq", "B"], "relations": ["LISTED_ON", "LISTED_ON"]}]})
    assert g["layout"] == "path"
    assert [e["tag"] for e in g["edges"]] == ["G1", "G1", "G2", "G2"]
    types = {n["label"]: (n["type"], n["role"]) for n in g["nodes"]}
    assert types["B"] == ("COMPANY", "anchor")   # endpoint, even though it ends the hop X-[FACES_RISK]-B
    assert types["X"] == ("RISK", "target")
    assert types["Nasdaq"] == ("STOCK_EXCHANGE", "target")


def test_aggregates_become_metrics_not_edges():
    g = build_graph({"query_type": "compare_subsidiary_counts", "params": {"names": ["A", "B"]},
                     "rows": [{"company": "B", "count": 14}, {"company": "A", "count": 9}]})
    assert g["layout"] == "none" and g["edges"] == []
    assert g["metrics"] == [{"label": "B", "value": 14, "tag": "G1"}, {"label": "A", "value": 9, "tag": "G2"}]


def test_empty_or_errored_graph():
    assert build_graph(None)["layout"] == "none"
    assert build_graph({"error": "boom", "rows": []})["error"] == "boom"


def test_cited_tags_in_order_without_duplicates():
    assert cited_tags("A [G2] b [P1] c [G2] d [G1].") == ["G2", "P1", "G1"]


# ---------- benchmark ----------

def test_published_benchmark_is_consistent_with_readme(tmp_path):
    b = benchmark_summary(tmp_path / "missing.json")
    assert b["source"] == "published"
    assert sum(c["n"] for c in b["categories"]) == 36
    assert round(sum(c["graphrag"] * c["n"] for c in b["categories"])) == 28   # 78%
    assert round(sum(c["vector"] * c["n"] for c in b["categories"])) == 20     # 56%


def test_benchmark_from_results_file(tmp_path):
    f = tmp_path / "benchmark_results.json"
    f.write_text(json.dumps({"rows": [
        {"cat": "three_hop", "ours": True, "vanilla": False},
        {"cat": "three_hop", "ours": True, "vanilla": False},
        {"cat": "single_hop", "ours": False, "vanilla": True},
    ], "summary": {"ours_median_s": 24.0, "van_median_s": 7.5}}))
    b = benchmark_summary(f)
    assert b["source"] == "results-file"
    three = next(c for c in b["categories"] if c["key"] == "three_hop")
    assert (three["graphrag"], three["vector"], three["n"]) == (1.0, 0.0, 2)
    assert b["costs"][1]["graphrag"] == "24 s"


# ---------- endpoints ----------

def test_ask_shapes_a_real_answer(client, monkeypatch):
    def fake_answer(q):
        return {"question": q, "classified": "HYBRID", "confidence": 0.5, "path_used": "HYBRID (low confidence)",
                "answer": "Apple faces export controls [G1] and relies on partners in Asia [P1].",
                "sources": [{"tag": "G1", "kind": "graph-derived fact", "text": "Apple faces export controls."},
                            {"tag": "P1", "kind": "retrieved passage", "cite": "AAPL-10K", "text": "partners"}],
                "graph_result": {"query_type": "risks_of", "params": {"name": "Apple"},
                                 "rows": [{"risk": "Export controls"}]},
                "vector_hits": [{"chunk_id": "abc", "doc_id": "AAPL-10K", "section_path": "Item 1A",
                                 "filing_date": "2025-11-01", "similarity": 0.8, "text": "partners  in Asia"}],
                "attempts": 1, "citations_valid": True, "uncited_claims": []}
    monkeypatch.setattr(main, "DEMO", False)
    monkeypatch.setattr("src.answer.answer", fake_answer)
    r = client.post("/api/ask", json={"question": "What risks does Apple face?"})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "HYBRID" and body["cited"] == ["G1", "P1"]
    assert body["sources"][0]["kind"] == "graph" and body["passages"][0]["tag"] == "P1"
    assert body["passages"][0]["text"] == "partners in Asia"
    assert body["graph"]["layout"] == "radial" and body["demo"] is False


def test_ask_backend_failure_is_a_503(client, monkeypatch):
    def boom(q):
        raise ConnectionError("neo4j down")
    monkeypatch.setattr(main, "DEMO", False)
    monkeypatch.setattr("src.answer.answer", boom)
    r = client.post("/api/ask", json={"question": "anything at all"})
    assert r.status_code == 503 and "server log" in r.json()["detail"]


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


@pytest.mark.parametrize("q,layout", [
    ("What are Apple's subsidiaries?", "radial"),
    ("Which companies face export control risk?", "bipartite"),
    ("How is NVIDIA connected to Microsoft?", "path"),
    ("Compare the number of subsidiaries of Apple and Microsoft.", "none"),
])
def test_demo_mode(client, monkeypatch, q, layout):
    monkeypatch.setattr(main, "DEMO", True)
    body = client.post("/api/ask", json={"question": q}).json()
    assert body["demo"] is True and body["graph"]["layout"] == layout
    tags = {s["tag"] for s in body["sources"]}
    assert set(body["cited"]) <= tags  # every demo citation resolves to a source


def test_meta_and_benchmark_endpoints(client):
    assert len(client.get("/api/meta").json()["examples"]) >= 4
    assert client.get("/api/benchmark").json()["overall"]["n"] > 0
