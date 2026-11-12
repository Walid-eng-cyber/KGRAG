"""Demo mode (KGRAG_DEMO=1): canned, clearly-labelled sample responses so the UI
can be explored without Neo4j, Postgres or Ollama. Shapes match src.answer.answer().
The facts are illustrative samples, not output from a real run."""
from __future__ import annotations

import re

_P = {
    "aapl_supply": {"chunk_id": "a1f3c9d2e7b04411", "doc_id": "AAPL-10K", "section_path": "Item 1A · Risk Factors",
                    "filing_date": "2025-11-01", "similarity": 0.812,
                    "text": "The Company depends on component and product manufacturing and logistical services "
                            "provided by outsourcing partners, many of which are located outside of the U.S. "
                            "Substantially all of the Company's manufacturing is performed in whole or in part by "
                            "outsourcing partners located primarily in Asia."},
    "aapl_trade": {"chunk_id": "b7e2d1c0a9f84422", "doc_id": "AAPL-10K", "section_path": "Item 1A · Risk Factors",
                   "filing_date": "2025-11-01", "similarity": 0.774,
                   "text": "Restrictions on international trade, such as tariffs and other controls on imports or "
                           "exports of goods, technology or data, can materially adversely affect the Company's "
                           "business and supply chain."},
    "nvda_foundry": {"chunk_id": "c4a8e6f1b2d34433", "doc_id": "NVDA-10K", "section_path": "Item 1A · Risk Factors",
                     "filing_date": "2025-02-26", "similarity": 0.709,
                     "text": "We depend on foundries to manufacture our semiconductor wafers and on third parties "
                             "for assembly and testing; a disruption at any of them could delay supply."},
    "aapl_mda": {"chunk_id": "d9b1f7a3c6e54444", "doc_id": "AAPL-10K", "section_path": "Item 7 · MD&A",
                 "filing_date": "2025-11-01", "similarity": 0.662,
                 "text": "Gross margins are subject to volatility due to component costs, product mix and "
                         "the effects of the Company's supply chain."},
}


def _graph(qt, rows, params, cypher):
    return {"query_type": qt, "rows": rows, "params": params, "cypher": cypher}


FIXTURES = [
    {   # GRAPH, one anchor -> radial
        "match": r"subsidiar(y|ies) of apple|apple'?s subsidiar",
        "question": "What are Apple's subsidiaries?",
        "classified": "GRAPH", "confidence": 0.93, "path_used": "GRAPH",
        "graph_result": _graph("subsidiaries_of", [{"subsidiary": s} for s in [
            "Braeburn Capital", "Apple Operations International", "Apple Sales International",
            "Apple Operations Europe", "Apple Distribution International", "Beats Electronics"]],
            {"name": "Apple"}, "MATCH (c:COMPANY {name:'Apple'})-[r:HAS_SUBSIDIARY]->(s) RETURN s.name AS subsidiary"),
        "vector_hits": [],
        "answer": "Apple's subsidiaries in the graph include Braeburn Capital, Apple Operations International, "
                  "Apple Sales International, Apple Operations Europe, Apple Distribution International and "
                  "Beats Electronics [G1].",
    },
    {   # HYBRID: graph risks + passages
        "match": r"risk|supply chain",
        "question": "What risks does Apple face, and how does it describe its supply chain?",
        "classified": "HYBRID", "confidence": 0.54, "path_used": "HYBRID (low confidence)",
        "graph_result": _graph("risks_of", [{"risk": r} for r in [
            "Supply chain disruption", "Export controls", "Geopolitical tension", "Consumer demand",
            "Component shortages"]], {"name": "Apple"},
            "MATCH (c {name:'Apple'})-[r:FACES_RISK]->(rk) RETURN rk.name AS risk"),
        "vector_hits": [_P["aapl_supply"], _P["aapl_trade"], _P["nvda_foundry"], _P["aapl_mda"]],
        "answer": "Apple discloses risks including supply chain disruption, export controls, geopolitical tension, "
                  "consumer demand and component shortages [G1].\n\nIt describes its supply chain as concentrated: "
                  "substantially all manufacturing is performed by outsourcing partners located primarily in Asia [P1], "
                  "and trade restrictions on goods or technology could materially affect the business [P2].",
    },
    {   # GRAPH, several anchors -> bipartite
        "match": r"compan(y|ies).*export|export.*compan",
        "question": "Which companies face export control risk?",
        "classified": "GRAPH", "confidence": 0.91, "path_used": "GRAPH",
        "graph_result": _graph("companies_facing_risk", [{"company": c} for c in ["Apple", "NVIDIA", "Microsoft"]],
                               {"risk": "Export controls"},
                               "MATCH (c)-[:FACES_RISK]->(r) WHERE r.name = 'Export controls' RETURN DISTINCT c.name AS company"),
        "vector_hits": [],
        "answer": "Apple, NVIDIA and Microsoft all disclose export controls as a risk factor [G1].",
    },
    {   # GRAPH path -> chain
        "match": r"connect|path|linked",
        "question": "How is NVIDIA connected to Microsoft?",
        "classified": "GRAPH", "confidence": 0.88, "path_used": "GRAPH",
        "graph_result": _graph("path_between", [{"path": ["NVIDIA", "Export controls", "Microsoft"],
                                                 "relations": ["FACES_RISK", "FACES_RISK"]},
                                                {"path": ["NVIDIA", "Nasdaq", "Microsoft"],
                                                 "relations": ["LISTED_ON", "LISTED_ON"]}],
                               {"a": "NVIDIA", "b": "Microsoft"},
                               "MATCH p = shortestPath((a {name:'NVIDIA'})-[*..3]-(b {name:'Microsoft'})) RETURN p"),
        "vector_hits": [],
        "answer": "NVIDIA and Microsoft both face export controls risk [G1], and both are listed on Nasdaq [G2].",
    },
    {   # GRAPH aggregation -> metrics
        "match": r"how many|count|compare",
        "question": "Compare the number of subsidiaries of Apple and Microsoft.",
        "classified": "GRAPH", "confidence": 0.95, "path_used": "GRAPH",
        "graph_result": _graph("compare_subsidiary_counts", [{"company": "Microsoft", "count": 14},
                                                             {"company": "Apple", "count": 9}],
                               {"names": ["Apple", "Microsoft"]},
                               "MATCH (c:COMPANY)-[:HAS_SUBSIDIARY]->(s) WHERE c.name IN ['Apple','Microsoft'] "
                               "RETURN c.name AS company, count(s) AS count"),
        "vector_hits": [],
        "answer": "Microsoft has 14 subsidiaries recorded in the graph [G1], compared with 9 for Apple [G2].",
    },
    {   # VECTOR only
        "match": r"ceo|who is|what does|describe|policy",
        "question": "What does Apple say about manufacturing?",
        "classified": "VECTOR", "confidence": 0.86, "path_used": "VECTOR",
        "graph_result": {},
        "vector_hits": [_P["aapl_supply"], _P["aapl_mda"], _P["aapl_trade"]],
        "answer": "Apple relies on outsourcing partners, primarily in Asia, for substantially all of its "
                  "manufacturing [P1], and notes that component costs and its supply chain make gross margins "
                  "volatile [P2].",
    },
]

# Example questions for a live server loaded with SEC Form 3/4/5 data (src/form4.py).
LIVE_EXAMPLES = [
    {"label": "Graph · insiders", "path": "GRAPH", "question": "Who are NVIDIA's insiders?"},
    {"label": "Graph · trades", "path": "GRAPH", "question": "Which insiders traded Microsoft shares recently?"},
    {"label": "Graph · aggregation", "path": "GRAPH", "question": "Were insiders net buyers or sellers of Tesla?"},
    {"label": "Graph · ranking", "path": "GRAPH", "question": "What were the largest insider sales this quarter?"},
    {"label": "Graph · multi-hop", "path": "GRAPH", "question": "Which companies share an insider with Alphabet?"},
    {"label": "Graph · person", "path": "GRAPH", "question": "What did Colette Kress buy or sell?"},
]

EXAMPLES = [
    {"label": "Graph · list", "path": "GRAPH", "question": "What are Apple's subsidiaries?"},
    {"label": "Graph · aggregation", "path": "GRAPH", "question": "Compare the number of subsidiaries of Apple and Microsoft."},
    {"label": "Graph · multi-hop", "path": "GRAPH", "question": "How is NVIDIA connected to Microsoft?"},
    {"label": "Graph · reverse lookup", "path": "GRAPH", "question": "Which companies face export control risk?"},
    {"label": "Hybrid", "path": "HYBRID", "question": "What risks does Apple face, and how does it describe its supply chain?"},
    {"label": "Vector · explanation", "path": "VECTOR", "question": "What does Apple say about manufacturing?"},
]


def demo_answer(question: str) -> dict:
    """An answer()-shaped dict for the fixture that best matches the question."""
    q = question.lower()
    fx = next((f for f in FIXTURES if f["question"].lower() == q), None)
    fx = fx or next((f for f in FIXTURES if re.search(f["match"], q)), FIXTURES[1])
    sources = []
    from src.answer import format_passages, verbalize_graph
    for i, s in enumerate(verbalize_graph(fx["graph_result"])):
        sources.append({"tag": f"G{i + 1}", "kind": "graph-derived fact", "text": s})
    for i, (cite, text) in enumerate(format_passages(fx["vector_hits"])):
        sources.append({"tag": f"P{i + 1}", "kind": "retrieved passage", "cite": cite, "text": text})
    return {"question": fx["question"], "classified": fx["classified"], "confidence": fx["confidence"],
            "path_used": fx["path_used"], "answer": fx["answer"], "sources": sources,
            "graph_result": fx["graph_result"], "vector_hits": fx["vector_hits"],
            "attempts": 1, "citations_valid": True, "uncited_claims": []}
