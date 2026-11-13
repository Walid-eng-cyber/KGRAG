"""Turn a graph-path result into typed nodes and edges for the UI.

The React graph draws every entity with the shape of its ontology type
(company = hexagon, risk = diamond, ...). The graph templates return names,
not labels, so we infer each endpoint's type from the relation using the
same VALIDATION_SCHEMA the extractor is constrained by.
"""
from __future__ import annotations

import re

from src.answer import graph_edges
from src.schema import STRUCTURED_RELATIONS, VALIDATION_SCHEMA

# relation -> (subject type, object type). SEGMENT->OFFERS_PRODUCT is the only
# relation with two subject types; COMPANY is the common case, so it wins.
REL_TYPES: dict[str, tuple[str, str]] = {}
for _subj, _rel, _obj in [*VALIDATION_SCHEMA, *STRUCTURED_RELATIONS]:
    REL_TYPES.setdefault(_rel, (_subj, _obj))

_CITE = re.compile(r"\[(G\d+|P\d+|C\d+)\]")

# query types whose verbalization is ONE statement covering every row ([G1]),
# versus one statement per row ([G1], [G2], ...).
_SINGLE_STATEMENT = {"subsidiaries_of", "risks_of", "executives_of",
                     "competitors_of", "regulators_of", "companies_facing_risk", "insiders_of"}


def cited_tags(text: str) -> list[str]:
    """Citation labels used in an answer, in first-appearance order."""
    return list(dict.fromkeys(_CITE.findall(text or "")))


def _node_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "node"


def build_graph(graph_result: dict | None) -> dict:
    """{nodes, edges, layout, metrics} for the evidence graph.

    layout: 'path' for path_between (chain), 'radial' for one anchor,
    'bipartite' for several anchors, 'none' when there is nothing to draw.
    metrics: aggregate rows (counts) that are numbers, not edges.
    """
    g = graph_result or {}
    qt = g.get("query_type")
    rows = g.get("rows") or []

    metrics: list[dict] = []
    if qt == "count_subsidiaries" and rows:
        metrics = [{"label": g.get("params", {}).get("name", "count"),
                    "value": rows[0].get("count", 0), "tag": "G1"}]
    elif qt == "insider_net_activity" and rows and (rows[0].get("value_bought") or rows[0].get("value_sold")):
        metrics = [{"label": "Bought (open market, $)", "value": round(rows[0].get("value_bought") or 0), "tag": "G1"},
                   {"label": "Sold (open market, $)", "value": round(rows[0].get("value_sold") or 0), "tag": "G2"}]
    elif qt == "compare_subsidiary_counts":
        metrics = [{"label": r.get("company"), "value": r.get("count", 0), "tag": f"G{i + 1}"}
                   for i, r in enumerate(rows)]

    from src.corpus_graph import edges as corpus_edges
    typed = corpus_edges(g)            # document corpora carry their own entity types
    if typed is not None:
        return _build_typed(g, typed, metrics)

    edges_in = graph_edges(g)
    if qt == "path_between":
        # Paths are undirected: a hop may arrive as (Export controls)-[FACES_RISK]-(Microsoft).
        # Orient each hop so a queried endpoint (a/b, the entities asked about) is the subject.
        ends = {str(v) for k, v in g.get("params", {}).items() if k in ("a", "b")}
        edges_in = [(d, r, s, i) if d in ends and s not in ends else (s, r, d, i) for s, r, d, i in edges_in]
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for i, (src, rel, dst, row) in enumerate(edges_in):
        s_type, o_type = REL_TYPES.get(rel, ("COMPANY", "COMPANY"))
        # Form 4 rows say whether the insider is a person or a fund/firm (10% owners often are).
        row_type = rows[row].get("_insider_type") if row < len(rows) else None
        if row_type and rel in ("INSIDER_OF", "TRADED") and qt != "companies_sharing_insiders":
            s_type = row_type
        for name, typ, role in ((src, s_type, "anchor"), (dst, o_type, "target")):
            nid = _node_id(name)
            node = nodes.setdefault(nid, {"id": nid, "label": name, "type": typ, "role": role})
            if role == "anchor":
                node["role"] = "anchor"   # an entity that is ever a subject is an anchor
        tag = "G1" if qt in _SINGLE_STATEMENT else f"G{row + 1}"
        edges.append({"id": f"e{i}", "source": _node_id(src), "target": _node_id(dst),
                      "relation": rel, "tag": tag})

    anchors = [n for n in nodes.values() if n["role"] == "anchor"]
    if not edges:
        layout = "none"
    elif qt == "path_between":
        layout = "path"
    elif len(anchors) <= 1:
        layout = "radial"
    else:
        layout = "bipartite"

    return {"query_type": qt, "cypher": g.get("cypher"), "layout": layout,
            "nodes": list(nodes.values()), "edges": edges, "metrics": metrics,
            "error": g.get("error")}


def _build_typed(g: dict, typed: list[tuple], metrics: list[dict]) -> dict:
    """Graph view for a document corpus: node types come from the stored entity labels."""
    qt, rows = g.get("query_type"), g.get("rows") or []
    if qt in ("entities_of_type", "count_of_type") and rows:
        if qt == "count_of_type":
            metrics = [{"label": rows[0]["type"].replace("_", " ").title(), "value": rows[0]["count"], "tag": "G1"}]
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for i, (src, rel, dst, row, s_type, o_type) in enumerate(typed):
        for name, typ, role in ((src, s_type, "anchor"), (dst, o_type, "target")):
            nid = _node_id(name)
            node = nodes.setdefault(nid, {"id": nid, "label": name, "type": typ, "role": role})
            if role == "anchor":
                node["role"] = "anchor"
        edges.append({"id": f"e{i}", "source": _node_id(src), "target": _node_id(dst),
                      "relation": rel, "tag": f"G{row + 1}"})
    anchors = [n for n in nodes.values() if n["role"] == "anchor"]
    layout = "none" if not edges else "path" if qt == "path_between" else "radial" if len(anchors) <= 1 else "bipartite"
    return {"query_type": qt, "cypher": g.get("cypher"), "layout": layout, "nodes": list(nodes.values()),
            "edges": edges, "metrics": metrics, "error": g.get("error")}
