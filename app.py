"""KGRAG frontend — ask a question, see the grounded answer, and get a
visualization that adapts to how it was answered:

  GRAPH  → a node/edge diagram of the facts that were traversed
  VECTOR → the retrieved passages with similarity scores
  HYBRID → both, side by side

Run:  streamlit run app.py
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import config
from src.answer import answer, graph_triples

st.set_page_config(page_title="KGRAG", page_icon="🔎", layout="wide")

st.title("🔎 KGRAG — SEC filings knowledge-graph RAG")
st.caption("Ask about Apple, Microsoft, or NVIDIA 10-K filings. "
           "The router chooses graph, vector, or both — and the view adapts.")

EXAMPLES = {
    "Graph — aggregation": "How many subsidiaries does Microsoft have?",
    "Graph — comparison": "Compare the subsidiaries of Apple and Microsoft.",
    "Graph — list": "What are Apple's subsidiaries?",
    "Vector — explanation": "What does Apple say about supply chain risk?",
    "Vector — single fact": "Who is Microsoft's CEO?",
}

with st.sidebar:
    st.subheader("Try an example")
    for label, q in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["q"] = q
    st.divider()
    st.caption(f"router: `{config.ROUTER_MODEL}`  ·  answer: `{config.ANSWER_MODEL}`"
               f"  ·  embed: `{config.EMBED_MODEL}`")

question = st.text_input("Your question",
                         value=st.session_state.get("q", "How many subsidiaries does Microsoft have?"))
ask = st.button("Ask", type="primary")


def render_graph(g: dict) -> None:
    triples = graph_triples(g)
    qt = g.get("query_type")
    if triples:
        dot = ["digraph { rankdir=LR; bgcolor=transparent;",
               'node [shape=box, style="rounded,filled", fillcolor="#eeedfe", '
               'color="#534ab7", fontname="Helvetica"];',
               'edge [color="#888780", fontsize=10, fontname="Helvetica"];']
        for s, rel, t in triples[:40]:
            dot.append(f'"{html.escape(s)}" -> "{html.escape(t)}" '
                       f'[label="{html.escape(rel.lower())}"];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)
    elif qt == "count_subsidiaries" and g.get("rows"):
        st.metric(g["params"].get("name", "count"), g["rows"][0]["count"])
    elif qt == "compare_subsidiary_counts" and g.get("rows"):
        df = pd.DataFrame(g["rows"]).set_index("company")
        st.bar_chart(df)
    else:
        st.info("The graph path returned no rows to visualize.")


def render_vector(hits: list[dict]) -> None:
    if not hits:
        st.info("No passages retrieved.")
        return
    df = pd.DataFrame([{"chunk": h["chunk_id"][:8], "similarity": round(h["similarity"], 3)}
                       for h in hits]).set_index("chunk")
    st.bar_chart(df, horizontal=True)
    for h in hits:
        with st.expander(f"[{h['doc_id']} · {h['section_path']} · {h['filing_date']}]  "
                         f"sim={h['similarity']:.3f}  (chunk {h['chunk_id'][:8]})"):
            st.write(" ".join(h["text"].split())[:900] + "…")


if ask and question.strip():
    with st.spinner("Routing → retrieving → grounding (local models, ~20–40s)…"):
        try:
            r = answer(question)
        except Exception as e:
            st.error(f"Backend error (are Neo4j and Postgres up?): {e}")
            st.stop()

    # --- routing explanation ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Path used", r["path_used"].split(" ")[0])
    c2.metric("Classified", r["classified"])
    c3.metric("Confidence", f"{r['confidence']:.2f}")
    why = {
        "GRAPH": "A connection / comparison / aggregation question — answered by "
                 "traversing the knowledge graph (Neo4j).",
        "VECTOR": "A definition / single-fact / explanatory question — answered by "
                  "semantic search over passages (pgvector).",
        "HYBRID": "Ambiguous or multi-part (or low confidence) — both paths were "
                  "run and merged.",
    }.get(r["path_used"].split(" ")[0], "")
    st.caption("🧭 " + why)

    # --- the grounded answer + citation validation ---
    st.subheader("Answer")
    st.markdown(r["answer"])
    if r["citations_valid"]:
        st.success(f"✓ Every citation resolves to a retrieved source "
                   f"(validated in {r['attempts']} pass(es)).")
    else:
        st.warning("⚠ Some citations could not be validated and were removed.")

    # --- path-adaptive visualization ---
    st.subheader("How it was answered")
    has_graph = bool(r.get("graph_result", {}).get("rows"))
    has_vec = bool(r.get("vector_hits"))
    if has_graph and has_vec:
        t1, t2 = st.tabs(["🕸 Graph facts", "📄 Retrieved passages"])
        with t1:
            render_graph(r["graph_result"])
        with t2:
            render_vector(r["vector_hits"])
    elif has_graph:
        st.markdown("**Graph facts traversed:**")
        render_graph(r["graph_result"])
    elif has_vec:
        st.markdown("**Passages retrieved (by similarity):**")
        render_vector(r["vector_hits"])
    else:
        st.info("No structured evidence to visualize.")

    # --- sources ---
    with st.expander(f"Sources ({len(r['sources'])})"):
        for s in r["sources"]:
            tag = s["tag"]
            if s["kind"] == "graph-derived fact":
                st.markdown(f"**[{tag}]** _(graph)_ {s['text']}")
            else:
                st.markdown(f"**[{tag}]** _({s['cite']})_ {s['text'][:300]}…")
