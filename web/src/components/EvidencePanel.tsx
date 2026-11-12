import { useEffect, useState, type CSSProperties } from "react";
import { tagKind } from "../lib/answerText";
import type { AskResponse } from "../types";
import { GraphStage } from "./GraphStage";

type Tab = "graph" | "passages";

function Metrics({ r, selectedTag }: { r: AskResponse; selectedTag: string | null }) {
  const max = Math.max(...r.graph.metrics.map((m) => m.value), 1);
  return (
    <div className="card-b" style={{ padding: "4px 0" }}>
      {r.graph.metrics.map((m, i) => (
        <div className="metric-row" key={m.label}>
          <span>{m.label}</span>
          <span className="metric-track">
            <span className="metric-fill" style={{ display: "block", width: `${(m.value / max) * 100}%`, "--d": `${i * 0.12}s`,
              outline: selectedTag === m.tag ? "2px solid #d9d2ff" : undefined } as CSSProperties} />
          </span>
          <span className="metric-val">{m.value}</span>
          <span className={`cite ${tagKind(m.tag)}`}>{m.tag}</span>
        </div>
      ))}
    </div>
  );
}

function Passages({ r, selectedTag }: { r: AskResponse; selectedTag: string | null }) {
  return (
    <div className="card-b" style={{ padding: "4px 0", gap: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="muted">Retrieved by cosine similarity (pgvector · bge-m3)</span>
        <span className="muted">dimmed = retrieved but not sent to the model</span>
      </div>
      {r.passages.map((p, i) => (
        <details className="passage" key={p.chunk_id} open={selectedTag != null && p.tag === selectedTag}>
          <summary>
            <div className="sim-row">
              {p.tag ? <span className="cite passage">{p.tag}</span> : <span className="tagless">—</span>}
              <span className="sim-meta">{p.doc_id} · {p.section} · {p.filing_date}</span>
              <span className="sim-track">
                <span className={`sim-fill${p.tag ? "" : " unused"}`}
                  style={{ display: "block", width: `${Math.max(0, Math.min(1, p.similarity)) * 100}%`, "--d": `${i * 0.08}s` } as CSSProperties} />
              </span>
              <span className="sim-val">{p.similarity.toFixed(3)}</span>
            </div>
          </summary>
          <div className="sim-row" style={{ marginTop: 6 }}>
            <span />
            <p className="passage-text">{p.text}</p>
          </div>
        </details>
      ))}
    </div>
  );
}

export function EvidencePanel({ r, selectedTag }: { r: AskResponse; selectedTag: string | null }) {
  const hasGraph = r.graph.edges.length > 0 || r.graph.metrics.length > 0;
  const hasPassages = r.passages.length > 0;
  const [tab, setTab] = useState<Tab>(hasGraph ? "graph" : "passages");

  useEffect(() => setTab(hasGraph ? "graph" : "passages"), [r, hasGraph]);
  useEffect(() => {
    if (selectedTag) setTab(tagKind(selectedTag) === "graph" && hasGraph ? "graph" : hasPassages ? "passages" : "graph");
  }, [selectedTag, hasGraph, hasPassages]);

  const facts = r.graph.edges.length || r.graph.metrics.length;
  return (
    <section className="card" aria-label="How it was answered">
      <div className="card-h">
        <h2 className="card-t">How it was answered</h2>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === "graph"} disabled={!hasGraph} onClick={() => setTab("graph")}>
            Graph · {facts}
          </button>
          <button role="tab" aria-selected={tab === "passages"} disabled={!hasPassages} onClick={() => setTab("passages")}>
            Passages · {r.passages.length}
          </button>
        </div>
      </div>
      <div className="card-b">
        {tab === "graph" && hasGraph && (r.graph.edges.length > 0
          ? <GraphStage graph={r.graph} cited={r.cited} selectedTag={selectedTag} />
          : <Metrics r={r} selectedTag={selectedTag} />)}
        {tab === "passages" && hasPassages && <Passages r={r} selectedTag={selectedTag} />}
        {!hasGraph && !hasPassages && (
          <div className="empty">{r.graph.error ? `Graph path: ${r.graph.error}` : "No evidence was retrieved for this question."}</div>
        )}
        {tab === "graph" && r.graph.cypher && (
          <details className="cypher">
            <summary>Parameterized Cypher ({r.graph.query_type})</summary>
            <div className="code">{r.graph.cypher}</div>
          </details>
        )}
      </div>
    </section>
  );
}
