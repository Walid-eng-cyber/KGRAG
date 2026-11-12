import { useEffect, useState, type CSSProperties } from "react";
import { api, ApiError } from "../api";
import type { Benchmark } from "../types";

const pct = (x: number) => `${Math.round(x * 100)}%`;
const pp = (g: number, v: number) => {
  const d = Math.round((g - v) * 100);
  return { text: `${d > 0 ? "+" : d < 0 ? "−" : "+"}${Math.abs(d)}pp`, cls: d > 0 ? "pos" : d < 0 ? "neg" : "" };
};

export function BenchmarkPage() {
  const [b, setB] = useState<Benchmark | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.benchmark().then(setB).catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  if (error) return <div className="card error" role="alert">{error}</div>;
  if (!b) return <div className="empty">Loading benchmark…</div>;

  const cat = (k: string) => b.categories.find((c) => c.key === k);
  const three = cat("three_hop"), agg = cat("aggregation");
  const o = pp(b.overall.graphrag, b.overall.vector);
  const lat = b.latency;
  return (
    <>
      <div className="kpis">
        <div className="card kpi"><span className="kpi-k">Overall accuracy</span><span className="kpi-v">{pct(b.overall.graphrag)}</span>
          <span className="kpi-d"><span className={o.cls}>{o.text}</span> vs {pct(b.overall.vector)} vector</span></div>
        {three && <div className="card kpi"><span className="kpi-k">Three-hop questions</span><span className="kpi-v">{pct(three.graphrag)}</span>
          <span className="kpi-d"><span className={pp(three.graphrag, three.vector).cls}>{pp(three.graphrag, three.vector).text}</span> vs {pct(three.vector)} vector</span></div>}
        {agg && <div className="card kpi"><span className="kpi-k">Aggregation</span><span className="kpi-v">{pct(agg.graphrag)}</span>
          <span className="kpi-d"><span className={pp(agg.graphrag, agg.vector).cls}>{pp(agg.graphrag, agg.vector).text}</span> vs {pct(agg.vector)} vector</span></div>}
        <div className="card kpi"><span className="kpi-k">Median latency / query</span>
          <span className="kpi-v">{lat.graphrag_median_s != null ? `${Math.round(lat.graphrag_median_s)} s` : "—"}</span>
          <span className="kpi-d">vs {lat.vector_median_s != null ? `${Math.round(lat.vector_median_s)} s` : "—"} vector baseline</span></div>
      </div>

      <div className="bench-grid">
        <section className="card">
          <div className="card-h">
            <h2 className="card-t">Accuracy by difficulty</h2>
            <div className="legend-light">
              <span><span className="sw" style={{ background: "var(--graph)" }} />GraphRAG</span>
              <span><span className="sw" style={{ background: "#a8a59b" }} />Plain vector RAG</span>
            </div>
          </div>
          <table className="score-table">
            <thead>
              <tr><th scope="col">Difficulty</th><th scope="col">GraphRAG</th><th scope="col">Plain vector RAG</th><th scope="col" className="num">Delta</th></tr>
            </thead>
            <tbody>
              {[...b.categories, { key: "overall", label: "Overall", ...b.overall }].map((c, i) => {
                const d = pp(c.graphrag, c.vector);
                const cell = (v: number, other: number, cls: string) => (
                  <td className={v > other ? "best" : undefined}>
                    <span className="score">{pct(v)}</span>
                    <span className="bar-track"><span className={cls} style={{ display: "block", width: pct(v), "--d": `${i * 0.07}s` } as CSSProperties} /></span>
                  </td>
                );
                return (
                  <tr key={c.key} className={c.key === "overall" ? "overall" : undefined}>
                    <th scope="row">{c.label}<small>{c.key === "overall" ? `n=${c.n}` : `${c.key} · n=${c.n}`}</small></th>
                    {cell(c.graphrag, c.vector, "bar-g")}
                    {cell(c.vector, c.graphrag, "bar-v")}
                    <td className={`num ${d.cls}`}>{d.text}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>

        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <section className="card">
            <div className="card-h"><h2 className="card-t">The cost of that accuracy</h2></div>
            <table>
              <thead><tr><th scope="col"></th><th className="num" scope="col">Vector</th><th className="num" scope="col">GraphRAG</th></tr></thead>
              <tbody>
                {b.costs.map((c) => (
                  <tr key={c.label}><td>{c.label}</td><td className="num">{c.vector ?? "—"}</td><td className="num">{c.graphrag ?? "—"}</td></tr>
                ))}
              </tbody>
            </table>
          </section>
          <div className="note">
            <b>Honest caveats.</b> Small corpus (3 filings, 27 chunks) and keyword-based scoring, so the numbers are
            directional. The two-hop dip is real: the template library doesn’t cover those chains, so GraphRAG abstains
            while vector RAG scores by keyword luck.
          </div>
          <span className="muted">
            Source: {b.source === "results-file" ? "data/benchmark_results.json (your latest run)" : "published results in the README"} ·
            reproduce with <span className="mono">python -m src.benchmark --all</span>
          </span>
        </div>
      </div>
    </>
  );
}
