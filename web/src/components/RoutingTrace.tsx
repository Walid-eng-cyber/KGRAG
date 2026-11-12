import type { AskResponse } from "../types";

const THRESHOLD = 0.6; // ROUTER_CONFIDENCE_THRESHOLD: below it both paths run

export function RoutingTrace({ r }: { r: AskResponse }) {
  const conf = r.confidence ?? 0;
  const color = conf >= 0.8 ? "var(--ok)" : conf >= THRESHOLD ? "var(--warn)" : "var(--bad)";
  const facts = r.sources.filter((s) => s.kind === "graph").length;
  return (
    <section className="card trace" aria-label="How the question was routed">
      <div className="step">
        <span className="step-k">Router</span>
        <span className="step-v">Classified as <b>{(r.classified ?? "—").toLowerCase()}</b></span>
      </div>
      <span className="arrow" aria-hidden="true">→</span>
      <div className="step">
        <span className="step-k">Confidence</span>
        <span className="conf">
          <span className="conf-bar" title={`Below ${THRESHOLD} the router runs both paths`}>
            <span className="conf-fill" style={{ width: `${Math.round(conf * 100)}%`, background: color, display: "block" }} />
            <span className="conf-tick" style={{ left: `${THRESHOLD * 100}%` }} />
          </span>
          <span className="step-v mono">{r.confidence == null ? "—" : conf.toFixed(2)}</span>
        </span>
      </div>
      <span className="arrow" aria-hidden="true">→</span>
      <div className="step">
        <span className="step-k">Path</span>
        <span className={`badge ${r.path}`}>{r.path}</span>
      </div>
      {r.graph.query_type && (
        <>
          <span className="arrow" aria-hidden="true">→</span>
          <div className="step">
            <span className="step-k">Graph</span>
            <span className="step-v">
              {facts} fact{facts === 1 ? "" : "s"} · <span className="mono">{r.graph.query_type}</span>
            </span>
          </div>
        </>
      )}
      {r.passages.length > 0 && (
        <>
          <span className="arrow" aria-hidden="true">→</span>
          <div className="step">
            <span className="step-k">Vector</span>
            <span className="step-v">{r.passages.length} passage{r.passages.length === 1 ? "" : "s"}</span>
          </div>
        </>
      )}
      <div className="step trace-end">
        <span className="step-k">Latency</span>
        <span className="step-v mono">{r.demo ? "demo" : `${r.latency_s.toFixed(1)} s`}</span>
      </div>
    </section>
  );
}
