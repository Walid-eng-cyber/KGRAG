import { useEffect, useState } from "react";

/** Waiting state for a local-model answer (typically 20–40 s). */
export function Pending({ demo }: { demo: boolean }) {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const id = setInterval(() => setSecs(Math.floor((Date.now() - t0) / 1000)), 250);
    return () => clearInterval(id);
  }, []);
  return (
    <section className="card pending" aria-live="polite" aria-busy="true">
      <svg className="spinner" viewBox="-24 -24 48 48" aria-hidden="true">
        <polygon points="0,-20 17.3,-10 17.3,10 0,20 -17.3,10 -17.3,-10" fill="none" stroke="#5B4BD6" strokeWidth="3" strokeDasharray="20 10" />
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <span className="step-v">
          Running the pipeline{demo ? "" : " on local models"} · <span className="mono">{secs}s</span>
          {!demo && <span className="muted"> (usually 20–40 s)</span>}
        </span>
        <div className="pending-steps" aria-hidden="true">
          <span>route</span><span>retrieve</span><span>verbalize</span><span>ground</span><span>validate citations</span>
        </div>
      </div>
    </section>
  );
}
