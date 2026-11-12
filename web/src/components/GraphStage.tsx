import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { shapeFor, TYPE_LABEL } from "../lib/geometry";
import { citedSets, layoutGraph, STAGE, type PlacedEdge } from "../lib/layout";
import type { GraphView } from "../types";
import { Glyph, MatchShape, Shape } from "./Shape";

type Phase = "build" | "traverse" | "match";

const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** A glowing dot that rides an edge from its source to its target. */
function Pulse({ edge, delayMs, durationMs }: { edge: PlacedEdge; delayMs: number; durationMs: number }) {
  const pathRef = useRef<SVGPathElement>(null);
  const dotRef = useRef<SVGCircleElement>(null);
  useEffect(() => {
    const path = pathRef.current, dot = dotRef.current;
    if (!path || !dot) return;
    const total = path.getTotalLength();
    let raf = 0;
    const t0 = performance.now() + delayMs;
    const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
    const tick = (now: number) => {
      const t = Math.min(1, Math.max(0, (now - t0) / durationMs));
      const p = path.getPointAtLength(ease(t) * total);
      dot.setAttribute("cx", p.x.toFixed(1));
      dot.setAttribute("cy", p.y.toFixed(1));
      dot.style.opacity = t <= 0 || t >= 1 ? "0" : t < 0.12 || t > 0.88 ? "0.5" : "1";
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [delayMs, durationMs]);
  return (
    <g>
      <path ref={pathRef} d={edge.d} fill="none" stroke="none" />
      <circle ref={dotRef} className="pulse" r={4.5} style={{ opacity: 0 }} />
    </g>
  );
}

interface Props {
  graph: GraphView;
  cited: string[];
  selectedTag: string | null;
}

/**
 * The evidence graph. Choreography (same as the Figma motion spec):
 *  seed anchors → draw edges → pop targets by geometry → pulses ride the cited
 *  edges → cited nodes fill violet with a quarter-turn, everything else dims.
 */
export function GraphStage({ graph, cited, selectedTag }: Props) {
  const layout = useMemo(() => layoutGraph(graph), [graph]);
  const { edgeIds, nodeIds } = useMemo(() => citedSets(layout.edges, cited), [layout, cited]);
  const [phase, setPhase] = useState<Phase>("build");
  const [run, setRun] = useState(0);

  const edgeStep = Math.min(0.07, 1.4 / Math.max(1, layout.edges.length));
  const edgeStart = 0.5;
  const traverseAt = edgeStart + layout.edges.length * edgeStep + 0.55;
  const matchAt = traverseAt + 0.85;

  useEffect(() => {
    if (prefersReducedMotion()) {
      setPhase("match");
      return;
    }
    setPhase("build");
    const a = setTimeout(() => setPhase("traverse"), traverseAt * 1000);
    const b = setTimeout(() => setPhase("match"), matchAt * 1000);
    return () => { clearTimeout(a); clearTimeout(b); };
  }, [layout, run, traverseAt, matchAt]);

  if (!layout.nodes.length) return null;

  const pulses = layout.edges.filter((e) => edgeIds.has(e.id)).slice(0, 16);
  const types = [...new Set(layout.nodes.map((n) => n.type))];
  const anchorDelay = (order: number) => 0.1 + order * 0.12;
  const targetDelay = (order: number) => edgeStart + order * edgeStep + 0.3;

  return (
    <div className="stage-wrap">
      <svg
        key={run}
        className={`stage${edgeIds.size ? " has-cited" : ""}`}
        data-phase={phase}
        viewBox={`0 0 ${STAGE.w} ${STAGE.h}`}
        role="img"
        aria-label={`Graph of ${layout.nodes.length} entities and ${layout.edges.length} relations used to answer`}
      >
        <defs>
          <filter id="glow" x="-200%" y="-200%" width="500%" height="500%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="match-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#6E5CF0" floodOpacity="0.8" />
          </filter>
        </defs>

        <circle className="orbit" cx={STAGE.w / 2} cy={STAGE.h / 2} r={150} />
        <circle className="orbit" cx={STAGE.w / 2} cy={STAGE.h / 2} r={205} style={{ animationDirection: "reverse" }} />

        {layout.edges.map((e) => (
          <path
            key={e.id}
            d={e.d}
            pathLength={1}
            className={`edge${edgeIds.has(e.id) ? " cited" : ""}${selectedTag && e.tag === selectedTag ? " selected" : ""}`}
            style={{ "--d": `${edgeStart + e.order * edgeStep}s` } as CSSProperties}
          >
            <title>{`${e.relation}${e.tag ? ` · ${e.tag}` : ""}`}</title>
          </path>
        ))}

        {phase !== "build" &&
          pulses.map((e, i) => (
            <Pulse key={`${run}-${e.id}`} edge={e} delayMs={i * 60} durationMs={700} />
          ))}

        {layout.nodes.map((n) => {
          const anchor = n.role === "anchor";
          const d = anchor ? anchorDelay(n.order) : targetDelay(n.order);
          const cited = nodeIds.has(n.id);
          const vars = {
            "--d": `${d}s`,
            "--ld": `${d + 0.15}s`,
            "--rot": `${shapeFor(n.type).entryRotation}deg`,
            "--md": `${(n.order % 12) * 0.05}s`,
          } as CSSProperties;
          return (
            <g key={n.id} className={`node${anchor ? " anchor" : ""}${cited ? " cited" : ""}`} style={vars}>
              <title>{`${n.label} · ${TYPE_LABEL[n.type] ?? n.type}`}</title>
              <g transform={`translate(${n.x.toFixed(1)} ${n.y.toFixed(1)})`}>
                <g className="node-body">
                  <g className="beat">
                    <Shape type={n.type} size={n.size} />
                    {!anchor && cited && (
                      <g filter="url(#match-glow)">
                        <MatchShape type={n.type} size={n.size} className="match" />
                      </g>
                    )}
                  </g>
                </g>
              </g>
              <text
                className={`label${anchor ? " anchor" : ""}`}
                x={n.labelPos.x}
                y={n.labelPos.y}
                textAnchor={n.labelPos.anchor}
              >
                {n.label.length > 30 ? `${n.label.slice(0, 29)}…` : n.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="stage-foot">
        <div className="legend">
          {types.map((t) => (
            <span key={t}><Glyph type={t} size={16} />{TYPE_LABEL[t] ?? t}</span>
          ))}
          {layout.hidden > 0 && <span>+{layout.hidden} more not drawn</span>}
        </div>
        <button className="replay" onClick={() => setRun((r) => r + 1)}>Replay</button>
      </div>
    </div>
  );
}
