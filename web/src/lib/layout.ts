import type { GraphEdge, GraphNode, GraphView } from "../types";

// Deterministic layouts for the evidence graph, in a fixed viewBox that the
// SVG scales responsively. No physics: the same answer always draws the same.

export const STAGE = { w: 640, h: 440 };

export interface PlacedNode extends GraphNode {
  x: number;
  y: number;
  size: number;
  order: number; // animation order
  labelPos: { x: number; y: number; anchor: "start" | "middle" | "end" };
}

export interface PlacedEdge extends GraphEdge {
  d: string;
  order: number;
}

export interface Layout {
  nodes: PlacedNode[];
  edges: PlacedEdge[];
  hidden: number; // targets not drawn (radial cap)
}

const ANCHOR = 54;
const TARGET = 34;

function spread(n: number, lo: number, hi: number, maxGap: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [(lo + hi) / 2];
  const gap = Math.min(maxGap, (hi - lo) / (n - 1));
  const start = (lo + hi) / 2 - (gap * (n - 1)) / 2;
  return Array.from({ length: n }, (_, i) => start + i * gap);
}

const curve = (x1: number, y1: number, x2: number, y2: number) => {
  const mx = (x1 + x2) / 2;
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${mx.toFixed(1)} ${y1.toFixed(1)} ${mx.toFixed(1)} ${y2.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
};
const line = (x1: number, y1: number, x2: number, y2: number) =>
  `M ${x1.toFixed(1)} ${y1.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)}`;

function edgeOrder(edges: GraphEdge[]): Map<string, number> {
  // a node animates in right after the first edge that reaches it
  const first = new Map<string, number>();
  edges.forEach((e, i) => {
    if (!first.has(e.target)) first.set(e.target, i);
    if (!first.has(e.source)) first.set(e.source, i);
  });
  return first;
}

function radial(g: GraphView, w: number, h: number, maxTargets: number): Layout {
  const anchor = g.nodes.find((n) => n.role === "anchor") ?? g.nodes[0];
  const targetIds: string[] = [];
  for (const e of g.edges) {
    const other = e.source === anchor.id ? e.target : e.source;
    if (other !== anchor.id && !targetIds.includes(other)) targetIds.push(other);
  }
  const shown = targetIds.slice(0, maxTargets);
  const cx = w / 2, cy = h / 2;
  const r = Math.min(w, h) / 2 - 64;
  const size = shown.length > 14 ? 26 : TARGET;
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const nodes: PlacedNode[] = [
    { ...anchor, x: cx, y: cy, size: ANCHOR, order: 0, labelPos: { x: cx, y: cy + ANCHOR / 2 + 18, anchor: "middle" } },
  ];
  shown.forEach((id, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / shown.length;
    const x = cx + r * Math.cos(a), y = cy + r * Math.sin(a);
    const c = Math.cos(a), s = Math.sin(a);
    const off = size / 2 + 10;
    const anchorSide = c > 0.3 ? "start" : c < -0.3 ? "end" : "middle";
    nodes.push({
      ...(byId.get(id) as GraphNode), x, y, size, order: i + 1,
      labelPos: { x: x + c * off, y: y + s * off + (anchorSide === "middle" ? (s > 0 ? 10 : -2) : 4), anchor: anchorSide },
    });
  });
  const keep = new Set(nodes.map((n) => n.id));
  const pos = new Map(nodes.map((n) => [n.id, n]));
  const edges: PlacedEdge[] = g.edges
    .filter((e) => keep.has(e.source) && keep.has(e.target))
    .map((e, i) => {
      const a = pos.get(e.source)!, b = pos.get(e.target)!;
      return { ...e, d: line(a.x, a.y, b.x, b.y), order: i };
    });
  return { nodes, edges, hidden: targetIds.length - shown.length };
}

function bipartite(g: GraphView, w: number, h: number): Layout {
  const anchors = g.nodes.filter((n) => n.role === "anchor");
  const anchorIdx = new Map(anchors.map((n, i) => [n.id, i]));
  const targets = g.nodes.filter((n) => n.role !== "anchor");
  // barycentre ordering keeps edges from crossing more than they must
  const bary = (id: string) => {
    const ix = g.edges.filter((e) => e.target === id).map((e) => anchorIdx.get(e.source) ?? 0);
    return ix.length ? ix.reduce((a, b) => a + b, 0) / ix.length : 0;
  };
  targets.sort((a, b) => bary(a.id) - bary(b.id));
  const ax = w * 0.2, tx = w * 0.6;
  const ay = spread(anchors.length, 50, h - 50, 130);
  const ty = spread(targets.length, 36, h - 36, 70);
  const order = edgeOrder(g.edges);
  const nodes: PlacedNode[] = [
    ...anchors.map((n, i) => ({ ...n, x: ax, y: ay[i], size: ANCHOR, order: i,
      labelPos: { x: ax, y: ay[i] + ANCHOR / 2 + 18, anchor: "middle" as const } })),
    ...targets.map((n, i) => ({ ...n, x: tx, y: ty[i], size: TARGET, order: anchors.length + (order.get(n.id) ?? i),
      labelPos: { x: tx + TARGET / 2 + 12, y: ty[i] + 4, anchor: "start" as const } })),
  ];
  const pos = new Map(nodes.map((n) => [n.id, n]));
  const edges = g.edges.map((e, i) => {
    const a = pos.get(e.source)!, b = pos.get(e.target)!;
    return { ...e, d: curve(a.x, a.y, b.x, b.y), order: i };
  });
  return { nodes, edges, hidden: 0 };
}

function layered(g: GraphView, w: number, h: number): Layout {
  // BFS layers from the first edge's source: A — intermediates — B
  const start = g.edges[0]?.source ?? g.nodes[0]?.id;
  const depth = new Map<string, number>([[start, 0]]);
  const queue = [start];
  while (queue.length) {
    const id = queue.shift()!;
    for (const e of g.edges) {
      const next = e.source === id ? e.target : e.target === id ? e.source : null;
      if (next && !depth.has(next)) {
        depth.set(next, depth.get(id)! + 1);
        queue.push(next);
      }
    }
  }
  g.nodes.forEach((n) => depth.has(n.id) || depth.set(n.id, 0));
  const maxDepth = Math.max(...depth.values(), 1);
  const layers: GraphNode[][] = Array.from({ length: maxDepth + 1 }, () => []);
  g.nodes.forEach((n) => layers[depth.get(n.id)!].push(n));
  const xs = spread(maxDepth + 1, 70, w - 70, 260);
  const nodes: PlacedNode[] = [];
  layers.forEach((layer, li) => {
    const ys = spread(layer.length, 60, h - 70, 150);
    layer.forEach((n, i) => {
      const end = li === 0 || li === maxDepth;
      const size = end ? ANCHOR : TARGET;
      nodes.push({ ...n, x: xs[li], y: ys[i], size, order: li * 2 + i,
        labelPos: { x: xs[li], y: ys[i] + size / 2 + 18, anchor: "middle" } });
    });
  });
  const pos = new Map(nodes.map((n) => [n.id, n]));
  const edges = g.edges.map((e, i) => {
    let a = pos.get(e.source)!, b = pos.get(e.target)!;
    if (a.x > b.x) [a, b] = [b, a]; // always draw left → right
    return { ...e, d: curve(a.x, a.y, b.x, b.y), order: i };
  });
  return { nodes, edges, hidden: 0 };
}

export function layoutGraph(g: GraphView, w = STAGE.w, h = STAGE.h, maxTargets = 24): Layout {
  if (!g.edges.length || !g.nodes.length) return { nodes: [], edges: [], hidden: 0 };
  if (g.layout === "path") return layered(g, w, h);
  if (g.layout === "bipartite") return bipartite(g, w, h);
  return radial(g, w, h, maxTargets);
}

/** Nodes touched by an edge whose citation tag the answer used. */
export function citedSets(edges: GraphEdge[], cited: string[]) {
  const tags = new Set(cited);
  const edgeIds = new Set<string>();
  const nodeIds = new Set<string>();
  for (const e of edges) {
    if (e.tag && tags.has(e.tag)) {
      edgeIds.add(e.id);
      nodeIds.add(e.source);
      nodeIds.add(e.target);
    }
  }
  return { edgeIds, nodeIds };
}
