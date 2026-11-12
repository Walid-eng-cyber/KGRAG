import type { EntityType } from "../types";

// Every entity type has its own shape (same language as the Figma node
// components). Shapes are drawn in a 48×48 box centred on (0, 0).

export interface ShapeStyle {
  fill: string;
  stroke: string;
  strokeWidth: number;
}

export interface ShapeSpec {
  kind: "polygon" | "circle" | "ring" | "roundrect";
  points?: string; // polygon
  r?: number; // circle / ring
  name: string;
  entryRotation: number; // degrees the node turns through when it appears (its symmetry)
  style: ShapeStyle;
}

function poly(n: number, r: number, rotDeg: number, cy = 0): string {
  return Array.from({ length: n }, (_, k) => {
    const a = ((rotDeg + (360 * k) / n) * Math.PI) / 180;
    return `${(r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)}`;
  }).join(" ");
}

function star(n: number, outer: number, inner: number): string {
  return Array.from({ length: 2 * n }, (_, k) => {
    const r = k % 2 ? inner : outer;
    const a = ((-90 + (180 * k) / n) * Math.PI) / 180;
    return `${(r * Math.cos(a)).toFixed(2)},${(2 + r * Math.sin(a)).toFixed(2)}`;
  }).join(" ");
}

export const SHAPES: Record<EntityType, ShapeSpec> = {
  COMPANY: { kind: "polygon", points: poly(6, 23, -90), name: "hexagon", entryRotation: -60,
    style: { fill: "#F4F2EC", stroke: "none", strokeWidth: 0 } },
  RISK: { kind: "polygon", points: "0,-20 20,0 0,20 -20,0", name: "diamond", entryRotation: -90,
    style: { fill: "#221F3D", stroke: "#A99CFF", strokeWidth: 2 } },
  PERSON: { kind: "circle", r: 17, name: "circle", entryRotation: 0,
    style: { fill: "#2B2216", stroke: "#F0B86E", strokeWidth: 2 } },
  PRODUCT: { kind: "roundrect", name: "rounded square", entryRotation: -90,
    style: { fill: "#12302C", stroke: "#7FD1C4", strokeWidth: 2 } },
  SEGMENT: { kind: "ring", r: 15, name: "ring", entryRotation: 0,
    style: { fill: "none", stroke: "#7FD1C4", strokeWidth: 7 } },
  AUDITOR: { kind: "polygon", points: poly(5, 22, -90, 1), name: "pentagon", entryRotation: -72,
    style: { fill: "#1A2340", stroke: "#9FB4FF", strokeWidth: 2 } },
  LOCATION: { kind: "polygon", points: poly(3, 22, -90, 4), name: "triangle", entryRotation: -120,
    style: { fill: "#3A1F1A", stroke: "#E58F7B", strokeWidth: 2 } },
  GOVERNMENT_AGENCY: { kind: "polygon", points: poly(8, 21, -67.5), name: "octagon", entryRotation: -45,
    style: { fill: "#262626", stroke: "#C9C6BD", strokeWidth: 2 } },
  STOCK_EXCHANGE: { kind: "polygon", points: star(5, 22, 9.5), name: "star", entryRotation: -144,
    style: { fill: "#F5D76E", stroke: "none", strokeWidth: 0 } },
};

export const TYPE_LABEL: Record<EntityType, string> = {
  COMPANY: "Company",
  RISK: "Risk",
  PERSON: "Person",
  PRODUCT: "Product",
  SEGMENT: "Segment",
  AUDITOR: "Auditor",
  LOCATION: "Location",
  GOVERNMENT_AGENCY: "Government agency",
  STOCK_EXCHANGE: "Stock exchange",
};

// Custom schemas bring types without a dedicated glyph: a ringed circle in a colour
// derived from the type name, so each type is consistent across the UI.
const PALETTE = ["#7FD1C4", "#A99CFF", "#F0B86E", "#E58F7B", "#9FB4FF", "#C9C6BD", "#B5D98A", "#E8A3C8"];
const ALIASES: Record<string, EntityType> = { ORGANIZATION: "COMPANY", INSTITUTION: "AUDITOR", RISK_FACTOR: "RISK" };

export function shapeFor(t: string): ShapeSpec {
  const known = SHAPES[(t as EntityType)] ?? SHAPES[ALIASES[t]];
  if (known) return known;
  let h = 0;
  for (const ch of t) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const color = PALETTE[h % PALETTE.length];
  return { kind: "circle", r: 16, name: "circle", entryRotation: 0, style: { fill: "#1d1e24", stroke: color, strokeWidth: 3 } };
}

export const typeLabel = (t: string): string =>
  TYPE_LABEL[(t as EntityType)] ?? t.toLowerCase().replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
