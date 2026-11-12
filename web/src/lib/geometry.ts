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

export const shapeFor = (t: string): ShapeSpec => SHAPES[(t as EntityType)] ?? SHAPES.COMPANY;
