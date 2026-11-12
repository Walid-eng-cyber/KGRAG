import { describe, expect, it } from "vitest";
import { parseAnswer, parseInline } from "./answerText";
import { citedSets, layoutGraph, STAGE } from "./layout";
import type { GraphView } from "../types";

describe("answer text", () => {
  it("splits citations and bold into tokens", () => {
    expect(parseInline("Apple has **9** subsidiaries [G1].")).toEqual([
      { kind: "text", text: "Apple has " },
      { kind: "bold", text: "9" },
      { kind: "text", text: " subsidiaries " },
      { kind: "cite", tag: "G1" },
      { kind: "text", text: "." },
    ]);
  });

  it("recognises paragraphs and bullet lists", () => {
    const blocks = parseAnswer("Intro [P1].\n\n- one [G1]\n- two [G2]");
    expect(blocks.map((b) => b.kind)).toEqual(["p", "ul"]);
    expect(blocks[1].kind === "ul" && blocks[1].items.length).toBe(2);
  });
});

const g = (over: Partial<GraphView>): GraphView => ({
  query_type: null, cypher: null, layout: "radial", nodes: [], edges: [], metrics: [], ...over,
});

const inside = (l: ReturnType<typeof layoutGraph>) =>
  l.nodes.every((n) => n.x >= 0 && n.x <= STAGE.w && n.y >= 0 && n.y <= STAGE.h);

describe("layout", () => {
  const star = g({
    layout: "radial",
    nodes: [{ id: "apple", label: "Apple", type: "COMPANY", role: "anchor" },
      ...Array.from({ length: 30 }, (_, i) => ({ id: `r${i}`, label: `Risk ${i}`, type: "RISK" as const, role: "target" as const }))],
    edges: Array.from({ length: 30 }, (_, i) => ({ id: `e${i}`, source: "apple", target: `r${i}`, relation: "FACES_RISK", tag: "G1" })),
  });

  it("radial: anchor in the centre, targets capped and inside the stage", () => {
    const l = layoutGraph(star);
    expect(l.nodes[0]).toMatchObject({ id: "apple", x: STAGE.w / 2, y: STAGE.h / 2 });
    expect(l.nodes.length).toBe(25);
    expect(l.hidden).toBe(6);
    expect(inside(l)).toBe(true);
  });

  it("bipartite: anchors left of targets", () => {
    const l = layoutGraph(g({
      layout: "bipartite",
      nodes: [{ id: "a", label: "A", type: "COMPANY", role: "anchor" }, { id: "b", label: "B", type: "COMPANY", role: "anchor" },
        { id: "x", label: "X", type: "RISK", role: "target" }],
      edges: [{ id: "e0", source: "a", target: "x", relation: "FACES_RISK", tag: "G1" },
        { id: "e1", source: "b", target: "x", relation: "FACES_RISK", tag: "G1" }],
    }));
    const x = (id: string) => l.nodes.find((n) => n.id === id)!.x;
    expect(x("a")).toBeLessThan(x("x"));
    expect(inside(l)).toBe(true);
  });

  it("path: nodes are layered left to right by hop", () => {
    const l = layoutGraph(g({
      layout: "path",
      nodes: [{ id: "a", label: "A", type: "COMPANY", role: "anchor" }, { id: "m", label: "M", type: "RISK", role: "target" },
        { id: "b", label: "B", type: "COMPANY", role: "anchor" }],
      edges: [{ id: "e0", source: "a", target: "m", relation: "FACES_RISK", tag: "G1" },
        { id: "e1", source: "b", target: "m", relation: "FACES_RISK", tag: "G1" }],
    }));
    const x = (id: string) => l.nodes.find((n) => n.id === id)!.x;
    expect(x("a")).toBeLessThan(x("m"));
    expect(x("m")).toBeLessThan(x("b"));
  });

  it("citedSets picks edges and nodes for the cited tags", () => {
    const { edgeIds, nodeIds } = citedSets(star.edges.slice(0, 2), ["G1"]);
    expect(edgeIds.size).toBe(2);
    expect(nodeIds.has("apple")).toBe(true);
    expect(citedSets(star.edges, ["P1"]).edgeIds.size).toBe(0);
  });
});
