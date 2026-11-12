// Turn the model's answer into blocks of inline tokens: plain text, **bold**,
// and [G#]/[P#] citations rendered as chips. Deliberately tiny — the answer
// prompt only produces paragraphs, bullets, bold and citations.

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "cite"; tag: string };

export type Block = { kind: "p"; inlines: Inline[] } | { kind: "ul"; items: Inline[][] };

const TOKEN = /\*\*(.+?)\*\*|\[((?:G|P)\d+)\]/g;

export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ kind: "text", text: text.slice(last, at) });
    if (m[1] !== undefined) out.push({ kind: "bold", text: m[1] });
    else out.push({ kind: "cite", tag: m[2] });
    last = at + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

const BULLET = /^\s*(?:[-*•]|\d+[.)])\s+/;

export function parseAnswer(text: string): Block[] {
  const blocks: Block[] = [];
  for (const chunk of text.trim().split(/\n\s*\n/)) {
    const lines = chunk.split("\n").map((l) => l.trimEnd()).filter(Boolean);
    if (!lines.length) continue;
    if (lines.every((l) => BULLET.test(l))) {
      blocks.push({ kind: "ul", items: lines.map((l) => parseInline(l.replace(BULLET, ""))) });
    } else {
      blocks.push({ kind: "p", inlines: parseInline(lines.join(" ")) });
    }
  }
  return blocks;
}

export const tagKind = (tag: string): "graph" | "passage" => (tag.startsWith("G") ? "graph" : "passage");
