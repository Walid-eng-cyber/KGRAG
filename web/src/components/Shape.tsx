import { shapeFor, type ShapeStyle } from "../lib/geometry";

interface Props {
  type: string;
  size: number; // rendered box size in px (shapes are authored in 48×48)
  style?: Partial<ShapeStyle>;
  className?: string;
}

const MATCH: ShapeStyle = { fill: "#6E5CF0", stroke: "#D9D2FF", strokeWidth: 2 };

/** One entity glyph, centred on (0, 0). */
export function Shape({ type, size, style, className }: Props) {
  const spec = shapeFor(type);
  const s = { ...spec.style, ...style };
  const k = size / 48;
  const common = {
    fill: s.fill,
    stroke: s.stroke === "none" ? undefined : s.stroke,
    strokeWidth: s.strokeWidth / k || undefined,
    strokeLinejoin: "round" as const,
  };
  return (
    <g className={className} transform={`scale(${k.toFixed(3)})`}>
      {spec.kind === "polygon" && <polygon points={spec.points} {...common} />}
      {spec.kind === "circle" && <circle r={spec.r} {...common} />}
      {spec.kind === "ring" && <circle r={spec.r} fill="none" stroke={s.stroke} strokeWidth={s.strokeWidth / k} />}
      {spec.kind === "roundrect" && <rect x={-17} y={-17} width={34} height={34} rx={8} {...common} />}
    </g>
  );
}

/** The violet "matched" version of a shape (ring/circle become a solid disc). */
export function MatchShape({ type, size, className }: { type: string; size: number; className?: string }) {
  const spec = shapeFor(type);
  const solid = spec.kind === "ring" ? "PERSON" : type;
  return <Shape type={solid} size={size} style={MATCH} className={className} />;
}

/** A standalone glyph for legends and the empty state. */
export function Glyph({ type, size = 28 }: { type: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="-26 -26 52 52" aria-hidden="true">
      <Shape type={type} size={48} />
    </svg>
  );
}
