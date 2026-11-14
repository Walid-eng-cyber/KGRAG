import { useMemo } from "react";
import { parseAnswer, tagKind, type Inline } from "../lib/answerText";

interface Props {
  text: string;
  route: string | null;
  revising: boolean;
  model: string;
}

const inline = (inl: Inline, i: number) =>
  inl.kind === "text" ? <span key={i}>{inl.text}</span>
    : inl.kind === "bold" ? <b key={i}>{inl.text}</b>
    : <span key={i} className={`cite ${tagKind(inl.tag)}`}>{inl.tag}</span>;

/** The answer while the model is still writing it. Sources and the evidence panel appear when it is done. */
export function LiveAnswer({ text, route, revising, model }: Props) {
  const blocks = useMemo(() => parseAnswer(text), [text]);
  return (
    <section className="card" aria-label="Answer, being written" aria-busy="true">
      <div className="card-h">
        <h2 className="card-t">Answer</h2>
        <span className="muted mono">{route ? `${route} · ` : ""}{model} · writing…</span>
      </div>
      <div className="card-b">
        {revising && !text && <div className="live-note">Checking the citations: the model is writing the answer again.</div>}
        <div className="answer live">
          {blocks.map((b, i) =>
            b.kind === "p" ? <p key={i}>{b.inlines.map(inline)}</p>
              : <ul key={i}>{b.items.map((it, j) => <li key={j}>{it.map(inline)}</li>)}</ul>,
          )}
          <span className="caret" aria-hidden="true" />
        </div>
      </div>
    </section>
  );
}
