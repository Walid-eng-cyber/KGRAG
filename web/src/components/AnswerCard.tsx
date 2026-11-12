import { useEffect, useMemo, useRef } from "react";
import { parseAnswer, tagKind, type Inline } from "../lib/answerText";
import type { AskResponse } from "../types";

interface Props {
  r: AskResponse;
  model: string;
  selectedTag: string | null;
  onSelectTag: (tag: string | null) => void;
}

function Cite({ tag, known, selected, onClick }: { tag: string; known: boolean; selected: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      className={`cite ${known ? tagKind(tag) : "invalid"}`}
      aria-pressed={selected}
      title={known ? `Show source ${tag}` : `${tag} does not resolve to a retrieved source`}
      onClick={onClick}
    >
      {tag}
    </button>
  );
}

export function AnswerCard({ r, model, selectedTag, onSelectTag }: Props) {
  const blocks = useMemo(() => parseAnswer(r.answer), [r.answer]);
  const known = useMemo(() => new Set(r.sources.map((s) => s.tag)), [r.sources]);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (!selectedTag) return;
    listRef.current?.querySelector(`[data-tag="${selectedTag}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedTag]);

  const toggle = (tag: string) => onSelectTag(selectedTag === tag ? null : tag);
  const renderInline = (inl: Inline, i: number) =>
    inl.kind === "text" ? <span key={i}>{inl.text}</span>
      : inl.kind === "bold" ? <b key={i}>{inl.text}</b>
      : <Cite key={i} tag={inl.tag} known={known.has(inl.tag)} selected={selectedTag === inl.tag} onClick={() => toggle(inl.tag)} />;

  const problems = r.uncited_claims.length;
  return (
    <section className="card" aria-label="Answer">
      <div className="card-h">
        <h2 className="card-t">Answer</h2>
        <span className="muted mono">{model} · {r.attempts} pass{r.attempts === 1 ? "" : "es"}</span>
      </div>
      <div className="card-b">
        {r.demo && <div className="demo-note">Demo mode: a sample answer, not a live run against your filings.</div>}
        <div className="answer">
          {blocks.map((b, i) =>
            b.kind === "p" ? <p key={i}>{b.inlines.map(renderInline)}</p>
              : <ul key={i}>{b.items.map((it, j) => <li key={j}>{it.map(renderInline)}</li>)}</ul>,
          )}
        </div>
        {r.sources.length > 0 && (
          r.citations_valid && problems === 0 ? (
            <div className="valid ok"><span className="check" aria-hidden="true">✓</span>Every citation resolves to a retrieved source. 0 uncited claims.</div>
          ) : (
            <div className="valid warn">
              <span className="check" aria-hidden="true">!</span>
              {!r.citations_valid ? "Some citations could not be validated and were removed. " : ""}
              {problems > 0 ? `${problems} claim${problems === 1 ? "" : "s"} without a citation.` : ""}
            </div>
          )
        )}
      </div>
      {r.sources.length > 0 && (
        <ul className="sources" ref={listRef} aria-label="Sources">
          {r.sources.map((s) => (
            <li key={s.tag} data-tag={s.tag} className={`src ${s.kind}${selectedTag === s.tag ? " selected" : ""}`}>
              <Cite tag={s.tag} known selected={selectedTag === s.tag} onClick={() => toggle(s.tag)} />
              <span className="kind">{s.kind}</span>
              <div className="src-txt">
                {s.kind === "passage" ? `“${s.text.slice(0, 260)}${s.text.length > 260 ? "…" : ""}”` : s.text}
                {s.cite && <div className="src-meta">{s.cite}</div>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
