import { useRef, useState, type FormEvent } from "react";
import { api, ApiError } from "../api";
import { AnswerCard } from "../components/AnswerCard";
import { EvidencePanel } from "../components/EvidencePanel";
import { Pending } from "../components/Pending";
import { RoutingTrace } from "../components/RoutingTrace";
import { Glyph } from "../components/Shape";
import { typeLabel } from "../lib/geometry";
import type { AskResponse, Corpus, CorpusProfile, Meta } from "../types";

interface HeroProps { corpus: Corpus | null; profile: CorpusProfile | null; demo: boolean; onNewCorpus: () => void }

function heroCopy(corpus: Corpus | null, demo: boolean): { title: string; body: string } {
  if (demo || !corpus) {
    return { title: "Answers you can trace back to the filing.",
      body: "Ask about Apple, Microsoft or NVIDIA’s 10-K filings. A small router sends each question to the knowledge graph, to vector search, or to both, and the answer only cites evidence it actually retrieved." };
  }
  if (corpus.kind === "sec_insiders") {
    return { title: "Who bought, who sold, who sits on which board.",
      body: "Insider transactions from SEC Forms 3/4/5. Ask about a company’s insiders, recent trades, net buying or selling, or which companies share a director; every answer cites the SEC filing." };
  }
  const body = corpus.description?.trim()
    || "Your private documents, turned into a knowledge graph and a search index. Answers cite the passages and facts they come from.";
  return { title: corpus.name, body };
}

function EmptyState({ corpus, profile, demo, onNewCorpus }: HeroProps) {
  const { title, body } = heroCopy(corpus, demo);
  const types = demo || !corpus ? ["COMPANY", "RISK", "PERSON", "PRODUCT", "SEGMENT", "AUDITOR", "LOCATION", "GOVERNMENT_AGENCY", "STOCK_EXCHANGE"]
    : corpus.schema.entities;
  const counts = new Map((profile?.profile?.types ?? []).map((t) => [t.type, t.n]));
  const noDocs = corpus?.kind === "documents" && corpus.ready === 0;
  return (
    <section className="horizon" aria-label="Introduction">
      <svg className="horizon-arc" viewBox="0 0 1200 420" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
        <defs>
          <filter id="arc-wide" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="16" /></filter>
          <filter id="arc-soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="4" /></filter>
          <filter id="arc-fine" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="1.2" /></filter>
        </defs>
        <g className="arc-glow">
          <ellipse cx="600" cy="1486" rx="1560" ry="1250" fill="none" stroke="#5b4bd6" strokeWidth="30" filter="url(#arc-wide)" />
          <ellipse cx="600" cy="1492" rx="1560" ry="1250" fill="none" stroke="#8b7cff" strokeWidth="10" filter="url(#arc-soft)" />
        </g>
        <ellipse cx="600" cy="1500" rx="1560" ry="1250" fill="#0d0d0c" stroke="#f4f2ec" strokeWidth="5" filter="url(#arc-fine)" />
        <ellipse cx="600" cy="1505" rx="1560" ry="1250" fill="#0d0d0c" stroke="#c8643c" strokeOpacity="0.7" strokeWidth="2" filter="url(#arc-fine)" />
      </svg>
      <div className="horizon-copy">
        {corpus && !demo && <div className="horizon-kicker">{corpus.kind === "sec_insiders" ? "SEC insider data" : "Private corpus"}</div>}
        <h2>{title}</h2>
        <p>{body}</p>
        {!demo && !corpus && (
          <button className="btn light" onClick={onNewCorpus}>Create your first corpus</button>
        )}
        {noDocs && <p className="horizon-hint">No documents ingested yet: add files in the sidebar and press Ingest.</p>}
      </div>
      <div className="horizon-shapes" aria-label="Entity types in this corpus">
        {types.map((t) => (
          <span key={t}><Glyph type={t} size={18} />{typeLabel(t)}{counts.has(t) ? <em> {counts.get(t)}</em> : null}</span>
        ))}
      </div>
    </section>
  );
}

interface Props {
  meta: Meta | null;
  corpus: Corpus | null;
  profile: CorpusProfile | null;
  onNewCorpus: () => void;
}

export function AskPage({ meta, corpus, profile, onNewCorpus }: Props) {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  async function ask(q: string) {
    const text = q.trim();
    if (text.length < 3 || loading) return;
    setQuestion(text);
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setLoading(true);
    setError(null);
    setSelectedTag(null);
    try {
      setResult(await api.ask(text, corpus?.id ?? null, ctrl.signal));
    } catch (e) {
      if (ctrl.signal.aborted) return;
      setError(e instanceof ApiError ? e.message : String(e));
      setResult(null);
    } finally {
      if (abort.current === ctrl) setLoading(false);
    }
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void ask(question);
  };

  const examples = meta?.demo ? meta.examples : profile?.examples ?? [];
  const placeholder = corpus?.kind === "sec_insiders" ? "e.g. Which insiders sold NVIDIA shares?"
    : corpus ? `Ask about ${corpus.name}…` : "e.g. Which companies face export control risk?";
  const canAsk = !!meta?.demo || !!corpus;
  return (
    <>
      <form className="card ask" onSubmit={onSubmit} role="search">
        <label htmlFor="q" className="sr-only" style={{ position: "absolute", left: -9999 }}>Question</label>
        <input
          id="q"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={placeholder}
          disabled={!canAsk}
          maxLength={500}
          autoComplete="off"
          autoFocus
        />
        {loading && <button type="button" className="btn ghost" onClick={() => { abort.current?.abort(); setLoading(false); }}>Cancel</button>}
        <button className="btn" type="submit" disabled={!canAsk || loading || question.trim().length < 3}>Ask</button>
      </form>
      <div className="chips" aria-label="Example questions">
        {examples.map((ex) => (
          <button key={ex.question} className="chip" type="button" onClick={() => void ask(ex.question)} disabled={loading}>
            <i className={ex.path}>{ex.path === "HYBRID" ? "Hybrid" : ex.path === "GRAPH" ? "Graph" : "Vector"}</i>
            {ex.question}
          </button>
        ))}
      </div>

      {loading && <Pending demo={!!meta?.demo} />}
      {error && !loading && <div className="card error" role="alert">{error}</div>}
      {!loading && !error && !result && <EmptyState corpus={corpus} profile={profile} demo={!!meta?.demo} onNewCorpus={onNewCorpus} />}
      {result && !loading && (
        <>
          <RoutingTrace r={result} />
          <div className="grid">
            <AnswerCard r={result} model={meta?.models.answer ?? "answer model"} selectedTag={selectedTag} onSelectTag={setSelectedTag} />
            <EvidencePanel r={result} selectedTag={selectedTag} />
          </div>
        </>
      )}
    </>
  );
}
