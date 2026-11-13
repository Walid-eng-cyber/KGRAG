import type { Corpus, Health, Meta, User } from "../types";
import { DocumentsPanel } from "./DocumentsPanel";

export type Page = "ask" | "benchmark" | "admin" | "usage";

const LABEL: Record<Page, string> = { ask: "Ask", benchmark: "Benchmark", admin: "Accounts", usage: "Clients & usage" };

const SERVICES: { key: keyof Health["services"]; name: string }[] = [
  { key: "neo4j", name: "Neo4j" },
  { key: "postgres", name: "pgvector" },
  { key: "ollama", name: "Ollama" },
];

export function Mark() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path d="M4 4 L14 5 M4 4 L9 14 M14 5 L9 14" stroke="#fff" strokeWidth="1.2" strokeOpacity="0.6" />
      <circle cx="4" cy="4" r="2.4" fill="#A99CFF" /><circle cx="14" cy="5" r="2.4" fill="#7FD1C4" /><circle cx="9" cy="14" r="2.4" fill="#fff" />
    </svg>
  );
}

interface Props {
  page: Page;
  onNavigate: (p: Page) => void;
  meta: Meta | null;
  health: Health | null;
  user: User | null;
  onSignOut: () => void;
  onToggleUpdates: (on: boolean) => void;
  corpora: Corpus[] | null;
  corpus: Corpus | null;
  onSelectCorpus: (id: string) => void;
  onNewCorpus: () => void;
  onEditCorpus: () => void;
  onCorpusChanged: () => void;
}

export function Sidebar({ page, onNavigate, meta, health, user, onSignOut, onToggleUpdates, corpora, corpus,
  onSelectCorpus, onNewCorpus, onEditCorpus, onCorpusChanged }: Props) {
  // The benchmark compares GraphRAG with vector RAG on SEC filings: only shown for the SEC corpus.
  // Clients & usage is the operator's view: admins, or anyone on a local setup with sign-in off.
  const operator = user?.role === "admin" || (meta != null && !meta.auth.enabled && !meta.demo && !meta.acting_as);
  const pages: Page[] = ["ask", ...(corpus?.kind === "sec_insiders" ? ["benchmark" as Page] : []),
    ...(user?.role === "admin" ? ["admin" as Page] : []), ...(operator ? ["usage" as Page] : [])];
  return (
    <aside className="side">
      <div className="brand">
        <div className="brand-mark"><Mark /></div>
        <div><div className="brand-name">KGRAG</div><div className="brand-sub">Knowledge-graph RAG · local</div></div>
      </div>
      <section className="corpus-switch" aria-label="Corpus">
        <div className="side-label docs-head"><span>Corpus</span>
          <button className="linkish small" onClick={onNewCorpus}>+ New</button>
        </div>
        {corpora && corpora.length > 0 ? (
          <div className="corpus-select">
            <select value={corpus?.id ?? ""} onChange={(e) => onSelectCorpus(e.target.value)} aria-label="Select corpus">
              {corpora.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
            <button className="corpus-gear" aria-label="Corpus settings" title="Corpus settings" onClick={onEditCorpus}>⚙</button>
          </div>
        ) : (
          <p className="docs-empty">{corpora ? "No corpus yet." : "Loading…"}</p>
        )}
        {corpus && (
          <p className="corpus-meta">
            {corpus.kind === "sec_insiders" ? "SEC Forms 3/4/5 data set" : `${corpus.ready}/${corpus.documents} documents ready`}
            {" · "}{corpus.schema.entities.length} entity types
          </p>
        )}
      </section>
      <nav className="nav" aria-label="Pages">
        {pages.map((p) => (
          <button key={p} aria-current={page === p ? "page" : undefined} onClick={() => onNavigate(p)}>
            <span className="nav-dot" />{LABEL[p]}
          </button>
        ))}
      </nav>
      {meta?.demo && (
        <div className="corpus">
          <div className="side-label">Corpus</div>
          {meta.tickers.map((t) => (
            <div className="side-row" key={t}><span className="ticker">{t}</span><span className="muted">10-K</span></div>
          ))}
        </div>
      )}
      {corpus?.kind === "documents" && <DocumentsPanel key={corpus.id} corpusId={corpus.id} onChanged={onCorpusChanged} />}
      {corpus?.kind === "sec_insiders" && (
        <div className="corpus">
          <div className="side-label">Data set</div>
          <div className="side-row"><span className="ticker">SEC Form 4</span><span className="muted">insider trades</span></div>
          <p className="docs-empty">Loaded with <span className="mono">python -m src.form4</span>. Read-only in the app.</p>
        </div>
      )}
      <div className="spacer" />
      <div className="status" aria-label="Local stack status">
        <div className="side-label" style={{ padding: "0 0 4px" }}>Local stack</div>
        {SERVICES.map(({ key, name }) => {
          const s = health?.services[key];
          const cls = !health ? "" : health.demo ? "demo" : s?.ok ? "ok" : "down";
          return (
            <div className="status-row" key={key} title={s?.detail}>
              <span><span className={`dot ${cls}`} />{name}</span>
              <span className="model">{!health ? "…" : health.demo ? "demo" : s?.ok ? "up" : "down"}</span>
            </div>
          );
        })}
        {meta && (
          <>
            <div className="model">router {meta.models.router}</div>
            <div className="model">answer {meta.models.answer} · embed {meta.models.embed}</div>
          </>
        )}
      </div>
      {user && (
        <div className="account">
          <div className="account-row">
            <div className="account-who" title={user.email}>
              <span className="account-name">{user.name}</span>
              <span className="model">{user.email}</span>
            </div>
            <button className="btn ghost" onClick={onSignOut}>Sign out</button>
          </div>
          <label className="updates-toggle">
            <input type="checkbox" checked={user.updates_opt_in} onChange={(e) => onToggleUpdates(e.target.checked)} />
            Email me about updates
          </label>
        </div>
      )}
    </aside>
  );
}
