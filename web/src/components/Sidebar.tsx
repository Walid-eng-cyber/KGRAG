import type { Health, Meta, User } from "../types";

export type Page = "ask" | "benchmark" | "admin";

const LABEL: Record<Page, string> = { ask: "Ask", benchmark: "Benchmark", admin: "Accounts" };

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
}

export function Sidebar({ page, onNavigate, meta, health, user, onSignOut, onToggleUpdates }: Props) {
  const pages: Page[] = user?.role === "admin" ? ["ask", "benchmark", "admin"] : ["ask", "benchmark"];
  return (
    <aside className="side">
      <div className="brand">
        <div className="brand-mark"><Mark /></div>
        <div><div className="brand-name">KGRAG</div><div className="brand-sub">Knowledge-graph RAG · local</div></div>
      </div>
      <nav className="nav" aria-label="Pages">
        {pages.map((p) => (
          <button key={p} aria-current={page === p ? "page" : undefined} onClick={() => onNavigate(p)}>
            <span className="nav-dot" />{LABEL[p]}
          </button>
        ))}
      </nav>
      <div className="corpus">
        <div className="side-label">Corpus</div>
        {(meta?.tickers ?? ["AAPL", "MSFT", "NVDA"]).map((t) => (
          <div className="side-row" key={t}><span className="ticker">{t}</span><span className="muted">10-K</span></div>
        ))}
      </div>
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
