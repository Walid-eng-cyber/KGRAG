import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, ApiError } from "../api";
import type { AccountOverview, Billing, Credit, DayUsage, StageUsage, UsageReport, User } from "../types";

// ---------- formatting ----------

const EUR = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR" });
const EUR_FINE = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", minimumFractionDigits: 4,
  maximumFractionDigits: 4 });
const NUM = new Intl.NumberFormat("en-IE");
const COMPACT = new Intl.NumberFormat("en-IE", { notation: "compact", maximumFractionDigits: 1 });

/** Euros: cents normally, four decimals for sub-euro amounts so small costs don't read as €0.00. */
export const eur = (v: number | null | undefined): string =>
  v == null ? "—" : v > 0 && v < 1 ? EUR_FINE.format(v) : EUR.format(v);
const secs = (ms: number | null | undefined) => (ms == null ? "—" : `${(ms / 1000).toFixed(1)} s`);
const pct = (v: number | null | undefined, digits = 1) => (v == null ? "—" : `${v.toFixed(digits)}%`);
const day = (iso: string) => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
const ago = (iso: string | null) => (iso ? new Date(iso).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" }) : "—");

const STATUS_LABEL: Record<Credit["status"], string> = {
  ok: "Within credit", at_risk: "At risk", over: "Credit used up", no_credit: "No credit set",
};
const STAGE_LABEL: Record<string, string> = {
  router: "Routing", graph_plan: "Graph planning", answer: "Answer", retry: "Citation retry",
  extract: "Ingest extraction", vector: "Vector search", graph: "Graph query", eval: "Evaluation", other: "Other",
};
const PERIODS = [7, 30, 90] as const;

const errText = (e: unknown) => (e instanceof ApiError ? e.message : String(e));

// ---------- small pieces ----------

function StatusPill({ status }: { status: Credit["status"] }) {
  const icon = status === "ok" ? "●" : status === "at_risk" ? "▲" : status === "over" ? "■" : "○";
  return <span className={`credit-pill ${status}`}><span aria-hidden="true">{icon}</span> {STATUS_LABEL[status]}</span>;
}

/** Credit meter: the fill carries severity, the track is a lighter step of the same hue. */
function CreditMeter({ credit, label }: { credit: Credit; label?: string }) {
  const used = Math.min(100, credit.used_pct ?? 0);
  return (
    <div className="credit-meter">
      <div className={`meter ${credit.status}`} role="meter" aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={Math.round(used * 10) / 10} aria-label={label ?? "Monthly credit used"}>
        {/* At least a sliver once anything is spent, so a small spend is visible. */}
        <span style={{ width: `${credit.spent_eur > 0 ? Math.max(used, 0.6) : 0}%` }} />
      </div>
      <div className="meter-legend">
        <span><b>{eur(credit.spent_eur)}</b> of {eur(credit.monthly_credit_eur)}</span>
        <span className="muted">{pct(credit.used_pct, credit.used_pct != null && credit.used_pct < 10 ? 2 : 1)} used</span>
      </div>
    </div>
  );
}

function Tile({ k, v, d }: { k: string; v: string; d?: ReactNode }) {
  return (
    <div className="card kpi">
      <div className="kpi-k">{k}</div>
      <div className="kpi-v usage-v">{v}</div>
      {d && <div className="kpi-d">{d}</div>}
    </div>
  );
}

// ---------- daily spend column chart (single series: no legend, the title names it) ----------

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return ([1, 2, 2.5, 5, 10].find((m) => m * p >= v) ?? 10) * p;
}

function topRounded(x: number, y: number, w: number, h: number): string {
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

function DailySpend({ daily }: { daily: DayUsage[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 720, H = 220, L = 52, R = 8, T = 12, B = 26;
  const max = niceMax(Math.max(...daily.map((d) => d.cost_eur)));
  const slot = (W - L - R) / daily.length;
  const bw = Math.min(24, Math.max(2, slot - 2));  // capped thickness; the rest of the slot is air (>= 2px gap)
  const y = (v: number) => T + (H - T - B) * (1 - v / max);
  const ticks = [0, max / 2, max];
  const every = Math.ceil(daily.length / 8);
  const empty = daily.every((d) => d.cost_eur === 0);
  const h = hover != null ? daily[hover] : null;

  return (
    <div className="chart-wrap" onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img"
        aria-label={`Daily spend, ${daily.length} days. Highest ${eur(Math.max(...daily.map((d) => d.cost_eur)))}.`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid" />
            <text x={L - 8} y={y(t) + 4} textAnchor="end" className="axis">€{Number(t.toFixed(2))}</text>
          </g>
        ))}
        {daily.map((d, i) => {
          const x = L + i * slot + (slot - bw) / 2;
          const bh = (H - T - B) * (d.cost_eur / max);
          return (
            <g key={d.date}>
              {bh > 0 && <path d={topRounded(x, y(d.cost_eur), bw, bh)} className={`bar${hover === i ? " on" : ""}`} />}
              {i % every === 0 && <text x={x + bw / 2} y={H - 8} textAnchor="middle" className="axis">{day(d.date)}</text>}
              {/* Hit target: the whole column, wider than the bar. */}
              <rect x={L + i * slot} y={T} width={slot} height={H - T - B} fill="transparent"
                onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} tabIndex={-1} />
            </g>
          );
        })}
      </svg>
      {empty && <div className="chart-empty">No usage in this period yet. Spending starts at €0 and fills in as people ask questions.</div>}
      {h && hover != null && (
        <div className="chart-tip" style={{ left: `${((L + hover * slot + slot / 2) / W) * 100}%` }}>
          <b>{day(h.date)}</b>
          <span>{eur(h.cost_eur)}</span>
          <span className="muted">{NUM.format(h.questions)} questions · {NUM.format(h.requests)} requests</span>
          {h.ingests > 0 && <span className="muted">{NUM.format(h.ingests)} documents ingested</span>}
        </div>
      )}
    </div>
  );
}

// ---------- cost by stage: horizontal bars, value at the tip ----------

function StageBars({ stages }: { stages: StageUsage[] }) {
  const billed = stages.filter((s) => s.calls > 0);
  if (billed.length === 0) return <div className="empty">No model calls yet.</div>;
  const max = Math.max(...billed.map((s) => s.cost_eur), 1e-9);
  return (
    <ul className="hbars">
      {billed.map((s) => (
        <li key={s.stage} title={`${NUM.format(s.calls)} calls · ${COMPACT.format(s.tokens_in)} in / ${COMPACT.format(s.tokens_out)} out tokens · p95 ${secs(s.p95_ms)}`}>
          <span className="hbar-k">{STAGE_LABEL[s.stage] ?? s.stage}</span>
          <span className="hbar-track">
            <span className="hbar" style={{ width: `${Math.max(1, (100 * s.cost_eur) / max)}%` }} />
          </span>
          <span className="hbar-v">{eur(s.cost_eur)} <span className="muted">{s.cost_share_pct.toFixed(0)}%</span></span>
        </li>
      ))}
    </ul>
  );
}

// ---------- one account ----------

function AccountDetail({ id, billing, onBack, onChanged }: { id: string; billing: Billing | null; onBack: () => void;
  onChanged: () => void }) {
  const [days, setDays] = useState<number>(30);
  const [rep, setRep] = useState<UsageReport | null>(null);
  const [members, setMembers] = useState<User[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [creditDraft, setCreditDraft] = useState("");
  const [newMember, setNewMember] = useState("");
  const [showTable, setShowTable] = useState(false);

  const load = useCallback(() => {
    setError(null);
    api.accountUsage(id, days).then(setRep).catch((e) => setError(errText(e)));
    api.accountMembers(id).then((r) => setMembers(r.members)).catch(() => setMembers([]));
  }, [id, days]);
  useEffect(load, [load]);

  const saveCredit = async () => {
    const v = Number(creditDraft.replace(/[^\d.]/g, ""));
    if (!Number.isFinite(v) || v < 0) return setError("Enter the monthly credit in euros, e.g. 2500.");
    try {
      await api.updateClientAccount(id, { monthly_credit_eur: v });
      setEditing(false);
      load();
      onChanged();
    } catch (e) { setError(errText(e)); }
  };
  const addMember = async () => {
    if (!newMember.trim()) return;
    try {
      await api.addAccountMember(id, newMember.trim());
      setNewMember("");
      load();
      onChanged();
    } catch (e) { setError(errText(e)); }
  };
  const removeMember = async (email: string) => {
    try {
      await api.removeAccountMember(id, email);
      load();
      onChanged();
    } catch (e) { setError(errText(e)); }
  };

  if (!rep) return error ? <div className="card error" role="alert">{error}</div> : <div className="empty">Loading…</div>;
  const { account: a, credit: c, totals: t, latency, quality: q } = rep;
  const b = billing ?? rep.billing;

  return (
    <div className="usage">
      <div className="usage-head">
        <div>
          <button className="linkish small" onClick={onBack}>← All clients</button>
          <h2 className="usage-title">{a.name}</h2>
          <div className="muted">{a.industry || "No industry"} · {a.members} {a.members === 1 ? "user" : "users"} · {a.id}</div>
        </div>
        <div className="tabs" role="tablist" aria-label="Period">
          {PERIODS.map((p) => (
            <button key={p} role="tab" aria-selected={days === p} onClick={() => setDays(p)}>{p} days</button>
          ))}
        </div>
      </div>
      {error && <div className="card error" role="alert">{error}</div>}

      <section className="card credit-card" aria-label="Monthly credit">
        <div className="credit-top">
          <div>
            <div className="kpi-k">Usage credit · {new Date(`${c.month}-01T00:00:00Z`).toLocaleDateString("en-GB", { month: "long", year: "numeric", timeZone: "UTC" })}</div>
            {editing ? (
              <div className="credit-edit">
                <span>€</span>
                <input value={creditDraft} onChange={(e) => setCreditDraft(e.target.value)} inputMode="decimal"
                  aria-label="Monthly credit in euros" autoFocus onKeyDown={(e) => e.key === "Enter" && saveCredit()} />
                <span className="muted">per month</span>
                <button className="btn small" onClick={saveCredit}>Save</button>
                <button className="btn ghost small" onClick={() => setEditing(false)}>Cancel</button>
              </div>
            ) : (
              <div className="credit-amount">
                {eur(c.remaining_eur)} <span className="muted">left of {eur(c.monthly_credit_eur)} per month</span>
                <button className="linkish small" onClick={() => { setCreditDraft(String(c.monthly_credit_eur)); setEditing(true); }}>Change</button>
              </div>
            )}
          </div>
          <StatusPill status={c.status} />
        </div>
        <CreditMeter credit={c} />
        <div className="muted">
          Projected month-end {eur(c.projected_eur)} at the current pace · resets {new Date(c.resets_at).toLocaleDateString("en-GB", { day: "numeric", month: "long", timeZone: "UTC" })}
        </div>
      </section>

      <div className="kpis">
        <Tile k={`Spend, last ${days} days`} v={eur(t.cost_eur)}
          d={<>questions {eur(t.ask_cost_eur)} · ingest {eur(t.ingest_cost_eur)}</>} />
        <Tile k="Questions" v={NUM.format(t.questions)}
          d={<>{eur(t.cost_per_question_eur)} each · {t.active_users} active {t.active_users === 1 ? "user" : "users"}</>} />
        <Tile k="Model requests" v={NUM.format(t.requests)}
          d={<>{eur(t.cost_per_request_eur)} each · {COMPACT.format(t.tokens_in)} in / {COMPACT.format(t.tokens_out)} out tokens</>} />
        <Tile k="Answer time (p95)" v={secs(latency.p95_ms)} d={<>median {secs(latency.p50_ms)}</>} />
      </div>

      <section className="card" aria-label="Daily spend">
        <div className="card-h">
          <h3 className="card-t">Daily spend</h3>
          <button className="linkish small" onClick={() => setShowTable((v) => !v)}>{showTable ? "Hide table" : "Show as table"}</button>
        </div>
        <div className="card-b">
          <DailySpend daily={rep.daily} />
          {showTable && (
            <table className="accounts compact">
              <thead><tr><th scope="col">Day</th><th scope="col">Spend</th><th scope="col">Questions</th><th scope="col">Requests</th><th scope="col">Ingests</th><th scope="col">Users</th></tr></thead>
              <tbody>
                {[...rep.daily].reverse().map((d) => (
                  <tr key={d.date}><td>{day(d.date)}</td><td>{eur(d.cost_eur)}</td><td>{d.questions}</td><td>{d.requests}</td><td>{d.ingests}</td><td>{d.active_users}</td></tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      <div className="usage-two">
        <section className="card" aria-label="Cost by stage">
          <div className="card-h"><h3 className="card-t">Where the money goes</h3></div>
          <div className="card-b"><StageBars stages={rep.stages} /></div>
        </section>
        <section className="card" aria-label="Quality">
          <div className="card-h"><h3 className="card-t">Answer quality</h3></div>
          <div className="card-b">
            <dl className="facts">
              <dt>Answered</dt><dd>{pct(q.answered_pct)}</dd>
              <dt>No evidence found</dt><dd>{pct(q.no_evidence_pct)}</dd>
              <dt>Errors</dt><dd>{pct(q.error_pct)}</dd>
              <dt>Citation retries</dt><dd>{pct(q.retry_pct)}</dd>
              <dt>Low-confidence routing</dt><dd>{pct(q.low_confidence_pct)}</dd>
              <dt>Failed ingests</dt><dd>{pct(q.ingest_failed_pct)}</dd>
              <dt>Paths</dt>
              <dd>{Object.keys(rep.paths).length ? Object.entries(rep.paths).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k.toLowerCase()} ${v}`).join(" · ") : "—"}</dd>
            </dl>
          </div>
        </section>
      </div>

      <section className="card" aria-label="Usage by user">
        <div className="card-h"><h3 className="card-t">By user</h3><span className="muted">last {days} days</span></div>
        {rep.users.length === 0 ? <div className="empty">Nobody has used KGRAG in this period yet.</div> : (
          <div className="table-scroll"><table className="accounts">
            <thead>
              <tr><th scope="col">User</th><th scope="col">Questions</th><th scope="col">Requests</th><th scope="col">Spend</th>
                <th scope="col">Share</th><th scope="col">p95</th><th scope="col">Errors</th><th scope="col">Last active</th></tr>
            </thead>
            <tbody>
              {rep.users.map((u) => (
                <tr key={u.user}>
                  <td className="mono-cell">{u.user}</td><td>{NUM.format(u.questions)}</td><td>{NUM.format(u.requests)}</td>
                  <td>{eur(u.cost_eur)}</td><td>{u.cost_share_pct.toFixed(0)}%</td><td>{secs(u.p95_ms)}</td>
                  <td>{u.errors}</td><td className="muted">{ago(u.last_seen)}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
        )}
      </section>

      <section className="card" aria-label="Members">
        <div className="card-h">
          <h3 className="card-t">Members</h3>
          <form className="member-add" onSubmit={(e) => { e.preventDefault(); void addMember(); }}>
            <input value={newMember} onChange={(e) => setNewMember(e.target.value)} placeholder="email of a signed-up user"
              aria-label="Add member by email" />
            <button className="btn small" type="submit" disabled={!newMember.trim()}>Add</button>
          </form>
        </div>
        {!members ? <div className="empty">Loading…</div> : members.length === 0 ? <div className="empty">No members yet.</div> : (
          <table className="accounts">
            <tbody>
              {members.map((m) => (
                <tr key={m.email}>
                  <td><b>{m.name}</b><div className="model">{m.email}</div></td>
                  <td><span className={`status-pill ${m.status}`}>{m.status}</span></td>
                  <td className="actions"><button className="btn ghost small" onClick={() => removeMember(m.email)}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {b && (
        <p className="muted billing-note">
          Every model call is priced as {b.priced_as ?? "the model that ran it"} — €{b.eur_per_1m_in.toFixed(2)} per 1M input and
          €{b.eur_per_1m_out.toFixed(2)} per 1M output tokens (1 USD = {b.usd_to_eur} EUR), even when a local model answered.
          Requests aren't charged on their own; they're counted to show how many model calls each question needs.
          {t.estimated_cost_share_pct > 0 && ` ${t.estimated_cost_share_pct}% of this spend is estimated from text length (ingest extraction).`}
        </p>
      )}
    </div>
  );
}

// ---------- all accounts ----------

export function UsagePage() {
  const [rows, setRows] = useState<AccountOverview[] | null>(null);
  const [billing, setBilling] = useState<Billing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(() => {
    const m = window.location.hash.match(/^#\/usage\/(.+)$/);
    return m ? decodeURIComponent(m[1]) : null;
  });
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState({ name: "", industry: "" });

  const load = useCallback(() => {
    api.clientAccounts().then((r) => { setRows(r.accounts); setBilling(r.billing); setError(null); })
      .catch((e) => setError(errText(e)));
  }, []);
  useEffect(load, [load]);

  const open = (id: string | null) => {
    window.history.replaceState(null, "", id ? `#/usage/${encodeURIComponent(id)}` : "#/usage");
    setSelected(id);
  };
  const create = async () => {
    try {
      const r = await api.createClientAccount({ name: draft.name.trim(), industry: draft.industry.trim() });
      setCreating(false);
      setDraft({ name: "", industry: "" });
      load();
      open(r.account.id);
    } catch (e) { setError(errText(e)); }
  };

  const totals = useMemo(() => rows && {
    spent: rows.reduce((s, r) => s + r.credit.spent_eur, 0),
    credit: rows.reduce((s, r) => s + r.credit.monthly_credit_eur, 0),
    questions: rows.reduce((s, r) => s + r.last_7d.questions, 0),
  }, [rows]);

  if (selected) return <AccountDetail id={selected} billing={billing} onBack={() => open(null)} onChanged={load} />;

  return (
    <div className="usage">
      <div className="usage-head">
        <div>
          <h2 className="usage-title">Clients</h2>
          {billing && (
            <div className="muted">
              Priced as {billing.priced_as ?? "each model"} · €{billing.eur_per_1m_in.toFixed(2)} / €{billing.eur_per_1m_out.toFixed(2)} per 1M
              input / output tokens · {eur(billing.default_monthly_credit_eur)} monthly credit per client
            </div>
          )}
        </div>
        <button className="btn" onClick={() => setCreating((v) => !v)}>{creating ? "Cancel" : "New client"}</button>
      </div>
      {error && <div className="card error" role="alert">{error}</div>}

      {creating && (
        <form className="card card-b new-client" onSubmit={(e) => { e.preventDefault(); if (draft.name.trim()) void create(); }}>
          <label>Company name<input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} autoFocus /></label>
          <label>Industry<input value={draft.industry} onChange={(e) => setDraft({ ...draft, industry: e.target.value })} placeholder="e.g. Recruitment agency" /></label>
          <button className="btn" type="submit" disabled={!draft.name.trim()}>Create with {eur(billing?.default_monthly_credit_eur ?? 2500)} credit</button>
        </form>
      )}

      {totals && rows && rows.length > 0 && (
        <div className="kpis three">
          <Tile k="Spent this month, all clients" v={eur(totals.spent)} d={<>of {eur(totals.credit)} total credit</>} />
          <Tile k="Clients" v={NUM.format(rows.length)} d={<>{rows.reduce((s, r) => s + r.members, 0)} users</>} />
          <Tile k="Questions, last 7 days" v={NUM.format(totals.questions)} />
        </div>
      )}

      {!rows ? <div className="empty">Loading…</div> : rows.length === 0 ? (
        <div className="card empty">No client accounts yet. Create one, or run <span className="mono">python -m server.seed_accounts</span>.</div>
      ) : (
        <div className="client-grid">
          {rows.map((r) => (
            <button key={r.id} className="card client" onClick={() => open(r.id)} aria-label={`Open ${r.name}`}>
              <div className="client-top">
                <div>
                  <div className="client-name">{r.name}</div>
                  <div className="muted">{r.industry || "No industry"} · {r.members} {r.members === 1 ? "user" : "users"}</div>
                </div>
                <StatusPill status={r.credit.status} />
              </div>
              <CreditMeter credit={r.credit} label={`${r.name}: monthly credit used`} />
              <div className="client-stats">
                <span><b>{NUM.format(r.last_7d.questions)}</b> questions · 7 d</span>
                <span><b>{secs(r.last_7d.p95_ms)}</b> p95</span>
                <span><b>{eur(r.credit.projected_eur)}</b> projected</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
