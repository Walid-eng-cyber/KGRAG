import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { Account, AccountStatus, User } from "../types";

const FILTERS: { key: AccountStatus | "all"; label: string }[] = [
  { key: "pending", label: "Pending" },
  { key: "approved", label: "Approved" },
  { key: "all", label: "All" },
];

const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : "—");

export function AdminPage({ me }: { me: User }) {
  const [filter, setFilter] = useState<AccountStatus | "all">("pending");
  const [users, setUsers] = useState<Account[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    api.adminUsers(filter === "all" ? undefined : filter)
      .then((r) => setUsers(r.users))
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [filter]);
  useEffect(load, [load]);

  const decide = async (email: string, status: AccountStatus) => {
    setBusy(email);
    try {
      await api.adminSetStatus(email, status);
      load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="card" aria-label="Accounts">
      <div className="card-h">
        <h2 className="card-t">Accounts</h2>
        <div className="tabs" role="tablist">
          {FILTERS.map((f) => (
            <button key={f.key} role="tab" aria-selected={filter === f.key} onClick={() => setFilter(f.key)}>{f.label}</button>
          ))}
        </div>
      </div>
      {error && <div className="card error" role="alert" style={{ margin: 16 }}>{error}</div>}
      {!users ? <div className="empty">Loading…</div> : users.length === 0 ? (
        <div className="empty">{filter === "pending" ? "No sign-ups waiting for approval." : "No accounts."}</div>
      ) : (
        <table className="accounts">
          <thead>
            <tr><th scope="col">Account</th><th scope="col">Status</th><th scope="col">Signed up</th><th scope="col">Updates</th><th scope="col"></th></tr>
          </thead>
          <tbody>
            {users.map((u) => {
              const self = u.email === me.email;
              return (
                <tr key={u.email}>
                  <td><b>{u.name}</b><div className="model">{u.email}{u.role === "admin" ? " · admin" : ""}</div></td>
                  <td><span className={`status-pill ${u.status}`}>{u.status}</span></td>
                  <td className="muted">{when(u.created_at)}</td>
                  <td className="muted">{u.updates_opt_in ? "opted in" : "no"}</td>
                  <td className="actions">
                    {!self && u.status !== "approved" && (
                      <button className="btn" disabled={busy === u.email} onClick={() => decide(u.email, "approved")}>Approve</button>
                    )}
                    {!self && u.status === "pending" && (
                      <button className="btn ghost" disabled={busy === u.email} onClick={() => decide(u.email, "rejected")}>Reject</button>
                    )}
                    {!self && u.status === "approved" && (
                      <button className="btn ghost" disabled={busy === u.email} onClick={() => decide(u.email, "disabled")}>Disable</button>
                    )}
                    {self && <span className="muted">you</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
