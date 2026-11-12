import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { Sidebar, type Page } from "./components/Sidebar";
import { AskPage } from "./pages/AskPage";
import { AdminPage } from "./pages/AdminPage";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { LoginPage } from "./pages/LoginPage";
import type { Health, Meta, User } from "./types";

const pageFromHash = (): Page =>
  window.location.hash === "#/benchmark" ? "benchmark" : window.location.hash === "#/admin" ? "admin" : "ask";

type Session = "loading" | "signed-out" | "ready";

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session>("loading");

  // Decide whether sign-in is needed before showing anything protected.
  useEffect(() => {
    api.meta()
      .then(async (m) => {
        setMeta(m);
        if (!m.auth?.enabled) return setSession("ready");
        try {
          setUser((await api.me()).user);
          setSession("ready");
        } catch {
          setSession("signed-out");
        }
      })
      .catch(() => setSession("ready")); // API unreachable: pages show their own errors
    const onHash = () => setPage(pageFromHash());
    const onSignedOut = () => { setUser(null); setSession("signed-out"); };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("kgrag:signed-out", onSignedOut);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("kgrag:signed-out", onSignedOut);
    };
  }, []);

  useEffect(() => {
    if (session === "ready") api.health().then(setHealth).catch(() => setHealth(null));
  }, [session]);

  const onSignedIn = useCallback((u: User) => { setUser(u); setSession("ready"); }, []);
  const signOut = async () => {
    await api.signOut().catch(() => undefined);
    setUser(null);
    setSession("signed-out");
  };
  const toggleUpdates = async (on: boolean) => {
    try {
      setUser((await api.setUpdates(on)).user);
    } catch {
      /* keep the old value; a 401 already sends the user back to sign-in */
    }
  };
  const navigate = (p: Page) => {
    window.location.hash = p === "ask" ? "#/" : `#/${p}`;
    setPage(p);
  };

  if (session === "loading") return <div className="empty" aria-busy="true">Loading…</div>;
  if (session === "signed-out" && meta?.auth.client_id) {
    return <LoginPage clientId={meta.auth.client_id} consentText={meta.auth.consent_text} onSignedIn={onSignedIn} />;
  }

  return (
    <div className="shell">
      <Sidebar page={page} onNavigate={navigate} meta={meta} health={health} user={user} onSignOut={signOut} onToggleUpdates={toggleUpdates} />
      <main className="main">
        <div className="topline">
          <div className="crumb">
            {page === "ask" ? <>Ask <b>/ grounded answer</b></>
              : page === "admin" ? <>Accounts <b>/ sign-ups and access</b></>
              : <>Benchmark <b>/ GraphRAG vs plain vector RAG</b></>}
          </div>
          {meta?.demo ? <span className="pill demo">Demo mode · sample data</span>
            : <span className="pill local">All models on-device · $0 per query</span>}
        </div>
        {page === "admin" && user?.role === "admin" ? <AdminPage me={user} />
          : page === "benchmark" ? <BenchmarkPage /> : <AskPage meta={meta} />}
      </main>
    </div>
  );
}
