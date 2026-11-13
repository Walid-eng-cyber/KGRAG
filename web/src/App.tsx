import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { CorpusDialog } from "./components/CorpusDialog";
import { Sidebar, type Page } from "./components/Sidebar";
import { AskPage } from "./pages/AskPage";
import { AdminPage } from "./pages/AdminPage";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { LoginPage } from "./pages/LoginPage";
import { UsagePage } from "./pages/UsagePage";
import type { Corpus, CorpusProfile, Health, Meta, Preset, User } from "./types";

const pageFromHash = (): Page => {
  const h = window.location.hash;
  return h === "#/benchmark" ? "benchmark" : h === "#/admin" ? "admin"
    : h === "#/usage" || h.startsWith("#/usage/") ? "usage" : "ask";
};

type Session = "loading" | "signed-out" | "ready";
const CORPUS_KEY = "kgrag.corpus";

function remembered(): string | null {
  try { return localStorage.getItem(CORPUS_KEY); } catch { return null; }
}
function remember(id: string): void {
  try { localStorage.setItem(CORPUS_KEY, id); } catch { /* private mode: fine */ }
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session>("loading");
  const [corpora, setCorpora] = useState<Corpus[] | null>(null);
  const [presets, setPresets] = useState<Record<string, Preset>>({});
  const [corpusId, setCorpusId] = useState<string | null>(remembered);
  const [profile, setProfile] = useState<CorpusProfile | null>(null);
  const [dialog, setDialog] = useState<"new" | "edit" | null>(null);

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

  const loadCorpora = useCallback(async () => {
    if (meta?.demo) return setCorpora([]);
    try {
      const r = await api.corpora();
      setCorpora(r.corpora);
      setPresets(r.presets);
      setCorpusId((cur) => (cur && r.corpora.some((c) => c.id === cur) ? cur : r.corpora[0]?.id ?? null));
    } catch {
      setCorpora([]);
    }
  }, [meta?.demo]);
  useEffect(() => { if (session === "ready" && meta) void loadCorpora(); }, [session, meta, loadCorpora]);

  const corpus = useMemo(() => corpora?.find((c) => c.id === corpusId) ?? null, [corpora, corpusId]);
  const loadProfile = useCallback(() => {
    if (!corpus) return setProfile(null);
    api.corpusProfile(corpus.id).then(setProfile).catch(() => setProfile(null));
  }, [corpus]);
  useEffect(() => { setProfile(null); loadProfile(); }, [loadProfile]);

  // The SEC-only benchmark page makes no sense for a document corpus.
  useEffect(() => {
    if (page === "benchmark" && corpus && corpus.kind !== "sec_insiders") navigate("ask");
  }, [page, corpus]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectCorpus = (id: string) => { setCorpusId(id); remember(id); };
  const corpusChanged = useCallback(() => { void loadCorpora(); loadProfile(); }, [loadCorpora, loadProfile]);

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
      <Sidebar page={page} onNavigate={navigate} meta={meta} health={health} user={user} onSignOut={signOut}
        onToggleUpdates={toggleUpdates} corpora={corpora} corpus={corpus} onSelectCorpus={selectCorpus}
        onNewCorpus={() => setDialog("new")} onEditCorpus={() => corpus && setDialog("edit")} onCorpusChanged={corpusChanged} />
      <main className="main">
        <div className="topline">
          <div className="crumb">
            {page === "ask" ? <>{corpus ? corpus.name : "Ask"} <b>/ grounded answer</b></>
              : page === "admin" ? <>Accounts <b>/ sign-ups and access</b></>
              : page === "usage" ? <>Clients <b>/ usage, cost and credit per account</b></>
              : <>Benchmark <b>/ GraphRAG vs plain vector RAG</b></>}
          </div>
          <div className="pills">
          {meta?.acting_as && (
            <span className="pill acting" title={meta.acting_as.email}>
              {meta.acting_as.account ?? "No account"} · {meta.acting_as.name}
            </span>
          )}
          {meta?.demo ? <span className="pill demo">Demo mode · sample data</span>
            : meta?.billing?.priced_as
              ? <span className="pill local" title="Models run on-device; usage is priced as if a hosted provider answered">
                  On-device · priced as {meta.billing.priced_as}</span>
              : <span className="pill local">All models on-device · $0 per query</span>}
          </div>
        </div>
        {page === "usage" && (user?.role === "admin" || (meta && !meta.auth.enabled && !meta.acting_as)) ? <UsagePage />
          : page === "admin" && user?.role === "admin" ? <AdminPage me={user} />
          : page === "benchmark" ? <BenchmarkPage />
          : <AskPage key={corpus?.id ?? "none"} meta={meta} corpus={corpus} profile={profile} onNewCorpus={() => setDialog("new")} />}
      </main>
      {dialog && (
        <CorpusDialog
          presets={presets}
          editing={dialog === "edit" ? corpus : null}
          onClose={() => setDialog(null)}
          onSaved={(c) => { setDialog(null); selectCorpus(c.id); void loadCorpora().then(loadProfile); }}
          onDeleted={() => { setDialog(null); setCorpusId(null); void loadCorpora(); }}
        />
      )}
    </div>
  );
}
