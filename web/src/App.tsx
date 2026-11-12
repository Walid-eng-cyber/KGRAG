import { useEffect, useState } from "react";
import { api } from "./api";
import { Sidebar, type Page } from "./components/Sidebar";
import { AskPage } from "./pages/AskPage";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import type { Health, Meta } from "./types";

const pageFromHash = (): Page => (window.location.hash === "#/benchmark" ? "benchmark" : "ask");

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => setMeta(null));
    api.health().then(setHealth).catch(() => setHealth(null));
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = (p: Page) => {
    window.location.hash = p === "benchmark" ? "#/benchmark" : "#/";
    setPage(p);
  };

  return (
    <div className="shell">
      <Sidebar page={page} onNavigate={navigate} meta={meta} health={health} />
      <main className="main">
        <div className="topline">
          <div className="crumb">
            {page === "ask" ? <>Ask <b>/ grounded answer</b></> : <>Benchmark <b>/ GraphRAG vs plain vector RAG</b></>}
          </div>
          {meta?.demo ? <span className="pill demo">Demo mode · sample data</span>
            : <span className="pill local">All models on-device · $0 per query</span>}
        </div>
        {page === "ask" ? <AskPage meta={meta} /> : <BenchmarkPage />}
      </main>
    </div>
  );
}
