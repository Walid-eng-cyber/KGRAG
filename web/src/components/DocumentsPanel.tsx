import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import type { DocumentInfo } from "../types";
import { DocumentViewer } from "./DocumentViewer";

const ACTIVE = new Set(["queued", "ingesting"]);
const PENDING = new Set(["uploaded", "failed"]);

const size = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

/** Sidebar section: the documents KGRAG answers from, with upload and ingest. */
export function DocumentsPanel({ corpusId, onChanged }: { corpusId: string; onChanged?: () => void }) {
  const canManage = true; // corpora are private: the owner manages their own documents
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  const [limits, setLimits] = useState({ maxMb: 20, accepted: [".pdf", ".txt", ".md", ".html"] });
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(0);
  const [viewing, setViewing] = useState<DocumentInfo | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const r = await api.documents(corpusId);
      setDocs(r.documents);
      setLimits({ maxMb: r.max_upload_mb, accepted: r.accepted });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, [corpusId]);

  useEffect(() => { setDocs(null); setError(null); void load(); }, [load]);

  // Poll while something is queued or ingesting.
  const busy = !!docs?.some((d) => ACTIVE.has(d.status));
  const wasBusy = useRef(false);
  useEffect(() => {
    if (wasBusy.current && !busy) onChanged?.();
    wasBusy.current = busy;
  }, [busy, onChanged]);
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => void load(), 2500);
    return () => clearInterval(id);
  }, [busy, load]);

  const onFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setError(null);
    setUploading(files.length);
    for (const f of Array.from(files)) {
      if (f.size > limits.maxMb * 1024 * 1024) {
        setError(`${f.name} is larger than ${limits.maxMb} MB.`);
        continue;
      }
      try {
        await api.uploadDocument(corpusId, f);
      } catch (e) {
        setError(`${f.name}: ${e instanceof ApiError ? e.message : String(e)}`);
      } finally {
        setUploading((n) => n - 1);
      }
    }
    if (inputRef.current) inputRef.current.value = "";
    void load();
  };

  const ingest = async () => {
    setError(null);
    try {
      await api.ingestDocuments(corpusId);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
    void load();
    onChanged?.();
  };

  const remove = async (d: DocumentInfo) => {
    if (!window.confirm(`Remove “${d.title}”? Its text will no longer be used for answers.`)) return;
    try {
      await api.deleteDocument(d.id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
    void load();
  };

  const pending = docs?.filter((d) => PENDING.has(d.status)).length ?? 0;

  return (
    <section className="docs" aria-label="Documents">
      <div className="side-label docs-head">
        <span>Documents</span>
        {docs && docs.length > 0 && <span className="docs-count">{docs.filter((d) => d.status === "ready").length}/{docs.length} ready</span>}
      </div>

      {docs && docs.length === 0 && (
        <p className="docs-empty">Add PDFs or text files to this corpus, then press Ingest.</p>
      )}

      <ul className="docs-list">
        {docs?.map((d) => (
          <li key={d.id} className={`doc ${d.status}`} title={`${d.filename} · ${size(d.size_bytes)}`}>
            <div className="doc-row">
              <span className="doc-kind">{d.kind === "pdf" ? "PDF" : d.kind === "html" ? "HTML" : "TXT"}</span>
              {d.status === "ready" ? (
                <button className="doc-title doc-open" aria-haspopup="dialog" title={`Read ${d.title}`}
                  onClick={() => setViewing(d)}>{d.title}</button>
              ) : (
                <span className="doc-title">{d.title}</span>
              )}
              {canManage && !ACTIVE.has(d.status) && (
                <button className="doc-remove" aria-label={`Remove ${d.title}`} onClick={() => void remove(d)}>×</button>
              )}
            </div>
            {ACTIVE.has(d.status) ? (
              <>
                <span className="doc-bar"><span style={{ width: `${Math.round((d.progress ?? 0) * 100)}%` }} /></span>
                <span className="doc-msg">{d.message}</span>
              </>
            ) : (
              <span className={`doc-msg ${d.status}`}>
                {d.status === "ready" ? `Ready · ${d.chunks ?? 0} chunks`
                  : d.status === "uploaded" ? "Not ingested yet"
                  : d.message}
              </span>
            )}
          </li>
        ))}
      </ul>

      {canManage && (
        <div className="docs-actions">
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={limits.accepted.join(",")}
            hidden
            onChange={(e) => void onFiles(e.target.files)}
          />
          <button className="btn ghost" onClick={() => inputRef.current?.click()} disabled={uploading > 0}>
            {uploading > 0 ? "Uploading…" : "Add files"}
          </button>
          <button className="btn" onClick={() => void ingest()} disabled={pending === 0 || busy}>
            {busy ? "Ingesting…" : pending > 0 ? `Ingest ${pending}` : "Ingest"}
          </button>
        </div>
      )}
      {canManage && <span className="docs-hint">PDF, TXT, Markdown or HTML · up to {limits.maxMb} MB · a few minutes each on local models</span>}
      {error && <div className="docs-error" role="alert">{error}</div>}
      {viewing && <DocumentViewer doc={viewing} onClose={() => setViewing(null)} />}
    </section>
  );
}
