import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, ApiError } from "../api";
import type { DocumentContent, DocumentInfo } from "../types";

const size = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

/** Overlay with a document's text. Click outside it, press Esc or use the × to go back. */
export function DocumentViewer({ doc, onClose }: { doc: DocumentInfo; onClose: () => void }) {
  const [content, setContent] = useState<DocumentContent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dialog = useRef<HTMLElement>(null);
  const close = useRef(onClose);
  close.current = onClose;

  useEffect(() => {
    let live = true;
    api.documentContent(doc.id)
      .then((r) => live && setContent(r))
      .catch((e) => live && setError(e instanceof ApiError ? e.message : String(e)));
    return () => { live = false; };
  }, [doc.id]);

  // Esc closes; the page behind stays put; focus goes back to the document that was clicked.
  useEffect(() => {
    const back = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close.current(); };
    window.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.current?.focus();
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
      back?.focus?.();
    };
  }, []);

  // PDFs mark page breaks with a form feed.
  const pages = content ? content.text.split("\f") : [];

  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <section className="modal doc-modal" role="dialog" aria-modal="true" aria-labelledby="doc-viewer-title"
        tabIndex={-1} ref={dialog}>
        <div className="modal-h">
          <div className="doc-modal-title">
            <h2 id="doc-viewer-title">{doc.title}</h2>
            <span className="doc-meta">
              {doc.filename} · {size(doc.size_bytes)}{doc.chunks != null ? ` · ${doc.chunks} chunks` : ""}
              {content ? ` · ${content.chars.toLocaleString("en-GB")} characters` : ""}
            </span>
          </div>
          <button className="modal-x" aria-label="Close" onClick={onClose}>×</button>
        </div>
        <div className="modal-b doc-modal-b">
          {error ? <div className="error" role="alert">{error}</div>
            : !content ? <div className="empty" aria-busy="true">Loading…</div>
            : pages.map((page, i) => (
              <div key={i}>
                {pages.length > 1 && <div className="doc-page-break">Page {i + 1}</div>}
                <div className="doc-text">{page.trim()}</div>
              </div>
            ))}
          {content?.truncated && (
            <p className="doc-meta">Showing the first {content.text.length.toLocaleString("en-GB")} of {content.chars.toLocaleString("en-GB")} characters.</p>
          )}
        </div>
        <div className="modal-f">
          <span className="doc-meta">Click outside this window or press Esc to go back.</span>
          <button className="btn ghost" onClick={onClose}>Close</button>
        </div>
      </section>
    </div>,
    document.body,
  );
}
