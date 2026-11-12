import type {
  Account, AccountStatus, AskResponse, Benchmark, Corpus, CorpusInput, CorpusProfile, DocumentInfo, Health, Meta,
  Preset, User,
} from "./types";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

// Empty = same origin (the API serves the UI). Set VITE_API_URL at build time when the
// UI is hosted elsewhere (e.g. Netlify) and the API runs on another domain.
const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { credentials: "include", ...init });
  } catch {
    throw new ApiError("Can't reach the KGRAG API. Is `uvicorn server.main:app` running on :8000?", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    if (res.status === 401) window.dispatchEvent(new Event("kgrag:signed-out"));
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => request<Meta>("/api/meta"),
  me: () => request<{ user: User | null }>("/api/auth/me"),
  signInWithGoogle: (credential: string) =>
    request<{ user: User }>("/api/auth/google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential }),
    }),
  signUpWithGoogle: (credential: string, updates_opt_in: boolean) =>
    request<{ user: User }>("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential, updates_opt_in }),
    }),
  signOut: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),
  setUpdates: (opt_in: boolean) =>
    request<{ user: User }>("/api/me/updates", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ opt_in }),
    }),
  documents: (corpusId: string) =>
    request<{ documents: DocumentInfo[]; max_upload_mb: number; accepted: string[] }>(
      `/api/corpora/${encodeURIComponent(corpusId)}/documents`),
  uploadDocument: (corpusId: string, file: File, title = "") => {
    const body = new FormData();
    body.append("file", file);
    body.append("title", title);
    return request<{ document: DocumentInfo }>(`/api/corpora/${encodeURIComponent(corpusId)}/documents`, { method: "POST", body });
  },
  ingestDocuments: (corpusId: string) =>
    request<{ queued: string[] }>(`/api/corpora/${encodeURIComponent(corpusId)}/ingest`, { method: "POST" }),
  deleteDocument: (id: string) => request<{ ok: boolean }>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  adminUsers: (status?: AccountStatus) =>
    request<{ users: Account[] }>(`/api/admin/users${status ? `?status=${status}` : ""}`),
  adminSetStatus: (email: string, status: AccountStatus) =>
    request<{ user: User }>(`/api/admin/users/${encodeURIComponent(email)}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),
  health: () => request<Health>("/api/health"),
  benchmark: () => request<Benchmark>("/api/benchmark"),
  ask: (question: string, corpusId: string | null, signal?: AbortSignal) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, corpus_id: corpusId }),
      signal,
    }),
  corpora: () => request<{ corpora: Corpus[]; presets: Record<string, Preset> }>("/api/corpora"),
  createCorpus: (body: CorpusInput) =>
    request<{ corpus: Corpus }>("/api/corpora", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }),
  updateCorpus: (id: string, body: Partial<CorpusInput>) =>
    request<{ corpus: Corpus }>(`/api/corpora/${encodeURIComponent(id)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }),
  deleteCorpus: (id: string) => request<{ ok: boolean }>(`/api/corpora/${encodeURIComponent(id)}`, { method: "DELETE" }),
  corpusProfile: (id: string) => request<CorpusProfile>(`/api/corpora/${encodeURIComponent(id)}/profile`),
};
