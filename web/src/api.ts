import type {
  Account, AccountOverview, AccountStatus, AskEvent, AskResponse, Benchmark, Billing, ClientAccount, Corpus, CorpusInput,
  CorpusProfile, DocumentContent, DocumentInfo, Health, Meta, Preset, UsageReport, User,
} from "./types";

import { drainNdjson } from "./lib/ndjson";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

// Empty = same origin (the API serves the UI). Set VITE_API_URL at build time when the
// UI is hosted elsewhere (e.g. Netlify) and the API runs on another domain.
const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

const UNREACHABLE = "Can't reach the KGRAG API. Is `uvicorn server.main:app` running on :8000?";

/** The ApiError for a non-2xx response (and the sign-out signal for a 401). */
async function failure(res: Response): Promise<ApiError> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* keep statusText */
  }
  if (res.status === 401) window.dispatchEvent(new Event("kgrag:signed-out"));
  return new ApiError(detail, res.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { credentials: "include", ...init });
  } catch {
    throw new ApiError(UNREACHABLE, 0);
  }
  if (!res.ok) throw await failure(res);
  return res.json() as Promise<T>;
}

/** Ask a question and receive the answer as it is written: onEvent gets route, token, reset and finally done. */
async function askStream(question: string, corpusId: string | null, onEvent: (e: AskEvent) => void,
  signal?: AbortSignal): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/ask/stream`, {
      method: "POST", credentials: "include", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, corpus_id: corpusId }), signal,
    });
  } catch (e) {
    if (signal?.aborted) throw e;
    throw new ApiError(UNREACHABLE, 0);
  }
  if (!res.ok) throw await failure(res);
  if (!res.body) throw new ApiError("This browser can't show the answer as it is written.", 0);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  const deliver = (events: unknown[]) => {
    for (const e of events as AskEvent[]) {
      if (e.type === "done" || e.type === "error") finished = true;
      onEvent(e);
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const drained = drainNdjson(buffer + decoder.decode(value, { stream: true }));
    buffer = drained.rest;
    deliver(drained.events);
  }
  deliver(drainNdjson(buffer + decoder.decode() + "\n").events);   // a last line without its newline
  if (!finished) throw new ApiError("The answer stopped before it was finished. Try asking again.", 0);
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
  documentContent: (id: string) => request<DocumentContent>(`/api/documents/${encodeURIComponent(id)}`),
  deleteDocument: (id: string) => request<{ ok: boolean }>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  adminUsers: (status?: AccountStatus) =>
    request<{ users: Account[] }>(`/api/admin/users${status ? `?status=${status}` : ""}`),
  adminSetStatus: (email: string, status: AccountStatus) =>
    request<{ user: User }>(`/api/admin/users/${encodeURIComponent(email)}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),
  clientAccounts: () => request<{ accounts: AccountOverview[]; billing: Billing }>("/api/admin/accounts"),
  createClientAccount: (body: { name: string; industry: string; monthly_credit_eur?: number }) =>
    request<{ account: ClientAccount }>("/api/admin/accounts", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }),
  updateClientAccount: (id: string, body: Partial<Pick<ClientAccount, "name" | "industry" | "monthly_credit_eur" | "status">>) =>
    request<{ account: ClientAccount }>(`/api/admin/accounts/${encodeURIComponent(id)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }),
  accountUsage: (id: string, days = 30) =>
    request<UsageReport>(`/api/admin/accounts/${encodeURIComponent(id)}/usage?days=${days}`),
  accountMembers: (id: string) => request<{ members: User[] }>(`/api/admin/accounts/${encodeURIComponent(id)}/members`),
  addAccountMember: (id: string, email: string) =>
    request<{ user: User }>(`/api/admin/accounts/${encodeURIComponent(id)}/members`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email }),
    }),
  removeAccountMember: (id: string, email: string) =>
    request<{ ok: boolean }>(`/api/admin/accounts/${encodeURIComponent(id)}/members/${encodeURIComponent(email)}`,
      { method: "DELETE" }),
  health: () => request<Health>("/api/health"),
  benchmark: () => request<Benchmark>("/api/benchmark"),
  askStream,
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
