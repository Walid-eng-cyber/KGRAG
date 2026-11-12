import type { AskResponse, Benchmark, Health, Meta, User } from "./types";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
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
  signOut: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),
  health: () => request<Health>("/api/health"),
  benchmark: () => request<Benchmark>("/api/benchmark"),
  ask: (question: string, signal?: AbortSignal) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal,
    }),
};
