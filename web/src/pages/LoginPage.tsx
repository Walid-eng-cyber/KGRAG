import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import { Mark } from "../components/Sidebar";
import type { User } from "../types";

// Minimal typing for Google Identity Services (https://accounts.google.com/gsi/client).
interface GoogleId {
  initialize(opts: { client_id: string; callback: (r: { credential?: string }) => void; ux_mode?: "popup" }): void;
  renderButton(el: HTMLElement, opts: Record<string, string | number>): void;
}
declare global {
  interface Window {
    google?: { accounts: { id: GoogleId } };
  }
}

const GSI_SRC = "https://accounts.google.com/gsi/client";

function loadGsi(): Promise<void> {
  if (window.google?.accounts?.id) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${GSI_SRC}"]`);
    const script = existing ?? Object.assign(document.createElement("script"), { src: GSI_SRC, async: true });
    script.addEventListener("load", () => resolve());
    script.addEventListener("error", () => reject(new Error("Couldn't load Google sign-in. Check your connection or ad blocker.")));
    if (!existing) document.head.appendChild(script);
  });
}

export function LoginPage({ clientId, onSignedIn }: { clientId: string; onSignedIn: (u: User) => void }) {
  const buttonRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadGsi()
      .then(() => {
        if (cancelled || !buttonRef.current || !window.google) return;
        window.google.accounts.id.initialize({
          client_id: clientId,
          ux_mode: "popup",
          callback: async ({ credential }) => {
            if (!credential) return setError("Google didn't return a sign-in. Please try again.");
            setBusy(true);
            setError(null);
            try {
              onSignedIn((await api.signInWithGoogle(credential)).user);
            } catch (e) {
              setError(e instanceof ApiError ? e.message : String(e));
            } finally {
              setBusy(false);
            }
          },
        });
        window.google.accounts.id.renderButton(buttonRef.current, {
          theme: "filled_black", size: "large", shape: "pill", text: "signin_with", width: 280,
        });
      })
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => { cancelled = true; };
  }, [clientId, onSignedIn]);

  return (
    <main className="login">
      <section className="login-card" aria-label="Sign in">
        <div className="brand">
          <div className="brand-mark"><Mark /></div>
          <div><div className="brand-name">KGRAG</div><div className="brand-sub">Knowledge-graph RAG · local</div></div>
        </div>
        <h1>Sign in to ask questions</h1>
        <p>
          Access is limited to invited Google accounts. Google confirms your email address; KGRAG never sees or
          stores a password.
        </p>
        <div ref={buttonRef} className="gsi-slot" aria-busy={busy} />
        {busy && <span className="muted">Checking your account…</span>}
        {error && <div className="card error" role="alert">{error}</div>}
      </section>
    </main>
  );
}
