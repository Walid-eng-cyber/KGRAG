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

type Mode = "signin" | "signup";
type Notice = { kind: "error" | "info"; text: string; offerSignup?: boolean } | null;

interface Props {
  clientId: string;
  consentText: string;
  onSignedIn: (u: User) => void;
}

export function LoginPage({ clientId, consentText, onSignedIn }: Props) {
  const buttonRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<Mode>("signin");
  const [optIn, setOptIn] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);
  const [requested, setRequested] = useState<User | null>(null);
  // Google calls back long after render; read the current choices from refs.
  const modeRef = useRef(mode);
  const optInRef = useRef(optIn);
  modeRef.current = mode;
  optInRef.current = optIn;

  useEffect(() => {
    let cancelled = false;
    loadGsi()
      .then(() => {
        if (cancelled || !window.google) return;
        window.google.accounts.id.initialize({
          client_id: clientId,
          ux_mode: "popup",
          callback: async ({ credential }) => {
            if (!credential) return setNotice({ kind: "error", text: "Google didn't return a sign-in. Please try again." });
            setBusy(true);
            setNotice(null);
            try {
              if (modeRef.current === "signup") {
                const { user } = await api.signUpWithGoogle(credential, optInRef.current);
                if (user.status === "approved") onSignedIn(user);
                else setRequested(user);
              } else {
                onSignedIn((await api.signInWithGoogle(credential)).user);
              }
            } catch (e) {
              const err = e instanceof ApiError ? e : null;
              setNotice({
                kind: err?.status === 403 && /waiting/.test(err.message) ? "info" : "error",
                text: err ? err.message : String(e),
                offerSignup: err?.status === 404,
              });
            } finally {
              setBusy(false);
            }
          },
        });
      })
      .catch((e: Error) => !cancelled && setNotice({ kind: "error", text: e.message }));
    return () => { cancelled = true; };
  }, [clientId, onSignedIn]);

  // (Re)draw Google's button whenever the mode changes so its label matches.
  useEffect(() => {
    let cancelled = false;
    loadGsi().then(() => {
      const el = buttonRef.current;
      if (cancelled || !el || !window.google) return;
      el.innerHTML = "";
      window.google.accounts.id.renderButton(el, {
        theme: "filled_black", size: "large", shape: "pill", width: 300,
        text: mode === "signup" ? "signup_with" : "signin_with",
      });
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [mode, requested]);

  const switchMode = (m: Mode) => { setMode(m); setNotice(null); };

  return (
    <main className="login">
      <section className="login-card" aria-label={mode === "signup" ? "Create an account" : "Sign in"}>
        <div className="brand">
          <div className="brand-mark"><Mark /></div>
          <div><div className="brand-name">KGRAG</div><div className="brand-sub">Knowledge-graph RAG · local</div></div>
        </div>

        {requested ? (
          <>
            <h1>Thanks, {requested.name.split(" ")[0]}.</h1>
            <p>
              Your account for <b>{requested.email}</b> is waiting for approval. Once an admin approves it, come
              back and sign in with the same Google account.
            </p>
            <button className="btn ghost" onClick={() => { setRequested(null); switchMode("signin"); }}>Back to sign in</button>
          </>
        ) : (
          <>
            <div className="tabs login-tabs" role="tablist">
              <button role="tab" aria-selected={mode === "signin"} onClick={() => switchMode("signin")}>Sign in</button>
              <button role="tab" aria-selected={mode === "signup"} onClick={() => switchMode("signup")}>Create account</button>
            </div>
            <h1>{mode === "signup" ? "Create your account" : "Sign in to ask questions"}</h1>
            <p>
              {mode === "signup"
                ? "Sign up with your Google account. Google confirms your email address; an admin then approves your access."
                : "Use the Google account you signed up with. KGRAG never sees or stores a password."}
            </p>
            {mode === "signup" && (
              <label className="consent">
                <input type="checkbox" checked={optIn} onChange={(e) => setOptIn(e.target.checked)} />
                <span>{consentText}</span>
              </label>
            )}
            <div ref={buttonRef} className="gsi-slot" aria-busy={busy} />
            {busy && <span className="muted">Checking your account…</span>}
            {notice && (
              <div className={`card ${notice.kind === "error" ? "error" : "notice"}`} role={notice.kind === "error" ? "alert" : "status"}>
                {notice.text}
                {notice.offerSignup && (
                  <> <button className="linkish" onClick={() => switchMode("signup")}>Create an account</button></>
                )}
              </div>
            )}
          </>
        )}
      </section>
    </main>
  );
}
