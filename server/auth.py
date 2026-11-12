"""Sign up / sign in with Google, admin approval, signed session cookie, rate limit.

Flow
  * Sign up:  the browser gets a Google ID token and POSTs it to /api/auth/signup.
    We verify it (signature, audience, issuer, expiry), require `email_verified`,
    and create a `pending` account (see server/users.py). Emails in ADMIN_EMAILS
    become approved admins straight away and are signed in.
  * Approval: an admin approves the account (admin page or `python -m server.users`).
  * Sign in:  /api/auth/google with a Google ID token works only for an `approved`
    account; it then sets an HttpOnly signed session cookie. No passwords are stored.

Every request re-reads the account, so rejecting or disabling someone takes effect
immediately. Auth is enabled when GOOGLE_CLIENT_ID is set; it fails closed (a short
SESSION_SECRET stops the server from starting).
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from pathlib import Path

from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, URLSafeTimedSerializer

from server.users import UserStore

COOKIE = "kgrag_session"
_FALSE = {"0", "false", "no"}
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "users.db"


def _emails(raw: str) -> set[str]:
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


class Settings:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        env = os.environ if env is None else env
        self.client_id = env.get("GOOGLE_CLIENT_ID", "").strip()
        # ALLOWED_EMAILS is the pre-sign-up name for the same list; still honoured.
        self.admins = _emails(env.get("ADMIN_EMAILS", "")) | _emails(env.get("ALLOWED_EMAILS", ""))
        self.secret = env.get("SESSION_SECRET", "")
        self.ttl_s = int(float(env.get("SESSION_TTL_HOURS", "12")) * 3600)
        # Secure cookies need HTTPS; set COOKIE_SECURE=0 only for plain-http local testing.
        self.cookie_secure = env.get("COOKIE_SECURE", "1").lower() not in _FALSE
        # "lax" when the UI and API share a site; "none" when the UI is on another domain (e.g. Netlify).
        self.cookie_samesite = env.get("COOKIE_SAMESITE", "lax").strip().lower()
        if self.cookie_samesite not in {"lax", "strict", "none"}:
            raise RuntimeError("COOKIE_SAMESITE must be lax, strict or none.")
        if self.cookie_samesite == "none" and not self.cookie_secure:
            raise RuntimeError("COOKIE_SAMESITE=none requires COOKIE_SECURE=1 (HTTPS); browsers drop it otherwise.")
        self.rate_per_min = int(env.get("RATE_LIMIT_PER_MIN", "10"))
        self.users_db = env.get("USERS_DB", str(_DEFAULT_DB))
        self.enabled = bool(self.client_id)
        self._store: UserStore | None = None
        if self.enabled and len(self.secret) < 32:
            raise RuntimeError(
                "GOOGLE_CLIENT_ID is set, so SESSION_SECRET must be set to a random string of "
                "at least 32 characters. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(48))\"")

    @property
    def store(self) -> UserStore:
        if self._store is None:
            self._store = UserStore(self.users_db)
        return self._store


settings = Settings()


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret, salt="kgrag-session-v1")


def public(user: dict) -> dict:
    """What the browser may see about an account."""
    return {"email": user["email"], "name": user["name"], "picture": user.get("picture"),
            "role": user["role"], "status": user["status"], "updates_opt_in": bool(user["updates_opt_in"])}


def verify_google_token(credential: str) -> dict:
    """Verify a Google ID token (signature, audience, issuer, expiry). Raises on failure."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    return id_token.verify_oauth2_token(credential, google_requests.Request(), settings.client_id)


def _identity(credential: str) -> dict:
    """Google ID token -> {email, name, picture} for a verified address, or 401/403/404."""
    if not settings.enabled:
        raise HTTPException(status_code=404, detail="Sign-in is not enabled on this server.")
    try:
        claims = verify_google_token(credential)
    except Exception as e:  # noqa: BLE001 - any verification failure is a failed sign-in
        raise HTTPException(status_code=401, detail="Google sign-in could not be verified.") from e
    email = str(claims.get("email", "")).strip().lower()
    if not email or claims.get("email_verified") is not True:
        raise HTTPException(status_code=403, detail="Google has not verified this email address.")
    return {"email": email, "name": claims.get("name") or email, "picture": claims.get("picture")}


_STATUS_MESSAGES = {
    "pending": "Your sign-up is waiting for approval. You can sign in once an admin approves it.",
    "rejected": "Your sign-up was not approved.",
    "disabled": "This account has been disabled.",
}


def signup(credential: str, updates_opt_in: bool) -> dict:
    """Create an account. Returns the account; admins come back already approved."""
    ident = _identity(credential)
    store = settings.store
    if store.get(ident["email"]):
        raise HTTPException(status_code=409, detail="You already have an account. Sign in instead.")
    return store.create(ident["email"], ident["name"], ident["picture"],
                        admin=ident["email"] in settings.admins, updates_opt_in=updates_opt_in)


def login(credential: str) -> dict:
    """Sign in an existing, approved account, or 403/404."""
    ident = _identity(credential)
    store = settings.store
    if ident["email"] in settings.admins:
        if store.get(ident["email"]) is None:  # admins don't need to sign up first
            store.create(ident["email"], ident["name"], ident["picture"], admin=True, updates_opt_in=False)
        store.promote_admin(ident["email"])
    user = store.get(ident["email"])
    if user is None:
        raise HTTPException(status_code=404, detail="No account for this Google address yet. Sign up first.")
    if user["status"] != "approved":
        raise HTTPException(status_code=403, detail=_STATUS_MESSAGES[user["status"]])
    return user


def start_session(response: Response, user: dict) -> None:
    response.set_cookie(COOKIE, _serializer().dumps({"email": user["email"]}), max_age=settings.ttl_s,
                        httponly=True, secure=settings.cookie_secure, samesite=settings.cookie_samesite, path="/")


def end_session(response: Response) -> None:
    response.delete_cookie(COOKIE, path="/", httponly=True, secure=settings.cookie_secure,
                           samesite=settings.cookie_samesite)


def current_user(request: Request) -> dict | None:
    """The signed-in, still-approved account, or None."""
    token = request.cookies.get(COOKIE)
    if not token or not settings.enabled:
        return None
    try:
        data = _serializer().loads(token, max_age=settings.ttl_s)
    except BadSignature:  # includes SignatureExpired
        return None
    if not isinstance(data, dict) or not isinstance(data.get("email"), str):
        return None
    user = settings.store.get(data["email"])
    return user if user and user["status"] == "approved" else None


def require_user(request: Request) -> dict | None:
    """FastAPI dependency: 401 unless signed in. A no-op when auth is disabled."""
    if not settings.enabled:
        return None
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to use KGRAG.")
    return user


def require_admin(request: Request) -> dict:
    """FastAPI dependency: 403 unless signed in as an admin (404 when auth is disabled)."""
    if not settings.enabled:
        raise HTTPException(status_code=404, detail="Accounts are not enabled on this server.")
    user = require_user(request)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admins only.")
    return user


# ---- rate limit: sliding 60 s window per user (or per client IP when auth is off) ----
_hits: dict[str, deque[float]] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    """FastAPI dependency for expensive routes. In-memory: one process only."""
    if settings.rate_per_min <= 0:
        return
    user = current_user(request)
    key = user["email"] if user else f"ip:{request.client.host if request.client else 'unknown'}"
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.rate_per_min:
        retry = max(1, int(60 - (now - q[0])) + 1)
        raise HTTPException(status_code=429, detail=f"Too many questions. Try again in {retry} s.",
                            headers={"Retry-After": str(retry)})
    q.append(now)
