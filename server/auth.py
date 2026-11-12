"""Sign in with Google + email allowlist + signed session cookie + rate limit.

Flow: the browser gets a Google ID token (Google Identity Services button) and
POSTs it to /api/auth/google. We verify the token's signature, audience and
expiry with Google's public keys, require `email_verified`, check the address
against ALLOWED_EMAILS, and set an HttpOnly signed session cookie. No passwords
are stored and no token is kept server-side.

Auth is enabled when GOOGLE_CLIENT_ID is set. With it unset (local use) the API
is open, as before. When enabled it fails closed: a missing/short SESSION_SECRET
stops the server from starting, and an empty ALLOWED_EMAILS admits nobody.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from collections.abc import Mapping

from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, URLSafeTimedSerializer

COOKIE = "kgrag_session"
_FALSE = {"0", "false", "no"}


class Settings:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        env = os.environ if env is None else env
        self.client_id = env.get("GOOGLE_CLIENT_ID", "").strip()
        self.allowed = {e.strip().lower() for e in env.get("ALLOWED_EMAILS", "").split(",") if e.strip()}
        self.secret = env.get("SESSION_SECRET", "")
        self.ttl_s = int(float(env.get("SESSION_TTL_HOURS", "12")) * 3600)
        # Secure cookies need HTTPS; set COOKIE_SECURE=0 only for plain-http local testing.
        self.cookie_secure = env.get("COOKIE_SECURE", "1").lower() not in _FALSE
        self.rate_per_min = int(env.get("RATE_LIMIT_PER_MIN", "10"))
        self.enabled = bool(self.client_id)
        if self.enabled and len(self.secret) < 32:
            raise RuntimeError(
                "GOOGLE_CLIENT_ID is set, so SESSION_SECRET must be set to a random string of "
                "at least 32 characters. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(48))\"")


settings = Settings()


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret, salt="kgrag-session-v1")


def verify_google_token(credential: str) -> dict:
    """Verify a Google ID token (signature, audience, issuer, expiry). Raises on failure."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    return id_token.verify_oauth2_token(credential, google_requests.Request(), settings.client_id)


def login(credential: str) -> dict:
    """Google ID token -> user dict, or 401/403."""
    if not settings.enabled:
        raise HTTPException(status_code=404, detail="Sign-in is not enabled on this server.")
    try:
        claims = verify_google_token(credential)
    except Exception as e:  # noqa: BLE001 - any verification failure is a failed sign-in
        raise HTTPException(status_code=401, detail="Google sign-in could not be verified.") from e
    email = str(claims.get("email", "")).strip().lower()
    if not email or claims.get("email_verified") is not True:
        raise HTTPException(status_code=403, detail="Google has not verified this email address.")
    if email not in settings.allowed:
        raise HTTPException(status_code=403, detail="This Google account is not on the access list.")
    return {"email": email, "name": claims.get("name") or email, "picture": claims.get("picture")}


def start_session(response: Response, user: dict) -> None:
    response.set_cookie(COOKIE, _serializer().dumps(user), max_age=settings.ttl_s, httponly=True,
                        secure=settings.cookie_secure, samesite="lax", path="/")


def end_session(response: Response) -> None:
    response.delete_cookie(COOKIE, path="/", httponly=True, secure=settings.cookie_secure, samesite="lax")


def current_user(request: Request) -> dict | None:
    """The signed-in user, or None. Re-checks the allowlist so removals take effect."""
    token = request.cookies.get(COOKIE)
    if not token or not settings.enabled:
        return None
    try:
        user = _serializer().loads(token, max_age=settings.ttl_s)
    except BadSignature:  # includes SignatureExpired
        return None
    if not isinstance(user, dict) or user.get("email") not in settings.allowed:
        return None
    return user


def require_user(request: Request) -> dict | None:
    """FastAPI dependency: 401 unless signed in. A no-op when auth is disabled."""
    if not settings.enabled:
        return None
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to use KGRAG.")
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
