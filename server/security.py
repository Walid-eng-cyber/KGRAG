"""HTTP hardening: security headers, CORS origins, API-docs switch, startup checks."""
from __future__ import annotations

import os
from collections.abc import Mapping

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_TRUE = {"1", "true", "yes"}
WEAK_PASSWORDS = {"", "kgrag-password", "password", "neo4j", "postgres", "change-me"}

# Google Identity Services needs its script, iframe, styles and XHR endpoints.
GSI = "https://accounts.google.com/gsi/"
CSP = "; ".join([
    "default-src 'self'",
    f"script-src 'self' {GSI}client",
    f"style-src 'self' 'unsafe-inline' {GSI}style",  # React sets inline style attributes
    f"frame-src {GSI}",
    f"connect-src 'self' {GSI}",
    "img-src 'self' data: https://lh3.googleusercontent.com",
    "font-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


def cors_origins(env: Mapping[str, str] | None = None) -> list[str]:
    """CORS_ORIGINS (comma-separated) or the Vite dev server. Same-origin requests never need CORS."""
    env = os.environ if env is None else env
    raw = env.get("CORS_ORIGINS", "").strip()
    if not raw:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    if "*" in origins:
        raise RuntimeError("CORS_ORIGINS must list exact origins (e.g. https://kgrag.example.com), not '*'.")
    return origins


def docs_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Interactive API docs (/docs, /openapi.json) are off unless KGRAG_DOCS=1."""
    env = os.environ if env is None else env
    return env.get("KGRAG_DOCS", "").lower() in _TRUE


def check_public_config(auth_enabled: bool, secure_cookies: bool, demo: bool,
                        neo4j_password: str, pg_password: str) -> None:
    """Sign-in on + HTTPS cookies = a real deployment: refuse weak database passwords.

    Local testing (COOKIE_SECURE=0) and demo mode are not checked."""
    if not auth_enabled or not secure_cookies or demo:
        return
    weak = [name for name, pw in (("NEO4J_PASSWORD", neo4j_password), ("PG_PASSWORD", pg_password))
            if pw.strip().lower() in WEAK_PASSWORDS]
    if weak:
        raise RuntimeError(
            f"This looks like a deployment (GOOGLE_CLIENT_ID set, COOKIE_SECURE=1) but {', '.join(weak)} "
            "is empty or a default. "
            "Set strong database passwords in .env before exposing the server. Generate one with: "
            "python -c \"import secrets; print(secrets.token_urlsafe(24))\"")


class SecurityHeaders(BaseHTTPMiddleware):
    def __init__(self, app, hsts: bool) -> None:
        super().__init__(app)
        self.hsts = hsts

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        h = response.headers
        h.setdefault("Content-Security-Policy", CSP)
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        h.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        # Google's sign-in popup must be able to talk back to this window.
        h.setdefault("Cross-Origin-Opener-Policy", "same-origin-allow-popups")
        if request.url.path.startswith("/api/"):
            h.setdefault("Cache-Control", "no-store")
        if self.hsts:  # only meaningful (and only sent) when served over HTTPS
            h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response
