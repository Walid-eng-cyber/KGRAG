"""Hardening: headers, docs switch, error messages, CORS, startup checks, cookie settings."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server import auth
from server.security import check_public_config, cors_origins, docs_enabled


@pytest.fixture
def client():
    return TestClient(main.app)


def test_security_headers_on_api_and_pages(client):
    for path in ("/api/meta", "/api/benchmark"):
        h = client.get(path).headers
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert "https://accounts.google.com/gsi/client" in h["content-security-policy"]
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "strict-origin-when-cross-origin"
        assert h["cross-origin-opener-policy"] == "same-origin-allow-popups"
        assert h["cache-control"] == "no-store"


def test_api_docs_are_off_by_default(client):
    assert not docs_enabled({})
    assert docs_enabled({"KGRAG_DOCS": "1"})
    assert client.get("/openapi.json").status_code in (404, 200)  # 200 only if the SPA fallback serves index.html
    assert "openapi" not in client.get("/openapi.json").text[:200].lower()
    assert "swagger" not in client.get("/docs").text.lower()


def test_backend_errors_do_not_leak_internals(client, monkeypatch):
    def boom(q):
        raise ConnectionError("Couldn't connect to bolt://db-internal.example:7687 as neo4j/hunter2")
    monkeypatch.setattr(main, "DEMO", False)
    monkeypatch.setattr("src.answer.answer", boom)
    detail = client.post("/api/ask", json={"question": "anything at all"}).json()["detail"]
    assert "db-internal" not in detail and "hunter2" not in detail and "ConnectionError" not in detail


def test_cors_origins_from_env():
    assert cors_origins({}) == ["http://localhost:5173", "http://127.0.0.1:5173"]
    assert cors_origins({"CORS_ORIGINS": "https://kgrag.netlify.app/, https://kgrag.example.com"}) == [
        "https://kgrag.netlify.app", "https://kgrag.example.com"]
    with pytest.raises(RuntimeError, match="exact origins"):
        cors_origins({"CORS_ORIGINS": "*"})


def test_cors_refuses_unknown_origins(client):
    r = client.options("/api/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


@pytest.mark.parametrize("neo,pg", [("", "strong-pg-password-123"), ("kgrag-password", "strong-pg-password-123"),
                                    ("strong-neo-password-123", "postgres")])
def test_deployment_refuses_weak_database_passwords(neo, pg):
    with pytest.raises(RuntimeError, match="empty or a default"):
        check_public_config(True, True, False, neo, pg)


def test_weak_passwords_allowed_for_local_and_demo():
    check_public_config(False, True, False, "", "")           # auth off: local use
    check_public_config(True, False, False, "kgrag-password", "kgrag-password")  # COOKIE_SECURE=0: localhost testing
    check_public_config(True, True, True, "", "")             # demo mode touches no database
    check_public_config(True, True, False, "strong-neo-password-123", "strong-pg-password-123")


BASE = {"GOOGLE_CLIENT_ID": "abc", "SESSION_SECRET": "x" * 40}


def test_cookie_samesite_settings(tmp_path):
    assert auth.Settings({**BASE, "USERS_DB": str(tmp_path / "a.db")}).cookie_samesite == "lax"
    s = auth.Settings({**BASE, "COOKIE_SAMESITE": "None", "USERS_DB": str(tmp_path / "b.db")})
    assert s.cookie_samesite == "none" and s.cookie_secure
    with pytest.raises(RuntimeError, match="COOKIE_SECURE=1"):
        auth.Settings({**BASE, "COOKIE_SAMESITE": "none", "COOKIE_SECURE": "0"})
    with pytest.raises(RuntimeError, match="lax, strict or none"):
        auth.Settings({**BASE, "COOKIE_SAMESITE": "sometimes"})
