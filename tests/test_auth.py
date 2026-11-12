"""Auth tests: Google's token check is stubbed; everything else is real."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server import auth

ENV = {"GOOGLE_CLIENT_ID": "test-client.apps.googleusercontent.com", "ALLOWED_EMAILS": "Ok@Gmail.com, second@gmail.com",
       "SESSION_SECRET": "x" * 40, "COOKIE_SECURE": "0", "RATE_LIMIT_PER_MIN": "3"}
CLAIMS = {"good": {"email": "ok@gmail.com", "email_verified": True, "name": "Ok User"},
          "unverified": {"email": "ok@gmail.com", "email_verified": False},
          "stranger": {"email": "stranger@gmail.com", "email_verified": True}}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth, "settings", auth.Settings(ENV))
    monkeypatch.setattr(main, "DEMO", True)

    def fake_verify(credential: str) -> dict:
        key = credential.removeprefix("token-for-").strip("-")
        if key not in CLAIMS:
            raise ValueError("bad token")
        return CLAIMS[key]
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)
    return TestClient(main.app)


def sign_in(client, who="good"):
    return client.post("/api/auth/google", json={"credential": f"token-for-{who}-----------"})


ASK = {"question": "What are Apple's subsidiaries?"}


def test_protected_routes_need_a_session(client):
    assert client.post("/api/ask", json=ASK).status_code == 401
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/benchmark").status_code == 401
    assert client.get("/api/auth/me").status_code == 401
    meta = client.get("/api/meta").json()   # public: the login page needs the client id
    assert meta["auth"] == {"enabled": True, "client_id": ENV["GOOGLE_CLIENT_ID"]}


def test_sign_in_sets_an_httponly_cookie_and_unlocks_the_api(client):
    r = sign_in(client)
    assert r.status_code == 200 and r.json()["user"]["email"] == "ok@gmail.com"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert client.get("/api/auth/me").json()["user"]["name"] == "Ok User"
    assert client.post("/api/ask", json=ASK).status_code == 200


def test_invalid_token_is_401(client):
    assert sign_in(client, "forged").status_code == 401
    assert client.get("/api/auth/me").status_code == 401


def test_unverified_email_is_rejected(client):
    r = sign_in(client, "unverified")
    assert r.status_code == 403 and "verified" in r.json()["detail"]


def test_email_not_on_the_allowlist_is_rejected(client):
    r = sign_in(client, "stranger")
    assert r.status_code == 403 and "access list" in r.json()["detail"]
    assert client.post("/api/ask", json=ASK).status_code == 401


def test_tampered_cookie_is_rejected(client):
    sign_in(client)
    client.cookies.set(auth.COOKIE, client.cookies.get(auth.COOKIE)[:-3] + "abc")
    assert client.get("/api/auth/me").status_code == 401


def test_cookie_signed_with_another_secret_is_rejected(client, monkeypatch):
    sign_in(client)
    monkeypatch.setattr(auth, "settings", auth.Settings({**ENV, "SESSION_SECRET": "y" * 40}))
    assert client.get("/api/auth/me").status_code == 401


def test_removing_an_email_from_the_allowlist_revokes_the_session(client, monkeypatch):
    sign_in(client)
    monkeypatch.setattr(auth, "settings", auth.Settings({**ENV, "ALLOWED_EMAILS": "second@gmail.com"}))
    assert client.get("/api/auth/me").status_code == 401


def test_expired_session_is_rejected(client, monkeypatch):
    sign_in(client)
    monkeypatch.setattr(auth, "settings", auth.Settings({**ENV, "SESSION_TTL_HOURS": "-1"}))
    assert client.get("/api/auth/me").status_code == 401


def test_logout_clears_the_session(client):
    sign_in(client)
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_rate_limit_per_user(client):
    sign_in(client)
    codes = [client.post("/api/ask", json=ASK).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = client.post("/api/ask", json=ASK)
    assert int(r.headers["retry-after"]) >= 1


def test_missing_session_secret_refuses_to_start():
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        auth.Settings({"GOOGLE_CLIENT_ID": "abc"})


def test_empty_allowlist_admits_nobody(client, monkeypatch):
    monkeypatch.setattr(auth, "settings", auth.Settings({**ENV, "ALLOWED_EMAILS": ""}))
    assert sign_in(client).status_code == 403


def test_auth_disabled_leaves_the_api_open():
    c = TestClient(main.app)   # conftest: auth off
    assert c.get("/api/benchmark").status_code == 200
    assert c.get("/api/auth/me").json() == {"user": None}
    assert c.post("/api/auth/google", json={"credential": "x" * 30}).status_code == 404
