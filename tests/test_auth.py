"""Sign-up / sign-in / approval tests. Google's token check is stubbed; everything else is real."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server import auth
from server.users import CONSENT_TEXT, UserStore

CLAIMS = {
    "admin": {"email": "Boss@Gmail.com", "email_verified": True, "name": "The Admin"},
    "alice": {"email": "alice@gmail.com", "email_verified": True, "name": "Alice"},
    "bob": {"email": "bob@gmail.com", "email_verified": True, "name": "Bob"},
    "unverified": {"email": "carol@gmail.com", "email_verified": False},
}
ASK = {"question": "What are Apple's subsidiaries?"}


@pytest.fixture
def env(tmp_path):
    return {"GOOGLE_CLIENT_ID": "test-client.apps.googleusercontent.com", "ADMIN_EMAILS": "boss@gmail.com",
            "SESSION_SECRET": "x" * 40, "COOKIE_SECURE": "0", "RATE_LIMIT_PER_MIN": "3",
            "USERS_DB": str(tmp_path / "users.db")}


@pytest.fixture
def client(monkeypatch, env):
    monkeypatch.setattr(auth, "settings", auth.Settings(env))
    monkeypatch.setattr(main, "DEMO", True)

    def fake_verify(credential: str) -> dict:
        key = credential.removeprefix("token-for-").strip("-")
        if key not in CLAIMS:
            raise ValueError("bad token")
        return CLAIMS[key]
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)
    return TestClient(main.app)


def cred(who: str) -> str:
    return f"token-for-{who}-----------"


def signup(c, who, updates=False):
    return c.post("/api/auth/signup", json={"credential": cred(who), "updates_opt_in": updates})


def signin(c, who):
    return c.post("/api/auth/google", json={"credential": cred(who)})


def as_admin(env, monkeypatch):
    """A second client signed in as the admin."""
    c = TestClient(main.app)
    signup(c, "admin")
    return c


# ---------- sign-up and approval ----------

def test_protected_routes_need_a_session(client):
    for r in (client.post("/api/ask", json=ASK), client.get("/api/health"),
              client.get("/api/benchmark"), client.get("/api/auth/me")):
        assert r.status_code == 401
    meta = client.get("/api/meta").json()["auth"]   # public: the sign-in page needs it
    assert meta["enabled"] and meta["client_id"] and meta["consent_text"] == CONSENT_TEXT


def test_sign_in_before_sign_up_is_refused(client):
    r = signin(client, "alice")
    assert r.status_code == 404 and "Sign up first" in r.json()["detail"]


def test_sign_up_creates_a_pending_account_without_a_session(client):
    r = signup(client, "alice")
    assert r.status_code == 200
    assert r.json()["user"] == {"email": "alice@gmail.com", "name": "Alice", "picture": None, "role": "user",
                                "status": "pending", "updates_opt_in": False, "account_id": None}
    assert "set-cookie" not in r.headers
    r = signin(client, "alice")
    assert r.status_code == 403 and "waiting for approval" in r.json()["detail"]
    assert client.post("/api/ask", json=ASK).status_code == 401


def test_signing_up_twice_is_a_409(client):
    signup(client, "alice")
    assert signup(client, "alice").status_code == 409


def test_unverified_google_email_cannot_sign_up(client):
    r = signup(client, "unverified")
    assert r.status_code == 403 and "verified" in r.json()["detail"]


def test_forged_token_is_401(client):
    assert signup(client, "forged").status_code == 401
    assert signin(client, "forged").status_code == 401


def test_admin_email_is_approved_and_signed_in_at_sign_up(client):
    r = signup(client, "admin")
    assert r.json()["user"]["role"] == "admin" and r.json()["user"]["status"] == "approved"
    assert client.get("/api/auth/me").json()["user"]["email"] == "boss@gmail.com"
    assert client.post("/api/ask", json=ASK).status_code == 200


def test_full_flow_sign_up_then_admin_approves_then_sign_in(client, env, monkeypatch):
    signup(client, "alice")
    admin = as_admin(env, monkeypatch)
    pending = admin.get("/api/admin/users", params={"status": "pending"}).json()["users"]
    assert [u["email"] for u in pending] == ["alice@gmail.com"]
    r = admin.post("/api/admin/users/alice@gmail.com/status", json={"status": "approved"})
    assert r.status_code == 200 and r.json()["user"]["status"] == "approved"

    r = signin(client, "alice")
    assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
    assert client.post("/api/ask", json=ASK).status_code == 200


def test_rejected_account_cannot_sign_in(client, env, monkeypatch):
    signup(client, "alice")
    as_admin(env, monkeypatch).post("/api/admin/users/alice@gmail.com/status", json={"status": "rejected"})
    r = signin(client, "alice")
    assert r.status_code == 403 and "not approved" in r.json()["detail"]


def test_disabling_an_account_ends_its_session_immediately(client, env, monkeypatch):
    signup(client, "alice")
    admin = as_admin(env, monkeypatch)
    admin.post("/api/admin/users/alice@gmail.com/status", json={"status": "approved"})
    signin(client, "alice")
    assert client.get("/api/auth/me").status_code == 200
    admin.post("/api/admin/users/alice@gmail.com/status", json={"status": "disabled"})
    assert client.get("/api/auth/me").status_code == 401


def test_admin_routes_are_admin_only(client, env, monkeypatch):
    assert client.get("/api/admin/users").status_code == 401
    signup(client, "alice")
    as_admin(env, monkeypatch).post("/api/admin/users/alice@gmail.com/status", json={"status": "approved"})
    signin(client, "alice")
    assert client.get("/api/admin/users").status_code == 403
    assert client.post("/api/admin/users/alice@gmail.com/status", json={"status": "approved"}).status_code == 403


def test_admin_cannot_change_own_status(client):
    signup(client, "admin")
    assert client.post("/api/admin/users/boss@gmail.com/status", json={"status": "disabled"}).status_code == 400


def test_admin_can_sign_in_without_signing_up_first(client, env):
    r = signin(client, "admin")
    assert r.status_code == 200 and r.json()["user"]["role"] == "admin"
    assert UserStore(env["USERS_DB"]).get("boss@gmail.com")["updates_opt_in"] == 0  # no implied consent


def test_existing_account_added_to_admin_emails_becomes_admin_on_sign_in(client, env, monkeypatch):
    signup(client, "bob")
    monkeypatch.setattr(auth, "settings", auth.Settings({**env, "ADMIN_EMAILS": "boss@gmail.com,bob@gmail.com"}))
    r = signin(client, "bob")
    assert r.status_code == 200 and r.json()["user"]["role"] == "admin"


# ---------- update emails: consent and unsubscribe ----------

def test_update_consent_is_recorded_only_when_ticked(client, env):
    signup(client, "alice", updates=True)
    signup(client, "bob", updates=False)
    store = UserStore(env["USERS_DB"])
    a, b = store.get("alice@gmail.com"), store.get("bob@gmail.com")
    assert a["updates_opt_in"] == 1 and a["consent_text"] == CONSENT_TEXT and a["consent_at"]
    assert b["updates_opt_in"] == 0 and b["consent_text"] is None and b["consent_at"] is None
    assert store.subscribers() == []  # pending accounts never get emails


def test_user_can_turn_updates_on_and_off(client):
    signup(client, "admin")
    assert client.put("/api/me/updates", json={"opt_in": True}).json()["user"]["updates_opt_in"] is True
    assert client.put("/api/me/updates", json={"opt_in": False}).json()["user"]["updates_opt_in"] is False


def test_unsubscribe_link_opts_out_without_signing_in(client, env):
    signup(client, "admin", updates=True)
    store = UserStore(env["USERS_DB"])
    token = store.subscribers()[0]["unsubscribe_token"]
    r = TestClient(main.app).get("/api/unsubscribe", params={"token": token})
    assert r.status_code == 200 and "unsubscribed" in r.text
    assert store.get("boss@gmail.com")["updates_opt_in"] == 0
    assert TestClient(main.app).get("/api/unsubscribe", params={"token": "nope"}).status_code == 404


# ---------- sessions and limits ----------

def test_tampered_cookie_is_rejected(client):
    signup(client, "admin")
    tampered = client.cookies.get(auth.COOKIE)[:-3] + "abc"
    # Replace, don't add: the jar would otherwise hold the genuine cookie (domain testserver.local)
    # next to the tampered one, and which of the two gets sent is not fixed.
    client.cookies.clear()
    client.cookies.set(auth.COOKIE, tampered)
    assert client.get("/api/auth/me").status_code == 401


def test_expired_session_is_rejected(client, env, monkeypatch):
    signup(client, "admin")
    monkeypatch.setattr(auth, "settings", auth.Settings({**env, "SESSION_TTL_HOURS": "-1"}))
    assert client.get("/api/auth/me").status_code == 401


def test_logout_clears_the_session(client):
    signup(client, "admin")
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_rate_limit_per_user(client):
    signup(client, "admin")
    codes = [client.post("/api/ask", json=ASK).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert int(client.post("/api/ask", json=ASK).headers["retry-after"]) >= 1


def test_missing_session_secret_refuses_to_start():
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        auth.Settings({"GOOGLE_CLIENT_ID": "abc"})


def test_old_allowed_emails_setting_still_makes_admins(tmp_path):
    s = auth.Settings({"GOOGLE_CLIENT_ID": "abc", "SESSION_SECRET": "x" * 40, "ALLOWED_EMAILS": "Old@Gmail.com",
                       "USERS_DB": str(tmp_path / "u.db")})
    assert s.admins == {"old@gmail.com"}


def test_auth_disabled_leaves_the_api_open():
    c = TestClient(main.app)   # conftest: auth off
    assert c.get("/api/benchmark").status_code == 200
    assert c.get("/api/auth/me").json() == {"user": None}
    assert c.post("/api/auth/google", json={"credential": "x" * 30}).status_code == 404
    assert c.post("/api/auth/signup", json={"credential": "x" * 30}).status_code == 404
    assert c.get("/api/admin/users").status_code == 404
