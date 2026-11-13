"""Client accounts and usage monitoring: metering, the usage store, the API hooks and the demo seed."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import config
import server.main as main
from server import auth, seed_accounts
from server.corpora import CorpusStore
from server.usage import UsageStore
from server.users import UserStore
from src import llm, meter

NOW = datetime(2026, 11, 13, 15, 0, tzinfo=timezone.utc)


# ---- metering ---------------------------------------------------------------

class _Resp:
    status_code, ok, headers, text = 200, True, {}, ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _sonnet_billing(monkeypatch):
    """Bill as Claude Sonnet 5.5 ($2 / $10 per 1M) at a round FX rate, whatever the local .env says."""
    monkeypatch.setattr(config, "BILLING_PRICE_AS", "claude-sonnet-5-5")
    monkeypatch.setattr(config, "USD_TO_EUR", 0.5)
    monkeypatch.setattr(config, "DEFAULT_MONTHLY_CREDIT_EUR", 2500.0)


SONNET_EUR = lambda tin, tout: (tin / 1e6 * 2.0 + tout / 1e6 * 10.0) * 0.5  # noqa: E731


def _hosted(monkeypatch, payload):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "PRICE_PER_1M_INPUT", 0.0)
    monkeypatch.setattr(config, "PRICE_PER_1M_OUTPUT", 0.0)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(payload))


def test_hosted_call_is_metered_with_the_providers_token_counts(monkeypatch):
    _hosted(monkeypatch, {"choices": [{"message": {"content": "hi"}}],
                          "usage": {"prompt_tokens": 1000, "completion_tokens": 200}})
    with meter.meter() as m:
        llm.chat_client("router").chat(model="kimi-k3.1-pro", messages=[{"role": "user", "content": "q"}])
    assert [(c.stage, c.tokens_in, c.tokens_out, c.estimated) for c in m.calls] == [("router", 1000, 200, False)]
    assert m.cost_eur == pytest.approx(SONNET_EUR(1000, 200)) and m.priced_as == "claude-sonnet-5-5"


def test_a_local_model_is_priced_as_sonnet_so_usage_shows_the_external_cost():
    with meter.meter() as m:
        meter.record("answer", "llama3.1:8b", 3000, 300, 4000.0)
    assert m.cost_eur == pytest.approx(SONNET_EUR(3000, 300)) and m.requests == 1


def test_without_shadow_pricing_a_local_model_is_free(monkeypatch):
    monkeypatch.setattr(config, "BILLING_PRICE_AS", "")
    with meter.meter() as m:
        meter.record("answer", "llama3.1:8b", 3000, 300, 4000.0)
    assert m.cost_eur == 0.0


def test_missing_token_counts_are_estimated_not_recorded_as_zero(monkeypatch):
    _hosted(monkeypatch, {"choices": [{"message": {"content": "x" * 400}}]})
    with meter.meter() as m:
        llm.chat_client("answer").chat(model="kimi-k3.1-pro", messages=[{"role": "user", "content": "y" * 800}])
    (c,) = m.calls
    assert (c.tokens_in, c.tokens_out, c.estimated) == (200, 100, True) and m.estimated


def test_stage_can_change_between_calls_and_spans_are_timed(monkeypatch):
    _hosted(monkeypatch, {"choices": [{"message": {"content": "a"}}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
    with meter.meter() as m:
        client = llm.chat_client("answer")
        client.chat(model="kimi-k3.1-pro", messages=[])
        client.stage = "retry"
        client.chat(model="kimi-k3.1-pro", messages=[])
        with meter.span("vector"):
            pass
    stages = m.stages()
    assert stages["answer"]["calls"] == 1 and stages["retry"]["calls"] == 1
    assert stages["vector"]["calls"] == 0 and stages["vector"]["ms"] >= 0


def test_recording_outside_a_meter_is_a_no_op():
    meter.record("router", "kimi-k3.1-pro", 10, 10, 1.0)  # CLI runs, evals: must not raise or leak


# ---- usage store --------------------------------------------------------------

def _store(tmp_path) -> UsageStore:
    return UsageStore(tmp_path / "usage.db")


def test_report_aggregates_cost_latency_users_and_stages(tmp_path):
    s = _store(tmp_path)
    acct = {"id": "a1", "name": "Acme", "industry": "Retail", "monthly_credit_eur": 10.0}
    stages = {"router": {"calls": 1, "tokens_in": 100, "tokens_out": 5, "cost_eur": 0.25, "ms": 500.0},
              "answer": {"calls": 1, "tokens_in": 900, "tokens_out": 95, "cost_eur": 0.75, "ms": 4000.0}}
    for i, (user, outcome) in enumerate([("ann@x.example", "answered"), ("ann@x.example", "answered"),
                                         ("bob@x.example", "no_evidence"), ("bob@x.example", "error")]):
        s.record(kind="ask", outcome=outcome, account_id="a1", user_email=user, path="HYBRID",
                 retried=i == 0, citation_ok=True, requests=2, tokens_in=1000, tokens_out=100, cost_eur=1.0,
                 latency_ms=1000 * (i + 1), stages=stages, ts=NOW - timedelta(hours=i + 1))
    s.record(kind="ingest", outcome="ok", account_id="a1", user_email="ann@x.example", cost_eur=2.0, units=12,
             estimated=True, ts=NOW - timedelta(hours=1))
    s.record(kind="ask", outcome="answered", account_id="other", cost_eur=50.0, ts=NOW)  # someone else's

    r = s.report(acct, days=7, now=NOW)
    t = r["totals"]
    assert (t["questions"], t["ingests"], t["documents_chunks"]) == (4, 1, 12)
    assert t["cost_eur"] == 6.0 and t["ingest_cost_eur"] == 2.0 and t["requests"] == 8
    assert t["cost_per_question_eur"] == 1.0 and t["cost_per_answer_eur"] == 2.0
    assert t["cost_per_request_eur"] == 0.75
    assert r["quality"]["error_pct"] == 25.0 and r["quality"]["retry_pct"] == 25.0
    assert r["latency"]["p50_ms"] == 2000.0  # the error isn't a latency sample
    assert [u["user"] for u in r["users"]] == ["ann@x.example", "bob@x.example"]
    assert r["users"][0]["cost_eur"] == 4.0 and r["users"][0]["ingests"] == 1
    assert {st["stage"]: st["calls"] for st in r["stages"]} == {"router": 4, "answer": 4}
    assert r["credit"]["spent_eur"] == 6.0 and r["credit"]["remaining_eur"] == 4.0
    assert r["credit"]["status"] == "at_risk" and r["billing"]["priced_as"] == "claude-sonnet-5-5"


@pytest.mark.parametrize("spent, credit, status", [(5.0, 2500.0, "ok"), (2100.0, 2500.0, "at_risk"),
                                                   (2600.0, 2500.0, "over"), (5.0, 0.0, "no_credit")])
def test_credit_status(tmp_path, spent, credit, status):
    s = _store(tmp_path)
    s.record(kind="ask", outcome="answered", account_id="a1", cost_eur=spent, ts=NOW.replace(day=1, hour=1))
    # Mid-month, so a small spend doesn't project past the credit.
    assert s.credit({"id": "a1", "monthly_credit_eur": credit}, now=NOW)["status"] == status


def test_last_months_usage_does_not_count_against_this_months_credit(tmp_path):
    s = _store(tmp_path)
    s.record(kind="ask", outcome="answered", account_id="a1", cost_eur=900.0,
             ts=NOW.replace(day=1, hour=0) - timedelta(minutes=1))  # 31 October, 23:59
    c = s.credit({"id": "a1", "monthly_credit_eur": 2500.0}, now=NOW)
    assert c["spent_eur"] == 0.0 and c["remaining_eur"] == 2500.0 and c["resets_at"].startswith("2026-12-01")


def test_reset_puts_every_account_back_to_zero(tmp_path):
    s = _store(tmp_path)
    for a in ("a1", "a2"):
        s.record(kind="ask", outcome="answered", account_id=a, cost_eur=3.0, ts=NOW)
    assert s.reset() == 2
    assert s.credit({"id": "a1", "monthly_credit_eur": 2500.0}, now=NOW)["spent_eur"] == 0.0


def test_bad_rows_are_dropped_without_raising(tmp_path):
    s = _store(tmp_path)
    s.record(kind="nope", outcome="answered")
    assert s.report({"id": None, "name": "-", "monthly_credit_eur": 0}, now=NOW)["totals"]["questions"] == 0


# ---- accounts -----------------------------------------------------------------

def test_accounts_link_users_and_old_databases_gain_the_column(tmp_path):
    import sqlite3
    path = tmp_path / "users.db"
    old = sqlite3.connect(path)  # a users table from before accounts existed
    old.execute("CREATE TABLE users (email TEXT PRIMARY KEY, name TEXT NOT NULL, picture TEXT, status TEXT NOT NULL, "
                "role TEXT NOT NULL DEFAULT 'user', created_at TEXT NOT NULL, decided_at TEXT, decided_by TEXT, "
                "updates_opt_in INTEGER NOT NULL DEFAULT 0, consent_at TEXT, consent_text TEXT, "
                "unsubscribe_token TEXT NOT NULL UNIQUE)")
    old.commit()
    old.close()
    store = UserStore(path)
    assert store.create_account("Default", "Retail")["monthly_credit_eur"] == 2500.0
    a = store.create_account("Acme", "Retail", 50)
    store.create("ann@acme.example", "Ann", None, admin=False, updates_opt_in=False)
    assert store.set_account("ann@acme.example", a["id"])["account_id"] == a["id"]
    assert store.get_account(a["id"])["members"] == 1
    with pytest.raises(ValueError):
        store.set_account("ann@acme.example", "acct_missing")
    store.delete_account(a["id"])
    assert store.get("ann@acme.example")["account_id"] is None


# ---- demo seed ----------------------------------------------------------------

def test_seed_creates_three_businesses_including_the_recruiting_agency_and_is_idempotent(tmp_path):
    users, corpora, usage = UserStore(tmp_path / "u.db"), CorpusStore(tmp_path / "d.db"), _store(tmp_path)
    first = seed_accounts.seed(users, corpora, usage, days=10, now=NOW)
    again = seed_accounts.seed(users, corpora, usage, days=10, now=NOW)
    accts = {a["id"]: a for a in users.list_accounts()}
    assert set(accts) == {"acct_northbridge", "acct_halden", "acct_meridian"}
    assert accts["acct_northbridge"]["industry"] == "Recruitment agency" and accts["acct_northbridge"]["members"] == 7
    assert {a["monthly_credit_eur"] for a in accts.values()} == {2500.0}
    assert corpora.get("c_seed_northbridge")["schema"]["preset"] == "career"
    assert first == again  # same seed -> same events, and re-running replaced rather than duplicated
    rep = usage.report(accts["acct_northbridge"], days=10, now=NOW)
    assert rep["totals"]["questions"] > 100 and rep["totals"]["ingests"] > 10
    assert rep["totals"]["active_users"] == 7 and rep["totals"]["cost_eur"] > 0

    # Without sample usage (the default), the accounts start at €0.
    seed_accounts.seed(users, corpora, usage, now=NOW)
    assert usage.credit(accts["acct_northbridge"], now=NOW)["spent_eur"] == 0.0

    seed_accounts.clear(users, corpora, usage)
    assert users.list_accounts() == [] and users.list() == [] and corpora.get("c_seed_northbridge") is None
    assert usage.report({"id": "acct_northbridge", "name": "-", "monthly_credit_eur": 0},
                        now=NOW)["totals"]["questions"] == 0


# ---- API ----------------------------------------------------------------------

@pytest.fixture
def usage(monkeypatch, tmp_path):
    s = _store(tmp_path)
    monkeypatch.setattr(main, "usage_store", s)
    monkeypatch.setattr(main, "DEMO", False)
    return s


def _fake_answer(question, corpus=None):
    meter.record("router", "kimi-k3.1-pro", 1200, 20, 800.0)
    meter.record("answer", "kimi-k3.1-pro", 3000, 300, 4000.0)
    return {"question": question, "classified": "vector", "confidence": 0.9, "path_used": "VECTOR",
            "answer": "Yes [P1].", "sources": [{"tag": "P1", "kind": "passage", "text": "t"}],
            "vector_hits": [], "graph_result": {}, "attempts": 2, "citations_valid": True}


def test_ask_records_one_event_with_tokens_and_no_question_text(usage, monkeypatch):
    import src.answer
    monkeypatch.setattr(src.answer, "answer", _fake_answer)
    r = TestClient(main.app).post("/api/ask", json={"question": "Who knows Kubernetes in Lyon?"})
    assert r.status_code == 200
    (e,) = usage._events(None, NOW - timedelta(days=3650))
    assert (e["kind"], e["outcome"], e["path"], e["retried"]) == ("ask", "answered", "VECTOR", 1)
    assert (e["tokens_in"], e["tokens_out"]) == (4200, 320)
    assert e["cost_eur"] == pytest.approx(SONNET_EUR(4200, 320)) and e["requests"] == 2
    assert e["priced_as"] == "claude-sonnet-5-5"
    assert "Kubernetes" not in str(e)  # we monitor measurements, never the customer's content


def test_failed_ask_is_recorded_as_an_error(usage, monkeypatch):
    import src.answer

    def boom(question, corpus=None):
        meter.record("router", "kimi-k3.1-pro", 1200, 20, 800.0)
        raise ConnectionError("neo4j down")
    monkeypatch.setattr(src.answer, "answer", boom)
    assert TestClient(main.app).post("/api/ask", json={"question": "anything?"}).status_code == 503
    (e,) = usage._events(None, NOW - timedelta(days=3650))
    assert e["outcome"] == "error" and e["tokens_in"] == 1200  # the router call still cost money


ADMIN = {"email": "boss@gmail.com", "email_verified": True, "name": "Boss"}
RECRUITER = {"email": "amira@northbridge.example", "email_verified": True, "name": "Amira"}


def _login(client, who):
    return client.post("/api/auth/google", json={"credential": who})


def test_operator_sees_each_accounts_usage_and_members_are_attributed(monkeypatch, tmp_path, usage):
    env = {"GOOGLE_CLIENT_ID": "x.apps.googleusercontent.com", "ADMIN_EMAILS": "boss@gmail.com",
           "SESSION_SECRET": "s" * 40, "COOKIE_SECURE": "0", "USERS_DB": str(tmp_path / "users.db")}
    monkeypatch.setattr(auth, "settings", auth.Settings(env))
    claims = {"token-admin-xxxxxxxxxxxx": ADMIN, "token-recruiter-xxxxxxxx": RECRUITER}
    monkeypatch.setattr(auth, "verify_google_token", lambda c: claims[c])
    boss, worker = TestClient(main.app), TestClient(main.app)

    assert _login(boss, "token-admin-xxxxxxxxxxxx").status_code == 200
    acct = boss.post("/api/admin/accounts", json={"name": "Northbridge Talent",
                                                  "industry": "Recruitment agency"}).json()["account"]
    assert acct["monthly_credit_eur"] == 2500.0
    assert worker.post("/api/auth/signup", json={"credential": "token-recruiter-xxxxxxxx"}).status_code == 200
    auth.settings.store.set_status(RECRUITER["email"], "approved", by="test")
    assert boss.post(f"/api/admin/accounts/{acct['id']}/members",
                     json={"email": RECRUITER["email"]}).status_code == 200

    # The recruiter asks in their own corpus; the event lands on their account.
    import src.answer
    monkeypatch.setattr(src.answer, "answer", _fake_answer)
    corpus = main.corpus_store.create("Profiles", "", "documents", {"preset": "career", "entities": [],
                                                                    "relations": []}, RECRUITER["email"])
    assert _login(worker, "token-recruiter-xxxxxxxx").status_code == 200
    assert worker.post("/api/ask", json={"question": "Who fits job J-42?", "corpus_id": corpus["id"]}).status_code == 200

    rows = boss.get("/api/admin/accounts").json()["accounts"]
    assert [(r["name"], r["members"], r["last_7d"]["questions"]) for r in rows] == [("Northbridge Talent", 1, 1)]
    rep = boss.get(f"/api/admin/accounts/{acct['id']}/usage", params={"days": 7}).json()
    assert rep["totals"]["questions"] == 1 and rep["users"][0]["user"] == RECRUITER["email"]
    assert rep["credit"]["monthly_credit_eur"] == 2500.0 and rep["credit"]["spent_eur"] > 0
    assert boss.patch(f"/api/admin/accounts/{acct['id']}",
                      json={"monthly_credit_eur": 3000}).json()["account"]["monthly_credit_eur"] == 3000.0

    # Workers can't see the operator views.
    assert worker.get("/api/admin/accounts").status_code == 403
    assert worker.get(f"/api/admin/accounts/{acct['id']}/usage").status_code == 403
    assert boss.get("/api/admin/accounts/acct_missing/usage").status_code == 404


def test_with_sign_in_off_the_operator_views_are_open_and_new_clients_get_the_default_credit(usage, monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "settings", auth.Settings({"USERS_DB": str(tmp_path / "users.db")}))  # sign-in off
    c = TestClient(main.app)
    made = c.post("/api/admin/accounts", json={"name": "Clinic", "industry": "Healthcare"})
    assert made.status_code == 200 and made.json()["account"]["monthly_credit_eur"] == 2500.0
    rows = c.get("/api/admin/accounts").json()
    assert [(a["name"], a["credit"]["spent_eur"]) for a in rows["accounts"]] == [("Clinic", 0.0)]
    assert rows["billing"]["priced_as"] == "claude-sonnet-5-5" and rows["billing"]["currency"] == "EUR"
    assert c.get("/api/meta").json()["billing"] == {"currency": "EUR", "priced_as": "claude-sonnet-5-5"}


def _acting(monkeypatch, tmp_path, who="pinned.amira@northbridge.example"):
    """A sign-in-off server pinned to one client user (KGRAG_AS_USER), with that user in an account."""
    env = {"USERS_DB": str(tmp_path / "users.db"), "KGRAG_AS_USER": who}
    monkeypatch.setattr(auth, "settings", auth.Settings(env))
    store = auth.settings.store
    acct = store.create_account("Northbridge Talent", "Recruitment agency")
    store.create(who, "Amira", None, admin=False, updates_opt_in=False)
    store.set_status(who, "approved", by="test")
    store.set_account(who, acct["id"])
    return acct


def test_a_pinned_server_attributes_questions_to_that_users_account(usage, monkeypatch, tmp_path):
    import src.answer
    acct = _acting(monkeypatch, tmp_path)
    monkeypatch.setattr(src.answer, "answer", _fake_answer)
    corpus = main.corpus_store.create("Profiles", "", "documents", {"preset": "career", "entities": [],
                                                                    "relations": []}, "pinned.amira@northbridge.example")
    c = TestClient(main.app)
    assert c.get("/api/meta").json()["acting_as"] == {"email": "pinned.amira@northbridge.example", "name": "Amira",
                                                       "account": "Northbridge Talent"}
    assert [x["id"] for x in c.get("/api/corpora").json()["corpora"]] == [corpus["id"]]   # their own corpus
    assert c.post("/api/ask", json={"question": "Who fits job J-42?"}).status_code == 400   # must name a corpus
    assert c.post("/api/ask", json={"question": "Who fits job J-42?", "corpus_id": corpus["id"]}).status_code == 200
    (e,) = usage._events(None, NOW - timedelta(days=3650))
    assert e["account_id"] == acct["id"] and e["user_email"] == "pinned.amira@northbridge.example"
    # Operator views belong to the main server, not to a client's.
    assert c.get("/api/admin/accounts").status_code == 403


def test_a_pinned_server_with_an_unknown_user_says_so(usage, monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "settings", auth.Settings({"USERS_DB": str(tmp_path / "u.db"),
                                                        "KGRAG_AS_USER": "nobody@x.example"}))
    c = TestClient(main.app)
    assert c.get("/api/meta").json()["acting_as"] is None
    r = c.get("/api/corpora")
    assert r.status_code == 503 and "nobody@x.example" in r.json()["detail"]


def test_acting_as_a_user_is_refused_when_sign_in_is_on():
    with pytest.raises(RuntimeError, match="only works with sign-in off"):
        auth.Settings({"GOOGLE_CLIENT_ID": "x.apps.googleusercontent.com", "SESSION_SECRET": "s" * 40,
                       "KGRAG_AS_USER": "pinned.amira@northbridge.example"})


def test_a_clients_users_share_its_corpora_and_other_clients_cannot_see_them(monkeypatch, tmp_path):
    from server.corpora import CorpusStore
    monkeypatch.setattr(main, "corpus_store", CorpusStore(tmp_path / "documents.db"))
    users = UserStore(tmp_path / "users.db")
    a, b = users.create_account("Client A"), users.create_account("Client B")
    for email, acct in [("a1@a.example", a), ("a2@a.example", a), ("b1@b.example", b), ("solo@x.example", None)]:
        users.create(email, email.split("@")[0], None, admin=False, updates_opt_in=False)
        users.set_status(email, "approved", by="test")
        if acct:
            users.set_account(email, acct["id"])

    def as_(email):   # a sign-in-off server acting as this user
        monkeypatch.setattr(auth, "settings", auth.Settings({"USERS_DB": str(tmp_path / "users.db"),
                                                            "KGRAG_AS_USER": email}))
        return TestClient(main.app)

    made = as_("a1@a.example").post("/api/corpora", json={"name": "Profiles", "preset": "career"}).json()["corpus"]
    assert made["account_id"] == a["id"]

    mate = as_("a2@a.example")                                   # a teammate in the same client
    assert [c["id"] for c in mate.get("/api/corpora").json()["corpora"]] == [made["id"]]
    assert mate.get(f"/api/corpora/{made['id']}/documents").status_code == 200
    assert mate.patch(f"/api/corpora/{made['id']}", json={"name": "Renamed"}).json()["corpus"]["name"] == "Renamed"

    other = as_("b1@b.example")                                  # another client: nothing, not even a hint
    assert other.get("/api/corpora").json()["corpora"] == []
    assert other.get(f"/api/corpora/{made['id']}/documents").status_code == 404
    assert as_("solo@x.example").get("/api/corpora").json()["corpora"] == []

    mine = as_("solo@x.example").post("/api/corpora", json={"name": "Mine", "preset": "general"}).json()["corpus"]
    assert mine["account_id"] is None                            # no account: private to its owner, as before
    assert as_("a1@a.example").get(f"/api/corpora/{mine['id']}/documents").status_code == 404


def test_corpora_from_before_client_accounts_gain_the_account_column(tmp_path):
    import sqlite3
    from server.corpora import CorpusStore
    path = tmp_path / "documents.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE corpora (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', "
                "kind TEXT NOT NULL, schema_json TEXT NOT NULL, owner TEXT, created_at TEXT NOT NULL)")
    old.execute("INSERT INTO corpora VALUES ('c_old', 'Old', '', 'documents', '{}', 'ann@x.example', '2026-01-01')")
    old.commit()
    old.close()
    store = CorpusStore(path)
    assert [c["id"] for c in store.list_for("ann@x.example")] == ["c_old"]            # still theirs, still private
    assert store.visible("c_old", "ann@x.example", "acct_1") is not None
    store.assign("c_old", "acct_1")
    assert store.visible("c_old", "ann@x.example") is None                              # now the account's, not personal
    assert store.visible("c_old", None, "acct_1") is not None


def test_ingest_usage_is_attributed_to_the_uploader_and_their_client(usage, monkeypatch, tmp_path):
    from server.corpora import CorpusStore
    monkeypatch.setattr(main, "corpus_store", CorpusStore(tmp_path / "documents.db"))
    corpus = main.corpus_store.create("Profiles", "", "documents", {"preset": "career", "entities": [], "relations": []},
                                      "ann@x.example", account_id="acct_1")
    monkeypatch.setattr(main.corpus_graph, "ingest_text", lambda c, doc_id, title, text, progress:
                        {"chunks": 3, "entities": 5, "relations": 4})
    main._ingest({"id": "DOC-1", "title": "CV", "corpus_id": corpus["id"], "uploaded_by": "ann@x.example"},
                 "x" * 3000, lambda *a: None)
    (e,) = usage._events(None, NOW - timedelta(days=3650))
    assert (e["kind"], e["outcome"], e["account_id"], e["user_email"], e["units"]) == \
           ("ingest", "ok", "acct_1", "ann@x.example", 3)
    assert e["cost_eur"] > 0 and e["estimated"] == 1 and e["priced_as"] == "claude-sonnet-5-5"
