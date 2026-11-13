"""Usage monitoring: one row per question or ingest, per client account.

What we record is ours to know as the operator — cost, tokens, requests, latency,
path, outcome — never the customer's content (no question or answer text, no
document text). That keeps the same monitoring valid for every industry a client
works in.

Cost is in EUR. Every LLM call is priced at the BILLING_PRICE_AS model's list
rates (default Claude Sonnet 5.5) converted with USD_TO_EUR, even when a local
model ran it, so the numbers show what an external provider would charge. Each
account has a monthly usage credit (DEFAULT_MONTHLY_CREDIT_EUR, €2,500).

Reports (operator view):
    python -m server.usage                      # every account, this month
    python -m server.usage --account acct_xxx   # one account, last 30 days
    python -m server.usage --account acct_xxx --days 7
    python -m server.usage --reset              # delete every usage event (all accounts back to €0)
"""
from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("kgrag.usage")

KINDS = ("ask", "ingest")
# ask:    answered | no_evidence | error        ingest: ok | failed
OUTCOMES = ("answered", "no_evidence", "error", "ok", "failed")
CREDIT_WARN = 0.8  # share of the monthly credit at which an account is flagged

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    account_id     TEXT,
    user_email     TEXT,
    corpus_id      TEXT,
    kind           TEXT NOT NULL CHECK (kind IN ('ask','ingest')),
    outcome        TEXT NOT NULL CHECK (outcome IN ('answered','no_evidence','error','ok','failed')),
    path           TEXT,
    low_confidence INTEGER,
    retried        INTEGER,
    citation_ok    INTEGER,
    requests       INTEGER NOT NULL DEFAULT 0,
    tokens_in      INTEGER NOT NULL DEFAULT 0,
    tokens_out     INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0,
    cost_eur       REAL NOT NULL DEFAULT 0,
    priced_as      TEXT,
    estimated      INTEGER NOT NULL DEFAULT 0,
    latency_ms     REAL,
    units          INTEGER,
    stages_json    TEXT NOT NULL DEFAULT '{}',
    source         TEXT NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS usage_by_account_ts ON usage_events (account_id, ts);
CREATE INDEX IF NOT EXISTS usage_by_ts ON usage_events (ts);
"""
# Columns added after the first version of the table; older databases gain them on open.
_ADDED = {"requests": "INTEGER NOT NULL DEFAULT 0", "cost_eur": "REAL NOT NULL DEFAULT 0", "priced_as": "TEXT"}


def default_path() -> str:
    from config import DATA_DIR
    return os.getenv("USAGE_DB", str(DATA_DIR / "usage.db"))


def billing() -> dict:
    """How usage is priced, for the UI and reports."""
    import config
    from src.pricing import billing_prices
    as_model, p_in, p_out = billing_prices(config.BILLING_PRICE_AS or "")
    return {"currency": config.BILLING_CURRENCY, "priced_as": config.BILLING_PRICE_AS or None,
            "usd_per_1m_in": p_in, "usd_per_1m_out": p_out, "usd_to_eur": config.USD_TO_EUR,
            "eur_per_1m_in": round(p_in * config.USD_TO_EUR, 4), "eur_per_1m_out": round(p_out * config.USD_TO_EUR, 4),
            "default_monthly_credit_eur": config.DEFAULT_MONTHLY_CREDIT_EUR}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 1)


def month_bounds(now: datetime) -> tuple[datetime, datetime, int]:
    """(start of this month, start of next month, days in month), UTC."""
    start = now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    days = calendar.monthrange(start.year, start.month)[1]
    return start, start + timedelta(days=days), days


class UsageStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path or default_path())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            have = {r["name"] for r in self._db.execute("PRAGMA table_info(usage_events)")}
            for col, ddl in _ADDED.items():
                if col not in have:
                    self._db.execute(f"ALTER TABLE usage_events ADD COLUMN {col} {ddl}")
            self._db.commit()

    # ---- writes ----
    def record(self, *, kind: str, outcome: str, account_id: str | None = None, user_email: str | None = None,
               corpus_id: str | None = None, path: str | None = None, low_confidence: bool | None = None,
               retried: bool | None = None, citation_ok: bool | None = None, requests: int = 0,
               tokens_in: int = 0, tokens_out: int = 0, cost_usd: float = 0.0, cost_eur: float = 0.0,
               priced_as: str | None = None, estimated: bool = False, latency_ms: float | None = None,
               units: int | None = None, stages: dict | None = None, ts: datetime | None = None,
               source: str = "live") -> None:
        """Store one event. Never raises: monitoring must not break the request it measures."""
        try:
            if kind not in KINDS or outcome not in OUTCOMES:
                raise ValueError(f"bad kind/outcome: {kind}/{outcome}")
            b = lambda x: None if x is None else int(bool(x))  # noqa: E731
            with self._lock:
                self._db.execute(
                    "INSERT INTO usage_events (ts, account_id, user_email, corpus_id, kind, outcome, path, "
                    "low_confidence, retried, citation_ok, requests, tokens_in, tokens_out, cost_usd, cost_eur, "
                    "priced_as, estimated, latency_ms, units, stages_json, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (_iso(ts or datetime.now(timezone.utc)), account_id, user_email and user_email.lower(),
                     corpus_id, kind, outcome, path, b(low_confidence), b(retried), b(citation_ok), int(requests),
                     int(tokens_in), int(tokens_out), round(float(cost_usd), 6), round(float(cost_eur), 6),
                     priced_as, int(bool(estimated)), None if latency_ms is None else round(float(latency_ms), 1),
                     units, json.dumps(stages or {}), source))
                self._db.commit()
        except Exception:  # noqa: BLE001
            log.exception("usage event not recorded")

    def record_many(self, rows: list[dict]) -> None:
        for r in rows:
            self.record(**r)

    def delete_source(self, source: str, account_id: str | None = None) -> int:
        q, args = "DELETE FROM usage_events WHERE source = ?", [source]
        if account_id:
            q += " AND account_id = ?"
            args.append(account_id)
        with self._lock:
            n = self._db.execute(q, args).rowcount
            self._db.commit()
        return n

    def reset(self, account_id: str | None = None) -> int:
        """Delete every usage event (or one account's): spending starts again from €0."""
        q, args = "DELETE FROM usage_events", []
        if account_id:
            q += " WHERE account_id = ?"
            args.append(account_id)
        with self._lock:
            n = self._db.execute(q, args).rowcount
            self._db.commit()
        return n

    # ---- reads ----
    def _events(self, account_id: str | None, since: datetime, until: datetime | None = None) -> list[dict]:
        q, args = "SELECT * FROM usage_events WHERE ts >= ?", [_iso(since)]
        if until:
            q += " AND ts < ?"
            args.append(_iso(until))
        if account_id is not None:
            q += " AND account_id = ?"
            args.append(account_id)
        return [dict(r) for r in self._db.execute(q + " ORDER BY ts", args)]

    def spend(self, account_id: str, since: datetime, until: datetime | None = None) -> float:
        q, args = "SELECT COALESCE(SUM(cost_eur), 0) FROM usage_events WHERE account_id = ? AND ts >= ?", \
                  [account_id, _iso(since)]
        if until:
            q += " AND ts < ?"
            args.append(_iso(until))
        return float(self._db.execute(q, args).fetchone()[0])

    def credit(self, account: dict, now: datetime | None = None) -> dict:
        """This month's spend against the account's monthly credit, with a straight-line projection."""
        now = now or datetime.now(timezone.utc)
        start, end, days = month_bounds(now)
        spent = self.spend(account["id"], start)
        elapsed = max((now - start).total_seconds() / 86400, 1 / 24)
        projected = spent / elapsed * days
        limit = float(account.get("monthly_credit_eur") or 0)
        used = spent / limit if limit else None
        status = ("no_credit" if not limit else "over" if spent >= limit else
                  "at_risk" if projected >= limit or used >= CREDIT_WARN else "ok")
        return {"monthly_credit_eur": limit, "spent_eur": round(spent, 4),
                "remaining_eur": round(max(limit - spent, 0.0), 4), "projected_eur": round(projected, 4),
                "used_pct": None if used is None else round(100 * used, 2), "status": status,
                "month": start.strftime("%Y-%m"), "resets_at": _iso(end)}

    def report(self, account: dict, days: int = 30, now: datetime | None = None) -> dict:
        """Everything the operator needs about one account over the last `days` days."""
        now = now or datetime.now(timezone.utc)
        since = now - timedelta(days=days)
        ev = self._events(account["id"], since)
        asks = [e for e in ev if e["kind"] == "ask"]
        ingests = [e for e in ev if e["kind"] == "ingest"]
        answered = [e for e in asks if e["outcome"] == "answered"]
        lat = [e["latency_ms"] for e in asks if e["latency_ms"] is not None and e["outcome"] != "error"]
        cost = sum(e["cost_eur"] for e in ev)
        ask_cost = sum(e["cost_eur"] for e in asks)
        requests = sum(e["requests"] for e in ev)

        def rate(rows, pred) -> float | None:
            return round(100 * sum(1 for e in rows if pred(e)) / len(rows), 1) if rows else None

        by_user: dict[str, dict] = {}
        for e in ev:
            u = by_user.setdefault(e["user_email"] or "(unknown)", {
                "user": e["user_email"] or "(unknown)", "questions": 0, "ingests": 0, "requests": 0,
                "cost_eur": 0.0, "tokens": 0, "_lat": [], "errors": 0, "last_seen": None})
            u["cost_eur"] += e["cost_eur"]
            u["requests"] += e["requests"]
            u["tokens"] += e["tokens_in"] + e["tokens_out"]
            u["last_seen"] = e["ts"]
            if e["kind"] == "ask":
                u["questions"] += 1
                u["errors"] += e["outcome"] == "error"
                if e["latency_ms"] is not None and e["outcome"] != "error":
                    u["_lat"].append(e["latency_ms"])
            else:
                u["ingests"] += 1
        users = []
        for u in by_user.values():
            lats = u.pop("_lat")
            users.append({**u, "cost_eur": round(u["cost_eur"], 4), "p95_ms": _pct(lats, 0.95),
                          "cost_share_pct": round(100 * u["cost_eur"] / cost, 1) if cost else 0.0})
        users.sort(key=lambda u: (-u["cost_eur"], -u["questions"]))

        by_day: dict[str, dict] = {}
        for i in range(days - 1, -1, -1):
            d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            by_day[d] = {"date": d, "questions": 0, "ingests": 0, "requests": 0, "cost_eur": 0.0,
                         "active_users": set()}
        for e in ev:
            d = by_day.get(e["ts"][:10])
            if d is None:
                continue
            d["cost_eur"] += e["cost_eur"]
            d["requests"] += e["requests"]
            d["questions" if e["kind"] == "ask" else "ingests"] += 1
            if e["user_email"]:
                d["active_users"].add(e["user_email"])
        daily = [{**d, "cost_eur": round(d["cost_eur"], 4), "active_users": len(d["active_users"])}
                 for d in by_day.values()]

        stages: dict[str, dict] = {}
        for e in ev:
            for name, s in json.loads(e["stages_json"] or "{}").items():
                t = stages.setdefault(name, {"stage": name, "calls": 0, "tokens_in": 0, "tokens_out": 0,
                                             "cost_eur": 0.0, "_ms": []})
                t["calls"] += s.get("calls", 0)
                t["tokens_in"] += s.get("tokens_in", 0)
                t["tokens_out"] += s.get("tokens_out", 0)
                t["cost_eur"] += s.get("cost_eur", 0.0)
                if s.get("ms") is not None:
                    t["_ms"].append(s["ms"])
        stage_rows = []
        for t in stages.values():
            ms = t.pop("_ms")
            stage_rows.append({**t, "cost_eur": round(t["cost_eur"], 4),
                               "cost_share_pct": round(100 * t["cost_eur"] / cost, 1) if cost else 0.0,
                               "p50_ms": _pct(ms, 0.5), "p95_ms": _pct(ms, 0.95)})
        stage_rows.sort(key=lambda t: (-t["cost_eur"], t["stage"]))

        paths: dict[str, int] = {}
        for e in asks:
            if e["path"]:
                paths[e["path"]] = paths.get(e["path"], 0) + 1
        week_ago = _iso(now - timedelta(days=7))
        return {
            "account": account,
            "billing": billing(),
            "window": {"days": days, "since": _iso(since), "until": _iso(now)},
            "credit": self.credit(account, now),
            "totals": {
                "questions": len(asks), "ingests": len(ingests), "requests": requests,
                "documents_chunks": sum(e["units"] or 0 for e in ingests),
                "cost_eur": round(cost, 4), "ask_cost_eur": round(ask_cost, 4),
                "ingest_cost_eur": round(cost - ask_cost, 4),
                "tokens_in": sum(e["tokens_in"] for e in ev), "tokens_out": sum(e["tokens_out"] for e in ev),
                "cost_per_question_eur": round(ask_cost / len(asks), 6) if asks else None,
                "cost_per_answer_eur": round(ask_cost / len(answered), 6) if answered else None,
                "cost_per_request_eur": round(cost / requests, 6) if requests else None,
                "active_users": len({e["user_email"] for e in ev if e["user_email"]}),
                "active_users_7d": len({e["user_email"] for e in ev if e["user_email"] and e["ts"] >= week_ago}),
                "estimated_cost_share_pct": round(100 * sum(e["cost_eur"] for e in ev if e["estimated"]) / cost, 1)
                if cost else 0.0,
            },
            "latency": {"p50_ms": _pct(lat, 0.5), "p95_ms": _pct(lat, 0.95), "p99_ms": _pct(lat, 0.99)},
            "quality": {
                "answered_pct": rate(asks, lambda e: e["outcome"] == "answered"),
                "no_evidence_pct": rate(asks, lambda e: e["outcome"] == "no_evidence"),
                "error_pct": rate(asks, lambda e: e["outcome"] == "error"),
                "retry_pct": rate(asks, lambda e: e["retried"]),
                "citation_fail_pct": rate([e for e in asks if e["citation_ok"] is not None],
                                          lambda e: not e["citation_ok"]),
                "low_confidence_pct": rate(asks, lambda e: e["low_confidence"]),
                "ingest_failed_pct": rate(ingests, lambda e: e["outcome"] == "failed"),
            },
            "paths": paths,
            "stages": stage_rows,
            "users": users,
            "daily": daily,
        }

    def overview(self, accounts: list[dict], now: datetime | None = None) -> list[dict]:
        """One line per account: this month's credit plus the last 7 days of activity."""
        now = now or datetime.now(timezone.utc)
        out = []
        for a in accounts:
            week = self._events(a["id"], now - timedelta(days=7))
            asks = [e for e in week if e["kind"] == "ask"]
            lat = [e["latency_ms"] for e in asks if e["latency_ms"] is not None and e["outcome"] != "error"]
            out.append({
                "id": a["id"], "name": a["name"], "industry": a.get("industry", ""), "status": a.get("status"),
                "members": a.get("members", 0), "credit": self.credit(a, now),
                "last_7d": {"questions": len(asks), "requests": sum(e["requests"] for e in week),
                            "cost_eur": round(sum(e["cost_eur"] for e in week), 4),
                            "active_users": len({e["user_email"] for e in week if e["user_email"]}),
                            "p95_ms": _pct(lat, 0.95),
                            "error_pct": round(100 * sum(e["outcome"] == "error" for e in asks) / len(asks), 1)
                            if asks else None},
            })
        return out


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------

def _eur(v: float | None) -> str:
    return "—" if v is None else f"€{v:,.2f}" if v >= 1 or v == 0 else f"€{v:.4f}"


def _ms(v: float | None) -> str:
    return "—" if v is None else f"{v / 1000:.1f}s"


def _print_overview(rows: list[dict]) -> None:
    if not rows:
        print("No accounts yet. Create some with: python -m server.seed_accounts")
        return
    b = billing()
    print(f"Priced as {b['priced_as'] or 'each model'} (€{b['eur_per_1m_in']:g} / €{b['eur_per_1m_out']:g} per 1M "
          f"input/output tokens, 1 USD = {b['usd_to_eur']} EUR)\n")
    print(f"{'ACCOUNT':<26} {'INDUSTRY':<28} {'USERS':>5}  {'SPENT':>10} {'CREDIT':>10} {'USED':>6} "
          f"{'PROJECTED':>10}  {'STATUS':<9} {'Q 7d':>5} {'p95 7d':>7}")
    for r in rows:
        c, w = r["credit"], r["last_7d"]
        used = "—" if c["used_pct"] is None else f"{c['used_pct']:.1f}%"
        print(f"{r['name'][:26]:<26} {r['industry'][:28]:<28} {r['members']:>5}  {_eur(c['spent_eur']):>10} "
              f"{_eur(c['monthly_credit_eur']):>10} {used:>6} {_eur(c['projected_eur']):>10}  "
              f"{c['status']:<9} {w['questions']:>5} {_ms(w['p95_ms']):>7}")


def _print_report(r: dict) -> None:
    a, c, t, lat, q = r["account"], r["credit"], r["totals"], r["latency"], r["quality"]
    print(f"\n{a['name']}  ({a['id']}, {a.get('industry') or 'no industry'})  — last {r['window']['days']} days")
    print(f"Credit {c['month']}: {_eur(c['spent_eur'])} of {_eur(c['monthly_credit_eur'])} used "
          f"({c['used_pct'] if c['used_pct'] is not None else '—'}%), {_eur(c['remaining_eur'])} left, "
          f"projected {_eur(c['projected_eur'])} -> {c['status'].upper()}")
    print(f"Cost {_eur(t['cost_eur'])} (questions {_eur(t['ask_cost_eur'])}, ingest {_eur(t['ingest_cost_eur'])})"
          f" · {t['questions']} questions · {t['requests']} LLM requests · {t['ingests']} ingests · "
          f"{t['active_users']} active users")
    print(f"Per question {_eur(t['cost_per_question_eur'])} · per request {_eur(t['cost_per_request_eur'])} · "
          f"latency p50 {_ms(lat['p50_ms'])} p95 {_ms(lat['p95_ms'])}")
    print(f"Answered {q['answered_pct']}% · no evidence {q['no_evidence_pct']}% · errors {q['error_pct']}% · "
          f"retried {q['retry_pct']}% · low-confidence routing {q['low_confidence_pct']}%")
    if r["paths"]:
        print("Paths: " + ", ".join(f"{k} {v}" for k, v in sorted(r["paths"].items(), key=lambda kv: -kv[1])))
    if r["stages"]:
        print(f"\n{'STAGE':<12} {'CALLS':>6} {'COST':>10} {'SHARE':>6} {'p50':>7} {'p95':>7}")
        for s in r["stages"]:
            print(f"{s['stage']:<12} {s['calls']:>6} {_eur(s['cost_eur']):>10} {s['cost_share_pct']:>5.0f}% "
                  f"{_ms(s['p50_ms']):>7} {_ms(s['p95_ms']):>7}")
    if r["users"]:
        print(f"\n{'USER':<44} {'QUESTIONS':>9} {'REQUESTS':>9} {'COST':>10} {'SHARE':>6} {'p95':>7} {'ERR':>4}")
        for u in r["users"]:
            print(f"{u['user'][:44]:<44} {u['questions']:>9} {u['requests']:>9} {_eur(u['cost_eur']):>10} "
                  f"{u['cost_share_pct']:>5.0f}% {_ms(u['p95_ms']):>7} {u['errors']:>4}")
    if t["estimated_cost_share_pct"]:
        print(f"\n{t['estimated_cost_share_pct']}% of the cost is estimated from text length "
              "(ingest extraction doesn't report token counts yet).")


def _main() -> None:
    import config  # noqa: F401 - load .env
    from server.auth import settings

    ap = argparse.ArgumentParser(prog="python -m server.usage", description="Usage and cost per client account.")
    ap.add_argument("--account", help="account id for a detailed report (default: overview of all accounts)")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="print the raw report as JSON")
    ap.add_argument("--reset", action="store_true",
                    help="delete all usage events (only --account's, if given): spending back to €0")
    args = ap.parse_args()

    store, users = UsageStore(), settings.store
    if args.reset:
        n = store.reset(args.account)
        print(f"Deleted {n} usage events{' for ' + args.account if args.account else ''}. Spending is back to €0.")
        return
    if args.account:
        acct = users.get_account(args.account)
        if acct is None:
            raise SystemExit(f"No account {args.account}. List them with: python -m server.usage")
        rep = store.report(acct, args.days)
        print(json.dumps(rep, indent=2)) if args.json else _print_report(rep)
    else:
        rows = store.overview(users.list_accounts())
        print(json.dumps(rows, indent=2)) if args.json else _print_overview(rows)


if __name__ == "__main__":
    _main()
