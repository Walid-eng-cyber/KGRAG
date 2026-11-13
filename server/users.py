"""Registered users (SQLite, stdlib only).

One row per Google account that signed up. An account is `pending` until an
admin approves it; only `approved` accounts can sign in. Emails listed in
ADMIN_EMAILS are approved admins as soon as they sign up.

Opting in to update emails stores the consent time and the exact wording shown,
plus a random unsubscribe token for the link in every email.

Admin CLI (no web UI needed):
    python -m server.users list [--status pending]
    python -m server.users approve someone@gmail.com
    python -m server.users reject someone@gmail.com
    python -m server.users disable someone@gmail.com
"""
from __future__ import annotations

import argparse
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("pending", "approved", "rejected", "disabled")

# Shown next to the sign-up checkbox and stored with every opt-in.
CONSENT_TEXT = ("Email me occasional updates about KGRAG (new features and releases). "
                "I can unsubscribe at any time.")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email             TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    picture           TEXT,
    status            TEXT NOT NULL CHECK (status IN ('pending','approved','rejected','disabled')),
    role              TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user','admin')),
    created_at        TEXT NOT NULL,
    decided_at        TEXT,
    decided_by        TEXT,
    updates_opt_in    INTEGER NOT NULL DEFAULT 0,
    consent_at        TEXT,
    consent_text      TEXT,
    unsubscribe_token TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS accounts (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    industry           TEXT NOT NULL DEFAULT '',
    monthly_credit_eur REAL NOT NULL DEFAULT 2500,
    status             TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','suspended')),
    created_at         TEXT NOT NULL
);
"""

ACCOUNT_STATUSES = ("active", "suspended")

_COLUMNS = ("email", "name", "picture", "status", "role", "created_at", "decided_at", "decided_by",
            "updates_opt_in", "consent_at", "consent_text", "account_id")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class UserStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            # Databases from before client accounts: every user gains an (initially empty) account link.
            cols = {r["name"] for r in self._db.execute("PRAGMA table_info(users)")}
            if "account_id" not in cols:
                self._db.execute("ALTER TABLE users ADD COLUMN account_id TEXT REFERENCES accounts(id)")
            # Accounts first had a USD budget; they now have a monthly usage credit in EUR.
            acols = {r["name"] for r in self._db.execute("PRAGMA table_info(accounts)")}
            if "monthly_budget_usd" in acols and "monthly_credit_eur" not in acols:
                self._db.execute("ALTER TABLE accounts RENAME COLUMN monthly_budget_usd TO monthly_credit_eur")
            self._db.commit()

    # ---- reads ----
    def get(self, email: str) -> dict | None:
        row = self._db.execute(f"SELECT {', '.join(_COLUMNS)} FROM users WHERE email = ?",
                               (email.lower(),)).fetchone()
        return dict(row) if row else None

    def list(self, status: str | None = None) -> list[dict]:
        q = f"SELECT {', '.join(_COLUMNS)} FROM users"
        args: tuple = ()
        if status:
            q += " WHERE status = ?"
            args = (status,)
        return [dict(r) for r in self._db.execute(q + " ORDER BY created_at DESC", args)]

    def subscribers(self) -> list[dict]:
        """Approved users who opted in, with their unsubscribe tokens (for sending)."""
        rows = self._db.execute("SELECT email, name, unsubscribe_token FROM users "
                                "WHERE status = 'approved' AND updates_opt_in = 1")
        return [dict(r) for r in rows]

    # ---- writes ----
    def create(self, email: str, name: str, picture: str | None, *, admin: bool, updates_opt_in: bool) -> dict:
        email = email.lower()
        now = _now()
        with self._lock:
            self._db.execute(
                "INSERT INTO users (email, name, picture, status, role, created_at, decided_at, decided_by, "
                "updates_opt_in, consent_at, consent_text, unsubscribe_token) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (email, name, picture, "approved" if admin else "pending", "admin" if admin else "user", now,
                 now if admin else None, "ADMIN_EMAILS" if admin else None, int(updates_opt_in),
                 now if updates_opt_in else None, CONSENT_TEXT if updates_opt_in else None,
                 secrets.token_urlsafe(24)))
            self._db.commit()
        return self.get(email)  # type: ignore[return-value]

    def set_status(self, email: str, status: str, by: str) -> dict | None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self._lock:
            cur = self._db.execute("UPDATE users SET status = ?, decided_at = ?, decided_by = ? WHERE email = ?",
                                   (status, _now(), by, email.lower()))
            self._db.commit()
        return self.get(email) if cur.rowcount else None

    def promote_admin(self, email: str) -> None:
        """Make an ADMIN_EMAILS account an approved admin (e.g. added to the env after signing up)."""
        with self._lock:
            self._db.execute("UPDATE users SET role = 'admin', status = 'approved', "
                             "decided_at = COALESCE(decided_at, ?), decided_by = COALESCE(decided_by, 'ADMIN_EMAILS') "
                             "WHERE email = ? AND (role != 'admin' OR status != 'approved')", (_now(), email.lower()))
            self._db.commit()

    def set_updates(self, email: str, opt_in: bool) -> dict | None:
        with self._lock:
            self._db.execute(
                "UPDATE users SET updates_opt_in = ?, consent_at = ?, consent_text = ? WHERE email = ?",
                (int(opt_in), _now() if opt_in else None, CONSENT_TEXT if opt_in else None, email.lower()))
            self._db.commit()
        return self.get(email)

    # ---- client accounts (one per customer company; users belong to at most one) ----
    def create_account(self, name: str, industry: str = "", monthly_credit_eur: float | None = None,
                       account_id: str | None = None) -> dict:
        """A client company. Its monthly usage credit defaults to DEFAULT_MONTHLY_CREDIT_EUR (€2,500)."""
        if monthly_credit_eur is None:
            import config
            monthly_credit_eur = config.DEFAULT_MONTHLY_CREDIT_EUR
        aid = account_id or "acct_" + secrets.token_hex(5)
        with self._lock:
            self._db.execute("INSERT INTO accounts (id, name, industry, monthly_credit_eur, status, created_at) "
                             "VALUES (?, ?, ?, ?, 'active', ?)",
                             (aid, name.strip()[:80], industry.strip()[:80], max(0.0, float(monthly_credit_eur)),
                              _now()))
            self._db.commit()
        return self.get_account(aid)  # type: ignore[return-value]

    def get_account(self, account_id: str) -> dict | None:
        row = self._db.execute("SELECT a.*, (SELECT count(*) FROM users u WHERE u.account_id = a.id) AS members "
                               "FROM accounts a WHERE a.id = ?", (account_id,)).fetchone()
        return dict(row) if row else None

    def list_accounts(self) -> list[dict]:
        rows = self._db.execute("SELECT a.*, (SELECT count(*) FROM users u WHERE u.account_id = a.id) AS members "
                                "FROM accounts a ORDER BY a.created_at, a.name")
        return [dict(r) for r in rows]

    def update_account(self, account_id: str, **fields) -> dict | None:
        allowed: dict = {}
        if fields.get("name") is not None:
            allowed["name"] = str(fields["name"]).strip()[:80]
        if fields.get("industry") is not None:
            allowed["industry"] = str(fields["industry"]).strip()[:80]
        if fields.get("monthly_credit_eur") is not None:
            allowed["monthly_credit_eur"] = max(0.0, float(fields["monthly_credit_eur"]))
        if fields.get("status") is not None:
            if fields["status"] not in ACCOUNT_STATUSES:
                raise ValueError(f"status must be one of {ACCOUNT_STATUSES}")
            allowed["status"] = fields["status"]
        if allowed:
            with self._lock:
                self._db.execute(f"UPDATE accounts SET {', '.join(f'{k} = ?' for k in allowed)} WHERE id = ?",
                                 (*allowed.values(), account_id))
                self._db.commit()
        return self.get_account(account_id)

    def delete_account(self, account_id: str) -> None:
        """Remove the account; its users stay, unlinked."""
        with self._lock:
            self._db.execute("UPDATE users SET account_id = NULL WHERE account_id = ?", (account_id,))
            self._db.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
            self._db.commit()

    def set_account(self, email: str, account_id: str | None) -> dict | None:
        """Put a user in an account (None takes them out). Returns the user, or None if unknown."""
        if account_id is not None and self.get_account(account_id) is None:
            raise ValueError(f"no such account: {account_id}")
        with self._lock:
            cur = self._db.execute("UPDATE users SET account_id = ? WHERE email = ?", (account_id, email.lower()))
            self._db.commit()
        return self.get(email) if cur.rowcount else None

    def members(self, account_id: str) -> list[dict]:
        rows = self._db.execute(f"SELECT {', '.join(_COLUMNS)} FROM users WHERE account_id = ? ORDER BY name",
                                (account_id,))
        return [dict(r) for r in rows]

    def delete(self, email: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM users WHERE email = ?", (email.lower(),))
            self._db.commit()

    def unsubscribe(self, token: str) -> str | None:
        """Opt out by the token from an email link. Returns the email, or None if the token is unknown."""
        row = self._db.execute("SELECT email FROM users WHERE unsubscribe_token = ?", (token,)).fetchone()
        if not row:
            return None
        self.set_updates(row["email"], False)
        return row["email"]


def _main() -> None:
    import config  # noqa: F401 - load .env
    from server.auth import settings

    ap = argparse.ArgumentParser(prog="python -m server.users", description="Manage KGRAG accounts.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="list accounts")
    ls.add_argument("--status", choices=STATUSES)
    for cmd in ("approve", "reject", "disable"):
        sub.add_parser(cmd, help=f"{cmd} an account").add_argument("email")
    args = ap.parse_args()

    store = UserStore(settings.users_db)
    if args.cmd == "list":
        rows = store.list(args.status)
        if not rows:
            print("No accounts.")
        for u in rows:
            print(f"{u['status']:<9} {u['role']:<5} {u['email']:<36} updates={'yes' if u['updates_opt_in'] else 'no':<3} "
                  f"signed up {u['created_at']}")
        return
    status = {"approve": "approved", "reject": "rejected", "disable": "disabled"}[args.cmd]
    u = store.set_status(args.email, status, by="cli")
    raise SystemExit(f"{args.email}: now {status}." if u else f"{args.email}: no such account.")


if __name__ == "__main__":
    _main()
