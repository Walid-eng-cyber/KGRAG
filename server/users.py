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
"""

_COLUMNS = ("email", "name", "picture", "status", "role", "created_at", "decided_at", "decided_by",
            "updates_opt_in", "consent_at", "consent_text")


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
