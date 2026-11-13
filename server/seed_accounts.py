"""Demo client accounts, to try the per-account monitoring.

Three fictional customers in different businesses, each with its own users, a
corpus with a fitting extraction schema and the default monthly usage credit
(€2,500). By default they start at €0 of usage; --sample-usage also generates N
days of synthetic activity (questions and ingests), priced like live usage:

  Northbridge Talent      recruitment agency — 7 recruiters match job seekers' profiles to open jobs
  Halden & Ruiz Legal     law firm — 4 lawyers query client contracts and case files
  Meridian Supply         wholesale distributor — 5 support agents answer from catalogs and policies

All names and email domains are invented (`.example` domains can never receive
mail). Every generated event is stored with source='seed', so it never mixes
with real usage and is removed cleanly:

    python -m server.seed_accounts                          # create (or refresh) the accounts, €0 usage
    python -m server.seed_accounts --sample-usage --days 30 # plus 30 days of synthetic usage
    python -m server.seed_accounts --clear    # remove demo accounts, users, corpora and events
"""
from __future__ import annotations

import argparse
import math
import os
import random
from datetime import datetime, timedelta, timezone

SEED_SOURCE = "seed"
MODEL = "kimi-k3.1-pro"  # what "ran" the synthetic calls; billed at BILLING_PRICE_AS rates like live calls

# Per account: who works there, what they ingest, and how they use the chat.
#   weight       share of the team's questions this person asks
#   q_per_day    team questions on a working day
#   paths        VECTOR / GRAPH / HYBRID mix
#   answer_in    typical answer-prompt size (tokens): longer documents -> bigger context
#   docs_per_day documents ingested on a working day, chunks per document (min, max)
ACCOUNTS: list[dict] = [
    {
        "id": "acct_northbridge", "name": "Northbridge Talent", "industry": "Recruitment agency",
        "domain": "northbridge-talent.example",
        "members": [("amira.haddad", "Amira Haddad", 1.8), ("lucas.moreau", "Lucas Moreau", 1.3),
                    ("sofia.rinaldi", "Sofia Rinaldi", 1.2), ("daniel.okoro", "Daniel Okoro", 1.0),
                    ("ines.duarte", "Inès Duarte", 0.9), ("marc.lefevre", "Marc Lefèvre", 0.6),
                    ("yuki.tanaka", "Yuki Tanaka", 0.4)],
        "corpus": ("Candidate profiles", "CVs and profiles of job seekers registered with the agency, "
                   "matched against open job descriptions.", "career"),
        "q_per_day": 62, "paths": {"VECTOR": 0.30, "GRAPH": 0.25, "HYBRID": 0.45},
        "answer_in": (2600, 4800), "answer_out": (250, 520), "retry": 0.12, "no_evidence": 0.07,
        "docs_per_day": 9, "chunks": (3, 8),
    },
    {
        "id": "acct_halden", "name": "Halden & Ruiz Legal", "industry": "Law firm",
        "domain": "halden-ruiz.example",
        "members": [("clara.halden", "Clara Halden", 1.4), ("mateo.ruiz", "Mateo Ruiz", 1.2),
                    ("nora.becker", "Nora Becker", 1.0), ("samuel.price", "Samuel Price", 0.5)],
        "corpus": ("Client contracts", "Signed client contracts, amendments and case files.", "company"),
        "q_per_day": 22, "paths": {"VECTOR": 0.45, "GRAPH": 0.15, "HYBRID": 0.40},
        "answer_in": (4500, 9000), "answer_out": (400, 900), "retry": 0.18, "no_evidence": 0.05,
        "docs_per_day": 2, "chunks": (20, 60),
    },
    {
        "id": "acct_meridian", "name": "Meridian Supply", "industry": "Wholesale distribution (customer support)",
        "domain": "meridian-supply.example",
        "members": [("leila.benali", "Leïla Benali", 1.2), ("tom.whitaker", "Tom Whitaker", 1.1),
                    ("aiko.sato", "Aiko Sato", 1.0), ("pablo.ortega", "Pablo Ortega", 1.0),
                    ("grace.mensah", "Grace Mensah", 0.8)],
        "corpus": ("Catalog & policies", "Product catalogs, price lists, shipping and returns policies.", "general"),
        "q_per_day": 85, "paths": {"VECTOR": 0.65, "GRAPH": 0.20, "HYBRID": 0.15},
        "answer_in": (1800, 3200), "answer_out": (150, 320), "retry": 0.08, "no_evidence": 0.11,
        "docs_per_day": 1, "chunks": (30, 80),
    },
]


def _price(tokens_in: int, tokens_out: int) -> float:
    """USD at the billing model's list rates (Claude Sonnet 5.5 by default)."""
    from src.pricing import billing_prices
    _, p_in, p_out = billing_prices(MODEL)
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def _costs(tokens_in: int, tokens_out: int) -> dict:
    from src.pricing import billing_prices, usd_to_eur
    usd = _price(tokens_in, tokens_out)
    return {"cost_usd": usd, "cost_eur": usd_to_eur(usd), "priced_as": billing_prices(MODEL)[0]}


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:  # normal approximation keeps big counts fast
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    k, p, limit = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _stage(calls: int, tin: int, tout: int, ms: float) -> dict:
    from src.pricing import usd_to_eur
    return {"calls": calls, "tokens_in": tin, "tokens_out": tout, "cost_eur": round(usd_to_eur(_price(tin, tout)), 6),
            "ms": round(ms, 1)}


def _question(rng: random.Random, spec: dict) -> dict:
    """One synthetic question's measurements, shaped like a real /api/ask row."""
    path = rng.choices(list(spec["paths"]), weights=list(spec["paths"].values()))[0]
    low = rng.random() < (0.22 if path == "HYBRID" else 0.03)
    stages = {"router": _stage(1, rng.randint(1100, 1400), rng.randint(15, 40), rng.uniform(450, 1300))}
    if path in ("VECTOR", "HYBRID"):
        stages["vector"] = _stage(0, 0, 0, rng.uniform(60, 260))
    if path in ("GRAPH", "HYBRID"):
        stages["graph_plan"] = _stage(1, rng.randint(700, 1100), rng.randint(30, 90), rng.uniform(400, 1100))
        stages["graph"] = _stage(0, 0, 0, rng.uniform(120, 700))

    if rng.random() < 0.015:  # a backend hiccup: the request dies before the answer step
        tin = sum(s["tokens_in"] for s in stages.values())
        tout = sum(s["tokens_out"] for s in stages.values())
        ms = sum(s["ms"] for s in stages.values()) + rng.uniform(2000, 30000)
        return {"kind": "ask", "outcome": "error", "path": path, "low_confidence": low, "retried": False,
                "citation_ok": None, "tokens_in": tin, "tokens_out": tout, **_costs(tin, tout),
                "requests": sum(s["calls"] for s in stages.values()),
                "latency_ms": ms, "stages": stages, "model": MODEL}

    a_in = rng.randint(*spec["answer_in"]) + (1200 if path == "HYBRID" else 0)
    a_out = rng.randint(*spec["answer_out"])
    stages["answer"] = _stage(1, a_in, a_out, a_out * rng.uniform(9, 16) + rng.uniform(400, 900))
    retried = rng.random() < spec["retry"]
    if retried:
        r_in, r_out = a_in + a_out + 150, rng.randint(*spec["answer_out"])
        stages["retry"] = _stage(1, r_in, r_out, r_out * rng.uniform(9, 16) + rng.uniform(400, 900))
    no_ev = rng.random() < spec["no_evidence"] * (1.6 if low else 1.0)
    tin = sum(s["tokens_in"] for s in stages.values())
    tout = sum(s["tokens_out"] for s in stages.values())
    return {"kind": "ask", "outcome": "no_evidence" if no_ev else "answered", "path": path, "low_confidence": low,
            "retried": retried, "citation_ok": not (retried and rng.random() < 0.25),
            "tokens_in": tin, "tokens_out": tout, **_costs(tin, tout),
            "requests": sum(s["calls"] for s in stages.values()),
            "latency_ms": sum(s["ms"] for s in stages.values()) + rng.uniform(30, 120), "stages": stages}


def _ingest(rng: random.Random, spec: dict) -> dict:
    chunks = rng.randint(*spec["chunks"])
    tin, tout = chunks * rng.randint(1000, 1250), chunks * rng.randint(250, 340)
    failed = rng.random() < 0.03
    ms = chunks * rng.uniform(2500, 5500)
    return {"kind": "ingest", "outcome": "failed" if failed else "ok", "tokens_in": tin, "tokens_out": tout,
            **_costs(tin, tout), "requests": chunks, "estimated": True, "latency_ms": ms, "units": chunks,
            "stages": {"extract": _stage(chunks, tin, tout, ms)}}


def generate(spec: dict, corpus_id: str, days: int, now: datetime, rng: random.Random) -> list[dict]:
    """Working-hours activity for one account over the last `days` days (light on weekends)."""
    emails = [f"{m[0]}@{spec['domain']}" for m in spec["members"]]
    weights = [m[2] for m in spec["members"]]
    rows: list[dict] = []
    for back in range(days, -1, -1):
        day = (now - timedelta(days=back)).replace(hour=0, minute=0, second=0, microsecond=0)
        workday = day.weekday() < 5
        trend = 0.85 + 0.3 * (days - back) / max(days, 1)  # usage grows over the period
        factor = (1.0 if workday else 0.08) * trend
        events = [_question(rng, spec) for _ in range(_poisson(rng, spec["q_per_day"] * factor))]
        events += [_ingest(rng, spec) for _ in range(_poisson(rng, spec["docs_per_day"] * factor))]
        for e in events:
            ts = day + timedelta(hours=rng.uniform(8, 18.5))
            if ts > now:
                continue
            e.pop("model", None)
            rows.append({**e, "ts": ts, "account_id": spec["id"], "corpus_id": corpus_id,
                         "user_email": rng.choices(emails, weights=weights)[0], "source": SEED_SOURCE})
    return rows


def _stores():
    import config
    from server.auth import settings
    from server.corpora import CorpusStore
    from server.usage import UsageStore
    corpora = CorpusStore(os.getenv("DOCUMENTS_DB", str(config.DATA_DIR / "documents.db")))
    return settings.store, corpora, UsageStore()


def clear(users, corpora, usage) -> None:
    for spec in ACCOUNTS:
        usage.delete_source(SEED_SOURCE, spec["id"])
        for u in users.members(spec["id"]):
            if u["email"].endswith("@" + spec["domain"]):
                users.delete(u["email"])
        if users.get_account(spec["id"]):
            users.delete_account(spec["id"])
        corpora.delete(f"c_seed_{spec['id'].removeprefix('acct_')}")


def seed(users, corpora, usage, days: int = 0, now: datetime | None = None, seed_value: int = 7) -> dict:
    """Create the demo accounts (idempotent). With days > 0, also (re)generate that much synthetic usage;
    with days = 0 their seeded usage is removed, so they start from €0."""
    import config
    from src import corpus_schema
    now = now or datetime.now(timezone.utc)
    rng = random.Random(seed_value)
    summary = {}
    for spec in ACCOUNTS:
        if users.get_account(spec["id"]) is None:
            users.create_account(spec["name"], spec["industry"], account_id=spec["id"])  # default €2,500 credit
        else:
            users.update_account(spec["id"], name=spec["name"], industry=spec["industry"],
                                 monthly_credit_eur=config.DEFAULT_MONTHLY_CREDIT_EUR)
        emails = []
        for local, name, _w in spec["members"]:
            email = f"{local}@{spec['domain']}"
            if users.get(email) is None:
                users.create(email, name, None, admin=False, updates_opt_in=False)
                users.set_status(email, "approved", by="seed")
            users.set_account(email, spec["id"])
            emails.append(email)
        cid = f"c_seed_{spec['id'].removeprefix('acct_')}"
        if corpora.get(cid) is None:
            title, desc, preset = spec["corpus"]
            corpora.create(title, desc, "documents", corpus_schema.from_preset(preset), emails[0], corpus_id=cid)
        usage.delete_source(SEED_SOURCE, spec["id"])
        rows = generate(spec, cid, days, now, rng) if days > 0 else []
        usage.record_many(rows)
        summary[spec["id"]] = {"name": spec["name"], "users": len(emails), "corpus": cid, "events": len(rows)}
    return summary


def _main() -> None:
    import config  # noqa: F401 - load .env

    ap = argparse.ArgumentParser(prog="python -m server.seed_accounts", description=__doc__.split("\n")[0])
    ap.add_argument("--sample-usage", action="store_true", help="also generate synthetic usage (default: €0)")
    ap.add_argument("--days", type=int, default=30, help="days of synthetic usage with --sample-usage (default 30)")
    ap.add_argument("--clear", action="store_true", help="remove the demo accounts and everything seeded for them")
    args = ap.parse_args()
    users, corpora, usage = _stores()
    if args.clear:
        clear(users, corpora, usage)
        print("Removed the demo accounts, their users, corpora and seeded usage.")
        return
    for aid, s in seed(users, corpora, usage, args.days if args.sample_usage else 0).items():
        print(f"{s['name']:<22} {aid:<18} {s['users']} users · corpus {s['corpus']} · {s['events']} events")
    print("\nSee them with:  python -m server.usage   (or --account <id> for one account)")


if __name__ == "__main__":
    _main()
