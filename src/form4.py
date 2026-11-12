"""Load SEC insider-transaction data (Forms 3/4/5) into the knowledge graph.

Source: the SEC's quarterly "Insider Transactions Data Sets"
https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
Each quarter is a zip of flattened TSV tables. This loader reads SUBMISSION,
REPORTINGOWNER and NONDERIV_TRANS — structured data, so no LLM extraction.

Graph written (all MERGE, so re-running a quarter is idempotent):

  (:SecEntity:COMPANY {cik, name, ticker, aliases})           the issuer
  (:SecEntity:PERSON|COMPANY:Insider {cik, name, aliases})    the reporting owner
  (insider)-[:INSIDER_OF {roles, title, last_filed, form}]->(issuer)
  (insider)-[:TRADED {id, date, code, code_label, shares, price, value,
                      acquired, owned_after, plan_10b5_1, form, accession,
                      co_filers}]->(issuer)

Every TRADED edge carries its SEC accession number, which becomes the citation.

Usage:
    python -m src.form4 data/raw/form345/2026q3            # an unzipped quarter
    python -m src.form4 data/raw/form345/2026q3_form345.zip
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# Transaction codes (SEC Form 4 General Instructions, section 8).
CODE_LABELS = {
    "P": "open-market purchase", "S": "open-market sale", "A": "grant or award",
    "D": "disposition to the issuer", "F": "tax withholding", "I": "discretionary transaction",
    "M": "option exercise", "C": "derivative conversion", "E": "derivative expiration",
    "H": "derivative expiration or cancellation", "O": "out-of-the-money option exercise",
    "X": "in-the-money option exercise", "G": "gift", "L": "small acquisition",
    "W": "acquisition or disposition by will", "Z": "voting trust deposit or withdrawal",
    "J": "other acquisition or disposition", "K": "equity swap", "U": "tender of shares",
    "V": "voluntarily reported transaction",
}

# Reporting owners that are funds or firms rather than people (common for 10% owners).
_ENTITY = re.compile(
    r"\b(INC|CORP|CORPORATION|CO|COMPANY|LLC|L\.L\.C|LP|L\.P|LLP|LTD|LIMITED|PLC|SA|AG|NV|BV|GMBH|"
    r"FUND|FUNDS|TRUST|CAPITAL|PARTNERS|HOLDINGS|HOLDING|GROUP|MANAGEMENT|ADVISORS|ADVISERS|"
    r"INVESTMENTS?|VENTURES|BANK|FOUNDATION|ASSOCIATES|SECURITIES|ASSET|ASSETS)\b\.?", re.I)
_SUFFIX = re.compile(r"\b(JR|SR|II|III|IV|MD|PHD|DR)\.?$", re.I)

TABLES = ("SUBMISSION.tsv", "REPORTINGOWNER.tsv", "NONDERIV_TRANS.tsv")


# ---------------------------------------------------------------------------
# Parsing helpers (pure, unit-tested)
# ---------------------------------------------------------------------------

def parse_date(s: str) -> str | None:
    """'28-SEP-2026' -> '2026-09-28'."""
    s = (s or "").strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def to_float(s: str) -> float | None:
    try:
        return float((s or "").replace(",", "").strip())
    except ValueError:
        return None


def is_entity(name: str) -> bool:
    return bool(_ENTITY.search(name or ""))


def person_aliases(raw: str) -> list[str]:
    """SEC lists people as 'LAST FIRST MIDDLE'. Add 'First Last' and 'First Middle Last'."""
    parts = [p for p in re.split(r"\s+", raw.strip().rstrip(",")) if p]
    if len(parts) < 2:
        return []
    suffix = parts.pop() if _SUFFIX.match(parts[-1]) and len(parts) > 2 else None
    last, first, middle = parts[0], parts[1], parts[2:]
    out = {f"{first} {last}", " ".join([first, *middle, last])}
    if suffix:
        out.add(f"{first} {last} {suffix}")
    return sorted(a.title() for a in out)


def company_aliases(name: str, ticker: str) -> list[str]:
    """Ticker plus the name without legal suffixes: 'NVIDIA CORP' -> 'Nvidia'."""
    out = set()
    if ticker:
        out.add(ticker.upper())
    base = re.sub(r"[,.]", " ", name or "")
    base = re.sub(r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|HOLDINGS|GROUP|LLC|LP|NV|SA|AG)\b", " ",
                  base, flags=re.I)
    base = re.sub(r"\s+", " ", base).strip()
    if base and base.lower() != (name or "").lower():
        out.add(base.title())
    return sorted(out)


def is_10b5_1(flag: str) -> bool | None:
    f = (flag or "").strip().lower()
    return True if f in {"1", "true", "y"} else False if f in {"0", "false", "n"} else None


# ---------------------------------------------------------------------------
# Reading a quarter
# ---------------------------------------------------------------------------

def _open_tables(path: Path) -> dict[str, list[dict]]:
    def read(fh) -> list[dict]:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t"))

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = {Path(n).name: n for n in z.namelist()}
            return {t: read(z.open(names[t])) for t in TABLES}
    return {t: read(open(path / t, "rb")) for t in TABLES}


def build_records(tables: dict[str, list[dict]], include_amendments: bool = False) -> dict[str, list[dict]]:
    """Flatten the three tables into node and edge rows for Neo4j."""
    subs = {}
    for r in tables["SUBMISSION.tsv"]:
        form = r["DOCUMENT_TYPE"].strip()
        if form.endswith("/A") and not include_amendments:
            continue  # amendments restate earlier filings; skipping avoids double counting
        if not r["ISSUERCIK"].strip():
            continue
        subs[r["ACCESSION_NUMBER"]] = r

    owners_by_filing: dict[str, list[dict]] = defaultdict(list)
    for r in tables["REPORTINGOWNER.tsv"]:
        if r["ACCESSION_NUMBER"] in subs and r["RPTOWNERCIK"].strip():
            owners_by_filing[r["ACCESSION_NUMBER"]].append(r)

    issuers, insiders, roles = {}, {}, {}
    for acc, s in subs.items():
        cik = s["ISSUERCIK"].strip().lstrip("0")
        name = s["ISSUERNAME"].strip()
        ticker = s["ISSUERTRADINGSYMBOL"].strip().upper()
        issuers[cik] = {"cik": cik, "name": name, "ticker": ticker, "aliases": company_aliases(name, ticker)}
        filed = parse_date(s["FILING_DATE"])
        for o in owners_by_filing.get(acc, []):
            ocik = o["RPTOWNERCIK"].strip().lstrip("0")
            oname = o["RPTOWNERNAME"].strip()
            entity = is_entity(oname)
            insiders[ocik] = {"cik": ocik, "name": oname, "kind": "COMPANY" if entity else "PERSON",
                              "aliases": [] if entity else person_aliases(oname)}
            key = (ocik, cik)
            rel = [x.strip() for x in o["RPTOWNER_RELATIONSHIP"].split(",") if x.strip()]
            prev = roles.get(key)
            if prev is None or (filed or "") >= (prev["last_filed"] or ""):
                roles[key] = {"insider": ocik, "issuer": cik, "roles": sorted(set(rel) | set(prev["roles"] if prev else [])),
                              "title": o["RPTOWNER_TITLE"].strip() or (prev["title"] if prev else ""),
                              "last_filed": filed, "form": s["DOCUMENT_TYPE"].strip()}
            else:
                prev["roles"] = sorted(set(prev["roles"]) | set(rel))

    trades = []
    for t in tables["NONDERIV_TRANS.tsv"]:
        acc = t["ACCESSION_NUMBER"]
        s, owners = subs.get(acc), owners_by_filing.get(acc)
        if not s or not owners:
            continue
        code = t["TRANS_CODE"].strip().upper()
        shares, price = to_float(t["TRANS_SHARES"]), to_float(t["TRANS_PRICEPERSHARE"])
        # Joint filings list several owners for one transaction: attribute it to the
        # first-listed owner and keep the rest as co-filers, so totals aren't double counted.
        primary, others = owners[0], owners[1:]
        trades.append({
            "id": f"{acc}#{t['NONDERIV_TRANS_SK']}",
            "insider": primary["RPTOWNERCIK"].strip().lstrip("0"),
            "issuer": s["ISSUERCIK"].strip().lstrip("0"),
            "date": parse_date(t["TRANS_DATE"]),
            "code": code, "code_label": CODE_LABELS.get(code, "other"),
            "security": t["SECURITY_TITLE"].strip(),
            "shares": shares, "price": price,
            "value": round(shares * price, 2) if shares is not None and price else None,
            "acquired": t["TRANS_ACQUIRED_DISP_CD"].strip().upper() == "A",
            "owned_after": to_float(t["SHRS_OWND_FOLWNG_TRANS"]),
            "plan_10b5_1": is_10b5_1(s.get("AFF10B5ONE", "")),
            "form": s["DOCUMENT_TYPE"].strip(), "accession": acc,
            "co_filers": [o["RPTOWNERNAME"].strip() for o in others],
        })
    flag_suspect_prices(trades)
    return {"issuers": list(issuers.values()), "insiders": list(insiders.values()),
            "roles": list(roles.values()), "trades": trades}


MAX_PLAUSIBLE_VALUE = 10e9   # no single insider trade this quarter should exceed $10B
PRICE_RATIO = 20.0           # a price 20x away from the company's median trade price is a typo
MAX_PRICE_NO_BASELINE = 5000.0  # with <3 priced trades there's no median; above this is almost always a typo


def flag_suspect_prices(trades: list[dict]) -> None:
    """Mark obvious data-entry errors (SEC publishes filings as filed).

    Some filers type the total amount into the per-share price field. Those rows
    get `price_suspect = True` and are left out of value rankings and totals."""
    by_issuer: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        if t["price"] and t["price"] > 0:
            by_issuer[t["issuer"]].append(t["price"])
    medians = {k: sorted(v)[len(v) // 2] for k, v in by_issuer.items() if len(v) >= 3}
    for t in trades:
        suspect = bool(t["value"] and t["value"] > MAX_PLAUSIBLE_VALUE)
        med, price = medians.get(t["issuer"]), t["price"]
        if med and price and (price > med * PRICE_RATIO or price < med / PRICE_RATIO):
            suspect = True
        if not med and price and price > MAX_PRICE_NO_BASELINE:
            suspect = True
        t["price_suspect"] = suspect


# ---------------------------------------------------------------------------
# Writing to Neo4j
# ---------------------------------------------------------------------------

SCHEMA = [
    "CREATE CONSTRAINT sec_entity_cik IF NOT EXISTS FOR (n:SecEntity) REQUIRE n.cik IS UNIQUE",
    "CREATE INDEX traded_id IF NOT EXISTS FOR ()-[t:TRADED]-() ON (t.id)",
    "CREATE INDEX company_name IF NOT EXISTS FOR (n:COMPANY) ON (n.name)",
    "CREATE INDEX person_name IF NOT EXISTS FOR (n:PERSON) ON (n.name)",
]

_ISSUERS = """
UNWIND $rows AS r
MERGE (c:SecEntity {cik: r.cik})
SET c:COMPANY, c.name = r.name, c.ticker = r.ticker,
    c.aliases = apoc.coll.toSet(coalesce(c.aliases, []) + r.aliases), c.source = 'SEC Form 3/4/5'
"""
_INSIDERS_PERSON = """
UNWIND $rows AS r
MERGE (i:SecEntity {cik: r.cik})
SET i:PERSON, i:Insider, i.name = r.name,
    i.aliases = apoc.coll.toSet(coalesce(i.aliases, []) + r.aliases), i.source = 'SEC Form 3/4/5'
"""
_INSIDERS_ENTITY = """
UNWIND $rows AS r
MERGE (i:SecEntity {cik: r.cik})
SET i:COMPANY, i:Insider, i.name = coalesce(i.name, r.name), i.source = coalesce(i.source, 'SEC Form 3/4/5')
"""
_ROLES = """
UNWIND $rows AS r
MATCH (i:SecEntity {cik: r.insider}), (c:SecEntity {cik: r.issuer})
MERGE (i)-[x:INSIDER_OF]->(c)
SET x.roles = apoc.coll.toSet(coalesce(x.roles, []) + r.roles),
    x.title = CASE WHEN r.title <> '' THEN r.title ELSE x.title END,
    x.last_filed = CASE WHEN x.last_filed IS NULL OR r.last_filed > x.last_filed THEN r.last_filed ELSE x.last_filed END,
    x.form = r.form, x.source = 'SEC Form 3/4/5'
"""
_TRADES = """
UNWIND $rows AS r
MATCH (i:SecEntity {cik: r.insider}), (c:SecEntity {cik: r.issuer})
MERGE (i)-[t:TRADED {id: r.id}]->(c)
SET t += r {.date, .code, .code_label, .security, .shares, .price, .value, .acquired,
            .owned_after, .plan_10b5_1, .form, .accession, .co_filers, .price_suspect},
    t.source = 'SEC Form ' + r.form
"""


def _batched(rows: list[dict], n: int = 2000):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def load(path: Path, include_amendments: bool = False, corpus_id: str | None = None) -> dict[str, int]:
    import config
    from neo4j import GraphDatabase

    recs = build_records(_open_tables(path), include_amendments)
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            for q in SCHEMA:
                s.run(q)
            steps = [
                (_ISSUERS, recs["issuers"]),
                (_INSIDERS_PERSON, [r for r in recs["insiders"] if r["kind"] == "PERSON"]),
                (_INSIDERS_ENTITY, [r for r in recs["insiders"] if r["kind"] == "COMPANY"]),
                (_ROLES, recs["roles"]),
                (_TRADES, recs["trades"]),
            ]
            for query, rows in steps:
                for batch in _batched(rows):
                    s.run(query, rows=batch).consume()
            if corpus_id:
                s.run("MATCH (n:SecEntity) WHERE n.corpus IS NULL OR n.corpus <> $c SET n.corpus = $c", c=corpus_id).consume()
    finally:
        driver.close()
    return {k: len(v) for k, v in recs.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load an SEC Form 3/4/5 quarterly data set into Neo4j.")
    ap.add_argument("path", type=Path, help="unzipped quarter folder or the downloaded .zip")
    ap.add_argument("--include-amendments", action="store_true", help="also load 3/A, 4/A, 5/A filings")
    ap.add_argument("--corpus", help="corpus id to attach the data to (see python -m server.corpus_admin)")
    args = ap.parse_args()
    counts = load(args.path, args.include_amendments, args.corpus)
    print("Loaded: " + ", ".join(f"{v:,} {k}" for k, v in counts.items()))
