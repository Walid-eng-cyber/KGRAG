"""Download 10-K filings from SEC EDGAR and turn them into clean text.

EDGAR is free and public but requires a descriptive User-Agent header with a
contact email (set SEC_USER_AGENT in .env). Rate limit is ~10 requests/sec;
we stay well under that.

Flow:  ticker -> CIK -> latest 10-K accession -> primary document -> text
Downloaded text is cached under data/ so re-runs don't re-hit EDGAR.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import DATA_DIR, MAX_FILING_CHARS, SEC_USER_AGENT

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


def _headers() -> dict[str, str]:
    return {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    time.sleep(0.2)  # be polite to EDGAR
    return resp


def ticker_to_cik(ticker: str) -> int:
    """Resolve a stock ticker (e.g. 'AAPL') to its zero-padded SEC CIK."""
    cache = DATA_DIR / "company_tickers.json"
    if not cache.exists():
        cache.write_text(_get(TICKERS_URL).text, encoding="utf-8")
    table = json.loads(cache.read_text(encoding="utf-8"))
    ticker = ticker.upper()
    for row in table.values():
        if row["ticker"].upper() == ticker:
            return int(row["cik_str"])
    raise ValueError(f"Ticker {ticker!r} not found in EDGAR company list.")


def latest_10k(cik: int) -> tuple[str, str, str]:
    """Return (accession_no_nodashes, primary_document, company_name) for the
    most recent 10-K filing of a given CIK."""
    data = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    name = data.get("name", str(cik))
    recent = data["filings"]["recent"]
    for form, accession, doc in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form == "10-K":
            return accession.replace("-", ""), doc, name
    raise ValueError(f"No 10-K found for CIK {cik}.")


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):  # drop noisy financial tables
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()


# Matches "Item 1.", "Item 1A.", "ITEM 7:" etc. — the 10-K section headers.
_ITEM_HEADER = re.compile(r"\bItem\s+(\d+[A-C]?)\s*[\.\:]", re.IGNORECASE)


ITEM_TITLES = {"1": "Item 1 - Business", "1A": "Item 1A - Risk Factors"}


def narrative_sections(text: str, items: tuple[str, ...] = ("1", "1A")) -> dict[str, str]:
    """Map each requested Item to its section text.

    A 10-K is organized into numbered Items. Each header appears at least twice
    (table of contents + the real section), so for each Item we keep the
    *longest* slice — the actual section rather than the TOC line.
    """
    matches = list(_ITEM_HEADER.finditer(text))
    if len(matches) < 2:
        return {}

    longest: dict[str, str] = {}
    for i, m in enumerate(matches):
        item = m.group(1).upper()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[m.start():end]
        if item not in longest or len(segment) > len(longest[item]):
            longest[item] = segment
    return {it: longest[it] for it in items if it in longest}


def extract_narrative(text: str, items: tuple[str, ...] = ("1", "1A")) -> str | None:
    """Concatenated narrative for extraction (Item 1 + Item 1A), each capped to
    an equal share of the budget. Returns None if the structure can't be parsed
    (caller falls back to a raw slice)."""
    sections = narrative_sections(text, items)
    if not sections:
        return None
    per_section = max(4000, MAX_FILING_CHARS // len(items))
    parts = [sections[it][:per_section] for it in items if it in sections]
    return "\n\n".join(parts) if parts else None


def latest_filing_date(ticker: str) -> str | None:
    """Report/period date of the latest 10-K, parsed from the document name
    (e.g. aapl-20250927.htm -> 2025-09-27)."""
    cik = ticker_to_cik(ticker)
    _, doc, _ = latest_10k(cik)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", doc)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def fetch_10k_text(ticker: str) -> tuple[str, str]:
    """Return (company_name, narrative_text) for a ticker's latest 10-K.

    Targets Item 1 / Item 1A (falls back to a raw slice if parsing fails).
    Caches the result under data/<TICKER>_10k.txt.
    """
    cache = DATA_DIR / f"{ticker.upper()}_10k.txt"
    name_cache = DATA_DIR / f"{ticker.upper()}_name.txt"
    if cache.exists() and name_cache.exists():
        return name_cache.read_text(encoding="utf-8"), cache.read_text(encoding="utf-8")

    cik = ticker_to_cik(ticker)
    accession, doc, name = latest_10k(cik)
    url = f"{ARCHIVE_BASE}/{cik}/{accession}/{doc}"
    print(f"  [edgar] {ticker}: {name} -> {url}")

    full = _html_to_text(_get(url).text)
    narrative = extract_narrative(full)
    if narrative:
        text = narrative
        print(f"  [edgar] {ticker}: extracted narrative sections ({len(text):,} chars)")
    else:
        text = full[:MAX_FILING_CHARS]
        print(f"  [edgar] {ticker}: section parse failed, used raw slice")

    cache.write_text(text, encoding="utf-8")
    name_cache.write_text(name, encoding="utf-8")
    return name, text


if __name__ == "__main__":
    # Quick manual test: python -m src.edgar AAPL
    import sys

    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    company, body = fetch_10k_text(tk)
    print(f"{company}: {len(body):,} chars")
    print(body[:500], "...")
