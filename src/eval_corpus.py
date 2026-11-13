"""Evaluate one corpus against a gold question set: our system vs plain vector RAG.

The gold set (JSON, kept in data/evals/ because it holds the corpus owner's
personal data) lists questions, expected answers and how to score each one.
Dates are pinned to the gold set's "today" so age/duration answers stay stable.

  python -m src.eval_corpus data/evals/walid_gold.json            # both systems
  python -m src.eval_corpus data/evals/walid_gold.json --only ours --ids 1-5,26

Every answer is appended to data/evals/results/<run>.jsonl as it finishes, and a
summary (accuracy per category, citations, tools, latency) is printed and saved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_REFUSE = re.compile(
    r"(don'?t have enough information|not enough information|no information|insufficient|"
    r"not (?:mentioned|available|provided|found|stated|specified|listed|included|contain)|"
    r"(?:do|does) not (?:contain|mention|say|include|state|provide)|no (?:grade|data|record|mention)|"
    r"cannot (?:find|answer|determine)|can'?t (?:find|answer|determine)|unable to|"
    r"keine (?:information|angabe|note)|nicht (?:genannt|angegeben|enthalten|erwähnt|verfügbar)|"
    r"not yet (?:graded|been graded)|hasn'?t been graded)", re.I)

_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "null": 0, "eins": 1, "ein": 1, "eine": 1, "zwei": 2,
          "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11}


def _clean(ans: str) -> str:
    return re.sub(r"\[(?:G|P|C)\d+\]", " ", ans or "")


def _numbers(ans: str) -> set[int]:
    """Standalone numbers, not parts of dates (06.05.2001), decimals (2,6) or tags."""
    a = _clean(ans)
    a = re.sub(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", " ", a)
    out = {int(n) for n in re.findall(r"(?<![\d.,])\d+(?![.,]\d)", a)}
    out |= {_WORDS[w] for w in re.findall(r"[a-zäöüß]+", a.lower()) if w in _WORDS}
    return out


def _has(text: str, term) -> bool:
    """term is a string or a list of variants; whole-word-ish, case-insensitive."""
    variants = term if isinstance(term, list) else [term]
    t = text.lower()
    return any(re.search(rf"(?<![\w]){re.escape(v.lower())}", t) for v in variants)


def _duration(ans: str) -> tuple[int, int] | None:
    a = _clean(ans).lower()
    y = re.search(r"(\d+)\s*(?:years?|jahre?n?|ans?)\b", a)
    m = re.search(r"(\d+)\s*(?:months?|monate?n?|mois)\b", a)
    if not y and not m:
        return None
    return (int(y[1]) if y else 0, int(m[1]) if m else 0)


def score(ans: str, spec: dict) -> bool:
    t, a = spec["type"], _clean(ans)
    refused = bool(_REFUSE.search(a))
    if t == "refuse":
        return refused
    if any(_has(a, f) for f in spec.get("forbid", [])):
        return False
    if t == "number":
        return spec["v"] in _numbers(a)
    if t == "contains_any":
        return any(_has(a, v) for v in spec["v"])
    if t in ("contains_all", "set"):
        return all(_has(a, v) for v in spec["v"])
    if t == "duration":
        got = _duration(a)
        if got is not None and got[0] * 12 + got[1] == spec["years"] * 12 + spec["months"]:
            return True
        # the same period stated exactly in days ("30 days") is right too, when the gold set gives the days
        days = re.search(r"(\d+)\s*(?:days?|tage?n?|jours?)\b", a.lower())
        return bool(spec.get("days") and days and int(days[1]) == spec["days"])
    raise ValueError(f"unknown score type {t!r}")


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def ours(question: str, corpus: dict, today: date) -> dict:
    from src.answer import answer
    r = answer(question, corpus, today)
    g = r.get("graph_result") or {}
    return {"answer": r["answer"], "citations_valid": r.get("citations_valid"), "tools": r.get("tools_used", []),
            "template": g.get("query_type"), "graph_error": g.get("error"), "path": r.get("path_used"),
            "attempts": r.get("attempts")}


def baseline(question: str, corpus: dict, today: date, k: int = 5) -> dict:
    """Plain vector RAG over the same corpus: top-k passages + the same answer model.
    It gets today's date too, so the comparison is about retrieval and tools, not the calendar."""
    import config
    from src.llm import chat_client
    from src.vectorize import retrieve
    hits = retrieve(question, k, corpus["id"], hybrid=False)
    if not hits:
        return {"answer": "I don't have enough information."}
    ctx = "\n".join(f"[{i + 1}] {' '.join(h['text'].split())[:1200]}" for i, h in enumerate(hits))
    system = (f"Today is {today:%d %B %Y}. Answer the question using ONLY the passages below. If they do not "
              "contain the answer, reply exactly: \"I don't have enough information.\" Be concise.")
    resp = chat_client("eval").chat(
        model=config.ANSWER_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Question: {question}\n\nPassages:\n{ctx}"}])
    return {"answer": resp["message"]["content"].strip()}


SYSTEMS = {"ours": ours, "baseline": baseline}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _ids(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def summarize(rows: list[dict], systems: list[str]) -> dict:
    cats: dict = defaultdict(lambda: {s: [0, 0] for s in systems})
    for r in rows:
        c = cats[r["cat"]][r["system"]]
        c[0] += r["correct"]
        c[1] += 1
    summary = {"by_category": {k: {s: f"{v[s][0]}/{v[s][1]}" for s in systems} for k, v in cats.items()}}
    for s in systems:
        mine = [r for r in rows if r["system"] == s]
        if not mine:
            continue
        lat = sorted(r["latency_s"] for r in mine)
        summary[s] = {"correct": sum(r["correct"] for r in mine), "total": len(mine),
                      "accuracy": round(sum(r["correct"] for r in mine) / len(mine), 3),
                      "median_latency_s": lat[len(lat) // 2]}
        if s == "ours":
            summary[s]["citations_valid"] = sum(bool(r.get("citations_valid")) for r in mine)
            summary[s]["tool_runs"] = sum(bool(r.get("tools")) for r in mine)
    return summary


def print_summary(summary: dict, systems: list[str]) -> None:
    print("\n" + "=" * 64)
    print(f"{'category':<14}" + "".join(f"{s:>14}" for s in systems))
    print("-" * 64)
    for cat, v in summary["by_category"].items():
        print(f"{cat:<14}" + "".join(f"{v[s]:>14}" for s in systems))
    print("-" * 64)
    cells = []
    for s in systems:
        t = summary.get(s)
        cells.append(f"{t['correct']}/{t['total']} ({t['accuracy']:.0%})" if t else "-")
    print(f"{'OVERALL':<14}" + "".join(f"{c:>14}" for c in cells))
    for s in systems:
        if s in summary:
            extra = (f", citations valid {summary[s]['citations_valid']}/{summary[s]['total']}, "
                     f"tools used on {summary[s]['tool_runs']}") if s == "ours" else ""
            print(f"{s}: median {summary[s]['median_latency_s']}s per question{extra}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--only", choices=list(SYSTEMS))
    ap.add_argument("--ids", help="e.g. 1-5,26")
    args = ap.parse_args(argv)

    from server.corpora import CorpusStore
    import os
    gold = json.loads(Path(args.gold).read_text())
    corpus = CorpusStore(os.environ.get("DOCUMENTS_DB", "data/documents.db")).get(gold["corpus"])
    if not corpus:
        sys.exit(f"corpus {gold['corpus']} not found")
    today = date.fromisoformat(gold["today"])
    systems = [args.only] if args.only else list(SYSTEMS)
    wanted = _ids(args.ids)
    items = [q for q in gold["questions"] if wanted is None or q["id"] in wanted]

    out_dir = Path(args.gold).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run = f"{Path(args.gold).stem}-{datetime.now():%Y%m%d-%H%M%S}"
    log = out_dir / f"{run}.jsonl"
    print(f"{len(items)} questions × {len(systems)} system(s) on {corpus['name']} as of {today} → {log}", flush=True)

    rows = []
    for it in items:
        for s in systems:
            t0 = perf_counter()
            try:
                res = SYSTEMS[s](it["q"], corpus, today)
            except Exception as e:  # noqa: BLE001 — record and keep going
                res = {"answer": "", "error": f"{type(e).__name__}: {str(e)[:120]}"}
            lat = round(perf_counter() - t0, 1)
            ok = score(res.get("answer", ""), it["score"])
            row = {"id": it["id"], "cat": it["cat"], "q": it["q"], "expected": it["expected"], "system": s,
                   "correct": ok, "latency_s": lat, **res}
            rows.append(row)
            with log.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"#{it['id']:>2} {s:<8} {'OK' if ok else 'X '} {lat:>5.0f}s [{it['cat']}] {it['q'][:50]}", flush=True)

    summary = summarize(rows, systems)
    (out_dir / f"{run}.summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary, systems)


if __name__ == "__main__":
    main()
