"""Number tools: totals, averages, highest and lowest over graph results.

Graph templates return rows (trades, counts, values). The model is asked to
read numbers, never to add or compare them; this does the arithmetic.
"""
from __future__ import annotations

import re

_LABEL_KEYS = ("name", "company", "insider", "entity", "other", "label", "ticker", "title")
_MAX = re.compile(r"\b(highest|largest|biggest|most|maximum|max|top|größte|höchste|meiste|plus grand)\b", re.I)
_MIN = re.compile(r"\b(lowest|smallest|least|fewest|minimum|min|kleinste|niedrigste|wenigste|plus petit)\b", re.I)
_SUM = re.compile(r"\b(total|sum|in total|altogether|combined|insgesamt|gesamt|summe|au total)\b", re.I)
_AVG = re.compile(r"\b(average|mean|avg|typical|durchschnitt\w*|moyenne)\b", re.I)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 or x == int(x) else f"{x:,.2f}"


def aggregate(question, evidence, graph, today):
    rows = (graph or {}).get("rows") or []
    if len(rows) < 2:
        return []
    fields = [k for k in rows[0] if not k.startswith("_") and all(_num(r.get(k)) is not None for r in rows)]
    label_key = next((k for k in _LABEL_KEYS if isinstance(rows[0].get(k), str)), None)
    if not fields:
        return []
    want_sum, want_avg = bool(_SUM.search(question)), bool(_AVG.search(question))
    want_max, want_min = bool(_MAX.search(question)), bool(_MIN.search(question))
    out = []
    for f in fields[:4]:
        vals = [_num(r[f]) for r in rows]
        name = f.replace("_", " ")
        parts = []
        if want_sum:
            parts.append(f"total {_fmt(sum(vals))}")
        if want_avg:
            parts.append(f"average {_fmt(sum(vals) / len(vals))}")
        for wanted, pick, word in ((want_max, max, "highest"), (want_min, min, "lowest")):
            if wanted:
                i = vals.index(pick(vals))
                who = f" ({rows[i][label_key]})" if label_key else ""
                parts.append(f"{word} {_fmt(vals[i])}{who}")
        if parts:
            out.append((f"Over the {len(rows)} graph results, {name}: {', '.join(parts)}.", "G1"))
    return out
