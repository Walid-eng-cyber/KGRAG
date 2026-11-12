"""Benchmark data for the UI: computed from data/benchmark_results.json when a
run exists, otherwise the published results from the README."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import config

CATEGORIES = [
    ("single_hop", "Single hop"),
    ("two_hop", "Two hop"),
    ("three_hop", "Three hop"),
    ("aggregation", "Aggregation"),
    ("out_of_scope", "Out of scope"),
]

# Published numbers (README, n = 36). Used until `python -m src.benchmark --all`
# has written data/benchmark_results.json.
# Per-tier n is the only split consistent with the published percentages and
# overall 28/36 vs 20/36.
PUBLISHED = {
    "single_hop": (0.90, 0.70, 10), "two_hop": (0.25, 0.50, 4),
    "three_hop": (1.00, 0.00, 2), "aggregation": (0.60, 0.10, 10),
    "out_of_scope": (1.00, 1.00, 10),
}
PUBLISHED_OVERALL = (0.78, 0.56, 36)
PUBLISHED_LATENCY = (25.0, 8.0)

COSTS = [
    {"label": "One-time ingestion (27 chunks)", "vector": "11.5 s (embed only)", "graphrag": "~6–7 min"},
    {"label": "Latency / query (median, local)", "vector": None, "graphrag": None},
    {"label": "Model calls / query", "vector": "2", "graphrag": "3–5"},
    {"label": "API cost", "vector": "$0", "graphrag": "$0"},
]


def _published() -> dict:
    cats = [{"key": k, "label": lbl, "graphrag": PUBLISHED[k][0], "vector": PUBLISHED[k][1],
             "n": PUBLISHED[k][2]} for k, lbl in CATEGORIES]
    g, v, n = PUBLISHED_OVERALL
    return {"source": "published", "categories": cats,
            "overall": {"graphrag": g, "vector": v, "n": n},
            "latency": {"graphrag_median_s": PUBLISHED_LATENCY[0],
                        "vector_median_s": PUBLISHED_LATENCY[1]}}


def _from_results(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # ours, vanilla, total
    for r in data.get("rows", []):
        t = tally[r["cat"]]
        t[0] += bool(r["ours"]); t[1] += bool(r["vanilla"]); t[2] += 1
    cats, to, tv, tn = [], 0, 0, 0
    for k, lbl in CATEGORIES:
        o, v, n = tally.get(k, [0, 0, 0])
        if n:
            cats.append({"key": k, "label": lbl, "graphrag": o / n, "vector": v / n, "n": n})
            to, tv, tn = to + o, tv + v, tn + n
    s = data.get("summary", {})
    return {"source": "results-file", "categories": cats,
            "overall": {"graphrag": to / tn if tn else 0, "vector": tv / tn if tn else 0, "n": tn},
            "latency": {"graphrag_median_s": s.get("ours_median_s"),
                        "vector_median_s": s.get("van_median_s")}}


def benchmark_summary(path: Path | None = None) -> dict:
    path = path or (config.DATA_DIR / "benchmark_results.json")
    out = _from_results(path) if path.exists() else _published()
    lat = out["latency"]
    costs = [dict(c) for c in COSTS]
    costs[1]["vector"] = f"{lat['vector_median_s']:.0f} s" if lat["vector_median_s"] else "—"
    costs[1]["graphrag"] = f"{lat['graphrag_median_s']:.0f} s" if lat["graphrag_median_s"] else "—"
    out["costs"] = costs
    return out
