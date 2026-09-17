"""Phase 5 — benchmark GraphRAG (ours) vs plain vector RAG, and publish the delta.

The thesis: our system should roughly TIE plain vector RAG on simple semantic
lookups, WIN clearly on aggregation and multi-hop questions (the graph's home
turf), and both should REFUSE out-of-scope questions (grounding).

The question set is stratified by difficulty:
  single_hop   — one direct semantic/factual lookup (vector-friendly)
  two_hop      — one relationship traversal
  three_hop    — two relationship traversals
  aggregation  — counting / comparison over relationships
  out_of_scope — the system SHOULD refuse (not in the corpus)

Both systems are grounded (answer only from retrieved evidence) so the comparison
is fair — the delta reflects capability, not one system hallucinating more.

Usage:
  python -m src.benchmark            # run the demo subset (fast)
  python -m src.benchmark --all      # run the full set (long batch)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from time import perf_counter

import ollama

import config
from src.answer import answer
from src.vectorize import retrieve

# --- Scoring expectations -----------------------------------------------------
# type: number | contains_any | contains_all | refuse
Q: list[dict] = [
    # ---------- single_hop (vector-friendly; expect a TIE) ----------
    dict(id=1, cat="single_hop", q="What does Apple say about supply chain risk?",
         exp=dict(type="contains_any", v=["supply chain", "supplier", "accident"])),
    dict(id=2, cat="single_hop", q="What does NVIDIA say about its data center business?",
         exp=dict(type="contains_any", v=["data center", "gpu", "ai", "accelerat"])),
    dict(id=3, cat="single_hop", q="What products does Apple sell?",
         exp=dict(type="contains_any", v=["iphone", "mac", "ipad", "wearable"])),
    dict(id=4, cat="single_hop", q="What does Microsoft say about competition?",
         exp=dict(type="contains_any", v=["compet"])),
    dict(id=5, cat="single_hop", q="What does Apple say about geopolitical risk?",
         exp=dict(type="contains_any", v=["geopolit", "tension", "conflict", "tariff"])),
    dict(id=6, cat="single_hop", q="What does NVIDIA say about GPU-powered AI?",
         exp=dict(type="contains_any", v=["gpu", "ai"])),
    dict(id=7, cat="single_hop", q="How does Microsoft describe Microsoft 365?",
         exp=dict(type="contains_any", v=["365", "productivity", "cloud", "office"])),
    dict(id=8, cat="single_hop", q="What does Apple say about macroeconomic conditions?",
         exp=dict(type="contains_any", v=["economic", "macro"])),
    dict(id=9, cat="single_hop", q="What does NVIDIA say about competitors with fabrication facilities?",
         exp=dict(type="contains_any", v=["fabricat", "compet"])),
    dict(id=10, cat="single_hop", q="What does Apple say about industrial accidents at suppliers?",
         exp=dict(type="contains_any", v=["accident", "injur", "supplier"])),

    # ---------- aggregation (graph's home turf; expect ours to WIN) ----------
    dict(id=11, cat="aggregation", q="How many subsidiaries does Microsoft have?",
         exp=dict(type="number", v=7)),
    dict(id=12, cat="aggregation", q="How many subsidiaries does Apple have?",
         exp=dict(type="number", v=12)),
    dict(id=13, cat="aggregation", q="How many subsidiaries does NVIDIA have?",
         exp=dict(type="number", v=4)),
    dict(id=14, cat="aggregation", q="Which has more subsidiaries, Apple or Microsoft?",
         exp=dict(type="contains_any", v=["apple"])),
    dict(id=15, cat="aggregation", q="Which company has the most subsidiaries?",
         exp=dict(type="contains_any", v=["apple"])),
    dict(id=16, cat="aggregation", q="Which has fewer subsidiaries, NVIDIA or Microsoft?",
         exp=dict(type="contains_any", v=["nvidia"])),
    dict(id=17, cat="aggregation", q="Compare the subsidiary counts of Apple and Microsoft.",
         exp=dict(type="contains_all", v=["12", "7"])),
    dict(id=18, cat="aggregation", q="Does Apple have more subsidiaries than NVIDIA?",
         exp=dict(type="contains_any", v=["yes", "more", "apple"])),
    dict(id=19, cat="aggregation", q="Rank Apple, Microsoft and NVIDIA by subsidiary count.",
         exp=dict(type="contains_all", v=["apple", "microsoft", "nvidia"])),
    dict(id=20, cat="aggregation", q="How many companies in the data are regulated by the SEC?",
         exp=dict(type="number", v=1)),

    # ---------- two_hop (one traversal; graph-favored) ----------
    dict(id=21, cat="two_hop", q="What risks do Apple's suppliers face?",
         exp=dict(type="contains_any", v=["risk", "supply", "disrupt", "accident"])),
    dict(id=22, cat="two_hop", q="Which risks are faced by companies regulated by the SEC?",
         exp=dict(type="contains_any", v=["risk", "competition", "economic", "supply"])),
    dict(id=23, cat="two_hop", q="What are the subsidiaries of the company Tim Cook leads?",
         exp=dict(type="contains_any", v=["beats", "subsidiary", "apple"])),
    dict(id=24, cat="two_hop", q="Who are the executives of the company headquartered in Cupertino?",
         exp=dict(type="contains_any", v=["cook", "executive"])),

    # ---------- three_hop (two traversals; hardest) ----------
    dict(id=25, cat="three_hop",
         q="What risks are faced by the subsidiaries of the company Tim Cook leads?",
         exp=dict(type="contains_any", v=["risk", "supply", "competition"])),
    dict(id=26, cat="three_hop",
         q="Which regulators oversee the companies that compete with Apple?",
         exp=dict(type="contains_any", v=["sec"])),

    # ---------- out_of_scope (should REFUSE) ----------
    dict(id=31, cat="out_of_scope", q="What is the capital of France?", exp=dict(type="refuse")),
    dict(id=32, cat="out_of_scope", q="What was Tesla's 2023 revenue?", exp=dict(type="refuse")),
    dict(id=33, cat="out_of_scope", q="Who is the CEO of Google?", exp=dict(type="refuse")),
    dict(id=34, cat="out_of_scope", q="What is Apple's current stock price?", exp=dict(type="refuse")),
    dict(id=35, cat="out_of_scope", q="Who won the 2022 World Cup?", exp=dict(type="refuse")),
    dict(id=36, cat="out_of_scope", q="What was Amazon's net income last year?", exp=dict(type="refuse")),
    dict(id=37, cat="out_of_scope", q="Summarize the plot of Star Wars.", exp=dict(type="refuse")),
    dict(id=38, cat="out_of_scope", q="What is the price of Bitcoin today?", exp=dict(type="refuse")),
    dict(id=39, cat="out_of_scope", q="Who is the Chancellor of Germany?", exp=dict(type="refuse")),
    dict(id=40, cat="out_of_scope", q="What is Microsoft's dividend yield right now?", exp=dict(type="refuse")),
]

# A representative subset for a fast live run (2-3 per category).
DEMO_IDS = [1, 3, 11, 12, 14, 17, 21, 31, 32, 34]

_REFUSE = re.compile(
    r"(don'?t have|do not have|not enough|no (?:relevant )?information"
    r"|cannot (?:provide|answer|find)|can'?t (?:provide|answer|find)"
    r"|could(?:n'?t| not) find|do(?:es)? not contain|isn'?t (?:any )?information"
    r"|no information (?:about|on|regarding)|unable to|not covered|insufficient"
    r"|not (?:mentioned|available|provided|found|in the)|out of scope|no data)", re.I)


def score(ans: str, exp: dict) -> bool:
    a = ans.lower()
    t = exp["type"]
    if t == "refuse":
        return bool(_REFUSE.search(ans))
    # a non-refusal that actually answered
    if _REFUSE.search(ans) and t != "contains_any":
        pass
    if t == "number":
        return re.search(rf"\b{exp['v']}\b", ans) is not None
    if t == "contains_any":
        return any(s.lower() in a for s in exp["v"])
    if t == "contains_all":
        return all(s.lower() in a for s in exp["v"])
    return False


# --- Baseline: plain vector RAG (no graph, no router) -------------------------
def vanilla_rag(question: str, k: int = 5) -> str:
    hits = retrieve(question, k)
    if not hits:
        return "I don't have enough information."
    ctx = "\n".join(f"[{i+1}] {' '.join(h['text'].split())[:400]}"
                    for i, h in enumerate(hits))
    system = ("Answer the question using ONLY the passages below. If they do not "
              "contain the answer, reply exactly: \"I don't have enough "
              "information.\" Be concise.")
    resp = ollama.Client(host=config.OLLAMA_BASE_URL).chat(
        model=config.ANSWER_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Question: {question}\n\nPassages:\n{ctx}"}])
    return resp["message"]["content"].strip()


def run(ids: list[int]) -> None:
    items = [x for x in Q if x["id"] in ids]
    by_cat_ours: dict = defaultdict(lambda: [0, 0])   # [correct, total]
    by_cat_van: dict = defaultdict(lambda: [0, 0])
    lat_ours, lat_van = [], []
    rows = []

    for it in items:
        t0 = perf_counter(); ours_ans = answer(it["q"])["answer"]; ours_lat = perf_counter() - t0
        t0 = perf_counter(); van_ans = vanilla_rag(it["q"]); van_lat = perf_counter() - t0
        o = score(ours_ans, it["exp"]); v = score(van_ans, it["exp"])
        by_cat_ours[it["cat"]][0] += o; by_cat_ours[it["cat"]][1] += 1
        by_cat_van[it["cat"]][0] += v;  by_cat_van[it["cat"]][1] += 1
        lat_ours.append(ours_lat); lat_van.append(van_lat)
        rows.append(dict(id=it["id"], cat=it["cat"], q=it["q"], ours=o, vanilla=v,
                         ours_lat=round(ours_lat, 1), van_lat=round(van_lat, 1),
                         ours_ans=ours_ans, van_ans=van_ans))
        print(f"#{it['id']:>2} [{it['cat']:<12}] ours={'OK' if o else 'X '} "
              f"vanilla={'OK' if v else 'X '}  ({ours_lat:.0f}s/{van_lat:.0f}s) | {it['q'][:44]}")

    print("\n" + "=" * 60)
    print(f"{'tier':<14}{'ours':>11}{'vanilla':>11}{'delta':>8}")
    print("-" * 60)
    cats = ["single_hop", "two_hop", "three_hop", "aggregation", "out_of_scope"]
    to, tv, tn = 0, 0, 0
    for c in cats:
        if by_cat_ours[c][1] == 0:
            continue
        oc, on = by_cat_ours[c]; vc, vn = by_cat_van[c]
        to += oc; tv += vc; tn += on
        op, vp = oc / on, vc / vn
        print(f"{c:<14}{f'{oc}/{on} ({op:.0%})':>11}{f'{vc}/{vn} ({vp:.0%})':>11}"
              f"{f'{(op-vp)*100:+.0f}pp':>8}")
    print("-" * 60)
    op, vp = to / tn, tv / tn
    print(f"{'OVERALL':<14}{f'{to}/{tn} ({op:.0%})':>11}{f'{tv}/{tn} ({vp:.0%})':>11}"
          f"{f'{(op-vp)*100:+.0f}pp':>8}")

    def med(xs): xs = sorted(xs); return xs[len(xs)//2] if xs else 0
    print(f"\nLatency/query (local):  ours median {med(lat_ours):.0f}s "
          f"(mean {sum(lat_ours)/len(lat_ours):.0f}s)  |  vanilla median {med(lat_van):.0f}s "
          f"(mean {sum(lat_van)/len(lat_van):.0f}s)")
    print("Model calls/query:      ours ~3-5 (router + [graph plan] + answer + citation "
          "checks)  |  vanilla 2 (embed + answer)")

    out = {"rows": rows,
           "summary": {"ours_median_s": round(med(lat_ours), 1),
                       "van_median_s": round(med(lat_van), 1),
                       "n": len(items)}}
    with open(config.DATA_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved detail to data/benchmark_results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="run the full set (slow)")
    args = p.parse_args()
    run([x["id"] for x in Q] if args.all else DEMO_IDS)
