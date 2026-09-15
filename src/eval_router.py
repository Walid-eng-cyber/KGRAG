"""Measure the router on a labeled question set — does it pick the right path?

Each question is tagged with the acceptable path(s). We run the classifier (not
full retrieval) and compare the routed path to the label. Reports per-question
results, overall accuracy, and a confusion matrix. This is the routing gate
flagged in the Phase 3 doc.

Usage:  python -m src.eval_router
"""
from __future__ import annotations

from collections import Counter

import config
from src.router import Path, classify

G, V, H = {"GRAPH"}, {"VECTOR"}, {"GRAPH", "HYBRID"}

# (question, acceptable paths). Hybrid/multi-hop accept GRAPH or HYBRID.
QUESTIONS: list[tuple[str, set]] = [
    # GRAPH — aggregations & counts
    ("How many subsidiaries does Microsoft have?", G),
    ("How many subsidiaries does Apple have?", G),
    ("How many subsidiaries does NVIDIA have?", G),
    ("How many companies face supply chain risk?", G),
    ("Which companies face competition risk?", G),
    ("How many risks does Apple disclose?", G),
    ("Which companies are regulated by the SEC?", G),
    ("How many companies are in the knowledge base?", G),
    # GRAPH — lists
    ("What are Apple's subsidiaries?", G),
    ("What are Microsoft's subsidiaries?", G),
    ("What are NVIDIA's subsidiaries?", G),
    ("What risks does Apple face?", G),
    ("What risks does Microsoft face?", G),
    ("What risks does NVIDIA face?", G),
    ("Who are Apple's executives?", G),
    ("Who regulates Apple?", G),
    ("Who are Apple's competitors?", G),
    # GRAPH — comparisons
    ("Compare the subsidiaries of Apple and Microsoft.", G),
    ("Compare the number of subsidiaries of Apple and NVIDIA.", G),
    ("Compare Microsoft and NVIDIA by subsidiary count.", G),
    ("Which has more subsidiaries, Apple or Microsoft?", G),
    # GRAPH — connections & multi-hop
    ("How is Apple connected to Taiwan?", G),
    ("What is Apple connected to?", G),
    ("How is Apple connected to its contract manufacturers?", G),
    ("What is Microsoft connected to?", G),
    ("How is NVIDIA connected to its suppliers?", H),
    # VECTOR — definitions
    ("What is a reportable business segment?", V),
    ("What is a contract manufacturer?", V),
    ("What is supply chain concentration?", V),
    ("What does 'material adverse effect' mean in a 10-K?", V),
    # VECTOR — explanations
    ("What does Apple say about supply chain risk?", V),
    ("What does Microsoft say about competition?", V),
    ("What does NVIDIA say about its data center business?", V),
    ("What does Apple say about geopolitical risk?", V),
    ("How does Microsoft describe its business segments?", V),
    ("What does NVIDIA say about GPU-powered AI?", V),
    ("What does Apple say about industrial accidents at suppliers?", V),
    ("How does Microsoft describe Microsoft 365?", V),
    ("What does NVIDIA say about competitors with fabrication facilities?", V),
    ("What does Apple say about macroeconomic conditions?", V),
    # VECTOR — single facts / policy
    ("Who is Apple's CEO?", V),
    ("Where is Apple headquartered?", V),
    ("What products does Apple sell?", V),
    ("What is Apple's fiscal year?", V),
    ("What tariffs does Apple mention?", V),
    # HYBRID — needs both
    ("What risks do Apple's suppliers face?", H),
    ("What risks do companies regulated by the SEC face?", H),
    ("Which of Apple's competitors face similar risks?", H),
    ("What business does NVIDIA do and what risks does it face?", H),
    ("How is Microsoft's business structured and what threatens it?", H),
]


def routed_path(question: str) -> tuple[str, str, float]:
    p, conf = classify(question)
    low = conf < config.ROUTER_CONFIDENCE_THRESHOLD
    used = "HYBRID" if (p == Path.HYBRID or low) else p.value
    return used, p.value, round(conf, 2)


def main() -> None:
    correct = 0
    confusion: Counter = Counter()
    print(f"Routing eval — {len(QUESTIONS)} questions "
          f"(router={config.ROUTER_MODEL}, threshold={config.ROUTER_CONFIDENCE_THRESHOLD})\n")
    for i, (q, ok) in enumerate(QUESTIONS, 1):
        used, cls, conf = routed_path(q)
        hit = used in ok
        correct += hit
        exp = "/".join(sorted(ok))
        confusion[(exp, used)] += 1
        print(f"{i:>2} [{'OK ' if hit else 'MISS'}] used={used:<6} "
              f"(cls={cls},{conf})  exp={exp:<12} | {q[:52]}")

    print(f"\nAccuracy: {correct}/{len(QUESTIONS)} = {correct/len(QUESTIONS):.1%}")
    print("\nConfusion (expected -> used : count):")
    for (exp, used), n in sorted(confusion.items()):
        flag = "" if used in set(exp.split("/")) else "  <-- misroute"
        print(f"  {exp:<14} -> {used:<6} : {n}{flag}")


if __name__ == "__main__":
    main()
