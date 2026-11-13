"""Skills: per-corpus playbooks for the answer step.

A skill picks the tools that fit a corpus's documents and adds a short rule
for how answers about them should read. It is chosen from the corpus's kind and
extraction preset, so a CV corpus gets age/duration/experience tools while the
SEC corpus gets totals and highest/lowest over trades.
"""
from __future__ import annotations

from src.tools import TOOLS

SKILLS: dict[str, dict] = {
    "career": {
        "label": "CV & career",
        "tools": ["age", "duration", "timeline", "experience", "aggregate"],
        "guidance": "These documents describe a person's education, jobs and projects. For ages, "
                    "durations, dates in order and years of experience, use the COMPUTED FACTS; "
                    "a planned or future position is not experience yet.",
    },
    "company": {
        "label": "Company filings",
        "tools": ["duration", "timeline", "aggregate"],
        "guidance": "These are company documents. Quote figures with their units and period "
                    "(fiscal year, quarter); never combine figures from different periods.",
    },
    "sec_insiders": {
        "label": "SEC insider trades",
        "tools": ["aggregate", "duration", "timeline"],
        "guidance": "These are SEC Forms 3/4/5. Distinguish open-market buys and sells from grants "
                    "and option exercises; give dollar values and share counts exactly as stated.",
    },
    "general": {
        "label": "General",
        "tools": list(TOOLS),
        "guidance": "",
    },
}


def skill_for(corpus: dict | None) -> dict:
    if not corpus:
        return SKILLS["company"]                 # the legacy filings graph
    if corpus.get("kind") == "sec_insiders":
        return SKILLS["sec_insiders"]
    return SKILLS.get((corpus.get("schema") or {}).get("preset", ""), SKILLS["general"])
