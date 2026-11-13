"""Deterministic tools the answer step runs before the model writes.

Each tool turns evidence (graph rows, passages) into exact, citable COMPUTED
FACTS ([C1], [C2], ...): ages, durations, timelines, totals. The model only
phrases them. Which tools a question gets is decided by:
  - the corpus's skill (src/skills.py): which tools make sense for its documents
  - each tool's trigger: words in the question, in English, German and French
Tools are plain Python over the evidence: no code from the model, no network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

from src.tools import dates, numbers

Evidence = list[tuple[str, str]]          # (tag, text)
Fact = tuple[str, str]                    # (statement, the source tag it is based on)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    trigger: re.Pattern
    run: Callable[[str, Evidence, dict, date], list[Fact]]


def _rx(s: str) -> re.Pattern:
    return re.compile(s, re.I)


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("age", "Age from a birth date, as of today",
         _rx(r"\b(how old|age of|his age|her age|their age|wie alt|alter von|quel âge)\b"), dates.age),
    Tool("duration", "How long a job, degree, project or other period lasted (or has lasted so far)",
         _rx(r"\b(how long|how many (years|months|weeks|days)|duration|wie lange|wie viele (jahre|monate|wochen|tage)|"
             r"dauer|combien de temps|depuis quand|since when|seit wann)\b"), dates.duration),
    Tool("timeline", "Periods in chronological order: first, latest, current",
         re.compile("|".join(r.pattern for r in (dates._CURRENT, dates._LATEST, dates._FIRST)) +
                    r"|\b(timeline|chronolog\w*|in order|order of|reihenfolge|werdegang|career path|parcours)\b", re.I),
         dates.timeline),
    Tool("experience", "Total work experience, overlaps counted once",
         _rx(r"\b(years? of (work )?experience|how much (work )?experience|berufserfahrung|"
             r"wie viel erfahrung|expérience)\b"), dates.experience),
    Tool("aggregate", "Total, average, highest or lowest of numbers in the graph results",
         _rx(r"\b(total|sum|altogether|combined|average|mean|highest|largest|biggest|most|lowest|smallest|least|"
             r"insgesamt|gesamt|durchschnitt\w*|größte|höchste|kleinste|niedrigste)\b"), numbers.aggregate),
]}


# Tools that need every dated passage of a corpus, not just the top-k similar ones,
# with a POSIX regex that finds those chunks in Postgres.
CORPUS_SCAN = {
    "age": r"geboren|geburtsdatum|birth|born|naissance",
    "duration": r"(19|20)[0-9]{2}",
    "timeline": r"(19|20)[0-9]{2}",
    "experience": r"(19|20)[0-9]{2}",
}


def scan_patterns(question: str, allowed: list[str]) -> list[str]:
    """The corpus-scan regexes for the allowed tools this question triggers."""
    return list(dict.fromkeys(CORPUS_SCAN[n] for n in allowed
                              if n in CORPUS_SCAN and n in TOOLS and TOOLS[n].trigger.search(question)))


def run_tools(question: str, evidence: Evidence, graph: dict | None, allowed: list[str],
              today: date | None = None) -> tuple[list[Fact], list[str]]:
    """Run every allowed tool whose trigger matches. Returns (facts, names of the tools that produced them)."""
    today = today or date.today()
    facts, used = [], []
    for name in allowed:
        tool = TOOLS.get(name)
        if not tool or not tool.trigger.search(question):
            continue
        try:
            out = tool.run(question, evidence, graph or {}, today)
        except Exception:  # noqa: BLE001 — a tool must never break an answer
            out = []
        if out:
            facts += out
            used.append(name)
    return list(dict.fromkeys(facts)), used
