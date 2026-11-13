"""Date tools: age, duration of a period, timeline, total experience.

Documents write periods as "01.09.2019 – 01.06.2022", "09/2021 - heute",
"seit 15.02.2026" or "May 2020 to present", usually right after the title of the
job, degree or project. We parse them, attach the text just before each period as
its label, and do the date math here instead of in the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS.update({"sept": 9, "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "maerz": 3, "mai": 5, "juni": 6,
               "juli": 7, "oktober": 10, "okt": 10, "dezember": 12, "dez": 12, "janvier": 1, "février": 2,
               "mars": 3, "avril": 4, "juin": 6, "juillet": 7, "août": 8, "septembre": 9, "octobre": 10,
               "novembre": 11, "décembre": 12})

# ---------------------------------------------------------------- single dates

_NUM = re.compile(r"(?P<d>\d{1,2})[./-](?P<m>\d{1,2})[./-](?P<y>(?:19|20)\d{2})")
_ISO = re.compile(r"(?P<y>(?:19|20)\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})")
_WORDS = re.compile(r"(?:(?P<d1>\d{1,2})\.?\s+(?P<m1>[^\W\d_]+)|(?P<m2>[^\W\d_]+)\s+(?P<d2>\d{1,2}),?)\s+(?P<y>(?:19|20)\d{2})")


def parse_date(text: str) -> date | None:
    """A full date in `text` (06.05.2001, 2001-05-06, May 6, 2001, 6. Mai 2001). Day-first for numbers."""
    for rx in (_ISO, _NUM):
        m = rx.search(text)
        if m:
            try:
                return date(int(m["y"]), int(m["m"]), int(m["d"]))
            except ValueError:
                pass
    m = _WORDS.search(text)
    if m:
        month = MONTHS.get((m["m1"] or m["m2"]).lower())
        if month:
            try:
                return date(int(m["y"]), month, int(m["d1"] or m["d2"]))
            except ValueError:
                return None
    return None


def age_on(born: date, today: date) -> int:
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


_BIRTH = re.compile(r"(geboren(?:\s+am)?|geburtsdatum|date of birth|birth ?date|born(?:\s+on)?|d\.?o\.?b\.?|"
                    r"né(?:e)?\s+le|date de naissance)\s*[:,]?\s*(?P<d>[^\n;|]{6,30})", re.I)


def birth_dates(text: str) -> list[date]:
    out = []
    for m in _BIRTH.finditer(text):
        d = parse_date(m["d"])
        if d and d not in out:
            out.append(d)
    return out


# ---------------------------------------------------------------- periods

_D = (r"(?:\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}"            # 01.09.2019
      r"|\d{1,2}[./](?:19|20)\d{2}"                         # 09/2019
      r"|(?:19|20)\d{2}-\d{2}(?:-\d{2})?"                   # 2019-09(-01)
      r"|[^\W\d_]{3,9}\.?\s+(?:19|20)\d{2}"                 # Sep 2019, September 2019
      r"|(?:19|20)\d{2})")                                   # 2019
_OPEN = r"(?:heute|present|today|now|current(?:ly)?|aktuell|laufend|ongoing|jetzt|dato|aujourd'hui)"
_RANGE = re.compile(rf"(?P<a>{_D})\s*(?:[–—-]+|\bto\b|\bbis\b|\bau\b|\buntil\b)\s*(?P<b>{_D}|{_OPEN})", re.I)
_SINCE = re.compile(rf"\b(?:seit|since|depuis|ab)\s+(?P<a>{_D})", re.I)


def _point(s: str, end: bool) -> date | None:
    """A period boundary. Month-only dates take the 1st (start) or the last day (end)."""
    s = s.strip().rstrip(".")
    d = parse_date(s)
    if d:
        return d
    m = re.fullmatch(r"(\d{1,2})[./]((?:19|20)\d{2})", s) or re.fullmatch(r"((?:19|20)\d{2})-(\d{2})", s)
    if m:
        a, b = m.groups()
        y, mo = (int(b), int(a)) if len(a) <= 2 else (int(a), int(b))
    else:
        m = re.fullmatch(r"([^\W\d_]{3,9})\.?\s+((?:19|20)\d{2})", s)
        if m and m[1].lower() in MONTHS:
            y, mo = int(m[2]), MONTHS[m[1].lower()]
        elif re.fullmatch(r"(?:19|20)\d{2}", s):
            y, mo = int(s), 12 if end else 1
        else:
            return None
    if not 1 <= mo <= 12:
        return None
    if not end:
        return date(y, mo, 1)
    nxt = date(y + mo // 12, mo % 12 + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


@dataclass
class Period:
    label: str
    start: date
    end: date
    ongoing: bool
    tag: str
    planned: bool = False      # starts in the future ("ab 02/2027 (Zusage)")
    after: str = ""            # the text right after it (place, thesis title, ...): helps matching


def _label(before: str) -> str:
    """The title a period belongs to: the text right before it, trimmed to the last ~2 lines' worth."""
    t = re.sub(r"\s+", " ", before).strip(" |,;:–-●•")
    t = re.split(r"[●•]|(?<=[a-zäöüß]{3})\.\s", t)[-1]      # a sentence end, not "B.Sc. " or "Std./Woche"
    t = re.split(r"\b[A-ZÄÖÜ]{5,}\b(?:\s+[A-ZÄÖÜ]{3,}\b)*", t)[-1]   # a section heading (BERUFSERFAHRUNG)
    t = re.split(r"\b[A-Z][\w]+:\s(?=[^|]*\|)", t)[-1] if "|" in t else t   # "Technologien: A | B" lists
    if len(t) > 160:
        t = t[-160:].split(" ", 1)[-1]         # start at a whole word
    return t.lstrip(" |,;:–-").strip()


def periods(text: str, tag: str, today: date) -> list[Period]:
    out, last = [], 0
    hits = [(m.start(), m.end(), m["a"], m.groupdict().get("b")) for m in _RANGE.finditer(text)]
    taken = [(s, e) for s, e, *_ in hits]
    hits += [(m.start(), m.end(), m["a"], None) for m in _SINCE.finditer(text)
             if not any(s <= m.start() < e for s, e in taken)]
    hits.sort()
    for n, (s, e, a, b) in enumerate(hits):
        start = _point(a, end=False)
        open_ = b is None or re.fullmatch(_OPEN, b.strip(), re.I) is not None
        end = today if open_ else _point(b, end=True)
        if not start or not end:
            continue
        planned = start > today
        if end < start and not planned:
            continue
        nxt = hits[n + 1][0] if n + 1 < len(hits) else len(text)
        after = text[e:min(nxt, e + 80)]
        out.append(Period(_label(text[last:s]) or "(untitled)", start, max(end, start), open_ and not planned, tag,
                          planned, after))
        last = e
    return out


def span(start: date, end: date) -> str:
    e = date.fromordinal(end.toordinal() + 1)      # periods include their last day: 01.07–30.09 is 3 months
    months = max((e.year - start.year) * 12 + e.month - start.month - (e.day < start.day), 0)
    y, m = divmod(months, 12)
    parts = ([f"{y} year{'s' * (y != 1)}"] if y else []) + ([f"{m} month{'s' * (m != 1)}"] if m else [])
    return f"{' '.join(parts) or 'less than a month'} ({(end - start).days + 1} days)"


def _fmt(p: Period) -> str:
    end = "today (ongoing)" if p.ongoing else f"{p.end:%d %b %Y}"
    return f"{p.start:%d %b %Y} – {end}"


# Words that tie a question to a period's label.
# SYNONYMS name the same thing in other words or languages: as strong as a literal match.
# RELATED words only hint at it ("study" -> "university"): weak, so they decide only ties.
SYNONYMS = {
    "master": "m.sc msc mba masterstudium", "bachelor": "b.sc bsc licence licenciatura bachelorstudium",
    "phd": "doktor promotion ph.d doctorate", "doctorate": "phd ph.d promotion",
    "internship": "praktikum intern stage", "intern": "praktikum internship", "praktikum": "internship intern",
    "project": "projekt projet", "projekt": "project", "thesis": "arbeit bachelorarbeit masterarbeit",
    "job": "stelle", "company": "firma unternehmen", "school": "schule",
}
RELATED = {
    "study": "studium studied studies student university universität hochschule college b.sc m.sc degree",
    "studying": "studium student university universität hochschule college b.sc m.sc degree",
    "studied": "studium university universität hochschule college b.sc m.sc",
    "degree": "b.sc m.sc bachelor master diplom abschluss",
    "work": "praktikum werkstudent mitarbeiter intern internship job arbeit entwickler developer engineer employee",
    "worked": "praktikum werkstudent mitarbeiter intern internship job entwickler developer engineer",
    "working": "praktikum werkstudent mitarbeiter intern internship job entwickler developer engineer",
    "project": "github", "projekt": "github",
}
_STOP = set(("how long did does was were the and for with what when which many years months days his her their "
            "has have had take took last lasted wie lange hat war ist der die das ein eine sein ihre "
            "combien temps est son").split())


def _words(s: str) -> set[str]:
    out = set()
    for w in re.findall(r"[\w.]+", s.lower()):
        w = w.strip(".")
        w = re.sub(r"(?<=\w{3})'?s$", "", w) if w.endswith("'s") else w
        if len(w) >= 3 and w not in _STOP:
            out.add(w)
    return out


_ONGOING_Q = re.compile(r"\b(?:has|have)\s+(?:\w+\s+)?been\b|\bseit\b|\bschon\b|\bstill\b|\bso far\b|\bdepuis\b", re.I)


def _score(question: str, p: Period) -> int:
    """Each question word counts once: 3 for a literal or synonym match in the label,
    1 for a related word in the label or a literal match in the text after the period."""
    label, after = _words(p.label), _words(p.after)
    score = 2 if p.ongoing and _ONGOING_Q.search(question) else 0
    hit = lambda w, words: any(w == x or (len(w) >= 5 and (w in x or x in w)) for x in words)  # noqa: E731
    for w in _words(question):
        if hit(w, label) or label & set(SYNONYMS.get(w, "").split()):
            score += 3
        elif label & set(RELATED.get(w, "").split()):
            score += 1
        elif hit(w, after):
            score += 1
    return score


def all_periods(evidence: list[tuple[str, str]], today: date) -> list[Period]:
    """Periods from all evidence. The same period seen twice (overlapping chunks, two copies of a CV)
    is kept once, with the most complete label."""
    out: list[Period] = []
    for tag, text in evidence:
        for p in periods(text, tag, today):
            twin = next((i for i, q in enumerate(out) if (q.start, q.end) == (p.start, p.end) and
                         (q.label.endswith(p.label[-25:]) or p.label.endswith(q.label[-25:]))), None)
            if twin is None:
                out.append(p)
            elif len(p.label) > len(out[twin].label):
                out[twin] = p
    return out


# ---------------------------------------------------------------- tools

def age(question, evidence, graph, today):
    for tag, text in evidence:
        for born in birth_dates(text):
            if born <= today:
                return [(f"Born on {born:%d %B %Y} (per [{tag}]); as of today, {today:%d %B %Y}, "
                         f"that is {age_on(born, today)} years old.", tag)]
    return []


def duration(question, evidence, graph, today):
    ps = [p for p in all_periods(evidence, today) if not p.planned]
    if not ps:
        return []
    scored = sorted(((_score(question, p), i, p) for i, p in enumerate(ps)), key=lambda t: (-t[0], t[1]))
    best = [p for s, _, p in scored if s == scored[0][0] and s > 0][:3]   # no match: say nothing, don't guess
    return [(f"“{p.label}”: {_fmt(p)} (per [{p.tag}]) — a duration of {span(p.start, p.end)}.", p.tag) for p in best]


_CURRENT = re.compile(r"\b(current(ly)?|now|today|at the moment|presently|aktuell\w*|derzeit|zurzeit|jetzt|"
                      r"gerade|actuel\w*|en ce moment)\b", re.I)
_LATEST = re.compile(r"\b(latest|most recent|last|newest|zuletzt|letzte\w*|neueste\w*|dernier\w*)\b", re.I)
_FIRST = re.compile(r"\b(first|earliest|oldest|zuerst|erste\w*|früheste\w*|premier\w*)\b", re.I)
_NEXT = re.compile(r"\b(next|upcoming|future|planned|will (?:start|begin|join)|going to|about to|zukünftig\w*|"
                   r"künftig\w*|nächste\w*|geplant\w*|demnächst|bald|prochain\w*|futur\w*)\b", re.I)
_JOBQ = re.compile(r"\b(job|work\w*|employ\w*|position|role|arbeit\w*|stelle|beruf\w*|travail\w*|poste)\b", re.I)


def _item(p: Period) -> str:
    return f"{p.label} — {_fmt(p)}{' (planned)' if p.planned else ''} [{p.tag}]"


def _narrow(question: str, cands: list[Period]) -> list[Period]:
    """Keep the periods of the kind the question names ("first internship", "latest project")."""
    scores = [_score(question, p) for p in cands]
    if scores and max(scores) > 0:
        return [p for p, s in zip(cands, scores) if s == max(scores)]
    return cands


def timeline(question, evidence, graph, today):
    ps = sorted(all_periods(evidence, today), key=lambda p: (p.start, p.end))
    if _JOBQ.search(question):            # "current job" is about work, not the degree in progress
        ps = [p for p in ps if _WORK.search(p.label)] or ps
    if not ps:
        return []
    past = [p for p in ps if not p.planned]
    out = []
    if _NEXT.search(question):
        plans = _narrow(question, sorted((p for p in ps if p.planned), key=lambda p: p.start))
        out.append(("Next, planned and not started yet: " + "; ".join(map(_item, plans)) + "." if plans else
                    "Nothing in these documents is planned for the future (no period starts after today).",
                    (plans or ps)[0].tag))
    if _CURRENT.search(question):
        now = _narrow(question, [p for p in ps if p.ongoing])
        out.append((f"As of today, {today:%d %B %Y}, ongoing: " + ("; ".join(map(_item, now)) if now else
                    "nothing in these documents is marked as ongoing") + ".", (now or ps)[0].tag))
        plans = [p for p in ps if p.planned]
        if plans and not _NEXT.search(question):
            out.append(("Planned, not started yet: " + "; ".join(map(_item, plans)) + ".", plans[0].tag))
    if _LATEST.search(question) and past:
        p = max(_narrow(question, past), key=lambda p: (p.start, p.end))
        out.append((f"The most recently started: {_item(p)}.", p.tag))
    if _FIRST.search(question) and past:
        p = min(_narrow(question, past), key=lambda p: (p.start, p.end))
        out.append((f"The earliest: {_item(p)}.", p.tag))
    if not out and len(ps) >= 2:
        ps = _narrow(question, ps)
        out.append(("In chronological order: " + "; ".join(f"{i + 1}) {_item(p)}" for i, p in enumerate(ps)) + ".",
                    ps[0].tag))
    return out


# Job words. English ones must be whole words ("engineer" is not "Engineering von KI-Systemen");
# German ones may end a compound ("Softwareentwickler", "Werkstudentin").
_WORK = re.compile(r"(?:praktik|werkstudent|mitarbeiter|entwickler|hilfskraft|angestellt|berater|referent)\w*|"
                   r"\b(?:intern|internship|job|developer|engineer|employee|assistant|researcher|manager|lead|"
                   r"consultant|analyst|stelle|position|work|worked|working|stagiaire|stage)\b", re.I)


def experience(question, evidence, graph, today):
    ps = sorted((p for p in all_periods(evidence, today) if not p.planned and _WORK.search(p.label)),
                key=lambda p: p.start)
    if not ps:
        return []
    merged: list[list[date]] = []
    for p in ps:                       # union, so overlapping jobs aren't counted twice
        if merged and p.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], p.end)
        else:
            merged.append([p.start, p.end])
    days = sum((b - a).days + 1 for a, b in merged)
    y, rem = divmod(round(days / 30.4375), 12)
    listed = "; ".join(f"{p.label} ({_fmt(p)}) [{p.tag}]" for p in ps)
    return [(f"Work periods found: {listed}. Together, without double-counting overlaps, that is "
             f"{y} year{'s' * (y != 1)} {rem} month{'s' * (rem != 1)} ({days} days) of work experience "
             f"as of {today:%d %B %Y}.", ps[0].tag)]


_YEAR_Q = re.compile(r"\b(?:in|during|im|en|seit|since|bis|until|before|after|vor|nach|von)\s+((?:19|20)\d{2})\b", re.I)


def in_year(question, evidence, graph, today):
    """What a person was doing in a given year: the periods that overlap it, or that none do."""
    years = sorted({int(y) for y in _YEAR_Q.findall(question)})
    if not years:
        return []
    ps = all_periods(evidence, today)
    if not ps:
        return []
    out = []
    for y in years:
        hits = [p for p in ps if p.start.year <= y <= p.end.year and not p.planned]
        if hits:
            out.append((f"Periods in these documents that overlap {y}: " + "; ".join(map(_item, hits)) + ".", hits[0].tag))
        else:
            first = min(ps, key=lambda p: p.start)
            out.append((f"No dated period in these documents overlaps {y}: the earliest starts "
                        f"{first.start:%d %b %Y} ({first.label}) [{first.tag}]. So there is no record of {y}.",
                        first.tag))
    return out
