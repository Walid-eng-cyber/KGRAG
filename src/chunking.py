"""Layout-aware chunking: cut documents where their structure changes, not every N tokens.

Fixed-size chunks split a CV project from its "Technologien:" line or a job from
its dates, so the extractor never sees them together and the graph misses the
links. Here:

  1. Boilerplate: a line repeated at the top or bottom of several pages
     (headers, footers, legends) is kept once. Repeats in the body (two jobs
     with the same title) are left alone.
  2. Sections start at headings: short lines in capitals ("BERUFSERFAHRUNG",
     "RISK FACTORS").
  3. Blocks start at entries: a title line followed within three lines by a
     date or date range. Each block (one project, one job) stays whole.
  4. Blocks are packed into chunks up to a size limit, never across sections,
     and a chunk that doesn't start with its section's heading gets it as a
     first line, so the extractor and the embeddings know the context.

Documents with no headings or dated entries (letters, plain reports) fall back
to page-by-page sentence chunking, as before.
"""
from __future__ import annotations

import re

PAGE_BREAK = "\f"
_DIGITS = re.compile(r"\d+")
_BULLET = re.compile(r"^\s*[●•▪◦\-–*]\s")
_FIELD = re.compile(r"^\s*[\w][\w &/().-]{1,30}:\s")       # "Technologien: …" belongs to the entry above
_DATE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}\b|\b\d{1,2}/(?:19|20)\d{2}\b|"
                   r"\b(?:19|20)\d{2}\s*[–—-]\s*(?:(?:19|20)\d{2}|heute|present|today|aktuell|laufend)\b|"
                   r"\b(?:seit|since|ab|depuis)\s+\d", re.I)


def pages(text: str) -> list[str]:
    return text.split(PAGE_BREAK) if PAGE_BREAK in text else [text]


def strip_boilerplate(text: str, edge: int = 7, min_len: int = 15) -> str:
    """Drop repeats (digits ignored: "Seite 1" = "Seite 2") of lines that sit near a page's top or bottom."""
    ps = [p.split("\n") for p in pages(text)]
    if len(ps) < 2:
        return text

    def edges(lines):
        idx = [i for i, l in enumerate(lines) if l.strip()]
        return set(idx[:edge] + idx[-edge:])

    seen_on: dict[str, set[int]] = {}
    for pi, lines in enumerate(ps):
        for i in edges(lines):
            k = _DIGITS.sub("#", lines[i].strip().lower())
            if len(k) >= min_len:
                seen_on.setdefault(k, set()).add(pi)
    boiler = {k for k, on in seen_on.items() if len(on) >= 2}
    kept_once: set[str] = set()
    out = []
    for lines in ps:
        e = edges(lines)
        page = []
        for i, l in enumerate(lines):
            k = _DIGITS.sub("#", l.strip().lower())
            if i in e and k in boiler:
                if k in kept_once:
                    continue
                kept_once.add(k)
            page.append(l)
        out.append("\n".join(page))
    return PAGE_BREAK.join(out)


def is_heading(line: str) -> bool:
    s = line.strip()
    letters = [c for c in s if c.isalpha()]
    return (4 <= len(s) <= 50 and len(letters) >= 4 and len(s.split()) <= 6
            and sum(c.isupper() for c in letters) / len(letters) >= 0.9 and not _DATE.search(s))


def blocks(text: str) -> list[tuple[str, str]]:
    """(section heading, block text) pairs in document order."""
    out: list[tuple[str, str]] = []
    for page in pages(text):
        lines = page.split("\n")
        section, start = "", 0
        cut = set()
        from src.tables import parse_row
        for i, l in enumerate(lines):
            if is_heading(l):
                cut.add(i)
            elif _DATE.search(l) and not parse_row(l, i):       # a dated table row is not a new entry
                # walk back over the entry's title lines (no bullets, no sentence ends, no headings)
                j = i
                while j - 1 >= 0 and i - (j - 1) <= 3 and lines[j - 1].strip() and not _BULLET.match(lines[j - 1]) \
                        and not lines[j - 1].rstrip().endswith((".", ":")) and not is_heading(lines[j - 1]) \
                        and not _DATE.search(lines[j - 1]) and not _FIELD.match(lines[j - 1]) \
                        and (j - 1) not in cut:
                    j -= 1
                cut.add(j)
        bounds = sorted(cut | {0}) + [len(lines)]
        for a, b in zip(bounds, bounds[1:]):
            part = lines[a:b]
            if part and is_heading(part[0]):
                section = part[0].strip()
                part = part[1:]
            body = "\n".join(part).strip()
            if body:
                out.append((section, body))
            start = b
    return out


def chunk_texts(text: str, max_chars: int = 1400, min_chars: int = 500) -> list[str]:
    """Layout-aware chunks. Falls back to the input's pages when it has no structure."""
    bs = blocks(text)
    structured = sum(1 for _, b in bs if _DATE.search(b)) >= 3 or len({s for s, _ in bs if s}) >= 2
    if not structured:
        return [p.strip() for p in pages(text) if p.strip()]
    out: list[str] = []
    cur_sec, cur = None, ""
    for sec, body in bs:
        new_sec = sec != cur_sec
        piece = (f"{sec}\n" if sec and new_sec else "") + body + "\n"
        too_big = cur and len(cur) + len(piece) > max_chars
        if cur and len(cur) >= min_chars and (new_sec or too_big):
            out.append(cur)
            cur = f"{sec}\n" if sec and not new_sec else ""     # a continued section repeats its heading
        cur += piece
        cur_sec = sec
    if cur.strip():
        out.append(cur)
    # a block longer than the limit is split at sentence or line ends
    final: list[str] = []
    for c in out:
        while len(c) > max_chars * 1.6:
            cut_at = max(c.rfind("\n", 0, max_chars), c.rfind(". ", 0, max_chars))
            cut_at = cut_at if cut_at > max_chars // 2 else max_chars
            final.append(c[:cut_at + 1])
            c = c[cut_at + 1:]
        final.append(c)
    return [c.strip() for c in final if c.strip()]
