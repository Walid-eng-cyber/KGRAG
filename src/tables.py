"""Tables in documents, read as rows and answered in code.

Chunking a document into ~512-token pieces cuts tables apart, so neither the
graph extractor nor vector search can count rows, add a column or find the
lowest value. This module reads tables straight from the extracted text:

  1. Rows: a line that is a name followed by a run of "cells" (numbers,
     decimals, dates, short codes like BE/ZU, yes/no, "-"). Lines with the same
     cell pattern (or a prefix of it: a row with empty trailing cells) form a table.
  2. Tables split across pages are merged; a copy of the same table in another
     language (same numbers in the same order) is dropped, its names kept as aliases.
  3. Legends ("BE = Bestanden, NB = Nicht bestanden") give code meanings.
  4. One model call per table names its columns (English snake_case). Only the
     names come from the model: every value is the document's own text.

At answer time `facts()` turns the tables that match a question into exact,
citable statements: row values, counts per code, totals, lowest and highest,
and which rows have no value in a column.

Nothing here is specific to one kind of document: transcripts, financial
statements, price lists and schedules all have this shape.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------- cells

_CELL = [
    ("date", re.compile(r"^(?:\d{1,2}[./]\d{1,2}[./](?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{4})$")),
    ("pct", re.compile(r"^[-+]?\d+(?:[.,]\d+)?%$")),
    ("money", re.compile(r"^[$€£][-+]?\d[\d.,]*[kKmMbB]?$|^[-+]?\d[\d.,]*[$€£]$")),
    ("num", re.compile(r"^[-+(]?\d{1,3}(?:[.,]\d{3})+\)?$|^[-+(]?\d+\)?$")),
    ("dec", re.compile(r"^[-+(]?\d+[.,]\d{1,2}\)?$")),
    ("code", re.compile(r"^[A-ZÄÖÜ]{1,4}\d?$")),
    ("yesno", re.compile(r"^(?:ja|nein|yes|no|oui|non|x|✓|✔|✗)$", re.I)),
    ("dash", re.compile(r"^[-–—]$")),
]
_LABEL_TOKEN = re.compile(r"^[A-Z][a-zA-Z]{1,3}\.?:?$")     # "PNr", "Nr.", "No" before an id


def cell_class(tok: str) -> str | None:
    for name, rx in _CELL:
        if rx.match(tok):
            return name
    return None


def to_number(v: str) -> float | None:
    """'2,3' -> 2.3; '1,234' / '1.234' -> 1234; '(5)' -> -5; '12%' -> 12; '€5' -> 5."""
    s = v.strip().replace("$", "").replace("€", "").replace("£", "").rstrip("%")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()+")
    if re.fullmatch(r"-?\d{1,3}(?:[.,]\d{3})+", s):
        s = re.sub(r"[.,]", "", s)
    elif re.fullmatch(r"-?\d+,\d{1,2}", s):
        s = s.replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


@dataclass
class Row:
    name: str
    cells: list[str]
    classes: list[str]
    line: int


def parse_row(line: str, n: int) -> Row | None:
    toks = line.split()
    i = len(toks)
    while i > 0 and cell_class(toks[i - 1]):
        i -= 1
    cells = toks[i:]
    name_toks = toks[:i]
    while name_toks and _LABEL_TOKEN.match(name_toks[-1]) and len(name_toks) > 1:
        name_toks = name_toks[:-1]                      # "Computer Vision 1 PNr" -> "Computer Vision 1"
    name = " ".join(name_toks).strip(" :–-")
    classes = [cell_class(c) for c in cells]
    if len(cells) < 2 or not re.search(r"[^\W\d_]{2}", name) or len(name) > 90:
        return None
    if not any(c in ("num", "dec", "date", "pct", "money") for c in classes):
        return None
    return Row(name, cells, classes, n)


# ---------------------------------------------------------------- tables

@dataclass
class Table:
    doc_id: str
    title: str
    header: str
    rows: list[dict]                       # {"name", "cells", "aliases"}
    classes: list[str]                     # the full (longest) cell pattern
    columns: list[str] = field(default_factory=list)
    legend: dict[str, str] = field(default_factory=dict)
    name_column: str = "name"

    def row_key(self, r: dict) -> tuple:
        """A row's numbers and dates: the same in every language the table is printed in."""
        return tuple(c for c, k in zip(r["cells"], self.classes) if k in ("num", "dec", "date", "pct", "money"))


_LEGEND = re.compile(r"\b([A-ZÄÖÜ]{1,4})\s*=\s*([^,=]+?)(?=,\s*[A-ZÄÖÜ]{1,4}\s*=|$)")


def legends(text: str) -> dict[str, str]:
    """Code meanings from lines with at least two "CODE = meaning" pairs. Several languages are joined."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        pairs = _LEGEND.findall(line)
        if len(pairs) >= 2:
            for code, meaning in pairs:
                m = meaning.strip(" .;")
                if m and m not in out.setdefault(code, []):
                    out[code].append(m)
    return {k: " / ".join(v) for k, v in out.items()}


def _is_prefix(short: list[str], long: list[str]) -> bool:
    return len(short) <= len(long) and long[:len(short)] == short


def _copy_of(t: Table, other: Table) -> bool:
    """t repeats `other` (another language or a reprint): most of its rows have the same numbers."""
    if not _is_prefix(t.classes, other.classes) and not _is_prefix(other.classes, t.classes):
        return False
    keys = {other.row_key(r) for r in other.rows}
    mine = [t.row_key(r) for r in t.rows]
    return bool(mine) and sum(k in keys for k in mine) >= 0.8 * len(mine)


def find_tables(text: str, doc_id: str = "", title: str = "", min_rows: int = 3) -> list[Table]:
    lines = text.splitlines()
    parsed = [parse_row(l, i) for i, l in enumerate(lines)]
    # regions: runs of rows with at most 1 non-row line between them (a wrapped name); a longer gap is a
    # new page or a new table, which the merge step below joins again when it continues the same table
    regions, cur, gap = [], [], 0
    for r in parsed:
        if r:
            cur.append(r)
            gap = 0
        elif cur:
            gap += 1
            if gap >= 2:
                regions.append(cur)
                cur, gap = [], 0
    if cur:
        regions.append(cur)

    found: list[Table] = []
    fragments: list[list[Row]] = []            # too short on their own: may continue a table on the next page
    for reg in regions:
        groups: list[list[Row]] = []
        for r in sorted(reg, key=lambda r: -len(r.classes)):
            g = next((g for g in groups if _is_prefix(r.classes, g[0].classes)), None)
            (g.append(r) if g else groups.append([r]))
        # A row that fits no table may have a number that belongs to its name ("Computer Vision 1"):
        # move its first cell back into the name and try again.
        for g in [g for g in groups if len(g) < min_rows]:
            for r in list(g):
                if len(r.classes) > 2 and r.classes[0] in ("num", "code"):
                    moved = Row(f"{r.name} {r.cells[0]}", r.cells[1:], r.classes[1:], r.line)
                    home = next((h for h in groups if len(h) >= min_rows and _is_prefix(moved.classes, h[0].classes)), None)
                    if home:
                        home.append(moved)
                        g.remove(r)
        for g in groups:
            if len(g) < min_rows:
                if g:
                    fragments.append(sorted(g, key=lambda r: r.line))
                continue
            g.sort(key=lambda r: r.line)
            year_head = None
            if all(re.fullmatch(r"(?:19|20)\d{2}", c) for c in g[0].cells):    # "Posten 2024 2023" is a header
                year_head, g = g[0], g[1:]
                if len(g) < min_rows - 1:
                    continue
            first = (year_head or g[0]).line
            head = [lines[i].strip() for i in range(max(0, first - 6), first)
                    if lines[i].strip() and not parsed[i]]
            if year_head:
                head.append(f"{year_head.name} " + " ".join(year_head.cells))
            found.append(Table(doc_id, title, " · ".join(head[-4:])[:300],
                               [{"name": r.name, "cells": r.cells, "aliases": []} for r in g],
                               max(g, key=lambda r: len(r.classes)).classes))

    # one table split over several pages: same cell pattern, separate regions
    merged: list[Table] = []
    for t in found:
        same = next((m for m in merged if m.classes == t.classes and not _copy_of(t, m)), None)
        twin = next((m for m in merged if _copy_of(t, m)), None)
        if twin:                               # the same rows again, e.g. the English printout
            by_key = {twin.row_key(r): r for r in twin.rows}
            for r in t.rows:
                a = by_key.get(t.row_key(r))
                if a and r["name"] != a["name"] and r["name"] not in a["aliases"]:
                    a["aliases"].append(r["name"])
        elif same:
            same.rows += t.rows
        else:
            merged.append(t)
    for frag in fragments:                     # a page-2 remainder of an earlier table
        longest = max(frag, key=lambda r: len(r.classes)).classes
        home = next((m for m in merged if _is_prefix(longest, m.classes)), None)
        if home:
            piece = Table(home.doc_id, home.title, "", [{"name": r.name, "cells": r.cells, "aliases": []}
                                                        for r in frag], longest)
            if not _copy_of(piece, home):
                home.rows += piece.rows
    for t in merged:                           # a page printed twice inside one region
        seen, rows = set(), []
        for r in t.rows:
            k = (r["name"].lower(), tuple(r["cells"]))
            if k not in seen:
                seen.add(k)
                rows.append(r)
        t.rows = rows
    leg = legends(text)
    for t in merged:
        codes = {c for r in t.rows for c, k in zip(r["cells"], t.classes) if k == "code"}
        t.legend = {c: m for c, m in leg.items() if c in codes}
    return merged


# ---------------------------------------------------------------- column names

def _fallback_columns(t: Table) -> list[str]:
    return [f"{k}_{i + 1}" for i, k in enumerate(t.classes)]


def name_columns(t: Table, chat=None) -> Table:
    """Ask the model what each cell position means. Any malformed reply falls back to generic names."""
    full = [r for r in t.rows if len(r["cells"]) == len(t.classes)][:4] or t.rows[:4]
    names = ", ".join(r["name"] for r in t.rows[:4])
    cells = "\n".join(
        f"cell {i + 1}: e.g. " + ", ".join(repr(r["cells"][i]) for r in full if i < len(r["cells"]))
        for i in range(len(t.classes)))
    prompt = (
        "A table was read from a document. Each row has a NAME and then cells. Say what the names are "
        "and what each cell position holds.\n"
        f"Text above the table: {t.header or '(none)'}\n"
        f"Code meanings: {json.dumps(t.legend, ensure_ascii=False) if t.legend else '(none)'}\n"
        f"Row names, e.g.: {names}\n{cells}\n\n"
        f"Reply ONLY JSON: {{\"name_column\": \"what the row names are, e.g. module or product\", "
        f"\"columns\": [exactly {len(t.classes)} short English snake_case names, cell 1 first; the row "
        "name is NOT a cell. Examples: \"exam_number\", \"exam_date\", \"status\", \"ects_credits\", "
        "\"grade\", \"revenue_2024\"]}"
    )
    try:
        if chat is None:
            import ollama

            import config
            client = ollama.Client(host=config.OLLAMA_BASE_URL)
            chat = lambda p: client.chat(model=config.ANSWER_MODEL, format="json",   # noqa: E731
                                         options={"temperature": 0},
                                         messages=[{"role": "user", "content": p}])["message"]["content"]
        reply = json.loads(chat(prompt))
        cols = [re.sub(r"[^a-z0-9_]", "", str(c).lower().replace(" ", "_"))[:32] or f"col_{i + 1}"
                for i, c in enumerate(reply.get("columns", []))]
        if len(cols) != len(t.classes):
            raise ValueError("wrong column count")
        if any(_mismatch(c, k) for c, k in zip(cols, t.classes)):
            raise ValueError("names don't fit the cells (shifted by one?)")
        t.columns = list(_dedup(cols))
        t.name_column = re.sub(r"[^a-z0-9_ ]", "", str(reply.get("name_column", "name")).lower())[:32] or "name"
    except Exception:  # noqa: BLE001
        t.columns = _fallback_columns(t)
    return t


_DATEY = re.compile(r"date|day|time|when|deadline|period|datum")
_AMOUNTY = re.compile(r"grade|point|ects|credit|amount|price|count|total|revenue|cost|score|value|number")


def _mismatch(name: str, cls: str) -> bool:
    """A column name that can't describe its cells: the model counted the row name as a column."""
    if cls == "date":
        return not _DATEY.search(name)
    if _DATEY.search(name):
        return True
    if cls in ("yesno", "dash", "code"):
        return bool(_AMOUNTY.search(name)) and cls != "code" or (cls == "code" and "number" in name)
    return False


def _dedup(cols: list[str]):
    seen: dict[str, int] = {}
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        yield c if seen[c] == 1 else f"{c}_{seen[c]}"


# ---------------------------------------------------------------- storage

_DDL = """CREATE TABLE IF NOT EXISTS doc_tables (
    id serial PRIMARY KEY, corpus_id text NOT NULL, doc_id text NOT NULL, data jsonb NOT NULL)"""


def save(corpus_id: str, doc_id: str, tables: list[Table]) -> None:
    from src.vectorize import pg_connect
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            cur.execute("DELETE FROM doc_tables WHERE corpus_id = %s AND doc_id = %s", (corpus_id, doc_id))
            for t in tables:
                cur.execute("INSERT INTO doc_tables (corpus_id, doc_id, data) VALUES (%s, %s, %s)",
                            (corpus_id, doc_id, json.dumps(asdict(t), ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()


def load(corpus_id: str) -> list[Table]:
    from src.vectorize import pg_connect
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            conn.commit()
            cur.execute("SELECT data FROM doc_tables WHERE corpus_id = %s ORDER BY id", (corpus_id,))
            return [Table(**(d if isinstance(d, dict) else json.loads(d))) for (d,) in cur.fetchall()]
    finally:
        conn.close()


def delete(corpus_id: str, doc_id: str | None = None) -> None:
    from src.vectorize import pg_connect
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            if doc_id:
                cur.execute("DELETE FROM doc_tables WHERE corpus_id = %s AND doc_id = %s", (corpus_id, doc_id))
            else:
                cur.execute("DELETE FROM doc_tables WHERE corpus_id = %s", (corpus_id,))
        conn.commit()
    finally:
        conn.close()


def ingest(corpus_id: str, doc_id: str, title: str, text: str, chat=None) -> int:
    tables = [name_columns(t, chat) for t in find_tables(text, doc_id, title)]
    save(corpus_id, doc_id, tables)
    return len(tables)


# ---------------------------------------------------------------- answering

_Q_STOP = set("what which who how many much does did has have the and for with his her their was were are is "
              "of in on at to all any wie viele welche welcher hat der die das und".split())
_ID_COL = re.compile(r"(^|_)(id|nr|no|number|code|pnr|ref)($|_)")


def _qwords(q: str) -> set[str]:
    return {w for w in re.findall(r"[a-zäöüß]{3,}|\d{4}", q.lower()) if w not in _Q_STOP}


def _vocab(t: Table) -> set[str]:
    words = set()
    for s in [t.header, t.title, t.name_column, *t.columns, *t.legend.values()]:
        words |= set(re.findall(r"[a-zäöüß]{3,}|\d{4}", s.lower().replace("_", " ")))
    return words


def _hit(w: str, vocab: set[str]) -> bool:
    return any(w == v or (len(w) >= 4 and len(v) >= 4 and (w in v or v in w)) for v in vocab)


def _row_names(r: dict) -> list[str]:
    return [r["name"], *r.get("aliases", [])]


def _rows_named(q: str, t: Table) -> list[dict]:
    ql = f" {q.lower()} "
    return [r for r in t.rows if any(len(n) >= 4 and f" {n.lower()} " in ql or
                                     (len(n) >= 6 and n.lower() in ql) for n in _row_names(r))]


def relevance(q: str, t: Table) -> int:
    words = _qwords(q)
    return sum(_hit(w, _vocab(t)) for w in words) + 3 * len(_rows_named(q, t))


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if v == int(v) and abs(v) >= 1000 else (f"{v:g}" if v == int(v) else f"{v:.2f}".rstrip("0"))


def _val(t: Table, r: dict, i: int) -> str | None:
    return r["cells"][i] if i < len(r["cells"]) else None


def describe_row(t: Table, r: dict) -> str:
    parts = []
    for i, col in enumerate(t.columns):
        v = _val(t, r, i)
        if v is None or t.classes[i] == "dash":
            continue
        meaning = t.legend.get(v) if t.classes[i] == "code" else None
        parts.append(f"{col} {v}" + (f" ({meaning})" if meaning else ""))
    missing = [c for i, c in enumerate(t.columns) if _val(t, r, i) is None]
    return f"{r['name']}: " + "; ".join(parts) + (f"; no value for {', '.join(missing)}" if missing else "")


def _looks_like_ids(present: list) -> bool:
    vals = [v for _, v in present]
    return len(vals) >= 3 and len(set(vals)) == len(vals) and all(re.fullmatch(r"\d{3,}", v) for v in vals)


def summarize(t: Table, max_names: int = 20) -> str:
    n = len(t.rows)
    out = [f"Table “{t.title}” ({t.header[:80]}{'…' if len(t.header) > 80 else ''}) has {n} {t.name_column} rows"]
    for i, col in enumerate(t.columns):
        k = t.classes[i]
        vals = [(r, _val(t, r, i)) for r in t.rows]
        present = [(r, v) for r, v in vals if v is not None and k != "dash"]
        empty = [r["name"] for r, v in vals if v is None]
        if k == "code":
            groups: dict[str, list[str]] = {}
            for r, v in present:
                groups.setdefault(v, []).append(r["name"])
            if 1 < len(groups) <= 12:
                out.append(f"{col}: " + "; ".join(
                    f"{v}{' (' + t.legend[v] + ')' if v in t.legend else ''} ×{len(names)}"
                    + (f" = {', '.join(names)}" if len(names) <= max_names else "")
                    for v, names in sorted(groups.items(), key=lambda kv: -len(kv[1]))))
        elif k in ("num", "dec", "pct", "money") and not _ID_COL.search(col) and not _looks_like_ids(present):
            nums = [(r, to_number(v)) for r, v in present]
            nums = [(r, x) for r, x in nums if x is not None]
            if not nums:
                continue
            lo = min(nums, key=lambda t_: t_[1])
            hi = max(nums, key=lambda t_: t_[1])
            s = (f"{col}: {len(nums)} rows have a value, total {_fmt(sum(x for _, x in nums))}, "
                 f"lowest {lo[0]['cells'][i]} ({lo[0]['name']}), highest {hi[0]['cells'][i]} ({hi[0]['name']})")
            if empty:
                s += f"; no {col} for {len(empty)}: " + (", ".join(empty) if len(empty) <= max_names else "…")
            out.append(s)
        elif k == "date" and present:
            out.append(f"{col}: from {present[0][1]} … ({len(present)} dated rows)")
    return ". ".join(out) + "."


def facts(question: str, tables: list[Table]) -> list[str]:
    """Statements for the table(s) that match the question: the rows it names, then the table summary."""
    scored = sorted(((relevance(question, t), -len(t.rows), i, t) for i, t in enumerate(tables)),
                    key=lambda x: (-x[0], x[1], x[2]))
    if not scored or scored[0][0] < 2:
        return []
    best = scored[0][3]
    out = [f"In “{best.title}”, {describe_row(best, r)}." for r in _rows_named(question, best)[:6]]
    out.append(summarize(best))
    return out


if __name__ == "__main__":
    # Read the tables of every ready document in a corpus (for documents ingested before tables existed).
    #   python -m src.tables <corpus_id>
    import argparse
    import os
    from pathlib import Path

    import config
    from server.doc_store import DocumentStore
    from src.documents import extract_text

    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_id")
    a = ap.parse_args()
    store = DocumentStore(Path(os.getenv("DOCUMENTS_DB", str(config.DATA_DIR / "documents.db"))),
                          Path(os.getenv("UPLOAD_DIR", str(config.DATA_DIR / "uploads"))))
    for d in store.list(a.corpus_id):
        if d["status"] != "ready":
            continue
        d = store.get(d["id"])                   # the full row, with the file extension
        n = ingest(a.corpus_id, d["id"], d["title"], extract_text(store.path(d), d["kind"]))
        print(f"{d['title']}: {n} table(s)")
