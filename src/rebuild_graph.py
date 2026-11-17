"""Rebuild the graph of existing corpora, e.g. with a stronger model on another machine, in detailed mode.

Every ready document is read again from its uploaded file and extracted again; once extraction succeeds, its
old graph, vectors and tables are replaced. A document that fails keeps its old graph, and the run moves on.

    # 1. check: the extraction server answers, the model is there, how long one document takes, ETA
    python -m src.rebuild_graph --all --mode detailed --dry-run

    # 2. the run (the report is saved after every document)
    python -m src.rebuild_graph --all --mode detailed

    # 3. after a stop (Ctrl-C, power cut, server down): continue where it stopped
    python -m src.rebuild_graph --all --mode detailed --resume data/rebuilds/rebuild-20261118-0700.json

Where the model runs comes from .env / the environment: EXTRACT_OLLAMA_URL (default: OLLAMA_BASE_URL) and
EXTRACT_MODEL. Embeddings (bge-m3) still run on OLLAMA_BASE_URL. The report (data/rebuilds/*.json) holds, per
document, the time, tokens and what each pass found or dropped, plus each corpus's graph size before and after.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

import config

REPORT_DIR = config.DATA_DIR / "rebuilds"
_DB = os.getenv("DOCUMENTS_DB", str(config.DATA_DIR / "documents.db"))
_UPLOADS = os.getenv("UPLOAD_DIR", str(config.DATA_DIR / "uploads"))
MAX_FAILURES_IN_A_ROW = 3


def _stores():
    from server.corpora import CorpusStore
    from server.doc_store import DocumentStore
    return CorpusStore(_DB), DocumentStore(_DB, _UPLOADS)


def plan(corpora, doc_store, corpus_ids: list[str], all_corpora: bool, only_doc: str | None = None,
         limit: int | None = None) -> list[tuple[dict, dict]]:
    """(corpus, document) pairs to rebuild: ready documents of document corpora, by corpus then title."""
    if all_corpora:
        import sqlite3
        with sqlite3.connect(_DB) as con:
            corpus_ids = [r[0] for r in con.execute("SELECT id FROM corpora ORDER BY id")]
    todo = []
    for cid in corpus_ids:
        c = corpora.get(cid)
        if c is None:
            raise SystemExit(f"No corpus {cid!r}. Corpora: see the sidebar, or --all.")
        if c.get("kind") != "documents":
            continue                     # e.g. SEC insider data: not built from uploaded documents
        for d in sorted(doc_store.list(cid), key=lambda d: (d["title"] or "").lower()):
            if d["status"] == "ready" and (only_doc is None or d["id"] == only_doc):
                todo.append((c, doc_store.get(d["id"])))     # the full record (its file extension included)
    return todo[:limit] if limit else todo


def check_extraction_server() -> str:
    """Fail early, with the fix, when the extraction model can't be reached. Returns a one-line description."""
    from src.llm import is_hosted
    if is_hosted():
        return f"{config.EXTRACT_MODEL} via {config.LLM_BASE_URL} (hosted)"
    url = config.EXTRACT_OLLAMA_URL.rstrip("/")
    try:
        tags = requests.get(f"{url}/api/tags", timeout=10).json()
    except requests.RequestException as e:
        raise SystemExit(
            f"Can't reach the extraction server at {url} ({type(e).__name__}).\n"
            "On that machine, start Ollama so other computers can reach it:  OLLAMA_HOST=0.0.0.0 ollama serve\n"
            "(on Windows: set OLLAMA_HOST=0.0.0.0 in the system environment variables, restart Ollama, and allow "
            "port 11434 through the firewall). Then check EXTRACT_OLLAMA_URL here.") from e
    names = {m.get("name", "") for m in tags.get("models", [])}
    want = config.EXTRACT_MODEL
    if want not in names and f"{want}:latest" not in names:
        raise SystemExit(f"{url} doesn't have {want}. On that machine run:  ollama pull {want}\n"
                         f"(it has: {', '.join(sorted(names)) or 'no models'})")
    return f"{want} on {url}"


def graph_size(corpus_ids: list[str]) -> dict:
    """Entities and relations (MENTIONS excluded) per corpus."""
    from src.corpus_graph import _driver
    d = _driver()
    try:
        with d.session() as s:
            out = {}
            for cid in corpus_ids:
                e = s.run("MATCH (e:DocEntity {corpus: $c}) RETURN count(e) AS n", c=cid).single()["n"]
                r = s.run("MATCH (:DocEntity {corpus: $c})-[r]->(:DocEntity {corpus: $c}) WHERE type(r) <> 'MENTIONS' "
                          "RETURN count(r) AS n", c=cid).single()["n"]
                out[cid] = {"entities": e, "relations": r}
            return out
    finally:
        d.close()


def _eta(done_secs: list[float], done_chars: int, remaining_chars: int, remaining_docs: int) -> datetime:
    """Finish time from the pace so far: seconds per character, with a per-document floor (each document
    costs at least three calls however short it is)."""
    spent = sum(done_secs)
    per_char = spent / max(done_chars, 1)
    per_doc = spent / max(len(done_secs), 1)
    left = max(remaining_chars * per_char, remaining_docs * per_doc * 0.6)
    return datetime.now() + timedelta(seconds=left)


class Report:
    def __init__(self, path: Path, settings: dict, resume: dict | None = None):
        self.path = path
        self.data = resume or {"settings": settings, "started": datetime.now().isoformat(timespec="seconds"),
                               "graph_before": {}, "graph_after": {}, "documents": {}}

    def done_ok(self) -> set[str]:
        return {k for k, v in self.data["documents"].items() if v.get("ok")}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.path)


def rebuild_one(corpus: dict, doc: dict, doc_store, mode: str, dry_run: bool) -> dict:
    """Extract (and unless dry_run, replace) one document's graph. Returns its report row."""
    from src import corpus_graph, meter
    from src import documents as docs

    text = docs.extract_text(doc_store.path(doc), doc["kind"])
    t0 = time.perf_counter()
    steps: list[str] = []

    def progress(msg: str, _frac: float | None = None) -> None:
        steps.append(msg)

    with meter.meter() as m:
        if dry_run:
            chunks, found, detail, _ = corpus_graph.extract(corpus, doc["id"], doc["title"], text, progress, mode)
            out = {"chunks": len(chunks), "entities": len({e["name"] for ents, _ in found.values() for e in ents}),
                   "relations": sum(len(r) for _, r in found.values()), "detail": detail,
                   "sample": [f"{r['source']} -[{r['type']}]-> {r['target']}" for _, rels in found.values() for r in rels][:12]}
        else:
            out = corpus_graph.ingest_text(corpus, doc["id"], doc["title"], text, progress, mode=mode, replace=True)
    secs = time.perf_counter() - t0
    if not dry_run:
        extra = f", {out['entities']} entities, {out['relations']} relations ({mode})"
        doc_store.update(doc["id"], chunks=out["chunks"], message=f"Ready: {out['chunks']} chunks{extra}",
                         ingested_at=datetime.now().astimezone().isoformat(timespec="seconds"))
    return {"ok": True, "corpus": corpus["id"], "title": doc["title"], "chars": len(text), "seconds": round(secs, 1),
            "calls": m.requests, "tokens_in": m.tokens_in, "tokens_out": m.tokens_out, **out}


def _resume_hint(argv: list[str], path: Path) -> str:
    """The command that continues this run (the original arguments, with --resume pointing at its report)."""
    args, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x == "--resume":
            skip = True
        elif not x.startswith("--resume="):
            args.append(x)
    return f"python -m src.rebuild_graph {' '.join(args)} --resume {path}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", default=[], help="corpus id to rebuild (repeatable)")
    ap.add_argument("--all", action="store_true", help="every document corpus")
    ap.add_argument("--doc", help="only this document id")
    ap.add_argument("--mode", choices=["detailed", "fast"], default="detailed")
    ap.add_argument("--limit", type=int, help="at most this many documents")
    ap.add_argument("--dry-run", action="store_true",
                    help="extract the first document(s) (default 1) without writing anything, show what was found "
                         "and the estimated finish time for the full run")
    ap.add_argument("--resume", type=Path, help="a report from an earlier run: skip the documents it finished")
    a = ap.parse_args(argv)
    if not a.all and not a.corpus:
        ap.error("name a corpus with --corpus ID, or use --all")

    corpora, doc_store = _stores()
    todo = plan(corpora, doc_store, a.corpus, a.all, a.doc)
    if not todo:
        print("Nothing to rebuild: no ready documents in those corpora.")
        return 0
    server = check_extraction_server()
    total_chars = sum(d.get("chars") or 0 for _, d in todo)
    n_corpora = len({c["id"] for c, _ in todo})
    print(f"Rebuilding {len(todo)} documents in {n_corpora} corpora ({total_chars:,} characters), mode {a.mode}")
    print(f"Extraction: {server}")

    if a.dry_run:
        sample = todo[: a.limit or 1]
        secs, chars = [], 0
        for c, d in sample:
            print(f"\n{c['id']} · {d['title']} ({d.get('chars') or 0} chars): extracting, nothing will be written…")
            row = rebuild_one(c, d, doc_store, a.mode, dry_run=True)
            secs.append(row["seconds"])
            chars += row["chars"]
            det = row.get("detail") or {}
            print(f"  {row['seconds']}s, {row['calls']} model calls, {row['tokens_in']:,} tokens in / {row['tokens_out']:,} out")
            print(f"  {row['entities']} entities, {row['relations']} facts kept")
            if det:
                print(f"  pass 1 found {det['extracted']}, pass 2 added {det['gleaned']}, dropped "
                      f"{det['quote_not_found']} (quote not in the text) + {det['rejected_by_check']} (failed the check); "
                      f"seconds per pass {det['seconds']}")
            for line in row["sample"]:
                print(f"    {line}")
        rest = todo[len(sample):]
        finish = _eta(secs, chars, sum(d.get("chars") or 0 for _, d in rest), len(rest))
        took = finish - datetime.now()
        print(f"\nEstimated full run: about {took.total_seconds() / 3600:.1f} h for {len(todo)} documents, "
              f"finishing around {finish:%H:%M} if started now (plus ~10 min for embeddings and cleanup).")
        print(f"Started at 07:00, it would finish around {(datetime(2000, 1, 1, 7, 0) + took):%H:%M}.")
        return 0

    resume = json.loads(a.resume.read_text()) if a.resume else None
    path = a.resume or REPORT_DIR / f"rebuild-{datetime.now():%Y%m%d-%H%M}.json"
    settings = {"mode": a.mode, "extract_model": config.EXTRACT_MODEL, "server": server,
                "num_ctx": config.EXTRACT_NUM_CTX, "window_chars": config.DETAILED_WINDOW_CHARS}
    report = Report(path, settings, resume)
    corpus_ids = sorted({c["id"] for c, _ in todo})
    if not report.data["graph_before"]:
        report.data["graph_before"] = graph_size(corpus_ids)
    skip = report.done_ok()
    pending = [(c, d) for c, d in todo if d["id"] not in skip]
    print(f"Report: {path}" + (f" (resuming: {len(skip)} already done)" if skip else ""))
    report.save()

    secs: list[float] = []
    chars_done, failures_in_a_row = 0, 0
    try:
        for i, (c, d) in enumerate(pending, 1):
            head = f"[{i:>3}/{len(pending)}] {c['id']} · {d['title']}"
            try:
                row = rebuild_one(c, d, doc_store, a.mode, dry_run=False)
                failures_in_a_row = 0
                secs.append(row["seconds"])
                chars_done += row["chars"]
                det = row.get("detail") or {}
                dropped = det.get("quote_not_found", 0) + det.get("rejected_by_check", 0)
                rest = pending[i:]
                finish = _eta(secs, chars_done, sum(x.get("chars") or 0 for _, x in rest), len(rest))
                print(f"{head} · {row['seconds']}s · {row['entities']} entities, {row['relations']} facts"
                      + (f" ({dropped} dropped)" if det else "") + (f" · ETA {finish:%H:%M}" if rest else ""), flush=True)
            except Exception as e:  # noqa: BLE001 - one bad document must not stop the night's run
                failures_in_a_row += 1
                row = {"ok": False, "corpus": c["id"], "title": d["title"], "error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"{head} · FAILED (old graph kept): {row['error']}", flush=True)
            report.data["documents"][d["id"]] = row
            report.save()
            if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                print(f"\n{MAX_FAILURES_IN_A_ROW} documents failed in a row: the extraction server, Neo4j or Postgres is "
                      "probably down. Fix it, then continue with:")
                print(f"  {_resume_hint(argv, path)}")
                return 1
    except KeyboardInterrupt:
        report.save()
        print(f"\nStopped. Continue later with:\n  {_resume_hint(argv, path)}")
        return 130

    report.data["graph_after"] = graph_size(corpus_ids)
    report.data["finished"] = datetime.now().isoformat(timespec="seconds")
    report.save()
    rows = report.data["documents"].values()
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"\nDone: {ok}/{len(todo)} documents rebuilt, {len(todo) - ok} failed (their old graph is kept).")
    for cid in corpus_ids:
        b, f = report.data["graph_before"].get(cid, {}), report.data["graph_after"][cid]
        print(f"  {cid}: entities {b.get('entities', '?')} -> {f['entities']}, "
              f"relations {b.get('relations', '?')} -> {f['relations']}")
    print(f"Report: {path}")
    return 0 if ok == len(todo) else 1


if __name__ == "__main__":
    raise SystemExit(main())
