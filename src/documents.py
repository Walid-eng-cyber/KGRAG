"""Ingest user-supplied documents (PDF, TXT, Markdown, HTML) into the graph and
the vector index: the same pipeline as the 10-Ks, driven per document.

    text extraction -> SentenceSplitter chunks (stable ids)
                    -> SchemaLLMPathExtractor triples -> Neo4j (Chunk nodes + entities)
                    -> bge-m3 embeddings of the same chunks -> pgvector

A document's chunks carry `ticker = doc_id`, so vectorize/retrieval treat it like
any filing, and `remove_document` can find and delete exactly its chunks.

CLI:
    python -m src.documents path/to/report.pdf [--title "Annual report 2025"]
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

import config

SUPPORTED = {".pdf": "pdf", ".txt": "text", ".md": "text", ".markdown": "text", ".html": "html", ".htm": "html"}
MAX_PDF_PAGES = 400
CHUNK_SIZE, CHUNK_OVERLAP, BATCH = 512, 64, 4

DOC_HINT = (
    "You are reading a company document (an annual report, filing, investor "
    "presentation or similar). Extract only facts that are explicitly stated. "
    "Identify companies, their subsidiaries, competitors, partners, executives and "
    "directors, products, business segments, auditor, regulators, stock exchanges, "
    "headquarters/operating locations, and the key risk factors described. Use each "
    "company's common name."
)

Progress = Callable[[str, float | None], None]  # (message, fraction 0..1 or None)


def _noop(_msg: str, _frac: float | None = None) -> None:
    pass


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

class DocumentError(ValueError):
    """A file that can't be turned into usable text."""


def kind_for(filename: str) -> str:
    kind = SUPPORTED.get(Path(filename).suffix.lower())
    if not kind:
        raise DocumentError(f"Unsupported file type. Use one of: {', '.join(sorted(SUPPORTED))}.")
    return kind


def extract_text(path: Path, kind: str) -> str:
    if kind == "pdf":
        from pypdf import PdfReader

        with open(path, "rb") as fh:
            if fh.read(5) != b"%PDF-":
                raise DocumentError("This file isn't a valid PDF.")
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                raise DocumentError("This PDF is password-protected.")
            pages = reader.pages[:MAX_PDF_PAGES]
            # pages are separated by a form feed, so layout-aware chunking can find page tops and bottoms
            text = "\n\f\n".join((p.extract_text() or "") for p in pages)
        except DocumentError:
            raise
        except Exception as e:  # noqa: BLE001 - malformed PDFs raise many kinds of errors
            raise DocumentError("This PDF couldn't be read.") from e
    elif kind == "html":
        from src.edgar import _html_to_text

        text = _html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 200:
        raise DocumentError("Too little text found. Scanned (image-only) PDFs need OCR first.")
    return text


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Graph + vector ingestion
# ---------------------------------------------------------------------------

def _chunk_id(i: int, doc) -> str:
    return hashlib.sha1(f"{doc.doc_id}:{i}".encode()).hexdigest()


def ingest_text(doc_id: str, title: str, text: str, progress: Progress = _noop) -> dict:
    """Extract a graph from `text` into Neo4j, then embed its chunks into pgvector."""
    from llama_index.core import Document, PropertyGraphIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter

    from src.ingest import build_extractor, build_graph_store, build_llm

    text = text[: config.MAX_FILING_CHARS]
    llm = build_llm()
    Settings.llm = llm
    doc = Document(doc_id=doc_id, text=f"{DOC_HINT}\n\nDocument: {title}\n\n{text}",
                   metadata={"ticker": doc_id, "title": title, "source": "upload"})
    nodes = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                             id_func=_chunk_id).get_nodes_from_documents([doc])
    total = len(nodes)
    eta_min = max(1, round(total * config.SECONDS_PER_CHUNK / 60))
    progress(f"Extracting graph: 0/{total} chunks (about {eta_min} min)", 0.0)

    store = build_graph_store()
    for i in range(0, total, BATCH):
        batch = nodes[i:i + BATCH]
        # LlamaIndex runs each index build in a fresh asyncio event loop, and the
        # Ollama client caches its async session on the first loop. Reusing it for
        # a second batch fails with "Event loop is closed", so build a new one per batch.
        batch_llm = build_llm()
        PropertyGraphIndex(nodes=batch, llm=batch_llm, kg_extractors=[build_extractor(batch_llm)],
                           property_graph_store=store, embed_kg_nodes=False, show_progress=False)
        done = min(i + BATCH, total)
        progress(f"Extracting graph: {done}/{total} chunks", 0.85 * done / total)

    progress("Embedding chunks for vector search", 0.9)
    embedded = _embed_chunks(doc_id, title)
    progress("Ready", 1.0)
    return {"chunks": total, "embedded": embedded}


def _embed_chunks(doc_id: str, title: str) -> int:
    from neo4j import GraphDatabase

    from src.embeddings import embed_texts
    from src.vectorize import DDL, UPSERT, _vec, chunks_for, pg_connect

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
                                  notifications_min_severity="OFF")
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
        rows = chunks_for(driver, doc_id)
        for r in rows:
            r["text"] = strip_hint(r["text"], title)
        vectors = embed_texts([r["text"] for r in rows]) if rows else []
        with conn.cursor() as cur:
            for r, vec in zip(rows, vectors):
                cur.execute(UPSERT, (r["chunk_id"], doc_id, title, date.today(), r["entity_ids"], r["text"], _vec(vec)))
        conn.commit()
        return len(rows)
    finally:
        conn.close()
        driver.close()


def strip_hint(chunk_text: str, title: str) -> str:
    """Remove the extraction instructions + title header that prefix a document's
    first chunk, so they are never embedded or shown to the answer model as content."""
    head = f"{DOC_HINT}\n\nDocument: {title}\n\n"
    if chunk_text.startswith(head):
        return chunk_text[len(head):]
    return chunk_text.replace(DOC_HINT, "").lstrip()


def remove_document(doc_id: str) -> None:
    """Delete a document's chunks from Neo4j and pgvector, plus entities that no other
    document mentions."""
    from neo4j import GraphDatabase

    from src.vectorize import pg_connect

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
                                  notifications_min_severity="OFF")
    try:
        with driver.session() as s:
            # Entities mentioned ONLY by this document's chunks go too; anything another
            # document mentions, and the structured SEC data (:SecEntity), stays.
            s.run("""
                MATCH (ch:Chunk {ticker: $id})-[:MENTIONS]->(e)
                WHERE NOT e:SecEntity
                  AND NOT EXISTS { MATCH (other:Chunk)-[:MENTIONS]->(e) WHERE other.ticker <> $id }
                WITH DISTINCT e DETACH DELETE e
            """, id=doc_id).consume()
            s.run("MATCH (ch:Chunk {ticker: $id}) DETACH DELETE ch", id=doc_id).consume()
    finally:
        driver.close()
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunk_embeddings WHERE doc_id = %s", (doc_id,))
        conn.commit()
    except Exception:  # noqa: BLE001 - no vector table yet means nothing to delete
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest a document into the KGRAG graph and vector index.")
    ap.add_argument("path", type=Path)
    ap.add_argument("--title")
    a = ap.parse_args()
    body = extract_text(a.path, kind_for(a.path.name))
    did = "DOC-" + content_hash(body)[:10]
    print(ingest_text(did, a.title or a.path.stem, body, progress=lambda m, f=None: print(" ", m)))
