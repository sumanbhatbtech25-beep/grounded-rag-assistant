"""Index PDFs into the vector store."""
from __future__ import annotations

from pathlib import Path

import config
from rag.ingest import load_pdf
from rag.security import detect_injection
from rag.store import VectorStore


def index_pdf(store: VectorStore, pdf_path: str | Path,
              size: int = config.CHUNK_LINES, overlap: int = config.CHUNK_OVERLAP) -> int:
    """(Re)index one PDF. Returns the number of chunks stored (0 = no extractable text)."""
    pdf_path = Path(pdf_path)
    store.delete_source(pdf_path.name)
    chunks = load_pdf(pdf_path, size=size, overlap=overlap)
    return store.add(chunks) if chunks else 0


def index_pdf_with_report(store: VectorStore, pdf_path: str | Path) -> tuple[int, int]:
    """Index a PDF and also count passages that contain instruction-like text
    (they stay searchable but are quarantined from the LLM at question time)."""
    pdf_path = Path(pdf_path)
    store.delete_source(pdf_path.name)
    chunks = load_pdf(pdf_path, size=config.CHUNK_LINES, overlap=config.CHUNK_OVERLAP)
    flagged = sum(1 for c in chunks if detect_injection(c.text))
    return (store.add(chunks) if chunks else 0), flagged


def index_folder(store: VectorStore, folder: str | Path = config.DOCS_DIR) -> dict[str, int]:
    return {p.name: index_pdf(store, p) for p in sorted(Path(folder).glob("*.pdf"))}
