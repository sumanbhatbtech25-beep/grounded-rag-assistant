"""Index PDFs into the vector store - safely.

A file is only added to docs/ and the index if EVERY page can be read and the passages
are stored successfully. Damaged files are rejected (uploads) or moved to
docs/_rejected/ (folder re-index), so they can never break the app.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import config
from rag.ingest import DamagedPDFError, load_pdf, load_pdf_bytes
from rag.security import detect_injection, safe_filename, validate_pdf
from rag.store import VectorStore

REJECTED_DIR_NAME = "_rejected"


@dataclass
class IndexResult:
    name: str
    ok: bool
    passages: int = 0
    flagged: int = 0
    reason: str = ""


def index_pdf(store: VectorStore, pdf_path: str | Path,
              size: int = config.CHUNK_LINES, overlap: int = config.CHUNK_OVERLAP) -> int:
    """(Re)index one PDF already on disk. Returns passages stored. Raises DamagedPDFError."""
    pdf_path = Path(pdf_path)
    chunks = load_pdf(pdf_path, size=size, overlap=overlap)
    if not chunks:
        store.delete_source(pdf_path.name)
        return 0
    return store.replace_source(pdf_path.name, chunks)


def safe_add_pdf(store: VectorStore, data: bytes, original_name: str,
                 docs_dir: Path = config.DOCS_DIR) -> IndexResult:
    """Validate -> parse every page in memory -> index -> only then save to docs/."""
    ok, reason, name = validate_pdf(data, original_name)
    if not ok:
        return IndexResult(name, False, reason=reason)
    try:
        chunks = load_pdf_bytes(data, name, size=config.CHUNK_LINES, overlap=config.CHUNK_OVERLAP)
    except DamagedPDFError as e:
        return IndexResult(name, False, reason=f"damaged PDF: {e}")
    if not chunks:
        return IndexResult(name, False, reason="no selectable text (scanned PDF? run OCR first)")

    flagged = sum(1 for c in chunks if detect_injection(c.text))
    try:
        n = store.replace_source(name, chunks)
    except Exception as e:
        return IndexResult(name, False, reason=f"could not be indexed ({type(e).__name__})")

    # write the file atomically, only after indexing succeeded
    docs_dir.mkdir(parents=True, exist_ok=True)
    tmp = docs_dir / f".{name}.part"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, docs_dir / name)
    except OSError as e:
        store.delete_source(name)
        tmp.unlink(missing_ok=True)
        return IndexResult(name, False, reason=f"could not be saved ({e.strerror})")
    return IndexResult(name, True, passages=n, flagged=flagged)


def index_pdf_with_report(store: VectorStore, pdf_path: str | Path) -> tuple[int, int]:
    pdf_path = Path(pdf_path)
    chunks = load_pdf(pdf_path, size=config.CHUNK_LINES, overlap=config.CHUNK_OVERLAP)
    flagged = sum(1 for c in chunks if detect_injection(c.text))
    return (store.replace_source(pdf_path.name, chunks) if chunks else 0), flagged


def _quarantine(path: Path, reason: str) -> None:
    dest_dir = path.parent / REJECTED_DIR_NAME
    dest_dir.mkdir(exist_ok=True)
    try:
        shutil.move(str(path), dest_dir / path.name)
        (dest_dir / f"{path.name}.reason.txt").write_text(reason, encoding="utf-8")
    except OSError:
        pass


def index_folder_report(store: VectorStore, folder: str | Path = config.DOCS_DIR) -> list[IndexResult]:
    """Re-index every PDF in `folder`. Damaged files are moved to folder/_rejected/
    and removed from the index; the rest are indexed normally. Never raises."""
    results = []
    for p in sorted(Path(folder).glob("*.pdf")):
        try:
            data = p.read_bytes()
            ok, reason, _ = validate_pdf(data, p.name)
            if not ok:
                raise DamagedPDFError(reason)
            chunks = load_pdf_bytes(data, p.name, size=config.CHUNK_LINES, overlap=config.CHUNK_OVERLAP)
            if not chunks:
                store.delete_source(p.name)
                results.append(IndexResult(p.name, True, 0, reason="no selectable text"))
                continue
            n = store.replace_source(p.name, chunks)
            results.append(IndexResult(p.name, True, n, sum(1 for c in chunks if detect_injection(c.text))))
        except Exception as e:
            reason = str(e) if isinstance(e, DamagedPDFError) else f"{type(e).__name__}: {e}"
            try:
                store.delete_source(p.name)
            except Exception:
                pass
            _quarantine(p, reason)
            results.append(IndexResult(p.name, False, reason=reason))
    return results


def index_folder(store: VectorStore, folder: str | Path = config.DOCS_DIR) -> dict[str, int]:
    return {r.name: r.passages for r in index_folder_report(store, folder) if r.ok}
