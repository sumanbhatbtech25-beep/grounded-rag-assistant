"""PDF -> lines (with page + line numbers + bounding boxes) -> chunks.

Line numbers are counted per page, top to bottom, over non-empty text lines.
The same function (`page_lines`) is used for ingestion and for highlighting,
so a citation "page 3, lines 12-14" always points at the same pixels.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF


@dataclass
class Line:
    text: str
    page: int        # 1-based page number
    line_no: int     # 1-based line number within the page
    bbox: tuple      # (x0, y0, x1, y1) in PDF points


@dataclass
class Chunk:
    id: str
    source: str      # file name, e.g. "club_rules.pdf"
    page: int
    line_start: int
    line_end: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


def page_lines(page: "fitz.Page") -> list[Line]:
    """Return the visual text lines of one page in reading order."""
    data = page.get_text("dict", sort=True)
    rows: list[tuple[tuple, str]] = []
    for block in data["blocks"]:
        if block.get("type", 0) != 0:  # skip images
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                rows.append((tuple(line["bbox"]), text))

    # Merge fragments that sit on the same baseline (e.g. "1." + "Members must...")
    merged: list[list] = []
    for bbox, text in sorted(rows, key=lambda r: (round(r[0][1]), r[0][0])):
        if merged and abs(merged[-1][0][1] - bbox[1]) < 2 and abs(merged[-1][0][3] - bbox[3]) < 2:
            prev_bbox, prev_text = merged[-1]
            merged[-1] = [
                (min(prev_bbox[0], bbox[0]), min(prev_bbox[1], bbox[1]),
                 max(prev_bbox[2], bbox[2]), max(prev_bbox[3], bbox[3])),
                f"{prev_text} {text}",
            ]
        else:
            merged.append([bbox, text])

    pno = page.number + 1
    return [
        Line(text=t, page=pno, line_no=i, bbox=tuple(round(v, 2) for v in b))
        for i, (b, t) in enumerate(merged, start=1)
    ]


def extract_lines(pdf_path: str | Path) -> list[Line]:
    out: list[Line] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            out.extend(page_lines(page))
    return out


def chunk_lines(lines: list[Line], source: str, size: int = 8, overlap: int = 2) -> list[Chunk]:
    """Sliding window of `size` lines per chunk. Chunks never cross a page boundary,
    so every chunk maps to exactly one page and a contiguous line range."""
    if size <= 0:
        raise ValueError("size must be > 0")
    step = max(1, size - max(0, overlap))

    by_page: dict[int, list[Line]] = {}
    for ln in lines:
        by_page.setdefault(ln.page, []).append(ln)

    chunks: list[Chunk] = []
    for page, plines in sorted(by_page.items()):
        start = 0
        while start < len(plines):
            window = plines[start:start + size]
            cid = hashlib.sha1(f"{source}|{page}|{window[0].line_no}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                id=cid, source=source, page=page,
                line_start=window[0].line_no, line_end=window[-1].line_no,
                lines=[l.text for l in window],
            ))
            if start + size >= len(plines):
                break
            start += step
    return chunks


def load_pdf(pdf_path: str | Path, size: int = 8, overlap: int = 2) -> list[Chunk]:
    pdf_path = Path(pdf_path)
    lines = extract_lines(pdf_path)
    return chunk_lines(lines, pdf_path.name, size=size, overlap=overlap)
