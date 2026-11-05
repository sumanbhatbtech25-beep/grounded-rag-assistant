"""Render a PDF page as an image with the cited lines highlighted."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from rag.ingest import page_lines


def render_highlighted_page(pdf_path: str | Path, page: int, lines: list[int],
                            zoom: float = 1.6) -> bytes:
    """PNG bytes of `page` (1-based) with `lines` (1-based line numbers) highlighted."""
    with fitz.open(pdf_path) as doc:
        pg = doc[page - 1]
        wanted = set(lines)
        for ln in page_lines(pg):
            if ln.line_no in wanted:
                rect = fitz.Rect(ln.bbox)
                annot = pg.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 0.85, 0.1))
                annot.update()
                # line-number tag in the left margin
                pg.insert_text((max(2, rect.x0 - 26), rect.y1 - 1), f"L{ln.line_no}",
                               fontsize=7, color=(0.85, 0.1, 0.1))
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), annots=True)
        return pix.tobytes("png")
