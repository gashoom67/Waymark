from __future__ import annotations

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from waymark.ingest.chunker import PageText, strip_running_lines
from waymark.ingest.ocr import OcrEngine, lines_to_text

log = logging.getLogger(__name__)


@dataclass
class PdfExtraction:
    pages: list[PageText]
    page_count: int
    title: str = ""
    ocr_pages: int = 0
    figures_read: int = 0
    warnings: list[str] = field(default_factory=list)


def _page_image(page: fitz.Page, zoom: float = 2.0) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _page_text(page) -> str:
    blocks = page.get_text("blocks", sort=True) or []
    parts = [b[4].strip() for b in blocks if len(b) > 6 and b[6] == 0 and b[4].strip()]
    return "\n\n".join(parts)


def render_page_png(path: Path, page_number: int, zoom: float = 1.6) -> bytes:
    with fitz.open(path) as doc:
        if page_number < 1 or page_number > doc.page_count:
            raise IndexError(page_number)
        pix = doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")


def extract_pdf(
    path: Path,
    ocr: bool = True,
    min_page_chars: int = 60,
    min_image_px: int = 120,
    figure_budget: int = 40,
    progress: Callable[[int, int], None] | None = None,
) -> PdfExtraction:
    pages: list[PageText] = []
    ocr_pages = 0
    figures = 0
    warnings: list[str] = []
    engine = OcrEngine.shared() if ocr else None
    if ocr and engine is None:
        warnings.append("OCR engine unavailable; scanned pages and figures were skipped")
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        title = (doc.metadata or {}).get("title") or ""
        total = doc.page_count
        for index in range(total):
            page = doc[index]
            text = _page_text(page)
            if len(text.strip()) < min_page_chars and engine is not None:
                try:
                    lines, _ = engine.read(_page_image(page))
                    scanned = lines_to_text(lines)
                    if scanned.strip():
                        text = scanned
                        ocr_pages += 1
                        pages.append(PageText(page=index + 1, text=text, origin="ocr"))
                        if progress:
                            progress(index + 1, total)
                        continue
                except Exception as exc:
                    warnings.append(f"OCR failed on page {index + 1}: {exc}")
            if text.strip():
                pages.append(PageText(page=index + 1, text=text, origin="text"))
            if engine is not None and figures < figure_budget:
                figures += _read_figures(doc, page, index + 1, engine, min_image_px, figure_budget - figures, pages, warnings)
            if progress:
                progress(index + 1, total)
    cleaned = strip_running_lines([p for p in pages if p.origin == "text"])
    merged = cleaned + [p for p in pages if p.origin != "text"]
    merged.sort(key=lambda p: (p.page, p.origin != "text"))
    return PdfExtraction(pages=merged, page_count=total, title=title.strip(), ocr_pages=ocr_pages, figures_read=figures, warnings=warnings)


def _read_figures(doc, page, page_no: int, engine: OcrEngine, min_px: int, budget: int, out: list[PageText], warnings: list[str]) -> int:
    used = 0
    seen: set[int] = set()
    for info in page.get_images(full=True):
        if used >= budget:
            break
        xref = info[0]
        if xref in seen:
            continue
        seen.add(xref)
        width, height = info[2], info[3]
        if width < min_px or height < min_px or width * height > 25_000_000:
            continue
        try:
            raw = doc.extract_image(xref)
            image = Image.open(io.BytesIO(raw["image"]))
            image.load()
        except Exception:
            continue
        try:
            lines, _ = engine.read(image, min_score=0.6)
        except Exception as exc:
            warnings.append(f"figure OCR failed on page {page_no}: {exc}")
            continue
        used += 1
        text = lines_to_text(lines)
        if len(text) >= 12 and len(lines) >= 2:
            out.append(PageText(page=page_no, text=text, origin="figure"))
    return used
