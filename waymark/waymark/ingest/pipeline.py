from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from waymark.config import Settings
from waymark.ingest.chunker import Chunker, PageText
from waymark.ingest.images import IMAGE_SUFFIXES, analyse_image
from waymark.ingest.pdf import extract_pdf
from waymark.nlp.annotator import Annotator, PassageInput, PassageNote

log = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".text"})
SUPPORTED = frozenset({".pdf"}) | IMAGE_SUFFIXES | TEXT_SUFFIXES

Progress = Callable[[str, float], None]


@dataclass
class IngestResult:
    kind: str
    pages: int
    notes: list[PassageNote]
    ocr_pages: int = 0
    figures: int = 0
    title: str = ""
    report: dict | None = None
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def detect_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES:
        return "text"
    raise ValueError(f"unsupported file type: {suffix or 'unknown'}")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class Ingestor:
    def __init__(self, settings: Settings, annotator: Annotator):
        self.settings = settings
        self.annotator = annotator

    def ingest(self, path: Path, doc_id: str, progress: Progress | None = None) -> IngestResult:
        started = time.perf_counter()
        report = progress or (lambda stage, fraction: None)
        kind = detect_kind(path)
        report("reading", 0.02)
        pages: list[PageText]
        result = IngestResult(kind=kind, pages=0, notes=[])
        if kind == "pdf":
            extraction = extract_pdf(
                path,
                ocr=self.settings.ocr_enabled,
                min_page_chars=self.settings.ocr_min_page_chars,
                min_image_px=self.settings.ocr_min_image_px,
                progress=lambda done, total: report("reading", 0.02 + 0.33 * done / max(total, 1)),
            )
            pages = extraction.pages
            result.pages = extraction.page_count
            result.ocr_pages = extraction.ocr_pages
            result.figures = extraction.figures_read
            result.title = extraction.title
            result.warnings.extend(extraction.warnings)
        elif kind == "image":
            report("analysing image", 0.1)
            analysis = analyse_image(path, ocr=self.settings.ocr_enabled)
            result.report = analysis.to_dict()
            result.pages = 1
            pages = [PageText(page=1, text=analysis.text, origin="ocr")] if analysis.text.strip() else []
            if not pages:
                result.warnings.append("no readable text found in image")
        else:
            text = _read_text(path)
            sheets = [t for t in text.split("\f")] if "\f" in text else [text]
            pages = [PageText(page=i + 1, text=t) for i, t in enumerate(sheets) if t.strip()]
            result.pages = max(1, len(sheets))
        inputs: list[PassageInput] = []
        chunker = Chunker(target=self.settings.passage_chars)
        for page in pages:
            for chunk in chunker.chunk(page):
                inputs.append(PassageInput(doc_id=doc_id, page=page.page, text=chunk.text, title=chunk.title, origin=page.origin))
        report("understanding", 0.36)
        notes: list[PassageNote] = []
        if inputs:
            for note in self.annotator.annotate(
                inputs,
                progress=lambda done, total: report("understanding", 0.36 + 0.6 * done / max(total, 1)),
            ):
                notes.append(note)
        result.notes = notes
        result.seconds = time.perf_counter() - started
        report("indexed", 1.0)
        log.info("ingested %s: %d pages, %d passages in %.1fs", path.name, result.pages, len(notes), result.seconds)
        return result
