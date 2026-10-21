from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_spaces = re.compile(r"[ \t ]+")
_sentence_end = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])")
_page_number = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "‐": "-", "­": ""}


@dataclass(slots=True)
class PageText:
    page: int
    text: str
    origin: str = "text"


def strip_running_lines(pages: list[PageText]) -> list[PageText]:
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [l.strip() for l in page.text.split("\n") if l.strip()]
        edges = set(lines[:2] + lines[-2:])
        counts.update(_shape(l) for l in edges)
    threshold = max(3, int(len(pages) * 0.4))
    noisy = {shape for shape, c in counts.items() if c >= threshold and shape}
    cleaned = []
    for page in pages:
        lines = page.text.split("\n")
        kept = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            near_edge = i < 3 or i >= len(lines) - 3
            if near_edge and (_shape(stripped) in noisy or _page_number.match(stripped)):
                continue
            kept.append(line)
        cleaned.append(PageText(page=page.page, text="\n".join(kept), origin=page.origin))
    return cleaned


def _shape(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())[:80]


def split_sentences(paragraph: str) -> list[str]:
    parts = _sentence_end.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


GENERIC_HEADINGS = frozenset("""
abstract introduction background conclusion conclusions results discussion methods method methodology materials
related work references bibliography acknowledgements acknowledgments appendix contents summary overview chapter
section preface foreword index keywords notes glossary table figure evaluation experiments experiment analysis
limitations future work data dataset approach implementation setup results and discussion executive summary
""".split())
_numbering = re.compile(r"^(?:#{1,6}\s*|(?:chapter|section|part)\s+\w+[:.\s-]*|[ivxlc]+\.\s+|\d+(?:\.\d+)*\.?\s+)", re.IGNORECASE)


@dataclass(slots=True)
class Chunk:
    text: str
    title: str = ""


def _heading_like(line: str) -> bool:
    stripped = line.strip().lstrip("#").strip()
    if not stripped or len(stripped) > 90 or stripped[-1] in ".,;:!?" or stripped.endswith(("-", "\u2013")):
        return False
    words = stripped.split()
    if len(words) > 12 or not (stripped[0].isupper() or stripped[0].isdigit() or line.lstrip().startswith("#")):
        return False
    letters = [w for w in words if w[:1].isalpha()]
    if not letters:
        return False
    capitalised = sum(1 for w in letters if w[0].isupper())
    return len(words) <= 5 or capitalised / len(letters) >= 0.5


def _heading_title(line: str) -> str:
    cleaned = _numbering.sub("", line.strip()).strip(" #:-\u2013")
    words = cleaned.lower().split()
    if not words or " ".join(words) in GENERIC_HEADINGS or all(w in GENERIC_HEADINGS for w in words):
        return ""
    if not any(ch.isupper() for ch in cleaned):
        return ""
    return cleaned


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        elif out:
            out += " " + line
        else:
            out = line
    return " ".join(out.split())


class Chunker:
    def __init__(self, target: int = 900):
        self.target = target
        self.heading = ""

    def chunk(self, page: PageText) -> list[Chunk]:
        text = page.text
        for bad, good in _ligatures.items():
            text = text.replace(bad, good)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _spaces.sub(" ", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        size = 0
        target = self.target

        def flush() -> None:
            nonlocal current, size
            if current:
                body = " ".join(current).strip()
                if len(body) > 1:
                    chunks.append(Chunk(text=body, title=self.heading))
            current, size = [], 0

        for block in re.split(r"\n\s*\n", text):
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue
            if _heading_like(lines[0]) and (len(lines) == 1 or len(lines[0]) < 0.6 * max(len(l) for l in lines[1:])):
                flush()
                self.heading = _heading_title(lines[0])
                lines = lines[1:]
                if not lines:
                    continue
            para = _join_lines(lines)
            pieces = split_sentences(para) if len(para) > target else [para]
            for sent in pieces:
                if len(sent) > target * 2:
                    for i in range(0, len(sent), target):
                        if size and size + target > target:
                            flush()
                        current.append(sent[i:i + target])
                        size += target
                    continue
                if size + len(sent) > target and current:
                    flush()
                current.append(sent)
                size += len(sent) + 1
            if size >= target * 0.35:
                flush()
        if current:
            tail = " ".join(current)
            if chunks and len(tail) < target * 0.2 and chunks[-1].title == self.heading:
                chunks[-1] = Chunk(text=chunks[-1].text + " " + tail, title=chunks[-1].title)
                current = []
            else:
                flush()
        return chunks
