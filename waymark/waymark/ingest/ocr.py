from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

MAX_SIDE = 2200


@dataclass(slots=True)
class OcrLine:
    text: str
    score: float
    box: list[list[float]]

    @property
    def top(self) -> float:
        return min(p[1] for p in self.box)

    @property
    def left(self) -> float:
        return min(p[0] for p in self.box)

    @property
    def bottom(self) -> float:
        return max(p[1] for p in self.box)

    @property
    def height(self) -> float:
        return self.bottom - self.top


class OcrEngine:
    _instance: "OcrEngine | None" = None
    _guard = threading.Lock()

    def __init__(self):
        for name in ("RapidOCR", "rapidocr"):
            logging.getLogger(name).setLevel(logging.ERROR)
        from rapidocr import RapidOCR

        self._engine = RapidOCR()
        self._lock = threading.Lock()

    @classmethod
    def shared(cls) -> "OcrEngine | None":
        with cls._guard:
            if cls._instance is None:
                try:
                    cls._instance = cls()
                except Exception:
                    log.exception("OCR engine unavailable")
                    return None
            return cls._instance

    def read(self, image: Image.Image, min_score: float = 0.5) -> tuple[list[OcrLine], float]:
        rgb = image.convert("RGB")
        scale = 1.0
        longest = max(rgb.size)
        if longest > MAX_SIDE:
            scale = MAX_SIDE / longest
            rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.LANCZOS)
        elif longest < 640:
            scale = 640 / longest
            rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.LANCZOS)
        array = np.asarray(rgb)[:, :, ::-1].copy()
        with self._lock:
            result = self._engine(array)
        lines: list[OcrLine] = []
        if result is None or result.txts is None:
            return lines, scale
        for text, score, box in zip(result.txts, result.scores, result.boxes):
            if float(score) < min_score or not str(text).strip():
                continue
            pts = [[float(x) / scale, float(y) / scale] for x, y in np.asarray(box).tolist()]
            lines.append(OcrLine(text=str(text).strip(), score=float(score), box=pts))
        return lines, scale


def lines_to_text(lines: list[OcrLine]) -> str:
    if not lines:
        return ""
    ordered = sorted(lines, key=lambda l: (round(l.top / max(8.0, l.height * 0.6)), l.left))
    heights = [l.height for l in ordered if l.height > 0]
    unit = float(np.median(heights)) if heights else 12.0
    paragraphs: list[list[str]] = [[]]
    previous: OcrLine | None = None
    for line in ordered:
        if previous is not None and line.top - previous.bottom > unit * 1.1:
            paragraphs.append([])
        paragraphs[-1].append(line.text)
        previous = line
    blocks = []
    for para in paragraphs:
        joined = ""
        for piece in para:
            if joined.endswith("-") and piece[:1].islower():
                joined = joined[:-1] + piece
            elif joined:
                joined += " " + piece
            else:
                joined = piece
        if joined:
            blocks.append(joined)
    return "\n\n".join(blocks)
