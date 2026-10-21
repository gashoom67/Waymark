from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageOps

from waymark.ingest.ocr import OcrEngine, OcrLine, lines_to_text

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"})
_number = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


@dataclass
class ImageReport:
    width: int
    height: int
    format: str
    mode: str
    kind: str
    kind_reasons: list[str]
    brightness: float
    contrast: float
    colorfulness: float
    edge_density: float
    text_coverage: float
    palette: list[dict]
    lines: list[dict]
    numbers: list[str]
    exif: dict = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "mode": self.mode,
            "kind": self.kind,
            "kind_reasons": self.kind_reasons,
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
            "colorfulness": round(self.colorfulness, 3),
            "edge_density": round(self.edge_density, 3),
            "text_coverage": round(self.text_coverage, 3),
            "palette": self.palette,
            "lines": self.lines,
            "numbers": self.numbers,
            "exif": self.exif,
            "text": self.text,
        }


def _palette(image: Image.Image, colors: int = 6) -> list[dict]:
    small = image.convert("RGB")
    small.thumbnail((160, 160))
    quant = small.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    raw = quant.getpalette() or []
    counts = sorted(quant.getcolors() or [], reverse=True)
    total = sum(c for c, _ in counts) or 1
    out = []
    for count, idx in counts[:colors]:
        r, g, b = raw[idx * 3: idx * 3 + 3]
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "share": round(count / total, 3)})
    return out


def _exif(image: Image.Image) -> dict:
    try:
        data = image.getexif()
    except Exception:
        return {}
    keep = {"DateTime", "DateTimeOriginal", "Make", "Model", "Software", "Orientation"}
    out = {}
    for tag, value in data.items():
        name = ExifTags.TAGS.get(tag, str(tag))
        if name in keep:
            out[name] = str(value)[:80]
    return out


def _metrics(image: Image.Image) -> tuple[float, float, float, float, int]:
    rgb = image.convert("RGB")
    rgb.thumbnail((640, 640))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    gy, gx = np.gradient(lum)
    edges = float((np.hypot(gx, gy) > 0.08).mean())
    quant = (arr * 15).astype(np.uint8)
    unique = len(np.unique(quant.reshape(-1, 3), axis=0))
    return float(lum.mean()), float(lum.std()), colorfulness, edges, unique


def _classify(coverage: float, colorfulness: float, edges: float, unique: int, lines: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if coverage > 0.12 and (lines >= 6 or colorfulness < 0.08) and colorfulness < 0.25:
        reasons.append(f"text covers {coverage:.0%} of the frame across {lines} lines")
        reasons.append("low colour variance typical of paper or scans")
        return "document", reasons
    if lines >= 4 and unique < 900 and edges > 0.04:
        reasons.append(f"{lines} text lines over flat colour regions")
        reasons.append("sharp rectilinear edges suggest an interface capture")
        return "screenshot", reasons
    if unique < 700 and edges > 0.02:
        reasons.append("limited palette with crisp edges")
        if lines:
            reasons.append(f"{lines} short labels detected")
        return "diagram", reasons
    reasons.append("continuous tones and a wide palette")
    if lines:
        reasons.append(f"incidental text: {lines} lines")
    return "photo", reasons


def analyse_image(path: Path, ocr: bool = True) -> ImageReport:
    with Image.open(path) as opened:
        opened.load()
        fmt = opened.format or path.suffix.lstrip(".").upper()
        mode = opened.mode
        exif = _exif(opened)
        image = ImageOps.exif_transpose(opened.convert("RGBA") if opened.mode in {"P", "LA"} else opened)
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        else:
            image = image.convert("RGB")
    width, height = image.size
    brightness, contrast, colorfulness, edges, unique = _metrics(image)
    lines: list[OcrLine] = []
    engine = OcrEngine.shared() if ocr else None
    if engine is not None:
        lines, _ = engine.read(image)
    area = float(width * height) or 1.0
    coverage = 0.0
    for line in lines:
        xs = [p[0] for p in line.box]
        ys = [p[1] for p in line.box]
        coverage += max(0.0, (max(xs) - min(xs))) * max(0.0, (max(ys) - min(ys)))
    coverage = min(1.0, coverage / area)
    kind, reasons = _classify(coverage, colorfulness, edges, unique, len(lines))
    text = lines_to_text(lines)
    numbers = list(dict.fromkeys(_number.findall(text)))[:40]
    return ImageReport(
        width=width,
        height=height,
        format=fmt,
        mode=mode,
        kind=kind,
        kind_reasons=reasons,
        brightness=brightness,
        contrast=contrast,
        colorfulness=colorfulness,
        edge_density=edges,
        text_coverage=coverage,
        palette=_palette(image),
        lines=[
            {
                "text": l.text,
                "score": round(l.score, 3),
                "box": [[round(x / width, 4), round(y / height, 4)] for x, y in l.box],
            }
            for l in lines
        ],
        numbers=numbers,
        exif=exif,
        text=text,
    )
