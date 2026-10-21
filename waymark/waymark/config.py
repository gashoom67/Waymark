from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("WAYMARK_DATA_DIR", "data")).resolve())
    spacy_model: str = field(default_factory=lambda: os.environ.get("WAYMARK_SPACY_MODEL", "en_core_web_md"))
    max_upload_mb: int = field(default_factory=lambda: _env_int("WAYMARK_MAX_UPLOAD_MB", 200))
    ocr_enabled: bool = field(default_factory=lambda: _env_bool("WAYMARK_OCR", True))
    ocr_min_page_chars: int = field(default_factory=lambda: _env_int("WAYMARK_OCR_MIN_PAGE_CHARS", 60))
    ocr_min_image_px: int = field(default_factory=lambda: _env_int("WAYMARK_OCR_MIN_IMAGE_PX", 120))
    nlp_batch_size: int = field(default_factory=lambda: _env_int("WAYMARK_NLP_BATCH", 48))
    passage_chars: int = field(default_factory=lambda: _env_int("WAYMARK_PASSAGE_CHARS", 900))
    host: str = field(default_factory=lambda: os.environ.get("WAYMARK_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("WAYMARK_PORT", 8000))

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.workspaces_dir.mkdir(parents=True, exist_ok=True)
    return settings
