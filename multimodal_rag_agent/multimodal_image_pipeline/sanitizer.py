"""OCR text sanitizer."""

from __future__ import annotations

import re


def sanitize_ocr_text(text: str) -> str:
    """Normalize OCR output and drop obvious non-content responses."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    lowered = cleaned.lower()
    if lowered in {"", "no text content.", "no text content", "none", "n/a"}:
        return ""
    return cleaned
