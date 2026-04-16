"""Helpers for preparing image payloads for VLM requests."""

from __future__ import annotations

import base64
from io import BytesIO


def build_normalized_image_data_uri(image_path: str) -> str:
    """Return a stable PNG data URI for OCR/caption models."""
    from PIL import Image, ImageOps

    with Image.open(image_path) as image:
        image.load()
        normalized = ImageOps.exif_transpose(image)
        if getattr(normalized, "is_animated", False):
            normalized.seek(0)
            normalized = normalized.copy()
        else:
            normalized = normalized.copy()

    if normalized.mode not in {"RGB", "RGBA"}:
        if "A" in normalized.getbands():
            normalized = normalized.convert("RGBA")
        else:
            normalized = normalized.convert("RGB")

    buffer = BytesIO()
    normalized.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"
