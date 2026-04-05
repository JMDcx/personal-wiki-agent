"""Schemas for docreader service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseRequest:
    """Input for docreader parsing."""

    file_name: str | None = None
    file_type: str | None = None
    file_content: bytes | None = None
    url: str | None = None
    title: str = ""
    parser_engine: str = ""
    engine_overrides: dict[str, Any] = field(default_factory=dict)
