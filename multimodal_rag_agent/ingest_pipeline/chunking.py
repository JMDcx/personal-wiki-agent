"""Markdown chunking with protected patterns and header tracking."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from multimodal_rag_agent.models import ChunkRecord, ResolvedImage

PROTECTED_PATTERNS = [
    re.compile(r"(?s)\$\$.*?\$\$"),
    re.compile(r"!\[[^\]]*\]\([^)]+\)"),
    re.compile(r"\[[^\]]*\]\([^)]+\)"),
    re.compile(r"(?m)[ ]*(?:\|[^|\n]*)+\|[\r\n]+\s*(?:\|\s*:?-{3,}:?\s*)+\|[\r\n]+"),
    re.compile(r"(?m)[ ]*(?:\|[^|\n]*)+\|[\r\n]+"),
    re.compile(r"(?s)```(?:\w+)?[\r\n].*?```"),
]
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
IMAGE_URL_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<url>[^)\n]+)\)")


@dataclass
class TextChunk:
    start: int
    end: int
    content: str
    headers: list[str]


class HeaderTracker:
    """Lightweight markdown header tracker."""

    def headers_for_offset(self, text: str, offset: int) -> list[str]:
        active: dict[int, str] = {}
        for match in HEADER_RE.finditer(text):
            if match.start() > offset:
                break
            level = len(match.group(1))
            active[level] = match.group(2).strip()
            active = {key: value for key, value in active.items() if key <= level}
        return [active[key] for key in sorted(active)]


class MarkdownChunker:
    """Split markdown while keeping images, links, tables, code, and formulas intact."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_tracker = HeaderTracker()

    def split(self, document_id: str, markdown: str, metadata: dict[str, object] | None = None) -> list[ChunkRecord]:
        base_metadata = dict(metadata or {})
        chunks = [
            ChunkRecord(
                chunk_id=uuid.uuid4().hex,
                document_id=document_id,
                chunk_type="text",
                content=text_chunk.content,
                metadata={
                    **base_metadata,
                    "headers": text_chunk.headers,
                    "section_path": " > ".join(text_chunk.headers),
                    "start_offset": text_chunk.start,
                    "end_offset": text_chunk.end,
                    "page_number": base_metadata.get("page_number", ""),
                    "public_image_url": self._first_image_url(text_chunk.content),
                    "ocr_text": "",
                    "caption_text": "",
                },
            )
            for text_chunk in self._split_markdown(markdown)
            if text_chunk.content.strip()
        ]
        return chunks

    def find_parent_chunk(self, image: ResolvedImage, chunks: list[ChunkRecord]) -> str | None:
        for chunk in chunks:
            if image.public_url in chunk.content:
                return chunk.chunk_id
        return chunks[0].chunk_id if chunks else None

    def _split_markdown(self, text: str) -> list[TextChunk]:
        if not text:
            return []
        protected = self._protected_spans(text)
        units = self._build_units(text, protected)
        results: list[TextChunk] = []
        current_parts: list[str] = []
        current_start = 0
        current_len = 0
        current_end = 0
        for start, end, part in units:
            part_len = len(part)
            if current_parts and current_len + part_len > self.chunk_size:
                content = "".join(current_parts)
                results.append(
                    TextChunk(
                        start=current_start,
                        end=current_end,
                        content=content,
                        headers=self.header_tracker.headers_for_offset(text, current_start),
                    )
                )
                overlap_text = content[-self.chunk_overlap :] if self.chunk_overlap else ""
                current_parts = [overlap_text, part] if overlap_text else [part]
                current_start = max(0, end - len("".join(current_parts)))
                current_len = len("".join(current_parts))
            else:
                if not current_parts:
                    current_start = start
                current_parts.append(part)
                current_len += part_len
            current_end = end
        if current_parts:
            content = "".join(current_parts)
            results.append(
                TextChunk(
                    start=current_start,
                    end=current_end,
                    content=content,
                    headers=self.header_tracker.headers_for_offset(text, current_start),
                )
            )
        return results

    def _protected_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for pattern in PROTECTED_PATTERNS:
            spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
        spans.sort()
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if not merged or start >= merged[-1][1]:
                merged.append((start, end))
        return merged

    def _build_units(self, text: str, protected: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
        separators = ["\n\n", "\n", "。", " "]
        units: list[tuple[int, int, str]] = []
        cursor = 0
        for start, end in protected:
            if cursor < start:
                units.extend(self._split_plain(text, cursor, start, separators))
            units.append((start, end, text[start:end]))
            cursor = end
        if cursor < len(text):
            units.extend(self._split_plain(text, cursor, len(text), separators))
        return units

    def _split_plain(self, text: str, start: int, end: int, separators: list[str]) -> list[tuple[int, int, str]]:
        segment = text[start:end]
        if not segment:
            return []
        for sep in separators:
            if sep in segment:
                units: list[tuple[int, int, str]] = []
                cursor = start
                for part in segment.split(sep):
                    piece = part
                    if cursor != start:
                        piece = sep + piece
                    units.append((cursor, cursor + len(piece), piece))
                    cursor += len(piece)
                return [unit for unit in units if unit[2]]
        return [(start, end, segment)]

    @staticmethod
    def _first_image_url(text: str) -> str:
        match = IMAGE_URL_RE.search(text)
        return match.group("url") if match else ""
