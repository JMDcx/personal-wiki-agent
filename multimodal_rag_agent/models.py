"""Shared models for multimodal RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ImageRef:
    filename: str
    original_ref: str
    mime_type: str
    image_data: bytes = b""


@dataclass
class ParsedDocument:
    markdown_content: str
    image_refs: list[ImageRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedImage:
    image_id: str
    original_ref: str
    stored_path: str
    public_url: str
    mime_type: str
    source_type: str


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    chunk_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    image_id: str | None = None
    parent_chunk_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "chunk_id": self.chunk_id,
                "document_id": self.document_id,
                "chunk_type": self.chunk_type,
                "image_id": self.image_id or "",
                "parent_chunk_id": self.parent_chunk_id or "",
                "content": self.content,
            }
        )
        return payload


@dataclass
class IngestResult:
    document_id: str
    status: str
    chunk_count: int
    image_count: int
    resolved_markdown: str
    chunks: list[ChunkRecord] = field(default_factory=list)
    images: list[ResolvedImage] = field(default_factory=list)


@dataclass
class QueryBundle:
    raw_query: str
    rewritten_query: str
    query_keywords: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    chunk_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    query_bundle: QueryBundle | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
