"""Structured tool-side protocol models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass(slots=True)
class RetrievalRequest:
    """Normalized retrieval request passed into the retrieval tool boundary."""

    query: str
    with_sources: bool = True

    @classmethod
    def from_query(cls, query: str, *, with_sources: bool = True) -> "RetrievalRequest":
        return cls(query=_normalize_text(query), with_sources=with_sources)


@dataclass(slots=True)
class DepositRequestContext:
    """Normalized deposit request inputs before calling the deposit pipeline."""

    text: str
    image_paths: list[str] = field(default_factory=list)
    invalid_image_paths_json: bool = False
    dropped_image_path_count: int = 0

    @classmethod
    def from_inputs(cls, *, text: str, image_paths_json: str = "[]") -> "DepositRequestContext":
        try:
            parsed_paths = json.loads(image_paths_json) if image_paths_json.strip() else []
            invalid_image_paths_json = False
        except json.JSONDecodeError:
            parsed_paths = []
            invalid_image_paths_json = True
        normalized_candidates = [str(path).strip() for path in parsed_paths]
        image_paths = [path for path in normalized_candidates if path]
        dropped_image_path_count = len(normalized_candidates) - len(image_paths)
        return cls(
            text=str(text or "").strip(),
            image_paths=image_paths,
            invalid_image_paths_json=invalid_image_paths_json,
            dropped_image_path_count=dropped_image_path_count,
        )


@dataclass(slots=True)
class RetrievalResult:
    """Normalized retrieval payload before rendering to tool text."""

    query: str
    result_status: str
    context: str
    sources: list[dict[str, object]] = field(default_factory=list)
    chunk_count: int = 0

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @classmethod
    def empty(cls, *, query: str) -> "RetrievalResult":
        return cls(
            query=query,
            result_status="empty",
            context="",
            sources=[],
            chunk_count=0,
        )

    @classmethod
    def from_prepared_context(cls, query: str, prepared: Any) -> "RetrievalResult":
        merged_chunks = list(getattr(prepared, "merged_chunks", []) or [])
        if not merged_chunks:
            return cls.empty(query=query)
        return cls(
            query=query,
            result_status="completed",
            context=str(getattr(prepared, "context", "") or "").strip(),
            sources=list(getattr(prepared, "sources", []) or []),
            chunk_count=len(merged_chunks),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "result_status": self.result_status,
            "context": self.context,
            "sources": self.sources,
            "chunk_count": self.chunk_count,
            "source_count": self.source_count,
        }


@dataclass(slots=True)
class DepositResult:
    """Normalized deposit payload before rendering to tool text."""

    result_status: str
    message: str
    source_type: str
    local_document_id: str = ""
    feishu_doc_url: str = ""
    wiki_node_token: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pipeline_result(cls, result: Any) -> "DepositResult":
        draft = getattr(result, "draft", None)
        return cls(
            result_status=str(getattr(result, "status", "") or "").strip(),
            message=str(getattr(result, "message", "") or "").strip(),
            source_type=str(getattr(draft, "source_type", "") or "").strip(),
            local_document_id=str(getattr(result, "local_document_id", "") or "").strip(),
            feishu_doc_url=str(getattr(result, "feishu_doc_url", "") or "").strip(),
            wiki_node_token=str(getattr(result, "wiki_node_token", "") or "").strip(),
            metadata=dict(getattr(result, "metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_status": self.result_status,
            "message": self.message,
            "source_type": self.source_type,
            "local_document_id": self.local_document_id,
            "feishu_doc_url": self.feishu_doc_url,
            "wiki_node_token": self.wiki_node_token,
            "metadata": self.metadata,
        }
