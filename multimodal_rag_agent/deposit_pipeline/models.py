"""Shared models for the knowledge deposit flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DepositRequest:
    text: str = ""
    urls: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    target_space_id: str = ""
    target_parent_node_token: str = ""
    auto_write: bool | None = None


@dataclass(slots=True)
class SourceMaterial:
    source_type: str
    source_uri: str
    title: str
    author: str = ""
    published_at: str = ""
    raw_markdown: str = ""
    extra_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeDraft:
    source_type: str
    source_uri: str
    source_title: str
    author: str
    published_at: str
    raw_content_markdown: str
    summary_markdown: str
    key_points: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    feishu_doc_title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DepositResult:
    status: str
    message: str
    draft: KnowledgeDraft
    final_markdown: str
    local_document_id: str = ""
    feishu_doc_token: str = ""
    feishu_doc_url: str = ""
    wiki_node_token: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "draft": self.draft.to_dict(),
            "final_markdown": self.final_markdown,
            "local_document_id": self.local_document_id,
            "feishu_doc_token": self.feishu_doc_token,
            "feishu_doc_url": self.feishu_doc_url,
            "wiki_node_token": self.wiki_node_token,
            "metadata": self.metadata,
        }
