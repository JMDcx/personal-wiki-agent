"""Retrieval helpers backed by the multimodal Qdrant pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feishu_wiki_rag_agent.config import Settings, get_settings
from feishu_wiki_rag_agent.schemas import IndexManifest, utcnow_iso
from multimodal_rag_agent.config import get_multimodal_settings
from multimodal_rag_agent.rag_query_pipeline.retrieval import Retriever

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:  # pragma: no cover - local fallback
    @dataclass
    class Document:
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)


def search_knowledge(
    query: str,
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
) -> list[Document]:
    """Search the Qdrant-backed multimodal store and adapt results to LangChain documents."""
    _ = settings or get_settings()
    retriever = Retriever(get_multimodal_settings())
    chunks = retriever.retrieve(query, top_k=top_k)
    return [
        Document(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type,
                "title": chunk.metadata.get("title", "Untitled"),
                "source_url": chunk.metadata.get("source_uri", chunk.metadata.get("source_url", "")),
                "source_uri": chunk.metadata.get("source_uri", ""),
            },
        )
        for chunk in chunks
    ]


def format_retrieved_context(documents: list[Document]) -> str:
    """Format retrieved chunks for debug or fallback response rendering."""
    if not documents:
        return "当前索引中未找到相关内容。"

    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        title = str(metadata.get("title", "Untitled"))
        source_url = str(metadata.get("source_url", metadata.get("source_uri", "")))
        chunk_type = str(metadata.get("chunk_type", "text"))
        header = f"[{index}] {title} [{chunk_type}]"
        if source_url:
            header += f" ({source_url})"
        sections.append(f"{header}\n{document.page_content.strip()}")
    return "\n\n".join(sections)


def write_index_manifest(
    root_tokens: list[str],
    document_count: int,
    chunk_count: int,
    settings: Settings | None = None,
) -> IndexManifest:
    """Write the local manifest describing the latest index build."""
    resolved = settings or get_settings()
    resolved.ensure_directories()
    manifest = IndexManifest(
        indexed_at=utcnow_iso(),
        root_tokens=root_tokens,
        document_count=document_count,
        chunk_count=chunk_count,
    )
    resolved.manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_index_manifest(settings: Settings | None = None) -> dict[str, Any] | None:
    """Load the local manifest if one exists."""
    resolved = settings or get_settings()
    if not resolved.manifest_path.exists():
        return None
    return json.loads(resolved.manifest_path.read_text(encoding="utf-8"))
