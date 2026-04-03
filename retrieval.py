"""Retrieval helpers for the Feishu Wiki RAG example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from feishu_wiki_rag_agent.config import Settings, get_settings
from feishu_wiki_rag_agent.schemas import IndexManifest, utcnow_iso


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create the embeddings model used for index and query."""
    resolved = settings or get_settings()
    kwargs: dict[str, Any] = {"model": resolved.embedding_model}
    if resolved.embedding_api_key:
        kwargs["api_key"] = resolved.embedding_api_key
    if resolved.embedding_base_url:
        kwargs["base_url"] = resolved.embedding_base_url
    return OpenAIEmbeddings(**kwargs)


def load_vector_store(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Load the local Chroma vector store."""
    resolved = settings or get_settings()
    return Chroma(
        collection_name=resolved.chroma_collection_name,
        persist_directory=str(resolved.chroma_dir),
        embedding_function=embeddings or get_embeddings(resolved),
    )


def search_knowledge(
    query: str,
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
    embeddings: Embeddings | None = None,
) -> list[Document]:
    """Search the local vector store for relevant Feishu document chunks."""
    resolved = settings or get_settings()
    vector_store = load_vector_store(resolved, embeddings=embeddings)
    return vector_store.similarity_search(query, k=top_k or resolved.rag_top_k)


def format_retrieved_context(documents: list[Document]) -> str:
    """Format retrieved chunks for the RAG tool response."""
    if not documents:
        return "当前索引中未找到相关内容。"

    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        title = str(metadata.get("title", "Untitled"))
        source_url = str(metadata.get("source_url", ""))
        header = f"[{index}] {title}"
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


def chroma_dir_has_index(path: Path) -> bool:
    """Return whether the Chroma persistence directory contains files."""
    return path.exists() and any(path.iterdir())
