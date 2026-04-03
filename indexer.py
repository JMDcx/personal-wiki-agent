"""Manual indexing command for Feishu Wiki/Docs content."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from rich.console import Console

from feishu_wiki_rag_agent.config import Settings, get_settings
from feishu_wiki_rag_agent.feishu_client import FeishuClient
from feishu_wiki_rag_agent.retrieval import get_embeddings, write_index_manifest

console = Console()


def rebuild_index(
    documents: list[Document],
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
) -> dict[str, Any]:
    """Rebuild the local Chroma index from Feishu documents."""
    resolved = settings or get_settings()
    resolved.ensure_directories()
    non_empty_documents = [document for document in documents if document.page_content and document.page_content.strip()]
    if not non_empty_documents:
        msg = (
            "No non-empty Feishu documents were collected from the configured roots. "
            "Check that the root tokens point to Wiki nodes with readable doc/docx content."
        )
        raise RuntimeError(msg)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(non_empty_documents)
    non_empty_chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
    if not non_empty_chunks:
        msg = (
            "Documents were fetched, but all generated chunks were empty. "
            "This usually means the selected Feishu pages have no extractable raw text content."
        )
        raise RuntimeError(msg)

    shutil.rmtree(resolved.chroma_dir, ignore_errors=True)
    resolved.chroma_dir.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=non_empty_chunks,
        embedding=embeddings or get_embeddings(resolved),
        persist_directory=str(resolved.chroma_dir),
        collection_name=resolved.chroma_collection_name,
    )

    manifest = write_index_manifest(
        root_tokens=resolved.feishu_wiki_root_tokens,
        document_count=len(non_empty_documents),
        chunk_count=len(non_empty_chunks),
        settings=resolved,
    )
    return manifest.to_dict()


def main() -> None:
    """Run a full Feishu Wiki crawl and rebuild the local vector index."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    settings = get_settings()

    if not settings.feishu_wiki_root_tokens:
        msg = "FEISHU_WIKI_ROOT_TOKENS is required to build the local index."
        raise RuntimeError(msg)

    console.print("[bold blue]Feishu Wiki RAG Indexer[/]")
    console.print(f"Roots: {', '.join(settings.feishu_wiki_root_tokens)}")

    client = FeishuClient(settings)
    documents = client.crawl_documents(settings.feishu_wiki_root_tokens)
    console.print(f"Collected {len(documents)} non-empty documents before chunking")
    manifest = rebuild_index(documents, settings)

    console.print(f"[green]Indexed {manifest['document_count']} documents[/]")
    console.print(f"[green]Created {manifest['chunk_count']} chunks[/]")
    console.print(f"[green]Manifest written to {settings.manifest_path}[/]")


if __name__ == "__main__":
    main()
