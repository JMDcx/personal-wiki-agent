"""Manual indexing command for Feishu Wiki/Docs content via the multimodal ingest pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from feishu_wiki_rag_agent.channel.feishu.feishu_client import FeishuClient
from feishu_wiki_rag_agent.config import Settings, get_settings
from feishu_wiki_rag_agent.retrieval import write_index_manifest
from multimodal_rag_agent.config import get_multimodal_settings
from multimodal_rag_agent.ingest_pipeline.pipeline import IngestPipeline

console = Console()


def rebuild_index(
    documents: list[Any],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Rebuild the Qdrant-backed index from Feishu documents."""
    resolved = settings or get_settings()
    non_empty_documents = [document for document in documents if document.page_content and document.page_content.strip()]
    if not non_empty_documents:
        msg = (
            "No non-empty Feishu documents were collected from the configured roots. "
            "Check that the root tokens point to Wiki nodes with readable doc/docx content."
        )
        raise RuntimeError(msg)

    pipeline = IngestPipeline(get_multimodal_settings())
    results = pipeline.ingest_documents(non_empty_documents, reset_index=True)
    total_chunks = sum(item.chunk_count for item in results)
    manifest = write_index_manifest(
        root_tokens=resolved.feishu_wiki_root_tokens,
        document_count=len(results),
        chunk_count=total_chunks,
        settings=resolved,
    )
    return manifest.to_dict()


def main() -> None:
    """Run a full Feishu Wiki crawl and rebuild the Qdrant index."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    settings = get_settings()

    if not settings.feishu_wiki_root_tokens:
        msg = "FEISHU_WIKI_ROOT_TOKENS is required to build the local index."
        raise RuntimeError(msg)

    console.print("[bold blue]Feishu Wiki Multimodal RAG Indexer[/]")
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
