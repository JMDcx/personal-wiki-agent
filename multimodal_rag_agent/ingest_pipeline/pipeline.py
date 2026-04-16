"""Synchronous ingest pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path
from time import perf_counter

try:
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.events import log_event, log_exception

from multimodal_rag_agent.models import ParsedDocument
from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.docreader_service.client import DocreaderService
from multimodal_rag_agent.docreader_service.schemas import ParseRequest
from multimodal_rag_agent.image_resolver.resolver import ImageResolver
from multimodal_rag_agent.image_resolver.storage import LocalImageStorage
from multimodal_rag_agent.ingest_pipeline.chunking import MarkdownChunker
from multimodal_rag_agent.ingest_pipeline.embedder import EmbeddingService
from multimodal_rag_agent.ingest_pipeline.qdrant_index import QdrantIndex
from multimodal_rag_agent.models import ChunkRecord, IngestResult
from multimodal_rag_agent.multimodal_image_pipeline.pipeline import MultimodalImagePipeline


class IngestPipeline:
    """Main document ingest flow."""

    def __init__(
        self,
        settings: MultimodalRAGSettings | None = None,
        *,
        docreader: DocreaderService | None = None,
        image_resolver: ImageResolver | None = None,
        chunker: MarkdownChunker | None = None,
        embedder: EmbeddingService | None = None,
        qdrant_index: QdrantIndex | None = None,
        multimodal_pipeline: MultimodalImagePipeline | None = None,
    ) -> None:
        self.settings = settings or get_multimodal_settings()
        self.docreader = docreader or DocreaderService(self.settings)
        self.image_resolver = image_resolver or ImageResolver(
            LocalImageStorage(self.settings.asset_root, self.settings.image_url_prefix)
        )
        self.chunker = chunker or MarkdownChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        self.embedder = embedder or EmbeddingService(self.settings)
        self.qdrant_index = qdrant_index or QdrantIndex(self.settings)
        self.multimodal_pipeline = multimodal_pipeline or MultimodalImagePipeline(self.settings)

    def ingest_file(
        self,
        file_name: str,
        file_content: bytes,
        *,
        metadata: dict[str, object] | None = None,
    ) -> IngestResult:
        document_id = uuid.uuid4().hex
        parsed = self.docreader.parse(
            ParseRequest(
                file_name=file_name,
                file_type=Path(file_name).suffix.lstrip("."),
                file_content=file_content,
            )
        )
        return self._ingest_parsed(document_id, parsed, metadata or {})

    def ingest_url(
        self,
        url: str,
        *,
        title: str = "",
        metadata: dict[str, object] | None = None,
    ) -> IngestResult:
        document_id = uuid.uuid4().hex
        parsed = self.docreader.parse(ParseRequest(url=url, title=title))
        merged = dict(metadata or {})
        merged.setdefault("source_uri", url)
        if title:
            merged.setdefault("title", title)
        return self._ingest_parsed(document_id, parsed, merged)

    def ingest_markdown(
        self,
        markdown_content: str,
        *,
        title: str,
        metadata: dict[str, object] | None = None,
        document_id: str | None = None,
    ) -> IngestResult:
        parsed = ParsedDocument(markdown_content=markdown_content, metadata={"title": title})
        return self._ingest_parsed(document_id or uuid.uuid4().hex, parsed, metadata or {})

    def ingest_documents(
        self,
        documents: list[object],
        *,
        reset_index: bool = True,
    ) -> list[IngestResult]:
        if reset_index:
            self.qdrant_index.reset_collection()
        results: list[IngestResult] = []
        for document in documents:
            page_content = str(getattr(document, "page_content", "") or "")
            metadata = dict(getattr(document, "metadata", {}) or {})
            if not page_content.strip():
                continue
            image_refs = metadata.pop("image_refs", [])
            document_id = str(metadata.get("doc_token") or metadata.get("node_token") or uuid.uuid4().hex)
            title = str(metadata.get("title", "Untitled Document"))
            metadata.setdefault("title", title)
            metadata.setdefault("source_uri", str(metadata.get("source_url", "")))
            parsed = ParsedDocument(
                markdown_content=page_content,
                image_refs=list(image_refs) if isinstance(image_refs, list) else [],
                metadata={"title": title},
            )
            results.append(self._ingest_parsed(document_id, parsed, metadata))
        return results

    def _ingest_parsed(self, document_id: str, parsed, metadata: dict[str, object]) -> IngestResult:
        started_at = perf_counter()
        document_metadata = dict(parsed.metadata)
        document_metadata.update(metadata)
        document_metadata.setdefault("document_id", document_id)
        document_metadata.setdefault("title", document_metadata.get("title", "Untitled Document"))
        document_metadata.setdefault("source_uri", document_metadata.get("source_uri", ""))
        document_metadata.setdefault("source_url", document_metadata.get("source_uri", ""))
        document_metadata.setdefault("source_type", "file" if document_metadata.get("source_uri", "") == "" else "url")
        log_event(
            "ingest_started",
            document_id=document_id,
            title=str(document_metadata.get("title", "Untitled Document")),
            source_type=str(document_metadata.get("source_type", "")),
        )
        try:
            resolved = self.image_resolver.resolve(document_id, parsed)
            text_chunks = self.chunker.split(document_id, resolved.markdown_content, document_metadata)
            image_chunks = self.multimodal_pipeline.process_images(document_id, resolved.images, text_chunks, document_metadata)
            all_chunks: list[ChunkRecord] = text_chunks + image_chunks
            vectors = self.embedder.embed_texts([chunk.content for chunk in all_chunks]) if all_chunks else []
            if all_chunks:
                self.qdrant_index.upsert_chunks(all_chunks, vectors)
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_event(
                "ingest_completed",
                document_id=document_id,
                text_chunk_count=len(text_chunks),
                image_chunk_count=len(image_chunks),
                chunk_count=len(all_chunks),
                image_count=len(resolved.images),
                duration_ms=round(elapsed_ms, 1),
            )
            return IngestResult(
                document_id=document_id,
                status="completed",
                chunk_count=len(all_chunks),
                image_count=len(resolved.images),
                resolved_markdown=resolved.markdown_content,
                chunks=all_chunks,
                images=resolved.images,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_exception(
                "ingest_failed",
                exc,
                document_id=document_id,
                title=str(document_metadata.get("title", "Untitled Document")),
                duration_ms=round(elapsed_ms, 1),
            )
            raise
