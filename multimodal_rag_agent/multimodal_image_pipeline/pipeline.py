"""Generate OCR and caption chunks for images."""

from __future__ import annotations

import uuid

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.models import ChunkRecord, ResolvedImage
from multimodal_rag_agent.multimodal_image_pipeline.caption import CaptionService
from multimodal_rag_agent.multimodal_image_pipeline.ocr import OCRService
from multimodal_rag_agent.multimodal_image_pipeline.sanitizer import sanitize_ocr_text


class MultimodalImagePipeline:
    """Create image OCR and image caption chunks."""

    def __init__(
        self,
        settings: MultimodalRAGSettings | None = None,
        *,
        ocr_service: OCRService | None = None,
        caption_service: CaptionService | None = None,
    ) -> None:
        self.settings = settings or get_multimodal_settings()
        self.ocr_service = ocr_service or OCRService(self.settings)
        self.caption_service = caption_service or CaptionService(self.settings)

    def process_images(
        self,
        document_id: str,
        images: list[ResolvedImage],
        text_chunks: list[ChunkRecord],
        document_metadata: dict[str, object],
    ) -> list[ChunkRecord]:
        results: list[ChunkRecord] = []
        for image in images:
            parent_chunk_id = self._find_parent_chunk(image, text_chunks)
            ocr_text = sanitize_ocr_text(self.ocr_service.extract_text(image.stored_path))
            caption = (self.caption_service.caption(image.stored_path) or "").strip()
            common_metadata = {
                **document_metadata,
                "image_id": image.image_id,
                "source_uri": document_metadata.get("source_uri", ""),
                "source_type": image.source_type,
                "public_image_url": image.public_url,
                "headers": [],
                "section_path": "",
                "page_number": document_metadata.get("page_number", ""),
                "ocr_text": ocr_text,
                "caption_text": caption,
                "title": document_metadata.get("title", "Untitled Document"),
            }
            if ocr_text:
                results.append(
                    ChunkRecord(
                        chunk_id=uuid.uuid4().hex,
                        document_id=document_id,
                        chunk_type="image_ocr",
                        content=ocr_text,
                        metadata=common_metadata,
                        image_id=image.image_id,
                        parent_chunk_id=parent_chunk_id,
                    )
                )
            if caption:
                results.append(
                    ChunkRecord(
                        chunk_id=uuid.uuid4().hex,
                        document_id=document_id,
                        chunk_type="image_caption",
                        content=caption,
                        metadata=common_metadata,
                        image_id=image.image_id,
                        parent_chunk_id=parent_chunk_id,
                    )
                )
        return results

    @staticmethod
    def _find_parent_chunk(image: ResolvedImage, text_chunks: list[ChunkRecord]) -> str | None:
        for chunk in text_chunks:
            if image.public_url in chunk.content:
                return chunk.chunk_id
        return text_chunks[0].chunk_id if text_chunks else None
