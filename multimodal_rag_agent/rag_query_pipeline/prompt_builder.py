"""Prompt context builder."""

from __future__ import annotations

from multimodal_rag_agent.models import QueryBundle, RetrievedChunk


class PromptContextBuilder:
    """Build prompt context from retrieved chunks."""

    def build(self, query_bundle: QueryBundle, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "未检索到相关上下文。"
        sections = [
            f"用户问题：{query_bundle.raw_query}",
            f"改写查询：{query_bundle.rewritten_query}",
            "检索片段：",
        ]
        for index, chunk in enumerate(chunks, start=1):
            title = chunk.metadata.get("title", "Untitled")
            source_uri = chunk.metadata.get("source_uri", "")
            label = "文本"
            if chunk.chunk_type == "image_ocr":
                label = "图片OCR"
            elif chunk.chunk_type == "image_caption":
                label = "图片描述"
            sections.append(f"[{index}] {title} | {label} | {source_uri}\n{chunk.content}")
        return "\n\n".join(sections)
