"""Merge and deduplicate retrieved chunks."""

from __future__ import annotations

from multimodal_rag_agent.models import RetrievedChunk


class ResultMerger:
    """Deduplicate and lightly merge multimodal chunks."""

    def merge(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen_ids: set[str] = set()
        seen_content: set[str] = set()
        merged: list[RetrievedChunk] = []
        for chunk in chunks:
            normalized = " ".join(chunk.content.split()).lower()
            if chunk.chunk_id in seen_ids or normalized in seen_content:
                continue
            seen_ids.add(chunk.chunk_id)
            seen_content.add(normalized)
            merged.append(chunk)
        return merged
