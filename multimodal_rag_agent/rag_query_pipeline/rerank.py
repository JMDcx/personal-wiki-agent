"""Simple reranking."""

from __future__ import annotations

from multimodal_rag_agent.models import RetrievedChunk


class Reranker:
    """Rerank retrieved chunks with lightweight lexical boosts."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        rescored: list[RetrievedChunk] = []
        for chunk in chunks:
            boost = 0.0
            haystack = f"{chunk.content} {chunk.metadata.get('ocr_text', '')} {chunk.metadata.get('caption_text', '')}".lower()
            for term in query_terms:
                if term and term in haystack:
                    boost += 0.05
            chunk.score += boost
            rescored.append(chunk)
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
