"""Embedding helpers."""

from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings


class EmbeddingService:
    """Thin OpenAI-compatible embeddings wrapper.

    Uses the raw OpenAI SDK instead of `langchain_openai.OpenAIEmbeddings`
    because some compatible providers accept the plain `/embeddings` wire shape
    but reject LangChain's internal batching/token-safe request variants.
    """

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()
        self._client = OpenAI(
            api_key=self.settings.embedding_api_key or None,
            base_url=self.settings.embedding_base_url or None,
            timeout=60,
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        clean_texts = [text if isinstance(text, str) else str(text) for text in texts]
        if not clean_texts:
            return []
        response = self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=clean_texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding
