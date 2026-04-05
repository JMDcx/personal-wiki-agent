"""Dense retrieval from Qdrant."""

from __future__ import annotations

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.ingest_pipeline.embedder import EmbeddingService
from multimodal_rag_agent.ingest_pipeline.qdrant_index import QdrantIndex
from multimodal_rag_agent.models import RetrievedChunk


class Retriever:
    """Retrieve multimodal chunks from Qdrant."""

    def __init__(
        self,
        settings: MultimodalRAGSettings | None = None,
        *,
        embedder: EmbeddingService | None = None,
        qdrant_index: QdrantIndex | None = None,
    ) -> None:
        self.settings = settings or get_multimodal_settings()
        self.embedder = embedder or EmbeddingService(self.settings)
        self.qdrant_index = qdrant_index or QdrantIndex(self.settings)

    def retrieve(self, query: str, *, top_k: int | None = None, filters: dict[str, object] | None = None) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(query)
        return self.qdrant_index.query(vector, top_k=top_k or self.settings.retrieval_top_k, filters=filters)
