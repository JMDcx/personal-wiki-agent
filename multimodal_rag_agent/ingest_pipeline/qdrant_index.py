"""Qdrant index wrapper."""

from __future__ import annotations

from typing import Any

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.models import ChunkRecord, RetrievedChunk


class QdrantIndex:
    """Qdrant collection operations for multimodal chunks."""

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key or None)
        return self._client

    def ensure_collection(self) -> None:
        from qdrant_client.http import models as rest

        client = self._get_client()
        collections = {item.name for item in client.get_collections().collections}
        if self.settings.qdrant_collection not in collections:
            client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=rest.VectorParams(size=self.settings.qdrant_vector_size, distance=rest.Distance.COSINE),
            )

    def reset_collection(self) -> None:
        client = self._get_client()
        collections = {item.name for item in client.get_collections().collections}
        if self.settings.qdrant_collection in collections:
            client.delete_collection(collection_name=self.settings.qdrant_collection)
        self.ensure_collection()

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        from qdrant_client.http import models as rest

        self.ensure_collection()
        points = [
            rest.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.to_payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._get_client().upsert(collection_name=self.settings.qdrant_collection, points=points)

    def query(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        from qdrant_client.http import models as rest

        query_filter = None
        if filters:
            query_filter = rest.Filter(
                must=[
                    rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )
        records = self._get_client().query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [
            RetrievedChunk(
                chunk_id=str(point.id),
                score=float(point.score or 0.0),
                chunk_type=str(point.payload.get("chunk_type", "")),
                content=str(point.payload.get("content", "")),
                metadata=dict(point.payload),
            )
            for point in records
        ]
