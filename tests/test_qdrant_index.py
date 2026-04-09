from __future__ import annotations

import qdrant_client

from multimodal_rag_agent.config import MultimodalRAGSettings
from multimodal_rag_agent.ingest_pipeline.qdrant_index import QdrantIndex


def test_qdrant_index_disables_trust_env_for_local_qdrant(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyQdrantClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_client, "QdrantClient", DummyQdrantClient)

    settings = MultimodalRAGSettings(qdrant_url="http://127.0.0.1:6333")

    QdrantIndex(settings)._get_client()

    assert captured["url"] == "http://127.0.0.1:6333"
    assert captured["trust_env"] is False
