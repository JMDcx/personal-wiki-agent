"""Query understanding."""

from __future__ import annotations

import re

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.models import QueryBundle


class QueryUnderstander:
    """Lightweight local normalization and keyword extraction.

    The controller layer already performs the LLM-based query understanding step.
    Retrieval should stay cheap and deterministic here, so we do not call another
    chat model inside the retrieval pipeline.
    """

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()

    def understand(self, query: str) -> QueryBundle:
        rewritten = query.strip()
        keywords = [part for part in re.split(r"[\s,，。！？?]+", rewritten) if part]
        return QueryBundle(raw_query=query, rewritten_query=rewritten, query_keywords=keywords[:8])
