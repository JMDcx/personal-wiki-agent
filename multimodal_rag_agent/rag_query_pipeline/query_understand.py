"""Query understanding."""

from __future__ import annotations

import re

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.models import QueryBundle


class QueryUnderstander:
    """Lightweight query rewrite and keyword extraction."""

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()

    def understand(self, query: str) -> QueryBundle:
        rewritten = query.strip()
        if self.settings.chat_api_key or self.settings.chat_base_url:
            try:
                from langchain_openai import ChatOpenAI

                model = ChatOpenAI(
                    model=self.settings.chat_model,
                    api_key=self.settings.chat_api_key or None,
                    base_url=self.settings.chat_base_url or None,
                    temperature=0.0,
                )
                message = model.invoke(
                    "将下面的问题改写为更适合知识检索的一句话，并只输出改写结果：\n" + query
                )
                rewritten = str(getattr(message, "content", "") or rewritten).strip() or rewritten
            except Exception:
                rewritten = query.strip()
        keywords = [part for part in re.split(r"[\s,，。！？?]+", rewritten) if part]
        intent = "retrieve" if keywords else "chat"
        return QueryBundle(raw_query=query, rewritten_query=rewritten, intent=intent, query_keywords=keywords[:8])
