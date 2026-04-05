"""LLM answer generation."""

from __future__ import annotations

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings


class AnswerGenerator:
    """Generate answers from retrieved context."""

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()

    def generate(self, query: str, context: str) -> str:
        if context.strip() == "未检索到相关上下文。":
            return "当前知识库中未找到相关内容。"
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=self.settings.chat_model,
            api_key=self.settings.chat_api_key or None,
            base_url=self.settings.chat_base_url or None,
            temperature=0.1,
        )
        result = model.invoke(
            "你是多模态 RAG 助手。只能依据提供的上下文回答，回答简洁，并附上来源标题或链接。\n\n"
            f"{context}\n\n用户问题：{query}"
        )
        return str(getattr(result, "content", "") or "").strip() or "当前知识库中未找到相关内容。"
