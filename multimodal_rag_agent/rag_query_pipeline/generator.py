"""LLM answer generation."""

from __future__ import annotations

from time import perf_counter

try:
    from feishu_wiki_rag_agent.observability.context import record_request_timing, update_request_state
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception, preview_text
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.context import record_request_timing, update_request_state
    from observability.events import log_event, log_exception, preview_text

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings


EMPTY_CONTEXT = "未检索到相关上下文。"
EMPTY_ANSWER = "当前知识库中未找到相关内容。"


class AnswerGenerator:
    """Generate answers from retrieved context."""

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()

    def generate(self, query: str, context: str) -> str:
        if context.strip() == EMPTY_CONTEXT:
            return EMPTY_ANSWER

        from langchain_openai import ChatOpenAI

        started_at = perf_counter()
        log_event(
            "generation_started",
            model_name=self.settings.chat_model,
            query_preview=preview_text(query),
            context_length=len(context),
        )
        try:
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
            answer = str(getattr(result, "content", "") or "").strip() or EMPTY_ANSWER
            elapsed_ms = (perf_counter() - started_at) * 1000
            record_request_timing("llm_ms", elapsed_ms)
            update_request_state(answer_length=len(answer))
            log_event(
                "generation_completed",
                model_name=self.settings.chat_model,
                answer_length=len(answer),
                duration_ms=round(elapsed_ms, 1),
            )
            return answer
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_exception(
                "generation_failed",
                exc,
                model_name=self.settings.chat_model,
                query_preview=preview_text(query),
                duration_ms=round(elapsed_ms, 1),
            )
            raise
