"""Deep Agents entrypoint backed by the multimodal Qdrant RAG pipeline."""

from __future__ import annotations

import atexit
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

try:
    from feishu_wiki_rag_agent.config import Settings, get_settings
    from feishu_wiki_rag_agent.observability.context import (
        bind_log_context,
        bind_request_context,
        get_log_context,
        has_request_state,
        record_request_timing,
        update_request_state,
    )
    from feishu_wiki_rag_agent.observability.events import emit_request_summary, log_event, log_exception, preview_text
    from feishu_wiki_rag_agent.observability.deepagents_middleware import AgentObservabilityMiddleware
    from feishu_wiki_rag_agent.observability.logging import configure_logging
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import Settings, get_settings
    from observability.context import (
        bind_log_context,
        bind_request_context,
        get_log_context,
        has_request_state,
        record_request_timing,
        update_request_state,
    )
    from observability.events import emit_request_summary, log_event, log_exception, preview_text
    from observability.deepagents_middleware import AgentObservabilityMiddleware
    from observability.logging import configure_logging
try:
    from feishu_wiki_rag_agent.schemas import MessageContext
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from schemas import MessageContext

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.rag_query_pipeline.pipeline import RAGQueryPipeline
from multimodal_rag_agent.rag_query_pipeline.controller_input_prompts import render_controller_user_input
from multimodal_rag_agent.rag_query_pipeline.intent_prompts import (
    render_base_controller_system_prompt,
    render_intent_system_prompt,
)
from multimodal_rag_agent.rag_query_pipeline.query_understand_service import (
    HistoryTurn,
    ModelConfig,
    QueryUnderstandResult,
    QueryUnderstandService,
)
from multimodal_rag_agent.deposit_pipeline.models import DepositRequest
from multimodal_rag_agent.deposit_pipeline.pipeline import DepositPipeline
try:
    from feishu_wiki_rag_agent.protocols.controller_models import ControllerDecision
    from feishu_wiki_rag_agent.protocols.renderers import render_deposit_result_text, render_retrieval_result_text
    from feishu_wiki_rag_agent.protocols.tool_models import (
        DepositRequestContext,
        DepositResult,
        RetrievalRequest,
        RetrievalResult,
    )
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from protocols.controller_models import ControllerDecision
    from protocols.renderers import render_deposit_result_text, render_retrieval_result_text
    from protocols.tool_models import DepositRequestContext, DepositResult, RetrievalRequest, RetrievalResult

try:
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
except ModuleNotFoundError:  # pragma: no cover - local fallback
    from dataclasses import dataclass

    @dataclass
    class AIMessage:
        content: Any

    def tool(fn):  # type: ignore[no-redef]
        return fn

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # pragma: no cover - imported lazily in tests
    ChatOpenAI = None  # type: ignore[assignment]


NON_RETRIEVAL_INTENTS = {
    "greeting",
    "summarize",
    "web_search",
    "follow_up",
    "image_only",
    "chitchat",
    "knowledge_deposit",
}

_LOCAL_GREETING_TEXTS = {
    "hi",
    "hello",
    "hey",
    "你好",
    "您好",
    "哈喽",
    "嗨",
    "thanks",
    "thankyou",
    "谢谢",
    "多谢",
    "感谢",
    "bye",
    "goodbye",
    "再见",
    "拜拜",
    "回头见",
}
_LOCAL_INTENT_NORMALIZER_RE = re.compile(r"[\s\.,，。!！?？~～、…]+")

DEPOSIT_TRIGGER_PATTERN = re.compile(r"(沉淀到知识库|沉淀到库|保存到知识库|收录到知识库|入库|沉淀一下|归档到知识库)")

_AGENT_RUNTIME_CACHE: dict[tuple[str, str, str, str], Any] = {}
_CHECKPOINTER_CACHE: dict[str, Any] = {}
_CHECKPOINTER_CONTEXTS: dict[str, Any] = {}
_LEGACY_FEISHU_CLEANUP_CACHE: set[str] = set()

_RUNTIME_METADATA_HEADER = "[Runtime Metadata - for assistant control, not for direct user display]"
_CURRENT_USER_QUESTION_RE = re.compile(r"Current user question:\s*", re.DOTALL)
_FEISHU_THREAD_PREFIX = "feishu:"
_FEISHU_RUNTIME_THREAD_PREFIX = "feishu__"
_HISTORY_USER_TEXT_LIMIT = 200
_HISTORY_ASSISTANT_TEXT_LIMIT = 400


def _question_preview(text: str, limit: int = 80) -> str:
    return preview_text(text, limit=limit)


def _runtime_thread_id(thread_id: str) -> str:
    raw_thread_id = str(thread_id or "").strip()
    if raw_thread_id.startswith(_FEISHU_THREAD_PREFIX):
        return f"{_FEISHU_RUNTIME_THREAD_PREFIX}{raw_thread_id[len(_FEISHU_THREAD_PREFIX):]}"
    return raw_thread_id


def _maybe_fast_path_query_understand(question: str, images: list[str]) -> QueryUnderstandResult | None:
    if images:
        return None
    stripped = str(question or "").strip()
    if not stripped or len(stripped) > 16:
        return None
    normalized = _LOCAL_INTENT_NORMALIZER_RE.sub("", stripped).lower()
    if normalized not in _LOCAL_GREETING_TEXTS:
        return None
    return QueryUnderstandResult(
        rewrite_query=stripped,
        intent="greeting",
        image_description="",
        raw_output='{"source":"local_fast_path","intent":"greeting"}',
    )


@dataclass(slots=True)
class ControllerInputContext:
    """Preprocessed controller-agent turn payload."""

    raw_question: str
    rewrite_query: str
    intent: str
    image_description: str
    raw_output: str
    history: list[HistoryTurn]
    user_content: str
    base_system_prompt: str
    runtime_system_prompt: str | None


@dataclass(slots=True)
class ThreadHistoryView:
    """Debug-friendly view of a persisted thread history."""

    thread_id: str
    raw_messages: list[dict[str, str]]
    history_turns: list[HistoryTurn]

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "raw_messages": self.raw_messages,
            "history_turns": [
                {
                    "user_question": turn.user_question,
                    "assistant_answer": turn.assistant_answer,
                }
                for turn in self.history_turns
            ],
        }


def _settings_cache_key(settings: Settings) -> tuple[str, str, str, str]:
    return (
        settings.rag_model,
        settings.chat_api_key,
        settings.chat_base_url,
        str(settings.example_dir),
    )


def _build_agent_backend(settings: Settings) -> Any:
    from deepagents.backends import CompositeBackend, FilesystemBackend

    history_root = settings.rag_data_dir / "conversation_history"
    return CompositeBackend(
        default=FilesystemBackend(root_dir=settings.example_dir, virtual_mode=True),
        routes={
            "/conversation_history/": FilesystemBackend(root_dir=history_root, virtual_mode=True),
        },
    )


def _close_cached_checkpointers() -> None:
    for path, context_manager in list(_CHECKPOINTER_CONTEXTS.items()):
        try:
            context_manager.__exit__(None, None, None)
        except Exception:
            pass
        finally:
            _CHECKPOINTER_CONTEXTS.pop(path, None)
            _CHECKPOINTER_CACHE.pop(path, None)


atexit.register(_close_cached_checkpointers)


def _intent_requires_retrieval(intent: str) -> bool:
    return intent not in NON_RETRIEVAL_INTENTS


def _build_vlm_model_config(settings: MultimodalRAGSettings) -> ModelConfig | None:
    if not settings.vlm_model:
        return None
    return ModelConfig(
        model=settings.vlm_model,
        api_key=settings.vlm_api_key,
        base_url=settings.vlm_base_url,
    )


def _create_query_understand_service(settings: MultimodalRAGSettings | None = None) -> QueryUnderstandService:
    return QueryUnderstandService(settings=settings or get_multimodal_settings())


def _log_schema_warning(schema_stage: str, reason: str, **fields: object) -> None:
    log_event(
        "schema_normalization_warning",
        level=logging.WARNING,
        schema_stage=schema_stage,
        reason=reason,
        **fields,
    )


def _normalize_retrieval_request(query: str) -> RetrievalRequest:
    request = RetrievalRequest.from_query(query, with_sources=True)
    stripped_query = str(query or "").strip()
    if stripped_query and request.query != stripped_query:
        _log_schema_warning(
            "retrieval_request",
            "query_whitespace_normalized",
            original_query_preview=_question_preview(str(query)),
            normalized_query_preview=_question_preview(request.query),
        )
    if not request.query:
        _log_schema_warning(
            "retrieval_request",
            "empty_query_after_normalization",
            original_query_preview=_question_preview(str(query)),
        )
    return request


def _normalize_deposit_request_context(
    *,
    text: str,
    image_paths_json: str,
    urls_json: str = "[]",
    source_url: str = "",
    provided_content: str = "",
) -> DepositRequestContext:
    context = DepositRequestContext.from_inputs(
        text=text,
        image_paths_json=image_paths_json,
        urls_json=urls_json,
        source_url=source_url,
        provided_content=provided_content,
    )
    if context.invalid_image_paths_json:
        _log_schema_warning(
            "deposit_request",
            "invalid_image_paths_json",
            text_preview=_question_preview(context.text),
        )
    if context.invalid_urls_json:
        _log_schema_warning(
            "deposit_request",
            "invalid_urls_json",
            text_preview=_question_preview(context.text),
        )
    if context.dropped_image_path_count:
        _log_schema_warning(
            "deposit_request",
            "blank_image_paths_dropped",
            dropped_image_path_count=context.dropped_image_path_count,
            image_count=len(context.image_paths),
        )
    if context.dropped_url_count:
        _log_schema_warning(
            "deposit_request",
            "blank_urls_dropped",
            dropped_url_count=context.dropped_url_count,
            url_count=len(context.urls),
        )
    if not context.text:
        _log_schema_warning(
            "deposit_request",
            "empty_text_after_normalization",
        )
    return context


def _normalize_message_context(message_context: MessageContext | dict[str, object] | None) -> MessageContext:
    if hasattr(message_context, "to_dict") and hasattr(message_context, "mention_refs"):
        return message_context
    return MessageContext.from_dict(message_context)


def _message_state_fields(message_context: MessageContext | dict[str, object] | None) -> dict[str, object]:
    context = _normalize_message_context(message_context)
    reply_context = context.reply_context
    return {
        "is_reply": reply_context is not None,
        "reply_parent_id": reply_context.parent_id if reply_context is not None else "",
        "reply_root_id": reply_context.root_id if reply_context is not None else "",
        "mentioned_users": list(context.mentioned_users),
        "bot_mentioned": context.bot_mentioned,
    }


def _build_checkpointer(settings: Settings) -> Any:
    db_path = str(settings.checkpoint_db_path)
    cached = _CHECKPOINTER_CACHE.get(db_path)
    if cached is not None:
        return cached

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        msg = (
            "langgraph-checkpoint-sqlite is required for SQLite-backed episodic memory. "
            "Install it with `uv add langgraph-checkpoint-sqlite`."
        )
        raise RuntimeError(msg) from exc

    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    context_manager = SqliteSaver.from_conn_string(db_path)
    saver = context_manager.__enter__()
    _CHECKPOINTER_CONTEXTS[db_path] = context_manager
    _CHECKPOINTER_CACHE[db_path] = saver
    return saver


def _get_or_build_agent_runtime(settings: Settings | None = None) -> Any:
    resolved = settings or get_settings()
    _cleanup_legacy_feishu_state(resolved)
    key = _settings_cache_key(resolved)
    runtime = _AGENT_RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = build_agent(resolved)
        _AGENT_RUNTIME_CACHE[key] = runtime
    return runtime


def _extract_message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "") or message.get("type", "")).lower()
    message_type = getattr(message, "type", None)
    if isinstance(message_type, str):
        return message_type.lower()
    message_role = getattr(message, "role", None)
    if isinstance(message_role, str):
        return message_role.lower()
    return ""


def _extract_message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()
    return ""


def _extract_raw_messages(agent: Any, thread_id: str) -> list[dict[str, str]]:
    if not hasattr(agent, "get_state"):
        return []
    config = {"configurable": {"thread_id": _runtime_thread_id(thread_id)}}
    try:
        snapshot = agent.get_state(config)
    except Exception:  # noqa: BLE001
        return []
    values = getattr(snapshot, "values", None)
    if values is None and isinstance(snapshot, dict):
        values = snapshot.get("values", {})
    if not isinstance(values, dict):
        return []

    raw_messages = values.get("messages", [])
    normalized: list[dict[str, str]] = []
    for raw_message in raw_messages:
        role = _extract_message_role(raw_message)
        text = _extract_message_text(raw_message)
        if role and text:
            normalized.append({"role": role, "content": text})
    return normalized


def _extract_history_user_text(text: str) -> tuple[str, bool]:
    stripped = str(text or "").strip()
    if not stripped:
        return "", False
    if _RUNTIME_METADATA_HEADER not in stripped:
        return stripped, False
    matches = list(_CURRENT_USER_QUESTION_RE.finditer(stripped))
    if matches:
        question = stripped[matches[-1].end() :].strip()
        if question:
            return question, True
    return stripped, True


def _truncate_history_text(text: str, *, limit: int) -> tuple[str, bool]:
    stripped = str(text or "").strip()
    if len(stripped) <= limit:
        return stripped, False
    truncated = stripped[: max(limit - 3, 0)].rstrip()
    if not truncated:
        return stripped[:limit], True
    return f"{truncated}...", True


def _load_history_from_runtime(agent: Any, thread_id: str, limit: int = 5) -> list[HistoryTurn]:
    messages: list[tuple[str, str]] = []
    metadata_user_count = 0
    user_truncated_count = 0
    assistant_truncated_count = 0
    for raw_message in _extract_raw_messages(agent, thread_id):
        role = raw_message["role"]
        text = raw_message["content"]
        if role in {"human", "user"} and text:
            cleaned_text, had_runtime_metadata = _extract_history_user_text(text)
            if had_runtime_metadata:
                metadata_user_count += 1
            cleaned_text, was_truncated = _truncate_history_text(cleaned_text, limit=_HISTORY_USER_TEXT_LIMIT)
            if was_truncated:
                user_truncated_count += 1
            if cleaned_text:
                messages.append(("user", cleaned_text))
        elif role in {"ai", "assistant"} and text:
            cleaned_text, was_truncated = _truncate_history_text(text, limit=_HISTORY_ASSISTANT_TEXT_LIMIT)
            if was_truncated:
                assistant_truncated_count += 1
            if cleaned_text:
                messages.append(("assistant", cleaned_text))

    history: list[HistoryTurn] = []
    pending_user: str | None = None
    for role, text in messages:
        if role == "user":
            pending_user = text
            continue
        if role == "assistant" and pending_user:
            history.append(HistoryTurn(user_question=pending_user, assistant_answer=text))
            pending_user = None
    if metadata_user_count or user_truncated_count or assistant_truncated_count:
        log_event(
            "thread_history_sanitized",
            thread_id=thread_id,
            runtime_thread_id=_runtime_thread_id(thread_id),
            metadata_user_count=metadata_user_count,
            user_truncated_count=user_truncated_count,
            assistant_truncated_count=assistant_truncated_count,
            history_turn_count=len(history),
        )
    return history[-limit:]


def list_thread_history(
    thread_id: str,
    *,
    settings: Settings | None = None,
    limit: int = 5,
) -> ThreadHistoryView:
    """Inspect the current persisted history for a thread."""
    runtime = _get_or_build_agent_runtime(settings)
    raw_messages = _extract_raw_messages(runtime, thread_id)
    history_turns = _load_history_from_runtime(runtime, thread_id, limit=limit)
    return ThreadHistoryView(
        thread_id=thread_id,
        raw_messages=raw_messages,
        history_turns=history_turns,
    )


def reset_thread(thread_id: str, *, settings: Settings | None = None) -> None:
    """Delete persisted episodic memory for a specific thread."""
    resolved = settings or get_settings()
    checkpointer = _build_checkpointer(resolved)
    delete_thread = getattr(checkpointer, "delete_thread", None)
    if delete_thread is None:  # pragma: no cover - unexpected checkpointer mismatch
        msg = "Current checkpointer does not support deleting a thread."
        raise RuntimeError(msg)
    delete_thread(_runtime_thread_id(thread_id))


def _cleanup_legacy_feishu_state(settings: Settings) -> None:
    db_path = str(settings.checkpoint_db_path.resolve())
    if db_path in _LEGACY_FEISHU_CLEANUP_CACHE:
        return

    removed_checkpoints = 0
    removed_writes = 0
    if settings.checkpoint_db_path.exists():
        try:
            conn = sqlite3.connect(settings.checkpoint_db_path)
            try:
                removed_checkpoints = conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id LIKE ?",
                    (f"{_FEISHU_THREAD_PREFIX}%",),
                ).rowcount
                removed_writes = conn.execute(
                    "DELETE FROM writes WHERE thread_id LIKE ?",
                    (f"{_FEISHU_THREAD_PREFIX}%",),
                ).rowcount
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            removed_checkpoints = 0
            removed_writes = 0

    removed_history_file = False
    for broken_history_path in (
        settings.example_dir / "conversation_history" / "feishu",
        settings.rag_data_dir / "conversation_history" / "feishu",
    ):
        if broken_history_path.is_file():
            broken_history_path.unlink(missing_ok=True)
            removed_history_file = True

    _LEGACY_FEISHU_CLEANUP_CACHE.add(db_path)
    if removed_checkpoints or removed_writes or removed_history_file:
        log_event(
            "legacy_feishu_state_cleaned",
            checkpoint_db_path=db_path,
            removed_checkpoints=removed_checkpoints,
            removed_writes=removed_writes,
            removed_history_file=removed_history_file,
        )


def _build_controller_context(
    *,
    question: str,
    thread_id: str,
    images: list[str],
    message_context: dict[str, object] | MessageContext | None,
    language: str,
    settings: Settings,
    multimodal_settings: MultimodalRAGSettings,
    agent_runtime: Any,
    chat_model_supports_vision: bool = False,
) -> ControllerInputContext:
    started_at = perf_counter()
    normalized_message_context = _normalize_message_context(message_context)
    history = _load_history_from_runtime(agent_runtime, thread_id)
    understand_result = _maybe_fast_path_query_understand(question, images)
    if understand_result is not None:
        log_event(
            "query_understand_fast_path_hit",
            thread_id=thread_id,
            intent=understand_result.intent,
            question_preview=_question_preview(question),
        )
    else:
        understand_service = _create_query_understand_service(multimodal_settings)
        understand_result = understand_service.run(
            query=question,
            history=history,
            images=images,
            language=language,
            chat_model_supports_vision=chat_model_supports_vision,
            vlm_model=_build_vlm_model_config(multimodal_settings),
        )
    if _is_knowledge_deposit_request(question):
        understand_result = QueryUnderstandResult(
            rewrite_query=question.strip(),
            intent="knowledge_deposit",
            image_description=understand_result.image_description,
            raw_output=understand_result.raw_output,
        )
    allow_retrieval = _intent_requires_retrieval(understand_result.intent)
    controller_decision = ControllerDecision.from_query_understand(
        understand_result,
        fallback_question=question,
        allow_retrieval=allow_retrieval,
    )
    user_content = render_controller_user_input(
        question=question,
        understand_result=understand_result,
        history=history,
        allow_retrieval=controller_decision.allow_retrieval,
        images=images,
        message_context=normalized_message_context,
    )
    current_time = datetime.now().isoformat(timespec="seconds")
    context = ControllerInputContext(
        raw_question=question,
        rewrite_query=controller_decision.rewrite_query,
        intent=controller_decision.intent,
        image_description=understand_result.image_description,
        raw_output=understand_result.raw_output,
        history=history,
        user_content=user_content,
        base_system_prompt=render_base_controller_system_prompt(language),
        runtime_system_prompt=render_intent_system_prompt(
            understand_result.intent,
            language,
            current_time=current_time,
        ),
    )
    elapsed_ms = (perf_counter() - started_at) * 1000
    record_request_timing("intent_ms", elapsed_ms)
    update_request_state(
        intent=context.intent,
        allow_retrieval=controller_decision.allow_retrieval,
        rewrite_query=controller_decision.rewrite_query,
        question_preview=_question_preview(question),
        history_turn_count=len(history),
        image_count=len(images),
        **_message_state_fields(normalized_message_context),
    )
    log_event(
        "controller_context_built",
        thread_id=thread_id,
        intent=context.intent,
        allow_retrieval=allow_retrieval,
        rewrite_query=controller_decision.rewrite_query,
        image_count=len(images),
        history_turn_count=len(history),
        duration_ms=round(elapsed_ms, 1),
        question_preview=_question_preview(question),
    )
    return context


def search_knowledge_tool_text(
    query: str,
    settings: Settings | None = None,
    *,
    pipeline: RAGQueryPipeline | None = None,
) -> str:
    """Return prepared RAG context for the agent or fallback snippets when needed."""
    started_at = perf_counter()
    _ = settings or get_settings()
    rag_pipeline = pipeline or RAGQueryPipeline(get_multimodal_settings())
    retrieval_request = _normalize_retrieval_request(query)
    log_event(
        "tool_called",
        tool_name="search_feishu_knowledge",
        query_preview=_question_preview(retrieval_request.query),
    )
    try:
        prepared = rag_pipeline.prepare_context(retrieval_request.query, with_sources=retrieval_request.with_sources)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (perf_counter() - started_at) * 1000
        log_exception(
            "tool_failed",
            exc,
            tool_name="search_feishu_knowledge",
            duration_ms=round(elapsed_ms, 1),
            query_preview=_question_preview(retrieval_request.query),
        )
        raise
    retrieval_result = RetrievalResult.from_prepared_context(retrieval_request.query, prepared)
    if retrieval_result.match_status == "no_match":
        elapsed_ms = (perf_counter() - started_at) * 1000
        log_event(
            "tool_completed",
            tool_name="search_feishu_knowledge",
            success=True,
            result_status=retrieval_result.result_status,
            match_status=retrieval_result.match_status,
            chunk_count=retrieval_result.chunk_count,
            source_count=retrieval_result.source_count,
            duration_ms=round(elapsed_ms, 1),
            query_preview=_question_preview(retrieval_request.query),
        )
        return render_retrieval_result_text(retrieval_result)

    elapsed_ms = (perf_counter() - started_at) * 1000
    log_event(
        "tool_completed",
        tool_name="search_feishu_knowledge",
        success=True,
        result_status=retrieval_result.result_status,
        match_status=retrieval_result.match_status,
        chunk_count=retrieval_result.chunk_count,
        source_count=retrieval_result.source_count,
        duration_ms=round(elapsed_ms, 1),
        query_preview=_question_preview(retrieval_request.query),
    )
    return render_retrieval_result_text(retrieval_result)


def _is_knowledge_deposit_request(text: str) -> bool:
    return bool(DEPOSIT_TRIGGER_PATTERN.search(text or ""))


def deposit_knowledge_tool_text(
    text: str,
    *,
    image_paths_json: str = "[]",
    urls_json: str = "[]",
    source_url: str = "",
    provided_content: str = "",
    settings: Settings | None = None,
    pipeline: DepositPipeline | None = None,
) -> str:
    resolved = settings or get_settings()
    deposit_pipeline = pipeline or DepositPipeline(resolved, get_multimodal_settings())
    deposit_request_context = _normalize_deposit_request_context(
        text=text,
        image_paths_json=image_paths_json,
        urls_json=urls_json,
        source_url=source_url,
        provided_content=provided_content,
    )
    log_event(
        "tool_called",
        tool_name="deposit_to_feishu_knowledge",
        text_preview=_question_preview(deposit_request_context.text),
        image_count=len(deposit_request_context.image_paths),
    )
    try:
        result = deposit_pipeline.run(
            DepositRequest(
                text=deposit_request_context.text,
                urls=list(deposit_request_context.urls),
                provided_content=deposit_request_context.provided_content,
                image_paths=deposit_request_context.image_paths,
            )
        )
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "tool_failed",
            exc,
            tool_name="deposit_to_feishu_knowledge",
            text_preview=_question_preview(deposit_request_context.text),
            image_count=len(deposit_request_context.image_paths),
        )
        raise
    deposit_result = DepositResult.from_pipeline_result(result)
    log_event(
        "tool_completed",
        tool_name="deposit_to_feishu_knowledge",
        success=True,
        result_status=deposit_result.result_status,
        source_type=deposit_result.source_type,
        local_document_id=deposit_result.local_document_id,
        has_feishu_doc=bool(deposit_result.feishu_doc_url),
    )
    return render_deposit_result_text(deposit_result)


def _build_chat_model(settings: Settings) -> Any:
    if ChatOpenAI is None:  # pragma: no cover - local fallback
        msg = "langchain_openai is required to run the Deep Agent runtime."
        raise RuntimeError(msg)
    model_kwargs: dict[str, Any] = {"model": settings.rag_model}
    if settings.chat_api_key:
        model_kwargs["api_key"] = settings.chat_api_key
    if settings.chat_base_url:
        model_kwargs["base_url"] = settings.chat_base_url
    return ChatOpenAI(**model_kwargs)


def build_agent(settings: Settings | None = None) -> Any:
    """Build the Deep Agent runtime with a dedicated retrieval subagent."""
    resolved = settings or get_settings()
    multimodal_settings = get_multimodal_settings()
    rag_pipeline = RAGQueryPipeline(multimodal_settings)
    deposit_pipeline = DepositPipeline(resolved, multimodal_settings)

    from deepagents import create_deep_agent
    from langgraph.store.memory import InMemoryStore

    @tool
    def search_feishu_knowledge(query: str) -> str:
        """Retrieve relevant Feishu knowledge snippets and sources for a query."""
        return search_knowledge_tool_text(query, resolved, pipeline=rag_pipeline)

    @tool
    def deposit_to_feishu_knowledge(
        text: str,
        image_paths_json: str = "[]",
        urls_json: str = "[]",
        source_url: str = "",
        provided_content: str = "",
    ) -> str:
        """Deposit provided links, text, or images into the Feishu knowledge base and local index."""
        return deposit_knowledge_tool_text(
            text,
            image_paths_json=image_paths_json,
            urls_json=urls_json,
            source_url=source_url,
            provided_content=provided_content,
            settings=resolved,
            pipeline=deposit_pipeline,
        )

    main_observability_middleware = AgentObservabilityMiddleware(
        agent_name="feishu-wiki-rag-agent",
        agent_kind="main",
    )
    retrieval_observability_middleware = AgentObservabilityMiddleware(
        agent_name="knowledge_retriever",
        agent_kind="subagent",
        parent_agent_name="feishu-wiki-rag-agent",
        subagent_type="knowledge_retriever",
    )
    deposit_observability_middleware = AgentObservabilityMiddleware(
        agent_name="knowledge_depositor",
        agent_kind="subagent",
        parent_agent_name="feishu-wiki-rag-agent",
        subagent_type="knowledge_depositor",
    )

    return create_deep_agent(
        name="feishu-wiki-rag-agent",
        model=_build_chat_model(resolved),
        tools=[],
        subagents=[
            {
                "name": "knowledge_retriever",
                "description": "Retrieve documentation context from the indexed Feishu knowledge base.",
                "system_prompt": (
                    "You are a retrieval specialist for indexed Feishu documentation. "
                    "Always call `search_feishu_knowledge` before responding. "
                    "Return concise retrieval findings, likely sources, and note when nothing relevant was found."
                ),
                "tools": [search_feishu_knowledge],
                "skills": ["/skills/"],
                "middleware": [retrieval_observability_middleware],
            },
            {
                "name": "knowledge_depositor",
                "description": "Deposit external links, text, and images into the Feishu knowledge base.",
                "system_prompt": (
                    "You are a knowledge deposit specialist. "
                    "Always call `deposit_to_feishu_knowledge` before responding. "
                    "Use the user's full source text, and pass image_paths_json when runtime metadata contains image paths. "
                    "Return a concise completion note with the resulting document link when available."
                ),
                "tools": [deposit_to_feishu_knowledge],
                "skills": ["/skills/"],
                "middleware": [deposit_observability_middleware],
            }
        ],
        system_prompt=render_base_controller_system_prompt("中文"),
        backend=_build_agent_backend(resolved),
        skills=["/skills/"],
        memory=["/AGENTS.md"],
        checkpointer=_build_checkpointer(resolved),
        store=InMemoryStore(),
        middleware=[main_observability_middleware],
    )


def extract_final_text(result: dict[str, Any]) -> str:
    """Extract the final assistant text from an invocation result."""
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", None)
        if content:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [block.get("text", "") for block in content if isinstance(block, dict)]
                joined = "\n".join(part for part in text_parts if part)
                if joined:
                    return joined
    return "当前索引中未找到相关内容。"


def invoke_agent(
    question: str,
    *,
    settings: Settings | None = None,
    thread_id: str = "default",
    images: list[str] | None = None,
    message_context: dict[str, object] | None = None,
    language: str = "中文",
) -> str:
    """Run the Deep Agent orchestrator backed by the multimodal RAG pipeline."""
    resolved = settings or get_settings()
    configure_logging(resolved)
    multimodal_settings = get_multimodal_settings()
    context_fields: dict[str, object] = {
        "thread_id": thread_id,
        "language": language,
    }
    if not get_log_context().get("request_id"):
        context_fields["request_id"] = f"thread:{thread_id}"

    owns_request_state = not has_request_state()
    context_manager = bind_request_context if owns_request_state else bind_log_context

    with context_manager(**context_fields):
        log_event(
            "agent_invoke_started",
            question_preview=_question_preview(question),
            image_count=len(images or []),
        )
        try:
            agent = _get_or_build_agent_runtime(resolved)
            runtime_thread_id = _runtime_thread_id(thread_id)
            if runtime_thread_id != thread_id:
                log_event(
                    "thread_runtime_key_mapped",
                    thread_id=thread_id,
                    runtime_thread_id=runtime_thread_id,
                )
            controller_context = _build_controller_context(
                question=question,
                thread_id=thread_id,
                images=images or [],
                message_context=message_context,
                language=language,
                settings=resolved,
                multimodal_settings=multimodal_settings,
                agent_runtime=agent,
            )

            messages: list[dict[str, Any]] = []
            if controller_context.runtime_system_prompt:
                messages.append({"role": "system", "content": controller_context.runtime_system_prompt})
            messages.append({"role": "user", "content": controller_context.user_content})

            invoke_started_at = perf_counter()
            result = agent.invoke(
                {"messages": messages},
                config={"configurable": {"thread_id": runtime_thread_id}},
            )
            invoke_elapsed_ms = (perf_counter() - invoke_started_at) * 1000
            record_request_timing("llm_ms", invoke_elapsed_ms)
            final_text = extract_final_text(result)
            update_request_state(
                intent=controller_context.intent,
                allow_retrieval=_intent_requires_retrieval(controller_context.intent),
                answer_length=len(final_text),
                answer_preview=_question_preview(final_text),
                question_preview=_question_preview(question),
                language=language,
            )
            log_event(
                "agent_invoke_completed",
                intent=controller_context.intent,
                invoke_duration_ms=round(invoke_elapsed_ms, 1),
                answer_length=len(final_text),
                answer_preview=_question_preview(final_text),
                question_preview=_question_preview(question),
            )
            if owns_request_state:
                emit_request_summary(status="ok")
            return final_text
        except Exception as exc:  # noqa: BLE001
            update_request_state(
                question_preview=_question_preview(question),
                language=language,
            )
            if owns_request_state:
                log_exception(
                    "request_failed",
                    exc,
                    stage="agent_invoke",
                    question_preview=_question_preview(question),
                )
                emit_request_summary(status="error", level=logging.ERROR)
            raise
