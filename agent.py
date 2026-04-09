"""Deep Agents entrypoint backed by the multimodal Qdrant RAG pipeline."""

from __future__ import annotations

import atexit
import json
import logging
import re
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
    from observability.logging import configure_logging

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

DEPOSIT_TRIGGER_PATTERN = re.compile(r"(沉淀到知识库|沉淀到库|保存到知识库|收录到知识库|入库|沉淀一下|归档到知识库)")

_AGENT_RUNTIME_CACHE: dict[tuple[str, str, str, str], Any] = {}
_CHECKPOINTER_CACHE: dict[str, Any] = {}
_CHECKPOINTER_CONTEXTS: dict[str, Any] = {}


def _question_preview(text: str, limit: int = 80) -> str:
    return preview_text(text, limit=limit)


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
    config = {"configurable": {"thread_id": thread_id}}
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


def _load_history_from_runtime(agent: Any, thread_id: str, limit: int = 5) -> list[HistoryTurn]:
    messages: list[tuple[str, str]] = []
    for raw_message in _extract_raw_messages(agent, thread_id):
        role = raw_message["role"]
        text = raw_message["content"]
        if role in {"human", "user"} and text:
            messages.append(("user", text))
        elif role in {"ai", "assistant"} and text:
            messages.append(("assistant", text))

    history: list[HistoryTurn] = []
    pending_user: str | None = None
    for role, text in messages:
        if role == "user":
            pending_user = text
            continue
        if role == "assistant" and pending_user:
            history.append(HistoryTurn(user_question=pending_user, assistant_answer=text))
            pending_user = None
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
    delete_thread(thread_id)


def _build_controller_context(
    *,
    question: str,
    thread_id: str,
    images: list[str],
    language: str,
    settings: Settings,
    multimodal_settings: MultimodalRAGSettings,
    agent_runtime: Any,
    chat_model_supports_vision: bool = False,
) -> ControllerInputContext:
    started_at = perf_counter()
    history = _load_history_from_runtime(agent_runtime, thread_id)
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
    user_content = render_controller_user_input(
        question=question,
        understand_result=understand_result,
        history=history,
        allow_retrieval=allow_retrieval,
        images=images,
    )
    current_time = datetime.now().isoformat(timespec="seconds")
    context = ControllerInputContext(
        raw_question=question,
        rewrite_query=understand_result.rewrite_query or question,
        intent=understand_result.intent,
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
        allow_retrieval=allow_retrieval,
        question_preview=_question_preview(question),
        history_turn_count=len(history),
        image_count=len(images),
    )
    log_event(
        "controller_context_built",
        thread_id=thread_id,
        intent=context.intent,
        allow_retrieval=allow_retrieval,
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
    log_event(
        "tool_called",
        tool_name="search_feishu_knowledge",
        query_preview=_question_preview(query),
    )
    try:
        prepared = rag_pipeline.prepare_context(query, with_sources=True)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (perf_counter() - started_at) * 1000
        log_exception(
            "tool_failed",
            exc,
            tool_name="search_feishu_knowledge",
            duration_ms=round(elapsed_ms, 1),
            query_preview=_question_preview(query),
        )
        raise
    if not prepared.merged_chunks:
        elapsed_ms = (perf_counter() - started_at) * 1000
        log_event(
            "tool_completed",
            tool_name="search_feishu_knowledge",
            success=True,
            result_status="empty",
            chunk_count=0,
            source_count=0,
            duration_ms=round(elapsed_ms, 1),
            query_preview=_question_preview(query),
        )
        return "当前索引中未找到相关内容。"

    lines: list[str] = [prepared.context]
    if prepared.sources:
        source_line = "来源：" + "；".join(
            [
                str(source.get("title") or source.get("source_uri") or "Untitled")
                for source in prepared.sources
            ]
        )
        lines.append(source_line)
    elapsed_ms = (perf_counter() - started_at) * 1000
    log_event(
        "tool_completed",
        tool_name="search_feishu_knowledge",
        success=True,
        result_status="completed",
        chunk_count=len(prepared.merged_chunks),
        source_count=len(prepared.sources),
        duration_ms=round(elapsed_ms, 1),
        query_preview=_question_preview(query),
    )
    return "\n\n".join(lines)


def _is_knowledge_deposit_request(text: str) -> bool:
    return bool(DEPOSIT_TRIGGER_PATTERN.search(text or ""))


def deposit_knowledge_tool_text(
    text: str,
    *,
    image_paths_json: str = "[]",
    settings: Settings | None = None,
    pipeline: DepositPipeline | None = None,
) -> str:
    resolved = settings or get_settings()
    deposit_pipeline = pipeline or DepositPipeline(resolved, get_multimodal_settings())
    try:
        image_paths = json.loads(image_paths_json) if image_paths_json.strip() else []
    except json.JSONDecodeError:
        image_paths = []
    cleaned_image_paths = [str(path) for path in image_paths if str(path).strip()]
    log_event(
        "tool_called",
        tool_name="deposit_to_feishu_knowledge",
        text_preview=_question_preview(text),
        image_count=len(cleaned_image_paths),
    )
    try:
        result = deposit_pipeline.run(
            DepositRequest(
                text=text,
                image_paths=cleaned_image_paths,
            )
        )
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "tool_failed",
            exc,
            tool_name="deposit_to_feishu_knowledge",
            text_preview=_question_preview(text),
            image_count=len(cleaned_image_paths),
        )
        raise
    log_event(
        "tool_completed",
        tool_name="deposit_to_feishu_knowledge",
        success=True,
        result_status=result.status,
        source_type=result.draft.source_type,
        local_document_id=result.local_document_id,
        has_feishu_doc=bool(result.feishu_doc_url),
    )
    lines = [result.message, f"来源类型：{result.draft.source_type}"]
    if result.feishu_doc_url:
        lines.append(f"飞书文档：{result.feishu_doc_url}")
    if result.wiki_node_token:
        lines.append(f"Wiki 节点：{result.wiki_node_token}")
    return "\n".join(lines)


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
    from deepagents.backends import FilesystemBackend
    from langgraph.store.memory import InMemoryStore

    @tool
    def search_feishu_knowledge(query: str) -> str:
        """Retrieve relevant Feishu knowledge snippets and sources for a query."""
        return search_knowledge_tool_text(query, resolved, pipeline=rag_pipeline)

    @tool
    def deposit_to_feishu_knowledge(text: str, image_paths_json: str = "[]") -> str:
        """Deposit provided links, text, or images into the Feishu knowledge base and local index."""
        return deposit_knowledge_tool_text(
            text,
            image_paths_json=image_paths_json,
            settings=resolved,
            pipeline=deposit_pipeline,
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
            }
        ],
        system_prompt=render_base_controller_system_prompt("中文"),
        backend=FilesystemBackend(root_dir=resolved.example_dir, virtual_mode=True),
        skills=["/skills/"],
        memory=["/AGENTS.md"],
        checkpointer=_build_checkpointer(resolved),
        store=InMemoryStore(),
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
            controller_context = _build_controller_context(
                question=question,
                thread_id=thread_id,
                images=images or [],
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
                config={"configurable": {"thread_id": thread_id}},
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
