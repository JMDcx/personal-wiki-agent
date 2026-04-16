"""Structured observability middleware for Deep Agents internals."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4
import re

try:
    from feishu_wiki_rag_agent.observability.context import bind_log_context, increment_request_counter, update_request_state
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception, preview_text
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.context import bind_log_context, increment_request_counter, update_request_state
    from observability.events import log_event, log_exception, preview_text

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command


_REWRITE_QUERY_RE = re.compile(r"^rewrite_query:\s*(.+)$", re.MULTILINE)
_INTENT_RE = re.compile(r"^intent:\s*(.+)$", re.MULTILINE)
_ALLOW_RETRIEVAL_RE = re.compile(r"^allow_retrieval:\s*(.+)$", re.MULTILINE)
_CURRENT_QUESTION_RE = re.compile(r"Current user question:\s*(.+)$", re.DOTALL)


def _safe_model_name(model: Any) -> str:
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return type(model).__name__


def _message_text(message: BaseMessage | Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _preview_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 2:
        return preview_text(str(value), limit=120)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return preview_text(value, limit=160)
    if isinstance(value, list):
        return [_preview_value(item, depth=depth + 1) for item in value[:5]]
    if isinstance(value, tuple):
        return [_preview_value(item, depth=depth + 1) for item in value[:5]]
    if isinstance(value, dict):
        limited_items = list(value.items())[:10]
        return {str(key): _preview_value(item, depth=depth + 1) for key, item in limited_items}
    return preview_text(str(value), limit=160)


def _tool_args_preview(tool_call: dict[str, Any]) -> Any:
    return _preview_value(tool_call.get("args", {}))


def _extract_ai_messages(response: Any) -> list[BaseMessage]:
    if isinstance(response, ModelResponse):
        return [message for message in response.result if isinstance(message, BaseMessage)]
    if isinstance(response, AIMessage):
        return [response]
    model_response = getattr(response, "model_response", None)
    if isinstance(model_response, ModelResponse):
        return [message for message in model_response.result if isinstance(message, BaseMessage)]
    return []


def _response_preview(messages: list[BaseMessage]) -> tuple[str, list[str], bool]:
    if not messages:
        return "", [], False
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    tool_names = [call.get("name", "") for call in tool_calls if isinstance(call, dict) and call.get("name")]
    return preview_text(_message_text(last_message), limit=200), tool_names, bool(tool_calls)


def _request_input_preview(request: ModelRequest[Any]) -> dict[str, Any]:
    message_count = len(request.messages)
    tool_message_count = sum(1 for message in request.messages if isinstance(message, ToolMessage))
    user_preview = ""
    rewrite_query_preview = ""
    controller_intent = ""
    allow_retrieval = ""
    for message in reversed(request.messages):
        message_type = getattr(message, "type", "")
        if message_type in {"human", "user"}:
            raw_user_text = _message_text(message)
            if "[Runtime Metadata - for assistant control, not for direct user display]" in raw_user_text:
                matched_question = _CURRENT_QUESTION_RE.search(raw_user_text)
                matched_rewrite = _REWRITE_QUERY_RE.search(raw_user_text)
                matched_intent = _INTENT_RE.search(raw_user_text)
                matched_allow = _ALLOW_RETRIEVAL_RE.search(raw_user_text)
                if matched_question:
                    user_preview = preview_text(matched_question.group(1).strip(), limit=200)
                else:
                    user_preview = preview_text(raw_user_text, limit=200)
                if matched_rewrite:
                    rewrite_query_preview = preview_text(matched_rewrite.group(1).strip(), limit=160)
                if matched_intent:
                    controller_intent = matched_intent.group(1).strip()
                if matched_allow:
                    allow_retrieval = matched_allow.group(1).strip()
            else:
                user_preview = preview_text(raw_user_text, limit=200)
            break
    system_preview = preview_text(request.system_prompt or "", limit=160) if request.system_prompt else ""
    return {
        "message_count": message_count,
        "tool_message_count": tool_message_count,
        "user_message_preview": user_preview,
        "rewrite_query_preview": rewrite_query_preview,
        "controller_intent": controller_intent,
        "controller_allow_retrieval": allow_retrieval,
        "system_message_preview": system_preview,
        "has_tool_messages": tool_message_count > 0,
    }


def _command_preview(result: Command[Any]) -> tuple[str, bool]:
    update = result.update if isinstance(result.update, dict) else {}
    messages = update.get("messages", []) if isinstance(update, dict) else []
    has_messages_update = bool(messages)
    if messages:
        first = messages[0]
        if isinstance(first, BaseMessage):
            return preview_text(_message_text(first), limit=200), has_messages_update
        return preview_text(str(first), limit=200), has_messages_update
    return preview_text(str(update), limit=200), has_messages_update


@contextmanager
def bind_subagent_context(
    *,
    parent_agent_name: str,
    subagent_type: str,
    delegation_id: str,
) -> Any:
    with bind_log_context(
        parent_agent_name=parent_agent_name,
        subagent_type=subagent_type,
        delegation_id=delegation_id,
    ) as ctx:
        yield ctx


class AgentObservabilityMiddleware(AgentMiddleware[Any, Any, Any]):
    """Capture model and tool boundaries for main agents and subagents."""

    def __init__(
        self,
        *,
        agent_name: str,
        agent_kind: str,
        parent_agent_name: str = "",
        subagent_type: str = "",
    ) -> None:
        super().__init__()
        self._agent_name = agent_name
        self._agent_kind = agent_kind
        self._parent_agent_name = parent_agent_name
        self._subagent_type = subagent_type

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Any,
    ) -> Any:
        model_name = _safe_model_name(request.model)
        input_preview = _request_input_preview(request)
        increment_request_counter("agent_model_call_count")
        log_event(
            "agent_model_call_started",
            agent_name=self._agent_name,
            agent_kind=self._agent_kind,
            parent_agent_name=self._parent_agent_name,
            subagent_type=self._subagent_type,
            model=model_name,
            **input_preview,
        )
        started_at = perf_counter()
        try:
            response = handler(request)
        except Exception as exc:  # noqa: BLE001
            update_request_state(failed_agent_stage="model")
            log_exception(
                "agent_model_call_failed",
                exc,
                agent_name=self._agent_name,
                agent_kind=self._agent_kind,
                parent_agent_name=self._parent_agent_name,
                subagent_type=self._subagent_type,
                model=model_name,
                duration_ms=round((perf_counter() - started_at) * 1000, 1),
                **input_preview,
            )
            raise

        output_messages = _extract_ai_messages(response)
        output_preview, tool_names, has_tool_calls = _response_preview(output_messages)
        log_event(
            "agent_model_call_completed",
            agent_name=self._agent_name,
            agent_kind=self._agent_kind,
            parent_agent_name=self._parent_agent_name,
            subagent_type=self._subagent_type,
            model=model_name,
            duration_ms=round((perf_counter() - started_at) * 1000, 1),
            output_preview=output_preview,
            output_message_count=len(output_messages),
            has_tool_calls=has_tool_calls,
            tool_call_names=tool_names,
        )
        return response

    def wrap_tool_call(
        self,
        request: Any,
        handler: Any,
    ) -> Any:
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")
        args_preview = _tool_args_preview(tool_call)
        common_fields = {
            "agent_name": self._agent_name,
            "agent_kind": self._agent_kind,
            "parent_agent_name": self._parent_agent_name,
            "subagent_type": self._subagent_type,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        }
        increment_request_counter("agent_tool_call_count")
        log_event(
            "agent_tool_call_started",
            tool_args_preview=args_preview,
            **common_fields,
        )
        started_at = perf_counter()
        if tool_name == "task":
            delegation_id = uuid4().hex
            delegated_subagent_type = str(tool_call.get("args", {}).get("subagent_type", "") or "")
            delegation_instruction = str(tool_call.get("args", {}).get("description", "") or "")
            increment_request_counter("subagent_call_count")
            log_event(
                "subagent_task_started",
                delegation_id=delegation_id,
                delegation_instruction_preview=preview_text(delegation_instruction, limit=200),
                subagent_type=delegated_subagent_type,
                parent_agent_name=self._agent_name,
                tool_call_id=tool_call_id,
                agent_name=self._agent_name,
                agent_kind=self._agent_kind,
            )
            try:
                with bind_subagent_context(
                    parent_agent_name=self._agent_name,
                    subagent_type=delegated_subagent_type,
                    delegation_id=delegation_id,
                ):
                    result = handler(request)
            except Exception as exc:  # noqa: BLE001
                update_request_state(failed_agent_stage="subagent_task")
                log_exception(
                    "subagent_task_failed",
                    exc,
                    delegation_id=delegation_id,
                    subagent_type=delegated_subagent_type,
                    parent_agent_name=self._agent_name,
                    duration_ms=round((perf_counter() - started_at) * 1000, 1),
                    tool_call_id=tool_call_id,
                )
                log_exception(
                    "agent_tool_call_failed",
                    exc,
                    duration_ms=round((perf_counter() - started_at) * 1000, 1),
                    tool_args_preview=args_preview,
                    delegation_id=delegation_id,
                    **common_fields,
                )
                raise

            result_preview = ""
            result_type = type(result).__name__
            has_messages_update = False
            if isinstance(result, Command):
                result_preview, has_messages_update = _command_preview(result)
            elif isinstance(result, ToolMessage):
                result_preview = preview_text(_message_text(result), limit=200)
            else:
                result_preview = preview_text(str(result), limit=200)
            log_event(
                "subagent_task_completed",
                delegation_id=delegation_id,
                subagent_type=delegated_subagent_type,
                parent_agent_name=self._agent_name,
                duration_ms=round((perf_counter() - started_at) * 1000, 1),
                returned_message_preview=result_preview,
                returned_message_length=len(result_preview),
                result_type=result_type,
                has_messages_update=has_messages_update,
                tool_call_id=tool_call_id,
            )
            log_event(
                "agent_tool_call_completed",
                duration_ms=round((perf_counter() - started_at) * 1000, 1),
                tool_args_preview=args_preview,
                result_preview=result_preview,
                result_type=result_type,
                result_status="ok",
                has_messages_update=has_messages_update,
                delegation_id=delegation_id,
                **common_fields,
            )
            return result

        try:
            result = handler(request)
        except Exception as exc:  # noqa: BLE001
            update_request_state(failed_agent_stage="tool")
            log_exception(
                "agent_tool_call_failed",
                exc,
                duration_ms=round((perf_counter() - started_at) * 1000, 1),
                tool_args_preview=args_preview,
                **common_fields,
            )
            raise

        result_type = type(result).__name__
        has_messages_update = False
        if isinstance(result, Command):
            result_preview, has_messages_update = _command_preview(result)
        elif isinstance(result, ToolMessage):
            result_preview = preview_text(_message_text(result), limit=200)
        else:
            result_preview = preview_text(str(result), limit=200)
        log_event(
            "agent_tool_call_completed",
            duration_ms=round((perf_counter() - started_at) * 1000, 1),
            tool_args_preview=args_preview,
            result_preview=result_preview,
            result_type=result_type,
            result_status="ok",
            has_messages_update=has_messages_update,
            **common_fields,
        )
        return result
