"""Convenience helpers for structured event logging."""

from __future__ import annotations

import logging
import os
import re

try:
    from feishu_wiki_rag_agent.observability.context import (
        get_request_state,
        get_request_total_duration_ms,
        update_request_state,
    )
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.context import get_request_state, get_request_total_duration_ms, update_request_state


logger = logging.getLogger("feishu_wiki_rag_agent.events")
_DEFAULT_DEBUG_EVENTS = {
    "message_normalized",
    "message_ignored",
    "agent_invoke_started",
    "request_completed",
    "agent_invoke_completed",
    "controller_context_built",
    "tool_called",
    "tool_completed",
    "tool_failed",
    "retrieval_started",
    "retrieval_completed",
    "retrieval_failed",
    "generation_started",
    "generation_completed",
    "generation_failed",
    "deposit_started",
    "deposit_completed",
    "deposit_feishu_markdown_prepared",
    "deposit_ingest_markdown_prepared",
    "feishu_write_started",
    "feishu_write_completed",
    "stream_started",
    "stream_status_emitted",
    "stream_delta_emitted",
    "stream_update_sent",
    "stream_completed",
    "stream_failed",
    "stream_fallback_used",
    "reply_sent",
    "dispatch_submitted",
    "dispatch_started",
    "dispatch_completed",
}
_TOKEN_RE = re.compile(r"\b(sk-[A-Za-z0-9]{12,}|Bearer\s+[A-Za-z0-9._-]{12,}|cli_[A-Za-z0-9]{8,})\b")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_LONG_NUMBER_RE = re.compile(r"\b\d{11,}\b")
_ERROR_CODE_RE = re.compile(r"\b(?:code|status|err|tid|received)\s*[:=]?\s*(\d{3,6})\b", re.IGNORECASE)


def _is_empty_value(value: object) -> bool:
    return value is None or value == ""


def _preview_length() -> int:
    raw = os.getenv("FEISHU_LOG_PREVIEW_LENGTH", "120")
    try:
        return max(16, int(raw))
    except ValueError:
        return 120


def _should_redact_previews() -> bool:
    return os.getenv("FEISHU_LOG_REDACT_PREVIEWS", "false").strip().lower() in {"1", "true", "yes", "on"}


def _redact_preview(text: str) -> str:
    redacted = _TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = _LONG_NUMBER_RE.sub("[REDACTED_NUMBER]", redacted)
    return redacted


def preview_text(text: str, limit: int | None = None) -> str:
    """Return a whitespace-normalized preview safe for logs."""
    max_length = limit if limit is not None else _preview_length()
    normalized = " ".join((text or "").split())
    if _should_redact_previews():
        normalized = _redact_preview(normalized)
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3]}..."


def _normalize_fields(fields: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in fields.items():
        if _is_empty_value(value):
            continue
        normalized[key] = value
    return normalized


def _extract_error_info(exc: BaseException) -> dict[str, object]:
    error_type = type(exc).__name__
    error_message = str(exc) or error_type
    error_code: str | None = None

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is not None:
        error_code = str(status_code)

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        payload = body.get("error")
        if isinstance(payload, dict):
            provider_message = payload.get("message")
            provider_code = payload.get("code") or payload.get("type")
            if provider_message:
                error_message = str(provider_message)
            if provider_code and not error_code:
                error_code = str(provider_code)

    if not error_code:
        matched = _ERROR_CODE_RE.search(error_message)
        if matched:
            error_code = matched.group(1)

    return {
        "error_type": error_type,
        "error_code": error_code,
        "error_message": preview_text(error_message, limit=240),
    }


def log_event(
    event: str,
    *,
    level: int | None = None,
    message: str | None = None,
    **fields: object,
) -> None:
    """Emit a structured event log with stable fields."""
    resolved_level = logging.DEBUG if event in _DEFAULT_DEBUG_EVENTS else logging.INFO
    normalized_fields = _normalize_fields(fields)
    logger.log(
        resolved_level if level is None else level,
        message or event,
        extra={
            "event": event,
            **normalized_fields,
        },
    )
    if normalized_fields:
        update_request_state(**normalized_fields)


def log_exception(
    event: str,
    exc: BaseException,
    *,
    message: str | None = None,
    **fields: object,
) -> None:
    """Emit a structured exception log with traceback attached."""
    normalized_fields = _normalize_fields(fields)
    error_info = _extract_error_info(exc)
    logger.error(
        message or str(exc) or event,
        extra={
            "event": event,
            "status": "error",
            "error": str(exc),
            **error_info,
            **normalized_fields,
        },
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    update_request_state(status="error", **error_info, **normalized_fields)


def emit_request_summary(
    *,
    status: str,
    event: str = "request_summary",
    level: int = logging.INFO,
    **fields: object,
) -> None:
    """Emit one normalized request summary line with merged request metrics."""
    state = get_request_state()
    merged_fields: dict[str, object] = {}
    merged_fields.update(state.get("fields", {}))
    merged_fields.update(state.get("timings", {}))
    merged_fields["status"] = status
    total_ms = get_request_total_duration_ms()
    if total_ms is not None and "total_ms" not in merged_fields:
        merged_fields["total_ms"] = total_ms
    merged_fields.update(_normalize_fields(fields))
    logger.log(
        level,
        event,
        extra={
            "event": event,
            **_normalize_fields(merged_fields),
        },
    )
