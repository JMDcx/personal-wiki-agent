"""Convenience helpers for structured event logging."""

from __future__ import annotations

import logging


logger = logging.getLogger("feishu_wiki_rag_agent.events")


def _is_empty_value(value: object) -> bool:
    return value is None or value == ""


def preview_text(text: str, limit: int = 160) -> str:
    """Return a whitespace-normalized preview safe for logs."""
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _normalize_fields(fields: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in fields.items():
        if _is_empty_value(value):
            continue
        normalized[key] = value
    return normalized


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    message: str | None = None,
    **fields: object,
) -> None:
    """Emit a structured event log with stable fields."""
    logger.log(
        level,
        message or event,
        extra={
            "event": event,
            **_normalize_fields(fields),
        },
    )


def log_exception(
    event: str,
    exc: BaseException,
    *,
    message: str | None = None,
    **fields: object,
) -> None:
    """Emit a structured exception log with traceback attached."""
    logger.error(
        message or str(exc) or event,
        extra={
            "event": event,
            "error": str(exc),
            "error_type": type(exc).__name__,
            **_normalize_fields(fields),
        },
        exc_info=(type(exc), exc, exc.__traceback__),
    )
