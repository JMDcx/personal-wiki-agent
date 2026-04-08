"""Context propagation for structured logs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator
from uuid import uuid4


_LOG_CONTEXT: ContextVar[dict[str, object]] = ContextVar("log_context", default={})


def _is_empty_value(value: object) -> bool:
    return value is None or value == ""


def get_log_context() -> dict[str, object]:
    """Return a copy of the current log context."""
    return dict(_LOG_CONTEXT.get({}))


def make_request_id(prefix: str) -> str:
    """Return a stable opaque request id with a readable prefix."""
    return f"{prefix}:{uuid4().hex}"


@contextmanager
def bind_log_context(**fields: object) -> Iterator[dict[str, object]]:
    """Temporarily merge fields into the current log context."""
    current = get_log_context()
    merged = dict(current)
    for key, value in fields.items():
        if _is_empty_value(value):
            continue
        merged[key] = value
    token = _LOG_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _LOG_CONTEXT.reset(token)
