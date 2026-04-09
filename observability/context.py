"""Context propagation for structured logs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Iterator
from uuid import uuid4


_LOG_CONTEXT: ContextVar[dict[str, object]] = ContextVar("log_context", default={})
_REQUEST_STATE: ContextVar[dict[str, object]] = ContextVar("request_state", default={})


def _is_empty_value(value: object) -> bool:
    return value is None or value == ""


def get_log_context() -> dict[str, object]:
    """Return a copy of the current log context."""
    return dict(_LOG_CONTEXT.get({}))


def has_request_state() -> bool:
    """Return whether a request-scoped summary state is active."""
    return bool(_REQUEST_STATE.get({}))


def get_request_state() -> dict[str, object]:
    """Return a shallow copy of the current request summary state."""
    state = _REQUEST_STATE.get({})
    copied = dict(state)
    copied["fields"] = dict(state.get("fields", {}))
    copied["timings"] = dict(state.get("timings", {}))
    return copied


def make_request_id(prefix: str) -> str:
    """Return a stable opaque request id with a readable prefix."""
    return f"{prefix}:{uuid4().hex}"


def update_request_state(**fields: object) -> None:
    """Merge request-level summary fields into the current request state."""
    current = _REQUEST_STATE.get({})
    if not current:
        return
    merged = dict(current)
    stored_fields = dict(current.get("fields", {}))
    for key, value in fields.items():
        if _is_empty_value(value):
            continue
        stored_fields[key] = value
    merged["fields"] = stored_fields
    _REQUEST_STATE.set(merged)


def record_request_timing(name: str, duration_ms: float) -> None:
    """Store one named stage timing in the current request state."""
    current = _REQUEST_STATE.get({})
    if not current:
        return
    merged = dict(current)
    timings = dict(current.get("timings", {}))
    timings[name] = round(duration_ms, 1)
    merged["timings"] = timings
    _REQUEST_STATE.set(merged)


def get_request_total_duration_ms() -> float | None:
    """Return the current request total duration in milliseconds when available."""
    current = _REQUEST_STATE.get({})
    started_at = current.get("started_at")
    if not isinstance(started_at, (int, float)):
        return None
    return round((perf_counter() - started_at) * 1000, 1)


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


@contextmanager
def bind_request_context(**fields: object) -> Iterator[dict[str, object]]:
    """Bind log context and initialize request summary state for one request."""
    initial_fields = {key: value for key, value in fields.items() if not _is_empty_value(value)}
    token = _REQUEST_STATE.set(
        {
            "started_at": perf_counter(),
            "fields": dict(initial_fields),
            "timings": {},
        }
    )
    try:
        with bind_log_context(**fields) as merged:
            update_request_state(**merged)
            yield merged
    finally:
        _REQUEST_STATE.reset(token)
