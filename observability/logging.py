"""Logging configuration for structured console and file output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

try:
    from feishu_wiki_rag_agent.config import Settings, get_settings
    from feishu_wiki_rag_agent.observability.context import get_log_context
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import Settings, get_settings
    from observability.context import get_log_context


_RESERVED_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}


def _is_empty_value(value: object) -> bool:
    return value is None or value == ""


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    return str(value)


def _build_payload(record: logging.LogRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "timestamp": _iso_timestamp(),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }
    for key, value in record.__dict__.items():
        if key in _RESERVED_RECORD_KEYS:
            continue
        payload[key] = _normalize_value(value)
    payload.setdefault("event", record.getMessage())
    return payload


class ContextFilter(logging.Filter):
    """Inject request-scoped context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in get_log_context().items():
            if _is_empty_value(value):
                continue
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """Render logs as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _build_payload(record)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class KeyValueFormatter(logging.Formatter):
    """Render logs as compact key-value lines for local development."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _build_payload(record)
        exception = ""
        if record.exc_info:
            exception = self.formatException(record.exc_info)

        timestamp = str(payload.pop("timestamp"))
        level = str(payload.pop("level"))
        logger_name = str(payload.pop("logger"))
        message = str(payload.pop("message"))
        event = str(payload.pop("event", message))

        parts = [f"event={json.dumps(event, ensure_ascii=False)}"]
        for key in sorted(payload):
            parts.append(f"{key}={json.dumps(payload[key], ensure_ascii=False)}")
        if message != event:
            parts.append(f"message={json.dumps(message, ensure_ascii=False)}")

        line = f"{timestamp} {level} {logger_name} {' '.join(parts)}"
        if exception:
            return f"{line}\n{exception}"
        return line


def _resolve_level(level_name: str) -> int:
    return getattr(logging, (level_name or "INFO").strip().upper(), logging.INFO)


def _root_has_file_handler(root_logger: logging.Logger, target_path: str) -> bool:
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", "") == target_path:
            return True
    return False


def _handler_has_context_filter(handler: logging.Handler) -> bool:
    return any(isinstance(existing_filter, ContextFilter) for existing_filter in handler.filters)


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure root logging once for the whole application."""
    resolved = settings or get_settings()
    level = _resolve_level(resolved.log_level)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    context_filter = ContextFilter()

    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(KeyValueFormatter())
        if not _handler_has_context_filter(console_handler):
            console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            if not _handler_has_context_filter(handler):
                handler.addFilter(context_filter)

    log_file_path = resolved.log_file_path
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    if not _root_has_file_handler(root_logger, str(log_file_path)):
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=resolved.log_max_bytes,
            backupCount=resolved.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JsonFormatter())
        if not _handler_has_context_filter(file_handler):
            file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
