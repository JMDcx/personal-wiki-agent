"""Logging configuration for structured console and file output."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import monotonic
from uuid import uuid4

try:
    from feishu_wiki_rag_agent.config import Settings, get_settings
    from feishu_wiki_rag_agent.observability.context import get_log_context
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import Settings, get_settings
    from observability.context import get_log_context


_RESERVED_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}
_PROCESS_ID = os.getpid()
_RUNTIME_ID = uuid4().hex[:12]
_SERVICE_NAME = "feishu_wiki_rag_agent"
_BOUNDARY_EVENTS = {
    "message_received",
    "message_normalized",
    "message_ignored",
    "agent_invoke_started",
    "controller_context_built",
    "agent_invoke_completed",
    "request_completed",
    "request_summary",
    "reply_sent",
    "request_failed",
}
_CONSOLE_EVENTS = {
    "message_received",
    "request_summary",
    "request_failed",
    "reply_skipped",
    "index_rebuild_started",
    "index_rebuild_completed",
    "index_rebuild_failed",
    "channel_connection_connecting",
    "channel_connection_connected",
    "channel_connection_reconnecting",
    "channel_connection_recovered",
    "channel_connection_disconnected",
    "channel_connection_closed",
}
_NOISY_LARK_FRAGMENTS = (
    "processor not found, type: im.message.message_read_v1",
    "processor not found, type: im.chat.access_event.bot_p2p_chat_entered_v1",
)
_LARK_CONNECTION_ID_RE = re.compile(r"\[conn_id=(\d+)\]")
_LARK_ERROR_CODE_RE = re.compile(r"\b(?:received|sent)\s+(\d{4})\b")
_LARK_CONNECTED_MARKERS = (
    "open websocket connection success",
    "websocket connect success",
    "connected to websocket",
    "websocket connection established",
    "receive message loop start",
)
_LARK_RECONNECTING_MARKERS = (
    "reconnect",
    "reconnecting",
    "start to connect",
)


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
    event = getattr(record, "event", record.getMessage())
    payload: dict[str, object] = {
        "timestamp": _iso_timestamp(),
        "level": record.levelname,
        "logger": record.name,
        "service": _SERVICE_NAME,
        "pid": _PROCESS_ID,
        "runtime_id": _RUNTIME_ID,
        "event": event,
    }
    message = record.getMessage()
    if message != event:
        payload["message"] = message
    for key, value in record.__dict__.items():
        if key in _RESERVED_RECORD_KEYS:
            continue
        payload[key] = _normalize_value(value)
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


class BoundaryEventDedupFilter(logging.Filter):
    """Drop obviously duplicated boundary events inside the same process."""

    def __init__(self, *, window_seconds: float = 5.0) -> None:
        super().__init__()
        self.window_seconds = window_seconds
        self._recent: dict[tuple[object, ...], float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        event = getattr(record, "event", record.getMessage())
        if event not in _BOUNDARY_EVENTS:
            return True

        request_id = getattr(record, "request_id", "")
        message_id = getattr(record, "message_id", "")
        if not request_id or not message_id:
            return True

        signature = (
            record.name,
            record.levelno,
            str(event),
            str(request_id),
            str(message_id),
            str(getattr(record, "stage", "")),
            record.getMessage(),
        )
        now = monotonic()
        self._recent = {
            key: seen_at for key, seen_at in self._recent.items() if now - seen_at < self.window_seconds
        }
        if signature in self._recent:
            return False
        self._recent[signature] = now
        return True


class ThirdPartyNoiseFilter(logging.Filter):
    """Suppress known low-signal third-party SDK chatter."""

    def __init__(self, ignored_fragments: Iterable[str]) -> None:
        super().__init__()
        self.ignored_fragments = tuple(fragment for fragment in ignored_fragments if fragment)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in self.ignored_fragments)


class ConsoleEventFilter(logging.Filter):
    """Keep the console stream focused on high-signal lifecycle and summary events."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        event = getattr(record, "event", record.getMessage())
        return event in _CONSOLE_EVENTS


def _extract_lark_error_code(message: str) -> str | None:
    matched = _LARK_ERROR_CODE_RE.search(message)
    if matched:
        return matched.group(1)
    return None


def _extract_lark_connection_id(message: str) -> str | None:
    matched = _LARK_CONNECTION_ID_RE.search(message)
    if matched:
        return matched.group(1)
    return None


def _classify_lark_record(record: logging.LogRecord) -> dict[str, object] | None:
    if record.name != "Lark":
        return None

    message = record.getMessage()
    connection_id = _extract_lark_connection_id(message)
    error_code = _extract_lark_error_code(message)

    if "ping_timeout" in message or "receive message loop exit" in message:
        return {
            "event": "channel_connection_disconnected",
            "level": logging.WARNING,
            "channel": "feishu",
            "transport": "websocket",
            "connection_state": "disconnected",
            "connection_id": connection_id,
            "error_type": "WebSocketDisconnect",
            "error_code": error_code,
            "error_message": "ping_timeout" if "ping_timeout" in message else message,
        }

    lowered = message.lower()
    if any(marker in lowered for marker in _LARK_RECONNECTING_MARKERS):
        return {
            "event": "channel_connection_reconnecting",
            "level": logging.INFO,
            "channel": "feishu",
            "transport": "websocket",
            "connection_state": "reconnecting",
            "connection_id": connection_id,
        }

    if any(marker in lowered for marker in _LARK_CONNECTED_MARKERS):
        return {
            "event": "channel_connection_connected",
            "level": logging.INFO,
            "channel": "feishu",
            "transport": "websocket",
            "connection_state": "connected",
            "connection_id": connection_id,
        }

    return None


class LarkRawDropFilter(logging.Filter):
    """Suppress raw Lark SDK lines once we normalize them into structured events."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "Lark":
            return True
        if _classify_lark_record(record) is not None:
            return False
        message = record.getMessage()
        return not any(fragment in message for fragment in _NOISY_LARK_FRAGMENTS)


class LarkLifecycleBridgeHandler(logging.Handler):
    """Mirror raw Lark websocket logs into stable structured connection events."""

    def __init__(self, *, window_seconds: float = 10.0) -> None:
        super().__init__(level=logging.INFO)
        self.window_seconds = window_seconds
        self._recent: dict[tuple[object, ...], float] = {}
        self._was_disconnected = False

    def emit(self, record: logging.LogRecord) -> None:
        normalized = _classify_lark_record(record)
        if normalized is None:
            return

        event = str(normalized["event"])
        if event == "channel_connection_connected" and self._was_disconnected:
            normalized["event"] = "channel_connection_recovered"
            event = "channel_connection_recovered"
            normalized["connection_state"] = "connected"
        elif event == "channel_connection_disconnected":
            self._was_disconnected = True
        elif event == "channel_connection_connected":
            self._was_disconnected = False

        connection_id = str(normalized.get("connection_id", "") or "")
        signature = (
            event,
            connection_id,
            str(normalized.get("error_code", "") or ""),
            str(normalized.get("error_message", "") or ""),
        )
        now = monotonic()
        self._recent = {
            key: seen_at for key, seen_at in self._recent.items() if now - seen_at < self.window_seconds
        }
        if signature in self._recent:
            return
        self._recent[signature] = now

        logging.getLogger("feishu_wiki_rag_agent.events").log(
            int(normalized.pop("level", logging.INFO)),
            event,
            extra=normalized,
        )


class JsonFormatter(logging.Formatter):
    """Render logs as single-line JSON documents."""

    def __init__(self, *, include_tracebacks: bool = False) -> None:
        super().__init__()
        self.include_tracebacks = include_tracebacks

    def format(self, record: logging.LogRecord) -> str:
        payload = _build_payload(record)
        if record.exc_info and self.include_tracebacks:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class KeyValueFormatter(logging.Formatter):
    """Render logs as compact key-value lines for local development."""

    def __init__(self, *, include_tracebacks: bool = False) -> None:
        super().__init__()
        self.include_tracebacks = include_tracebacks

    def format(self, record: logging.LogRecord) -> str:
        payload = _build_payload(record)
        exception = ""
        if record.exc_info and self.include_tracebacks:
            exception = self.formatException(record.exc_info)

        timestamp = str(payload.pop("timestamp"))
        level = str(payload.pop("level"))
        logger_name = str(payload.pop("logger"))
        event = str(payload.pop("event", record.getMessage()))
        message = str(payload.pop("message", event))

        parts = [f"event={json.dumps(event, ensure_ascii=False)}"]
        for key in sorted(payload):
            parts.append(f"{key}={json.dumps(payload[key], ensure_ascii=False)}")
        if message != event:
            parts.append(f"message={json.dumps(message, ensure_ascii=False)}")

        line = f"{timestamp} {level} {logger_name} {' '.join(parts)}"
        if exception:
            return f"{line}\n{exception}"
        return line


class ConsoleFormatter(logging.Formatter):
    """Render concise, developer-friendly console summaries."""

    def __init__(self, *, include_tracebacks: bool = False) -> None:
        super().__init__()
        self.include_tracebacks = include_tracebacks

    def format(self, record: logging.LogRecord) -> str:
        payload = _build_payload(record)
        exception = ""
        if record.exc_info and self.include_tracebacks:
            exception = self.formatException(record.exc_info)

        timestamp = datetime.fromisoformat(str(payload.pop("timestamp"))).strftime("%H:%M:%S")
        level = str(payload.pop("level"))
        event = str(payload.pop("event", record.getMessage()))

        parts: list[str] = []
        if event == "request_summary":
            status = str(payload.get("status", "unknown"))
            request_id = str(payload.get("request_id", "") or "-")
            intent = str(payload.get("intent", "") or "-")
            channel = str(payload.get("channel", "") or "-")
            parts.extend([f"status={status}", f"channel={channel}", f"intent={intent}", f"req={request_id}"])
            for key in (
                "total_ms",
                "intent_ms",
                "history_ms",
                "intent_model_ms",
                "retrieval_ms",
                "generation_ms",
                "llm_ms",
                "reply_ms",
            ):
                if key in payload:
                    parts.append(f"{key}={payload[key]}")
            if "question_preview" in payload:
                parts.append(f"q={json.dumps(payload['question_preview'], ensure_ascii=False)}")
            if status != "ok":
                if "error_type" in payload:
                    parts.append(f"error_type={payload['error_type']}")
                if "error_code" in payload:
                    parts.append(f"error_code={payload['error_code']}")
                if "error_message" in payload:
                    parts.append(f"error={json.dumps(payload['error_message'], ensure_ascii=False)}")
        elif event.startswith("channel_connection_"):
            parts.append(f"channel={payload.get('channel', 'feishu')}")
            parts.append(f"state={payload.get('connection_state', '-')}")
            if "error_code" in payload:
                parts.append(f"error_code={payload['error_code']}")
            if "error_message" in payload:
                parts.append(f"error={json.dumps(payload['error_message'], ensure_ascii=False)}")
        else:
            for key in ("request_id", "channel", "chat_type", "message_type", "stage", "error_type", "error_code"):
                if key in payload:
                    parts.append(f"{key}={json.dumps(payload[key], ensure_ascii=False)}")
            if "question_preview" in payload:
                parts.append(f"q={json.dumps(payload['question_preview'], ensure_ascii=False)}")
            if "error_message" in payload:
                parts.append(f"error={json.dumps(payload['error_message'], ensure_ascii=False)}")

        line = f"{timestamp} {level:<5} {event}"
        if parts:
            line = f"{line} {' '.join(parts)}"
        if exception:
            return f"{line}\n{exception}"
        return line


def _resolve_level(level_name: str) -> int:
    return getattr(logging, (level_name or "INFO").strip().upper(), logging.INFO)


def _root_level(settings: Settings) -> int:
    return min(
        _resolve_level(settings.log_level),
        _resolve_level(settings.log_console_level),
        _resolve_level(settings.log_json_level),
    )


def _root_has_file_handler(root_logger: logging.Logger, target_path: str) -> bool:
    resolved_target = str(Path(target_path).expanduser().resolve())
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", "") == resolved_target:
            return True
    return False


def _handler_has_context_filter(handler: logging.Handler) -> bool:
    return any(isinstance(existing_filter, ContextFilter) for existing_filter in handler.filters)


def _handler_has_filter_type(handler: logging.Handler, filter_type: type[logging.Filter]) -> bool:
    return any(isinstance(existing_filter, filter_type) for existing_filter in handler.filters)


def _logger_has_handler_type(logger: logging.Logger, handler_type: type[logging.Handler]) -> bool:
    return any(isinstance(handler, handler_type) for handler in logger.handlers)


def _configure_third_party_loggers(settings: Settings) -> None:
    logging.getLogger("httpx").setLevel(_resolve_level(settings.log_httpx_level))
    logging.getLogger("httpcore").setLevel(_resolve_level(settings.log_httpx_level))
    logging.getLogger("openai").setLevel(_resolve_level(settings.log_openai_level))
    lark_logger = logging.getLogger("Lark")
    lark_logger.setLevel(_resolve_level(settings.log_lark_level))
    if not _logger_has_handler_type(lark_logger, LarkLifecycleBridgeHandler):
        lark_logger.addHandler(LarkLifecycleBridgeHandler())


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure root logging once for the whole application."""
    global _SERVICE_NAME
    resolved = settings or get_settings()
    _SERVICE_NAME = resolved.log_service_name
    level = _root_level(resolved)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    context_filter = ContextFilter()
    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(_resolve_level(resolved.log_console_level))
        console_handler.setFormatter(ConsoleFormatter(include_tracebacks=resolved.log_include_tracebacks))
        if not _handler_has_context_filter(console_handler):
            console_handler.addFilter(context_filter)
        if not _handler_has_filter_type(console_handler, BoundaryEventDedupFilter):
            console_handler.addFilter(BoundaryEventDedupFilter())
        if not _handler_has_filter_type(console_handler, ConsoleEventFilter):
            console_handler.addFilter(ConsoleEventFilter())
        if not _handler_has_filter_type(console_handler, LarkRawDropFilter):
            console_handler.addFilter(LarkRawDropFilter())
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            if not _handler_has_context_filter(handler):
                handler.addFilter(context_filter)
            if not _handler_has_filter_type(handler, BoundaryEventDedupFilter):
                handler.addFilter(BoundaryEventDedupFilter())
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) and not _handler_has_filter_type(handler, ConsoleEventFilter):
                handler.addFilter(ConsoleEventFilter())
            if not _handler_has_filter_type(handler, LarkRawDropFilter):
                handler.addFilter(LarkRawDropFilter())

    log_file_path = resolved.log_file_path.expanduser().resolve()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    if not _root_has_file_handler(root_logger, str(log_file_path)):
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=resolved.log_max_bytes,
            backupCount=resolved.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(_resolve_level(resolved.log_json_level))
        file_handler.setFormatter(JsonFormatter(include_tracebacks=resolved.log_include_tracebacks))
        if not _handler_has_context_filter(file_handler):
            file_handler.addFilter(context_filter)
        if not _handler_has_filter_type(file_handler, BoundaryEventDedupFilter):
            file_handler.addFilter(BoundaryEventDedupFilter())
        if not _handler_has_filter_type(file_handler, LarkRawDropFilter):
            file_handler.addFilter(LarkRawDropFilter())
        if resolved.log_suppress_noisy_lark_events and not _handler_has_filter_type(file_handler, ThirdPartyNoiseFilter):
            file_handler.addFilter(ThirdPartyNoiseFilter(_NOISY_LARK_FRAGMENTS))
        root_logger.addHandler(file_handler)
    else:
        for handler in root_logger.handlers:
            if not isinstance(handler, RotatingFileHandler):
                continue
            if not _handler_has_filter_type(handler, LarkRawDropFilter):
                handler.addFilter(LarkRawDropFilter())
            if resolved.log_suppress_noisy_lark_events and not _handler_has_filter_type(handler, ThirdPartyNoiseFilter):
                handler.addFilter(ThirdPartyNoiseFilter(_NOISY_LARK_FRAGMENTS))

    _configure_third_party_loggers(resolved)
    logging.captureWarnings(True)
