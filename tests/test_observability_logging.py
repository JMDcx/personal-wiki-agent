from __future__ import annotations

import json
import logging

from feishu_wiki_rag_agent.config import Settings
from feishu_wiki_rag_agent.observability.context import bind_request_context, record_request_timing
from feishu_wiki_rag_agent.observability.events import emit_request_summary, log_event
from feishu_wiki_rag_agent.observability.logging import (
    BoundaryEventDedupFilter,
    ConsoleFormatter,
    JsonFormatter,
    LarkLifecycleBridgeHandler,
    LarkRawDropFilter,
    ThirdPartyNoiseFilter,
    configure_logging,
)


def _make_record(
    *,
    name: str = "feishu_wiki_rag_agent.events",
    level: int = logging.INFO,
    message: str = "message_received",
    event: str = "message_received",
    request_id: str = "req-1",
    message_id: str = "msg-1",
) -> logging.LogRecord:
    record = logging.LogRecord(name, level, __file__, 1, message, (), None)
    record.event = event
    record.request_id = request_id
    record.message_id = message_id
    return record


def test_json_formatter_omits_traceback_by_default() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        record = _make_record(level=logging.ERROR, message="boom", event="request_failed")
        record.error = "boom"
        record.error_type = type(exc).__name__
        record.exc_info = (type(exc), exc, exc.__traceback__)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "request_failed"
    assert payload["error"] == "boom"
    assert "exception" not in payload
    assert payload["message"] == "boom"


def test_json_formatter_includes_traceback_when_enabled() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        record = _make_record(level=logging.ERROR, message="boom", event="request_failed")
        record.error = "boom"
        record.error_type = type(exc).__name__
        record.exc_info = (type(exc), exc, exc.__traceback__)

    payload = json.loads(JsonFormatter(include_tracebacks=True).format(record))

    assert payload["event"] == "request_failed"
    assert "RuntimeError: boom" in payload["exception"]


def test_json_formatter_omits_message_when_same_as_event() -> None:
    record = _make_record(message="request_summary", event="request_summary")

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "request_summary"
    assert "message" not in payload


def test_boundary_event_dedup_filter_suppresses_exact_duplicates() -> None:
    dedupe_filter = BoundaryEventDedupFilter(window_seconds=5.0)

    first = _make_record()
    duplicate = _make_record()
    different_message = _make_record(message_id="msg-2")

    assert dedupe_filter.filter(first) is True
    assert dedupe_filter.filter(duplicate) is False
    assert dedupe_filter.filter(different_message) is True


def test_third_party_noise_filter_drops_known_lark_read_receipts() -> None:
    noise_filter = ThirdPartyNoiseFilter(
        ("processor not found, type: im.message.message_read_v1",),
    )
    noisy_record = _make_record(
        name="Lark",
        level=logging.ERROR,
        message="handle message failed, processor not found, type: im.message.message_read_v1",
        event="handle message failed, processor not found, type: im.message.message_read_v1",
    )
    useful_record = _make_record(
        name="Lark",
        level=logging.ERROR,
        message="tenant_access_token expired",
        event="tenant_access_token expired",
    )

    assert noise_filter.filter(noisy_record) is False
    assert noise_filter.filter(useful_record) is True


def test_configure_logging_reuses_same_file_handler_for_resolved_path(tmp_path) -> None:
    settings = Settings(log_file_path=tmp_path / "logs" / "app.jsonl")
    settings.ensure_directories()

    configure_logging(settings, force=True)
    configure_logging(settings)

    root_logger = logging.getLogger()
    file_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "baseFilename", "").endswith("app.jsonl")
    ]

    assert len(file_handlers) == 1


def test_message_normalized_defaults_to_debug_level(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="feishu_wiki_rag_agent.events")

    log_event("message_normalized", text_preview="hello")

    record = next(record for record in caplog.records if getattr(record, "event", "") == "message_normalized")
    assert record.levelno == logging.DEBUG


def test_emit_request_summary_merges_stage_timings(caplog) -> None:
    caplog.set_level(logging.INFO, logger="feishu_wiki_rag_agent.events")

    with bind_request_context(request_id="req-123", channel="feishu"):
        log_event("message_received", message_type="text", chat_type="p2p")
        record_request_timing("intent_ms", 12.3)
        record_request_timing("retrieval_ms", 45.6)
        emit_request_summary(status="ok", intent="kb_search", question_preview="hello")

    record = next(record for record in caplog.records if getattr(record, "event", "") == "request_summary")
    assert record.status == "ok"
    assert record.intent == "kb_search"
    assert record.intent_ms == 12.3
    assert record.retrieval_ms == 45.6


def test_console_formatter_renders_compact_request_summary() -> None:
    record = _make_record(message="request_summary", event="request_summary")
    record.status = "ok"
    record.channel = "feishu"
    record.intent = "greeting"
    record.total_ms = 123.4
    record.intent_ms = 11.1
    record.reply_ms = 5.0

    line = ConsoleFormatter().format(record)

    assert "request_summary" in line
    assert "status=ok" in line
    assert "intent=greeting" in line
    assert "total_ms=123.4" in line


def test_lark_raw_drop_filter_suppresses_ping_timeout_records() -> None:
    drop_filter = LarkRawDropFilter()
    record = _make_record(
        name="Lark",
        level=logging.ERROR,
        message="receive message loop exit, err: received 3003 (registered) ping_timeout [conn_id=1]",
        event="receive message loop exit, err: received 3003 (registered) ping_timeout [conn_id=1]",
    )

    assert drop_filter.filter(record) is False


def test_lark_lifecycle_bridge_emits_structured_disconnect(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="feishu_wiki_rag_agent.events")
    bridge = LarkLifecycleBridgeHandler()
    record = _make_record(
        name="Lark",
        level=logging.ERROR,
        message="receive message loop exit, err: received 3003 (registered) ping_timeout [conn_id=7]",
        event="receive message loop exit, err: received 3003 (registered) ping_timeout [conn_id=7]",
    )

    bridge.emit(record)

    normalized = next(record for record in caplog.records if getattr(record, "event", "") == "channel_connection_disconnected")
    assert normalized.channel == "feishu"
    assert normalized.connection_state == "disconnected"
    assert normalized.error_code == "3003"
