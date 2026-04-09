# Observability Guide

## Goals

- Make one user request traceable from inbound message to outbound reply.
- Keep logs structured and easy to extend.
- Avoid logging secrets or oversized raw payloads.

## Current Setup

- `observability/logging.py`
  - Configures console logs plus rotating JSONL file logs.
- `observability/context.py`
  - Carries request-scoped context such as `request_id`, `thread_id`, `channel`, and `message_id`.
- `observability/events.py`
  - Provides `log_event`, `log_exception`, and `preview_text`.

## Default Config

Environment variables:

- `FEISHU_LOG_LEVEL`
- `FEISHU_LOG_CONSOLE_LEVEL`
- `FEISHU_LOG_JSON_LEVEL`
- `FEISHU_LOG_FILE_PATH`
- `FEISHU_LOG_MAX_BYTES`
- `FEISHU_LOG_BACKUP_COUNT`
- `FEISHU_LOG_SERVICE_NAME`
- `FEISHU_LOG_INCLUDE_TRACEBACKS`
- `FEISHU_LOG_HTTPX_LEVEL`
- `FEISHU_LOG_OPENAI_LEVEL`
- `FEISHU_LOG_LARK_LEVEL`
- `FEISHU_LOG_SUPPRESS_NOISY_LARK_EVENTS`
- `FEISHU_LOG_PREVIEW_LENGTH`
- `FEISHU_LOG_REDACT_PREVIEWS`

Default log file:

- `data/logs/app.jsonl`

## Event Naming

Use stable event names at request boundaries instead of ad hoc prose.

Current event families:

- Message lifecycle
  - `message_received`
  - `message_normalized`
  - `message_ignored`
  - `reply_sent`
  - `request_summary`
- Channel operations
  - `reply_skipped`
  - `channel_auth_refresh_requested`
  - `channel_poll_error`
  - `channel_poll_failed`
  - `attachment_image_saved`
  - `credentials_load_failed`
  - `terminal_render_failed`
  - `channel_connection_connecting`
  - `channel_connection_connected`
  - `channel_connection_reconnecting`
  - `channel_connection_recovered`
  - `channel_connection_disconnected`
  - `channel_connection_closed`
- Controller and tools
  - `controller_context_built`
  - `agent_invoke_started`
  - `agent_invoke_completed`
  - `tool_called`
  - `tool_completed`
  - `tool_failed`
- Retrieval and generation
  - `retrieval_started`
  - `retrieval_completed`
  - `retrieval_failed`
  - `generation_started`
  - `generation_completed`
  - `generation_failed`
- Deposit and ingest
  - `deposit_started`
  - `deposit_completed`
  - `deposit_failed`
  - `ingest_started`
  - `ingest_completed`
  - `ingest_failed`
  - `feishu_write_started`
  - `feishu_write_completed`
  - `feishu_write_failed`
- Maintenance
  - `index_rebuild_started`
  - `index_rebuild_completed`
  - `index_rebuild_failed`
  - `request_completed`
- Failures
  - `request_failed`
  - `attachment_adaptation_failed`

## Shared Fields

Prefer these shared fields when relevant:

- `request_id`
- `thread_id`
- `channel`
- `message_id`
- `service`
- `pid`
- `runtime_id`
- `duration_ms`
- `intent_ms`
- `retrieval_ms`
- `llm_ms`
- `reply_ms`
- `total_ms`
- `success`
- `status`
- `reason`
- `tool_name`
- `error_type`
- `error_code`
- `error_message`
- `query_preview`
- `text_preview`
- `answer_preview`

## Rules For New Features

1. Bind context at the outermost boundary.
   - Example: channel handlers, API routes, background jobs.
2. Log start and completion at major boundaries.
   - Example: tool call, retrieval, writeback, reply send.
3. Use `preview_text(...)` for user content.
   - Do not dump full prompts, OCR blobs, or secrets into logs.
4. Record timings as `duration_ms`.
5. Use `log_exception(...)` exactly once at the boundary that handles the error.
6. Keep field names stable.
   - Prefer `chunk_count`, `source_count`, `image_count`, `candidate_count`.

## Current Defaults

The logging setup now follows a leaner local-debug profile:

- Keep one `request_summary` line per handled request at `INFO`.
- Keep `message_received` at `INFO` so inbound delivery is still visible.
- Keep most step-by-step diagnostic events at `DEBUG`.
- Downshift noisy transport loggers such as `httpx`, `httpcore`, and `openai` to `WARNING` by default.
- Normalize common Feishu websocket disconnect messages into structured connection lifecycle events.
- Suppress known low-signal Lark read-receipt events and raw websocket timeout noise.
- Include `pid` and `runtime_id` on every record so duplicate processes are obvious.
- Emit stage timings such as `intent_ms`, `retrieval_ms`, `llm_ms`, and `reply_ms` into the request summary.
- Only include full traceback strings when `FEISHU_LOG_INCLUDE_TRACEBACKS=true` or `FEISHU_LOG_LEVEL=DEBUG`.
- Omit the JSON `message` field when it would only repeat the `event` name.

## Recommended Daily Workflow

Most debugging should start with the summary and boundary events instead of scanning the whole file.

Suggested commands:

```bash
tail -f data/logs/app.jsonl
```

```bash
rg '"event": "(message_received|request_summary|request_failed|channel_connection_disconnected|channel_connection_recovered)"' data/logs/app.jsonl | tail -n 50
```

```bash
rg 'feishu:om_' data/logs/app.jsonl | tail -n 30
```

Interpretation guide:

- `message_received` exists and `request_summary.status=ok`: the request completed normally.
- `request_summary.allow_retrieval=false`: the request was answered without knowledge retrieval.
- `request_summary` is the fastest way to inspect one successful request.
- `request_failed` is the fastest way to inspect one failed request.
- `channel_connection_disconnected` with `error_code=3003` usually means a websocket heartbeat timeout.
- Different `runtime_id` values for the same `request_id` usually indicate duplicate running processes.

By default, events such as `message_normalized`, `agent_invoke_started`, `tool_called`, `tool_completed`, `retrieval_started`, `retrieval_completed`, and `reply_sent` stay at `DEBUG` so the normal `INFO` view remains compact.

## Example

```python
from feishu_wiki_rag_agent.observability.context import bind_log_context
from feishu_wiki_rag_agent.observability.events import log_event, log_exception, preview_text

with bind_log_context(request_id="job:123", thread_id="sync:index"):
    log_event("my_feature_started", query_preview=preview_text(user_query))
    try:
        ...
        log_event("my_feature_completed", duration_ms=42.5, success=True)
    except Exception as exc:
        log_exception("my_feature_failed", exc, duration_ms=42.5)
        raise
```
