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
- `FEISHU_LOG_FILE_PATH`
- `FEISHU_LOG_MAX_BYTES`
- `FEISHU_LOG_BACKUP_COUNT`

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
- Failures
  - `request_failed`
  - `attachment_adaptation_failed`

## Shared Fields

Prefer these shared fields when relevant:

- `request_id`
- `thread_id`
- `channel`
- `message_id`
- `duration_ms`
- `success`
- `status`
- `tool_name`
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
