# PR: Build structured observability foundation for local integration

## Background
This PR focuses on observability hardening instead of new product features. The goal is to make one real request traceable end-to-end: inbound message, agent execution, RAG stages, outbound reply, and failures.

## What changed

### 1) Core observability infrastructure
- Added request-scoped state management and timing aggregation in `observability/context.py`.
- Standardized structured event and exception emission in `observability/events.py`.
- Upgraded logger pipeline in `observability/logging.py`:
  - console + JSONL outputs
  - stable common fields (`service`, `pid`, `runtime_id`)
  - handler dedup
  - third-party noise suppression
  - Lark websocket lifecycle normalization

### 2) Request summary model
- Introduced `request_summary` as the primary per-request summary event.
- Added stage timing fields to summaries:
  - `intent_ms`
  - `retrieval_ms`
  - `llm_ms`
  - `reply_ms`
  - `total_ms`
- Added merged request metadata such as intent, retrieval decision, previews, and result status.

### 3) Agent and RAG pipeline integration
- Wired request state/timing updates into:
  - `agent.py`
  - `multimodal_rag_agent/rag_query_pipeline/pipeline.py`
  - `multimodal_rag_agent/rag_query_pipeline/generator.py`

### 4) Channel logging unification
- Reworked channel-side ad hoc logs into structured events:
  - `channel/feishu/feishu_channel.py`
  - `channel/weixin/weixin_channel.py`
- Covered events including (not exhaustive):
  - `reply_skipped`
  - `channel_auth_refresh_requested`
  - `channel_poll_error`
  - `channel_poll_failed`
  - `attachment_image_saved`
  - `credentials_load_failed`
  - `terminal_render_failed`

### 5) Connection lifecycle observability
- Added normalized websocket lifecycle events:
  - `channel_connection_connecting`
  - `channel_connection_connected`
  - `channel_connection_reconnecting`
  - `channel_connection_recovered`
  - `channel_connection_disconnected`
  - `channel_connection_closed`

### 6) Configuration and documentation
- Extended logging configuration in:
  - `config.py`
  - `.env.example`
- Updated observability guide in:
  - `docs/observability.md`

### 7) Tests
- Added observability-focused tests:
  - `tests/test_observability_logging.py`
  - `tests/test_channel_observability.py`

## Validation
- Command: `pytest -q`
- Result: `15 passed`

## Notes
- A small follow-up fix was included so Feishu `request_failed` logs carry explicit structured context (`channel`, `message_id`, `chat_id`) without relying only on handler-injected fields.

## Operational impact
- Success path diagnosis is now centered on two high-value events:
  - `message_received`
  - `request_summary`
- Failure diagnosis is centered on:
  - `request_failed`
  - connection lifecycle events

## Known limitations / follow-up
- Websocket heartbeat timeout (`ping_timeout`) can still occur under long-running request pressure.
- This PR establishes logging foundations; metrics dashboards/alerts/tracing are still future work.
- Consider adding `conversation_history/` to `.gitignore` if it is local runtime output only.
