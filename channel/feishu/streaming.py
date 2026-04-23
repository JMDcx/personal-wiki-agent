"""Feishu streaming response adapter."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Protocol

try:
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception, preview_text
    from feishu_wiki_rag_agent.protocols.streaming import StreamEvent
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.events import log_event, log_exception, preview_text
    from protocols.streaming import StreamEvent


THINKING_REPLY_TEXT = "thinking..."
_EMPTY_FINAL_TEXT = "当前索引中未找到相关内容。"


class _FeishuStreamingClient(Protocol):
    def reply_text(self, message_id: str, text: str) -> str: ...

    def update_text(self, message_id: str, text: str) -> None: ...


class FeishuStreamingResponder:
    """Render stream events by replying once, then patching that reply."""

    def __init__(
        self,
        *,
        client: _FeishuStreamingClient,
        source_message_id: str,
        initial_reply_message_id: str = "",
        update_interval_seconds: float = 2.5,
        max_chars: int = 6000,
        max_update_count: int = 25,
        max_stream_seconds: float = 300.0,
        now_provider=time.monotonic,
    ) -> None:
        self.client = client
        self.source_message_id = source_message_id
        self.reply_message_id = initial_reply_message_id.strip()
        self.update_interval_seconds = max(0.0, update_interval_seconds)
        self.max_chars = max(200, max_chars)
        self.max_update_count = max(1, max_update_count)
        self.max_stream_seconds = max(1.0, max_stream_seconds)
        self.now_provider = now_provider
        self.placeholder_sent = bool(self.reply_message_id)
        self.update_available = True
        self.status_text = THINKING_REPLY_TEXT
        self.answer_text = ""
        self.stream_started_at = self.now_provider() if self.placeholder_sent else 0.0
        self.last_flush_at = 0.0
        self.non_final_update_count = 0

    def run(self, events: Iterable[StreamEvent]) -> str:
        final_text = ""
        self._ensure_placeholder()
        for event in events:
            if event.event_type == "started":
                if event.text:
                    self.status_text = event.text
                continue
            if event.event_type == "status":
                self.status_text = event.text or self.status_text
                log_event(
                    "stream_status_emitted",
                    stage=event.stage,
                    status_preview=preview_text(self.status_text),
                )
                self._flush(force=True)
                continue
            if event.event_type == "text_delta":
                self.answer_text += event.text
                log_event(
                    "stream_delta_emitted",
                    stage=event.stage,
                    delta_length=len(event.text),
                    answer_length=len(self.answer_text),
                )
                self._flush(force=False)
                continue
            if event.event_type == "final":
                final_text = event.text.strip() or _EMPTY_FINAL_TEXT
                self.answer_text = final_text
                self.status_text = ""
                self._flush(force=True, final=True)
                log_event("stream_completed", answer_length=len(final_text))
                return final_text
            if event.event_type == "error":
                final_text = event.text or "处理失败，请稍后重试。"
                self.answer_text = final_text
                self.status_text = ""
                self._flush(force=True, final=True)
                log_event("stream_failed", level=logging.ERROR, error_preview=preview_text(final_text))
                return final_text
        final_text = final_text or self.answer_text
        if final_text:
            self._flush(force=True, final=True)
        return final_text

    def _ensure_placeholder(self) -> None:
        if self.placeholder_sent:
            return
        self.reply_message_id = self.client.reply_text(self.source_message_id, THINKING_REPLY_TEXT)
        self.placeholder_sent = True
        self.stream_started_at = self.now_provider()
        if not self.reply_message_id:
            self.update_available = False
        log_event("stream_started", reply_message_id=self.reply_message_id)

    def _flush(self, *, force: bool, final: bool = False) -> None:
        if not self.placeholder_sent:
            self._ensure_placeholder()
        now = self.now_provider()
        if not force and now - self.last_flush_at < self.update_interval_seconds:
            return
        if not final and self.non_final_update_count >= self.max_update_count:
            return
        rendered = self._render_final() if final else self._render_progress()
        if not rendered:
            return
        stream_expired = now - self.stream_started_at > self.max_stream_seconds
        if stream_expired:
            log_event(
                "stream_fallback_used",
                level=logging.WARNING,
                reply_message_id=self.reply_message_id,
                final=final,
                reason="max_stream_seconds_exceeded",
                elapsed_seconds=round(now - self.stream_started_at, 1),
                max_stream_seconds=self.max_stream_seconds,
            )
        if self.update_available and self.reply_message_id and not stream_expired:
            try:
                self.client.update_text(self.reply_message_id, rendered)
                self.last_flush_at = now
                if not final:
                    self.non_final_update_count += 1
                log_event(
                    "stream_update_sent",
                    final=final,
                    rendered_length=len(rendered),
                    reply_message_id=self.reply_message_id,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.update_available = False
                log_exception(
                    "stream_fallback_used",
                    exc,
                    reply_message_id=self.reply_message_id,
                    final=final,
                )
        if final:
            self.client.reply_text(self.source_message_id, rendered)
            self.last_flush_at = now

    def _render_progress(self) -> str:
        lines = [THINKING_REPLY_TEXT, f"当前进度：{self.status_text or THINKING_REPLY_TEXT}"]
        if self.answer_text.strip():
            lines.extend(["", "已生成：", self._truncate(self.answer_text)])
        return "\n".join(lines)

    def _render_final(self) -> str:
        return self._truncate(self.answer_text).strip() or _EMPTY_FINAL_TEXT

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return f"{text[: self.max_chars - 20].rstrip()}\n\n...（内容过长，已截断）"
