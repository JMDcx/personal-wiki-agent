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


class _FeishuStreamingClient(Protocol):
    def reply_text(self, message_id: str, text: str) -> str: ...

    def patch_text(self, message_id: str, text: str) -> None: ...


class FeishuStreamingResponder:
    """Render stream events by replying once, then patching that reply."""

    def __init__(
        self,
        *,
        client: _FeishuStreamingClient,
        source_message_id: str,
        update_interval_seconds: float = 1.5,
        max_chars: int = 6000,
        now_provider=time.monotonic,
    ) -> None:
        self.client = client
        self.source_message_id = source_message_id
        self.update_interval_seconds = max(0.0, update_interval_seconds)
        self.max_chars = max(200, max_chars)
        self.now_provider = now_provider
        self.reply_message_id = ""
        self.placeholder_sent = False
        self.patch_available = True
        self.status_text = "正在处理..."
        self.answer_text = ""
        self.last_flush_at = 0.0

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
                final_text = event.text
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
        self.reply_message_id = self.client.reply_text(self.source_message_id, "正在处理...")
        self.placeholder_sent = True
        if not self.reply_message_id:
            self.patch_available = False
        log_event("stream_started", reply_message_id=self.reply_message_id)

    def _flush(self, *, force: bool, final: bool = False) -> None:
        if not self.placeholder_sent:
            self._ensure_placeholder()
        now = self.now_provider()
        if not force and now - self.last_flush_at < self.update_interval_seconds:
            return
        rendered = self._render_final() if final else self._render_progress()
        if not rendered:
            return
        if self.patch_available and self.reply_message_id:
            try:
                self.client.patch_text(self.reply_message_id, rendered)
                self.last_flush_at = now
                log_event(
                    "stream_update_sent",
                    final=final,
                    rendered_length=len(rendered),
                    reply_message_id=self.reply_message_id,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.patch_available = False
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
        lines = [f"状态：{self.status_text or '正在处理...'}"]
        if self.answer_text.strip():
            lines.extend(["", "已生成：", self._truncate(self.answer_text)])
        return "\n".join(lines)

    def _render_final(self) -> str:
        return self._truncate(self.answer_text).strip() or "处理完成。"

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return f"{text[: self.max_chars - 20].rstrip()}\n\n...（内容过长，已截断）"
