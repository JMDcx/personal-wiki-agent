"""Standalone Feishu websocket channel for the Feishu Wiki RAG example."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import time
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from dotenv import load_dotenv

try:
    from feishu_wiki_rag_agent.agent import invoke_agent
    from feishu_wiki_rag_agent.config import Settings, get_settings
    from feishu_wiki_rag_agent.channel.feishu.feishu_client import FeishuClient
    from feishu_wiki_rag_agent.observability.context import (
        bind_log_context,
        bind_request_context,
        record_request_timing,
        update_request_state,
    )
    from feishu_wiki_rag_agent.observability.events import (
        emit_request_summary,
        log_event,
        log_exception,
        preview_text,
    )
    from feishu_wiki_rag_agent.observability.logging import configure_logging
    from feishu_wiki_rag_agent.schemas import IncomingMessage
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from agent import invoke_agent
    from config import Settings, get_settings
    from channel.feishu.feishu_client import FeishuClient
    from observability.context import bind_log_context, bind_request_context, record_request_timing, update_request_state
    from observability.events import emit_request_summary, log_event, log_exception, preview_text
    from observability.logging import configure_logging
    from schemas import IncomingMessage

LARK_SDK_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None
lark = None

logging.getLogger("Lark").setLevel(logging.WARNING)


def _ensure_lark_imported():
    """Import `lark_oapi` lazily."""
    global lark
    if lark is None:
        import lark_oapi as imported_lark

        lark = imported_lark
    return lark


@dataclass
class MessageDeduper:
    """Simple TTL-based deduper for Feishu message ids."""

    ttl_seconds: int = 60 * 60
    _seen: dict[str, float] = field(default_factory=dict)

    def should_process(self, message_id: str, now: float | None = None) -> bool:
        """Return whether the message should be processed."""
        current = now if now is not None else time.time()
        self._seen = {key: ts for key, ts in self._seen.items() if current - ts < self.ttl_seconds}
        if message_id in self._seen:
            return False
        self._seen[message_id] = current
        return True


class FeishuChannel:
    """Standalone websocket channel that feeds incoming messages into Deep Agents."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: FeishuClient | None = None,
        agent_runner: Callable[[str, str], str] | None = None,
        deduper: MessageDeduper | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or FeishuClient(self.settings)
        self.agent_runner = agent_runner or (lambda text, thread_id: invoke_agent(text, settings=self.settings, thread_id=thread_id))
        self.deduper = deduper or MessageDeduper()
        self.bot_open_id: str | None = None

    def run(self) -> None:
        """Start the websocket client and listen for Feishu messages."""
        configure_logging(self.settings)
        if self.settings.feishu_event_mode != "websocket":
            msg = "This example only supports FEISHU_EVENT_MODE=websocket."
            raise RuntimeError(msg)
        if not LARK_SDK_AVAILABLE:
            msg = "lark-oapi is required for websocket mode."
            raise RuntimeError(msg)

        sdk = _ensure_lark_imported()
        self.bot_open_id = self.client.fetch_bot_open_id()
        log_event(
            "channel_connection_connecting",
            channel="feishu",
            transport="websocket",
            connection_state="connecting",
            mode=self.settings.feishu_event_mode,
        )

        def handle_message(data) -> None:
            try:
                payload = sdk.JSON.marshal(data)
            except Exception as exc:  # noqa: BLE001
                with bind_log_context(channel="feishu"):
                    log_exception(
                        "channel_payload_marshal_failed",
                        exc,
                        stage="feishu_websocket_payload_marshal",
                        channel="feishu",
                    )
                return
            self._handle_websocket_payload(payload)

        event_handler = (
            sdk.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(handle_message)
            .build()
        )
        websocket_client = sdk.ws.Client(
            self.settings.feishu_app_id,
            self.settings.feishu_app_secret,
            event_handler=event_handler,
            log_level=sdk.LogLevel.WARNING,
        )
        websocket_client.start()
        log_event(
            "channel_connection_closed",
            level=logging.WARNING,
            channel="feishu",
            transport="websocket",
            connection_state="closed",
        )

    def _handle_websocket_payload(self, payload: str) -> None:
        """Decode one websocket callback payload and keep failures structured."""
        try:
            event_dict = json.loads(payload)
        except Exception as exc:  # noqa: BLE001
            with bind_log_context(channel="feishu"):
                log_exception(
                    "channel_payload_decode_failed",
                    exc,
                    stage="feishu_websocket_payload_decode",
                    channel="feishu",
                )
            return

        event = event_dict.get("event", {}) if isinstance(event_dict, dict) else {}
        message = event.get("message", {}) if isinstance(event, dict) else {}
        message_id = str(message.get("message_id", ""))
        chat_id = str(message.get("chat_id", ""))
        thread_id = f"feishu:{chat_id}" if chat_id else ""
        request_id = f"feishu:{message_id}" if message_id else ""
        with bind_log_context(
            request_id=request_id,
            thread_id=thread_id,
            channel="feishu",
            message_id=message_id,
            chat_id=chat_id,
        ):
            try:
                self.handle_event(event)
            except Exception as exc:  # noqa: BLE001
                log_exception(
                    "channel_dispatch_failed",
                    exc,
                    stage="feishu_websocket_dispatch",
                    channel="feishu",
                    message_id=message_id,
                    chat_id=chat_id,
                )

    def handle_event(self, event: dict) -> str | None:
        """Parse and process a single Feishu event."""
        incoming = self._parse_incoming_message(event)
        if incoming is None:
            return None

        thread_id = f"feishu:{incoming.chat_id}"
        request_id = f"feishu:{incoming.message_id}"
        with bind_request_context(
            request_id=request_id,
            thread_id=thread_id,
            channel="feishu",
            message_id=incoming.message_id,
            chat_id=incoming.chat_id,
        ):
            update_request_state(
                message_type="text",
                chat_type=incoming.chat_type,
                sender_open_id=incoming.sender_open_id,
                mention_count=len(incoming.mentions),
                question_preview=preview_text(incoming.text),
            )
            log_event(
                "message_normalized",
                chat_type=incoming.chat_type,
                sender_open_id=incoming.sender_open_id,
                mention_count=len(incoming.mentions),
                text_preview=preview_text(incoming.text),
            )
            try:
                answer = self.agent_runner(incoming.text, thread_id)
                reply_started_at = perf_counter()
                self.client.reply_text(incoming.message_id, answer)
                reply_elapsed_ms = (perf_counter() - reply_started_at) * 1000
                record_request_timing("reply_ms", reply_elapsed_ms)
                update_request_state(
                    reply_channel="feishu",
                    answer_length=len(answer),
                    answer_preview=preview_text(answer),
                )
                log_event(
                    "reply_sent",
                    reply_channel="feishu",
                    answer_length=len(answer),
                    answer_preview=preview_text(answer),
                    duration_ms=round(reply_elapsed_ms, 1),
                )
                emit_request_summary(status="ok")
                return answer
            except Exception as exc:  # noqa: BLE001
                log_exception(
                    "request_failed",
                    exc,
                    stage="feishu_handle_event",
                    channel="feishu",
                    message_id=incoming.message_id,
                    chat_id=incoming.chat_id,
                    question_preview=preview_text(incoming.text),
                )
                emit_request_summary(status="error", level=logging.ERROR, stage="feishu_handle_event")
                return None

    def _parse_incoming_message(self, event: dict) -> IncomingMessage | None:
        """Convert a raw Feishu event into the local message schema."""
        message = event.get("message")
        sender = event.get("sender", {})
        if not message or message.get("message_type") != "text":
            return None

        message_id = str(message.get("message_id", ""))
        if not message_id or not self.deduper.should_process(message_id):
            return None
        chat_id = str(message.get("chat_id", ""))
        thread_id = f"feishu:{chat_id}" if chat_id else ""
        with bind_log_context(
            request_id=f"feishu:{message_id}",
            thread_id=thread_id,
            channel="feishu",
            message_id=message_id,
            chat_id=chat_id,
        ):
            log_event(
                "message_received",
                message_type=str(message.get("message_type", "")),
                chat_type=str(message.get("chat_type", "")),
            )

            content = json.loads(message.get("content", "{}"))
            text = str(content.get("text", "")).strip()
            mentions = message.get("mentions", [])
            chat_type = str(message.get("chat_type", ""))

            if chat_type == "group":
                if not mentions:
                    log_event("message_ignored", reason="group_without_mentions")
                    return None
                if not self._mentions_bot(mentions):
                    log_event("message_ignored", reason="group_not_mentioning_bot")
                    return None
                text = re.sub(r"@_user_\d+\s*", "", text).strip()

            if not text:
                log_event("message_ignored", reason="empty_text_after_normalization")
                return None

            return IncomingMessage(
                message_id=message_id,
                chat_id=chat_id,
                chat_type=chat_type,
                sender_open_id=str(sender.get("sender_id", {}).get("open_id", "")),
                text=text,
                mentions=mentions,
            )

    def _mentions_bot(self, mentions: list[dict]) -> bool:
        """Return whether the incoming group message mentioned the current bot."""
        if not self.bot_open_id:
            return True
        return any(mention.get("id", {}).get("open_id") == self.bot_open_id for mention in mentions)


def main() -> None:
    """Load env vars and run the Feishu websocket channel."""
    load_dotenv(Settings().env_path)
    settings = get_settings()
    configure_logging(settings)
    FeishuChannel(settings=settings).run()


if __name__ == "__main__":
    main()
