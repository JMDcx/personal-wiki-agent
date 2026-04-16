"""Standalone Feishu websocket channel for the Feishu Wiki RAG example."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

from dotenv import load_dotenv

if __package__ in {None, ""}:  # pragma: no cover - script execution fallback
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

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
    from feishu_wiki_rag_agent.schemas import IncomingMessage, MentionRef, ReplyContext
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from agent import invoke_agent
    from config import Settings, get_settings
    from channel.feishu.feishu_client import FeishuClient
    from observability.context import bind_log_context, bind_request_context, record_request_timing, update_request_state
    from observability.events import emit_request_summary, log_event, log_exception, preview_text
    from observability.logging import configure_logging
    from schemas import IncomingMessage, MentionRef, ReplyContext

LARK_SDK_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None
lark = None

logging.getLogger("Lark").setLevel(logging.WARNING)
MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")


@dataclass
class SingleInstanceGuard:
    """Best-effort file lock so only one Feishu channel process consumes events."""

    lock_path: Path
    _acquired: bool = False

    def acquire(self) -> None:
        if self._acquired:
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._clear_stale_lock():
                    continue
                msg = f"Feishu channel is already running: {self.lock_path}"
                raise RuntimeError(msg)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            self._acquired = True
            return

    def release(self) -> None:
        if not self._acquired:
            return
        with suppress(FileNotFoundError):
            self.lock_path.unlink()
        self._acquired = False

    def __enter__(self) -> SingleInstanceGuard:
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()

    def _clear_stale_lock(self) -> bool:
        try:
            pid_text = self.lock_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return True
        if pid_text.isdigit() and self._pid_is_running(int(pid_text)):
            return False
        with suppress(FileNotFoundError):
            self.lock_path.unlink()
        return True

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


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
        agent_runner: Callable[[str, str, dict[str, object] | None], str] | None = None,
        deduper: MessageDeduper | None = None,
        instance_guard: SingleInstanceGuard | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or FeishuClient(self.settings)
        self.agent_runner = agent_runner or (
            lambda text, thread_id, message_context=None: invoke_agent(
                text,
                settings=self.settings,
                thread_id=thread_id,
                message_context=message_context,
            )
        )
        self.deduper = deduper or MessageDeduper()
        self.instance_guard = instance_guard or SingleInstanceGuard(
            self.settings.rag_data_dir / "locks" / "feishu_channel.lock"
        )
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

        with self.instance_guard:
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
                is_reply=bool(incoming.reply_context),
                reply_parent_id=incoming.reply_context.parent_id if incoming.reply_context else "",
                reply_root_id=incoming.reply_context.root_id if incoming.reply_context else "",
                reply_parent_role=incoming.reply_context.parent_role if incoming.reply_context else "",
                reply_parent_preview=(
                    preview_text(incoming.reply_context.parent_text_preview) if incoming.reply_context else ""
                ),
                bot_mentioned=incoming.bot_mentioned,
                mentioned_users=incoming.mentioned_users,
                question_preview=preview_text(incoming.text),
            )
            log_event(
                "message_normalized",
                chat_type=incoming.chat_type,
                sender_open_id=incoming.sender_open_id,
                mention_count=len(incoming.mentions),
                is_reply=bool(incoming.reply_context),
                reply_parent_id=incoming.reply_context.parent_id if incoming.reply_context else "",
                reply_root_id=incoming.reply_context.root_id if incoming.reply_context else "",
                reply_parent_role=incoming.reply_context.parent_role if incoming.reply_context else "",
                reply_parent_preview=(
                    preview_text(incoming.reply_context.parent_text_preview) if incoming.reply_context else ""
                ),
                bot_mentioned=incoming.bot_mentioned,
                mentioned_users=incoming.mentioned_users,
                text_preview=preview_text(incoming.text),
            )
            try:
                answer = self.agent_runner(incoming.text, thread_id, incoming.to_message_context())
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
            raw_text = str(content.get("text", "")).strip()
            text = raw_text
            mentions = message.get("mentions", [])
            chat_type = str(message.get("chat_type", ""))
            bot_mentioned = False
            mentioned_users: list[str] = []
            mention_refs = self._build_mention_refs(mentions)
            reply_context = self._extract_reply_context(message)

            if chat_type == "group":
                if not mentions:
                    log_event("message_ignored", reason="group_without_mentions")
                    return None
                if not self._mentions_bot(mentions):
                    log_event("message_ignored", reason="group_not_mentioning_bot")
                    return None
                text, bot_mentioned, mentioned_users = self._normalize_group_message_text(raw_text, mentions)

            if not text:
                log_event("message_ignored", reason="empty_text_after_normalization")
                return None

            return IncomingMessage(
                message_id=message_id,
                chat_id=chat_id,
                chat_type=chat_type,
                sender_open_id=str(sender.get("sender_id", {}).get("open_id", "")),
                raw_text=raw_text,
                text=text,
                mentions=mentions,
                mention_refs=mention_refs,
                reply_context=reply_context,
                bot_mentioned=bot_mentioned,
                mentioned_users=mentioned_users,
            )

    def _mentions_bot(self, mentions: list[dict]) -> bool:
        """Return whether the incoming group message mentioned the current bot."""
        if not self.bot_open_id:
            return True
        return any(mention.get("id", {}).get("open_id") == self.bot_open_id for mention in mentions)

    def _normalize_group_message_text(self, raw_text: str, mentions: list[dict]) -> tuple[str, bool, list[str]]:
        """Remove the bot mention while preserving other mentioned users as readable text."""
        mention_index = 0
        bot_mentioned = False
        mentioned_users: list[str] = []

        def replace(_match: re.Match[str]) -> str:
            nonlocal mention_index, bot_mentioned
            mention = mentions[mention_index] if mention_index < len(mentions) else None
            mention_index += 1
            if mention is None:
                return ""
            if self._is_bot_mention(mention):
                bot_mentioned = True
                return ""
            display_name = self._mention_display_name(mention)
            if display_name not in mentioned_users:
                mentioned_users.append(display_name)
            return f"@{display_name}"

        normalized = MENTION_PLACEHOLDER_RE.sub(replace, raw_text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = normalized.lstrip("，, ")
        return normalized.strip(), bot_mentioned, mentioned_users

    def _is_bot_mention(self, mention: dict) -> bool:
        return bool(self.bot_open_id) and mention.get("id", {}).get("open_id") == self.bot_open_id

    def _build_mention_refs(self, mentions: list[dict]) -> list[MentionRef]:
        """Convert raw mention payloads into stable, transport-agnostic mention refs."""
        refs: list[MentionRef] = []
        for mention in mentions:
            refs.append(
                MentionRef(
                    display_name=self._mention_display_name(mention),
                    open_id=str(mention.get("id", {}).get("open_id", "")).strip(),
                    is_bot=self._is_bot_mention(mention),
                )
            )
        return refs

    def _extract_reply_context(self, message: dict) -> ReplyContext | None:
        """Build best-effort reply context from the current Feishu message."""
        parent_id = str(message.get("parent_id", "")).strip()
        if not parent_id:
            return None
        root_id = str(message.get("root_id", "")).strip() or parent_id
        parent_text_preview = ""
        parent_message_type = ""
        parent_role = ""
        parent_sender_name = ""
        try:
            parent_message = self.client.get_message(parent_id)
            parent_text_preview, parent_message_type = self._extract_message_preview(parent_message)
            parent_role = self._infer_parent_role(parent_message)
            parent_sender_name = self._extract_sender_name(parent_message)
        except Exception:  # noqa: BLE001
            pass
        return ReplyContext(
            parent_id=parent_id,
            root_id=root_id,
            parent_text_preview=parent_text_preview,
            parent_message_type=parent_message_type,
            parent_role=parent_role,
            parent_sender_name=parent_sender_name,
        )

    def _extract_message_preview(self, message: dict) -> tuple[str, str]:
        """Extract a compact preview from a Feishu message payload."""
        message_type = str(
            message.get("message_type")
            or message.get("msg_type")
            or message.get("type")
            or ""
        ).strip()
        content = message.get("content")
        if content is None and isinstance(message.get("body"), dict):
            content = message["body"].get("content")
        preview = self._extract_text_content(content)
        if not preview and message_type and message_type != "text":
            preview = f"[{message_type} message]"
        return preview_text(preview), message_type

    @staticmethod
    def _extract_text_content(content: object) -> str:
        """Extract plain text from text-message content returned by Feishu."""
        if isinstance(content, dict):
            text = content.get("text")
            return str(text).strip() if text else ""
        if isinstance(content, str):
            stripped = content.strip()
            if not stripped:
                return ""
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped.removeprefix("text:").strip()
            if isinstance(parsed, dict):
                text = parsed.get("text")
                return str(text).strip() if text else ""
            return stripped
        return ""

    def _infer_parent_role(self, message: dict) -> str:
        """Infer whether the replied-to message came from the bot or a user."""
        sender_open_id = self._extract_sender_open_id(message)
        sender_type = str(self._extract_sender_field(message, "sender_type")).strip().lower()
        if self.bot_open_id and sender_open_id and sender_open_id == self.bot_open_id:
            return "assistant"
        if sender_type in {"app", "bot"}:
            return "assistant"
        return "user"

    @staticmethod
    def _extract_sender_open_id(message: dict) -> str:
        sender = message.get("sender")
        if isinstance(sender, dict):
            sender_id = sender.get("sender_id")
            if isinstance(sender_id, dict):
                open_id = str(sender_id.get("open_id", "")).strip()
                if open_id:
                    return open_id
            open_id = str(sender.get("open_id", "")).strip()
            if open_id:
                return open_id
        return str(message.get("sender_open_id", "")).strip()

    @staticmethod
    def _extract_sender_name(message: dict) -> str:
        sender = message.get("sender")
        if isinstance(sender, dict):
            for key in ("sender_name", "name", "display_name"):
                value = str(sender.get(key, "")).strip()
                if value:
                    return value
            sender_id = sender.get("sender_id")
            if isinstance(sender_id, dict):
                for key in ("name", "display_name", "open_id"):
                    value = str(sender_id.get(key, "")).strip()
                    if value:
                        return value
        for key in ("sender_name", "sender_display_name"):
            value = str(message.get(key, "")).strip()
            if value:
                return value
        return ""

    @staticmethod
    def _extract_sender_field(message: dict, field_name: str) -> object:
        sender = message.get("sender")
        if isinstance(sender, dict) and field_name in sender:
            return sender.get(field_name)
        return message.get(field_name)

    @staticmethod
    def _mention_display_name(mention: dict) -> str:
        for key in ("name", "display_name", "key"):
            value = str(mention.get(key, "")).strip()
            if value:
                return value
        open_id = str(mention.get("id", {}).get("open_id", "")).strip()
        return open_id or "mentioned_user"


def main() -> None:
    """Load env vars and run the Feishu websocket channel."""
    load_dotenv(Settings().env_path)
    settings = get_settings()
    configure_logging(settings)
    FeishuChannel(settings=settings).run()


if __name__ == "__main__":
    main()
