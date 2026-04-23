"""Standalone Feishu websocket channel for the Feishu Wiki RAG example."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

from dotenv import load_dotenv
import requests

if __package__ in {None, ""}:  # pragma: no cover - script execution fallback
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

try:
    from feishu_wiki_rag_agent.agent import invoke_agent, invoke_agent_stream
    from feishu_wiki_rag_agent.config import Settings, get_settings
    from feishu_wiki_rag_agent.channel.dispatcher import ConcurrentMessageDispatcher, DispatchContext
    from feishu_wiki_rag_agent.channel.feishu.feishu_client import FeishuClient
    from feishu_wiki_rag_agent.channel.feishu.streaming import FeishuStreamingResponder
    from feishu_wiki_rag_agent.multimodal_rag_agent.deposit_pipeline.adapters import extract_urls
    from feishu_wiki_rag_agent.multimodal_rag_agent.deposit_pipeline.models import InlineImage
    from feishu_wiki_rag_agent.multimodal_rag_agent.docreader_service.client import DocreaderService
    from feishu_wiki_rag_agent.multimodal_rag_agent.docreader_service.schemas import ParseRequest
    from feishu_wiki_rag_agent.multimodal_rag_agent.models import ImageRef, ParsedDocument
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
    from agent import invoke_agent, invoke_agent_stream
    from config import Settings, get_settings
    from channel.dispatcher import ConcurrentMessageDispatcher, DispatchContext
    from channel.feishu.feishu_client import FeishuClient
    from channel.feishu.streaming import FeishuStreamingResponder
    from multimodal_rag_agent.deposit_pipeline.adapters import extract_urls
    from multimodal_rag_agent.deposit_pipeline.models import InlineImage
    from multimodal_rag_agent.docreader_service.client import DocreaderService
    from multimodal_rag_agent.docreader_service.schemas import ParseRequest
    from multimodal_rag_agent.models import ImageRef, ParsedDocument
    from observability.context import bind_log_context, bind_request_context, record_request_timing, update_request_state
    from observability.events import emit_request_summary, log_event, log_exception, preview_text
    from observability.logging import configure_logging
    from schemas import IncomingMessage, MentionRef, ReplyContext

LARK_SDK_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None
lark = None

logging.getLogger("Lark").setLevel(logging.WARNING)
MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")
BUSY_REPLY_TEXT = "当前排队较多，请稍后再试"


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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def should_process(self, message_id: str, now: float | None = None) -> bool:
        """Return whether the message should be processed."""
        current = now if now is not None else time.time()
        with self._lock:
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
        streaming_agent_runner: Callable[[str, str, dict[str, object] | None], object] | None = None,
        deduper: MessageDeduper | None = None,
        instance_guard: SingleInstanceGuard | None = None,
        dispatcher: ConcurrentMessageDispatcher | None = None,
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
        self.streaming_agent_runner = streaming_agent_runner or (
            lambda text, thread_id, message_context=None: invoke_agent_stream(
                text,
                settings=self.settings,
                thread_id=thread_id,
                message_context=message_context,
            )
        )
        self._docreader_local = threading.local()
        self.deduper = deduper or MessageDeduper()
        self.instance_guard = instance_guard or SingleInstanceGuard(
            self.settings.rag_data_dir / "locks" / "feishu_channel.lock"
        )
        self.dispatcher = dispatcher or self._build_dispatcher()
        self.bot_open_id: str | None = None
        self._startup_cutoff_ms: int = 0  # connection-start cutoff in epoch ms

    def _build_dispatcher(self) -> ConcurrentMessageDispatcher | None:
        if not self.settings.bot_concurrency_enabled:
            return None
        return ConcurrentMessageDispatcher(
            max_workers=self.settings.bot_concurrency_workers,
            queue_size=self.settings.bot_concurrency_queue_size,
            per_thread_serial=self.settings.bot_concurrency_per_thread_serial,
            thread_name_prefix="feishu-message-dispatcher",
        )

    def shutdown(self) -> None:
        """Release channel-owned worker resources."""
        if self.dispatcher is not None:
            self.dispatcher.shutdown(wait=True)

    def _get_docreader(self) -> DocreaderService:
        service = getattr(self._docreader_local, "service", None)
        if service is None:
            service = DocreaderService()
            self._docreader_local.service = service
        return service

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
            # Record the connection-start cutoff: messages created before this
            # timestamp (based on message.create_time) will be silently ignored.
            self._startup_cutoff_ms = int(time.time() * 1000)
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
        context_fields = {
            "request_id": request_id,
            "thread_id": thread_id,
            "channel": "feishu",
            "message_id": message_id,
            "chat_id": chat_id,
        }
        if self.settings.bot_concurrency_enabled and self.dispatcher is not None:
            self._submit_event(event, context_fields)
            return

        with bind_log_context(**context_fields):
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

    def _submit_event(self, event: dict, context_fields: dict[str, object]) -> None:
        thread_id = str(context_fields.get("thread_id", "") or "feishu:unknown")
        message_id = str(context_fields.get("message_id", "") or "")
        chat_id = str(context_fields.get("chat_id", "") or "")
        timing: dict[str, float] = {}

        def _on_started(context: DispatchContext) -> None:
            timing["queue_ms"] = context.queue_ms

        def _run_event() -> str | None:
            with bind_log_context(**context_fields):
                try:
                    return self.handle_event(event, dispatch_queue_ms=timing.get("queue_ms"))
                except Exception as exc:  # noqa: BLE001
                    log_exception(
                        "channel_dispatch_failed",
                        exc,
                        stage="feishu_websocket_dispatch",
                        channel="feishu",
                        message_id=message_id,
                        chat_id=chat_id,
                    )
                    return None

        self.dispatcher.submit(
            thread_id,
            _run_event,
            on_started=_on_started,
            on_rejected=lambda: self._reply_busy_if_targeted(event),
            metadata={
                "request_id": str(context_fields.get("request_id", "") or ""),
                "channel": "feishu",
                "message_id": message_id,
                "chat_id": chat_id,
                "transport": "websocket",
            },
        )

    def _reply_busy_if_targeted(self, event: dict) -> None:
        message = event.get("message", {}) if isinstance(event, dict) else {}
        if not isinstance(message, dict):
            return
        message_id = str(message.get("message_id", "")).strip()
        if not message_id or str(message.get("message_type", "")) != "text":
            return
        chat_type = str(message.get("chat_type", ""))
        mentions = message.get("mentions", [])
        if chat_type == "group":
            if not isinstance(mentions, list) or not mentions:
                return
            if not self._mentions_bot(mentions):
                return
        try:
            self.client.reply_text(message_id, BUSY_REPLY_TEXT)
            log_event(
                "reply_sent",
                reply_channel="feishu",
                reason="dispatch_rejected",
                answer_length=len(BUSY_REPLY_TEXT),
                answer_preview=BUSY_REPLY_TEXT,
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "dispatch_rejection_reply_failed",
                exc,
                stage="feishu_busy_reply",
                channel="feishu",
                message_id=message_id,
            )

    def handle_event(self, event: dict, *, dispatch_queue_ms: float | None = None) -> str | None:
        """Parse and process a single Feishu event."""
        worker_started_at = perf_counter()
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
            if dispatch_queue_ms is not None:
                record_request_timing("queue_ms", dispatch_queue_ms)
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
                agent_text, message_context = self._build_agent_turn(incoming)
                reply_started_at = perf_counter()
                if self.settings.feishu_streaming_enabled:
                    responder = FeishuStreamingResponder(
                        client=self.client,
                        source_message_id=incoming.message_id,
                        update_interval_seconds=self.settings.feishu_streaming_update_interval_ms / 1000,
                        max_chars=self.settings.feishu_streaming_max_chars,
                    )
                    answer = responder.run(self.streaming_agent_runner(agent_text, thread_id, message_context))
                else:
                    answer = self.agent_runner(agent_text, thread_id, message_context)
                    self.client.reply_text(incoming.message_id, answer)
                reply_elapsed_ms = (perf_counter() - reply_started_at) * 1000
                record_request_timing("reply_ms", reply_elapsed_ms)
                record_request_timing("worker_ms", (perf_counter() - worker_started_at) * 1000)
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
                record_request_timing("worker_ms", (perf_counter() - worker_started_at) * 1000)
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

    def _build_agent_turn(self, incoming: IncomingMessage) -> tuple[str, dict[str, object]]:
        message_context = incoming.to_message_context()
        urls = extract_urls(incoming.text)
        if not urls:
            return incoming.text, message_context

        link_contexts: list[tuple[str, str]] = []
        inline_images: list[InlineImage] = []
        source_title = ""
        for url in urls:
            parsed = self._parse_url(url)
            link_contexts.append((url, parsed.markdown_content.strip()))
            if not source_title:
                source_title = str(parsed.metadata.get("title", "")).strip()
            inline_images.extend(self._persist_inline_images(incoming.message_id, "link", url, parsed.image_refs))

        effective_question = self._strip_urls_from_text(incoming.text) or "请根据我提供的链接内容完成处理。"
        rendered = self._render_attachment_prompt(
            original_text=incoming.raw_text,
            user_question=effective_question,
            link_contexts=link_contexts,
        )
        return rendered, {
            **message_context,
            "source_title": source_title,
            "inline_images_json": json.dumps([image.to_dict() for image in inline_images], ensure_ascii=False),
        }

    def _parse_url(self, url: str) -> ParsedDocument:
        parsed = self._get_docreader().parse(ParseRequest(url=url, title=url))
        if not parsed.markdown_content.strip():
            raise RuntimeError(f"链接解析失败：{url}")
        return parsed

    def _persist_inline_images(
        self,
        message_id: str,
        source_kind: str,
        source_label: str,
        image_refs: list[ImageRef],
    ) -> list[InlineImage]:
        persisted_images: list[InlineImage] = []
        for index, image_ref in enumerate(image_refs):
            image_data = image_ref.image_data or self._download_remote_image_ref(image_ref)
            if not image_data:
                continue
            suffix = Path(image_ref.filename).suffix or ".png"
            file_name = f"{message_id}_{source_kind}_{index}{suffix}"
            save_path = self.settings.weixin_tmp_dir / file_name
            save_path.write_bytes(image_data)
            persisted_images.append(
                InlineImage(
                    placeholder="",
                    image_path=str(save_path),
                    original_ref=image_ref.original_ref,
                    order=index,
                )
            )
            log_event(
                "attachment_image_saved",
                level=logging.DEBUG,
                channel="feishu",
                source_kind=source_kind,
                source_label=source_label,
                save_path=str(save_path),
            )
        return persisted_images

    def _download_remote_image_ref(self, image_ref: ImageRef) -> bytes:
        original_ref = str(image_ref.original_ref or "").strip()
        if not original_ref.startswith(("http://", "https://")):
            return b""
        response = requests.get(original_ref, timeout=self.settings.feishu_request_timeout)
        response.raise_for_status()
        return response.content

    @staticmethod
    def _strip_urls_from_text(text: str) -> str:
        normalized = re.sub(r"https?://[^\s]+", " ", text or "")
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _render_attachment_prompt(
        *,
        original_text: str,
        user_question: str,
        link_contexts: list[tuple[str, str]],
    ) -> str:
        lines = [
            "以下是用户在本轮消息中附带的材料，请优先基于这些材料回答。",
            "如果回答主要依据以下链接或文件，请在结尾使用“来源：”并写明对应链接或文件名。",
            "",
        ]
        for url, content in link_contexts:
            lines.extend(
                [
                    "[来源类型] 链接",
                    f"[来源标识] {url}",
                    "[提取内容]",
                    content,
                    "",
                ]
            )
        lines.extend(
            [
                "[用户原始消息]",
                original_text.strip(),
                "",
                "[用户问题]",
                user_question.strip(),
            ]
        )
        return "\n".join(lines).strip()

    def _parse_incoming_message(self, event: dict) -> IncomingMessage | None:
        """Convert a raw Feishu event into the local message schema."""
        message = event.get("message")
        sender = event.get("sender", {})
        if not message or message.get("message_type") != "text":
            return None

        message_id = str(message.get("message_id", ""))
        chat_id = str(message.get("chat_id", ""))

        if self._startup_cutoff_ms and not self._is_after_startup_cutoff(message):
            with bind_log_context(
                request_id=f"feishu:{message_id}" if message_id else "",
                thread_id=f"feishu:{chat_id}" if chat_id else "",
                channel="feishu",
                message_id=message_id,
                chat_id=chat_id,
            ):
                log_event(
                    "message_ignored",
                    reason="before_startup_cutoff",
                    channel="feishu",
                    message_id=message_id,
                    chat_id=chat_id,
                )
            return None

        if not message_id or not self.deduper.should_process(message_id):
            return None
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

    def _is_after_startup_cutoff(self, message: dict) -> bool:
        """Check whether the message was created after the connection-start cutoff.

        The cutoff is recorded just before the websocket connection is established.
        Messages without a reliable ``create_time`` are logged and allowed through
        (degraded pass) rather than silently dropped.
        """
        create_time_ms = self._extract_create_time_ms(message)
        if create_time_ms is None:
            message_id = str(message.get("message_id", ""))
            chat_id = str(message.get("chat_id", ""))
            log_event(
                "startup_cutoff_degraded_pass",
                channel="feishu",
                message_id=message_id,
                chat_id=chat_id,
                reason="missing_create_time",
            )
            return True
        return create_time_ms >= self._startup_cutoff_ms

    @staticmethod
    def _extract_create_time_ms(message: dict) -> int | None:
        """Extract and normalize message.create_time to integer milliseconds."""
        raw = message.get("create_time")
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        if value < 1_000_000_000_000:
            value *= 1000
        return value

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
