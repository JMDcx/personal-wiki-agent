"""Standalone Weixin long-poll channel for the Feishu Wiki RAG example."""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import requests
from dotenv import load_dotenv
from PIL import Image

try:
    from feishu_wiki_rag_agent.agent import invoke_agent
    from feishu_wiki_rag_agent.channel.dispatcher import ConcurrentMessageDispatcher, DispatchContext
    from feishu_wiki_rag_agent.channel.weixin.weixin_api import (
        CDN_BASE_URL,
        DEFAULT_BASE_URL,
        WeixinApi,
    )
    from feishu_wiki_rag_agent.channel.weixin.weixin_api import download_media_from_cdn
    from feishu_wiki_rag_agent.channel.weixin.weixin_message import DownloadedAttachment, WeixinMessage
    from feishu_wiki_rag_agent.config import Settings, get_settings
    from feishu_wiki_rag_agent.observability.context import (
        bind_log_context,
        bind_request_context,
        record_request_timing,
    )
    from feishu_wiki_rag_agent.observability.events import emit_request_summary, log_event, log_exception, preview_text
    from feishu_wiki_rag_agent.observability.logging import configure_logging
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from agent import invoke_agent
    from channel.dispatcher import ConcurrentMessageDispatcher, DispatchContext
    from channel.weixin.weixin_api import CDN_BASE_URL, DEFAULT_BASE_URL, WeixinApi, download_media_from_cdn
    from channel.weixin.weixin_message import DownloadedAttachment, WeixinMessage
    from config import Settings, get_settings
    from observability.context import bind_log_context, bind_request_context, record_request_timing
    from observability.events import emit_request_summary, log_event, log_exception, preview_text
    from observability.logging import configure_logging
from multimodal_rag_agent.docreader_service.client import DocreaderService
from multimodal_rag_agent.deposit_pipeline.models import InlineImage
from multimodal_rag_agent.docreader_service.schemas import ParseRequest
from multimodal_rag_agent.models import ImageRef, ParsedDocument

QR_LOGIN_TIMEOUT_S = 480
QR_MAX_REFRESHES = 10
SESSION_EXPIRED_ERRCODE = -14
TEXT_CHUNK_LIMIT = 4000
MAX_CONSECUTIVE_FAILURES = 3
RETRY_DELAY = 2
BACKOFF_DELAY = 30
BUSY_REPLY_TEXT = "当前排队较多，请稍后再试"
THINKING_REPLY_TEXT = "thinking..."


@dataclass
class MessageDeduper:
    """Simple TTL-based deduper for inbound Weixin message ids."""

    ttl_seconds: int = 60 * 60
    _seen: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def should_process(self, message_id: str, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        with self._lock:
            self._seen = {key: ts for key, ts in self._seen.items() if current - ts < self.ttl_seconds}
            if message_id in self._seen:
                return False
            self._seen[message_id] = current
            return True


class UserFacingAttachmentError(RuntimeError):
    """Raised when a user-provided attachment cannot be processed safely."""


class WeixinChannel:
    """Standalone Weixin channel that adapts attachments into the existing Deep Agent runtime."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api: WeixinApi | None = None,
        agent_runner: Callable[[str, str, list[str], dict[str, object] | None], str] | None = None,
        deduper: MessageDeduper | None = None,
        docreader: DocreaderService | None = None,
        media_downloader: Callable[..., str] | None = None,
        dispatcher: ConcurrentMessageDispatcher | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.api = api
        self.agent_runner = agent_runner or (
            lambda question, thread_id, images, message_context=None: invoke_agent(
                question,
                settings=self.settings,
                thread_id=thread_id,
                images=images,
                message_context=message_context,
            )
        )
        self.deduper = deduper or MessageDeduper()
        self._provided_docreader = docreader
        self._docreader_local = threading.local()
        self.media_downloader = media_downloader or download_media_from_cdn
        self.dispatcher = dispatcher or self._build_dispatcher()
        self._cursor = ""
        self._context_tokens: dict[str, str] = {}
        self._context_tokens_lock = threading.Lock()

    def _build_dispatcher(self) -> ConcurrentMessageDispatcher | None:
        if not self.settings.bot_concurrency_enabled:
            return None
        return ConcurrentMessageDispatcher(
            max_workers=self.settings.bot_concurrency_workers,
            queue_size=self.settings.bot_concurrency_queue_size,
            per_thread_serial=self.settings.bot_concurrency_per_thread_serial,
            thread_name_prefix="weixin-message-dispatcher",
        )

    def shutdown(self) -> None:
        """Release channel-owned worker resources."""
        if self.dispatcher is not None:
            self.dispatcher.shutdown(wait=True)

    def _get_docreader(self) -> DocreaderService:
        if self._provided_docreader is not None:
            return self._provided_docreader
        service = getattr(self._docreader_local, "service", None)
        if service is None:
            service = DocreaderService()
            self._docreader_local.service = service
        return service

    def run(self) -> None:
        """Authenticate if needed and start the long-poll loop."""
        configure_logging(self.settings)
        self.api = self.api or self._build_authenticated_api()
        self._prime_cursor()
        self._poll_loop()

    def handle_raw_message(self, raw_msg: dict, *, dispatch_queue_ms: float | None = None) -> str | None:
        """Parse, adapt, answer, and reply to a single inbound Weixin message."""
        worker_started_at = perf_counter()
        if str(raw_msg.get("message_type", "")) != "1":
            return None

        message_id = str(raw_msg.get("message_id", raw_msg.get("seq", "")))
        if not message_id or not self.deduper.should_process(message_id):
            return None

        from_user_id = str(raw_msg.get("from_user_id", ""))
        context_token = str(raw_msg.get("context_token", ""))
        if from_user_id and context_token:
            with self._context_tokens_lock:
                self._context_tokens[from_user_id] = context_token

        if self.api is None:
            self.api = self._build_api(token=self.settings.weixin_token)
        thread_id = f"weixin:{from_user_id}"
        with bind_request_context(
            request_id=f"weixin:{message_id}",
            thread_id=thread_id,
            channel="weixin",
            message_id=message_id,
            from_user_id=from_user_id,
        ):
            if dispatch_queue_ms is not None:
                record_request_timing("queue_ms", dispatch_queue_ms)
            log_event(
                "message_received",
                message_type=str(raw_msg.get("message_type", "")),
                has_context_token=bool(context_token),
            )

            message = WeixinMessage(
                raw_msg,
                tmp_dir=self.settings.weixin_tmp_dir,
                cdn_base_url=self.api.cdn_base_url,
                media_downloader=self.media_downloader,
            )

            try:
                prompt, images, inline_images, source_title = self._build_agent_turn(message)
            except UserFacingAttachmentError as exc:
                record_request_timing("worker_ms", (perf_counter() - worker_started_at) * 1000)
                log_exception("attachment_adaptation_failed", exc, stage="weixin_build_agent_turn")
                self._reply_text(from_user_id, context_token, str(exc))
                emit_request_summary(status="error", level=logging.WARNING, stage="weixin_build_agent_turn")
                return str(exc)

            log_event(
                "message_normalized",
                url_count=len(message.urls),
                attachment_count=len(message.attachments),
                image_count=len(images),
                prompt_preview=preview_text(prompt),
            )
            try:
                answer = self.agent_runner(
                    prompt,
                    thread_id,
                    images,
                    {
                        "chat_type": "direct",
                        "raw_text": message.text,
                        "normalized_text": message.stripped_text_without_urls() or message.text,
                        "inline_images_json": json.dumps(
                            [image.to_dict() for image in inline_images],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "source_title": source_title,
                    },
                )
                self._reply_text(from_user_id, context_token, answer)
                record_request_timing("worker_ms", (perf_counter() - worker_started_at) * 1000)
                log_event(
                    "reply_sent",
                    reply_channel="weixin",
                    answer_length=len(answer),
                    answer_preview=preview_text(answer),
                )
                emit_request_summary(status="ok")
                return answer
            except Exception as exc:  # noqa: BLE001
                record_request_timing("worker_ms", (perf_counter() - worker_started_at) * 1000)
                log_exception(
                    "request_failed",
                    exc,
                    stage="weixin_handle_raw_message",
                    channel="weixin",
                    message_id=message_id,
                    from_user_id=from_user_id,
                )
                emit_request_summary(status="error", level=logging.ERROR, stage="weixin_handle_raw_message")
                raise

    def _dispatch_raw_message(self, raw_msg: dict) -> None:
        """Submit one raw Weixin message to the worker pool when enabled."""
        if str(raw_msg.get("message_type", "")) != "1":
            return

        message_id = str(raw_msg.get("message_id", raw_msg.get("seq", "")))
        from_user_id = str(raw_msg.get("from_user_id", ""))
        context_token = str(raw_msg.get("context_token", ""))
        thread_id = f"weixin:{from_user_id}" if from_user_id else "weixin:unknown"
        context_fields = {
            "request_id": f"weixin:{message_id}" if message_id else "",
            "thread_id": thread_id,
            "channel": "weixin",
            "message_id": message_id,
            "from_user_id": from_user_id,
        }

        if not self.settings.bot_concurrency_enabled or self.dispatcher is None:
            with bind_log_context(**context_fields):
                self.handle_raw_message(raw_msg)
            return

        timing: dict[str, float] = {}
        thinking_reply_ready = threading.Event()

        def _on_started(context: DispatchContext) -> None:
            timing["queue_ms"] = context.queue_ms

        def _run_message() -> str | None:
            thinking_reply_ready.wait()
            with bind_log_context(**context_fields):
                return self.handle_raw_message(raw_msg, dispatch_queue_ms=timing.get("queue_ms"))

        result = self.dispatcher.submit(
            thread_id,
            _run_message,
            on_started=_on_started,
            on_rejected=lambda: self._reply_text(from_user_id, context_token, BUSY_REPLY_TEXT),
            metadata={
                "request_id": str(context_fields.get("request_id", "") or ""),
                "channel": "weixin",
                "message_id": message_id,
                "from_user_id": from_user_id,
                "transport": "long_poll",
            },
        )
        if result.accepted:
            self._reply_text(from_user_id, context_token, THINKING_REPLY_TEXT)
        thinking_reply_ready.set()

    def _prime_cursor(self) -> None:
        """Drain the startup backlog to establish a cursor baseline before normal polling."""
        assert self.api is not None
        try:
            response = self.api.get_updates("")
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "channel_startup_priming_failed",
                exc,
                channel="weixin",
                stage="weixin_prime_cursor",
            )
            raise
        ret = int(response.get("ret", 0) or 0)
        errcode = int(response.get("errcode", 0) or 0)
        if ret != 0 or errcode != 0:
            log_event(
                "channel_startup_priming_failed",
                level=logging.ERROR,
                channel="weixin",
                ret=ret,
                errcode=errcode,
            )
            msg = f"Weixin startup priming returned error: ret={ret}, errcode={errcode}"
            raise RuntimeError(msg)
        primed_message_count = len(response.get("msgs", []))
        new_cursor = str(response.get("get_updates_buf", ""))
        if new_cursor:
            self._cursor = new_cursor
        log_event(
            "channel_startup_backlog_drained",
            channel="weixin",
            primed_message_count=primed_message_count,
            cursor_present=bool(new_cursor),
        )

    def _poll_loop(self) -> None:
        assert self.api is not None
        failures = 0
        while True:
            try:
                response = self.api.get_updates(self._cursor)
                ret = int(response.get("ret", 0) or 0)
                errcode = int(response.get("errcode", 0) or 0)
                if ret != 0 or errcode != 0:
                    if errcode == SESSION_EXPIRED_ERRCODE or ret == SESSION_EXPIRED_ERRCODE:
                        log_event(
                            "channel_auth_refresh_requested",
                            level=logging.WARNING,
                            channel="weixin",
                            reason="session_expired",
                            ret=ret,
                            errcode=errcode,
                        )
                        self.api = self._build_authenticated_api(force_relogin=True)
                        self._cursor = ""
                        failures = 0
                        continue
                    failures += 1
                    log_event(
                        "channel_poll_error",
                        level=logging.ERROR,
                        channel="weixin",
                        ret=ret,
                        errcode=errcode,
                        failure_count=failures,
                        cursor_present=bool(self._cursor),
                    )
                    time.sleep(BACKOFF_DELAY if failures >= MAX_CONSECUTIVE_FAILURES else RETRY_DELAY)
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        failures = 0
                    continue

                failures = 0
                new_cursor = str(response.get("get_updates_buf", ""))
                if new_cursor:
                    self._cursor = new_cursor
                for raw_msg in response.get("msgs", []):
                    message_id = str(raw_msg.get("message_id", raw_msg.get("seq", "")))
                    from_user_id = str(raw_msg.get("from_user_id", ""))
                    context_token = str(raw_msg.get("context_token", ""))
                    try:
                        with bind_log_context(
                            request_id=f"weixin:{message_id}" if message_id else "",
                            thread_id=f"weixin:{from_user_id}" if from_user_id else "",
                            channel="weixin",
                            message_id=message_id,
                            from_user_id=from_user_id,
                        ):
                            self._dispatch_raw_message(raw_msg)
                    except Exception as exc:  # noqa: BLE001
                        log_exception(
                            "request_failed",
                            exc,
                            stage="weixin_poll_loop_message",
                            channel="weixin",
                            message_id=message_id,
                            from_user_id=from_user_id,
                            has_context_token=bool(context_token),
                        )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log_exception(
                    "channel_poll_failed",
                    exc,
                    stage="weixin_poll_loop",
                    channel="weixin",
                    failure_count=failures,
                    cursor_present=bool(self._cursor),
                )
                time.sleep(BACKOFF_DELAY if failures >= MAX_CONSECUTIVE_FAILURES else RETRY_DELAY)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    failures = 0

    def _build_agent_turn(self, message: WeixinMessage) -> tuple[str, list[str], list[InlineImage], str]:
        images = list(message.image_paths)
        inline_images: list[InlineImage] = []
        text_without_urls = message.stripped_text_without_urls()
        source_title = ""

        link_contexts: list[tuple[str, str]] = []
        file_contexts: list[tuple[str, str]] = []

        for url in message.urls:
            parsed = self._parse_url(url)
            link_contexts.append((url, parsed.markdown_content.strip()))
            if not source_title:
                source_title = str(parsed.metadata.get("title", "")).strip()
            persisted = self._persist_inline_images(message.message_id, "link", url, parsed.image_refs)
            inline_images.extend(persisted)
            images.extend(image.image_path for image in persisted if image.image_path)

        for attachment in message.file_attachments:
            parsed = self._parse_file(attachment)
            file_contexts.append((attachment.display_name, parsed.markdown_content.strip()))
            persisted = self._persist_inline_images(message.message_id, "file", attachment.display_name, parsed.image_refs)
            inline_images.extend(persisted)
            images.extend(image.image_path for image in persisted if image.image_path)

        if link_contexts or file_contexts:
            effective_question = text_without_urls or self._default_attachment_question(
                has_links=bool(link_contexts),
                has_files=bool(file_contexts),
            )
            return self._render_attachment_prompt(
                original_text=message.text,
                user_question=effective_question,
                link_contexts=link_contexts,
                file_contexts=file_contexts,
            ), images, inline_images, source_title

        if images and not text_without_urls:
            return "请分析这张图片的内容，并尽量直接回答用户最可能想问的问题。", images, inline_images, source_title
        if text_without_urls:
            return text_without_urls, images, inline_images, source_title

        raise UserFacingAttachmentError("未识别到可处理的文本、链接、图片或文件内容。")

    def _parse_url(self, url: str) -> ParsedDocument:
        try:
            parsed = self._get_docreader().parse(ParseRequest(url=url, title=url))
        except Exception as exc:  # noqa: BLE001
            raise UserFacingAttachmentError(f"链接解析失败：{url}") from exc
        if not parsed.markdown_content.strip():
            raise UserFacingAttachmentError(f"链接解析失败：{url}")
        return parsed

    def _parse_file(self, attachment: DownloadedAttachment) -> ParsedDocument:
        try:
            file_content = Path(attachment.path).read_bytes()
            parsed = self._get_docreader().parse(
                ParseRequest(
                    file_name=attachment.display_name,
                    file_type=Path(attachment.display_name).suffix.lstrip("."),
                    file_content=file_content,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise UserFacingAttachmentError(f"文件解析失败或暂不支持该格式：{attachment.display_name}") from exc
        if not parsed.markdown_content.strip():
            raise UserFacingAttachmentError(f"文件解析失败或暂不支持该格式：{attachment.display_name}")
        return parsed

    def _persist_image_refs(
        self,
        message_id: str,
        source_kind: str,
        source_label: str,
        image_refs: list[ImageRef],
    ) -> list[str]:
        return [
            image.image_path
            for image in self._persist_inline_images(message_id, source_kind, source_label, image_refs)
            if image.image_path
        ]

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
            if self._should_filter_noise_image(save_path, image_ref):
                log_event(
                    "attachment_image_filtered",
                    channel="weixin",
                    source_kind=source_kind,
                    source_label=source_label,
                    save_path=str(save_path),
                    original_ref=image_ref.original_ref,
                )
                continue
            log_event(
                "attachment_image_saved",
                level=logging.DEBUG,
                channel="weixin",
                source_kind=source_kind,
                source_label=source_label,
                save_path=str(save_path),
            )
            persisted_images.append(
                InlineImage(
                    placeholder="",
                    image_path=str(save_path),
                    original_ref=image_ref.original_ref.strip(),
                    order=len(persisted_images),
                )
            )
        return persisted_images

    def _should_filter_noise_image(self, image_path: Path, image_ref: ImageRef) -> bool:
        suffix = image_path.suffix.lower()
        original_ref = image_ref.original_ref.lower()
        filename = image_path.name.lower()
        if suffix == ".gif":
            return True
        if "logo" in original_ref or "logo" in filename:
            return True
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                if width <= 80 or height <= 80:
                    return True
                if width == height and max(width, height) <= 600:
                    grayscale = image.convert("L")
                    colors = grayscale.getcolors(maxcolors=32)
                    if colors and len(colors) <= 8:
                        return True
                    if self._looks_like_qr(grayscale):
                        return True
        except Exception:
            return False
        return False

    @staticmethod
    def _looks_like_qr(image: Image.Image) -> bool:
        width, height = image.size
        if abs(width - height) > 8 or width < 120 or height < 120:
            return False
        sample = image.resize((32, 32))
        colors = sample.getcolors(maxcolors=8)
        if not colors:
            return False
        dark_pixels = 0
        total_pixels = 32 * 32
        for count, value in colors:
            if value < 100:
                dark_pixels += count
        ratio = dark_pixels / total_pixels
        return 0.2 <= ratio <= 0.8

    def _download_remote_image_ref(self, image_ref: ImageRef) -> bytes:
        original_ref = image_ref.original_ref.strip()
        if not original_ref.startswith(("http://", "https://")):
            return b""
        try:
            response = requests.get(original_ref, timeout=self.settings.weixin_request_timeout)
            response.raise_for_status()
        except Exception:
            log_exception(
                "attachment_image_download_failed",
                RuntimeError(f"Failed to download image ref: {preview_text(original_ref)}"),
                channel="weixin",
                source_kind="link",
                source_label=preview_text(original_ref),
            )
            return b""
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("image/") and not Path(image_ref.filename).suffix:
            suffix = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "image/bmp": ".bmp",
            }.get(content_type.lower(), "")
            if suffix:
                image_ref.filename = f"{image_ref.filename}{suffix}" if image_ref.filename else f"image{suffix}"
        return response.content

    @staticmethod
    def _default_attachment_question(*, has_links: bool, has_files: bool) -> str:
        if has_links and has_files:
            return "请结合这些链接和文件，总结主要内容并回答用户。"
        if has_links:
            return "请总结这个链接的主要内容并回答用户。"
        return "请总结这个文件的主要内容并回答用户。"

    @staticmethod
    def _render_attachment_prompt(
        *,
        original_text: str,
        user_question: str,
        link_contexts: list[tuple[str, str]],
        file_contexts: list[tuple[str, str]],
    ) -> str:
        lines = [
            "以下是用户在本轮消息中附带的材料，请优先基于这些材料回答。",
            "如果回答主要依据以下链接或文件，请在结尾使用“来源：”并写明对应链接或文件名。",
        ]
        for url, markdown in link_contexts:
            lines.extend(
                [
                    "",
                    "[来源类型] 链接",
                    f"[来源标识] {url}",
                    "[提取内容]",
                    markdown,
                ]
            )
        for file_name, markdown in file_contexts:
            lines.extend(
                [
                    "",
                    "[来源类型] 文件",
                    f"[来源标识] {file_name}",
                    "[提取内容]",
                    markdown,
                ]
            )
        lines.extend(
            [
                "",
                "[用户原始消息]",
                original_text or user_question,
                "",
                "[用户问题]",
                user_question,
            ]
        )
        return "\n".join(lines).strip()

    def _reply_text(self, to_user_id: str, context_token: str, text: str) -> None:
        if not context_token and to_user_id:
            with self._context_tokens_lock:
                context_token = self._context_tokens.get(to_user_id, "")
        if not to_user_id or not context_token:
            log_event(
                "reply_skipped",
                level=logging.WARNING,
                reply_channel="weixin",
                reason="missing_reply_target",
                has_to_user_id=bool(to_user_id),
                has_context_token=bool(context_token),
            )
            return
        assert self.api is not None
        for chunk in self._split_text(text, TEXT_CHUNK_LIMIT):
            self.api.send_text(to_user_id=to_user_id, text=chunk, context_token=context_token)

    @staticmethod
    def _split_text(text: str, limit: int) -> list[str]:
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break
            cut = remaining.rfind("\n\n", 0, limit)
            if cut <= 0:
                cut = remaining.rfind("\n", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(remaining[:cut])
            remaining = remaining[cut:].lstrip("\n")
        return chunks

    def _build_api(self, *, token: str = "", base_url: str | None = None) -> WeixinApi:
        return WeixinApi(
            base_url=base_url or self.settings.weixin_base_url or DEFAULT_BASE_URL,
            token=token,
            cdn_base_url=self.settings.weixin_cdn_base_url or CDN_BASE_URL,
            request_timeout=self.settings.weixin_request_timeout,
            long_poll_timeout=self.settings.weixin_long_poll_timeout,
        )

    def _build_authenticated_api(self, *, force_relogin: bool = False) -> WeixinApi:
        token = self.settings.weixin_token
        base_url = self.settings.weixin_base_url
        credentials = {} if force_relogin else self._load_credentials()
        if credentials and not token:
            token = str(credentials.get("token", ""))
            base_url = str(credentials.get("base_url", "") or base_url)
        if not token:
            login_result = self._qr_login(base_url or DEFAULT_BASE_URL)
            token = login_result["token"]
            base_url = login_result.get("base_url", base_url or DEFAULT_BASE_URL)
        return self._build_api(token=token, base_url=base_url)

    def _load_credentials(self) -> dict:
        try:
            if self.settings.weixin_credentials_path.exists():
                return json.loads(self.settings.weixin_credentials_path.read_text())
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "credentials_load_failed",
                exc,
                channel="weixin",
                stage="weixin_load_credentials",
                credentials_path=str(self.settings.weixin_credentials_path),
            )
        return {}

    def _save_credentials(self, payload: dict) -> None:
        self.settings.weixin_credentials_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.weixin_credentials_path.write_text(json.dumps(payload, indent=2))
        try:
            self.settings.weixin_credentials_path.chmod(0o600)
        except Exception:
            pass

    @staticmethod
    def _print_qr(qrcode_url: str) -> None:
        try:
            if importlib.util.find_spec("qrcode") is not None:
                import io
                import qrcode as qr_lib

                qr = qr_lib.QRCode(error_correction=qr_lib.constants.ERROR_CORRECT_L, box_size=1, border=1)
                qr.add_data(qrcode_url)
                qr.make(fit=True)
                buffer = io.StringIO()
                qr.print_ascii(out=buffer, invert=True)
                print(buffer.getvalue())
                return
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "terminal_render_failed",
                exc,
                channel="weixin",
                stage="weixin_print_qr",
            )
        print(f"请扫描微信二维码登录：{qrcode_url}")

    def _qr_login(self, base_url: str) -> dict:
        temp_api = self._build_api(base_url=base_url)
        qr_response = temp_api.fetch_qr_code()
        qrcode = str(qr_response.get("qrcode", "")).strip()
        qrcode_url = str(qr_response.get("qrcode_img_content", "")).strip()
        if not qrcode:
            msg = "No QR code returned from Weixin API"
            raise RuntimeError(msg)
        self._print_qr(qrcode_url or qrcode)
        print("等待扫码并在手机上确认...")

        deadline = time.time() + QR_LOGIN_TIMEOUT_S
        refresh_count = 0
        while time.time() < deadline:
            status_response = temp_api.poll_qr_status(qrcode)
            status = str(status_response.get("status", "wait"))
            if status == "wait":
                time.sleep(1)
                continue
            if status == "scaned":
                print("已扫码，请在手机上确认...")
                time.sleep(1)
                continue
            if status == "expired":
                refresh_count += 1
                if refresh_count >= QR_MAX_REFRESHES:
                    raise RuntimeError("Weixin QR code expired too many times")
                qr_response = temp_api.fetch_qr_code()
                qrcode = str(qr_response.get("qrcode", "")).strip()
                qrcode_url = str(qr_response.get("qrcode_img_content", "")).strip()
                self._print_qr(qrcode_url or qrcode)
                continue
            if status == "confirmed":
                bot_token = str(status_response.get("bot_token", "")).strip()
                result_base_url = str(status_response.get("baseurl", "")).strip() or base_url
                if not bot_token:
                    raise RuntimeError("Weixin login succeeded but no token was returned")
                self._save_credentials({"token": bot_token, "base_url": result_base_url})
                return {"token": bot_token, "base_url": result_base_url}
            time.sleep(1)
        raise RuntimeError("Weixin QR login timed out")


def main() -> None:
    load_dotenv(Settings().env_path)
    settings = get_settings()
    configure_logging(settings)
    WeixinChannel(settings=settings).run()


if __name__ == "__main__":
    main()
