"""Standalone Weixin long-poll channel for the Feishu Wiki RAG example."""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from feishu_wiki_rag_agent.agent import invoke_agent
from feishu_wiki_rag_agent.channel.weixin.weixin_api import (
    CDN_BASE_URL,
    DEFAULT_BASE_URL,
    WeixinApi,
)
from feishu_wiki_rag_agent.channel.weixin.weixin_api import download_media_from_cdn
from feishu_wiki_rag_agent.channel.weixin.weixin_message import DownloadedAttachment, WeixinMessage
from feishu_wiki_rag_agent.config import Settings, get_settings
from multimodal_rag_agent.docreader_service.client import DocreaderService
from multimodal_rag_agent.docreader_service.schemas import ParseRequest
from multimodal_rag_agent.models import ImageRef, ParsedDocument


logger = logging.getLogger(__name__)

QR_LOGIN_TIMEOUT_S = 480
QR_MAX_REFRESHES = 10
SESSION_EXPIRED_ERRCODE = -14
TEXT_CHUNK_LIMIT = 4000
MAX_CONSECUTIVE_FAILURES = 3
RETRY_DELAY = 2
BACKOFF_DELAY = 30


@dataclass
class MessageDeduper:
    """Simple TTL-based deduper for inbound Weixin message ids."""

    ttl_seconds: int = 60 * 60
    _seen: dict[str, float] = field(default_factory=dict)

    def should_process(self, message_id: str, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
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
        agent_runner: Callable[[str, str, list[str]], str] | None = None,
        deduper: MessageDeduper | None = None,
        docreader: DocreaderService | None = None,
        media_downloader: Callable[..., str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.api = api
        self.agent_runner = agent_runner or (
            lambda question, thread_id, images: invoke_agent(
                question,
                settings=self.settings,
                thread_id=thread_id,
                images=images,
            )
        )
        self.deduper = deduper or MessageDeduper()
        self.docreader = docreader or DocreaderService()
        self.media_downloader = media_downloader or download_media_from_cdn
        self._cursor = ""
        self._context_tokens: dict[str, str] = {}

    def run(self) -> None:
        """Authenticate if needed and start the long-poll loop."""
        self.api = self.api or self._build_authenticated_api()
        self._poll_loop()

    def handle_raw_message(self, raw_msg: dict) -> str | None:
        """Parse, adapt, answer, and reply to a single inbound Weixin message."""
        if str(raw_msg.get("message_type", "")) != "1":
            return None

        message_id = str(raw_msg.get("message_id", raw_msg.get("seq", "")))
        if not message_id or not self.deduper.should_process(message_id):
            return None

        from_user_id = str(raw_msg.get("from_user_id", ""))
        context_token = str(raw_msg.get("context_token", ""))
        if from_user_id and context_token:
            self._context_tokens[from_user_id] = context_token

        if self.api is None:
            self.api = self._build_api(token=self.settings.weixin_token)

        message = WeixinMessage(
            raw_msg,
            tmp_dir=self.settings.weixin_tmp_dir,
            cdn_base_url=self.api.cdn_base_url,
            media_downloader=self.media_downloader,
        )

        try:
            prompt, images = self._build_agent_turn(message)
        except UserFacingAttachmentError as exc:
            self._reply_text(from_user_id, context_token, str(exc))
            return str(exc)

        answer = self.agent_runner(prompt, f"weixin:{from_user_id}", images)
        self._reply_text(from_user_id, context_token, answer)
        return answer

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
                        logger.warning("Weixin session expired, triggering re-login")
                        self.api = self._build_authenticated_api(force_relogin=True)
                        self._cursor = ""
                        failures = 0
                        continue
                    failures += 1
                    logger.error("Weixin getUpdates error ret=%s errcode=%s", ret, errcode)
                    time.sleep(BACKOFF_DELAY if failures >= MAX_CONSECUTIVE_FAILURES else RETRY_DELAY)
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        failures = 0
                    continue

                failures = 0
                new_cursor = str(response.get("get_updates_buf", ""))
                if new_cursor:
                    self._cursor = new_cursor
                for raw_msg in response.get("msgs", []):
                    try:
                        self.handle_raw_message(raw_msg)
                    except Exception:
                        logger.exception("Failed to process Weixin message")
            except KeyboardInterrupt:
                raise
            except Exception:
                failures += 1
                logger.exception("Weixin long-poll failed")
                time.sleep(BACKOFF_DELAY if failures >= MAX_CONSECUTIVE_FAILURES else RETRY_DELAY)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    failures = 0

    def _build_agent_turn(self, message: WeixinMessage) -> tuple[str, list[str]]:
        images = list(message.image_paths)
        text_without_urls = message.stripped_text_without_urls()

        link_contexts: list[tuple[str, str]] = []
        file_contexts: list[tuple[str, str]] = []

        for url in message.urls:
            parsed = self._parse_url(url)
            link_contexts.append((url, parsed.markdown_content.strip()))
            images.extend(self._persist_image_refs(message.message_id, "link", url, parsed.image_refs))

        for attachment in message.file_attachments:
            parsed = self._parse_file(attachment)
            file_contexts.append((attachment.display_name, parsed.markdown_content.strip()))
            images.extend(
                self._persist_image_refs(message.message_id, "file", attachment.display_name, parsed.image_refs)
            )

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
            ), images

        if images and not text_without_urls:
            return "请分析这张图片的内容，并尽量直接回答用户最可能想问的问题。", images
        if text_without_urls:
            return text_without_urls, images

        raise UserFacingAttachmentError("未识别到可处理的文本、链接、图片或文件内容。")

    def _parse_url(self, url: str) -> ParsedDocument:
        try:
            parsed = self.docreader.parse(ParseRequest(url=url, title=url))
        except Exception as exc:  # noqa: BLE001
            raise UserFacingAttachmentError(f"链接解析失败：{url}") from exc
        if not parsed.markdown_content.strip():
            raise UserFacingAttachmentError(f"链接解析失败：{url}")
        return parsed

    def _parse_file(self, attachment: DownloadedAttachment) -> ParsedDocument:
        try:
            file_content = Path(attachment.path).read_bytes()
            parsed = self.docreader.parse(
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
        saved_paths: list[str] = []
        for index, image_ref in enumerate(image_refs):
            if not image_ref.image_data:
                continue
            suffix = Path(image_ref.filename).suffix or ".png"
            file_name = f"{message_id}_{source_kind}_{index}{suffix}"
            save_path = self.settings.weixin_tmp_dir / file_name
            save_path.write_bytes(image_ref.image_data)
            logger.debug("Saved parsed %s image %s to %s", source_kind, source_label, save_path)
            saved_paths.append(str(save_path))
        return saved_paths

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
            context_token = self._context_tokens.get(to_user_id, "")
        if not to_user_id or not context_token:
            logger.warning("Skipping Weixin reply because to_user_id/context_token is missing")
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
        except Exception:
            logger.exception("Failed to load Weixin credentials")
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
        except Exception:
            logger.exception("Failed to render QR code in terminal")
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
    logging.basicConfig(level=logging.INFO)
    load_dotenv(Settings().env_path)
    WeixinChannel().run()


if __name__ == "__main__":
    main()
