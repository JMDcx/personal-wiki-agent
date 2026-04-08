"""Normalization helpers for incoming Weixin iLink bot messages."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    from feishu_wiki_rag_agent.channel.weixin.weixin_api import CDN_BASE_URL, download_media_from_cdn
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from channel.weixin.weixin_api import CDN_BASE_URL, download_media_from_cdn


ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5

URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)


@dataclass(slots=True)
class DownloadedAttachment:
    """Downloaded inbound media attachment."""

    kind: str
    path: str
    display_name: str


@dataclass(slots=True)
class WeixinMessage:
    """Normalized Weixin message payload used by the standalone channel."""

    raw: dict
    tmp_dir: Path
    cdn_base_url: str = CDN_BASE_URL
    media_downloader: Callable[..., str] = download_media_from_cdn
    message_id: str = field(init=False)
    from_user_id: str = field(init=False)
    to_user_id: str = field(init=False)
    context_token: str = field(init=False)
    text: str = field(init=False, default="")
    urls: list[str] = field(init=False, default_factory=list)
    attachments: list[DownloadedAttachment] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.message_id = str(self.raw.get("message_id", self.raw.get("seq", "")))
        self.from_user_id = str(self.raw.get("from_user_id", ""))
        self.to_user_id = str(self.raw.get("to_user_id", ""))
        self.context_token = str(self.raw.get("context_token", ""))

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._parse_items()
        self.urls = URL_PATTERN.findall(self.text)

    @property
    def image_paths(self) -> list[str]:
        return [attachment.path for attachment in self.attachments if attachment.kind == "image"]

    @property
    def file_attachments(self) -> list[DownloadedAttachment]:
        return [attachment for attachment in self.attachments if attachment.kind == "file"]

    def stripped_text_without_urls(self) -> str:
        return URL_PATTERN.sub("", self.text).strip()

    def _parse_items(self) -> None:
        text_parts: list[str] = []
        for index, item in enumerate(self.raw.get("item_list", [])):
            item_type = int(item.get("type", 0) or 0)
            if item_type == ITEM_TEXT:
                content = str(item.get("text_item", {}).get("text", "")).strip()
                if content:
                    text_parts.append(content)
                continue
            if item_type == ITEM_IMAGE:
                self.attachments.append(self._download_attachment(item, index=index, kind="image"))
                continue
            if item_type == ITEM_FILE:
                self.attachments.append(self._download_attachment(item, index=index, kind="file"))

        self.text = "\n".join(part for part in text_parts if part).strip()

    def _download_attachment(self, item: dict, *, index: int, kind: str) -> DownloadedAttachment:
        info_key = "image_item" if kind == "image" else "file_item"
        info = item.get(info_key, {})
        media = info.get("media", {})

        encrypt_query_param = str(media.get("encrypt_query_param", "")).strip()
        aes_key = str(info.get("aeskey", "") or media.get("aes_key", "")).strip()
        if not encrypt_query_param or not aes_key:
            msg = f"Missing CDN info for inbound {kind} attachment"
            raise ValueError(msg)

        default_name = f"{self.message_id}_{index}"
        if kind == "image":
            display_name = f"{default_name}.jpg"
        else:
            display_name = str(info.get("file_name", "")).strip() or f"{default_name}.bin"
        safe_name = Path(display_name).name or display_name
        save_path = self.tmp_dir / safe_name

        self.media_downloader(
            cdn_base_url=self.cdn_base_url,
            encrypt_query_param=encrypt_query_param,
            aes_key=aes_key,
            save_path=str(save_path),
        )
        return DownloadedAttachment(kind=kind, path=str(save_path), display_name=os.path.basename(str(save_path)))
