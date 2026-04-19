"""Source adapters for the knowledge deposit flow."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

try:
    from feishu_wiki_rag_agent.config import Settings, get_settings
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import Settings, get_settings
from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.deposit_pipeline.models import DepositRequest, SourceMaterial
from multimodal_rag_agent.docreader_service.client import DocreaderService
from multimodal_rag_agent.docreader_service.schemas import ParseRequest
from multimodal_rag_agent.multimodal_image_pipeline.caption import CaptionService
from multimodal_rag_agent.multimodal_image_pipeline.ocr import OCRService


URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
TITLE_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
URL_TRAILING_PUNCTUATION = ")]}>,，。；;！？!?\"'`#"
URL_ALLOWED_PREFIX_RE = re.compile(r"^(https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)")
INVALID_PAGE_MARKERS = (
    "参数错误",
    "轻点两下取消赞",
    "轻点两下取消在看",
    "微信扫一扫",
    "继续滑动看下一个",
)


class DepositSourceError(RuntimeError):
    """Raised when source extraction fails."""


class BaseSourceAdapter(ABC):
    """Base adapter for deposit sources."""

    @abstractmethod
    def can_handle(self, request: DepositRequest) -> bool: ...

    @abstractmethod
    def fetch(self, request: DepositRequest) -> SourceMaterial: ...


def normalize_url(url: str) -> str:
    candidate = str(url or "").strip()
    if not candidate:
        return ""
    if candidate.startswith("[") and "](" in candidate and candidate.endswith(")"):
        candidate = candidate.split("](", 1)[1][:-1].strip()
    candidate = candidate.lstrip("([<")
    matched = URL_ALLOWED_PREFIX_RE.match(candidate)
    if matched:
        candidate = matched.group(1)
    while candidate and candidate[-1] in URL_TRAILING_PUNCTUATION:
        candidate = candidate[:-1]
    return candidate.strip()


def extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    normalized_urls: list[str] = []
    for matched in URL_PATTERN.findall(text or ""):
        normalized = normalize_url(matched)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_urls.append(normalized)
    return normalized_urls


def extract_title_from_markdown(markdown: str, *, fallback: str = "") -> str:
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = TITLE_LINE_RE.match(line)
        if matched:
            title = matched.group(1).strip()
            if title:
                return title
        if not line.startswith("http://") and not line.startswith("https://"):
            return line[:120].strip()
    return fallback.strip()


def is_invalid_extracted_content(markdown: str) -> bool:
    compact = str(markdown or "").strip()
    if not compact:
        return True
    hits = sum(1 for marker in INVALID_PAGE_MARKERS if marker in compact)
    return hits >= 2


class XiaohongshuAdapter(BaseSourceAdapter):
    """Fetch a Xiaohongshu post from the MCP server."""

    def __init__(self, settings: Settings | None = None, *, session: requests.Session | None = None) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()

    def can_handle(self, request: DepositRequest) -> bool:
        return any("xiaohongshu.com" in url or "xhslink.com" in url for url in request.urls)

    def fetch(self, request: DepositRequest) -> SourceMaterial:
        source_url = next(url for url in request.urls if "xiaohongshu.com" in url or "xhslink.com" in url)
        feed_id, xsec_token = self._extract_feed_and_token(source_url)
        if not feed_id or not xsec_token:
            raise DepositSourceError("小红书链接缺少 feed_id 或 xsec_token，无法获取帖子详情。")
        payload = self._call_mcp_tool(feed_id=feed_id, xsec_token=xsec_token)
        data = self._normalize_payload(payload)
        note = self._extract_note(data)
        title = str(note.get("title") or data.get("title") or "小红书帖子").strip()
        description = str(
            note.get("desc")
            or note.get("content")
            or data.get("desc")
            or data.get("content")
            or data.get("description")
            or ""
        ).strip()
        user = note.get("user") if isinstance(note.get("user"), dict) else {}
        author = str(
            user.get("nickname")
            or user.get("nickName")
            or note.get("nickname")
            or data.get("nickname")
            or data.get("author")
            or data.get("user_name")
            or ""
        ).strip()
        published_at = str(note.get("time") or data.get("publish_time") or data.get("time") or "").strip()
        interact_info = note.get("interactInfo") if isinstance(note.get("interactInfo"), dict) else {}
        interactions = []
        interaction_pairs = [
            ("点赞", interact_info.get("likedCount") or data.get("liked_count")),
            ("收藏", interact_info.get("collectedCount") or data.get("collected_count")),
            ("评论", interact_info.get("commentCount") or data.get("comment_count")),
            ("分享", interact_info.get("sharedCount") or data.get("share_count")),
        ]
        for label, value in interaction_pairs:
            if value not in {None, ""}:
                interactions.append(f"- {label}: {value}")
        comments = []
        for comment in self._extract_comments(data)[:5]:
            content = str(comment.get("content") or "").strip()
            if content:
                comments.append(f"- {content}")
        image_items = note.get("imageList") if isinstance(note.get("imageList"), list) else []
        image_lines = []
        for image in image_items[:10]:
            if isinstance(image, dict):
                url = str(image.get("urlDefault") or image.get("urlPre") or "").strip()
                if url:
                    image_lines.append(f"- {url}")
        parts = [f"# {title}", "", description]
        if author or published_at:
            parts.extend(
                [
                    "",
                    "## 作者信息",
                    f"- 作者：{author or '未知'}",
                    f"- 发布时间：{published_at or '未知'}",
                ]
            )
        if interactions:
            parts.extend(["", "## 互动数据", *interactions])
        if image_lines:
            parts.extend(["", "## 图片", *image_lines])
        if comments:
            parts.extend(["", "## 评论摘要", *comments])
        return SourceMaterial(
            source_type="xiaohongshu",
            source_uri=source_url,
            title=title,
            author=author,
            published_at=published_at,
            raw_markdown="\n".join(part for part in parts if part is not None).strip(),
            extra_summary=str(note.get("ipLocation") or data.get("ip_location") or "").strip(),
            metadata={
                "feed_id": feed_id,
                "xsec_token": xsec_token,
                "raw_payload": data,
                "note_id": str(note.get("noteId") or "").strip(),
                "image_urls": image_lines,
            },
        )

    @staticmethod
    def _extract_feed_and_token(url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        xsec_token = (query.get("xsec_token") or [""])[0].strip()
        path_parts = [part for part in parsed.path.split("/") if part]
        feed_id = ""
        if "explore" in path_parts:
            index = path_parts.index("explore")
            if index + 1 < len(path_parts):
                feed_id = path_parts[index + 1].strip()
        if not feed_id:
            for candidate in path_parts[::-1]:
                if len(candidate) >= 8:
                    feed_id = candidate.strip()
                    break
        return feed_id, xsec_token

    def _call_mcp_tool(self, *, feed_id: str, xsec_token: str) -> dict[str, Any]:
        session_id = self._initialize_mcp_session()
        payload = self._post_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_feed_detail",
                    "arguments": {"feed_id": feed_id, "xsec_token": xsec_token},
                },
            },
            session_id=session_id,
        )
        if "error" in payload:
            raise DepositSourceError(f"小红书详情抓取失败：{payload['error']}")
        result = payload.get("result", {})
        if isinstance(result, dict) and result.get("isError"):
            content = result.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text").strip():
                        raise DepositSourceError(item["text"].strip())
            raise DepositSourceError("小红书详情抓取失败，MCP 返回了错误结果。")
        return payload

    def _initialize_mcp_session(self) -> str:
        response = self.session.post(
            self.settings.xhs_mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "feishu-wiki-rag-agent", "version": "0.1.0"},
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        session_id = response.headers.get("Mcp-Session-Id", "").strip() or response.headers.get("mcp-session-id", "").strip()
        self._post_jsonrpc(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
            session_id=session_id,
        )
        return session_id

    def _post_jsonrpc(self, payload: dict[str, Any], *, session_id: str = "") -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        response = self.session.post(
            self.settings.xhs_mcp_url,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        if not response.content.strip():
            return {}
        return response.json()

    @staticmethod
    def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        result = payload.get("result", {})
        if isinstance(result, dict) and isinstance(result.get("content"), list) and result["content"]:
            block = result["content"][0]
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                try:
                    return json.loads(block["text"])
                except json.JSONDecodeError:
                    return {"description": block["text"]}
        if isinstance(result, dict):
            structured = result.get("structuredContent")
            if isinstance(structured, dict):
                return structured
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _extract_note(data: dict[str, Any]) -> dict[str, Any]:
        note = data.get("note")
        if isinstance(note, dict):
            return note
        nested_data = data.get("data")
        if isinstance(nested_data, dict) and isinstance(nested_data.get("note"), dict):
            return nested_data["note"]
        return {}

    @staticmethod
    def _extract_comments(data: dict[str, Any]) -> list[dict[str, Any]]:
        direct = data.get("comments")
        if isinstance(direct, list):
            return [item for item in direct if isinstance(item, dict)]
        nested_data = data.get("data")
        if isinstance(nested_data, dict):
            nested_comments = nested_data.get("comments")
            if isinstance(nested_comments, list):
                return [item for item in nested_comments if isinstance(item, dict)]
        return []


class ProvidedContentAdapter(BaseSourceAdapter):
    """Use caller-provided markdown content directly when available."""

    def can_handle(self, request: DepositRequest) -> bool:
        return bool(request.provided_content.strip())

    def fetch(self, request: DepositRequest) -> SourceMaterial:
        content = request.provided_content.strip()
        if is_invalid_extracted_content(content):
            raise DepositSourceError("上游提供的正文内容疑似异常页或壳页，已拒绝沉淀。")
        source_url = request.urls[0] if request.urls else ""
        fallback_title = source_url or "已提供正文内容"
        title = extract_title_from_markdown(content, fallback=fallback_title)
        source_type = "url" if source_url else "text"
        return SourceMaterial(
            source_type=source_type,
            source_uri=source_url,
            title=title,
            raw_markdown=content,
            metadata={
                "source_url": source_url,
                "content_origin": "provided_content",
                "provided_content_length": len(content),
            },
        )


class GenericUrlAdapter(BaseSourceAdapter):
    """Use docreader to parse generic URLs including WeChat articles."""

    def __init__(self, settings: MultimodalRAGSettings | None = None, *, docreader: DocreaderService | None = None) -> None:
        self.settings = settings or get_multimodal_settings()
        self.docreader = docreader or DocreaderService(self.settings)

    def can_handle(self, request: DepositRequest) -> bool:
        return bool(request.urls)

    def fetch(self, request: DepositRequest) -> SourceMaterial:
        url = request.urls[0]
        parsed = self.docreader.parse(ParseRequest(url=url, title=url))
        content = parsed.markdown_content.strip()
        if not content or is_invalid_extracted_content(content):
            raise DepositSourceError(f"链接解析失败：{url}")
        title = str(parsed.metadata.get("title") or "").strip() or extract_title_from_markdown(content, fallback=url)
        return SourceMaterial(
            source_type="url",
            source_uri=url,
            title=title,
            author=str(parsed.metadata.get("author") or "").strip(),
            published_at=str(parsed.metadata.get("published_at") or "").strip(),
            raw_markdown=content,
            metadata={
                **dict(parsed.metadata),
                "source_url": url,
                "content_origin": "fetched_url",
            },
        )


class PlainTextAdapter(BaseSourceAdapter):
    """Use plain text directly as knowledge source."""

    def can_handle(self, request: DepositRequest) -> bool:
        return bool(request.text.strip())

    def fetch(self, request: DepositRequest) -> SourceMaterial:
        cleaned = request.text.strip()
        return SourceMaterial(
            source_type="text",
            source_uri="",
            title=(extract_title_from_markdown(cleaned, fallback="用户文本输入")[:80] or "用户文本输入").strip(),
            raw_markdown=cleaned,
            metadata={"text_length": len(cleaned)},
        )


class ImageAdapter(BaseSourceAdapter):
    """Turn one or more local images into markdown source material."""

    def __init__(
        self,
        settings: MultimodalRAGSettings | None = None,
        *,
        ocr_service: OCRService | None = None,
        caption_service: CaptionService | None = None,
    ) -> None:
        self.settings = settings or get_multimodal_settings()
        self.ocr_service = ocr_service or OCRService(self.settings)
        self.caption_service = caption_service or CaptionService(self.settings)

    def can_handle(self, request: DepositRequest) -> bool:
        return bool(request.image_paths)

    def fetch(self, request: DepositRequest) -> SourceMaterial:
        sections: list[str] = []
        for image_path in request.image_paths:
            path = Path(image_path)
            caption = (self.caption_service.caption(str(path)) or "").strip()
            ocr_text = (self.ocr_service.extract_text(str(path)) or "").strip()
            sections.extend(
                [
                    f"## 图片：{path.name}",
                    f"- 描述：{caption or '无'}",
                    "",
                    "### OCR",
                    ocr_text or "无可提取正文",
                    "",
                ]
            )
        return SourceMaterial(
            source_type="image",
            source_uri=request.image_paths[0],
            title=Path(request.image_paths[0]).name,
            raw_markdown="\n".join(sections).strip(),
            metadata={"image_paths": list(request.image_paths)},
        )
