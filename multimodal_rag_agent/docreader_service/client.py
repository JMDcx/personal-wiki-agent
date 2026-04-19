"""In-process adapter for WeKnora docreader."""

from __future__ import annotations

import html
import importlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.docreader_service.schemas import ParseRequest
from multimodal_rag_agent.models import ImageRef, ParsedDocument


class DocreaderUnavailableError(RuntimeError):
    """Raised when WeKnora docreader cannot be loaded."""


class DocreaderService:
    """Adapter for the local WeKnora Python docreader parser."""

    _DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/135.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()
        self._parser_cls = None

    def _load_parser_class(self):
        if self._parser_cls is not None:
            return self._parser_cls

        base = self.settings.docreader_project_dir
        docreader_root = base / "docreader"
        if not docreader_root.exists():
            msg = f"WeKnora docreader project not found: {docreader_root}"
            raise DocreaderUnavailableError(msg)

        root_str = str(base)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            module = importlib.import_module("docreader.parser")
        except Exception as exc:  # pragma: no cover - import error path
            msg = f"Failed to import WeKnora docreader.parser: {exc}"
            raise DocreaderUnavailableError(msg) from exc

        self._parser_cls = getattr(module, "Parser")
        return self._parser_cls

    def _new_parser(self):
        return self._load_parser_class()()

    def parse(self, request: ParseRequest) -> ParsedDocument:
        if request.url:
            try:
                parser = self._new_parser()
                result = parser.parse_url(
                    request.url,
                    request.title,
                    parser_engine=request.parser_engine or None,
                    engine_overrides=request.engine_overrides or None,
                )
                parsed = self._to_parsed_document(result)
                if parsed.markdown_content.strip():
                    return parsed
            except Exception as exc:
                return self._parse_url_with_fallback(request, cause=exc)
            return self._parse_url_with_fallback(request)

        parser = self._new_parser()
        if not request.file_name or request.file_content is None:
            raise ValueError("file_name and file_content are required for file parsing")
        suffix = Path(request.file_name).suffix.lstrip(".")
        file_type = request.file_type or suffix
        result = parser.parse_file(
            request.file_name,
            file_type,
            request.file_content,
            parser_engine=request.parser_engine or None,
            engine_overrides=request.engine_overrides or None,
        )
        return self._to_parsed_document(result)

    def _to_parsed_document(self, result: object) -> ParsedDocument:
        image_refs = [
            ImageRef(
                filename=Path(ref_path).name or "image.png",
                original_ref=ref_path,
                mime_type=self._guess_mime(ref_path),
                image_data=self._decode_image(raw),
            )
            for ref_path, raw in (getattr(result, "images", {}) or {}).items()
        ]
        return ParsedDocument(
            markdown_content=getattr(result, "content", "") or "",
            image_refs=image_refs,
            metadata=dict(getattr(result, "metadata", {}) or {}),
        )

    def _parse_url_with_fallback(self, request: ParseRequest, cause: Exception | None = None) -> ParsedDocument:
        url = request.url or ""
        response = requests.get(url, headers=self._DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        response.encoding = response.encoding or response.apparent_encoding or "utf-8"
        html_text = response.text

        title = self._extract_title(html_text, fallback=request.title or url)
        author = self._extract_wechat_author(html_text)
        published_at = self._extract_wechat_published_at(html_text)
        content_html = self._extract_content_html(html_text)
        markdown_content = self._html_to_markdown(content_html, title=title)
        if not markdown_content.strip():
            msg = f"Fallback URL parser returned empty content for {url}"
            raise RuntimeError(msg) from cause

        metadata = {
            "title": title,
            "author": author,
            "published_at": published_at,
            "source_url": url,
            "parser_backend": "requests_html_fallback",
        }
        if cause is not None:
            metadata["fallback_reason"] = f"{type(cause).__name__}: {cause}"
        return ParsedDocument(markdown_content=markdown_content, metadata={k: v for k, v in metadata.items() if v})

    def _extract_title(self, html_text: str, *, fallback: str) -> str:
        candidates = [
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+name=["\']twitter:title["\'][^>]+content=["\'](.*?)["\']',
            r'<h1[^>]+id=["\']activity-name["\'][^>]*>(.*?)</h1>',
            r"<title[^>]*>(.*?)</title>",
        ]
        for pattern in candidates:
            matched = re.search(pattern, html_text, re.IGNORECASE | re.DOTALL)
            if matched:
                cleaned = self._clean_html_fragment(matched.group(1))
                if cleaned:
                    return cleaned
        return fallback.strip()

    def _extract_wechat_author(self, html_text: str) -> str:
        candidates = [
            r'<meta[^>]+name=["\']author["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+property=["\']article:author["\'][^>]+content=["\'](.*?)["\']',
            r'<span[^>]+id=["\']js_name["\'][^>]*>(.*?)</span>',
        ]
        for pattern in candidates:
            matched = re.search(pattern, html_text, re.IGNORECASE | re.DOTALL)
            if matched:
                cleaned = self._clean_html_fragment(matched.group(1))
                if cleaned:
                    return cleaned
        return ""

    def _extract_wechat_published_at(self, html_text: str) -> str:
        direct_match = re.search(
            r'<(em|span)[^>]+id=["\']publish_time["\'][^>]*>(.*?)</(em|span)>',
            html_text,
            re.IGNORECASE | re.DOTALL,
        )
        if direct_match:
            cleaned = self._clean_html_fragment(direct_match.group(2))
            if cleaned:
                return cleaned
        timestamp_match = re.search(r'\bvar\s+ct\s*=\s*["\']?(\d{10})["\']?', html_text)
        if not timestamp_match:
            timestamp_match = re.search(r'"publish_time"\s*:\s*"?(\\d{10})"?', html_text)
        if not timestamp_match:
            timestamp_match = re.search(r'"publish_time"\s*:\s*"?(\\d{10})"?', html_text.replace("\\", ""))
        if timestamp_match:
            try:
                ts = int(timestamp_match.group(1))
                return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
            except Exception:
                return timestamp_match.group(1)
        return ""

    def _extract_content_html(self, html_text: str) -> str:
        marker = re.search(r'<div[^>]+id=["\']js_content["\']', html_text, re.IGNORECASE)
        if marker:
            tail = html_text[marker.start() :]
            end_markers = [
                '<section class="wx_profile_card_inner">',
                '<script type="text/javascript">',
                "</body>",
            ]
            end_positions = [tail.find(candidate) for candidate in end_markers if tail.find(candidate) > 0]
            if end_positions:
                return tail[: min(end_positions)]
        candidates = [
            r'<article[^>]*>(.*?)</article>',
            r'<main[^>]*>(.*?)</main>',
            r"<body[^>]*>(.*?)</body>",
        ]
        for pattern in candidates:
            matched = re.search(pattern, html_text, re.IGNORECASE | re.DOTALL)
            if matched:
                return matched.group(1)
        return html_text

    def _html_to_markdown(self, html_fragment: str, *, title: str) -> str:
        content = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", html_fragment, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r"<br\s*/?>", "\n", content, flags=re.IGNORECASE)
        content = re.sub(r"</(p|div|section|article|li|ul|ol|h1|h2|h3|h4|h5|h6|tr)>", "\n", content, flags=re.IGNORECASE)
        content = re.sub(r"<li[^>]*>", "- ", content, flags=re.IGNORECASE)
        content = re.sub(r"<[^>]+>", "", content)
        content = html.unescape(content)
        content = content.replace("\r", "")
        lines = [line.strip() for line in content.splitlines()]
        compact = "\n".join(line for line in lines if line)
        compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
        if not compact:
            return f"# {title}".strip()
        return f"# {title}\n\n{compact}".strip()

    @staticmethod
    def _clean_html_fragment(fragment: str) -> str:
        text = re.sub(r"<[^>]+>", "", fragment)
        return html.unescape(text).strip()

    @staticmethod
    def _guess_mime(ref_path: str) -> str:
        suffix = Path(ref_path).suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }.get(suffix, "application/octet-stream")

    @staticmethod
    def _decode_image(raw: object) -> bytes:
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            import base64

            try:
                return base64.b64decode(raw)
            except Exception:
                return raw.encode("utf-8")
        return b""
