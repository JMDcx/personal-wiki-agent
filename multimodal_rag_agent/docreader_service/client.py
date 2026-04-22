"""In-process adapter for WeKnora docreader."""

from __future__ import annotations

import html
import importlib
import io
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

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
    _WECHAT_TAIL_MARKERS = (
        "预览时标签不可点",
        "微信扫一扫",
        "继续滑动看下一个",
        "轻触阅读原文",
        "使用小程序",
        "轻点两下取消赞",
        "轻点两下取消在看",
    )

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
        image_refs = self._filter_noise_image_refs(self._extract_image_refs(content_html, base_url=url))
        parser_backend = "requests_html_fallback"
        markitdown_failure = ""
        markdown_content = ""
        if self._should_use_markitdown(request):
            try:
                markdown_content = self._convert_html_with_markitdown(content_html, title=title, url=url)
                if markdown_content.strip():
                    parser_backend = "markitdown_html_fallback"
            except Exception as exc:
                markitdown_failure = f"{type(exc).__name__}: {exc}"
        if not markdown_content.strip():
            markdown_content = self._html_to_markdown(content_html, title=title, image_refs=image_refs)
        else:
            markdown_content = self._sanitize_wechat_markdown(markdown_content)
            markdown_content = self._ensure_markdown_has_images(markdown_content, image_refs)
        if not markdown_content.strip():
            msg = f"Fallback URL parser returned empty content for {url}"
            raise RuntimeError(msg) from cause

        metadata = {
            "title": title,
            "author": author,
            "published_at": published_at,
            "source_url": url,
            "parser_backend": parser_backend,
        }
        if cause is not None:
            metadata["fallback_reason"] = f"{type(cause).__name__}: {cause}"
        if markitdown_failure:
            metadata["markitdown_failure"] = markitdown_failure
        return ParsedDocument(
            markdown_content=markdown_content,
            image_refs=image_refs,
            metadata={k: v for k, v in metadata.items() if v},
        )

    @staticmethod
    def _should_use_markitdown(request: ParseRequest) -> bool:
        overrides = request.engine_overrides or {}
        if overrides.get("prefer_markitdown") or overrides.get("force_markitdown"):
            return True
        url = str(request.url or "")
        if "mp.weixin.qq.com" in url:
            return False
        return True

    def _convert_html_with_markitdown(self, html_fragment: str, *, title: str, url: str) -> str:
        module = importlib.import_module("markitdown")
        markitdown_cls = getattr(module, "MarkItDown")
        converter = markitdown_cls(enable_plugins=False)
        wrapped_html = (
            "<html><head>"
            f"<title>{html.escape(title or url)}</title>"
            "</head><body>"
            f"{html_fragment}"
            "</body></html>"
        )
        result = converter.convert_stream(
            io.BytesIO(wrapped_html.encode("utf-8")),
            file_extension=".html",
            url=url,
        )
        return str(getattr(result, "text_content", "") or "").strip()

    def _sanitize_wechat_markitdown(self, markdown_content: str) -> str:
        content = str(markdown_content or "").replace("\r", "")
        content = re.sub(r"!\[[^\]]*]\(\s*\)", "", content)
        content = re.sub(r"!\[(?:跳转二维码|作者头像)]\(\s*[^)]*\)", "", content)
        tail_positions = [content.find(marker) for marker in self._WECHAT_TAIL_MARKERS if content.find(marker) >= 0]
        if tail_positions:
            content = content[: min(tail_positions)]
        lines = [line.rstrip() for line in content.splitlines()]
        compact_lines: list[str] = []
        previous_blank = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if not previous_blank:
                    compact_lines.append("")
                previous_blank = True
                continue
            if line in self._WECHAT_TAIL_MARKERS:
                continue
            compact_lines.append(raw_line.strip())
            previous_blank = False
        compact = "\n".join(compact_lines).strip()
        compact = re.sub(r"\n{3,}", "\n\n", compact)
        return compact.strip()

    def _sanitize_wechat_markdown(self, markdown_content: str) -> str:
        sanitized = self._sanitize_wechat_markitdown(markdown_content)
        return sanitized if sanitized.strip() else markdown_content.strip()

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

    def _html_to_markdown(self, html_fragment: str, *, title: str, image_refs: list[ImageRef] | None = None) -> str:
        content = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", html_fragment, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r"(?is)<img\b[^>]*>", lambda match: self._img_tag_to_markdown(match.group(0)), content)
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
            compact = f"# {title}".strip()
        else:
            compact = f"# {title}\n\n{compact}".strip()
        return self._ensure_markdown_has_images(compact, image_refs or [])

    def _extract_image_refs(self, html_fragment: str, *, base_url: str) -> list[ImageRef]:
        image_refs: list[ImageRef] = []
        seen: set[str] = set()
        for match in re.finditer(r"(?is)<img\b(?P<attrs>[^>]*)>", html_fragment):
            original_ref = self._extract_img_src(match.group("attrs"), base_url=base_url)
            if not original_ref or original_ref in seen:
                continue
            seen.add(original_ref)
            image_refs.append(
                ImageRef(
                    filename=self._guess_image_filename(original_ref),
                    original_ref=original_ref,
                    mime_type=self._guess_mime(original_ref),
                )
            )
        return image_refs

    def _extract_img_src(self, attrs: str, *, base_url: str) -> str:
        attributes = self._extract_img_attribute_map(attrs)
        raw_url = attributes.get("data-src") or attributes.get("src") or ""
        if not raw_url.strip():
            return ""
        return urljoin(base_url, html.unescape(raw_url.strip()))

    @staticmethod
    def _extract_img_attribute_map(attrs: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in re.findall(r'([:\w-]+)\s*=\s*["\'](.*?)["\']', attrs, flags=re.DOTALL):
            result[key.lower()] = value
        return result

    def _img_tag_to_markdown(self, tag: str) -> str:
        match = re.search(r"(?is)<img\b(?P<attrs>[^>]*)>", tag)
        if not match:
            return ""
        attributes = self._extract_img_attribute_map(match.group("attrs"))
        raw_url = attributes.get("data-src") or attributes.get("src") or ""
        if not raw_url.strip():
            return ""
        alt = html.unescape(attributes.get("alt", "")).strip()
        return f"\n\n![{alt}]({html.unescape(raw_url.strip())})\n\n"

    @staticmethod
    def _filter_noise_image_refs(image_refs: list[ImageRef]) -> list[ImageRef]:
        filtered: list[ImageRef] = []
        for ref in image_refs:
            original_ref = str(ref.original_ref or "").strip().lower()
            if not original_ref:
                continue
            if "pic_blank.gif" in original_ref:
                continue
            if "/0?wx_fmt=png" in original_ref and "mmbiz.qpic.cn" in original_ref:
                continue
            filtered.append(ref)
        return filtered

    @staticmethod
    def _ensure_markdown_has_images(markdown_content: str, image_refs: list[ImageRef]) -> str:
        if not image_refs:
            return markdown_content.strip()
        existing_urls = set(re.findall(r"!\[[^\]]*\]\(([^)\n]+)\)", markdown_content))
        missing_refs = [ref for ref in image_refs if ref.original_ref not in existing_urls]
        if not missing_refs:
            return markdown_content.strip()
        appendix = "\n\n".join(f"![]({ref.original_ref})" for ref in missing_refs)
        return f"{markdown_content.rstrip()}\n\n## 图片\n\n{appendix}".strip()

    @staticmethod
    def _markdown_contains_image_refs(markdown_content: str, image_refs: list[ImageRef]) -> bool:
        existing_urls = set(re.findall(r"!\[[^\]]*\]\(([^)\n]+)\)", markdown_content))
        return all(ref.original_ref in existing_urls for ref in image_refs)

    @staticmethod
    def _clean_html_fragment(fragment: str) -> str:
        text = re.sub(r"<[^>]+>", "", fragment)
        return html.unescape(text).strip()

    @staticmethod
    def _guess_mime(ref_path: str) -> str:
        if ref_path.startswith("data:image/"):
            return ref_path.split(";", 1)[0].removeprefix("data:")
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
    def _guess_image_filename(ref_path: str) -> str:
        if ref_path.startswith("data:image/"):
            mime_type = ref_path.split(";", 1)[0].removeprefix("data:")
            suffix = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "image/bmp": ".bmp",
            }.get(mime_type, ".bin")
            return f"{uuid.uuid4().hex}{suffix}"
        parsed = urlparse(ref_path)
        filename = Path(parsed.path).name
        return filename or f"{uuid.uuid4().hex}.bin"

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
