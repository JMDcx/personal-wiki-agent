"""Resolve and normalize document images."""

from __future__ import annotations

import base64
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests

from multimodal_rag_agent.image_resolver.storage import ImageStorage
from multimodal_rag_agent.models import ParsedDocument, ResolvedImage

IMAGE_MD_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<url>[^)\n]+)\)")
IMAGE_HTML_RE = re.compile(
    r"""(?is)<img\s+[^>]*src=["'](?P<src>[^"']+)["'][^>]*>"""
)


@dataclass
class ResolvedDocument:
    markdown_content: str
    images: list[ResolvedImage]


class ImageResolver:
    """Resolve docreader, data URI, HTML, and remote images into storage-backed URLs."""

    min_image_dimension = 128
    min_image_bytes = 512

    def __init__(self, storage: ImageStorage, request_timeout: int = 20) -> None:
        self.storage = storage
        self.request_timeout = request_timeout

    def resolve(self, document_id: str, parsed: ParsedDocument) -> ResolvedDocument:
        markdown = parsed.markdown_content
        stored_images: list[ResolvedImage] = []
        ref_map = {ref.original_ref: ref for ref in parsed.image_refs}

        def replace_markdown(match: re.Match[str]) -> str:
            original = match.group("url").strip()
            alt = match.group("alt")
            resolved = self._resolve_single(document_id, original, ref_map)
            if resolved is None:
                return ""
            stored_images.append(resolved)
            return f"![{alt}]({resolved.public_url})"

        markdown = IMAGE_MD_RE.sub(replace_markdown, markdown)

        def replace_html(match: re.Match[str]) -> str:
            original = match.group("src").strip()
            resolved = self._resolve_single(document_id, original, ref_map, source_type="html_img")
            if resolved is None:
                return ""
            stored_images.append(resolved)
            return f"![]({resolved.public_url})"

        markdown = IMAGE_HTML_RE.sub(replace_html, markdown)
        return ResolvedDocument(markdown_content=markdown, images=stored_images)

    def _resolve_single(
        self,
        document_id: str,
        original_ref: str,
        ref_map: dict[str, object],
        *,
        source_type: str | None = None,
    ) -> ResolvedImage | None:
        content = b""
        mime_type = "application/octet-stream"
        inferred_source_type = source_type or "doc_image"
        image_name = Path(urlparse(original_ref).path).name or f"{uuid.uuid4().hex}.bin"

        if original_ref in ref_map:
            ref = ref_map[original_ref]
            content = getattr(ref, "image_data", b"")
            mime_type = getattr(ref, "mime_type", mime_type)
            inferred_source_type = "doc_image"
            image_name = getattr(ref, "filename", image_name)
        elif original_ref.startswith("data:image/"):
            inferred_source_type = "data_uri"
            try:
                mime_type, content = self._decode_data_uri(original_ref)
            except (ValueError, base64.binascii.Error):
                return None
            image_name = f"{uuid.uuid4().hex}{self._suffix_from_mime(mime_type)}"
        elif original_ref.startswith("http://") or original_ref.startswith("https://"):
            inferred_source_type = "remote_url"
            mime_type, content = self._download_remote_image(original_ref)
        else:
            return None

        if not content or self._is_icon_image(content):
            return None

        stored_path, public_url = self.storage.save(document_id, image_name, content)
        return ResolvedImage(
            image_id=uuid.uuid4().hex,
            original_ref=original_ref,
            stored_path=stored_path,
            public_url=public_url,
            mime_type=mime_type,
            source_type=inferred_source_type,
        )

    @staticmethod
    def _decode_data_uri(data_uri: str) -> tuple[str, bytes]:
        if "," not in data_uri:
            raise ValueError("invalid data URI image payload")
        header, payload = data_uri.split(",", 1)
        mime_type = header.split(";")[0].removeprefix("data:")
        return mime_type, base64.b64decode(payload)

    def _download_remote_image(self, url: str) -> tuple[str, bytes]:
        response = requests.get(url, timeout=self.request_timeout)
        response.raise_for_status()
        return response.headers.get("Content-Type", "application/octet-stream"), response.content

    @classmethod
    def _is_icon_image(cls, content: bytes) -> bool:
        if len(content) < cls.min_image_bytes:
            return True
        try:
            from PIL import Image

            with Image.open(BytesIO(content)) as image:
                width, height = image.size
                return width < cls.min_image_dimension or height < cls.min_image_dimension
        except Exception:
            return False

    @staticmethod
    def _suffix_from_mime(mime_type: str) -> str:
        return {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
        }.get(mime_type, ".bin")
