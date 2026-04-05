"""In-process adapter for WeKnora docreader."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.docreader_service.schemas import ParseRequest
from multimodal_rag_agent.models import ImageRef, ParsedDocument


class DocreaderUnavailableError(RuntimeError):
    """Raised when WeKnora docreader cannot be loaded."""


class DocreaderService:
    """Adapter for the local WeKnora Python docreader parser."""

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
        parser = self._new_parser()
        if request.url:
            result = parser.parse_url(
                request.url,
                request.title,
                parser_engine=request.parser_engine or None,
                engine_overrides=request.engine_overrides or None,
            )
        else:
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
