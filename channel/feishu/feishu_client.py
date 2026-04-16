"""Feishu API client for websocket messaging and Wiki/Docs crawling."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    from feishu_wiki_rag_agent.config import Settings, get_settings
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import Settings, get_settings

try:
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    try:
        from observability.events import log_event, log_exception
    except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
        def log_event(event: str, **_: Any) -> None:
            return None

        def log_exception(event: str, exc: BaseException, **_: Any) -> None:
            return None

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:  # pragma: no cover - local fallback
    @dataclass
    class Document:
        page_content: str
        metadata: dict[str, Any]

from multimodal_rag_agent.models import ImageRef


SUPPORTED_DOC_TYPES = {"doc", "docx"}
FETCHABLE_DOC_TYPES = {"docx"}
IMAGE_TAG_RE = re.compile(r"<image\s+[^>]*token=\"(?P<token>[^\"]+)\"[^>]*/?>", re.IGNORECASE)
WHITEBOARD_TAG_RE = re.compile(r"<whiteboard\s+[^>]*token=\"(?P<token>[^\"]+)\"[^>]*/?>", re.IGNORECASE)


@dataclass
class _TokenCache:
    value: str = ""
    expires_at: float = 0.0


class FeishuClient:
    """Thin wrapper around the Feishu open platform APIs used by the example."""

    def __init__(self, settings: Settings | None = None, *, cli_runner: Any | None = None):
        self.settings = settings or get_settings()
        self._token_cache = _TokenCache()
        self.cli_runner = cli_runner or subprocess.run

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        auth: bool = True,
    ) -> dict[str, Any]:
        merged_headers = {"Content-Type": "application/json; charset=utf-8"}
        if headers:
            merged_headers.update(headers)
        if auth:
            merged_headers["Authorization"] = f"Bearer {self.fetch_tenant_access_token()}"

        response = requests.request(
            method=method,
            url=f"{self.settings.feishu_api_base}{path}",
            params=params,
            json=json_body,
            headers=merged_headers,
            timeout=self.settings.feishu_request_timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if detail:
                raise RuntimeError(f"HTTP error for {path}: {response.status_code} {detail}") from exc
            raise
        payload = response.json()
        if payload.get("code", 0) != 0:
            msg = payload.get("msg", "Unknown Feishu API error")
            raise RuntimeError(f"Feishu API error for {path}: {msg}")
        return payload

    def fetch_tenant_access_token(self) -> str:
        """Fetch and cache the tenant access token."""
        now = time.time()
        if self._token_cache.value and now < self._token_cache.expires_at:
            return self._token_cache.value

        payload = self._request(
            "POST",
            "/open-apis/auth/v3/tenant_access_token/internal",
            json_body={
                "app_id": self.settings.feishu_app_id,
                "app_secret": self.settings.feishu_app_secret,
            },
            auth=False,
        )
        token = str(payload["tenant_access_token"])
        expires_in = int(payload.get("expire", 7200))
        self._token_cache = _TokenCache(value=token, expires_at=now + max(expires_in - 60, 60))
        return token

    def fetch_bot_open_id(self) -> str | None:
        """Fetch the bot's open_id for mention matching."""
        payload = self._request("GET", "/open-apis/bot/v3/info/")
        bot = payload.get("bot", {})
        open_id = bot.get("open_id")
        return str(open_id) if open_id else None

    def reply_text(self, message_id: str, text: str) -> None:
        """Reply to a Feishu message with plain text."""
        self._request(
            "POST",
            f"/open-apis/im/v1/messages/{message_id}/reply",
            json_body={
                "content": json.dumps({"text": text}, ensure_ascii=False),
                "msg_type": "text",
            },
        )

    def get_message(self, message_id: str) -> dict[str, Any]:
        """Fetch a single Feishu message for reply-context reconstruction."""
        payload = self._request("GET", f"/open-apis/im/v1/messages/{message_id}")
        data = payload.get("data", {})
        items = data.get("items", [])
        if isinstance(items, list) and items:
            item = items[0]
            if isinstance(item, dict):
                return item
        message = data.get("message")
        if isinstance(message, dict):
            return message
        raise RuntimeError(f"Feishu message {message_id} not found")

    def get_wiki_node(self, node_token: str) -> dict[str, Any]:
        """Fetch metadata for a single wiki node."""
        try:
            payload = self._request(
                "GET",
                "/open-apis/wiki/v2/spaces/get_node",
                params={"token": node_token},
            )
        except RuntimeError as exc:
            error_text = str(exc)
            if "HTTP error" not in error_text:
                raise
            payload = self._request(
                "POST",
                "/open-apis/wiki/v2/spaces/get_node",
                json_body={"token": node_token},
            )
        data = payload.get("data", {})
        return data.get("node", data)

    def list_wiki_children(
        self,
        space_id: str,
        parent_node_token: str,
    ) -> list[dict[str, Any]]:
        """List child nodes for a wiki node with automatic pagination."""
        children: list[dict[str, Any]] = []
        page_token = ""

        while True:
            payload = self._request(
                "GET",
                f"/open-apis/wiki/v2/spaces/{space_id}/nodes",
                params={
                    "parent_node_token": parent_node_token,
                    "page_size": 50,
                    "page_token": page_token,
                },
            )
            data = payload.get("data", {})
            children.extend(data.get("items", []))
            if not data.get("has_more"):
                break
            page_token = str(data.get("page_token", ""))
            if not page_token:
                break

        return children

    def get_docx_raw_content(self, document_token: str) -> str:
        """Fetch plain-text raw content for a docx document."""
        payload = self._request(
            "GET",
            f"/open-apis/docx/v1/documents/{document_token}/raw_content",
        )
        data = payload.get("data", {})
        content = data.get("content", "")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected raw content response for document {document_token}")
        return content

    def get_legacy_doc_content(self, document_token: str) -> str:
        """Fetch content for a legacy `doc` document and flatten it to plain text."""
        payload = self._request(
            "GET",
            f"/open-apis/doc/v2/{document_token}/content",
        )
        data = payload.get("data", {})
        content = data.get("content", "")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return content
            return self._extract_text_from_legacy_doc(parsed)
        if isinstance(content, dict | list):
            return self._extract_text_from_legacy_doc(content)
        raise RuntimeError(f"Unexpected legacy doc response for document {document_token}")

    def _extract_text_from_legacy_doc(self, payload: dict[str, Any] | list[Any]) -> str:
        """Recursively extract human-readable text from legacy Feishu doc payloads."""
        lines: list[str] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                text_run = node.get("text_run")
                if isinstance(text_run, str) and text_run.strip():
                    lines.append(text_run.strip())
                text = node.get("text")
                if isinstance(text, str) and text.strip():
                    lines.append(text.strip())
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)
        return "\n".join(lines)

    def _build_document(
        self,
        *,
        page_content: str,
        doc_token: str,
        node_token: str,
        title: str,
        source_url: str,
        image_refs: list[ImageRef] | None = None,
    ) -> Document | None:
        """Create a LangChain document when the content is non-empty."""
        cleaned = page_content.strip()
        if not cleaned:
            return None
        return Document(
            page_content=cleaned,
            metadata={
                "doc_token": doc_token,
                "node_token": node_token,
                "title": title,
                "source_url": source_url,
                "image_refs": list(image_refs or []),
            },
        )

    def _run_lark_cli(self, args: list[str], *, cwd: str | None = None) -> dict[str, Any]:
        if not shutil.which("lark-cli"):
            raise RuntimeError("lark-cli is not installed or not in PATH.")

        profile = self.settings.feishu_lark_cli_profile.strip() or "feishu-wiki-rag-agent"
        command = ["lark-cli", *args, "--profile", profile, "--as", "bot"]
        result = self.cli_runner(
            command,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=self.settings.feishu_request_timeout,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"lark-cli {' '.join(args)} failed: {detail}")

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError(f"lark-cli {' '.join(args)} returned empty output.")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return {"ok": True, "data": {"raw": stdout}}

        if isinstance(payload, dict) and "ok" in payload:
            if not payload.get("ok"):
                error = payload.get("error") or {}
                message = error.get("message") or stdout
                raise RuntimeError(f"lark-cli {' '.join(args)} failed: {message}")
            return payload
        return {"ok": True, "data": payload}

    @staticmethod
    def _payload_data(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(payload, dict):
            return payload
        return {}

    def _download_cli_media_bytes(self, token: str, *, media_type: str, doc_ref: str = "") -> ImageRef | None:
        temp_root = self.settings.rag_data_dir / "tmp" / "feishu_media"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=temp_root) as temp_dir:
            output_name = token
            args = [
                "docs",
                "+media-download",
                "--token",
                token,
                "--output",
                output_name,
                "--overwrite",
            ]
            if media_type == "whiteboard":
                args.extend(["--type", "whiteboard"])
            try:
                payload = self._run_lark_cli(args, cwd=temp_dir)
            except Exception as exc:  # noqa: BLE001
                log_exception(
                    "feishu_doc_media_download_failed",
                    exc,
                    doc_ref=doc_ref,
                    media_type=media_type,
                    media_token=token,
                )
                raise
            data = self._payload_data(payload)
            saved_path = str(data.get("saved_path") or output_name).strip()
            content_type = str(data.get("content_type") or "").strip() or "application/octet-stream"
            path = Path(saved_path)
            if not path.is_absolute():
                path = Path(temp_dir) / path
            if not path.exists():
                log_event(
                    "feishu_doc_media_download_completed",
                    doc_ref=doc_ref,
                    media_type=media_type,
                    media_token=token,
                    success=False,
                    saved_path=str(path),
                )
                return None
            log_event(
                "feishu_doc_media_download_completed",
                doc_ref=doc_ref,
                media_type=media_type,
                media_token=token,
                success=True,
                saved_path=str(path),
                size_bytes=path.stat().st_size,
                content_type=content_type,
            )
            return ImageRef(
                filename=path.name,
                original_ref=self._media_placeholder(token, media_type=media_type),
                mime_type=content_type,
                image_data=path.read_bytes(),
            )

    @staticmethod
    def _media_placeholder(token: str, *, media_type: str) -> str:
        prefix = "__feishu_whiteboard__" if media_type == "whiteboard" else "__feishu_image__"
        return f"{prefix}{token}"

    def _replace_media_tags_with_markdown(self, markdown: str, *, doc_ref: str = "") -> tuple[str, list[ImageRef]]:
        image_refs: list[ImageRef] = []

        def replace_image(match: re.Match[str]) -> str:
            token = match.group("token").strip()
            try:
                image_ref = self._download_cli_media_bytes(token, media_type="media", doc_ref=doc_ref)
            except RuntimeError:
                return ""
            if image_ref is None:
                return ""
            image_refs.append(image_ref)
            return f"![]({image_ref.original_ref})"

        markdown = IMAGE_TAG_RE.sub(replace_image, markdown)

        def replace_whiteboard(match: re.Match[str]) -> str:
            token = match.group("token").strip()
            try:
                image_ref = self._download_cli_media_bytes(token, media_type="whiteboard", doc_ref=doc_ref)
            except RuntimeError:
                return ""
            if image_ref is None:
                return ""
            image_refs.append(image_ref)
            return f"![]({image_ref.original_ref})"

        markdown = WHITEBOARD_TAG_RE.sub(replace_whiteboard, markdown)
        return markdown, image_refs

    def get_docx_markdown_with_media(self, doc_or_url: str) -> tuple[str, list[ImageRef]]:
        log_event(
            "feishu_doc_media_fetch_started",
            doc_ref=doc_or_url,
        )
        payload = self._run_lark_cli(
            [
                "docs",
                "+fetch",
                "--doc",
                doc_or_url,
                "--format",
                "json",
            ]
        )
        data = self._payload_data(payload)
        markdown = str(data.get("markdown") or data.get("content") or data.get("raw") or "").strip()
        if not markdown:
            raise RuntimeError(f"Unexpected docs +fetch response for {doc_or_url}")
        image_token_count = len(IMAGE_TAG_RE.findall(markdown))
        whiteboard_token_count = len(WHITEBOARD_TAG_RE.findall(markdown))
        log_event(
            "feishu_doc_media_tokens_detected",
            doc_ref=doc_or_url,
            image_token_count=image_token_count,
            whiteboard_token_count=whiteboard_token_count,
        )
        return self._replace_media_tags_with_markdown(markdown, doc_ref=doc_or_url)

    def _build_direct_document_from_token(self, token: str) -> Document | None:
        """Try to interpret a root token as a direct docx/doc token."""
        source_url = f"feishu://document/{token}"
        try:
            markdown_content, image_refs = self.get_docx_markdown_with_media(token)
            return self._build_document(
                page_content=markdown_content,
                doc_token=token,
                node_token="",
                title=token,
                source_url=source_url,
                image_refs=image_refs,
            )
        except RuntimeError as exc:
            log_exception(
                "feishu_doc_media_fetch_fallback",
                exc,
                doc_ref=token,
                fallback="docx_raw_content",
            )
            pass

        try:
            return self._build_document(
                page_content=self.get_legacy_doc_content(token),
                doc_token=token,
                node_token="",
                title=token,
                source_url=source_url,
            )
        except RuntimeError:
            return None

    def crawl_documents(self, root_tokens: list[str]) -> list[Document]:
        """Traverse the configured wiki roots and direct doc tokens and return text documents."""
        documents: list[Document] = []
        visited: set[str] = set()
        queue: deque[str] = deque(root_tokens)

        while queue:
            node_token = queue.popleft()
            if node_token in visited:
                continue
            visited.add(node_token)

            try:
                node = self.get_wiki_node(node_token)
            except RuntimeError:
                direct_document = self._build_direct_document_from_token(node_token)
                if direct_document is not None:
                    documents.append(direct_document)
                continue

            space_id = str(node.get("space_id", ""))
            obj_type = str(node.get("obj_type", ""))
            obj_token = str(node.get("obj_token", ""))
            title = str(node.get("title", "Untitled"))
            source_url = str(node.get("url") or f"feishu://wiki/{node_token}")

            if obj_type in SUPPORTED_DOC_TYPES and obj_token:
                raw_content = ""
                if obj_type == "docx":
                    raw_content = self.get_docx_raw_content(obj_token)
                elif obj_type == "doc":
                    raw_content = self.get_legacy_doc_content(obj_token)
                image_refs: list[ImageRef] = []
                if obj_type in FETCHABLE_DOC_TYPES:
                    try:
                        raw_content, image_refs = self.get_docx_markdown_with_media(obj_token)
                    except RuntimeError as exc:
                        log_exception(
                            "feishu_doc_media_fetch_fallback",
                            exc,
                            doc_ref=obj_token,
                            obj_type=obj_type,
                            obj_token=obj_token,
                            fallback="raw_content",
                        )
                        image_refs = []

                document = self._build_document(
                    page_content=raw_content,
                    doc_token=obj_token,
                    node_token=node_token,
                    title=title,
                    source_url=source_url,
                    image_refs=image_refs,
                )
                if document is not None:
                    documents.append(document)

            if space_id:
                for child in self.list_wiki_children(space_id, node_token):
                    child_token = str(child.get("node_token", ""))
                    if child_token and child_token not in visited:
                        queue.append(child_token)

        return documents
