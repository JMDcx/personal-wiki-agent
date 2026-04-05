"""Feishu API client for websocket messaging and Wiki/Docs crawling."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import requests

from feishu_wiki_rag_agent.config import Settings, get_settings

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:  # pragma: no cover - local fallback
    @dataclass
    class Document:
        page_content: str
        metadata: dict[str, Any]


SUPPORTED_DOC_TYPES = {"doc", "docx"}


@dataclass
class _TokenCache:
    value: str = ""
    expires_at: float = 0.0


class FeishuClient:
    """Thin wrapper around the Feishu open platform APIs used by the example."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._token_cache = _TokenCache()

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
            },
        )

    def _build_direct_document_from_token(self, token: str) -> Document | None:
        """Try to interpret a root token as a direct docx/doc token."""
        source_url = f"feishu://document/{token}"
        try:
            return self._build_document(
                page_content=self.get_docx_raw_content(token),
                doc_token=token,
                node_token="",
                title=token,
                source_url=source_url,
            )
        except RuntimeError:
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

                document = self._build_document(
                    page_content=raw_content,
                    doc_token=obj_token,
                    node_token=node_token,
                    title=title,
                    source_url=source_url,
                )
                if document is not None:
                    documents.append(document)

            if space_id:
                for child in self.list_wiki_children(space_id, node_token):
                    child_token = str(child.get("node_token", ""))
                    if child_token and child_token not in visited:
                        queue.append(child_token)

        return documents
