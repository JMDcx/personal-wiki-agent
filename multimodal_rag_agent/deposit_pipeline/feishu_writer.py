"""Write markdown knowledge drafts into Feishu Docs and Wiki."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import requests

try:
    from feishu_wiki_rag_agent.config import Settings, get_settings
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import Settings, get_settings
    from observability.events import log_event, log_exception


@dataclass(slots=True)
class FeishuWriteResult:
    document_token: str
    document_url: str
    wiki_node_token: str = ""
    raw_payload: dict[str, Any] | None = None


class FeishuDepositWriter:
    """Import markdown into Feishu Docs, then move it into Wiki."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        feishu_client: Any | None = None,
        request_timeout: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.feishu_client = feishu_client or self._build_feishu_client()
        self.request_timeout = request_timeout or self.settings.feishu_request_timeout

    def _build_feishu_client(self) -> Any:
        try:
            from feishu_wiki_rag_agent.channel.feishu.feishu_client import FeishuClient
        except ModuleNotFoundError:  # pragma: no cover - source tree fallback
            import importlib.util
            from pathlib import Path
            import sys

            module_path = Path(__file__).resolve().parents[2] / "channel" / "feishu" / "feishu_client.py"
            spec = importlib.util.spec_from_file_location("local_feishu_client", module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to load local Feishu client module.")
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("local_feishu_client", module)
            spec.loader.exec_module(module)
            FeishuClient = getattr(module, "FeishuClient")
        return FeishuClient(self.settings)

    def write_markdown(
        self,
        *,
        title: str,
        markdown_content: str,
        target_space_id: str,
        target_parent_node_token: str,
    ) -> FeishuWriteResult:
        started_at = perf_counter()
        log_event(
            "feishu_write_started",
            title=title,
            target_space_id=target_space_id,
            has_parent_node=bool(target_parent_node_token),
        )
        try:
            file_token = self._upload_markdown_file(file_name=f"{title}.md", markdown_content=markdown_content)
            import_result = self._create_import_task(
                file_token=file_token,
                title=title,
                target_parent_node_token=target_parent_node_token,
            )
            move_result = self._move_doc_to_wiki(
                document_token=import_result.document_token,
                target_space_id=target_space_id,
                target_parent_node_token=target_parent_node_token,
            )
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_event(
                "feishu_write_completed",
                title=title,
                document_token=import_result.document_token,
                has_wiki_node=bool(move_result.wiki_node_token),
                duration_ms=round(elapsed_ms, 1),
            )
            return FeishuWriteResult(
                document_token=import_result.document_token,
                document_url=import_result.document_url,
                wiki_node_token=move_result.wiki_node_token,
                raw_payload={
                    "import": import_result.raw_payload or {},
                    "move": move_result.raw_payload or {},
                },
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_exception(
                "feishu_write_failed",
                exc,
                title=title,
                duration_ms=round(elapsed_ms, 1),
            )
            raise

    def _upload_markdown_file(self, *, file_name: str, markdown_content: str) -> str:
        token = self.feishu_client.fetch_tenant_access_token()
        response = requests.post(
            f"{self.settings.feishu_api_base}/open-apis/drive/v1/files/upload_all",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "file_name": file_name,
                "parent_type": "explorer",
                "parent_node": "",
                "size": str(len(markdown_content.encode('utf-8'))),
            },
            files={"file": (file_name, markdown_content.encode("utf-8"), "text/markdown")},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code", 0) != 0:
            raise RuntimeError(f"Feishu upload failed: {payload.get('msg', 'unknown error')}")
        file_token = str(payload.get("data", {}).get("file_token", "")).strip()
        if not file_token:
            raise RuntimeError("Feishu upload succeeded but file_token is missing.")
        return file_token

    def _create_import_task(
        self,
        *,
        file_token: str,
        title: str,
        target_parent_node_token: str,
    ) -> FeishuWriteResult:
        payload = self.feishu_client._request(  # noqa: SLF001
            "POST",
            "/open-apis/drive/v1/import_tasks",
            json_body={
                "file_token": file_token,
                "type": "docx",
                "file_extension": "md",
                "file_name": title,
                "point": {
                    "mount_type": 1,
                    "mount_key": target_parent_node_token,
                },
            },
        )
        ticket = str(payload.get("data", {}).get("ticket", "")).strip()
        if not ticket:
            raise RuntimeError("Feishu import task did not return a ticket.")
        return self._poll_import_task(ticket)

    def _poll_import_task(self, ticket: str, *, max_attempts: int = 15, poll_interval: float = 1.0) -> FeishuWriteResult:
        last_payload: dict[str, Any] | None = None
        for _ in range(max_attempts):
            payload = self.feishu_client._request("GET", f"/open-apis/drive/v1/import_tasks/{ticket}")
            last_payload = payload
            data = payload.get("data", {})
            result = data.get("result", {})
            job_status = int(result.get("job_status", data.get("job_status", -1)) or -1)
            if job_status == 0:
                document_token = str(
                    result.get("token") or result.get("obj_token") or result.get("document_token") or ""
                ).strip()
                document_url = str(result.get("url") or result.get("doc_url") or "").strip()
                if not document_token and document_url:
                    document_token = document_url.rstrip("/").split("/")[-1]
                if not document_token:
                    raise RuntimeError("Feishu import completed but document token is missing.")
                return FeishuWriteResult(
                    document_token=document_token,
                    document_url=document_url,
                    raw_payload=payload,
                )
            if job_status == 2:
                raise RuntimeError(f"Feishu import failed: {json.dumps(payload, ensure_ascii=False)}")
            time.sleep(poll_interval)
        raise RuntimeError(f"Feishu import task timed out: {json.dumps(last_payload or {}, ensure_ascii=False)}")

    def _move_doc_to_wiki(
        self,
        *,
        document_token: str,
        target_space_id: str,
        target_parent_node_token: str,
    ) -> FeishuWriteResult:
        payload = self.feishu_client._request(
            "POST",
            f"/open-apis/wiki/v2/spaces/{target_space_id}/nodes/move_docs_to_wiki",
            json_body={
                "parent_wiki_token": target_parent_node_token,
                "obj_type": "docx",
                "obj_token": document_token,
            },
        )
        data = payload.get("data", {})
        wiki_token = str(data.get("wiki_token") or data.get("node_token") or "").strip()
        task_id = str(data.get("task_id", "")).strip()
        if not wiki_token and task_id:
            wiki_token = self._poll_wiki_task(task_id)
        return FeishuWriteResult(
            document_token=document_token,
            document_url=f"{self.settings.feishu_api_base.replace('open.', '')}/docx/{document_token}",
            wiki_node_token=wiki_token,
            raw_payload=payload,
        )

    def _poll_wiki_task(self, task_id: str, *, max_attempts: int = 15, poll_interval: float = 1.0) -> str:
        last_payload: dict[str, Any] | None = None
        for _ in range(max_attempts):
            payload = self.feishu_client._request("GET", f"/open-apis/wiki/v2/tasks/{task_id}")
            last_payload = payload
            data = payload.get("data", {})
            move_results = data.get("move_result") or data.get("task_result", {}).get("move_result") or []
            if move_results:
                wiki_token = str(move_results[0].get("wiki_token") or move_results[0].get("node_token") or "").strip()
                if wiki_token:
                    return wiki_token
            if str(data.get("status", "")).lower() in {"success", "done"}:
                break
            time.sleep(poll_interval)
        raise RuntimeError(f"Wiki move task timed out: {json.dumps(last_payload or {}, ensure_ascii=False)}")
