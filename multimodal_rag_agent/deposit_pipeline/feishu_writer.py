"""Write markdown knowledge drafts into Feishu Docs and Wiki."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from time import perf_counter
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


@dataclass(slots=True)
class FeishuWriteResult:
    document_token: str
    document_url: str
    wiki_node_token: str = ""
    raw_payload: dict[str, Any] | None = None


class FeishuDepositWriter:
    """Write markdown through either the direct OpenAPI path or lark-cli."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        feishu_client: Any | None = None,
        request_timeout: int | None = None,
        cli_runner: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.feishu_client = feishu_client or self._build_feishu_client()
        self.request_timeout = request_timeout or self.settings.feishu_request_timeout
        self.cli_runner = cli_runner or subprocess.run

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
        backend = (self.settings.feishu_deposit_write_backend or "lark_cli").strip().lower()
        log_event(
            "feishu_write_started",
            title=title,
            backend=backend,
            target_space_id=target_space_id,
            has_parent_node=bool(target_parent_node_token),
        )
        try:
            if backend == "lark_cli":
                result = self._write_markdown_via_lark_cli(
                    title=title,
                    markdown_content=markdown_content,
                    target_space_id=target_space_id,
                    target_parent_node_token=target_parent_node_token,
                )
            elif backend == "openapi":
                result = self._write_markdown_via_openapi(
                    title=title,
                    markdown_content=markdown_content,
                    target_space_id=target_space_id,
                    target_parent_node_token=target_parent_node_token,
                )
            else:
                raise RuntimeError(f"Unsupported FEISHU_DEPOSIT_WRITE_BACKEND: {backend}")

            elapsed_ms = (perf_counter() - started_at) * 1000
            log_event(
                "feishu_write_completed",
                title=title,
                backend=backend,
                document_token=result.document_token,
                has_wiki_node=bool(result.wiki_node_token),
                duration_ms=round(elapsed_ms, 1),
            )
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_exception(
                "feishu_write_failed",
                exc,
                title=title,
                backend=backend,
                duration_ms=round(elapsed_ms, 1),
            )
            raise

    def _write_markdown_via_lark_cli(
        self,
        *,
        title: str,
        markdown_content: str,
        target_space_id: str,
        target_parent_node_token: str,
    ) -> FeishuWriteResult:
        if not shutil.which("lark-cli"):
            raise RuntimeError("lark-cli is not installed or not in PATH.")

        profile = self.settings.feishu_lark_cli_profile.strip() or "feishu-wiki-rag-agent"
        sanitized_title = self._sanitize_title(title)
        command = [
            "lark-cli",
            "docs",
            "+create",
            "--profile",
            profile,
            "--as",
            "bot",
            "--title",
            sanitized_title,
            "--markdown",
            markdown_content,
        ]
        if target_parent_node_token:
            command.extend(["--wiki-node", target_parent_node_token])
        elif target_space_id:
            command.extend(["--wiki-space", target_space_id])

        result = self.cli_runner(command, capture_output=True, text=True, timeout=self.request_timeout)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"lark-cli docs +create failed: {detail}")

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("lark-cli docs +create returned empty output.")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"lark-cli docs +create returned non-JSON output: {stdout}") from exc

        if not payload.get("ok"):
            error = payload.get("error") or {}
            message = error.get("message") or stdout
            raise RuntimeError(f"lark-cli docs +create failed: {message}")

        data = payload.get("data") or {}
        document_token = str(data.get("doc_id") or data.get("document_id") or "").strip()
        document_url = str(data.get("doc_url") or data.get("url") or "").strip()
        wiki_node_token = self._extract_token_from_url(document_url)
        if not document_token:
            raise RuntimeError(f"lark-cli docs +create succeeded but doc_id is missing: {stdout}")
        return FeishuWriteResult(
            document_token=document_token,
            document_url=document_url,
            wiki_node_token=wiki_node_token,
            raw_payload=payload,
        )

    def _write_markdown_via_openapi(
        self,
        *,
        title: str,
        markdown_content: str,
        target_space_id: str,
        target_parent_node_token: str,
    ) -> FeishuWriteResult:
        file_token = self._upload_markdown_file(
            file_name=self._sanitize_upload_file_name(f"{title}.md"),
            markdown_content=markdown_content,
        )
        import_result = self._create_import_task(file_token=file_token, title=title)
        move_result = self._move_doc_to_wiki(
            document_token=import_result.document_token,
            target_space_id=target_space_id,
            target_parent_node_token=target_parent_node_token,
        )
        return FeishuWriteResult(
            document_token=import_result.document_token,
            document_url=move_result.document_url or import_result.document_url,
            wiki_node_token=move_result.wiki_node_token,
            raw_payload={
                "import": import_result.raw_payload or {},
                "move": move_result.raw_payload or {},
            },
        )

    def _upload_markdown_file(self, *, file_name: str, markdown_content: str) -> str:
        token = self.feishu_client.fetch_tenant_access_token()
        response = requests.post(
            f"{self.settings.feishu_api_base}/open-apis/drive/v1/files/upload_all",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "file_name": file_name,
                "parent_type": "explorer",
                "parent_node": "",
                "size": str(len(markdown_content.encode("utf-8"))),
            },
            files={"file": (file_name, markdown_content.encode("utf-8"), "text/markdown")},
            timeout=self.request_timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if detail:
                raise RuntimeError(f"Feishu upload failed: HTTP {response.status_code} {detail}") from exc
            raise
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
    ) -> FeishuWriteResult:
        payload = self.feishu_client._request(  # noqa: SLF001
            "POST",
            "/open-apis/drive/v1/import_tasks",
            json_body={
                "file_token": file_token,
                "type": "docx",
                "file_extension": "md",
                "file_name": self._sanitize_title(title),
                "point": {
                    "mount_type": 1,
                    "mount_key": "",
                },
            },
        )
        ticket = str(payload.get("data", {}).get("ticket", "")).strip()
        if not ticket:
            raise RuntimeError("Feishu import task did not return a ticket.")
        return self._poll_import_task(ticket, file_token=file_token, title=title)

    def _poll_import_task(
        self,
        ticket: str,
        *,
        file_token: str,
        title: str,
        max_attempts: int = 15,
        poll_interval: float = 1.0,
    ) -> FeishuWriteResult:
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
                raise RuntimeError(
                    "Feishu import failed: "
                    f"title={title}, file_token={file_token}, payload={json.dumps(payload, ensure_ascii=False)}"
                )
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

    @staticmethod
    def _sanitize_upload_file_name(file_name: str) -> str:
        sanitized = re.sub(r'[\\/:*?"<>|]+', "_", file_name).strip().strip(".")
        return sanitized or "deposit.md"

    @staticmethod
    def _sanitize_title(title: str) -> str:
        sanitized = re.sub(r'[\\/:*?"<>|]+', "_", title).strip()
        return sanitized or "未命名沉淀"

    @staticmethod
    def _extract_token_from_url(url: str) -> str:
        value = url.rstrip("/")
        if not value:
            return ""
        return value.split("/")[-1]
