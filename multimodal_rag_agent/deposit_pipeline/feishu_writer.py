"""Write markdown knowledge drafts into Feishu Docs and Wiki."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
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

from multimodal_rag_agent.deposit_pipeline.models import InlineImage


@dataclass(slots=True)
class FeishuWriteResult:
    document_token: str
    document_url: str
    wiki_node_token: str = ""
    inline_rendered_count: int = 0
    fallback_appended_count: int = 0
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
        image_paths: list[str] | None = None,
        inline_images: list[InlineImage] | None = None,
        target_space_id: str,
        target_parent_node_token: str,
    ) -> FeishuWriteResult:
        started_at = perf_counter()
        backend = (self.settings.feishu_deposit_write_backend or "lark_cli").strip().lower()
        prepared_markdown, unresolved_inline_images = self._prepare_markdown_for_feishu(
            markdown_content,
            inline_images=inline_images or [],
        )
        inline_rendered_count = len((inline_images or [])) - len(unresolved_inline_images)
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
                    markdown_content=prepared_markdown,
                    target_space_id=target_space_id,
                    target_parent_node_token=target_parent_node_token,
                )
            elif backend == "openapi":
                result = self._write_markdown_via_openapi(
                    title=title,
                    markdown_content=prepared_markdown,
                    target_space_id=target_space_id,
                    target_parent_node_token=target_parent_node_token,
                )
            else:
                raise RuntimeError(f"Unsupported FEISHU_DEPOSIT_WRITE_BACKEND: {backend}")
            inserted_image_count = self._insert_images_if_needed(
                document_ref=result.document_token or result.document_url,
                image_paths=self._fallback_image_paths(
                    image_paths=image_paths or [],
                    unresolved_inline_images=unresolved_inline_images,
                ),
            )

            elapsed_ms = (perf_counter() - started_at) * 1000
            log_event(
                "feishu_write_completed",
                title=title,
                backend=backend,
                document_token=result.document_token,
                has_wiki_node=bool(result.wiki_node_token),
                inline_rendered_count=inline_rendered_count,
                fallback_appended_count=inserted_image_count,
                inserted_image_count=inserted_image_count,
                duration_ms=round(elapsed_ms, 1),
            )
            result.inline_rendered_count = inline_rendered_count
            result.fallback_appended_count = inserted_image_count
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

    def _prepare_markdown_for_feishu(
        self,
        markdown_content: str,
        *,
        inline_images: list[InlineImage],
    ) -> tuple[str, list[InlineImage]]:
        rendered = markdown_content
        unresolved: list[InlineImage] = []
        for image in inline_images:
            replacement = self._feishu_markdown_image(image.original_ref)
            if replacement:
                rendered = rendered.replace(image.placeholder, replacement)
            else:
                unresolved.append(image)
        return rendered, unresolved

    @staticmethod
    def _feishu_markdown_image(original_ref: str) -> str:
        candidate = str(original_ref or "").strip()
        if candidate.startswith(("http://", "https://")):
            return f"![]({candidate})"
        return ""

    @staticmethod
    def _fallback_image_paths(*, image_paths: list[str], unresolved_inline_images: list[InlineImage]) -> list[str]:
        if not unresolved_inline_images:
            return []
        unresolved_paths = [image.image_path.strip() for image in unresolved_inline_images if image.image_path.strip()]
        if unresolved_paths:
            return list(dict.fromkeys(unresolved_paths))
        return [path for path in image_paths if str(path).strip()]

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
        if not document_token and str(data.get("status", "")).strip().lower() == "running":
            task_id = str(data.get("task_id", "")).strip()
            if task_id:
                polled_token, polled_url, poll_payload = self._poll_lark_cli_create_task(
                    title=sanitized_title,
                    target_space_id=target_space_id,
                    target_parent_node_token=target_parent_node_token,
                )
                if polled_token or polled_url:
                    document_token = polled_token or document_token
                    document_url = polled_url or document_url
                    payload = {
                        **payload,
                        "task_poll": poll_payload,
                    }
        wiki_node_token = self._extract_token_from_url(document_url)
        if not document_token:
            raise RuntimeError(f"lark-cli docs +create succeeded but doc_id is missing: {stdout}")
        return FeishuWriteResult(
            document_token=document_token,
            document_url=document_url,
            wiki_node_token=wiki_node_token,
            inline_rendered_count=0,
            fallback_appended_count=0,
            raw_payload=payload,
        )

    def _poll_lark_cli_create_task(
        self,
        *,
        title: str,
        target_space_id: str,
        target_parent_node_token: str,
        max_attempts: int = 20,
        poll_interval: float = 1.0,
    ) -> tuple[str, str, dict[str, Any]]:
        last_payload: dict[str, Any] = {"strategy": "wiki_lookup"}
        for _ in range(max_attempts):
            lookup_token, lookup_url = self._find_created_doc_in_wiki(
                title=title,
                target_space_id=target_space_id,
                target_parent_node_token=target_parent_node_token,
            )
            if lookup_token or lookup_url:
                last_payload = {
                    "strategy": "wiki_lookup",
                    "document_token": lookup_token,
                    "document_url": lookup_url,
                }
                return lookup_token, lookup_url, last_payload
            time.sleep(poll_interval)
        raise RuntimeError(f"lark-cli create_doc task timed out: {json.dumps(last_payload, ensure_ascii=False)}")

    def _find_created_doc_in_wiki(
        self,
        *,
        title: str,
        target_space_id: str,
        target_parent_node_token: str,
    ) -> tuple[str, str]:
        if not target_space_id or not target_parent_node_token:
            return "", ""
        try:
            children = self.feishu_client.list_wiki_children(target_space_id, target_parent_node_token)
        except Exception:
            return "", ""
        normalized_title = self._sanitize_title(title)
        for child in children:
            child_title = str(child.get("title", "")).strip()
            if child_title != normalized_title:
                continue
            document_url = str(child.get("url") or "").strip()
            document_token = str(
                child.get("obj_token")
                or child.get("doc_id")
                or child.get("document_id")
                or child.get("token")
                or ""
            ).strip()
            if not document_token and document_url:
                document_token = self._extract_token_from_url(document_url)
            if document_token or document_url:
                return document_token, document_url
        return "", ""

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
                    inline_rendered_count=0,
                    fallback_appended_count=0,
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
            inline_rendered_count=0,
            fallback_appended_count=0,
            raw_payload=payload,
        )

    def _insert_images_if_needed(self, *, document_ref: str, image_paths: list[str]) -> int:
        valid_paths = [path for path in image_paths if Path(path).exists() and Path(path).is_file()]
        if not valid_paths or not document_ref:
            return 0
        if not shutil.which("lark-cli"):
            log_event(
                "feishu_image_insert_skipped",
                reason="lark_cli_unavailable",
                document_ref=document_ref,
                image_count=len(valid_paths),
            )
            return 0

        profile = self.settings.feishu_lark_cli_profile.strip() or "feishu-wiki-rag-agent"
        inserted = 0
        for image_path in valid_paths:
            path = Path(image_path)
            command = [
                "lark-cli",
                "docs",
                "+media-insert",
                "--profile",
                profile,
                "--as",
                "bot",
                "--doc",
                document_ref,
                "--file",
                str(path),
                "--type",
                "image",
                "--caption",
                path.name,
            ]
            result = self.cli_runner(command, capture_output=True, text=True, timeout=self.request_timeout)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
                raise RuntimeError(f"lark-cli docs +media-insert failed: {detail}")
            inserted += 1
            log_event(
                "feishu_image_inserted",
                document_ref=document_ref,
                image_path=str(path),
                caption=path.name,
            )
        return inserted

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
