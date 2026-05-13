"""Export runtime log records into eval-case drafts for manual review."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

_BADCASE_EVENTS = {
    "agent_invoke_failed",
    "request_failed",
    "retrieval_failed",
    "tool_failed",
}


def _as_str(value: object) -> str:
    return str(value or "").strip()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    log_path = Path(path)
    records: list[dict[str, Any]] = []
    if not log_path.exists():
        return records
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "request"


def _group_by_request(records: Iterable[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        request_id = _as_str(record.get("request_id")) or f"missing-request-{index + 1}"
        if request_id not in groups:
            order.append(request_id)
            groups[request_id] = []
        groups[request_id].append(record)
    return [(request_id, groups[request_id]) for request_id in order]


def _last_event(records: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    return next((record for record in reversed(records) if record.get("event") == event_name), {})


def _first_non_empty(records: list[dict[str, Any]], keys: list[str]) -> str:
    for record in reversed(records):
        for key in keys:
            value = _as_str(record.get(key))
            if value:
                return value
    return ""


def _is_badcase(records: list[dict[str, Any]], summary: dict[str, Any]) -> bool:
    status = _as_str(summary.get("status"))
    if status and status != "ok":
        return True
    for record in records:
        if record.get("event") in _BADCASE_EVENTS:
            return True
        if _as_str(record.get("level")).upper() == "ERROR":
            return True
    return False


def _bool_from_controller_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = _as_str(value).lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    return None


def _build_draft(request_id: str, records: list[dict[str, Any]], *, index: int) -> dict[str, Any]:
    summary = _last_event(records, "request_summary")
    status = _as_str(summary.get("status")) or ("error" if _is_badcase(records, summary) else "ok")
    thread_id = _first_non_empty(records, ["thread_id"])
    allow_retrieval = summary.get("allow_retrieval")
    if allow_retrieval is None:
        allow_retrieval = _bool_from_controller_value(summary.get("controller_allow_retrieval"))
    event_names = [_as_str(record.get("event")) for record in records if _as_str(record.get("event"))]
    error_record = next(
        (
            record
            for record in reversed(records)
            if record.get("event") in _BADCASE_EVENTS or _as_str(record.get("level")).upper() == "ERROR"
        ),
        {},
    )
    return {
        "id": f"badcase_{_slugify(request_id)}_{index:04d}",
        "user_query": _first_non_empty(records, ["question_preview", "user_message_preview"]),
        "thread_id": thread_id,
        "history": [],
        "expected_intent": "",
        "expected_allow_retrieval": None,
        "expected_rewrite_query": "",
        "expected_answer_must_include": [],
        "expected_answer_must_not_include": [],
        "expected_source_titles": [],
        "expected_source_uris": [],
        "expected_match_status": "",
        "reference_answer": "",
        "tags": ["badcase", "draft", f"status:{status}"],
        "actual_intent": _as_str(summary.get("intent") or summary.get("controller_intent")),
        "actual_allow_retrieval": allow_retrieval if isinstance(allow_retrieval, bool) else None,
        "actual_rewrite_query": _as_str(summary.get("rewrite_query") or summary.get("rewrite_query_preview")),
        "actual_answer": _first_non_empty(records, ["answer_preview", "output_preview"]),
        "metadata": {
            "request_id": request_id,
            "thread_id": thread_id,
            "status": status,
            "error_type": _as_str(summary.get("error_type") or error_record.get("error_type")),
            "error_message": _as_str(summary.get("error_message") or error_record.get("error_message") or error_record.get("error")),
            "total_ms": summary.get("total_ms"),
            "intent_ms": summary.get("intent_ms"),
            "retrieval_ms": summary.get("retrieval_ms"),
            "llm_ms": summary.get("llm_ms"),
            "reply_ms": summary.get("reply_ms"),
            "events": event_names,
        },
    }


def extract_badcase_drafts(
    records: Iterable[dict[str, Any]],
    *,
    include_ok: bool = False,
    request_ids: set[str] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Extract eval-case drafts from structured runtime log records."""
    drafts: list[dict[str, Any]] = []
    for request_id, grouped_records in _group_by_request(records):
        if request_ids and request_id not in request_ids:
            continue
        summary = _last_event(grouped_records, "request_summary")
        if not include_ok and not _is_badcase(grouped_records, summary):
            continue
        drafts.append(_build_draft(request_id, grouped_records, index=len(drafts) + 1))
        if limit > 0 and len(drafts) >= limit:
            break
    return drafts


def write_badcase_drafts(
    *,
    log_path: str | Path,
    output_path: str | Path,
    include_ok: bool = False,
    request_ids: set[str] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Read a JSONL log file and write badcase draft JSONL."""
    drafts = extract_badcase_drafts(
        _read_jsonl(log_path),
        include_ok=include_ok,
        request_ids=request_ids,
        limit=limit,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(draft, ensure_ascii=False) for draft in drafts]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return drafts
