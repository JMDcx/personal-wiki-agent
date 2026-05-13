"""JSONL dataset loading for Agent/RAG evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from multimodal_rag_agent.eval.models import EvalCase

_KNOWN_FIELDS = {
    "id",
    "user_query",
    "question",
    "thread_id",
    "language",
    "history",
    "message_context",
    "images",
    "expected_intent",
    "expected_allow_retrieval",
    "should_retrieve",
    "expected_rewrite_query",
    "expected_answer_must_include",
    "expected_answer_points",
    "expected_answer_must_not_include",
    "expected_source_titles",
    "expected_source_uris",
    "expected_match_status",
    "reference_answer",
    "scene",
    "function",
    "tags",
}


def _as_str(value: object) -> str:
    return str(value or "").strip()


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [item for item in (_as_str(item) for item in value) if item]
    return [_as_str(value)] if _as_str(value) else []


def _as_bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _as_history(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    turns: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        turns.append(
            {
                "user_question": _as_str(item.get("user_question")),
                "assistant_answer": _as_str(item.get("assistant_answer")),
            }
        )
    return turns


def _normalize_row(row: dict[str, Any], *, line_number: int, path: Path) -> EvalCase:
    case_id = _as_str(row.get("id"))
    user_query = _as_str(row.get("user_query") or row.get("question"))
    if not case_id:
        raise ValueError(f"Missing id on line {line_number} of {path}")
    if not user_query:
        raise ValueError(f"Missing user_query on line {line_number} of {path}")

    expected_allow = row.get("expected_allow_retrieval")
    if expected_allow is None and "should_retrieve" in row:
        expected_allow = row.get("should_retrieve")

    must_include = row.get("expected_answer_must_include")
    if must_include is None and "expected_answer_points" in row:
        must_include = row.get("expected_answer_points")

    message_context = row.get("message_context") if isinstance(row.get("message_context"), dict) else {}
    metadata = {key: value for key, value in row.items() if key not in _KNOWN_FIELDS}
    return EvalCase(
        id=case_id,
        user_query=user_query,
        thread_id=_as_str(row.get("thread_id")),
        language=_as_str(row.get("language")) or "中文",
        history=_as_history(row.get("history")),
        message_context=dict(message_context),
        images=_as_string_list(row.get("images")),
        expected_intent=_as_str(row.get("expected_intent")),
        expected_allow_retrieval=_as_bool_or_none(expected_allow),
        expected_rewrite_query=_as_str(row.get("expected_rewrite_query")),
        expected_answer_must_include=_as_string_list(must_include),
        expected_answer_must_not_include=_as_string_list(row.get("expected_answer_must_not_include")),
        expected_source_titles=_as_string_list(row.get("expected_source_titles")),
        expected_source_uris=_as_string_list(row.get("expected_source_uris")),
        expected_match_status=_as_str(row.get("expected_match_status")),
        reference_answer=_as_str(row.get("reference_answer")),
        scene=_as_str(row.get("scene")),
        function=_as_str(row.get("function")),
        tags=_as_string_list(row.get("tags")),
        metadata=metadata,
    )


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load and normalize a JSONL eval dataset."""
    dataset_path = Path(path)
    rows: list[EvalCase] = []
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object on line {line_number} of {dataset_path}")
        rows.append(_normalize_row(row, line_number=line_number, path=dataset_path))
    return rows


def select_cases(cases: list[EvalCase], *, ids: set[str] | None = None, limit: int = 0) -> list[EvalCase]:
    """Filter cases by id and optional limit while preserving dataset order."""
    selected = [case for case in cases if not ids or case.id in ids]
    if limit > 0:
        return selected[:limit]
    return selected
