"""Case runners for local Agent/RAG evaluation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from feishu_wiki_rag_agent.protocols.tool_models import RetrievalResult
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from protocols.tool_models import RetrievalResult

from multimodal_rag_agent.eval.metrics import evaluate_case, evaluate_retrieval_case
from multimodal_rag_agent.eval.models import EvalActual, EvalCase, EvalResult

AgentRunner = Callable[[EvalCase, "EvalRunContext"], EvalActual | str]
RetrievalRunner = Callable[[EvalCase, "EvalRunContext"], EvalActual]


@dataclass(slots=True)
class EvalRunContext:
    """Stable identifiers passed to the per-case runner."""

    run_id: str
    case_index: int
    thread_id: str
    request_id: str


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "eval"


def read_log_records(path: str | Path) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
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


def _to_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def collect_request_view(records: list[dict[str, Any]], request_id: str) -> dict[str, Any]:
    """Collect request summary and timing events for one request id."""
    matched = [record for record in records if str(record.get("request_id", "")) == request_id]
    summary = next((record for record in reversed(matched) if record.get("event") == "request_summary"), {})
    retrieval_events = [record for record in matched if record.get("event") == "retrieval_completed"]
    generation_event = next((record for record in reversed(matched) if record.get("event") == "generation_completed"), {})
    retrieval_ms_values = [_to_float(record.get("duration_ms")) for record in retrieval_events]
    retrieval_ms_sum = sum(value for value in retrieval_ms_values if value is not None) if retrieval_ms_values else None
    return {
        "summary": summary,
        "events": matched,
        "generation_ms": _to_float(generation_event.get("duration_ms")) if generation_event else None,
        "retrieval_ms_sum": retrieval_ms_sum,
        "retrieval_call_count": len(retrieval_events),
        "tool_call_count": sum(1 for record in matched if record.get("event") == "tool_called"),
    }


def actual_from_request_view(
    *,
    answer: str,
    request_view: dict[str, Any],
    status: str = "ok",
    error: str = "",
    thread_id: str = "",
    request_id: str = "",
) -> EvalActual:
    """Build EvalActual from agent answer plus structured log records."""
    summary = dict(request_view.get("summary") or {})
    retrieval_ms = _to_float(summary.get("retrieval_ms"))
    if retrieval_ms is None:
        retrieval_ms = _to_float(request_view.get("retrieval_ms_sum"))
    allow_retrieval = summary.get("allow_retrieval")
    raw_sources = summary.get("sources")
    sources = raw_sources if isinstance(raw_sources, list) else []
    events = request_view.get("events") if isinstance(request_view.get("events"), list) else []
    tool_completed = next(
        (
            event
            for event in reversed(events)
            if isinstance(event, dict)
            and event.get("event") == "tool_completed"
            and event.get("tool_name") == "search_feishu_knowledge"
        ),
        {},
    )
    match_status = str(summary.get("match_status") or tool_completed.get("match_status") or "")
    return EvalActual(
        answer=answer,
        status=str(summary.get("status") or status),
        error=error,
        intent=str(summary.get("intent", "") or ""),
        allow_retrieval=allow_retrieval if isinstance(allow_retrieval, bool) else None,
        rewrite_query=str(summary.get("rewrite_query", "") or ""),
        retrieval_call_count=int(request_view.get("retrieval_call_count") or 0),
        tool_call_count=int(request_view.get("tool_call_count") or 0),
        sources=[source for source in sources if isinstance(source, dict)],
        match_status=match_status,
        total_ms=_to_float(summary.get("total_ms")),
        intent_ms=_to_float(summary.get("intent_ms")),
        retrieval_ms=retrieval_ms,
        llm_ms=_to_float(summary.get("llm_ms")),
        reply_ms=_to_float(summary.get("reply_ms")),
        thread_id=thread_id,
        request_id=request_id,
    )


def actual_from_prepared_context(
    *,
    query: str,
    prepared: Any,
    duration_ms: float | None = None,
    status: str = "ok",
    error: str = "",
    thread_id: str = "",
    request_id: str = "",
) -> EvalActual:
    """Build EvalActual from RAG prepare_context output for retrieval-only scoring."""
    retrieval_result = RetrievalResult.from_prepared_context(query, prepared)
    return EvalActual(
        answer=retrieval_result.context,
        status=status,
        error=error,
        allow_retrieval=True,
        retrieval_call_count=1 if status == "ok" and not error else 0,
        sources=[source for source in retrieval_result.sources if isinstance(source, dict)],
        match_status=retrieval_result.match_status,
        total_ms=duration_ms,
        retrieval_ms=duration_ms,
        thread_id=thread_id,
        request_id=request_id,
    )


def run_cases(
    cases: list[EvalCase],
    *,
    run_id: str,
    agent_runner: AgentRunner,
    sleep_seconds: float = 0.0,
) -> list[EvalResult]:
    """Run cases with an injected agent runner and return scored results."""
    results: list[EvalResult] = []
    for index, case in enumerate(cases, start=1):
        thread_id = case.thread_id or f"eval:{run_id}:{slugify(case.id)}"
        context = EvalRunContext(
            run_id=run_id,
            case_index=index,
            thread_id=thread_id,
            request_id=f"thread:{thread_id}",
        )
        try:
            output = agent_runner(case, context)
            actual = output if isinstance(output, EvalActual) else EvalActual(answer=str(output))
            if not actual.thread_id:
                actual.thread_id = context.thread_id
            if not actual.request_id:
                actual.request_id = context.request_id
        except Exception as exc:  # noqa: BLE001
            actual = EvalActual(
                status="error",
                error=str(exc),
                thread_id=context.thread_id,
                request_id=context.request_id,
            )
        results.append(evaluate_case(case, actual))
        if sleep_seconds > 0 and index < len(cases):
            time.sleep(sleep_seconds)
    return results


def run_retrieval_cases(
    cases: list[EvalCase],
    *,
    run_id: str,
    retrieval_runner: RetrievalRunner,
    sleep_seconds: float = 0.0,
) -> list[EvalResult]:
    """Run retrieval-only cases and score retrieval quality without generation."""
    results: list[EvalResult] = []
    for index, case in enumerate(cases, start=1):
        thread_id = case.thread_id or f"eval:{run_id}:{slugify(case.id)}"
        context = EvalRunContext(
            run_id=run_id,
            case_index=index,
            thread_id=thread_id,
            request_id=f"thread:{thread_id}",
        )
        try:
            actual = retrieval_runner(case, context)
            if not actual.thread_id:
                actual.thread_id = context.thread_id
            if not actual.request_id:
                actual.request_id = context.request_id
        except Exception as exc:  # noqa: BLE001
            actual = EvalActual(
                status="error",
                error=str(exc),
                thread_id=context.thread_id,
                request_id=context.request_id,
            )
        results.append(evaluate_retrieval_case(case, actual))
        if sleep_seconds > 0 and index < len(cases):
            time.sleep(sleep_seconds)
    return results
