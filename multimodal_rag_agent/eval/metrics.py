"""Deterministic scoring for Agent/RAG evaluation results."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from statistics import mean
from typing import Any

from multimodal_rag_agent.eval.models import EvalActual, EvalCase, EvalResult

_PRIMARY_REASON_PRIORITY = [
    "runtime_error",
    "intent_mismatch",
    "allow_retrieval_mismatch",
    "retrieval_not_called",
    "unexpected_retrieval_called",
    "retrieval_miss",
    "match_status_mismatch",
    "citation_missing",
    "citation_source_mismatch",
    "answer_missing_required_point",
    "forbidden_claim",
    "no_match_answer_incorrect",
    "latency_regression",
]

_NO_MATCH_PHRASES = ["未找到", "没有找到", "暂无", "不知道", "无法找到", "当前知识库"]


def _collapse_ws(text: str) -> str:
    return " ".join((text or "").split())


def _normalize_for_match(text: str) -> str:
    return _collapse_ws(text).strip().lower()


def _contains_all(answer: str, required: list[str]) -> list[str]:
    normalized_answer = _normalize_for_match(answer)
    return [item for item in required if _normalize_for_match(item) not in normalized_answer]


def _contains_any(answer: str, forbidden: list[str]) -> list[str]:
    normalized_answer = _normalize_for_match(answer)
    return [item for item in forbidden if _normalize_for_match(item) in normalized_answer]


def _source_values(sources: list[dict[str, Any]], key: str) -> list[str]:
    return [str(source.get(key, "") or "") for source in sources]


def _matches_expected(actual: str, expected: str) -> bool:
    actual_norm = _normalize_for_match(actual)
    expected_norm = _normalize_for_match(expected)
    if not actual_norm or not expected_norm:
        return False
    return actual_norm == expected_norm or expected_norm in actual_norm or actual_norm in expected_norm


def _has_source_hit(case: EvalCase, actual: EvalActual) -> bool | None:
    expected_titles = case.expected_source_titles
    expected_uris = case.expected_source_uris
    if not expected_titles and not expected_uris:
        return None

    actual_titles = _source_values(actual.sources, "title")
    actual_uris = _source_values(actual.sources, "source_uri")
    title_hit = any(_matches_expected(actual_title, expected) for expected in expected_titles for actual_title in actual_titles)
    uri_hit = any(_matches_expected(actual_uri, expected) for expected in expected_uris for actual_uri in actual_uris)
    return title_hit or uri_hit


def _has_citation(answer: str) -> bool:
    normalized = answer or ""
    return "来源：" in normalized or "来源:" in normalized or "source:" in normalized.lower()


def _citation_hits_expected_source(case: EvalCase, answer: str) -> bool | None:
    expected = [*case.expected_source_titles, *case.expected_source_uris]
    if not expected:
        return None
    return any(_normalize_for_match(item) in _normalize_for_match(answer) for item in expected)


def _requires_citation(case: EvalCase) -> bool:
    if case.expected_allow_retrieval is not True:
        return False
    return case.expected_match_status != "no_match"


def _no_match_answer_is_clear(answer: str) -> bool:
    return any(phrase in answer for phrase in _NO_MATCH_PHRASES)


def _primary_failure_reason(reasons: list[str]) -> str:
    for reason in _PRIMARY_REASON_PRIORITY:
        if reason in reasons:
            return reason
    return reasons[0] if reasons else ""


def evaluate_case(case: EvalCase, actual: EvalActual, *, require_citation: bool = True) -> EvalResult:
    """Score one eval case using deterministic checks."""
    failure_reasons: list[str] = []

    intent_match: bool | None = None
    if case.expected_intent:
        intent_match = case.expected_intent == actual.intent
        if not intent_match:
            failure_reasons.append("intent_mismatch")

    allow_retrieval_match: bool | None = None
    if case.expected_allow_retrieval is not None:
        allow_retrieval_match = case.expected_allow_retrieval == actual.allow_retrieval
        if not allow_retrieval_match:
            failure_reasons.append("allow_retrieval_mismatch")

    rewrite_match: bool | None = None
    if case.expected_rewrite_query:
        rewrite_match = _normalize_for_match(case.expected_rewrite_query) == _normalize_for_match(actual.rewrite_query)
        if not rewrite_match:
            failure_reasons.append("rewrite_mismatch")

    if case.expected_allow_retrieval is True and actual.retrieval_call_count <= 0:
        failure_reasons.append("retrieval_not_called")
    if case.expected_allow_retrieval is False and actual.retrieval_call_count > 0:
        failure_reasons.append("unexpected_retrieval_called")

    source_hit = _has_source_hit(case, actual)
    if source_hit is False:
        failure_reasons.append("retrieval_miss")

    match_status_match: bool | None = None
    if case.expected_match_status:
        match_status_match = case.expected_match_status == actual.match_status
        if not match_status_match:
            failure_reasons.append("match_status_mismatch")

    citation_present: bool | None = None
    citation_source_hit: bool | None = None
    if require_citation and _requires_citation(case):
        citation_present = _has_citation(actual.answer)
        if not citation_present:
            failure_reasons.append("citation_missing")
        else:
            citation_source_hit = _citation_hits_expected_source(case, actual.answer)
            if citation_source_hit is False:
                failure_reasons.append("citation_source_mismatch")

    missing_required = _contains_all(actual.answer, case.expected_answer_must_include)
    forbidden_hits = _contains_any(actual.answer, case.expected_answer_must_not_include)
    answer_match = not missing_required and not forbidden_hits and bool((actual.answer or "").strip())
    if missing_required:
        failure_reasons.append("answer_missing_required_point")
    if forbidden_hits:
        failure_reasons.append("forbidden_claim")

    if case.expected_match_status == "no_match" and not _no_match_answer_is_clear(actual.answer):
        failure_reasons.append("no_match_answer_incorrect")

    if actual.status != "ok" or actual.error:
        failure_reasons.insert(0, "runtime_error")

    deduped_reasons = list(dict.fromkeys(failure_reasons))
    primary = _primary_failure_reason(deduped_reasons)
    return EvalResult(
        case_id=case.id,
        user_query=case.user_query,
        passed=not deduped_reasons,
        status=actual.status,
        error=actual.error,
        expected_intent=case.expected_intent,
        actual_intent=actual.intent,
        intent_match=intent_match,
        expected_allow_retrieval=case.expected_allow_retrieval,
        actual_allow_retrieval=actual.allow_retrieval,
        allow_retrieval_match=allow_retrieval_match,
        expected_rewrite_query=case.expected_rewrite_query,
        actual_rewrite_query=actual.rewrite_query,
        rewrite_match=rewrite_match,
        answer_match=answer_match,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        expected_source_titles=case.expected_source_titles,
        expected_source_uris=case.expected_source_uris,
        actual_sources=actual.sources,
        source_hit=source_hit,
        expected_match_status=case.expected_match_status,
        actual_match_status=actual.match_status,
        match_status_match=match_status_match,
        citation_present=citation_present,
        citation_source_hit=citation_source_hit,
        retrieval_call_count=actual.retrieval_call_count,
        tool_call_count=actual.tool_call_count,
        total_ms=actual.total_ms,
        intent_ms=actual.intent_ms,
        retrieval_ms=actual.retrieval_ms,
        llm_ms=actual.llm_ms,
        reply_ms=actual.reply_ms,
        actual_answer=actual.answer,
        reference_answer=case.reference_answer,
        failure_reasons=deduped_reasons,
        primary_failure_reason=primary,
        thread_id=actual.thread_id,
        request_id=actual.request_id,
        tags=case.tags,
    )


def evaluate_retrieval_case(case: EvalCase, actual: EvalActual) -> EvalResult:
    """Score retrieval-only output without grading agent intent or final-answer citations."""
    retrieval_case = replace(case, expected_intent="")
    return evaluate_case(retrieval_case, actual, require_citation=False)


def _accuracy(values: list[bool | None]) -> float | None:
    scored = [value for value in values if value is not None]
    if not scored:
        return None
    return sum(1 for value in scored if value) / len(scored)


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((percentile / 100) * len(ordered)) - 1))
    return ordered[index]


def summarize_results(run_id: str, dataset_path: str, results: list[EvalResult]) -> dict[str, Any]:
    """Build aggregate metrics for one eval run."""
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    latencies = [result.total_ms for result in results if result.total_ms is not None]
    reason_counts = Counter(reason for result in results for reason in result.failure_reasons)
    status_counts = Counter(result.status for result in results)
    return {
        "run_id": run_id,
        "dataset_path": dataset_path,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "intent_accuracy": _accuracy([result.intent_match for result in results]),
        "allow_retrieval_accuracy": _accuracy([result.allow_retrieval_match for result in results]),
        "rewrite_accuracy": _accuracy([result.rewrite_match for result in results]),
        "answer_match_rate": _accuracy([result.answer_match for result in results]),
        "source_hit_rate": _accuracy([result.source_hit for result in results]),
        "match_status_accuracy": _accuracy([result.match_status_match for result in results]),
        "citation_presence_rate": _accuracy([result.citation_present for result in results]),
        "citation_source_hit_rate": _accuracy([result.citation_source_hit for result in results]),
        "latency_ms": {
            "avg": mean(latencies) if latencies else None,
            "p50": _nearest_rank(latencies, 50),
            "p95": _nearest_rank(latencies, 95),
            "max": max(latencies) if latencies else None,
        },
        "failed_case_ids": [result.case_id for result in results if not result.passed],
    }
