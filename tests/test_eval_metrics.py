from __future__ import annotations

from multimodal_rag_agent.eval.metrics import evaluate_case, evaluate_retrieval_case, summarize_results
from multimodal_rag_agent.eval.models import EvalActual, EvalCase


def test_evaluate_case_passes_when_expectations_match():
    case = EvalCase(
        id="case-1",
        user_query="项目支持哪些能力？",
        expected_intent="kb_search",
        expected_allow_retrieval=True,
        expected_answer_must_include=["飞书", "知识库"],
        expected_answer_must_not_include=["不存在的政策"],
        expected_source_titles=["项目说明"],
        expected_match_status="matched",
    )
    actual = EvalActual(
        answer="项目支持飞书知识库问答。\n来源：项目说明",
        intent="kb_search",
        allow_retrieval=True,
        retrieval_call_count=1,
        sources=[{"title": "项目说明", "source_uri": "https://example.test/doc"}],
        match_status="matched",
        total_ms=120.0,
    )

    result = evaluate_case(case, actual)

    assert result.passed is True
    assert result.failure_reasons == []
    assert result.primary_failure_reason == ""
    assert result.answer_match is True
    assert result.source_hit is True
    assert result.citation_present is True


def test_evaluate_case_reports_retrieval_and_answer_failures():
    case = EvalCase(
        id="case-2",
        user_query="入职流程是什么？",
        expected_intent="kb_search",
        expected_allow_retrieval=True,
        expected_answer_must_include=["提交材料"],
        expected_source_titles=["入职流程"],
        expected_match_status="matched",
    )
    actual = EvalActual(
        answer="当前知识库里没有找到相关内容。",
        intent="kb_search",
        allow_retrieval=True,
        retrieval_call_count=1,
        sources=[{"title": "其他文档"}],
        match_status="no_match",
    )

    result = evaluate_case(case, actual)

    assert result.passed is False
    assert result.primary_failure_reason == "retrieval_miss"
    assert "retrieval_miss" in result.failure_reasons
    assert "match_status_mismatch" in result.failure_reasons
    assert "citation_missing" in result.failure_reasons
    assert "answer_missing_required_point" in result.failure_reasons
    assert result.missing_required == ["提交材料"]


def test_evaluate_case_detects_unexpected_retrieval_for_chat():
    case = EvalCase(
        id="case-3",
        user_query="你好",
        expected_intent="chat",
        expected_allow_retrieval=False,
        expected_answer_must_include=["你好"],
    )
    actual = EvalActual(
        answer="你好，有什么可以帮你？",
        intent="chat",
        allow_retrieval=True,
        retrieval_call_count=1,
    )

    result = evaluate_case(case, actual)

    assert result.passed is False
    assert result.primary_failure_reason == "allow_retrieval_mismatch"
    assert "unexpected_retrieval_called" in result.failure_reasons


def test_summarize_results_groups_quality_and_latency_metrics():
    passing = evaluate_case(
        EvalCase(id="pass", user_query="A", expected_answer_must_include=["ok"]),
        EvalActual(answer="ok", total_ms=100.0),
    )
    failing = evaluate_case(
        EvalCase(id="fail", user_query="B", expected_answer_must_include=["needed"]),
        EvalActual(answer="missing", total_ms=300.0),
    )

    summary = summarize_results("run-1", "cases.jsonl", [passing, failing])

    assert summary["total_cases"] == 2
    assert summary["passed_cases"] == 1
    assert summary["failed_cases"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["failure_reason_counts"]["answer_missing_required_point"] == 1
    assert summary["latency_ms"]["p50"] == 100.0
    assert summary["latency_ms"]["p95"] == 300.0


def test_evaluate_retrieval_case_ignores_agent_intent_and_final_citation():
    case = EvalCase(
        id="retrieval-1",
        user_query="alpha policy",
        expected_intent="kb_search",
        expected_allow_retrieval=True,
        expected_answer_must_include=["required fact"],
        expected_source_titles=["Alpha Handbook"],
        expected_match_status="matched",
    )
    actual = EvalActual(
        answer="retrieved context with required fact",
        allow_retrieval=True,
        retrieval_call_count=1,
        sources=[{"title": "Alpha Handbook", "source_uri": "https://example.test/alpha"}],
        match_status="matched",
        retrieval_ms=12.0,
        total_ms=12.0,
    )

    result = evaluate_retrieval_case(case, actual)

    assert result.passed is True
    assert result.intent_match is None
    assert result.citation_present is None
