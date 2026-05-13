from __future__ import annotations

from types import SimpleNamespace

from multimodal_rag_agent.eval.models import EvalActual, EvalCase
from multimodal_rag_agent.eval.runner import (
    EvalRunContext,
    actual_from_prepared_context,
    actual_from_request_view,
    run_cases,
    run_retrieval_cases,
)


def test_run_cases_evaluates_fake_agent_and_keeps_going_after_error():
    cases = [
        EvalCase(id="ok", user_query="A", expected_answer_must_include=["answer"]),
        EvalCase(id="boom", user_query="B"),
    ]

    def fake_agent(case: EvalCase, context: EvalRunContext) -> EvalActual:
        if case.id == "boom":
            raise RuntimeError("provider failed")
        assert context.thread_id.startswith("eval:run-1:")
        return EvalActual(answer="answer", total_ms=5.0)

    results = run_cases(cases, run_id="run-1", agent_runner=fake_agent)

    assert [result.case_id for result in results] == ["ok", "boom"]
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].status == "error"
    assert results[1].primary_failure_reason == "runtime_error"


def test_actual_from_request_view_extracts_summary_sources_and_timings():
    actual = actual_from_request_view(
        answer="答案\n来源：项目说明",
        request_view={
            "summary": {
                "status": "ok",
                "intent": "kb_search",
                "allow_retrieval": True,
                "rewrite_query": "项目能力",
                "match_status": "matched",
                "sources": [{"title": "项目说明", "source_uri": "https://example.test/doc"}],
                "total_ms": 42.0,
            },
            "retrieval_call_count": 1,
            "tool_call_count": 1,
        },
        thread_id="eval:run:case",
        request_id="thread:eval:run:case",
    )

    assert actual.intent == "kb_search"
    assert actual.allow_retrieval is True
    assert actual.rewrite_query == "项目能力"
    assert actual.match_status == "matched"
    assert actual.sources == [{"title": "项目说明", "source_uri": "https://example.test/doc"}]
    assert actual.total_ms == 42.0


def test_actual_from_request_view_uses_tool_completed_match_status_fallback():
    actual = actual_from_request_view(
        answer="当前知识库里没有找到相关内容。",
        request_view={
            "summary": {"status": "ok"},
            "events": [
                {
                    "event": "tool_completed",
                    "tool_name": "search_feishu_knowledge",
                    "match_status": "no_match",
                }
            ],
            "retrieval_call_count": 1,
            "tool_call_count": 1,
        },
    )

    assert actual.match_status == "no_match"


def test_actual_from_prepared_context_builds_retrieval_actual():
    prepared = SimpleNamespace(
        merged_chunks=[
            SimpleNamespace(
                content="alpha policy required fact",
                metadata={},
                score=0.9,
                chunk_type="text",
            )
        ],
        context="alpha policy required fact",
        sources=[{"title": "Alpha Handbook", "source_uri": "https://example.test/alpha"}],
    )

    actual = actual_from_prepared_context(
        query="alpha policy",
        prepared=prepared,
        duration_ms=12.0,
        thread_id="eval:run:case",
        request_id="thread:eval:run:case",
    )

    assert actual.answer == "alpha policy required fact"
    assert actual.allow_retrieval is True
    assert actual.retrieval_call_count == 1
    assert actual.sources == [{"title": "Alpha Handbook", "source_uri": "https://example.test/alpha"}]
    assert actual.match_status == "matched"
    assert actual.retrieval_ms == 12.0


def test_run_retrieval_cases_scores_fake_pipeline_without_generation():
    case = EvalCase(
        id="retrieval-ok",
        user_query="alpha policy",
        expected_intent="kb_search",
        expected_allow_retrieval=True,
        expected_answer_must_include=["required fact"],
        expected_source_titles=["Alpha Handbook"],
        expected_match_status="matched",
    )
    prepared = SimpleNamespace(
        merged_chunks=[
            SimpleNamespace(
                content="alpha policy required fact",
                metadata={},
                score=0.9,
                chunk_type="text",
            )
        ],
        context="alpha policy required fact",
        sources=[{"title": "Alpha Handbook", "source_uri": "https://example.test/alpha"}],
    )

    def fake_retrieval(case: EvalCase, context: EvalRunContext) -> EvalActual:
        assert context.thread_id.startswith("eval:run-1:")
        return actual_from_prepared_context(query=case.user_query, prepared=prepared, duration_ms=3.0)

    results = run_retrieval_cases([case], run_id="run-1", retrieval_runner=fake_retrieval)

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].citation_present is None
