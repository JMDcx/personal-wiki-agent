from __future__ import annotations

import json

from multimodal_rag_agent.eval.metrics import evaluate_case
from multimodal_rag_agent.eval.models import EvalActual, EvalCase
from multimodal_rag_agent.eval.report import build_markdown_report, write_run_artifacts


def test_build_markdown_report_contains_metrics_and_failures():
    result = evaluate_case(
        EvalCase(id="case-1", user_query="问题", expected_answer_must_include=["必须有"]),
        EvalActual(answer="没有"),
    )

    report = build_markdown_report(
        run_id="run-1",
        dataset_path="cases.jsonl",
        summary={
            "total_cases": 1,
            "passed_cases": 0,
            "failed_cases": 1,
            "pass_rate": 0.0,
            "failure_reason_counts": {"answer_missing_required_point": 1},
            "latency_ms": {},
        },
        results=[result],
        log_path="app.jsonl",
    )

    assert "# Agent/RAG Eval Report" in report
    assert "answer_missing_required_point" in report
    assert "case-1" in report


def test_write_run_artifacts_writes_json_jsonl_and_markdown(tmp_path):
    result = evaluate_case(
        EvalCase(id="case-1", user_query="问题", expected_answer_must_include=["答案"]),
        EvalActual(answer="答案", total_ms=10.0),
    )

    artifacts = write_run_artifacts(
        output_dir=tmp_path,
        run_id="run-1",
        dataset_path="cases.jsonl",
        results=[result],
        log_path="app.jsonl",
    )

    assert artifacts.summary_path.exists()
    assert artifacts.results_path.exists()
    assert artifacts.report_path.exists()
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    result_lines = artifacts.results_path.read_text(encoding="utf-8").splitlines()
    assert summary["passed_cases"] == 1
    assert len(result_lines) == 1
    assert "Agent/RAG Eval Report" in artifacts.report_path.read_text(encoding="utf-8")


def test_build_markdown_report_includes_baseline_comparison():
    report = build_markdown_report(
        run_id="run-2",
        dataset_path="cases.jsonl",
        summary={
            "total_cases": 2,
            "passed_cases": 1,
            "failed_cases": 1,
            "pass_rate": 0.5,
            "failure_reason_counts": {},
            "latency_ms": {},
            "baseline_comparison": {
                "baseline_path": "old/summary.json",
                "pass_rate_delta": 0.25,
                "source_hit_rate_delta": -0.1,
                "p95_latency_delta_ms": 100.0,
                "newly_failed_case_ids": ["case-new"],
                "fixed_case_ids": ["case-fixed"],
            },
        },
        results=[],
    )

    assert "## Baseline Comparison" in report
    assert "old/summary.json" in report
    assert "case-new" in report
    assert "case-fixed" in report
