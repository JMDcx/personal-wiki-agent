"""Report writers for Agent/RAG evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multimodal_rag_agent.eval.compare import build_baseline_comparison, load_summary
from multimodal_rag_agent.eval.metrics import summarize_results
from multimodal_rag_agent.eval.models import EvalResult, RunArtifacts


def _fmt_rate(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    return str(value)


def _fmt_ms(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return str(value)


def _fmt_delta(value: object, *, rate: bool = False, ms: bool = False) -> str:
    if value is None:
        return "-"
    if not isinstance(value, (int, float)):
        return str(value)
    sign = "+" if value > 0 else ""
    if rate:
        return f"{sign}{value:.1%}"
    if ms:
        return f"{sign}{value:.1f}"
    return f"{sign}{value}"


def build_markdown_report(
    *,
    run_id: str,
    dataset_path: str,
    summary: dict[str, Any],
    results: list[EvalResult],
    log_path: str = "",
) -> str:
    """Build a human-readable Markdown report for one eval run."""
    lines: list[str] = [
        "# Agent/RAG Eval Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Dataset: `{dataset_path}`",
        f"- Log file: `{log_path or '-'}`",
        f"- Total cases: `{summary.get('total_cases', 0)}`",
        f"- Passed: `{summary.get('passed_cases', 0)}`",
        f"- Failed: `{summary.get('failed_cases', 0)}`",
        f"- Pass rate: `{_fmt_rate(summary.get('pass_rate'))}`",
        "",
        "## Quality Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in [
        "intent_accuracy",
        "allow_retrieval_accuracy",
        "rewrite_accuracy",
        "answer_match_rate",
        "source_hit_rate",
        "match_status_accuracy",
        "citation_presence_rate",
        "citation_source_hit_rate",
    ]:
        if key in summary:
            lines.append(f"| {key} | {_fmt_rate(summary.get(key))} |")

    latency = summary.get("latency_ms") or {}
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| metric | ms |",
            "| --- | ---: |",
            f"| avg | {_fmt_ms(latency.get('avg'))} |",
            f"| p50 | {_fmt_ms(latency.get('p50'))} |",
            f"| p95 | {_fmt_ms(latency.get('p95'))} |",
            f"| max | {_fmt_ms(latency.get('max'))} |",
            "",
            "## Failure Reasons",
            "",
        ]
    )
    reason_counts = summary.get("failure_reason_counts") or {}
    if reason_counts:
        lines.extend(["| reason | count |", "| --- | ---: |"])
        for reason, count in reason_counts.items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("No failures.")

    comparison = summary.get("baseline_comparison")
    if isinstance(comparison, dict):
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                f"- Baseline: `{comparison.get('baseline_path') or '-'}`",
                "",
                "| metric | delta |",
                "| --- | ---: |",
                f"| pass_rate | {_fmt_delta(comparison.get('pass_rate_delta'), rate=True)} |",
                f"| source_hit_rate | {_fmt_delta(comparison.get('source_hit_rate_delta'), rate=True)} |",
                f"| match_status_accuracy | {_fmt_delta(comparison.get('match_status_accuracy_delta'), rate=True)} |",
                f"| answer_match_rate | {_fmt_delta(comparison.get('answer_match_rate_delta'), rate=True)} |",
                f"| p95_latency_ms | {_fmt_delta(comparison.get('p95_latency_delta_ms'), ms=True)} |",
                f"| failed_cases | {_fmt_delta(comparison.get('failed_cases_delta'))} |",
                "",
                f"- Newly failed: `{comparison.get('newly_failed_case_ids') or []}`",
                f"- Fixed: `{comparison.get('fixed_case_ids') or []}`",
                f"- Still failed: `{comparison.get('still_failed_case_ids') or []}`",
            ]
        )

    failed = [result for result in results if not result.passed]
    lines.extend(["", "## Failed Cases", ""])
    if not failed:
        lines.append("No failed cases.")
    for result in failed:
        lines.extend(
            [
                f"### {result.case_id}",
                "",
                f"- primary_failure_reason: `{result.primary_failure_reason or '-'}`",
                f"- failure_reasons: `{', '.join(result.failure_reasons) or '-'}`",
                f"- status: `{result.status}`",
                f"- expected_intent / actual_intent: `{result.expected_intent or '-'}` / `{result.actual_intent or '-'}`",
                f"- expected_allow_retrieval / actual_allow_retrieval: `{result.expected_allow_retrieval}` / `{result.actual_allow_retrieval}`",
                f"- missing_required: `{result.missing_required}`",
                f"- forbidden_hits: `{result.forbidden_hits}`",
                f"- retrieval_call_count / tool_call_count: `{result.retrieval_call_count}` / `{result.tool_call_count}`",
                f"- total_ms: `{_fmt_ms(result.total_ms)}`",
                "",
                "Actual answer:",
                "",
                result.actual_answer or "-",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def write_run_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    dataset_path: str,
    results: list[EvalResult],
    log_path: str = "",
    baseline_path: str = "",
) -> RunArtifacts:
    """Write summary.json, results.jsonl, and report.md for one eval run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(run_id, dataset_path, results)
    if baseline_path:
        summary["baseline_comparison"] = build_baseline_comparison(
            summary,
            load_summary(baseline_path),
            baseline_path=baseline_path,
        )
    summary_path = output / "summary.json"
    results_path = output / "results.jsonl"
    report_path = output / "report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result_lines = [json.dumps(result.to_dict(), ensure_ascii=False) for result in results]
    results_path.write_text("\n".join(result_lines) + ("\n" if result_lines else ""), encoding="utf-8")
    report_path.write_text(
        build_markdown_report(
            run_id=run_id,
            dataset_path=dataset_path,
            summary=summary,
            results=results,
            log_path=log_path,
        ),
        encoding="utf-8",
    )
    return RunArtifacts(
        output_dir=output,
        summary_path=summary_path,
        results_path=results_path,
        report_path=report_path,
        summary=summary,
    )
