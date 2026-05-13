from __future__ import annotations

from multimodal_rag_agent.eval.compare import build_baseline_comparison


def test_build_baseline_comparison_reports_metric_deltas_and_case_changes():
    baseline = {
        "passed_cases": 7,
        "failed_cases": 3,
        "pass_rate": 0.7,
        "source_hit_rate": 0.6,
        "match_status_accuracy": 0.8,
        "latency_ms": {"p95": 1200.0},
        "failed_case_ids": ["still-bad", "fixed-case"],
    }
    current = {
        "passed_cases": 8,
        "failed_cases": 2,
        "pass_rate": 0.8,
        "source_hit_rate": 0.75,
        "match_status_accuracy": 0.7,
        "latency_ms": {"p95": 1500.0},
        "failed_case_ids": ["still-bad", "new-bad"],
    }

    comparison = build_baseline_comparison(current, baseline, baseline_path="old/summary.json")

    assert comparison["baseline_path"] == "old/summary.json"
    assert comparison["pass_rate_delta"] == 0.1
    assert comparison["source_hit_rate_delta"] == 0.15
    assert comparison["match_status_accuracy_delta"] == -0.1
    assert comparison["p95_latency_delta_ms"] == 300.0
    assert comparison["failed_cases_delta"] == -1
    assert comparison["newly_failed_case_ids"] == ["new-bad"]
    assert comparison["fixed_case_ids"] == ["fixed-case"]
