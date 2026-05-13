"""Baseline comparison helpers for eval summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load one summary.json file."""
    summary_path = Path(path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object summary: {summary_path}")
    return data


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _round_delta(current: object, baseline: object) -> float | None:
    current_float = _as_float(current)
    baseline_float = _as_float(baseline)
    if current_float is None or baseline_float is None:
        return None
    return round(current_float - baseline_float, 6)


def _failed_case_ids(summary: dict[str, Any]) -> set[str]:
    raw_ids = summary.get("failed_case_ids")
    if not isinstance(raw_ids, list):
        return set()
    return {str(case_id) for case_id in raw_ids}


def build_baseline_comparison(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    baseline_path: str = "",
) -> dict[str, Any]:
    """Compare current summary metrics against a previous baseline summary."""
    current_failed_ids = _failed_case_ids(current)
    baseline_failed_ids = _failed_case_ids(baseline)
    current_latency = current.get("latency_ms") if isinstance(current.get("latency_ms"), dict) else {}
    baseline_latency = baseline.get("latency_ms") if isinstance(baseline.get("latency_ms"), dict) else {}
    failed_cases_delta = _round_delta(current.get("failed_cases"), baseline.get("failed_cases"))
    return {
        "baseline_path": baseline_path,
        "pass_rate_delta": _round_delta(current.get("pass_rate"), baseline.get("pass_rate")),
        "source_hit_rate_delta": _round_delta(current.get("source_hit_rate"), baseline.get("source_hit_rate")),
        "match_status_accuracy_delta": _round_delta(
            current.get("match_status_accuracy"),
            baseline.get("match_status_accuracy"),
        ),
        "answer_match_rate_delta": _round_delta(current.get("answer_match_rate"), baseline.get("answer_match_rate")),
        "p95_latency_delta_ms": _round_delta(current_latency.get("p95"), baseline_latency.get("p95")),
        "failed_cases_delta": int(failed_cases_delta) if failed_cases_delta is not None else None,
        "newly_failed_case_ids": sorted(current_failed_ids - baseline_failed_ids),
        "fixed_case_ids": sorted(baseline_failed_ids - current_failed_ids),
        "still_failed_case_ids": sorted(current_failed_ids & baseline_failed_ids),
    }
