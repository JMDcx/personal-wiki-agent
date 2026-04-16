"""Replay a JSONL QA dataset through invoke_agent and compare outputs.

This script is designed for local production-like evaluation against the
currently indexed knowledge base. It:

1. Loads one JSONL dataset of QA cases.
2. Replays optional multi-turn history into a dedicated thread state.
3. Calls `invoke_agent(...)` for each case.
4. Reads structured logs to compare actual intent, rewrite_query, answer, and timings.
5. Writes machine-readable results plus a Markdown summary report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - script execution fallback
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a QA dataset through invoke_agent and compare results.")
    parser.add_argument(
        "--dataset",
        default="data/evals/single_doc_job_kb_qa.jsonl",
        help="Path to the JSONL dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for this run's artifacts. Defaults to data/evals/runs/<timestamp>.",
    )
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated case ids to run. Default runs all cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N selected cases.",
    )
    parser.add_argument(
        "--language",
        default="中文",
        help="Language passed into invoke_agent. Default is 中文.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds between cases to reduce provider throttling.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status code 1 when any case fails.",
    )
    return parser


def _collapse_ws(text: str) -> str:
    return " ".join((text or "").split())


def _normalize_for_match(text: str) -> str:
    return _collapse_ws(text).strip()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "eval"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid user dataset
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def _read_log_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _select_cases(rows: list[dict[str, Any]], *, ids: set[str], limit: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if not ids or str(row.get("id", "")) in ids]
    if limit > 0:
        selected = selected[:limit]
    return selected


def _contains_all(answer: str, required: list[str]) -> list[str]:
    normalized_answer = _normalize_for_match(answer)
    return [item for item in required if _normalize_for_match(item) not in normalized_answer]


def _contains_any(answer: str, forbidden: list[str]) -> list[str]:
    normalized_answer = _normalize_for_match(answer)
    return [item for item in forbidden if _normalize_for_match(item) in normalized_answer]


def _to_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


@dataclass(slots=True)
class CaseResult:
    case_id: str
    scene: str
    function: str
    thread_id: str
    request_id: str
    status: str
    error: str
    expected_intent: str
    actual_intent: str
    intent_match: bool
    expected_allow_retrieval: bool
    actual_allow_retrieval: bool | None
    allow_retrieval_match: bool
    expected_rewrite_query: str
    actual_rewrite_query: str
    rewrite_match: bool
    answer_match: bool
    missing_required: list[str]
    forbidden_hits: list[str]
    retrieval_call_count: int
    tool_call_count: int
    total_ms: float | None
    intent_ms: float | None
    retrieval_ms: float | None
    llm_ms: float | None
    reply_ms: float | None
    actual_answer: str
    reference_answer: str


def _collect_request_view(records: list[dict[str, Any]], request_id: str) -> dict[str, Any]:
    matched = [record for record in records if str(record.get("request_id", "")) == request_id]
    if not matched:
        return {
            "summary": {},
            "events": [],
            "generation_ms": None,
            "retrieval_ms_sum": None,
            "retrieval_call_count": 0,
            "tool_call_count": 0,
        }

    summary = next((record for record in reversed(matched) if record.get("event") == "request_summary"), {})
    generation_event = next((record for record in reversed(matched) if record.get("event") == "generation_completed"), {})
    generation_ms = _to_float(generation_event.get("duration_ms")) if generation_event else None
    retrieval_events = [record for record in matched if record.get("event") == "retrieval_completed"]
    retrieval_ms_values = [_to_float(record.get("duration_ms")) for record in retrieval_events]
    retrieval_ms_sum = sum(value for value in retrieval_ms_values if value is not None) if retrieval_ms_values else None
    tool_call_count = sum(1 for record in matched if record.get("event") == "tool_called")
    return {
        "summary": summary,
        "events": matched,
        "generation_ms": generation_ms,
        "retrieval_ms_sum": retrieval_ms_sum,
        "retrieval_call_count": len(retrieval_events),
        "tool_call_count": tool_call_count,
    }


def _seed_history(runtime: Any, thread_id: str, history: list[dict[str, str]]) -> None:
    if not history:
        return

    from langchain_core.messages import AIMessage, HumanMessage

    messages: list[Any] = []
    for turn in history:
        user_question = str(turn.get("user_question", "")).strip()
        assistant_answer = str(turn.get("assistant_answer", "")).strip()
        if user_question:
            messages.append(HumanMessage(content=user_question))
        if assistant_answer:
            messages.append(AIMessage(content=assistant_answer))

    if not messages:
        return

    runtime.update_state({"configurable": {"thread_id": thread_id}}, {"messages": messages})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_markdown_report(
    *,
    run_id: str,
    dataset_path: Path,
    log_path: Path,
    results: list[CaseResult],
) -> str:
    completed = sum(1 for result in results if result.status == "ok")
    errored = len(results) - completed
    total_ms_values = [result.total_ms for result in results if result.total_ms is not None]
    slowest = sorted(
        [result for result in results if result.total_ms is not None],
        key=lambda item: item.total_ms or 0.0,
        reverse=True,
    )[:5]
    errored_results = [result for result in results if result.status != "ok"]

    lines: list[str] = [
        f"# 回放评测报告",
        "",
        f"- 运行 ID: `{run_id}`",
        f"- 数据集: `{dataset_path}`",
        f"- 用例数: `{len(results)}`",
        f"- 完成: `{completed}`",
        f"- 异常: `{errored}`",
        f"- 平均 total_ms: `{_format_ms(mean(total_ms_values) if total_ms_values else None)}`",
        f"- 日志文件: `{log_path}`",
        "",
        "## 最慢用例",
        "",
        "| case_id | intent | total_ms | intent_ms | retrieval_ms | llm_ms | reply_ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in slowest:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.case_id,
                    result.actual_intent or "-",
                    _format_ms(result.total_ms),
                    _format_ms(result.intent_ms),
                    _format_ms(result.retrieval_ms),
                    _format_ms(result.llm_ms),
                    _format_ms(result.reply_ms),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 异常用例", ""])
    if not errored_results:
        lines.append("无运行异常。")
        return "\n".join(lines) + "\n"

    for result in errored_results:
        lines.extend(
            [
                f"### {result.case_id}",
                "",
                f"- scene: `{result.scene}`",
                f"- function: `{result.function}`",
                f"- status: `{result.status}`",
                f"- error: `{result.error or '-'}`",
                f"- expected_intent / actual_intent: `{result.expected_intent}` / `{result.actual_intent or '-'}`",
                f"- expected_allow_retrieval / actual_allow_retrieval: `{result.expected_allow_retrieval}` / `{result.actual_allow_retrieval}`",
                f"- expected_rewrite / actual_rewrite: `{result.expected_rewrite_query}` / `{result.actual_rewrite_query or '-'}`",
                f"- missing_required: `{result.missing_required}`",
                f"- forbidden_hits: `{result.forbidden_hits}`",
                f"- retrieval_call_count / tool_call_count: `{result.retrieval_call_count}` / `{result.tool_call_count}`",
                f"- total_ms: `{_format_ms(result.total_ms)}`",
                "",
                "参考答案：",
                "",
                result.reference_answer or "-",
                "",
                "实际答案：",
                "",
                result.actual_answer or "-",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    selected_ids = {item.strip() for item in str(args.ids or "").split(",") if item.strip()}
    run_id = datetime.now().strftime("replay-%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (Path("data/evals/runs") / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("FEISHU_LOG_LEVEL", "DEBUG")
    os.environ.setdefault("FEISHU_LOG_JSON_LEVEL", "DEBUG")
    os.environ.setdefault("FEISHU_LOG_CONSOLE_LEVEL", "INFO")

    print("Loading agent module...", flush=True)
    from agent import _get_or_build_agent_runtime, invoke_agent
    print("Loading settings...", flush=True)
    from config import get_settings

    settings = get_settings()
    settings.log_file_path = output_dir / "app.jsonl"
    settings.checkpoint_db_path = output_dir / "eval-checkpoints.sqlite"
    settings.ensure_directories()

    rows = _load_jsonl(dataset_path)
    cases = _select_cases(rows, ids=selected_ids, limit=args.limit)
    if not cases:
        raise SystemExit("No cases selected.")

    print("Building agent runtime...", flush=True)
    runtime = _get_or_build_agent_runtime(settings)
    results: list[CaseResult] = []

    print(f"Running {len(cases)} case(s)")
    print(f"Dataset: {dataset_path}")
    print(f"Run dir: {output_dir}")

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case-{index}"))
        thread_id = f"eval:{run_id}:{_slugify(case_id)}"
        request_id = f"thread:{thread_id}"

        history = case.get("history", [])
        if isinstance(history, list):
            _seed_history(runtime, thread_id, [item for item in history if isinstance(item, dict)])

        actual_answer = ""
        error = ""
        status = "ok"
        summary: dict[str, Any] = {}
        try:
            actual_answer = invoke_agent(
                str(case.get("user_query", "")),
                settings=settings,
                thread_id=thread_id,
                language=str(args.language),
            )
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = str(exc)

        request_view = _collect_request_view(_read_log_records(settings.log_file_path), request_id)
        summary = dict(request_view.get("summary") or {})
        retrieval_ms_from_events = _to_float(request_view.get("retrieval_ms_sum"))
        retrieval_ms = _to_float(summary.get("retrieval_ms"))
        if retrieval_ms is None:
            retrieval_ms = retrieval_ms_from_events

        expected_intent = str(case.get("expected_intent", "") or "")
        actual_intent = str(summary.get("intent", "") or "")
        intent_match = expected_intent == actual_intent

        expected_allow_retrieval = bool(case.get("expected_allow_retrieval", False))
        summary_allow = summary.get("allow_retrieval")
        actual_allow_retrieval = summary_allow if isinstance(summary_allow, bool) else None
        allow_retrieval_match = actual_allow_retrieval == expected_allow_retrieval

        expected_rewrite_query = str(case.get("expected_rewrite_query", "") or "")
        actual_rewrite_query = str(summary.get("rewrite_query", "") or "")
        rewrite_match = _normalize_for_match(expected_rewrite_query) == _normalize_for_match(actual_rewrite_query)

        missing_required = _contains_all(actual_answer, [str(item) for item in case.get("expected_answer_must_include", [])])
        forbidden_hits = _contains_any(actual_answer, [str(item) for item in case.get("expected_answer_must_not_include", [])])
        answer_match = not missing_required and not forbidden_hits and bool(actual_answer)

        if status == "error" and summary:
            status = str(summary.get("status", "error"))

        result = CaseResult(
            case_id=case_id,
            scene=str(case.get("scene", "")),
            function=str(case.get("function", "")),
            thread_id=thread_id,
            request_id=request_id,
            status=status,
            error=error,
            expected_intent=expected_intent,
            actual_intent=actual_intent,
            intent_match=intent_match,
            expected_allow_retrieval=expected_allow_retrieval,
            actual_allow_retrieval=actual_allow_retrieval,
            allow_retrieval_match=allow_retrieval_match,
            expected_rewrite_query=expected_rewrite_query,
            actual_rewrite_query=actual_rewrite_query,
            rewrite_match=rewrite_match,
            answer_match=answer_match,
            missing_required=missing_required,
            forbidden_hits=forbidden_hits,
            retrieval_call_count=int(request_view.get("retrieval_call_count") or 0),
            tool_call_count=int(request_view.get("tool_call_count") or 0),
            total_ms=_to_float(summary.get("total_ms")),
            intent_ms=_to_float(summary.get("intent_ms")),
            retrieval_ms=retrieval_ms,
            llm_ms=_to_float(summary.get("llm_ms")),
            reply_ms=_to_float(summary.get("reply_ms")),
            actual_answer=actual_answer,
            reference_answer=str(case.get("reference_answer", "")),
        )
        results.append(result)

        print(
            f"[{index}/{len(cases)}] {case_id} "
            f"status={result.status} "
            f"intent={result.actual_intent or '-'} "
            f"rewrite_match={result.rewrite_match} "
            f"answer_match={result.answer_match} "
            f"total_ms={_format_ms(result.total_ms)}"
        )

        if args.sleep_seconds > 0 and index < len(cases):
            print(f"Sleeping {args.sleep_seconds:.1f}s before next case...", flush=True)
            time.sleep(args.sleep_seconds)

    result_rows = [asdict(result) for result in results]
    summary_payload = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "log_path": str(settings.log_file_path),
        "total_cases": len(results),
        "completed_cases": sum(1 for result in results if result.status == "ok"),
        "errored_cases": sum(1 for result in results if result.status != "ok"),
        "results": result_rows,
    }

    _write_json(output_dir / "summary.json", summary_payload)
    _write_jsonl(output_dir / "results.jsonl", result_rows)
    (output_dir / "report.md").write_text(
        _build_markdown_report(
            run_id=run_id,
            dataset_path=dataset_path,
            log_path=settings.log_file_path,
            results=results,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"Completed. Finished {summary_payload['completed_cases']}/{summary_payload['total_cases']}.")
    print(f"Artifacts: {output_dir}")

    if args.strict and summary_payload["errored_cases"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
