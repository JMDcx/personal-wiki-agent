"""Replay a JSONL eval dataset through the local Agent runtime."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - script execution fallback
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from multimodal_rag_agent.eval.dataset import load_eval_cases, select_cases
from multimodal_rag_agent.eval.report import write_run_artifacts
from multimodal_rag_agent.eval.runner import (
    AgentRunner,
    EvalRunContext,
    RetrievalRunner,
    actual_from_prepared_context,
    actual_from_request_view,
    collect_request_view,
    read_log_records,
    run_cases,
    run_retrieval_cases,
)
from multimodal_rag_agent.eval.models import EvalActual, EvalCase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay an Agent/RAG eval dataset and write report artifacts.")
    parser.add_argument(
        "--mode",
        choices=["agent", "retrieval"],
        default="agent",
        help="Run full Agent eval or retrieval-only eval.",
    )
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/evals/smoke_cases.jsonl",
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
        help="Language passed into invoke_agent when a case does not specify language.",
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
    parser.add_argument(
        "--baseline",
        default="",
        help="Optional previous summary.json path for baseline comparison.",
    )
    return parser


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

    if messages:
        runtime.update_state({"configurable": {"thread_id": thread_id}}, {"messages": messages})


def _build_real_agent_runner(
    *,
    output_dir: Path,
    default_language: str,
) -> tuple[AgentRunner, str]:
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

    print("Building agent runtime...", flush=True)
    runtime = _get_or_build_agent_runtime(settings)
    log_path = str(settings.log_file_path)

    def real_runner(case: EvalCase, context: EvalRunContext) -> EvalActual:
        _seed_history(runtime, context.thread_id, case.history)
        answer = invoke_agent(
            case.user_query,
            settings=settings,
            thread_id=context.thread_id,
            language=case.language or default_language,
        )
        request_view = collect_request_view(read_log_records(settings.log_file_path), context.request_id)
        return actual_from_request_view(
            answer=answer,
            request_view=request_view,
            thread_id=context.thread_id,
            request_id=context.request_id,
        )

    return real_runner, log_path


def _build_real_retrieval_runner() -> tuple[RetrievalRunner, str]:
    from multimodal_rag_agent.rag_query_pipeline.pipeline import RAGQueryPipeline

    pipeline = RAGQueryPipeline()

    def real_retrieval_runner(case: EvalCase, context: EvalRunContext) -> EvalActual:
        started_at = perf_counter()
        prepared = pipeline.prepare_context(case.user_query, with_sources=True)
        elapsed_ms = (perf_counter() - started_at) * 1000
        return actual_from_prepared_context(
            query=case.user_query,
            prepared=prepared,
            duration_ms=round(elapsed_ms, 1),
            thread_id=context.thread_id,
            request_id=context.request_id,
        )

    return real_retrieval_runner, ""


def _parse_ids(raw_ids: str) -> set[str]:
    return {item.strip() for item in str(raw_ids or "").split(",") if item.strip()}


def main(
    argv: list[str] | None = None,
    *,
    agent_runner: AgentRunner | None = None,
    retrieval_runner: RetrievalRunner | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    run_id = datetime.now().strftime("replay-%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (Path("data/evals/runs") / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = str(args.mode)
    cases = select_cases(
        load_eval_cases(dataset_path),
        ids=_parse_ids(args.ids),
        limit=0 if mode == "retrieval" else args.limit,
    )
    if not cases:
        raise SystemExit("No cases selected.")

    log_path = str(output_dir / "app.jsonl")
    skipped_non_retrieval = 0
    if mode == "retrieval":
        before_filter_count = len(cases)
        cases = [case for case in cases if case.expected_allow_retrieval is not False]
        if args.limit > 0:
            cases = cases[: args.limit]
        skipped_non_retrieval = before_filter_count - len(cases)
        if not cases:
            raise SystemExit("No retrieval cases selected.")
        if retrieval_runner is None:
            retrieval_runner, log_path = _build_real_retrieval_runner()
    elif agent_runner is None:
        agent_runner, log_path = _build_real_agent_runner(output_dir=output_dir, default_language=str(args.language))

    print(f"Mode: {mode}")
    print(f"Running {len(cases)} case(s)")
    if skipped_non_retrieval:
        print(f"Skipped {skipped_non_retrieval} non-retrieval case(s)")
    print(f"Dataset: {dataset_path}")
    print(f"Run dir: {output_dir}")

    if mode == "retrieval":
        if retrieval_runner is None:  # pragma: no cover - guarded above
            raise RuntimeError("retrieval_runner is not configured")
        results = run_retrieval_cases(
            cases,
            run_id=run_id,
            retrieval_runner=retrieval_runner,
            sleep_seconds=float(args.sleep_seconds),
        )
    else:
        if agent_runner is None:  # pragma: no cover - guarded above
            raise RuntimeError("agent_runner is not configured")
        results = run_cases(
            cases,
            run_id=run_id,
            agent_runner=agent_runner,
            sleep_seconds=float(args.sleep_seconds),
        )
    artifacts = write_run_artifacts(
        output_dir=output_dir,
        run_id=run_id,
        dataset_path=str(dataset_path),
        results=results,
        log_path=log_path,
        baseline_path=str(args.baseline or ""),
    )

    summary = artifacts.summary
    print("")
    print(
        "Completed. "
        f"Passed {summary['passed_cases']}/{summary['total_cases']}; "
        f"failed {summary['failed_cases']}."
    )
    print(f"Artifacts: {artifacts.output_dir}")

    if args.strict and summary["failed_cases"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
