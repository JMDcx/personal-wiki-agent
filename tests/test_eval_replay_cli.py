from __future__ import annotations

import json

from multimodal_rag_agent.eval.models import EvalActual, EvalCase
from multimodal_rag_agent.eval.runner import EvalRunContext
from scripts import replay_eval_dataset


def test_replay_cli_writes_artifacts_with_injected_runner(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "user_query": "问题",
                "expected_answer_must_include": ["答案"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    def fake_runner(case: EvalCase, context: EvalRunContext) -> EvalActual:
        assert case.id == "case-1"
        assert context.run_id.startswith("replay-")
        return EvalActual(answer="答案", total_ms=1.0)

    exit_code = replay_eval_dataset.main(
        ["--dataset", str(dataset_path), "--output-dir", str(output_dir)],
        agent_runner=fake_runner,
    )

    assert exit_code == 0
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "report.md").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed_cases"] == 1


def test_replay_cli_supports_retrieval_mode_with_injected_runner(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "retrieval-1",
                "user_query": "alpha policy",
                "expected_intent": "kb_search",
                "expected_allow_retrieval": True,
                "expected_answer_must_include": ["required fact"],
                "expected_source_titles": ["Alpha Handbook"],
                "expected_match_status": "matched",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "retrieval-run"

    def fake_retrieval(case: EvalCase, context: EvalRunContext) -> EvalActual:
        assert case.id == "retrieval-1"
        return EvalActual(
            answer="retrieved context with required fact",
            allow_retrieval=True,
            retrieval_call_count=1,
            sources=[{"title": "Alpha Handbook"}],
            match_status="matched",
            total_ms=2.0,
            retrieval_ms=2.0,
        )

    exit_code = replay_eval_dataset.main(
        ["--mode", "retrieval", "--dataset", str(dataset_path), "--output-dir", str(output_dir)],
        retrieval_runner=fake_retrieval,
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed_cases"] == 1
    result = json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert result["citation_present"] is None


def test_replay_cli_retrieval_mode_applies_limit_after_filtering(tmp_path):
    dataset_path = tmp_path / "mixed.jsonl"
    rows = [
        {
            "id": "chat-1",
            "user_query": "hello",
            "expected_allow_retrieval": False,
        },
        {
            "id": "retrieval-1",
            "user_query": "alpha policy",
            "expected_allow_retrieval": True,
            "expected_answer_must_include": ["required fact"],
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "retrieval-run"
    seen_ids: list[str] = []

    def fake_retrieval(case: EvalCase, context: EvalRunContext) -> EvalActual:
        seen_ids.append(case.id)
        return EvalActual(
            answer="required fact",
            allow_retrieval=True,
            retrieval_call_count=1,
            match_status="matched",
        )

    exit_code = replay_eval_dataset.main(
        ["--mode", "retrieval", "--dataset", str(dataset_path), "--output-dir", str(output_dir), "--limit", "1"],
        retrieval_runner=fake_retrieval,
    )

    assert exit_code == 0
    assert seen_ids == ["retrieval-1"]


def test_replay_cli_writes_baseline_comparison(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "user_query": "question",
                "expected_answer_must_include": ["answer"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline-summary.json"
    baseline_path.write_text(
        json.dumps(
            {
                "pass_rate": 0.0,
                "failed_cases": 1,
                "failed_case_ids": ["case-1"],
                "latency_ms": {"p95": 10.0},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    def fake_runner(case: EvalCase, context: EvalRunContext) -> EvalActual:
        return EvalActual(answer="answer", total_ms=5.0)

    exit_code = replay_eval_dataset.main(
        [
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--baseline",
            str(baseline_path),
        ],
        agent_runner=fake_runner,
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_comparison"]["pass_rate_delta"] == 1.0
    assert summary["baseline_comparison"]["fixed_case_ids"] == ["case-1"]
