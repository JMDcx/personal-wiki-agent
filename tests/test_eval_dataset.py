from __future__ import annotations

import json

import pytest

from multimodal_rag_agent.eval.dataset import load_eval_cases, select_cases


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_eval_cases_normalizes_current_schema(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "case-1",
                "user_query": "项目支持哪些能力？",
                "expected_intent": "kb_search",
                "expected_allow_retrieval": True,
                "expected_answer_must_include": ["飞书", "知识库"],
                "expected_source_titles": ["项目说明"],
                "expected_match_status": "matched",
                "tags": ["smoke"],
            }
        ],
    )

    cases = load_eval_cases(dataset_path)

    assert len(cases) == 1
    case = cases[0]
    assert case.id == "case-1"
    assert case.user_query == "项目支持哪些能力？"
    assert case.expected_intent == "kb_search"
    assert case.expected_allow_retrieval is True
    assert case.expected_answer_must_include == ["飞书", "知识库"]
    assert case.expected_source_titles == ["项目说明"]
    assert case.expected_match_status == "matched"
    assert case.tags == ["smoke"]


def test_load_eval_cases_accepts_legacy_fields(tmp_path):
    dataset_path = tmp_path / "legacy.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "legacy-1",
                "question": "旧数据集问题",
                "should_retrieve": False,
                "expected_answer_points": ["你好"],
            }
        ],
    )

    case = load_eval_cases(dataset_path)[0]

    assert case.user_query == "旧数据集问题"
    assert case.expected_allow_retrieval is False
    assert case.expected_answer_must_include == ["你好"]


def test_load_eval_cases_rejects_missing_required_fields(tmp_path):
    dataset_path = tmp_path / "bad.jsonl"
    _write_jsonl(dataset_path, [{"id": "missing-query"}])

    with pytest.raises(ValueError, match="user_query"):
        load_eval_cases(dataset_path)


def test_select_cases_filters_ids_and_limit(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {"id": "a", "user_query": "A"},
            {"id": "b", "user_query": "B"},
            {"id": "c", "user_query": "C"},
        ],
    )
    cases = load_eval_cases(dataset_path)

    selected = select_cases(cases, ids={"b", "c"}, limit=1)

    assert [case.id for case in selected] == ["b"]
