from __future__ import annotations

import json

from multimodal_rag_agent.eval.badcase import extract_badcase_drafts, write_badcase_drafts


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_extract_badcase_drafts_uses_failed_request_summary():
    records = [
        {
            "timestamp": "2026-05-14T00:00:00+00:00",
            "event": "agent_invoke_started",
            "request_id": "thread:abc",
            "thread_id": "abc",
            "question_preview": "How do I update the index?",
        },
        {
            "timestamp": "2026-05-14T00:00:01+00:00",
            "event": "request_summary",
            "request_id": "thread:abc",
            "thread_id": "abc",
            "status": "error",
            "intent": "kb_search",
            "allow_retrieval": True,
            "rewrite_query": "update index",
            "question_preview": "How do I update the index?",
            "answer_preview": "The request failed",
            "error_type": "RuntimeError",
            "error_message": "provider timeout",
            "total_ms": 1000.0,
        },
    ]

    drafts = extract_badcase_drafts(records)

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft["id"].startswith("badcase_thread-abc")
    assert draft["user_query"] == "How do I update the index?"
    assert draft["actual_intent"] == "kb_search"
    assert draft["actual_allow_retrieval"] is True
    assert draft["actual_rewrite_query"] == "update index"
    assert draft["actual_answer"] == "The request failed"
    assert draft["expected_answer_must_include"] == []
    assert draft["metadata"]["error_type"] == "RuntimeError"
    assert draft["metadata"]["error_message"] == "provider timeout"


def test_extract_badcase_drafts_skips_ok_requests_by_default():
    drafts = extract_badcase_drafts(
        [
            {
                "event": "request_summary",
                "request_id": "thread:ok",
                "status": "ok",
                "question_preview": "hello",
                "answer_preview": "hi",
            }
        ]
    )

    assert drafts == []


def test_extract_badcase_drafts_can_include_ok_requests_for_manual_review():
    drafts = extract_badcase_drafts(
        [
            {
                "event": "request_summary",
                "request_id": "thread:ok",
                "status": "ok",
                "question_preview": "hello",
                "answer_preview": "hi",
            }
        ],
        include_ok=True,
    )

    assert len(drafts) == 1
    assert drafts[0]["tags"] == ["badcase", "draft", "status:ok"]


def test_write_badcase_drafts_reads_log_and_writes_jsonl(tmp_path):
    log_path = tmp_path / "app.jsonl"
    output_path = tmp_path / "drafts.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "event": "request_summary",
                "request_id": "thread:abc",
                "status": "error",
                "question_preview": "broken question",
                "answer_preview": "broken answer",
            }
        ],
    )

    drafts = write_badcase_drafts(log_path=log_path, output_path=output_path)

    assert len(drafts) == 1
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["user_query"] == "broken question"
