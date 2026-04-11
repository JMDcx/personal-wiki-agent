from __future__ import annotations

import logging

from agent import (
    _message_state_fields,
    _normalize_deposit_request_context,
    _normalize_retrieval_request,
)
from schemas import MessageContext


def test_message_state_fields_accepts_message_context_object() -> None:
    context = MessageContext.from_dict(
        {
            "chat_type": "group",
            "bot_mentioned": True,
            "mentioned_users": ["张三"],
            "reply_context": {
                "is_reply": True,
                "parent_id": "om_parent_123",
                "root_id": "om_root_123",
                "parent_text_preview": "上一个问题在讨论 Agent",
            },
        }
    )

    fields = _message_state_fields(context)

    assert fields == {
        "is_reply": True,
        "reply_parent_id": "om_parent_123",
        "reply_root_id": "om_root_123",
        "mentioned_users": ["张三"],
        "bot_mentioned": True,
    }


def test_normalize_retrieval_request_logs_schema_warning_for_empty_query(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="feishu_wiki_rag_agent.events")

    request = _normalize_retrieval_request("   ")

    assert request.query == ""
    record = next(record for record in caplog.records if getattr(record, "event", "") == "schema_normalization_warning")
    assert record.schema_stage == "retrieval_request"
    assert record.reason == "empty_query_after_normalization"


def test_normalize_deposit_request_context_logs_schema_warning_for_invalid_image_paths_json(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="feishu_wiki_rag_agent.events")

    context = _normalize_deposit_request_context(text="store this", image_paths_json="{bad json}")

    assert context.image_paths == []
    record = next(record for record in caplog.records if getattr(record, "event", "") == "schema_normalization_warning")
    assert record.schema_stage == "deposit_request"
    assert record.reason == "invalid_image_paths_json"
