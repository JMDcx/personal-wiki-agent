from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from multimodal_rag_agent.rag_query_pipeline.query_understand_service import (
    ModelConfig,
    QueryUnderstandService,
)
from multimodal_rag_agent.deposit_pipeline.models import DepositResult as PipelineDepositResult
from multimodal_rag_agent.deposit_pipeline.models import KnowledgeDraft

from protocols.controller_models import ControllerDecision
from protocols.tool_models import DepositRequestContext, DepositResult, RetrievalRequest, RetrievalResult
from schemas import MessageContext


def test_controller_decision_normalizes_invalid_intent() -> None:
    decision = ControllerDecision.from_dict(
        {
            "intent": "unknown",
            "allow_retrieval": "yes",
            "rewrite_query": "Agent essence",
        }
    )

    assert decision.intent == "kb_search"
    assert decision.allow_retrieval is True
    assert decision.rewrite_query == "Agent essence"


def test_query_understand_parse_output_defaults_invalid_intent() -> None:
    service = QueryUnderstandService()

    result = service.parse_output('{"intent":"totally_unknown","rewrite_query":"Agent essence"}')

    assert result.intent == "kb_search"
    assert result.rewrite_query == "Agent essence"


def test_query_understand_run_falls_back_to_original_query_when_missing_rewrite() -> None:
    @dataclass
    class DummyResponse:
        content: Any

    class DummyModel:
        def invoke(self, _messages: list[Any]) -> DummyResponse:
            return DummyResponse(content='{"intent":"follow_up"}')

    service = QueryUnderstandService(
        chat_model=ModelConfig(model="dummy"),
        model_factory=lambda *_args, **_kwargs: DummyModel(),
    )

    result = service.run(
        query="Original question",
        history=[],
        images=[],
        language="English",
    )

    assert result.intent == "follow_up"
    assert result.rewrite_query == "Original question"


def test_retrieval_result_empty_factory() -> None:
    result = RetrievalResult.empty(query="Agent essence")

    assert result.query == "Agent essence"
    assert result.result_status == "empty"
    assert result.source_count == 0
    assert result.chunk_count == 0


def test_deposit_result_from_pipeline_result_preserves_doc_metadata() -> None:
    pipeline_result = PipelineDepositResult(
        status="completed",
        message="Stored successfully",
        draft=KnowledgeDraft(
            source_type="link",
            source_uri="https://example.com/source",
            source_title="Example Source",
            author="",
            published_at="",
            raw_content_markdown="# Example",
            summary_markdown="Summary",
        ),
        final_markdown="# Final",
        local_document_id="doc-123",
        feishu_doc_url="https://example.com/doc",
        wiki_node_token="wiki-node-456",
    )

    result = DepositResult.from_pipeline_result(pipeline_result)

    assert result.result_status == "completed"
    assert result.source_type == "link"
    assert result.feishu_doc_url == "https://example.com/doc"
    assert result.wiki_node_token == "wiki-node-456"


def test_retrieval_request_from_query_normalizes_whitespace() -> None:
    request = RetrievalRequest.from_query("  Agent   essence  ")

    assert request.query == "Agent essence"
    assert request.with_sources is True


def test_deposit_request_context_from_inputs_filters_blank_image_paths() -> None:
    context = DepositRequestContext.from_inputs(
        text="  store this  ",
        image_paths_json='["a.png", "", "  ", "b.png"]',
    )

    assert context.text == "store this"
    assert context.image_paths == ["a.png", "b.png"]


def test_deposit_request_context_from_inputs_recovers_from_invalid_json() -> None:
    context = DepositRequestContext.from_inputs(
        text="store this",
        image_paths_json="{not json}",
    )

    assert context.image_paths == []


def test_message_context_from_dict_normalizes_nested_protocol_objects() -> None:
    context = MessageContext.from_dict(
        {
            "chat_type": "group",
            "raw_text": "@_user_1 hello",
            "normalized_text": "@张三 hello",
            "bot_mentioned": True,
            "mentioned_users": ["张三"],
            "mentions": [
                {"display_name": "知识库机器人", "open_id": "ou_bot_123", "is_bot": True},
                {"display_name": "张三", "open_id": "ou_user_456", "is_bot": False},
            ],
            "reply_context": {
                "is_reply": True,
                "parent_id": "om_parent_123",
                "root_id": "om_root_123",
                "parent_text_preview": "上一条在讨论 Agent",
                "parent_role": "assistant",
            },
        }
    )

    assert context.chat_type == "group"
    assert context.bot_mentioned is True
    assert context.mention_refs[1].display_name == "张三"
    assert context.reply_context is not None
    assert context.reply_context.parent_role == "assistant"


def test_query_understand_parse_output_logs_schema_warning_for_text_fallback(caplog) -> None:
    service = QueryUnderstandService()

    caplog.set_level(logging.WARNING, logger="feishu_wiki_rag_agent.events")

    result = service.parse_output("plain text answer")

    assert result.rewrite_query == "plain text answer"
    record = next(record for record in caplog.records if getattr(record, "event", "") == "schema_normalization_warning")
    assert record.schema_stage == "query_understand"
    assert record.reason == "json_parse_failed"


def test_controller_decision_logs_schema_warning_for_invalid_intent(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="feishu_wiki_rag_agent.events")

    decision = ControllerDecision.from_dict(
        {
            "intent": "not_real",
            "allow_retrieval": "yes",
            "rewrite_query": "Agent essence",
        }
    )

    assert decision.intent == "kb_search"
    record = next(record for record in caplog.records if getattr(record, "event", "") == "schema_normalization_warning")
    assert record.schema_stage == "controller_decision"
    assert record.reason == "invalid_intent"


def test_message_context_logs_schema_warning_for_invalid_reply_context(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="feishu_wiki_rag_agent.events")

    context = MessageContext.from_dict(
        {
            "chat_type": "group",
            "reply_context": {
                "is_reply": True,
                "parent_text_preview": "missing ids",
            },
        }
    )

    assert context.reply_context is None
    record = next(record for record in caplog.records if getattr(record, "event", "") == "schema_normalization_warning")
    assert record.schema_stage == "message_context"
    assert record.reason == "invalid_reply_context"


def test_message_context_logs_schema_warning_for_non_dict_mentions(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="feishu_wiki_rag_agent.events")

    context = MessageContext.from_dict(
        {
            "mentions": [
                {"display_name": "张三", "open_id": "ou_user_123", "is_bot": False},
                "bad-mention",
            ],
        }
    )

    assert len(context.mention_refs) == 1
    record = next(record for record in caplog.records if getattr(record, "event", "") == "schema_normalization_warning")
    assert record.schema_stage == "message_context"
    assert record.reason == "invalid_mentions_skipped"
