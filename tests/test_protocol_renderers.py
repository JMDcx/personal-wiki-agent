from __future__ import annotations

from protocols.renderers import (
    render_controller_metadata_lines,
    render_deposit_result_text,
    render_message_context_lines,
    render_mention_details,
    render_reply_context_line,
    render_retrieval_result_text,
)
from protocols.tool_models import DepositResult, RetrievalResult
from schemas import MessageContext
from multimodal_rag_agent.rag_query_pipeline.query_understand_service import HistoryTurn


def test_render_mention_details_skips_bot_mentions() -> None:
    rendered = render_mention_details(
        [
            {"display_name": "Knowledge Bot", "open_id": "ou_bot_123", "is_bot": True},
            {"display_name": "Alice", "open_id": "ou_user_456", "is_bot": False},
            {"display_name": "Bob", "open_id": "", "is_bot": False},
        ]
    )

    assert rendered == "Alice(ou_user_456), Bob"


def test_render_reply_context_line_prefers_role_and_sender() -> None:
    line = render_reply_context_line(
        {
            "parent_role": "assistant",
            "parent_sender_name": "Knowledge Bot",
            "parent_text_preview": "Agent is a loop system",
        }
    )

    assert line == "Reply target: assistant/Knowledge Bot -> Agent is a loop system"


def test_render_retrieval_result_text_includes_sources() -> None:
    result = RetrievalResult(
        query="Agent essence",
        result_status="completed",
        context="Agent is a loop system",
        sources=[{"title": "Agent Guide"}],
        chunk_count=1,
    )

    text = render_retrieval_result_text(result)

    assert "Agent is a loop system" in text
    assert "来源：Agent Guide" in text


def test_render_deposit_result_text_includes_doc_link() -> None:
    result = DepositResult(
        result_status="completed",
        message="Stored successfully",
        source_type="link",
        feishu_doc_url="https://example.com/doc",
    )

    text = render_deposit_result_text(result)

    assert "Stored successfully" in text
    assert "来源类型：link" in text
    assert "https://example.com/doc" in text


def test_render_message_context_lines_renders_structured_summary() -> None:
    context = MessageContext.from_dict(
        {
            "chat_type": "group",
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
                "parent_text_preview": "上一个问题在讨论 Agent 的本质",
                "parent_message_type": "text",
                "parent_role": "assistant",
                "parent_sender_name": "知识库机器人",
            },
        }
    )

    lines = render_message_context_lines(context)

    assert "message_context:" in lines
    assert "chat_type: group" in lines
    assert "bot_mentioned: yes" in lines
    assert "mentioned_users: 张三" in lines
    assert "mention_details: 张三(ou_user_456)" in lines
    assert "reply_context:" in lines
    assert "Reply target: assistant/知识库机器人 -> 上一个问题在讨论 Agent 的本质" in lines


def test_render_controller_metadata_lines_includes_history_and_images() -> None:
    context = MessageContext.from_dict(
        {
            "chat_type": "group",
            "mentioned_users": ["张三"],
        }
    )

    lines = render_controller_metadata_lines(
        intent="kb_search",
        allow_retrieval=True,
        rewrite_query="Agent 的本质",
        image_description="一张关于 Agent 的图",
        images=["a.png"],
        history=[HistoryTurn(user_question="什么是 Agent？", assistant_answer="Agent 是循环系统。")],
        message_context=context,
    )

    assert "intent: kb_search" in lines
    assert "allow_retrieval: yes" in lines
    assert "rewrite_query: Agent 的本质" in lines
    assert "image_description:" in lines
    assert "image_paths:" in lines
    assert "recent_history:" in lines
    assert "message_context:" in lines
