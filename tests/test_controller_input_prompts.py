from __future__ import annotations

from multimodal_rag_agent.rag_query_pipeline.controller_input_prompts import render_controller_user_input
from multimodal_rag_agent.rag_query_pipeline.query_understand_service import HistoryTurn, QueryUnderstandResult


def test_render_controller_user_input_includes_group_mention_context() -> None:
    content = render_controller_user_input(
        question="刚刚@张三问的问题我很感兴趣，我想再深入询问一下Agent的本质",
        understand_result=QueryUnderstandResult(
            rewrite_query="Agent的本质",
            intent="kb_search",
            raw_output='{"intent":"kb_search"}',
        ),
        history=[
            HistoryTurn(
                user_question="上一个问题是什么？",
                assistant_answer="上一个问题在讨论Agent和工作流的区别。",
            )
        ],
        allow_retrieval=True,
        message_context={
            "chat_type": "group",
            "bot_mentioned": True,
            "mentioned_users": ["张三"],
            "mentions": [
                {"display_name": "知识库机器人", "open_id": "ou_bot_123", "is_bot": True},
                {"display_name": "张三", "open_id": "ou_user_456", "is_bot": False},
            ],
        },
    )

    assert "Current user question:" in content
    assert "刚刚@张三问的问题我很感兴趣" in content
    assert "message_context:" in content
    assert "chat_type: group" in content
    assert "bot_mentioned: yes" in content
    assert "mentioned_users: 张三" in content
    assert "mention_details: 张三(ou_user_456)" in content


def test_render_controller_user_input_includes_reply_context() -> None:
    content = render_controller_user_input(
        question="我想继续追问刚才那一点",
        understand_result=QueryUnderstandResult(
            rewrite_query="Agent 的本质",
            intent="kb_search",
            raw_output='{"intent":"kb_search"}',
        ),
        history=[],
        allow_retrieval=True,
        message_context={
            "chat_type": "group",
            "reply_context": {
                "is_reply": True,
                "parent_id": "om_parent_123",
                "root_id": "om_root_123",
                "parent_text_preview": "上一个问题在讨论 Agent 的本质",
                "parent_message_type": "text",
                "parent_role": "assistant",
                "parent_sender_name": "知识库机器人",
            },
        },
    )

    assert "reply_context:" in content
    assert "parent_id: om_parent_123" in content
    assert "root_id: om_root_123" in content
    assert "parent_message_type: text" in content
    assert "parent_role: assistant" in content
    assert "parent_sender_name: 知识库机器人" in content
    assert "parent_text_preview: 上一个问题在讨论 Agent 的本质" in content
