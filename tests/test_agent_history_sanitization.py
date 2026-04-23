from __future__ import annotations

import agent as agent_module
from agent import (
    _build_controller_context,
    _extract_history_user_text,
    _group_memory_history_turns,
    _merge_history_turns,
)
from config import Settings
from multimodal_rag_agent.config import get_multimodal_settings
from multimodal_rag_agent.rag_query_pipeline.query_understand_service import HistoryTurn, QueryUnderstandResult


def test_extract_history_user_text_uses_last_current_question():
    nested = """
    [Runtime Metadata - for assistant control, not for direct user display]
    intent: greeting
    allow_retrieval: no
    rewrite_query: 你好
    recent_history:
    - User: [Runtime Metadata - for assistant control, not for direct user display]
      intent: kb_search
      allow_retrieval: yes
      rewrite_query: 能否总结知识库中关于意图识别的相关知识？
      Current user question:
      我最近准备实习，你总结一下知识库里面关于意图识别的知识吧
    - Assistant: 这是上一轮很长的回答
    Current user question:
    你好
    """

    cleaned, had_metadata = _extract_history_user_text(nested)

    assert had_metadata is True
    assert cleaned == "你好"


def test_extract_history_user_text_keeps_plain_text_untouched():
    cleaned, had_metadata = _extract_history_user_text("你好")

    assert had_metadata is False
    assert cleaned == "你好"


def test_group_memory_history_turns_include_sender_identity():
    turns = _group_memory_history_turns(
        {
            "group_recent_turns": [
                {
                    "sender_open_id": "ou_a",
                    "question": "previous group question",
                    "answer": "previous group answer",
                }
            ]
        }
    )

    assert turns == [
        HistoryTurn(
            user_question="[群成员 ou_a] previous group question",
            assistant_answer="previous group answer",
        )
    ]


def test_merge_history_turns_deduplicates_and_keeps_recent_limit():
    group_turn = HistoryTurn("[群成员 ou_a] question", "answer")
    duplicate = HistoryTurn("[群成员 ou_a] question", "answer")
    personal_turn = HistoryTurn("personal question", "personal answer")

    merged = _merge_history_turns([group_turn], [duplicate, personal_turn], limit=2)

    assert merged == [group_turn, personal_turn]


def test_merge_history_turns_returns_empty_when_limit_is_zero():
    merged = _merge_history_turns([HistoryTurn("question", "answer")], [], limit=0)

    assert merged == []


def test_build_controller_context_includes_legacy_group_thread_history(monkeypatch):
    captured_history: list[HistoryTurn] = []
    legacy_turn = HistoryTurn("legacy group question", "legacy group answer")
    member_turn = HistoryTurn("member question", "member answer")

    def _fake_load_history(_agent, thread_id: str, limit: int = 5) -> list[HistoryTurn]:
        if thread_id == "feishu:oc_group":
            return [legacy_turn]
        if thread_id == "feishu:oc_group:ou_a":
            return [member_turn]
        return []

    class _FakeUnderstandService:
        def run(self, *, query, history, images, language, chat_model_supports_vision, vlm_model):  # noqa: ANN001
            captured_history[:] = list(history)
            return QueryUnderstandResult(rewrite_query=query, intent="kb_search")

    monkeypatch.setattr(agent_module, "_load_history_from_runtime", _fake_load_history)
    monkeypatch.setattr(agent_module, "_create_query_understand_service", lambda _settings: _FakeUnderstandService())

    context = _build_controller_context(
        question="tell me about the deployment policy",
        thread_id="feishu:oc_group:ou_a",
        images=[],
        message_context={"chat_type": "group", "group_thread_id": "feishu:oc_group"},
        language="中文",
        settings=Settings(),
        multimodal_settings=get_multimodal_settings(),
        agent_runtime=object(),
    )

    assert context.history == [legacy_turn, member_turn]
    assert captured_history == [legacy_turn, member_turn]
