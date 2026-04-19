from __future__ import annotations

from agent import _extract_history_user_text


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
