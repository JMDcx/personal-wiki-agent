from __future__ import annotations

from protocols.renderers import render_message_context_lines


def test_render_message_context_lines_include_group_recent_turns():
    lines = render_message_context_lines(
        {
            "chat_type": "group",
            "group_thread_id": "feishu:oc_group",
            "group_recent_turns": [
                {
                    "sender_open_id": "ou_a",
                    "question": "previous question",
                    "answer": "previous answer",
                }
            ],
        }
    )

    assert "group_thread_id: feishu:oc_group" in lines
    assert "group_recent_turns:" in lines
    assert "- sender: ou_a; user: previous question; assistant: previous answer" in lines
