from __future__ import annotations

from channel.feishu.group_memory import FeishuGroupMemoryStore


def test_group_memory_store_appends_and_reads_recent_turns(tmp_path):
    store = FeishuGroupMemoryStore(tmp_path, max_recent_turns=2)

    store.append_turn(
        chat_id="oc_group",
        sender_open_id="ou_a",
        message_id="om_1",
        question="question 1",
        answer="answer 1",
    )
    store.append_turn(
        chat_id="oc_group",
        sender_open_id="ou_b",
        message_id="om_2",
        question="question 2",
        answer="answer 2",
    )
    store.append_turn(
        chat_id="oc_group",
        sender_open_id="ou_c",
        message_id="om_3",
        question="question 3",
        answer="answer 3",
    )

    turns = store.recent_turns("oc_group", limit=2)

    assert [turn["message_id"] for turn in turns] == ["om_2", "om_3"]
    assert turns[-1]["sender_open_id"] == "ou_c"
    assert turns[-1]["question"] == "question 3"
    assert turns[-1]["answer"] == "answer 3"
    assert turns[-1]["created_at"]


def test_group_memory_store_isolates_chats(tmp_path):
    store = FeishuGroupMemoryStore(tmp_path)

    store.append_turn(
        chat_id="oc_a",
        sender_open_id="ou_user",
        message_id="om_a",
        question="question a",
        answer="answer a",
    )
    store.append_turn(
        chat_id="oc_b",
        sender_open_id="ou_user",
        message_id="om_b",
        question="question b",
        answer="answer b",
    )

    assert [turn["message_id"] for turn in store.recent_turns("oc_a")] == ["om_a"]
    assert [turn["message_id"] for turn in store.recent_turns("oc_b")] == ["om_b"]
