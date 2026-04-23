from __future__ import annotations

from protocols.streaming import StreamEvent


def test_stream_event_helpers_build_stable_payloads():
    event = StreamEvent.status("正在检索知识库...", stage="retrieval", metadata={"top_k": 4})

    assert event.event_type == "status"
    assert event.text == "正在检索知识库..."
    assert event.stage == "retrieval"
    assert event.to_dict() == {
        "event_type": "status",
        "text": "正在检索知识库...",
        "stage": "retrieval",
        "metadata": {"top_k": 4},
    }


def test_stream_event_final_defaults_to_final_stage():
    event = StreamEvent.final("最终答案")

    assert event.event_type == "final"
    assert event.text == "最终答案"
    assert event.stage == "final"
