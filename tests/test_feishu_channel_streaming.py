from __future__ import annotations

import json

from channel.feishu.feishu_channel import FeishuChannel, MessageDeduper
from config import Settings
from protocols.streaming import StreamEvent


class _FakeClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str]] = []
        self.updates: list[tuple[str, str]] = []

    def reply_text(self, message_id: str, text: str) -> str:
        self.replies.append((message_id, text))
        return "om_stream"

    def update_text(self, message_id: str, text: str) -> None:
        self.updates.append((message_id, text))


def _event(message_id: str = "om_source") -> dict:
    return {
        "message": {
            "message_id": message_id,
            "chat_id": "oc_chat",
            "message_type": "text",
            "chat_type": "p2p",
            "content": json.dumps({"text": "你好"}, ensure_ascii=False),
        },
        "sender": {"sender_id": {"open_id": "ou_user"}},
    }


def test_feishu_channel_uses_streaming_runner_when_enabled():
    client = _FakeClient()
    stream_calls: list[tuple[str, str, dict[str, object]]] = []

    def _sync_runner(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("sync runner should not be called")

    def _stream_runner(text: str, thread_id: str, message_context: dict[str, object]):
        stream_calls.append((text, thread_id, message_context))
        yield StreamEvent.status("正在生成回复...", stage="generation")
        yield StreamEvent.final("流式答案")

    channel = FeishuChannel(
        settings=Settings(feishu_streaming_enabled=True),
        client=client,
        agent_runner=_sync_runner,
        streaming_agent_runner=_stream_runner,
        deduper=MessageDeduper(),
    )

    answer = channel.handle_event(_event())

    assert answer == "流式答案"
    assert stream_calls[0][0] == "你好"
    assert stream_calls[0][1] == "feishu:oc_chat"
    assert client.replies == [("om_source", "thinking...")]
    assert client.updates[-1] == ("om_stream", "流式答案")


def test_feishu_channel_keeps_sync_runner_when_streaming_disabled():
    client = _FakeClient()

    def _sync_runner(text: str, thread_id: str, message_context: dict[str, object]):
        return f"同步答案：{text}:{thread_id}:{message_context['chat_type']}"

    def _stream_runner(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("stream runner should not be called")

    channel = FeishuChannel(
        settings=Settings(feishu_streaming_enabled=False),
        client=client,
        agent_runner=_sync_runner,
        streaming_agent_runner=_stream_runner,
        deduper=MessageDeduper(),
    )

    answer = channel.handle_event(_event("om_sync"))

    assert answer == "同步答案：你好:feishu:oc_chat:p2p"
    assert client.replies == [("om_sync", "同步答案：你好:feishu:oc_chat:p2p")]
    assert client.updates == []
