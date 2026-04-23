from __future__ import annotations

from dataclasses import dataclass

from channel.feishu.feishu_client import FeishuClient
from channel.feishu.streaming import FeishuStreamingResponder
from config import Settings
from protocols.streaming import StreamEvent


class _RecordingClient(FeishuClient):
    def __init__(self) -> None:
        super().__init__(Settings())
        self.calls: list[tuple[str, str, dict | None]] = []

    def _request(self, method, path, *, params=None, json_body=None, headers=None, auth=True):  # noqa: ANN001
        self.calls.append((method, path, json_body))
        if path.endswith("/reply"):
            return {"data": {"message_id": "om_reply"}}
        return {"data": {}}


def test_feishu_client_reply_text_returns_created_message_id():
    client = _RecordingClient()

    message_id = client.reply_text("om_source", "正在处理...")

    assert message_id == "om_reply"
    assert client.calls == [
        (
            "POST",
            "/open-apis/im/v1/messages/om_source/reply",
            {"content": '{"text": "正在处理..."}', "msg_type": "text"},
        )
    ]


def test_feishu_client_patch_text_updates_existing_message():
    client = _RecordingClient()

    client.patch_text("om_reply", "新的内容")

    assert client.calls == [
        (
            "PATCH",
            "/open-apis/im/v1/messages/om_reply",
            {"content": '{"text": "新的内容"}'},
        )
    ]


@dataclass
class _FakeFeishuClient:
    replies: list[tuple[str, str]]
    patches: list[tuple[str, str]]

    def reply_text(self, message_id: str, text: str) -> str:
        self.replies.append((message_id, text))
        return "om_stream"

    def patch_text(self, message_id: str, text: str) -> None:
        self.patches.append((message_id, text))


def test_streaming_responder_updates_placeholder_and_returns_final_answer():
    client = _FakeFeishuClient(replies=[], patches=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=999,
        now_provider=lambda: 100.0,
    )

    answer = responder.run(
        [
            StreamEvent.started("开始处理"),
            StreamEvent.status("正在理解问题...", stage="intent"),
            StreamEvent.text_delta("片段一"),
            StreamEvent.text_delta("片段二"),
            StreamEvent.final("完整答案"),
        ]
    )

    assert answer == "完整答案"
    assert client.replies == [("om_source", "正在处理...")]
    assert client.patches[0] == ("om_stream", "状态：正在理解问题...")
    assert client.patches[-1] == ("om_stream", "完整答案")


def test_streaming_responder_falls_back_to_final_reply_when_patch_fails():
    class _PatchFailingClient(_FakeFeishuClient):
        def patch_text(self, message_id: str, text: str) -> None:
            raise RuntimeError("patch failed")

    client = _PatchFailingClient(replies=[], patches=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=0,
        now_provider=lambda: 100.0,
    )

    answer = responder.run([StreamEvent.status("正在处理..."), StreamEvent.final("最终答案")])

    assert answer == "最终答案"
    assert client.replies == [
        ("om_source", "正在处理..."),
        ("om_source", "最终答案"),
    ]


def test_streaming_responder_falls_back_when_placeholder_id_is_missing():
    class _NoReplyIdClient(_FakeFeishuClient):
        def reply_text(self, message_id: str, text: str) -> str:
            self.replies.append((message_id, text))
            return ""

    client = _NoReplyIdClient(replies=[], patches=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=0,
        now_provider=lambda: 100.0,
    )

    answer = responder.run([StreamEvent.status("正在处理..."), StreamEvent.final("最终答案")])

    assert answer == "最终答案"
    assert client.replies == [
        ("om_source", "正在处理..."),
        ("om_source", "最终答案"),
    ]
    assert client.patches == []
