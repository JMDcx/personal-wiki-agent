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

    message_id = client.reply_text("om_source", "thinking...")

    assert message_id == "om_reply"
    assert client.calls == [
        (
            "POST",
            "/open-apis/im/v1/messages/om_source/reply",
            {"content": '{"text": "thinking..."}', "msg_type": "text"},
        )
    ]


def test_feishu_client_update_text_edits_existing_text_message():
    client = _RecordingClient()

    client.update_text("om_reply", "新的内容")

    assert client.calls == [
        (
            "PUT",
            "/open-apis/im/v1/messages/om_reply",
            {"content": '{"text": "新的内容"}', "msg_type": "text"},
        )
    ]


@dataclass
class _FakeFeishuClient:
    replies: list[tuple[str, str]]
    updates: list[tuple[str, str]]

    def reply_text(self, message_id: str, text: str) -> str:
        self.replies.append((message_id, text))
        return "om_stream"

    def update_text(self, message_id: str, text: str) -> None:
        self.updates.append((message_id, text))


def test_streaming_responder_updates_placeholder_and_returns_final_answer():
    client = _FakeFeishuClient(replies=[], updates=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=999,
        now_provider=lambda: 100.0,
    )

    answer = responder.run(
        [
            StreamEvent.started("开始思考"),
            StreamEvent.status("正在理解问题...", stage="intent"),
            StreamEvent.text_delta("片段一"),
            StreamEvent.text_delta("片段二"),
            StreamEvent.final("完整答案"),
        ]
    )

    assert answer == "完整答案"
    assert client.replies == [("om_source", "thinking...")]
    assert client.updates[0] == ("om_stream", "thinking...\n当前进度：正在理解问题...")
    assert client.updates[-1] == ("om_stream", "完整答案")


def test_streaming_responder_reuses_existing_placeholder_message():
    client = _FakeFeishuClient(replies=[], updates=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        initial_reply_message_id="om_existing",
        update_interval_seconds=0,
        now_provider=lambda: 100.0,
    )

    answer = responder.run([StreamEvent.status("正在生成回复..."), StreamEvent.final("完整答案")])

    assert answer == "完整答案"
    assert client.replies == []
    assert client.updates[-1] == ("om_existing", "完整答案")


def test_streaming_responder_falls_back_to_final_reply_when_update_fails():
    class _UpdateFailingClient(_FakeFeishuClient):
        def update_text(self, message_id: str, text: str) -> None:
            raise RuntimeError("update failed")

    client = _UpdateFailingClient(replies=[], updates=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=0,
        now_provider=lambda: 100.0,
    )

    answer = responder.run([StreamEvent.status("thinking..."), StreamEvent.final("最终答案")])

    assert answer == "最终答案"
    assert client.replies == [
        ("om_source", "thinking..."),
        ("om_source", "最终答案"),
    ]


def test_streaming_responder_falls_back_when_placeholder_id_is_missing():
    class _NoReplyIdClient(_FakeFeishuClient):
        def reply_text(self, message_id: str, text: str) -> str:
            self.replies.append((message_id, text))
            return ""

    client = _NoReplyIdClient(replies=[], updates=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=0,
        now_provider=lambda: 100.0,
    )

    answer = responder.run([StreamEvent.status("thinking..."), StreamEvent.final("最终答案")])

    assert answer == "最终答案"
    assert client.replies == [
        ("om_source", "thinking..."),
        ("om_source", "最终答案"),
    ]
    assert client.updates == []


def test_streaming_responder_uses_no_result_text_for_empty_final_event():
    client = _FakeFeishuClient(replies=[], updates=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=0,
        now_provider=lambda: 100.0,
    )

    answer = responder.run([StreamEvent.status("正在生成回复..."), StreamEvent.final("")])

    assert answer == "当前索引中未找到相关内容。"
    assert client.updates[-1] == ("om_stream", "当前索引中未找到相关内容。")


def test_streaming_responder_limits_non_final_update_count_but_keeps_final_update():
    client = _FakeFeishuClient(replies=[], updates=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=0,
        max_update_count=2,
        now_provider=lambda: 100.0,
    )

    answer = responder.run(
        [
            StreamEvent.status("正在生成回复..."),
            StreamEvent.text_delta("一"),
            StreamEvent.text_delta("二"),
            StreamEvent.text_delta("三"),
            StreamEvent.final("完整答案"),
        ]
    )

    assert answer == "完整答案"
    assert len(client.updates) == 3
    assert client.updates[-1] == ("om_stream", "完整答案")


def test_streaming_responder_falls_back_when_stream_exceeds_max_seconds():
    timestamps = iter([100.0, 100.0, 401.0])
    client = _FakeFeishuClient(replies=[], updates=[])
    responder = FeishuStreamingResponder(
        client=client,
        source_message_id="om_source",
        update_interval_seconds=0,
        max_stream_seconds=300,
        now_provider=lambda: next(timestamps),
    )

    answer = responder.run([StreamEvent.status("正在生成回复..."), StreamEvent.final("最终答案")])

    assert answer == "最终答案"
    assert client.updates == [("om_stream", "thinking...\n当前进度：正在生成回复...")]
    assert client.replies == [("om_source", "thinking..."), ("om_source", "最终答案")]
