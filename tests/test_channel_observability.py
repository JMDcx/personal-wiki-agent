from __future__ import annotations

import json
import logging
import pytest

from feishu_wiki_rag_agent.channel.feishu.feishu_channel import FeishuChannel
from feishu_wiki_rag_agent.channel.weixin.weixin_channel import WeixinChannel
from feishu_wiki_rag_agent.config import Settings


class DummyFeishuClient:
    def reply_text(self, message_id: str, text: str) -> None:
        raise AssertionError("reply_text should not be called in this test")


class RecordingFeishuClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str]] = []

    def reply_text(self, message_id: str, text: str) -> None:
        self.replies.append((message_id, text))

    def get_message(self, message_id: str) -> dict:
        raise AssertionError(f"get_message should not be called in this test: {message_id}")


class ReplyContextFeishuClient(RecordingFeishuClient):
    def __init__(self, message_payload: dict) -> None:
        super().__init__()
        self.message_payload = message_payload
        self.queries: list[str] = []

    def get_message(self, message_id: str) -> dict:
        self.queries.append(message_id)
        return self.message_payload


class DummyDocreader:
    pass


class FakePollingApi:
    def __init__(self) -> None:
        self.calls = 0

    def get_updates(self, cursor: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "ret": 0,
                "errcode": 0,
                "get_updates_buf": "cursor-1",
                "msgs": [
                    {
                        "message_type": "1",
                        "message_id": "wx_msg_123",
                        "from_user_id": "wx_user_456",
                        "context_token": "ctx_789",
                    }
                ],
            }
        raise KeyboardInterrupt()


def build_settings(tmp_path) -> Settings:
    settings = Settings(
        rag_data_dir=tmp_path / "data",
        weixin_credentials_path=tmp_path / "weixin" / "credentials.json",
        weixin_tmp_dir=tmp_path / "weixin" / "tmp",
        checkpoint_db_path=tmp_path / "deepagents" / "checkpoints.sqlite",
        log_file_path=tmp_path / "logs" / "app.jsonl",
    )
    settings.ensure_directories()
    return settings


def test_weixin_reply_skipped_emits_structured_event(tmp_path, caplog) -> None:
    channel = WeixinChannel(
        settings=build_settings(tmp_path),
        agent_runner=lambda *_args, **_kwargs: "unused",
        docreader=DummyDocreader(),
    )

    caplog.set_level(logging.WARNING, logger="feishu_wiki_rag_agent.events")

    channel._reply_text("user-1", "", "hello")

    record = next(record for record in caplog.records if getattr(record, "event", "") == "reply_skipped")
    assert record.reply_channel == "weixin"
    assert record.reason == "missing_reply_target"
    assert record.has_to_user_id is True
    assert record.has_context_token is False


def test_weixin_load_credentials_emits_structured_exception(tmp_path, caplog) -> None:
    settings = build_settings(tmp_path)
    settings.weixin_credentials_path.write_text("{broken json")
    channel = WeixinChannel(
        settings=settings,
        agent_runner=lambda *_args, **_kwargs: "unused",
        docreader=DummyDocreader(),
    )

    caplog.set_level(logging.ERROR, logger="feishu_wiki_rag_agent.events")

    assert channel._load_credentials() == {}

    record = next(record for record in caplog.records if getattr(record, "event", "") == "credentials_load_failed")
    assert record.stage == "weixin_load_credentials"
    assert record.channel == "weixin"
    assert record.credentials_path.endswith("credentials.json")


def test_feishu_websocket_dispatch_failure_is_structured(tmp_path, caplog) -> None:
    settings = build_settings(tmp_path)
    channel = FeishuChannel(
        settings=settings,
        client=DummyFeishuClient(),
        agent_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    payload = json.dumps(
        {
            "event": {
                "message": {
                    "message_id": "om_msg_123",
                    "chat_id": "oc_chat_456",
                    "message_type": "text",
                    "chat_type": "p2p",
                    "content": json.dumps({"text": "hello"}),
                },
                "sender": {"sender_id": {"open_id": "ou_sender_789"}},
            }
        }
    )

    caplog.set_level(logging.ERROR, logger="feishu_wiki_rag_agent.events")

    channel._handle_websocket_payload(payload)

    record = next(record for record in caplog.records if getattr(record, "event", "") == "request_failed")
    assert record.stage == "feishu_handle_event"
    assert record.channel == "feishu"
    assert record.message_id == "om_msg_123"
    assert record.chat_id == "oc_chat_456"
    assert record.error == "boom"

    summary = next(record for record in caplog.records if getattr(record, "event", "") == "request_summary")
    assert summary.status == "error"
    assert summary.stage == "feishu_handle_event"


def test_feishu_group_mentions_preserve_non_bot_context() -> None:
    channel = FeishuChannel(
        settings=Settings(),
        client=DummyFeishuClient(),
        agent_runner=lambda *_args, **_kwargs: "unused",
    )
    channel.bot_open_id = "ou_bot_123"

    event = {
        "message": {
            "message_id": "om_group_msg_123",
            "chat_id": "oc_group_chat_456",
            "message_type": "text",
            "chat_type": "group",
            "content": json.dumps(
                {
                    "text": "@_user_1 刚刚@_user_2问的问题我很感兴趣，我想再深入询问一下Agent的本质",
                }
            ),
            "mentions": [
                {"name": "知识库机器人", "id": {"open_id": "ou_bot_123"}},
                {"name": "张三", "id": {"open_id": "ou_user_456"}},
            ],
        },
        "sender": {"sender_id": {"open_id": "ou_sender_789"}},
    }

    incoming = channel._parse_incoming_message(event)

    assert incoming is not None
    assert incoming.raw_text == "@_user_1 刚刚@_user_2问的问题我很感兴趣，我想再深入询问一下Agent的本质"
    assert incoming.text == "刚刚@张三问的问题我很感兴趣，我想再深入询问一下Agent的本质"
    assert incoming.bot_mentioned is True
    assert incoming.mentioned_users == ["张三"]
    assert incoming.to_message_context()["mentions"] == [
        {"display_name": "知识库机器人", "open_id": "ou_bot_123", "is_bot": True},
        {"display_name": "张三", "open_id": "ou_user_456", "is_bot": False},
    ]


def test_feishu_group_request_summary_keeps_mention_fields(tmp_path, caplog) -> None:
    client = RecordingFeishuClient()
    settings = build_settings(tmp_path)
    channel = FeishuChannel(
        settings=settings,
        client=client,
        agent_runner=lambda *_args, **_kwargs: "好的，我继续展开讲一下。",
    )
    channel.bot_open_id = "ou_bot_123"

    event = {
        "message": {
            "message_id": "om_group_msg_summary",
            "chat_id": "oc_group_chat_summary",
            "message_type": "text",
            "chat_type": "group",
            "content": json.dumps(
                {
                    "text": "@_user_1 刚刚@_user_2问的问题我很感兴趣，我想再深入询问一下Agent的本质",
                }
            ),
            "mentions": [
                {"name": "知识库机器人", "id": {"open_id": "ou_bot_123"}},
                {"name": "张三", "id": {"open_id": "ou_user_456"}},
            ],
        },
        "sender": {"sender_id": {"open_id": "ou_sender_789"}},
    }

    caplog.set_level(logging.INFO, logger="feishu_wiki_rag_agent.events")

    answer = channel.handle_event(event)

    assert answer == "好的，我继续展开讲一下。"
    assert client.replies == [("om_group_msg_summary", "好的，我继续展开讲一下。")]

    summary = next(record for record in caplog.records if getattr(record, "event", "") == "request_summary")
    assert summary.status == "ok"
    assert summary.bot_mentioned is True
    assert summary.mentioned_users == ["张三"]
    assert summary.mention_count == 2


def test_feishu_reply_context_fetches_parent_preview() -> None:
    client = ReplyContextFeishuClient(
        {
            "message_id": "om_parent_123",
            "message_type": "text",
            "sender": {
                "sender_id": {"open_id": "ou_bot_123"},
                "sender_type": "app",
                "sender_name": "知识库机器人",
            },
            "content": json.dumps({"text": "上一个问题在讨论 Agent 的本质"}),
        }
    )
    channel = FeishuChannel(
        settings=Settings(),
        client=client,
        agent_runner=lambda *_args, **_kwargs: "unused",
    )
    channel.bot_open_id = "ou_bot_123"

    event = {
        "message": {
            "message_id": "om_group_reply_123",
            "chat_id": "oc_group_chat_456",
            "message_type": "text",
            "chat_type": "group",
            "root_id": "om_root_123",
            "parent_id": "om_parent_123",
            "content": json.dumps(
                {
                    "text": "@_user_1 我想继续追问刚才那一点",
                }
            ),
            "mentions": [
                {"name": "知识库机器人", "id": {"open_id": "ou_bot_123"}},
            ],
        },
        "sender": {"sender_id": {"open_id": "ou_sender_789"}},
    }

    incoming = channel._parse_incoming_message(event)

    assert incoming is not None
    assert client.queries == ["om_parent_123"]
    assert incoming.reply_context is not None
    assert incoming.reply_context.to_dict() == {
        "is_reply": True,
        "parent_id": "om_parent_123",
        "root_id": "om_root_123",
        "parent_text_preview": "上一个问题在讨论 Agent 的本质",
        "parent_message_type": "text",
        "parent_role": "assistant",
        "parent_sender_name": "知识库机器人",
    }


def test_weixin_poll_loop_message_failure_is_structured(tmp_path, caplog) -> None:
    channel = WeixinChannel(
        settings=build_settings(tmp_path),
        api=FakePollingApi(),
        agent_runner=lambda *_args, **_kwargs: "unused",
        docreader=DummyDocreader(),
    )
    channel.handle_raw_message = lambda _raw_msg: (_ for _ in ()).throw(RuntimeError("poll boom"))  # type: ignore[method-assign]

    caplog.set_level(logging.ERROR, logger="feishu_wiki_rag_agent.events")

    with pytest.raises(KeyboardInterrupt):
        channel._poll_loop()

    record = next(
        record
        for record in caplog.records
        if getattr(record, "event", "") == "request_failed" and getattr(record, "stage", "") == "weixin_poll_loop_message"
    )
    assert record.channel == "weixin"
    assert record.message_id == "wx_msg_123"
    assert record.from_user_id == "wx_user_456"
    assert record.has_context_token is True
    assert record.error == "poll boom"
