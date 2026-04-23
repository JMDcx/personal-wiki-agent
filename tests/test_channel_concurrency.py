from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from channel.dispatcher import DispatchResult
from channel.feishu.feishu_channel import BUSY_REPLY_TEXT as FEISHU_BUSY_REPLY_TEXT
from channel.feishu.feishu_channel import FeishuChannel, MessageDeduper
from channel.weixin.weixin_channel import BUSY_REPLY_TEXT as WEIXIN_BUSY_REPLY_TEXT
from channel.weixin.weixin_channel import WeixinChannel
from config import Settings


@dataclass
class _FakeDispatch:
    accepted: bool = True
    submissions: list[tuple[str, Callable[[], object]]] | None = None

    def __post_init__(self) -> None:
        if self.submissions is None:
            self.submissions = []

    def submit(self, thread_id: str, fn: Callable[[], object], *args, on_rejected=None, **kwargs):  # noqa: ANN002, ANN003
        assert self.submissions is not None
        self.submissions.append((thread_id, fn))
        if not self.accepted:
            if on_rejected is not None:
                on_rejected()
            return DispatchResult(accepted=False, rejected_reason="queue_full")
        return DispatchResult(accepted=True)

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        return None


class _FakeFeishuClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str]] = []

    def reply_text(self, message_id: str, text: str) -> str:
        self.replies.append((message_id, text))
        return "om_reply"

    def patch_text(self, message_id: str, text: str) -> None:
        return None


class _FakeWeixinApi:
    cdn_base_url = "https://cdn.example"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send_text(self, *, to_user_id: str, text: str, context_token: str) -> None:
        self.sent.append((to_user_id, text, context_token))


def _feishu_payload(message_id: str = "om_source", chat_id: str = "oc_chat") -> str:
    return json.dumps(
        {
            "event": {
                "message": {
                    "message_id": message_id,
                    "chat_id": chat_id,
                    "message_type": "text",
                    "chat_type": "p2p",
                    "content": json.dumps({"text": "你好"}, ensure_ascii=False),
                },
                "sender": {"sender_id": {"open_id": "ou_user"}},
            }
        },
        ensure_ascii=False,
    )


def test_feishu_websocket_payload_submits_to_dispatcher_when_enabled():
    dispatcher = _FakeDispatch()
    client = _FakeFeishuClient()
    channel = FeishuChannel(
        settings=Settings(bot_concurrency_enabled=True, feishu_streaming_enabled=False),
        client=client,
        agent_runner=lambda *_args, **_kwargs: "答案",
        deduper=MessageDeduper(),
        dispatcher=dispatcher,
    )

    channel._handle_websocket_payload(_feishu_payload())

    assert dispatcher.submissions is not None
    assert len(dispatcher.submissions) == 1
    assert dispatcher.submissions[0][0] == "feishu:oc_chat"
    assert client.replies == []


def test_feishu_dispatch_rejection_replies_busy_message():
    dispatcher = _FakeDispatch(accepted=False)
    client = _FakeFeishuClient()
    channel = FeishuChannel(
        settings=Settings(bot_concurrency_enabled=True),
        client=client,
        agent_runner=lambda *_args, **_kwargs: "答案",
        deduper=MessageDeduper(),
        dispatcher=dispatcher,
    )

    channel._handle_websocket_payload(_feishu_payload())

    assert client.replies == [("om_source", FEISHU_BUSY_REPLY_TEXT)]


def test_weixin_raw_message_submits_to_dispatcher_when_enabled():
    dispatcher = _FakeDispatch()
    api = _FakeWeixinApi()
    channel = WeixinChannel(
        settings=Settings(bot_concurrency_enabled=True),
        api=api,
        agent_runner=lambda *_args, **_kwargs: "答案",
        dispatcher=dispatcher,
    )

    channel._dispatch_raw_message(
        {
            "message_type": "1",
            "message_id": "wx_1",
            "from_user_id": "user_a",
            "context_token": "ctx_a",
            "text": "你好",
        }
    )

    assert dispatcher.submissions is not None
    assert len(dispatcher.submissions) == 1
    assert dispatcher.submissions[0][0] == "weixin:user_a"
    assert api.sent == []


def test_weixin_dispatch_rejection_replies_busy_message():
    dispatcher = _FakeDispatch(accepted=False)
    api = _FakeWeixinApi()
    channel = WeixinChannel(
        settings=Settings(bot_concurrency_enabled=True),
        api=api,
        agent_runner=lambda *_args, **_kwargs: "答案",
        dispatcher=dispatcher,
    )

    channel._dispatch_raw_message(
        {
            "message_type": "1",
            "message_id": "wx_1",
            "from_user_id": "user_a",
            "context_token": "ctx_a",
            "text": "你好",
        }
    )

    assert api.sent == [("user_a", WEIXIN_BUSY_REPLY_TEXT, "ctx_a")]
