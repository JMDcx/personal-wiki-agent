from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from channel.dispatcher import DispatchResult
from channel.feishu.feishu_channel import BUSY_REPLY_TEXT as FEISHU_BUSY_REPLY_TEXT
from channel.feishu.feishu_channel import FeishuChannel, MessageDeduper
from channel.feishu.feishu_channel import THINKING_REPLY_TEXT as FEISHU_THINKING_REPLY_TEXT
from channel.weixin.weixin_channel import BUSY_REPLY_TEXT as WEIXIN_BUSY_REPLY_TEXT
from channel.weixin.weixin_channel import THINKING_REPLY_TEXT as WEIXIN_THINKING_REPLY_TEXT
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
        self.updates: list[tuple[str, str]] = []

    def reply_text(self, message_id: str, text: str) -> str:
        self.replies.append((message_id, text))
        return "om_reply"

    def update_text(self, message_id: str, text: str) -> None:
        self.updates.append((message_id, text))


class _FakeWeixinApi:
    cdn_base_url = "https://cdn.example"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send_text(self, *, to_user_id: str, text: str, context_token: str) -> None:
        self.sent.append((to_user_id, text, context_token))


class _FakeGroupMemoryStore:
    def __init__(self, recent_turns: list[dict[str, object]] | None = None) -> None:
        self.recent = recent_turns or []
        self.recent_calls: list[tuple[str, int]] = []
        self.appended: list[dict[str, str]] = []

    def recent_turns(self, chat_id: str, limit: int) -> list[dict[str, object]]:
        self.recent_calls.append((chat_id, limit))
        return list(self.recent)

    def append_turn(
        self,
        *,
        chat_id: str,
        sender_open_id: str,
        message_id: str,
        question: str,
        answer: str,
    ) -> None:
        self.appended.append(
            {
                "chat_id": chat_id,
                "sender_open_id": sender_open_id,
                "message_id": message_id,
                "question": question,
                "answer": answer,
            }
        )


def _bot_mention() -> list[dict[str, object]]:
    return [{"id": {"open_id": "ou_bot"}, "name": "Bot"}]


def _feishu_payload(
    message_id: str = "om_source",
    chat_id: str = "oc_chat",
    *,
    chat_type: str = "p2p",
    sender_open_id: str = "ou_user",
    text: str = "hello",
    mentions: list[dict[str, object]] | None = None,
) -> str:
    message = {
        "message_id": message_id,
        "chat_id": chat_id,
        "message_type": "text",
        "chat_type": chat_type,
        "content": json.dumps({"text": text}, ensure_ascii=False),
    }
    if mentions is not None:
        message["mentions"] = mentions
    return json.dumps(
        {
            "event": {
                "message": message,
                "sender": {"sender_id": {"open_id": sender_open_id}},
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
    assert FEISHU_THINKING_REPLY_TEXT == "thinking..."
    assert client.replies == [("om_source", "thinking...")]


def test_feishu_group_messages_from_different_senders_use_member_dispatch_threads():
    dispatcher = _FakeDispatch()
    client = _FakeFeishuClient()
    channel = FeishuChannel(
        settings=Settings(bot_concurrency_enabled=True, feishu_streaming_enabled=False),
        client=client,
        agent_runner=lambda *_args, **_kwargs: "answer",
        deduper=MessageDeduper(),
        dispatcher=dispatcher,
    )
    channel.bot_open_id = "ou_bot"

    channel._handle_websocket_payload(
        _feishu_payload(
            message_id="om_a",
            chat_id="oc_group",
            chat_type="group",
            sender_open_id="ou_a",
            text="@_user_1 first",
            mentions=_bot_mention(),
        )
    )
    channel._handle_websocket_payload(
        _feishu_payload(
            message_id="om_b",
            chat_id="oc_group",
            chat_type="group",
            sender_open_id="ou_b",
            text="@_user_1 second",
            mentions=_bot_mention(),
        )
    )

    assert dispatcher.submissions is not None
    assert [thread_id for thread_id, _fn in dispatcher.submissions] == [
        "feishu:oc_group:ou_a",
        "feishu:oc_group:ou_b",
    ]


def test_feishu_group_messages_from_same_sender_keep_same_dispatch_thread():
    dispatcher = _FakeDispatch()
    channel = FeishuChannel(
        settings=Settings(bot_concurrency_enabled=True, feishu_streaming_enabled=False),
        client=_FakeFeishuClient(),
        agent_runner=lambda *_args, **_kwargs: "answer",
        deduper=MessageDeduper(),
        dispatcher=dispatcher,
    )
    channel.bot_open_id = "ou_bot"

    for message_id in ("om_a", "om_b"):
        channel._handle_websocket_payload(
            _feishu_payload(
                message_id=message_id,
                chat_id="oc_group",
                chat_type="group",
                sender_open_id="ou_a",
                text="@_user_1 hello",
                mentions=_bot_mention(),
            )
        )

    assert dispatcher.submissions is not None
    assert [thread_id for thread_id, _fn in dispatcher.submissions] == [
        "feishu:oc_group:ou_a",
        "feishu:oc_group:ou_a",
    ]


def test_feishu_group_concurrency_chat_mode_keeps_existing_group_thread():
    dispatcher = _FakeDispatch()
    channel = FeishuChannel(
        settings=Settings(
            bot_concurrency_enabled=True,
            feishu_streaming_enabled=False,
            feishu_group_concurrency_mode="chat",
        ),
        client=_FakeFeishuClient(),
        agent_runner=lambda *_args, **_kwargs: "answer",
        deduper=MessageDeduper(),
        dispatcher=dispatcher,
    )
    channel.bot_open_id = "ou_bot"

    channel._handle_websocket_payload(
        _feishu_payload(
            message_id="om_group",
            chat_id="oc_group",
            chat_type="group",
            sender_open_id="ou_a",
            text="@_user_1 hello",
            mentions=_bot_mention(),
        )
    )

    assert dispatcher.submissions is not None
    assert dispatcher.submissions[0][0] == "feishu:oc_group"


def test_feishu_dispatch_worker_reuses_queued_placeholder_for_final_answer_when_streaming_disabled():
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
    assert dispatcher.submissions[0][1]() == "答案"
    assert FEISHU_THINKING_REPLY_TEXT == "thinking..."
    assert client.replies == [("om_source", "thinking...")]
    assert client.updates == [("om_reply", "答案")]


def test_feishu_group_message_injects_and_appends_group_memory():
    client = _FakeFeishuClient()
    memory_store = _FakeGroupMemoryStore(
        recent_turns=[
            {
                "chat_id": "oc_group",
                "sender_open_id": "ou_other",
                "message_id": "om_old",
                "question": "previous question",
                "answer": "previous answer",
                "created_at": "2026-04-24T00:00:00+00:00",
            }
        ]
    )
    agent_calls: list[tuple[str, str, dict[str, object]]] = []

    def _agent_runner(text: str, thread_id: str, message_context: dict[str, object]) -> str:
        agent_calls.append((text, thread_id, message_context))
        return "final answer"

    channel = FeishuChannel(
        settings=Settings(
            bot_concurrency_enabled=False,
            feishu_streaming_enabled=False,
            feishu_group_memory_recent_turns=3,
        ),
        client=client,
        agent_runner=_agent_runner,
        deduper=MessageDeduper(),
        group_memory_store=memory_store,
    )
    channel.bot_open_id = "ou_bot"

    answer = channel.handle_event(
        json.loads(
            _feishu_payload(
                message_id="om_new",
                chat_id="oc_group",
                chat_type="group",
                sender_open_id="ou_a",
                text="@_user_1 current question",
                mentions=_bot_mention(),
            )
        )["event"]
    )

    assert answer == "final answer"
    assert memory_store.recent_calls == [("oc_group", 3)]
    assert agent_calls[0][0] == "current question"
    assert agent_calls[0][1] == "feishu:oc_group:ou_a"
    assert agent_calls[0][2]["group_thread_id"] == "feishu:oc_group"
    assert agent_calls[0][2]["group_recent_turns"] == memory_store.recent
    assert memory_store.appended == [
        {
            "chat_id": "oc_group",
            "sender_open_id": "ou_a",
            "message_id": "om_new",
            "question": "current question",
            "answer": "final answer",
        }
    ]


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
    assert WEIXIN_THINKING_REPLY_TEXT == "thinking..."
    assert api.sent == [("user_a", "thinking...", "ctx_a")]


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
