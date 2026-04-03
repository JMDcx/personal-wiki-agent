"""Standalone Feishu websocket channel for the Feishu Wiki RAG example."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv

from feishu_wiki_rag_agent.agent import invoke_agent
from feishu_wiki_rag_agent.config import Settings, get_settings
from feishu_wiki_rag_agent.feishu_client import FeishuClient
from feishu_wiki_rag_agent.schemas import IncomingMessage

LARK_SDK_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None
lark = None

logger = logging.getLogger(__name__)
logging.getLogger("Lark").setLevel(logging.WARNING)


def _ensure_lark_imported():
    """Import `lark_oapi` lazily."""
    global lark
    if lark is None:
        import lark_oapi as imported_lark

        lark = imported_lark
    return lark


@dataclass
class MessageDeduper:
    """Simple TTL-based deduper for Feishu message ids."""

    ttl_seconds: int = 60 * 60
    _seen: dict[str, float] = field(default_factory=dict)

    def should_process(self, message_id: str, now: float | None = None) -> bool:
        """Return whether the message should be processed."""
        current = now if now is not None else time.time()
        self._seen = {key: ts for key, ts in self._seen.items() if current - ts < self.ttl_seconds}
        if message_id in self._seen:
            return False
        self._seen[message_id] = current
        return True


class FeishuChannel:
    """Standalone websocket channel that feeds incoming messages into Deep Agents."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: FeishuClient | None = None,
        agent_runner: Callable[[str, str], str] | None = None,
        deduper: MessageDeduper | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or FeishuClient(self.settings)
        self.agent_runner = agent_runner or (lambda text, thread_id: invoke_agent(text, settings=self.settings, thread_id=thread_id))
        self.deduper = deduper or MessageDeduper()
        self.bot_open_id: str | None = None

    def run(self) -> None:
        """Start the websocket client and listen for Feishu messages."""
        if self.settings.feishu_event_mode != "websocket":
            msg = "This example only supports FEISHU_EVENT_MODE=websocket."
            raise RuntimeError(msg)
        if not LARK_SDK_AVAILABLE:
            msg = "lark-oapi is required for websocket mode."
            raise RuntimeError(msg)

        sdk = _ensure_lark_imported()
        self.bot_open_id = self.client.fetch_bot_open_id()

        def handle_message(data) -> None:
            try:
                event_dict = json.loads(sdk.JSON.marshal(data))
                event = event_dict.get("event", {})
                self.handle_event(event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to handle Feishu websocket event: %s", exc)

        event_handler = (
            sdk.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(handle_message)
            .build()
        )
        websocket_client = sdk.ws.Client(
            self.settings.feishu_app_id,
            self.settings.feishu_app_secret,
            event_handler=event_handler,
            log_level=sdk.LogLevel.WARNING,
        )
        websocket_client.start()

    def handle_event(self, event: dict) -> str | None:
        """Parse and process a single Feishu event."""
        incoming = self._parse_incoming_message(event)
        if incoming is None:
            return None

        thread_id = f"feishu:{incoming.chat_id}"
        answer = self.agent_runner(incoming.text, thread_id)
        self.client.reply_text(incoming.message_id, answer)
        return answer

    def _parse_incoming_message(self, event: dict) -> IncomingMessage | None:
        """Convert a raw Feishu event into the local message schema."""
        message = event.get("message")
        sender = event.get("sender", {})
        if not message or message.get("message_type") != "text":
            return None

        message_id = str(message.get("message_id", ""))
        if not message_id or not self.deduper.should_process(message_id):
            return None

        content = json.loads(message.get("content", "{}"))
        text = str(content.get("text", "")).strip()
        mentions = message.get("mentions", [])
        chat_type = str(message.get("chat_type", ""))

        if chat_type == "group":
            if not mentions:
                return None
            if not self._mentions_bot(mentions):
                return None
            text = re.sub(r"@_user_\d+\s*", "", text).strip()

        if not text:
            return None

        return IncomingMessage(
            message_id=message_id,
            chat_id=str(message.get("chat_id", "")),
            chat_type=chat_type,
            sender_open_id=str(sender.get("sender_id", {}).get("open_id", "")),
            text=text,
            mentions=mentions,
        )

    def _mentions_bot(self, mentions: list[dict]) -> bool:
        """Return whether the incoming group message mentioned the current bot."""
        if not self.bot_open_id:
            return True
        return any(mention.get("id", {}).get("open_id") == self.bot_open_id for mention in mentions)


def main() -> None:
    """Load env vars and run the Feishu websocket channel."""
    logging.basicConfig(level=logging.INFO)
    load_dotenv(Settings().env_path)
    FeishuChannel().run()


if __name__ == "__main__":
    main()
