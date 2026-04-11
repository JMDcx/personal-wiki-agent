"""Schemas used by the Feishu Wiki RAG example."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging

try:
    from feishu_wiki_rag_agent.observability.events import log_event
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.events import log_event


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log_schema_warning(reason: str, **fields: object) -> None:
    log_event(
        "schema_normalization_warning",
        level=logging.WARNING,
        schema_stage="message_context",
        reason=reason,
        **fields,
    )


@dataclass
class MentionRef:
    """Stable mention metadata preserved across channel normalization."""

    display_name: str
    open_id: str
    is_bot: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize the mention into a JSON-friendly dictionary."""
        return {
            "display_name": self.display_name,
            "open_id": self.open_id,
            "is_bot": self.is_bot,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MentionRef":
        """Build a mention reference from a JSON-friendly dictionary."""
        return cls(
            display_name=str(payload.get("display_name", "")).strip(),
            open_id=str(payload.get("open_id", "")).strip(),
            is_bot=bool(payload.get("is_bot", False)),
        )


@dataclass
class ReplyContext:
    """Reply/reference metadata preserved for group follow-up questions."""

    parent_id: str
    root_id: str
    parent_text_preview: str = ""
    parent_message_type: str = ""
    parent_role: str = ""
    parent_sender_name: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the reply context into a JSON-friendly dictionary."""
        return {
            "is_reply": True,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "parent_text_preview": self.parent_text_preview,
            "parent_message_type": self.parent_message_type,
            "parent_role": self.parent_role,
            "parent_sender_name": self.parent_sender_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ReplyContext | None":
        """Build reply context from a JSON-friendly dictionary when reply metadata is present."""
        if not payload or not payload.get("is_reply"):
            return None
        parent_id = str(payload.get("parent_id", "")).strip()
        root_id = str(payload.get("root_id", "")).strip()
        if not parent_id and not root_id:
            _log_schema_warning("invalid_reply_context", has_parent_id=False, has_root_id=False)
            return None
        return cls(
            parent_id=parent_id,
            root_id=root_id,
            parent_text_preview=str(payload.get("parent_text_preview", "")).strip(),
            parent_message_type=str(payload.get("parent_message_type", "")).strip(),
            parent_role=str(payload.get("parent_role", "")).strip(),
            parent_sender_name=str(payload.get("parent_sender_name", "")).strip(),
        )


@dataclass
class MessageContext:
    """Structured message context carried across agent protocol boundaries."""

    chat_type: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    reply_context: ReplyContext | None = None
    bot_mentioned: bool = False
    mentioned_users: list[str] = field(default_factory=list)
    mention_refs: list[MentionRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize the message context into a JSON-friendly dictionary."""
        return {
            "chat_type": self.chat_type,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "reply_context": self.reply_context.to_dict() if self.reply_context else None,
            "bot_mentioned": self.bot_mentioned,
            "mentioned_users": list(self.mentioned_users),
            "mentions": [mention.to_dict() for mention in self.mention_refs],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "MessageContext":
        """Normalize a message-context mapping into a structured protocol object."""
        payload = payload or {}
        mentions = payload.get("mentions", [])
        mention_refs = [MentionRef.from_dict(mention) for mention in mentions if isinstance(mention, dict)]
        skipped_mentions = len(list(mentions)) - len(mention_refs) if isinstance(mentions, list) else 0
        if skipped_mentions:
            _log_schema_warning("invalid_mentions_skipped", skipped_mentions=skipped_mentions)
        mentioned_users = [
            str(user).strip()
            for user in payload.get("mentioned_users", [])
            if str(user).strip()
        ]
        reply_context_payload = payload.get("reply_context")
        reply_context = (
            ReplyContext.from_dict(reply_context_payload)
            if isinstance(reply_context_payload, dict)
            else None
        )
        return cls(
            chat_type=str(payload.get("chat_type", "")).strip(),
            raw_text=str(payload.get("raw_text", "")).strip(),
            normalized_text=str(payload.get("normalized_text", "")).strip(),
            reply_context=reply_context,
            bot_mentioned=bool(payload.get("bot_mentioned", False)),
            mentioned_users=mentioned_users,
            mention_refs=mention_refs,
        )


@dataclass
class IncomingMessage:
    """Normalized Feishu message passed into the agent layer."""

    message_id: str
    chat_id: str
    chat_type: str
    sender_open_id: str
    raw_text: str
    text: str
    mentions: list[dict] = field(default_factory=list)
    mention_refs: list[MentionRef] = field(default_factory=list)
    reply_context: ReplyContext | None = None
    bot_mentioned: bool = False
    mentioned_users: list[str] = field(default_factory=list)

    @property
    def is_group(self) -> bool:
        """Whether the message came from a group chat."""
        return self.chat_type == "group"

    def to_message_context(self) -> dict[str, object]:
        """Return a transport-agnostic message context for the agent layer."""
        return MessageContext(
            chat_type=self.chat_type,
            raw_text=self.raw_text,
            normalized_text=self.text,
            reply_context=self.reply_context,
            bot_mentioned=self.bot_mentioned,
            mentioned_users=list(self.mentioned_users),
            mention_refs=list(self.mention_refs),
        ).to_dict()


@dataclass
class IndexManifest:
    """Metadata written after a local RAG index build."""

    indexed_at: str
    root_tokens: list[str]
    document_count: int
    chunk_count: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the manifest as a JSON-friendly dictionary."""
        return asdict(self)
