"""Schemas used by the Feishu Wiki RAG example."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
        return {
            "chat_type": self.chat_type,
            "raw_text": self.raw_text,
            "normalized_text": self.text,
            "reply_context": self.reply_context.to_dict() if self.reply_context else None,
            "bot_mentioned": self.bot_mentioned,
            "mentioned_users": list(self.mentioned_users),
            "mentions": [mention.to_dict() for mention in self.mention_refs],
        }


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
