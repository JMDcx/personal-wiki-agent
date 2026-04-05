"""Schemas used by the Feishu Wiki RAG example."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class IncomingMessage:
    """Normalized Feishu message passed into the agent layer."""

    message_id: str
    chat_id: str
    chat_type: str
    sender_open_id: str
    text: str
    mentions: list[dict] = field(default_factory=list)

    @property
    def is_group(self) -> bool:
        """Whether the message came from a group chat."""
        return self.chat_type == "group"


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
