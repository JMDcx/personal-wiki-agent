"""Shared streaming event protocol for agent-to-channel updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


StreamEventType = Literal["started", "status", "text_delta", "final", "error"]


@dataclass(slots=True)
class StreamEvent:
    """One user-safe streaming update emitted by the agent layer."""

    event_type: StreamEventType
    text: str = ""
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def started(
        cls,
        text: str = "开始处理",
        *,
        stage: str = "started",
        metadata: dict[str, Any] | None = None,
    ) -> "StreamEvent":
        return cls(event_type="started", text=text, stage=stage, metadata=dict(metadata or {}))

    @classmethod
    def status(cls, text: str, *, stage: str = "status", metadata: dict[str, Any] | None = None) -> "StreamEvent":
        return cls(event_type="status", text=text, stage=stage, metadata=dict(metadata or {}))

    @classmethod
    def text_delta(
        cls,
        text: str,
        *,
        stage: str = "generation",
        metadata: dict[str, Any] | None = None,
    ) -> "StreamEvent":
        return cls(event_type="text_delta", text=text, stage=stage, metadata=dict(metadata or {}))

    @classmethod
    def final(cls, text: str, *, stage: str = "final", metadata: dict[str, Any] | None = None) -> "StreamEvent":
        return cls(event_type="final", text=text, stage=stage, metadata=dict(metadata or {}))

    @classmethod
    def error(cls, text: str, *, stage: str = "error", metadata: dict[str, Any] | None = None) -> "StreamEvent":
        return cls(event_type="error", text=text, stage=stage, metadata=dict(metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "text": self.text,
            "stage": self.stage,
            "metadata": dict(self.metadata),
        }
