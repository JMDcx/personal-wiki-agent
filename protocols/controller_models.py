"""Structured controller-side protocol models."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

try:
    from feishu_wiki_rag_agent.multimodal_rag_agent.rag_query_pipeline.query_understand_service import (
        DEFAULT_INTENT,
        QueryUnderstandResult,
        VALID_INTENTS,
    )
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from multimodal_rag_agent.rag_query_pipeline.query_understand_service import (
        DEFAULT_INTENT,
        QueryUnderstandResult,
        VALID_INTENTS,
    )
try:
    from feishu_wiki_rag_agent.observability.events import log_event, preview_text
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.events import log_event, preview_text


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _log_schema_warning(reason: str, **fields: object) -> None:
    log_event(
        "schema_normalization_warning",
        level=logging.WARNING,
        schema_stage="controller_decision",
        reason=reason,
        **fields,
    )


@dataclass(slots=True)
class ControllerDecision:
    """Normalized controller decision carried across agent stages."""

    intent: str
    allow_retrieval: bool
    rewrite_query: str

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ControllerDecision":
        raw_intent = str(payload.get("intent", "")).strip()
        intent = raw_intent if raw_intent in VALID_INTENTS else DEFAULT_INTENT
        if raw_intent and intent == DEFAULT_INTENT and raw_intent != DEFAULT_INTENT:
            _log_schema_warning(
                "invalid_intent",
                original_intent=raw_intent,
                rewrite_query_preview=preview_text(str(payload.get("rewrite_query", ""))),
            )
        return cls(
            intent=intent,
            allow_retrieval=_normalize_bool(payload.get("allow_retrieval")),
            rewrite_query=str(payload.get("rewrite_query", "")).strip(),
        )

    @classmethod
    def from_query_understand(
        cls,
        result: QueryUnderstandResult,
        *,
        fallback_question: str,
        allow_retrieval: bool,
    ) -> "ControllerDecision":
        return cls.from_dict(
            {
                "intent": result.intent,
                "allow_retrieval": allow_retrieval,
                "rewrite_query": result.rewrite_query or fallback_question,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "allow_retrieval": self.allow_retrieval,
            "rewrite_query": self.rewrite_query,
        }
