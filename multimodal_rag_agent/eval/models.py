"""Data models for local Agent/RAG evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvalCase:
    """One normalized evaluation case loaded from JSONL."""

    id: str
    user_query: str
    thread_id: str = ""
    language: str = "中文"
    history: list[dict[str, str]] = field(default_factory=list)
    message_context: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    expected_intent: str = ""
    expected_allow_retrieval: bool | None = None
    expected_rewrite_query: str = ""
    expected_answer_must_include: list[str] = field(default_factory=list)
    expected_answer_must_not_include: list[str] = field(default_factory=list)
    expected_source_titles: list[str] = field(default_factory=list)
    expected_source_uris: list[str] = field(default_factory=list)
    expected_match_status: str = ""
    reference_answer: str = ""
    scene: str = ""
    function: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalActual:
    """Observed output and telemetry for one evaluation case."""

    answer: str = ""
    status: str = "ok"
    error: str = ""
    intent: str = ""
    allow_retrieval: bool | None = None
    rewrite_query: str = ""
    retrieval_call_count: int = 0
    tool_call_count: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    match_status: str = ""
    total_ms: float | None = None
    intent_ms: float | None = None
    retrieval_ms: float | None = None
    llm_ms: float | None = None
    reply_ms: float | None = None
    request_id: str = ""
    thread_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalResult:
    """Scored result for one evaluation case."""

    case_id: str
    user_query: str
    passed: bool
    status: str
    error: str
    expected_intent: str
    actual_intent: str
    intent_match: bool | None
    expected_allow_retrieval: bool | None
    actual_allow_retrieval: bool | None
    allow_retrieval_match: bool | None
    expected_rewrite_query: str
    actual_rewrite_query: str
    rewrite_match: bool | None
    answer_match: bool
    missing_required: list[str]
    forbidden_hits: list[str]
    expected_source_titles: list[str]
    expected_source_uris: list[str]
    actual_sources: list[dict[str, Any]]
    source_hit: bool | None
    expected_match_status: str
    actual_match_status: str
    match_status_match: bool | None
    citation_present: bool | None
    citation_source_hit: bool | None
    retrieval_call_count: int
    tool_call_count: int
    total_ms: float | None
    intent_ms: float | None
    retrieval_ms: float | None
    llm_ms: float | None
    reply_ms: float | None
    actual_answer: str
    reference_answer: str
    failure_reasons: list[str]
    primary_failure_reason: str
    thread_id: str = ""
    request_id: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunArtifacts:
    """Paths written by one eval run."""

    output_dir: Path
    summary_path: Path
    results_path: Path
    report_path: Path
    summary: dict[str, Any]
