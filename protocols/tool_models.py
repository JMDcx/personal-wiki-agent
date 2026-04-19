"""Structured tool-side protocol models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


_URL_TRAILING_PUNCTUATION = ")]}>,，。；;！？!?\"'`#"
_DEPOSIT_SOURCE_URL_RE = re.compile(r"##\s*来源链接\s+(https?://\S+)", re.IGNORECASE)
_DEPOSIT_PROVIDED_CONTENT_RE = re.compile(
    r"##\s*原文提取内容（已提供，无需再抓取）\s*(.+)$",
    re.DOTALL,
)
_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_URL_ALLOWED_PREFIX_RE = re.compile(r"^(https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)")


def _normalize_url_candidate(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("[") and "](" in raw and raw.endswith(")"):
        raw = raw.split("](", 1)[1][:-1].strip()
    raw = raw.lstrip("([<")
    matched = _URL_ALLOWED_PREFIX_RE.match(raw)
    if matched:
        raw = matched.group(1)
    while raw and raw[-1] in _URL_TRAILING_PUNCTUATION:
        raw = raw[:-1]
    return raw.strip()


def _extract_urls_from_text(text: str) -> list[str]:
    seen: set[str] = set()
    normalized_urls: list[str] = []
    for matched in _URL_PATTERN.findall(text or ""):
        normalized = _normalize_url_candidate(matched)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_urls.append(normalized)
    return normalized_urls


@dataclass(slots=True)
class RetrievalRequest:
    """Normalized retrieval request passed into the retrieval tool boundary."""

    query: str
    with_sources: bool = True

    @classmethod
    def from_query(cls, query: str, *, with_sources: bool = True) -> "RetrievalRequest":
        return cls(query=_normalize_text(query), with_sources=with_sources)


@dataclass(slots=True)
class DepositRequestContext:
    """Normalized deposit request inputs before calling the deposit pipeline."""

    text: str
    urls: list[str] = field(default_factory=list)
    source_url: str = ""
    provided_content: str = ""
    image_paths: list[str] = field(default_factory=list)
    invalid_image_paths_json: bool = False
    invalid_urls_json: bool = False
    dropped_image_path_count: int = 0
    dropped_url_count: int = 0

    @classmethod
    def from_inputs(
        cls,
        *,
        text: str,
        image_paths_json: str = "[]",
        urls_json: str = "[]",
        source_url: str = "",
        provided_content: str = "",
    ) -> "DepositRequestContext":
        try:
            parsed_paths = json.loads(image_paths_json) if image_paths_json.strip() else []
            invalid_image_paths_json = False
        except json.JSONDecodeError:
            parsed_paths = []
            invalid_image_paths_json = True
        try:
            parsed_urls = json.loads(urls_json) if urls_json.strip() else []
            invalid_urls_json = False
        except json.JSONDecodeError:
            parsed_urls = []
            invalid_urls_json = True
        normalized_candidates = [str(path).strip() for path in parsed_paths]
        image_paths = [path for path in normalized_candidates if path]
        dropped_image_path_count = len(normalized_candidates) - len(image_paths)
        normalized_url_candidates = [_normalize_url_candidate(url) for url in parsed_urls]
        urls = [url for url in normalized_url_candidates if url]
        dropped_url_count = len(normalized_url_candidates) - len(urls)
        normalized_text = str(text or "").strip()
        normalized_source_url = _normalize_url_candidate(source_url)
        if not normalized_source_url:
            matched_source = _DEPOSIT_SOURCE_URL_RE.search(normalized_text)
            if matched_source:
                normalized_source_url = _normalize_url_candidate(matched_source.group(1))
        normalized_provided_content = str(provided_content or "").strip()
        if not normalized_provided_content:
            matched_content = _DEPOSIT_PROVIDED_CONTENT_RE.search(normalized_text)
            if matched_content:
                normalized_provided_content = matched_content.group(1).strip()
        if not normalized_provided_content and normalized_source_url and normalized_text:
            # Treat direct tool calls that pass article markdown in `text`
            # plus the origin link in `source_url` as caller-provided content.
            normalized_provided_content = normalized_text
        if normalized_source_url and normalized_source_url not in urls:
            urls.insert(0, normalized_source_url)
        if not urls and not normalized_provided_content:
            urls = _extract_urls_from_text(normalized_text)
        return cls(
            text=normalized_text,
            urls=urls,
            source_url=normalized_source_url,
            provided_content=normalized_provided_content,
            image_paths=image_paths,
            invalid_image_paths_json=invalid_image_paths_json,
            invalid_urls_json=invalid_urls_json,
            dropped_image_path_count=dropped_image_path_count,
            dropped_url_count=dropped_url_count,
        )


# ---------------------------------------------------------------------------
# Weak-relevance thresholds (hardcoded per handoff: no new env / config)
# ---------------------------------------------------------------------------
_NO_MATCH_TOP_SCORE_THRESHOLD = 0.35
_NO_MATCH_KEYWORD_OVERLAP_MIN = 0.15


def _extract_ngrams(text: str, n: int = 2) -> set[str]:
    """Extract character *n*-grams, handling both CJK and whitespace-delimited text."""
    # If text contains spaces, split into whitespace tokens first and generate
    # n-grams per token (good for English-like text).
    parts = text.split()
    if len(parts) > 1:
        ngrams: set[str] = set()
        for part in parts:
            part_lower = part.lower()
            if len(part_lower) >= n:
                for i in range(len(part_lower) - n + 1):
                    ngrams.add(part_lower[i : i + n])
        return ngrams
    # No spaces — treat as CJK; generate character n-grams directly.
    lower = text.lower()
    if len(lower) < n:
        return {lower} if lower else set()
    return {lower[i : i + n] for i in range(len(lower) - n + 1)}


def _compute_keyword_overlap(query: str, text: str) -> float:
    """Fraction of query n-grams found in *text* (case-insensitive, CJK-aware)."""
    query_ngrams = _extract_ngrams(query)
    if not query_ngrams:
        return 0.0
    text_ngrams = _extract_ngrams(text)
    hits = len(query_ngrams & text_ngrams)
    return hits / len(query_ngrams)


def _classify_match_status(
    *,
    merged_chunks: list[object],
    top_score: float,
    query: str,
    top_chunk_content: str,
) -> str:
    """Return ``'no_match'`` when retrieval results are empty or weakly relevant."""
    if not merged_chunks:
        return "no_match"
    # Top chunk score too low → weak relevance
    if top_score < _NO_MATCH_TOP_SCORE_THRESHOLD:
        return "no_match"
    # Keyword overlap too low → likely unrelated
    if _compute_keyword_overlap(query, top_chunk_content) < _NO_MATCH_KEYWORD_OVERLAP_MIN:
        return "no_match"
    return "matched"


@dataclass(slots=True)
class RetrievalResult:
    """Normalized retrieval payload before rendering to tool text."""

    query: str
    result_status: str
    context: str
    sources: list[dict[str, object]] = field(default_factory=list)
    chunk_count: int = 0
    match_status: str = "matched"

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @classmethod
    def empty(cls, *, query: str) -> "RetrievalResult":
        return cls(
            query=query,
            result_status="empty",
            context="",
            sources=[],
            chunk_count=0,
            match_status="no_match",
        )

    @classmethod
    def from_prepared_context(cls, query: str, prepared: Any) -> "RetrievalResult":
        merged_chunks = list(getattr(prepared, "merged_chunks", []) or [])
        if not merged_chunks:
            return cls.empty(query=query)
        top_chunk = merged_chunks[0]
        top_score = float(getattr(top_chunk, "score", 0.0) or 0.0)
        # Align haystack with rerank.py: content + ocr_text + caption_text
        top_meta = getattr(top_chunk, "metadata", None) or {}
        haystack_parts = [
            str(getattr(top_chunk, "content", "") or ""),
            str(top_meta.get("ocr_text", "") or ""),
            str(top_meta.get("caption_text", "") or ""),
        ]
        top_haystack = " ".join(part for part in haystack_parts if part)
        match_status = _classify_match_status(
            merged_chunks=merged_chunks,
            top_score=top_score,
            query=query,
            top_chunk_content=top_haystack,
        )
        return cls(
            query=query,
            result_status="completed",
            context=str(getattr(prepared, "context", "") or "").strip(),
            sources=list(getattr(prepared, "sources", []) or []),
            chunk_count=len(merged_chunks),
            match_status=match_status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "result_status": self.result_status,
            "context": self.context,
            "sources": self.sources,
            "chunk_count": self.chunk_count,
            "source_count": self.source_count,
            "match_status": self.match_status,
        }


@dataclass(slots=True)
class DepositResult:
    """Normalized deposit payload before rendering to tool text."""

    result_status: str
    message: str
    source_type: str
    local_document_id: str = ""
    feishu_doc_url: str = ""
    wiki_node_token: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pipeline_result(cls, result: Any) -> "DepositResult":
        draft = getattr(result, "draft", None)
        return cls(
            result_status=str(getattr(result, "status", "") or "").strip(),
            message=str(getattr(result, "message", "") or "").strip(),
            source_type=str(getattr(draft, "source_type", "") or "").strip(),
            local_document_id=str(getattr(result, "local_document_id", "") or "").strip(),
            feishu_doc_url=str(getattr(result, "feishu_doc_url", "") or "").strip(),
            wiki_node_token=str(getattr(result, "wiki_node_token", "") or "").strip(),
            metadata=dict(getattr(result, "metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_status": self.result_status,
            "message": self.message,
            "source_type": self.source_type,
            "local_document_id": self.local_document_id,
            "feishu_doc_url": self.feishu_doc_url,
            "wiki_node_token": self.wiki_node_token,
            "metadata": self.metadata,
        }
