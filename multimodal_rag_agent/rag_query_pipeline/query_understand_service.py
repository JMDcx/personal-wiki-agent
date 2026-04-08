"""Standalone query-understand service.

This module reproduces WeKnora's QUERY_UNDERSTAND stage as a single model call
that performs query rewrite, intent classification, and image understanding.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.rag_query_pipeline.query_understand_prompts import (
    QUERY_UNDERSTAND_SYSTEM_PROMPT,
    QUERY_UNDERSTAND_USER_PROMPT,
)

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ModuleNotFoundError:  # pragma: no cover - local fallback
    HumanMessage = None  # type: ignore[assignment]
    SystemMessage = None  # type: ignore[assignment]

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # pragma: no cover - local fallback
    ChatOpenAI = None  # type: ignore[assignment]


QueryIntent = Literal[
    "greeting",
    "summarize",
    "web_search",
    "knowledge_deposit",
    "kb_search",
    "clarification",
    "follow_up",
    "image_only",
    "chitchat",
]

VALID_INTENTS: set[str] = {
    "greeting",
    "summarize",
    "web_search",
    "knowledge_deposit",
    "kb_search",
    "clarification",
    "follow_up",
    "image_only",
    "chitchat",
}
DEFAULT_INTENT: QueryIntent = "kb_search"


@dataclass(slots=True)
class HistoryTurn:
    """Minimal conversation turn used for query understanding."""

    user_question: str
    assistant_answer: str


@dataclass(slots=True)
class ModelConfig:
    """OpenAI-compatible model configuration."""

    model: str
    api_key: str = ""
    base_url: str = ""


@dataclass(slots=True)
class QueryUnderstandResult:
    """Structured output for the query-understand stage."""

    rewrite_query: str
    intent: QueryIntent = DEFAULT_INTENT
    image_description: str = ""
    raw_output: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "rewrite_query": self.rewrite_query,
            "intent": self.intent,
            "image_description": self.image_description,
        }


@dataclass(slots=True)
class SelectedModel:
    """Resolved runtime model choice."""

    config: ModelConfig
    use_images: bool
    max_tokens: int


class QueryUnderstandService:
    """Run rewrite, intent classification, and image analysis in one call."""

    def __init__(
        self,
        settings: MultimodalRAGSettings | None = None,
        *,
        chat_model: ModelConfig | None = None,
        model_factory: Callable[[ModelConfig, float, int], Any] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or get_multimodal_settings()
        self.chat_model = chat_model or ModelConfig(
            model=self.settings.chat_model,
            api_key=self.settings.chat_api_key,
            base_url=self.settings.chat_base_url,
        )
        self.model_factory = model_factory or self._default_model_factory
        self.now_provider = now_provider or datetime.now

    def select_model(
        self,
        *,
        images: list[str],
        chat_model_supports_vision: bool,
        vlm_model: ModelConfig | None = None,
    ) -> SelectedModel:
        has_images = bool(images)
        if has_images:
            if chat_model_supports_vision:
                return SelectedModel(config=self.chat_model, use_images=True, max_tokens=500)
            if vlm_model is not None:
                return SelectedModel(config=vlm_model, use_images=True, max_tokens=500)
        return SelectedModel(config=self.chat_model, use_images=False, max_tokens=150 if not has_images else 500)

    def build_prompts(
        self,
        *,
        query: str,
        history: list[HistoryTurn],
        language: str,
    ) -> tuple[str, str]:
        conversation = self.format_history(history)
        current_time = self.now_provider().isoformat(timespec="seconds")
        system_prompt = QUERY_UNDERSTAND_SYSTEM_PROMPT.format(language=language)
        user_prompt = QUERY_UNDERSTAND_USER_PROMPT.format(
            current_time=current_time,
            conversation=conversation,
            query=query,
        )
        return system_prompt, user_prompt

    def format_history(self, history: list[HistoryTurn]) -> str:
        if not history:
            return ""
        parts: list[str] = []
        for turn in history:
            parts.append("------BEGIN------")
            parts.append(f"User question: {turn.user_question}")
            parts.append(f"Assistant answer: {turn.assistant_answer}")
            parts.append("------END------")
        return "\n".join(parts)

    def merge_image_desc_and_ocr(self, desc: str, ocr: str) -> str:
        clean_desc = desc.strip()
        clean_ocr = ocr.strip()
        if not clean_desc and not clean_ocr:
            return ""
        if not clean_desc:
            return clean_ocr
        if not clean_ocr:
            return clean_desc
        if clean_ocr in clean_desc:
            return clean_desc
        return f"{clean_desc}\n\n[OCR]\n{clean_ocr}"

    def parse_output(self, raw_text: str) -> QueryUnderstandResult:
        content = (raw_text or "").strip()
        if not content:
            return QueryUnderstandResult(rewrite_query="", intent=DEFAULT_INTENT, image_description="", raw_output="")

        parsed = self._try_parse_json(content)
        if parsed is None:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end > start:
                parsed = self._try_parse_json(content[start : end + 1])

        if parsed is None:
            return QueryUnderstandResult(
                rewrite_query=content,
                intent=DEFAULT_INTENT,
                image_description="",
                raw_output=content,
            )

        rewrite_query = str(
            parsed.get("rewrite_query")
            or parsed.get("rewritten_query")
            or parsed.get("query")
            or parsed.get("question")
            or ""
        ).strip()
        intent = self._normalize_intent(parsed.get("intent"))
        image_description = self.merge_image_desc_and_ocr(
            str(
                parsed.get("image_description")
                or parsed.get("image_desc")
                or parsed.get("image_text")
                or parsed.get("image_ocr_text")
                or parsed.get("description")
                or ""
            ),
            str(
                parsed.get("ocr_text")
                or parsed.get("ocr")
                or parsed.get("full_ocr")
                or parsed.get("image_ocr")
                or parsed.get("ocr_content")
                or ""
            ),
        )
        return QueryUnderstandResult(
            rewrite_query=rewrite_query,
            intent=intent,
            image_description=image_description,
            raw_output=content,
        )

    def run(
        self,
        *,
        query: str,
        history: list[HistoryTurn] | None = None,
        images: list[str] | None = None,
        language: str = "中文",
        chat_model_supports_vision: bool = False,
        vlm_model: ModelConfig | None = None,
    ) -> QueryUnderstandResult:
        history = history or []
        images = images or []
        selected_model = self.select_model(
            images=images,
            chat_model_supports_vision=chat_model_supports_vision,
            vlm_model=vlm_model,
        )
        system_prompt, user_prompt = self.build_prompts(query=query, history=history, language=language)
        model = self.model_factory(selected_model.config, 0.3, selected_model.max_tokens)
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            use_images=selected_model.use_images,
        )
        response = model.invoke(messages)
        raw_output = self._extract_text(response)
        parsed = self.parse_output(raw_output)
        rewrite_query = parsed.rewrite_query or query.strip()
        image_description = parsed.image_description if images else ""
        return QueryUnderstandResult(
            rewrite_query=rewrite_query,
            intent=self._normalize_intent(parsed.intent),
            image_description=image_description,
            raw_output=raw_output,
        )

    def _build_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[str],
        use_images: bool,
    ) -> list[Any]:
        if SystemMessage is None or HumanMessage is None:  # pragma: no cover - local fallback
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._build_message_content_for_images(user_prompt, images) if use_images else user_prompt,
                },
            ]
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=self._build_message_content_for_images(user_prompt, images) if use_images else user_prompt
            ),
        ]

    def _build_message_content_for_images(self, user_prompt: str, images: list[str]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in images:
            path = Path(image_path)
            image_bytes = path.read_bytes()
            mime_subtype = path.suffix.lstrip(".").lower() or "png"
            data_uri = f"data:image/{mime_subtype};base64,{base64.b64encode(image_bytes).decode()}"
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        return content

    def _default_model_factory(self, config: ModelConfig, temperature: float, max_tokens: int) -> Any:
        if ChatOpenAI is None:  # pragma: no cover - local fallback
            msg = "langchain_openai is required to run QueryUnderstandService."
            raise RuntimeError(msg)
        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key or None,
            base_url=config.base_url or None,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        text_parts.append(item["text"])
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        text_parts.append(item["content"])
            return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()
        return str(content).strip()

    @staticmethod
    def _try_parse_json(raw_text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _normalize_intent(intent: Any) -> QueryIntent:
        value = str(intent or "").strip()
        if value in VALID_INTENTS:
            return value  # type: ignore[return-value]
        return DEFAULT_INTENT
