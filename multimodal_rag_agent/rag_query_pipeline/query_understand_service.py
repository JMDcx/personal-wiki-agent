"""Intent gate for controller-agent routing.

The controller agent now owns query rewriting and image understanding. This
service only classifies the current user turn into the small SetFit intent set
used before entering the Deep Agent runtime.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings

try:
    from feishu_wiki_rag_agent.observability.events import log_event, preview_text
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.events import log_event, preview_text


QueryIntent = Literal[
    "greeting",
    "knowledge_deposit",
    "kb_search",
    "follow_up",
    "chitchat",
]

INTENT_LABELS: dict[int, QueryIntent] = {
    0: "greeting",
    1: "knowledge_deposit",
    2: "kb_search",
    3: "follow_up",
    4: "chitchat",
}
VALID_INTENTS: set[str] = set(INTENT_LABELS.values())
DEFAULT_INTENT: QueryIntent = "kb_search"

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class _LightweightSetFitIntentModel:
    """Minimal SetFit inference path for exported sentence-transformer models."""

    def __init__(self, model_dir: str | Path) -> None:
        from tokenizers import Tokenizer
        from transformers import AutoModel
        import joblib
        import torch

        self.torch = torch
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.body = AutoModel.from_pretrained(str(model_dir))
        self.body.eval()
        self.head = joblib.load(Path(model_dir) / "model_head.pkl")

    def predict(self, queries: list[str]) -> list[Any]:
        if not queries:
            return []
        encoded = self.tokenizer.encode_batch(queries)
        max_len = min(128, max(len(item.ids) for item in encoded))
        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        token_types: list[list[int]] = []
        for item in encoded:
            ids = item.ids[:max_len]
            mask = [1] * len(ids)
            types = (item.type_ids or [0] * len(ids))[:max_len]
            pad = max_len - len(ids)
            input_ids.append(ids + [0] * pad)
            attention.append(mask + [0] * pad)
            token_types.append(types + [0] * pad)

        with self.torch.no_grad():
            output = self.body(
                input_ids=self.torch.tensor(input_ids),
                attention_mask=self.torch.tensor(attention),
                token_type_ids=self.torch.tensor(token_types),
            )
            token_embeddings = output.last_hidden_state
            mask = self.torch.tensor(attention).unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = (token_embeddings * mask).sum(1) / self.torch.clamp(mask.sum(1), min=1e-9)
        return list(self.head.predict(embeddings.numpy()))


@dataclass(slots=True)
class HistoryTurn:
    """Minimal conversation turn available to the controller agent."""

    user_question: str
    assistant_answer: str


@dataclass(slots=True)
class ModelConfig:
    """Compatibility shim for older query-understand call sites."""

    model: str
    api_key: str = ""
    base_url: str = ""


@dataclass(slots=True)
class QueryUnderstandResult:
    """Structured output consumed by the controller-agent runtime."""

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


class QueryUnderstandService:
    """Classify intent with a SetFit model and leave rewriting to the agent."""

    def __init__(
        self,
        settings: MultimodalRAGSettings | None = None,
        *,
        model_loader: Callable[[], Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.settings = settings or get_multimodal_settings()
        self.model_loader = model_loader or self._load_configured_model
        self.env = env if env is not None else os.environ

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
        del history, images, language, chat_model_supports_vision, vlm_model
        clean_query = str(query or "").strip()
        try:
            model = self.model_loader()
            label = self._predict_label(model, clean_query)
            intent = self._normalize_intent(label)
            raw_output = json.dumps(
                {
                    "source": "setfit_intent_model",
                    "label": str(label),
                    "intent": intent,
                },
                ensure_ascii=False,
            )
            log_event(
                "intent_model_classified",
                intent=intent,
                raw_label=str(label),
                question_preview=preview_text(clean_query),
            )
        except Exception as exc:  # pragma: no cover - exercised through tests with fake loader
            intent = DEFAULT_INTENT
            raw_output = json.dumps(
                {
                    "source": "setfit_intent_model",
                    "status": "fallback",
                    "intent": intent,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
            self._log_schema_warning(
                "intent_model_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                fallback_intent=intent,
                question_preview=preview_text(clean_query),
            )
        return QueryUnderstandResult(
            rewrite_query=clean_query,
            intent=intent,
            image_description="",
            raw_output=raw_output,
        )

    def _load_configured_model(self) -> Any:
        model_ref, source = self._resolve_model_ref()
        cache_key = (source, model_ref)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        token = str(self.env.get("HF_TOKEN", "")).strip() or None
        log_event(
            "intent_model_loading",
            model_source=source,
            model_ref=model_ref,
            has_hf_token=bool(token),
        )
        try:
            from setfit import SetFitModel

            model = SetFitModel.from_pretrained(model_ref, token=token)
        except TypeError:
            try:
                model = SetFitModel.from_pretrained(model_ref, use_auth_token=token)
            except TypeError:
                model = SetFitModel.from_pretrained(model_ref)
        except (ImportError, ModuleNotFoundError) as exc:
            log_event(
                "intent_model_setfit_unavailable",
                level=logging.WARNING,
                model_source=source,
                model_ref=model_ref,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            model = self._load_lightweight_model(model_ref, source=source, token=token)
        _MODEL_CACHE[cache_key] = model
        return model

    @staticmethod
    def _load_lightweight_model(model_ref: str, *, source: str, token: str | None) -> _LightweightSetFitIntentModel:
        if source == "local_path":
            return _LightweightSetFitIntentModel(model_ref)

        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:  # pragma: no cover - installed with setfit in normal env
            msg = "huggingface_hub is required to download the intent model fallback."
            raise RuntimeError(msg) from exc

        model_dir = snapshot_download(repo_id=model_ref, token=token)
        return _LightweightSetFitIntentModel(model_dir)

    def _resolve_model_ref(self) -> tuple[str, str]:
        local_path = str(self.env.get("FEISHU_INTENT_MODEL_PATH", "")).strip()
        if local_path:
            path = Path(local_path).expanduser()
            if not path.exists():
                msg = f"FEISHU_INTENT_MODEL_PATH does not exist: {path}"
                raise FileNotFoundError(msg)
            return str(path), "local_path"

        model_id = str(self.env.get("FEISHU_INTENT_MODEL_ID", "")).strip()
        if model_id:
            return model_id, "huggingface"

        msg = "Set FEISHU_INTENT_MODEL_ID or FEISHU_INTENT_MODEL_PATH to enable intent classification."
        raise RuntimeError(msg)

    @classmethod
    def _predict_label(cls, model: Any, query: str) -> Any:
        if hasattr(model, "predict"):
            prediction = model.predict([query])
        else:
            prediction = model([query])
        return cls._first_prediction(prediction)

    @classmethod
    def _first_prediction(cls, prediction: Any) -> Any:
        if hasattr(prediction, "detach"):
            prediction = prediction.detach().cpu().tolist()
        elif hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        if isinstance(prediction, Sequence) and not isinstance(prediction, str):
            if not prediction:
                return ""
            return cls._first_prediction(prediction[0])
        return prediction

    @classmethod
    def _normalize_intent(cls, intent: Any) -> QueryIntent:
        if isinstance(intent, str):
            value = intent.strip()
            if value in VALID_INTENTS:
                return value  # type: ignore[return-value]
            try:
                numeric = int(value)
            except ValueError:
                cls._log_schema_warning("invalid_intent", original_intent=value, fallback_intent=DEFAULT_INTENT)
                return DEFAULT_INTENT
        else:
            try:
                numeric = int(intent)
            except (TypeError, ValueError):
                cls._log_schema_warning("invalid_intent", original_intent=str(intent), fallback_intent=DEFAULT_INTENT)
                return DEFAULT_INTENT

        normalized = INTENT_LABELS.get(numeric)
        if normalized is None:
            cls._log_schema_warning("invalid_intent_label", original_label=numeric, fallback_intent=DEFAULT_INTENT)
            return DEFAULT_INTENT
        return normalized

    @staticmethod
    def _log_schema_warning(reason: str, **fields: object) -> None:
        log_event(
            "schema_normalization_warning",
            level=logging.WARNING,
            schema_stage="intent_classification",
            reason=reason,
            **fields,
        )
