"""Configuration for the multimodal RAG package."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass
class MultimodalRAGSettings:
    """Runtime settings for multimodal RAG."""

    project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "MULTIMODAL_RAG_DATA_DIR",
                str(Path(__file__).resolve().parents[1] / "data" / "multimodal_rag"),
            )
        )
    )
    asset_dir: Path | None = field(
        default_factory=lambda: Path(os.getenv("MULTIMODAL_RAG_ASSET_DIR"))
        if os.getenv("MULTIMODAL_RAG_ASSET_DIR")
        else None
    )
    image_dir_name: str = field(default_factory=lambda: os.getenv("MULTIMODAL_RAG_IMAGE_DIR_NAME", "images"))
    image_url_prefix: str = field(default_factory=lambda: os.getenv("MULTIMODAL_RAG_IMAGE_URL_PREFIX", "/assets/images"))
    qdrant_url: str = field(default_factory=lambda: os.getenv("MULTIMODAL_RAG_QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("MULTIMODAL_RAG_QDRANT_API_KEY", ""))
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_QDRANT_COLLECTION",
            os.getenv("FEISHU_RAG_COLLECTION", "multimodal_rag"),
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_EMBEDDING_MODEL",
            os.getenv("FEISHU_RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_EMBEDDING_API_KEY",
            os.getenv("FEISHU_RAG_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        )
    )
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_EMBEDDING_BASE_URL",
            os.getenv(
                "FEISHU_RAG_EMBEDDING_BASE_URL",
                os.getenv("OPENAI_BASE_URL", os.getenv("FEISHU_RAG_OPENAI_BASE_URL", "")),
            ),
        )
    )
    chat_model: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_CHAT_MODEL",
            os.getenv("FEISHU_RAG_MODEL", "gpt-4.1-mini"),
        )
    )
    chat_api_key: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_CHAT_API_KEY",
            os.getenv("FEISHU_RAG_CHAT_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        )
    )
    chat_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_CHAT_BASE_URL",
            os.getenv(
                "FEISHU_RAG_CHAT_BASE_URL",
                os.getenv("OPENAI_BASE_URL", os.getenv("FEISHU_RAG_OPENAI_BASE_URL", "")),
            ),
        )
    )
    rerank_model: str = field(default_factory=lambda: os.getenv("MULTIMODAL_RAG_RERANK_MODEL", ""))
    rerank_api_key: str = field(default_factory=lambda: os.getenv("MULTIMODAL_RAG_RERANK_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    rerank_base_url: str = field(default_factory=lambda: os.getenv("MULTIMODAL_RAG_RERANK_BASE_URL", os.getenv("OPENAI_BASE_URL", "")))
    vlm_model: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_VLM_MODEL",
            os.getenv("FEISHU_RAG_MODEL", "gpt-4.1-mini"),
        )
    )
    vlm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_VLM_API_KEY",
            os.getenv("FEISHU_RAG_CHAT_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        )
    )
    vlm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MULTIMODAL_RAG_VLM_BASE_URL",
            os.getenv(
                "FEISHU_RAG_CHAT_BASE_URL",
                os.getenv("OPENAI_BASE_URL", os.getenv("FEISHU_RAG_OPENAI_BASE_URL", "")),
            ),
        )
    )
    docreader_project_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("WEKNORA_DOCREADER_PROJECT_DIR", "/Users/jmdcx/Documents/GitHub/WeKnora")
        )
    )
    chunk_size: int = field(default_factory=lambda: _env_int("MULTIMODAL_RAG_CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("MULTIMODAL_RAG_CHUNK_OVERLAP", 128))
    retrieval_top_k: int = field(default_factory=lambda: _env_int("MULTIMODAL_RAG_TOP_K", 6))
    rerank_top_k: int = field(default_factory=lambda: _env_int("MULTIMODAL_RAG_RERANK_TOP_K", 4))
    qdrant_vector_size: int = field(default_factory=lambda: _env_int("MULTIMODAL_RAG_VECTOR_SIZE", 1536))
    query_understand_timeout_seconds: int = field(
        default_factory=lambda: _env_int("MULTIMODAL_RAG_QUERY_UNDERSTAND_TIMEOUT_SECONDS", 10)
    )
    query_understand_max_retries: int = field(
        default_factory=lambda: _env_int("MULTIMODAL_RAG_QUERY_UNDERSTAND_MAX_RETRIES", 0)
    )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.asset_root.mkdir(parents=True, exist_ok=True)

    @property
    def asset_root(self) -> Path:
        if self.asset_dir is not None:
            return self.asset_dir
        return self.data_dir / "assets" / self.image_dir_name


def get_multimodal_settings() -> MultimodalRAGSettings:
    settings = MultimodalRAGSettings()
    settings.ensure_directories()
    return settings
