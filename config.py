"""Configuration for the Feishu Wiki RAG example."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _split_csv_env(name: str) -> list[str]:
    """Split a comma-delimited environment variable into trimmed values."""
    raw = os.getenv(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass
class Settings:
    """Runtime settings for the Feishu Wiki RAG example."""

    example_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    feishu_api_base: str = field(default_factory=lambda: os.getenv("FEISHU_API_BASE", "https://open.feishu.cn"))
    feishu_app_id: str = field(default_factory=lambda: os.getenv("FEISHU_APP_ID", ""))
    feishu_app_secret: str = field(default_factory=lambda: os.getenv("FEISHU_APP_SECRET", ""))
    feishu_event_mode: str = field(default_factory=lambda: os.getenv("FEISHU_EVENT_MODE", "websocket"))
    feishu_wiki_root_tokens: list[str] = field(default_factory=lambda: _split_csv_env("FEISHU_WIKI_ROOT_TOKENS"))
    feishu_request_timeout: int = field(default_factory=lambda: int(os.getenv("FEISHU_REQUEST_TIMEOUT", "20")))
    rag_data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "FEISHU_RAG_DATA_DIR",
                str(Path(__file__).resolve().parent / "data"),
            )
        )
    )
    chroma_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "FEISHU_RAG_CHROMA_DIR",
                str(Path(__file__).resolve().parent / "data" / "chroma"),
            )
        )
    )
    chat_api_key: str = field(default_factory=lambda: os.getenv(
        "FEISHU_RAG_CHAT_API_KEY",
        os.getenv("OPENAI_API_KEY", ""),
    ))
    chat_base_url: str = field(default_factory=lambda: os.getenv(
        "FEISHU_RAG_CHAT_BASE_URL",
        os.getenv("OPENAI_BASE_URL", os.getenv("FEISHU_RAG_OPENAI_BASE_URL", "")),
    ))
    embedding_api_key: str = field(default_factory=lambda: os.getenv(
        "FEISHU_RAG_EMBEDDING_API_KEY",
        os.getenv("OPENAI_API_KEY", ""),
    ))
    embedding_base_url: str = field(default_factory=lambda: os.getenv(
        "FEISHU_RAG_EMBEDDING_BASE_URL",
        os.getenv("OPENAI_BASE_URL", os.getenv("FEISHU_RAG_OPENAI_BASE_URL", "")),
    ))
    rag_model: str = field(default_factory=lambda: os.getenv("FEISHU_RAG_MODEL", "gpt-4.1-mini"))
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("FEISHU_RAG_TOP_K", "4")))
    embedding_model: str = field(default_factory=lambda: os.getenv("FEISHU_RAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    chroma_collection_name: str = field(default_factory=lambda: os.getenv("FEISHU_RAG_COLLECTION", "feishu_wiki_docs"))
    manifest_name: str = field(default_factory=lambda: os.getenv("FEISHU_RAG_MANIFEST", "index_manifest.json"))

    def ensure_directories(self) -> None:
        """Create local directories used by the example."""
        self.rag_data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        """Return the path to the local index manifest."""
        return self.rag_data_dir / self.manifest_name

    @property
    def env_path(self) -> Path:
        """Return the example-local `.env` path."""
        return self.example_dir / ".env"


def get_settings() -> Settings:
    """Build settings and ensure on-disk directories exist."""
    settings = Settings()
    settings.ensure_directories()
    return settings
