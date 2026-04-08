"""Configuration for the Feishu Wiki RAG example."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")


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
    weixin_base_url: str = field(default_factory=lambda: os.getenv("WEIXIN_BASE_URL", "https://ilinkai.weixin.qq.com"))
    weixin_cdn_base_url: str = field(
        default_factory=lambda: os.getenv("WEIXIN_CDN_BASE_URL", "https://novac2c.cdn.weixin.qq.com/c2c")
    )
    weixin_token: str = field(default_factory=lambda: os.getenv("WEIXIN_TOKEN", ""))
    weixin_request_timeout: int = field(default_factory=lambda: int(os.getenv("WEIXIN_REQUEST_TIMEOUT", "15")))
    weixin_long_poll_timeout: int = field(default_factory=lambda: int(os.getenv("WEIXIN_LONG_POLL_TIMEOUT", "35")))
    rag_data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "FEISHU_RAG_DATA_DIR",
                str(Path(__file__).resolve().parent / "data"),
            )
        )
    )
    weixin_credentials_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "WEIXIN_CREDENTIALS_PATH",
                str(Path(__file__).resolve().parent / "data" / "weixin" / "credentials.json"),
            )
        )
    )
    weixin_tmp_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "WEIXIN_TMP_DIR",
                str(Path(__file__).resolve().parent / "data" / "weixin" / "tmp"),
            )
        )
    )
    checkpoint_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "FEISHU_AGENT_CHECKPOINT_DB_PATH",
                str(Path(__file__).resolve().parent / "data" / "deepagents" / "checkpoints.sqlite"),
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
    manifest_name: str = field(default_factory=lambda: os.getenv("FEISHU_RAG_MANIFEST", "index_manifest.json"))
    feishu_deposit_space_id: str = field(default_factory=lambda: os.getenv("FEISHU_DEPOSIT_SPACE_ID", ""))
    feishu_deposit_parent_node_token: str = field(
        default_factory=lambda: os.getenv("FEISHU_DEPOSIT_PARENT_NODE_TOKEN", "")
    )
    xhs_mcp_url: str = field(default_factory=lambda: os.getenv("XHS_MCP_URL", "http://127.0.0.1:18060/mcp"))
    deposit_enable_auto_write: bool = field(
        default_factory=lambda: os.getenv("DEPOSIT_ENABLE_AUTO_WRITE", "true").strip().lower() not in {"0", "false", "no"}
    )

    def ensure_directories(self) -> None:
        """Create local directories used by the example."""
        self.rag_data_dir.mkdir(parents=True, exist_ok=True)
        with suppress(Exception):
            self.weixin_credentials_path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(Exception):
            self.weixin_tmp_dir.mkdir(parents=True, exist_ok=True)
        with suppress(Exception):
            self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

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
