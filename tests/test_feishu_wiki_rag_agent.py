from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_chroma")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from feishu_wiki_rag_agent.agent import search_knowledge_tool_text
from feishu_wiki_rag_agent.config import Settings
from feishu_wiki_rag_agent.feishu_channel import FeishuChannel, MessageDeduper
from feishu_wiki_rag_agent.indexer import rebuild_index
from feishu_wiki_rag_agent.retrieval import format_retrieved_context, load_vector_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.count("a")), float(index)] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count("a")), 0.0]


class FakeFeishuClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str]] = []

    def fetch_bot_open_id(self) -> str:
        return "ou_bot"

    def reply_text(self, message_id: str, text: str) -> None:
        self.replies.append((message_id, text))


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        example_dir=Path(__file__).resolve().parents[1],
        rag_data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        feishu_wiki_root_tokens=["root_a"],
    )


def test_message_deduper_filters_duplicates() -> None:
    deduper = MessageDeduper(ttl_seconds=60)
    assert deduper.should_process("msg-1", now=100.0) is True
    assert deduper.should_process("msg-1", now=101.0) is False
    assert deduper.should_process("msg-2", now=101.0) is True


def test_channel_ignores_group_without_mention(tmp_path: Path) -> None:
    client = FakeFeishuClient()
    channel = FeishuChannel(
        settings=make_settings(tmp_path),
        client=client,  # type: ignore[arg-type]
        agent_runner=lambda text, thread_id: "ignored",
    )

    result = channel.handle_event(
        {
            "message": {
                "message_id": "msg-1",
                "message_type": "text",
                "chat_type": "group",
                "chat_id": "chat-1",
                "content": '{"text":"hello"}',
                "mentions": [],
            },
            "sender": {"sender_id": {"open_id": "ou_user"}},
        }
    )

    assert result is None
    assert client.replies == []


def test_channel_handles_private_text_message(tmp_path: Path) -> None:
    client = FakeFeishuClient()
    calls: list[tuple[str, str]] = []
    channel = FeishuChannel(
        settings=make_settings(tmp_path),
        client=client,  # type: ignore[arg-type]
        agent_runner=lambda text, thread_id: calls.append((text, thread_id)) or "answer",
    )

    result = channel.handle_event(
        {
            "message": {
                "message_id": "msg-1",
                "message_type": "text",
                "chat_type": "p2p",
                "chat_id": "chat-1",
                "content": '{"text":"如何发布版本？"}',
            },
            "sender": {"sender_id": {"open_id": "ou_user"}},
        }
    )

    assert result == "answer"
    assert calls == [("如何发布版本？", "feishu:chat-1")]
    assert client.replies == [("msg-1", "answer")]


def test_channel_filters_duplicate_message_id(tmp_path: Path) -> None:
    client = FakeFeishuClient()
    channel = FeishuChannel(
        settings=make_settings(tmp_path),
        client=client,  # type: ignore[arg-type]
        agent_runner=lambda text, thread_id: "answer",
    )

    event = {
        "message": {
            "message_id": "dup-1",
            "message_type": "text",
            "chat_type": "p2p",
            "chat_id": "chat-1",
            "content": '{"text":"hello"}',
        },
        "sender": {"sender_id": {"open_id": "ou_user"}},
    }

    assert channel.handle_event(event) == "answer"
    assert channel.handle_event(event) is None
    assert client.replies == [("dup-1", "answer")]


def test_format_retrieved_context_contains_sources() -> None:
    formatted = format_retrieved_context(
        [
            Document(
                page_content="Deployment steps",
                metadata={"title": "Release Guide", "source_url": "https://docs.example/release"},
            )
        ]
    )
    assert "Release Guide" in formatted
    assert "https://docs.example/release" in formatted


def test_search_tool_text_returns_not_found_when_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "feishu_wiki_rag_agent.agent.search_knowledge",
        lambda query, settings: [],
    )
    text = search_knowledge_tool_text("missing", make_settings(tmp_path))
    assert text == "当前索引中未找到相关内容。"


def test_rebuild_index_creates_persistent_chroma(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    documents = [
        Document(
            page_content="The release checklist requires staging verification.",
            metadata={
                "doc_token": "doc-1",
                "node_token": "node-1",
                "title": "Release Checklist",
                "source_url": "https://example.test/release",
            },
        )
    ]

    manifest = rebuild_index(documents, settings, embeddings=FakeEmbeddings())
    assert manifest["document_count"] == 1
    assert manifest["chunk_count"] >= 1

    vector_store = load_vector_store(settings, embeddings=FakeEmbeddings())
    results = vector_store.similarity_search("release checklist", k=1)
    assert results
    assert results[0].metadata["title"] == "Release Checklist"

    shutil.rmtree(settings.chroma_dir, ignore_errors=True)
