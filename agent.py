"""Deep Agent entrypoint for Feishu Wiki RAG."""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from feishu_wiki_rag_agent.config import Settings, get_settings
from feishu_wiki_rag_agent.retrieval import format_retrieved_context, search_knowledge


def search_knowledge_tool_text(query: str, settings: Settings | None = None) -> str:
    """Return formatted Feishu knowledge snippets for a user query."""
    resolved = settings or get_settings()
    documents = search_knowledge(query, resolved)
    return format_retrieved_context(documents)


def build_agent(settings: Settings | None = None) -> Any:
    """Build the Deep Agent used for Feishu Wiki RAG answers."""
    resolved = settings or get_settings()
    backend = FilesystemBackend(root_dir=resolved.example_dir, virtual_mode=True)
    model_kwargs: dict[str, Any] = {"model": resolved.rag_model}
    if resolved.chat_api_key:
        model_kwargs["api_key"] = resolved.chat_api_key
    if resolved.chat_base_url:
        model_kwargs["base_url"] = resolved.chat_base_url
    model = ChatOpenAI(**model_kwargs)

    @tool
    def search_feishu_knowledge(query: str) -> str:
        """Search the indexed Feishu Wiki knowledge base for relevant documentation.

        Use this before answering factual product, process, or documentation questions.

        Args:
            query: Search query describing the knowledge the user needs.
        """
        return search_knowledge_tool_text(query, resolved)

    return create_deep_agent(
        model=model,
        tools=[search_feishu_knowledge],
        system_prompt=(
            "You are a Feishu Wiki knowledge assistant. "
            "Always search the local Feishu knowledge index before answering factual questions. "
            "Base answers on retrieved content only. "
            "If the index has no relevant content, say '当前索引中未找到相关内容。' "
            "Do not invent missing details. "
            "When you do answer from retrieved content, end with a concise 来源 section that includes document titles or links."
        ),
        backend=backend,
        skills=["/skills/"],
        memory=["/AGENTS.md"],
        name="feishu-wiki-rag-agent",
    )


def extract_final_text(result: dict[str, Any]) -> str:
    """Extract the final assistant text from an agent invocation result."""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            if isinstance(message.content, str):
                return message.content
            if isinstance(message.content, list):
                text_parts = [block.get("text", "") for block in message.content if isinstance(block, dict)]
                joined = "\n".join(part for part in text_parts if part)
                if joined:
                    return joined
    return "当前索引中未找到相关内容。"


def invoke_agent(
    question: str,
    *,
    settings: Settings | None = None,
    thread_id: str = "default",
) -> str:
    """Invoke the Feishu Wiki Deep Agent and return the final text answer."""
    agent = build_agent(settings)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return extract_final_text(result)
