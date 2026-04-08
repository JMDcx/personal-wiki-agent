"""Feishu channel components for the Feishu Wiki RAG example."""

try:
    from feishu_wiki_rag_agent.channel.feishu.feishu_channel import FeishuChannel
    from feishu_wiki_rag_agent.channel.feishu.feishu_client import FeishuClient
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from channel.feishu.feishu_channel import FeishuChannel
    from channel.feishu.feishu_client import FeishuClient

__all__ = ["FeishuChannel", "FeishuClient"]
