"""Channel integrations for the Feishu Wiki RAG example."""

try:
    from feishu_wiki_rag_agent.channel.feishu import FeishuChannel, FeishuClient
    from feishu_wiki_rag_agent.channel.weixin import WeixinApi, WeixinChannel, WeixinMessage
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from channel.feishu import FeishuChannel, FeishuClient
    from channel.weixin import WeixinApi, WeixinChannel, WeixinMessage

__all__ = ["FeishuChannel", "FeishuClient", "WeixinApi", "WeixinChannel", "WeixinMessage"]
