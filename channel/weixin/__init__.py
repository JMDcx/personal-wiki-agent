"""Weixin channel components for the Feishu Wiki RAG example."""

try:
    from feishu_wiki_rag_agent.channel.weixin.weixin_api import WeixinApi
    from feishu_wiki_rag_agent.channel.weixin.weixin_channel import WeixinChannel
    from feishu_wiki_rag_agent.channel.weixin.weixin_message import WeixinMessage
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from channel.weixin.weixin_api import WeixinApi
    from channel.weixin.weixin_channel import WeixinChannel
    from channel.weixin.weixin_message import WeixinMessage

__all__ = ["WeixinApi", "WeixinChannel", "WeixinMessage"]
