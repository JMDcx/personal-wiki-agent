"""Shared renderers for structured control-plane context."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from feishu_wiki_rag_agent.protocols.tool_models import DepositResult, RetrievalResult
    except ModuleNotFoundError:  # pragma: no cover - source tree fallback
        from protocols.tool_models import DepositResult, RetrievalResult
    try:
        from feishu_wiki_rag_agent.multimodal_rag_agent.rag_query_pipeline.query_understand_service import HistoryTurn
    except ModuleNotFoundError:  # pragma: no cover - source tree fallback
        from multimodal_rag_agent.rag_query_pipeline.query_understand_service import HistoryTurn
    try:
        from feishu_wiki_rag_agent.schemas import MessageContext
    except ModuleNotFoundError:  # pragma: no cover - source tree fallback
        from schemas import MessageContext

try:
    from feishu_wiki_rag_agent.schemas import MessageContext
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from schemas import MessageContext


def render_mention_details(mentions: list[dict[str, object]] | None) -> str:
    """Render non-bot mentions as a compact, human-readable summary."""
    if not mentions:
        return ""

    rendered: list[str] = []
    for mention in mentions:
        if not isinstance(mention, dict) or mention.get("is_bot"):
            continue
        display_name = str(mention.get("display_name", "")).strip()
        open_id = str(mention.get("open_id", "")).strip()
        if not display_name:
            continue
        rendered.append(f"{display_name}({open_id})" if open_id else display_name)
    return ", ".join(rendered)


def render_reply_context_line(reply_context: dict[str, object] | None) -> str:
    """Render the replied-to target as one compact summary line."""
    if not isinstance(reply_context, dict):
        return ""

    role = str(reply_context.get("parent_role", "")).strip()
    sender = str(reply_context.get("parent_sender_name", "")).strip()
    preview = str(reply_context.get("parent_text_preview", "")).strip()

    target = "/".join(part for part in (role, sender) if part)
    if target and preview:
        return f"Reply target: {target} -> {preview}"
    if preview:
        return f"Reply target: {preview}"
    return f"Reply target: {target}" if target else ""


def render_message_context_lines(message_context: MessageContext | dict[str, object] | None) -> list[str]:
    """Render message context as stable controller prompt lines."""
    if message_context is None:
        return []

    if hasattr(message_context, "to_dict") and hasattr(message_context, "mention_refs"):
        context = message_context
    else:
        context = MessageContext.from_dict(message_context)
    lines: list[str] = ["message_context:"]
    if context.chat_type:
        lines.append(f"chat_type: {context.chat_type}")
    if context.bot_mentioned:
        lines.append("bot_mentioned: yes")
    if context.mentioned_users:
        lines.append(f"mentioned_users: {', '.join(context.mentioned_users)}")
    mention_details = render_mention_details([mention.to_dict() for mention in context.mention_refs])
    if mention_details:
        lines.append(f"mention_details: {mention_details}")
    if context.reply_context is not None:
        reply_context = context.reply_context.to_dict()
        lines.append("reply_context:")
        reply_target_line = render_reply_context_line(reply_context)
        if reply_target_line:
            lines.append(reply_target_line)
        for key in (
            "parent_id",
            "root_id",
            "parent_message_type",
            "parent_role",
            "parent_sender_name",
            "parent_text_preview",
        ):
            value = str(reply_context.get(key, "")).strip()
            if value:
                lines.append(f"{key}: {value}")
    return lines


def render_controller_metadata_lines(
    *,
    intent: str,
    allow_retrieval: bool,
    rewrite_query: str,
    image_description: str = "",
    images: list[str] | None = None,
    history: list["HistoryTurn"] | None = None,
    message_context: MessageContext | dict[str, object] | None = None,
) -> list[str]:
    """Render controller runtime metadata from normalized protocol objects."""
    lines: list[str] = [
        "[Runtime Metadata - for assistant control, not for direct user display]",
        f"intent: {intent}",
        f"allow_retrieval: {'yes' if allow_retrieval else 'no'}",
        f"rewrite_query: {rewrite_query}",
    ]
    if image_description:
        lines.extend(["image_description:", image_description])
    if images:
        lines.extend(["image_paths:", *images])
    if history:
        lines.append("recent_history:")
        for turn in history[-3:]:
            lines.append(f"- User: {turn.user_question}")
            lines.append(f"- Assistant: {turn.assistant_answer}")
    lines.extend(render_message_context_lines(message_context))
    return lines


def render_retrieval_result_text(result: RetrievalResult) -> str:
    """Render retrieval results into the existing tool-facing text contract."""
    if result.result_status == "empty" or not result.context.strip():
        return "当前索引中未找到相关内容。"

    lines: list[str] = [result.context.strip()]
    if result.sources:
        source_line = "来源：" + "；".join(
            [
                str(source.get("title") or source.get("source_uri") or "Untitled")
                for source in result.sources
            ]
        )
        lines.append(source_line)
    return "\n\n".join(lines)


def render_deposit_result_text(result: DepositResult) -> str:
    """Render deposit results into the existing tool-facing text contract."""
    lines = [result.message, f"来源类型：{result.source_type}"]
    if result.feishu_doc_url:
        lines.append(f"飞书文档：{result.feishu_doc_url}")
    if result.wiki_node_token:
        lines.append(f"Wiki 节点：{result.wiki_node_token}")
    return "\n".join(lines)
