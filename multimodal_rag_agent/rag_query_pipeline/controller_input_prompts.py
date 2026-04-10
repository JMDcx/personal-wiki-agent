"""Prompt helpers for controller-agent runtime metadata."""

from __future__ import annotations

from multimodal_rag_agent.rag_query_pipeline.query_understand_service import HistoryTurn, QueryUnderstandResult


def render_controller_user_input(
    *,
    question: str,
    understand_result: QueryUnderstandResult,
    history: list[HistoryTurn],
    allow_retrieval: bool,
    images: list[str] | None = None,
    message_context: dict[str, object] | None = None,
) -> str:
    lines: list[str] = [
        "[Runtime Metadata - for assistant control, not for direct user display]",
        f"intent: {understand_result.intent}",
        f"allow_retrieval: {'yes' if allow_retrieval else 'no'}",
        f"rewrite_query: {understand_result.rewrite_query or question}",
    ]

    if understand_result.image_description:
        lines.extend(
            [
                "image_description:",
                understand_result.image_description,
            ]
        )
    if images:
        lines.extend(["image_paths:", *images])

    if history:
        lines.append("recent_history:")
        for turn in history[-3:]:
            lines.append(f"- User: {turn.user_question}")
            lines.append(f"- Assistant: {turn.assistant_answer}")
    if message_context:
        lines.append("message_context:")
        chat_type = str(message_context.get("chat_type", "")).strip()
        if chat_type:
            lines.append(f"chat_type: {chat_type}")
        if message_context.get("bot_mentioned"):
            lines.append("bot_mentioned: yes")
        mentioned_users = [
            str(user).strip()
            for user in message_context.get("mentioned_users", [])
            if str(user).strip()
        ]
        if mentioned_users:
            lines.append(f"mentioned_users: {', '.join(mentioned_users)}")
        mention_details: list[str] = []
        for mention in message_context.get("mentions", []):
            if not isinstance(mention, dict) or mention.get("is_bot"):
                continue
            display_name = str(mention.get("display_name", "")).strip()
            open_id = str(mention.get("open_id", "")).strip()
            if not display_name:
                continue
            mention_details.append(f"{display_name}({open_id})" if open_id else display_name)
        if mention_details:
            lines.append(f"mention_details: {', '.join(mention_details)}")
        reply_context = message_context.get("reply_context", {})
        if isinstance(reply_context, dict) and reply_context.get("is_reply"):
            lines.append("reply_context:")
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

    lines.extend(
        [
            "turn_policy:",
            "- If allow_retrieval is yes, use rewrite_query when delegating retrieval.",
            "- If allow_retrieval is no, answer directly and do not call knowledge_retriever.",
            "- Treat mentioned_users as part of the user's intent and conversational reference.",
            "- Use mention_details when you need stable identity hints for group-chat references.",
            "- When reply_context is present, treat parent_text_preview as the conversational anchor for deictic follow-ups.",
            "- Use parent_role to distinguish whether the follow-up targets a user's question or the assistant's previous answer.",
            "",
            "Current user question:",
            question,
        ]
    )
    return "\n".join(lines)
