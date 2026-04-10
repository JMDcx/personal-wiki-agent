"""Prompt helpers for controller-agent runtime metadata."""

from __future__ import annotations

try:
    from feishu_wiki_rag_agent.protocols.renderers import render_controller_metadata_lines
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from protocols.renderers import render_controller_metadata_lines

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
    lines = render_controller_metadata_lines(
        intent=understand_result.intent,
        allow_retrieval=allow_retrieval,
        rewrite_query=understand_result.rewrite_query or question,
        image_description=understand_result.image_description,
        images=images,
        history=history,
        message_context=message_context,
    )

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
