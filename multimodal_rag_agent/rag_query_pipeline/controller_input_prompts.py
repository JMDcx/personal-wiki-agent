"""Prompt helpers for controller-agent runtime metadata."""

from __future__ import annotations

from multimodal_rag_agent.rag_query_pipeline.query_understand_service import HistoryTurn, QueryUnderstandResult


def render_controller_user_input(
    *,
    question: str,
    understand_result: QueryUnderstandResult,
    history: list[HistoryTurn],
    allow_retrieval: bool,
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

    if history:
        lines.append("recent_history:")
        for turn in history[-3:]:
            lines.append(f"- User: {turn.user_question}")
            lines.append(f"- Assistant: {turn.assistant_answer}")

    lines.extend(
        [
            "turn_policy:",
            "- If allow_retrieval is yes, use rewrite_query when delegating retrieval.",
            "- If allow_retrieval is no, answer directly and do not call knowledge_retriever.",
            "",
            "Current user question:",
            question,
        ]
    )
    return "\n".join(lines)
