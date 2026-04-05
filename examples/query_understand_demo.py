"""Minimal runnable demo for QueryUnderstandService."""

from __future__ import annotations

from pprint import pprint

from multimodal_rag_agent.rag_query_pipeline.query_understand_service import (
    HistoryTurn,
    ModelConfig,
    QueryUnderstandService,
)


def main() -> None:
    service = QueryUnderstandService()

    # no_image_result = service.run(
    #     query="你好",
    #     history=[
    #         HistoryTurn(
    #             user_question="什么是RAG架构",
    #             assistant_answer="RAG 是把检索和生成结合起来的问答架构。",
    #         )
    #     ],
    #     language="中文",
    #     chat_model_supports_vision=False,
    # )
    # print("=== No Image ===")
    # pprint(no_image_result.to_dict())

    with_image_result = service.run(
        query="这张图是什么意思",
        history=[],
        images=["examples/test.jpeg"],
        language="中文",
        chat_model_supports_vision=True,
        # vlm_model=ModelConfig(model="gpt-4.1-mini"),
    )
    print("=== With Image ===")
    pprint(with_image_result.to_dict())


if __name__ == "__main__":
    main()
