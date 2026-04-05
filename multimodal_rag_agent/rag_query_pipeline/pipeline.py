"""End-to-end RAG query pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.models import QueryResponse
from multimodal_rag_agent.rag_query_pipeline.generator import AnswerGenerator
from multimodal_rag_agent.rag_query_pipeline.merge import ResultMerger
from multimodal_rag_agent.rag_query_pipeline.prompt_builder import PromptContextBuilder
from multimodal_rag_agent.rag_query_pipeline.query_understand import QueryUnderstander
from multimodal_rag_agent.rag_query_pipeline.rerank import Reranker
from multimodal_rag_agent.rag_query_pipeline.retrieval import Retriever


@dataclass
class PreparedContext:
    """Intermediate retrieval result before final generation."""

    query_bundle: object
    merged_chunks: list[object]
    context: str
    sources: list[dict[str, object]]


class RAGQueryPipeline:
    """Query understand -> retrieval -> rerank -> merge -> build context -> generate."""

    def __init__(
        self,
        settings: MultimodalRAGSettings | None = None,
        *,
        understander: QueryUnderstander | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        merger: ResultMerger | None = None,
        prompt_builder: PromptContextBuilder | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.settings = settings or get_multimodal_settings()
        self.understander = understander or QueryUnderstander(self.settings)
        self.retriever = retriever or Retriever(self.settings)
        self.reranker = reranker or Reranker()
        self.merger = merger or ResultMerger()
        self.prompt_builder = prompt_builder or PromptContextBuilder()
        self.generator = generator or AnswerGenerator(self.settings)

    def prepare_context(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
        with_sources: bool = True,
    ) -> PreparedContext:
        """Run retrieval, rerank, and context assembly without final answer generation."""
        query_bundle = self.understander.understand(query)
        retrieved = self.retriever.retrieve(query_bundle.rewritten_query, top_k=top_k, filters=filters)
        reranked = self.reranker.rerank(query_bundle.rewritten_query, retrieved, top_k=top_k or self.settings.rerank_top_k)
        merged = self.merger.merge(reranked)
        context = self.prompt_builder.build(query_bundle, merged)
        sources: list[dict[str, object]] = []
        if with_sources:
            for chunk in merged:
                sources.append(
                    {
                        "title": chunk.metadata.get("title", "Untitled"),
                        "source_uri": chunk.metadata.get("source_uri", ""),
                        "chunk_type": chunk.chunk_type,
                    }
                )
        return PreparedContext(
            query_bundle=query_bundle,
            merged_chunks=merged,
            context=context,
            sources=sources,
        )

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
        with_sources: bool = True,
    ) -> QueryResponse:
        prepared = self.prepare_context(query, top_k=top_k, filters=filters, with_sources=with_sources)
        query_bundle = prepared.query_bundle
        merged = prepared.merged_chunks
        context = prepared.context
        answer = self.generator.generate(query, context)
        return QueryResponse(answer=answer, sources=prepared.sources, retrieved_chunks=merged, query_bundle=query_bundle)
