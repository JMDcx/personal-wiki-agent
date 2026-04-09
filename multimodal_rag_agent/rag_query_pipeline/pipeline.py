"""End-to-end RAG query pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

try:
    from feishu_wiki_rag_agent.observability.context import record_request_timing, update_request_state
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception, preview_text
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.context import record_request_timing, update_request_state
    from observability.events import log_event, log_exception, preview_text

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
        started_at = perf_counter()
        log_event(
            "retrieval_started",
            query_preview=preview_text(query),
            requested_top_k=top_k or self.settings.retrieval_top_k,
            filter_keys=sorted((filters or {}).keys()),
        )
        try:
            query_bundle = self.understander.understand(query)
            retrieved = self.retriever.retrieve(query_bundle.rewritten_query, top_k=top_k, filters=filters)
            reranked = self.reranker.rerank(
                query_bundle.rewritten_query,
                retrieved,
                top_k=top_k or self.settings.rerank_top_k,
            )
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
            elapsed_ms = (perf_counter() - started_at) * 1000
            record_request_timing("retrieval_ms", elapsed_ms)
            update_request_state(
                candidate_count=len(retrieved),
                reranked_count=len(reranked),
                merged_count=len(merged),
                source_count=len(sources),
                retrieval_query_preview=preview_text(query_bundle.rewritten_query),
            )
            log_event(
                "retrieval_completed",
                query_preview=preview_text(query_bundle.rewritten_query),
                candidate_count=len(retrieved),
                reranked_count=len(reranked),
                merged_count=len(merged),
                source_count=len(sources),
                duration_ms=round(elapsed_ms, 1),
            )
            return PreparedContext(
                query_bundle=query_bundle,
                merged_chunks=merged,
                context=context,
                sources=sources,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_exception(
                "retrieval_failed",
                exc,
                query_preview=preview_text(query),
                duration_ms=round(elapsed_ms, 1),
            )
            raise

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
