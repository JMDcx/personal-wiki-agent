"""API routes for query."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from multimodal_rag_agent.rag_query_pipeline.pipeline import RAGQueryPipeline

router = APIRouter(prefix="/api", tags=["query"])
pipeline = RAGQueryPipeline()


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    filters: dict[str, object] = Field(default_factory=dict)
    with_sources: bool = True


@router.post("/query")
async def query_knowledge(request: QueryRequest) -> dict[str, object]:
    result = pipeline.run(
        request.query,
        top_k=request.top_k,
        filters=request.filters or None,
        with_sources=request.with_sources,
    )
    return result.to_dict()
