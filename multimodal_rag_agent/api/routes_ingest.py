"""API routes for ingest."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

try:
    from feishu_wiki_rag_agent.observability.context import bind_log_context, make_request_id
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.context import bind_log_context, make_request_id

from multimodal_rag_agent.ingest_pipeline.pipeline import IngestPipeline

router = APIRouter(prefix="/api/ingest", tags=["ingest"])
pipeline = IngestPipeline()


class URLIngestRequest(BaseModel):
    url: str
    title: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


@router.post("/file")
async def ingest_file(file: UploadFile = File(...)) -> dict[str, object]:
    with bind_log_context(
        request_id=make_request_id("api-ingest-file"),
        channel="api",
        api_route="/api/ingest/file",
    ):
        payload = await file.read()
        result = pipeline.ingest_file(file.filename or "upload.bin", payload)
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "image_count": result.image_count,
        }


@router.post("/url")
async def ingest_url(request: URLIngestRequest) -> dict[str, object]:
    with bind_log_context(
        request_id=make_request_id("api-ingest-url"),
        channel="api",
        api_route="/api/ingest/url",
    ):
        result = pipeline.ingest_url(request.url, title=request.title, metadata=request.metadata)
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "image_count": result.image_count,
        }
