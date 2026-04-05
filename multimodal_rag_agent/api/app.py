"""FastAPI application for multimodal RAG."""

from __future__ import annotations

from fastapi import FastAPI

from multimodal_rag_agent.api.routes_assets import mount_asset_routes
from multimodal_rag_agent.api.routes_ingest import router as ingest_router
from multimodal_rag_agent.api.routes_query import router as query_router


def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal RAG Agent")
    app.include_router(ingest_router)
    app.include_router(query_router)
    mount_asset_routes(app)
    return app


app = create_app()
