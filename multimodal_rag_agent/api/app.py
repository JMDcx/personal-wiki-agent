"""FastAPI application for multimodal RAG."""

from __future__ import annotations

from fastapi import FastAPI

try:
    from feishu_wiki_rag_agent.config import get_settings
    from feishu_wiki_rag_agent.observability.logging import configure_logging
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import get_settings
    from observability.logging import configure_logging

from multimodal_rag_agent.api.routes_assets import mount_asset_routes
from multimodal_rag_agent.api.routes_deposit import router as deposit_router
from multimodal_rag_agent.api.routes_ingest import router as ingest_router
from multimodal_rag_agent.api.routes_query import router as query_router


def create_app() -> FastAPI:
    configure_logging(get_settings())
    app = FastAPI(title="Multimodal RAG Agent")
    app.include_router(ingest_router)
    app.include_router(deposit_router)
    app.include_router(query_router)
    mount_asset_routes(app)
    return app


app = create_app()
