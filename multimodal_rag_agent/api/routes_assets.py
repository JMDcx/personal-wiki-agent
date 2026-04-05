"""Helpers for static image routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from multimodal_rag_agent.config import get_multimodal_settings


def mount_asset_routes(app: FastAPI) -> None:
    settings = get_multimodal_settings()
    app.mount("/assets/images", StaticFiles(directory=str(settings.asset_root)), name="multimodal-images")
