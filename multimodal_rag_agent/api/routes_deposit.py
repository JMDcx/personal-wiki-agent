"""API routes for knowledge deposit."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from multimodal_rag_agent.deposit_pipeline.models import DepositRequest
from multimodal_rag_agent.deposit_pipeline.pipeline import DepositPipeline

router = APIRouter(prefix="/api/deposit", tags=["deposit"])
pipeline = DepositPipeline()


class DepositPayload(BaseModel):
    text: str = ""
    urls: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    target_space_id: str = ""
    target_parent_node_token: str = ""
    auto_write: bool | None = None


@router.post("")
async def deposit_knowledge(request: DepositPayload) -> dict[str, object]:
    result = pipeline.run(
        DepositRequest(
            text=request.text,
            urls=request.urls,
            image_paths=request.image_paths,
            target_space_id=request.target_space_id,
            target_parent_node_token=request.target_parent_node_token,
            auto_write=request.auto_write,
        )
    )
    return result.to_dict()
