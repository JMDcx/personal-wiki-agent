"""Image captioning service."""

from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings


class CaptionService:
    """OpenAI-compatible image captioning."""

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()

    def caption(self, image_path: str) -> str:
        image_bytes = Path(image_path).read_bytes()
        client = OpenAI(api_key=self.settings.vlm_api_key, base_url=self.settings.vlm_base_url or None, timeout=60)
        data_uri = f"data:image/{Path(image_path).suffix.lstrip('.') or 'png'};base64,{base64.b64encode(image_bytes).decode()}"
        response = client.chat.completions.create(
            model=self.settings.vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "请用中文简洁描述这张图片的主要内容。"},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
