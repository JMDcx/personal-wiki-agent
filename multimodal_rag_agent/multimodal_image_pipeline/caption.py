"""Image captioning service."""

from __future__ import annotations

from openai import OpenAI

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.multimodal_image_pipeline.image_payload import build_normalized_image_data_uri


class CaptionService:
    """OpenAI-compatible image captioning."""

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()

    def caption(self, image_path: str) -> str:
        client = OpenAI(api_key=self.settings.vlm_api_key, base_url=self.settings.vlm_base_url or None, timeout=60)
        data_uri = build_normalized_image_data_uri(image_path)
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
