"""OCR backends for images."""

from __future__ import annotations

from openai import OpenAI

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.multimodal_image_pipeline.image_payload import build_normalized_image_data_uri


class OCRService:
    """OCR service with PaddleOCR first and VLM fallback."""

    def __init__(self, settings: MultimodalRAGSettings | None = None) -> None:
        self.settings = settings or get_multimodal_settings()

    def extract_text(self, image_path: str) -> str:
        try:
            return self._extract_with_paddle(image_path)
        except Exception:
            return self._extract_with_vlm(image_path)

    def _extract_with_paddle(self, image_path: str) -> str:
        from paddleocr import PaddleOCR
        from PIL import Image
        import numpy as np

        ocr = PaddleOCR(use_gpu=False, use_doc_orientation_classify=True, lang="ch", show_log=False)
        with Image.open(image_path) as image:
            image_array = np.array(image.convert("RGB"))
        result = ocr.ocr(image_array, cls=False)
        if not result or not result[0]:
            return ""
        return " ".join(line[1][0].strip() for line in result[0] if line and len(line) > 1 and line[1][0].strip())

    def _extract_with_vlm(self, image_path: str) -> str:
        client = OpenAI(api_key=self.settings.vlm_api_key, base_url=self.settings.vlm_base_url or None, timeout=60)
        data_uri = build_normalized_image_data_uri(image_path)
        response = client.chat.completions.create(
            model=self.settings.vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {
                            "type": "text",
                            "text": (
                                "提取文档图片中的正文文本，忽略页眉页脚，"
                                "按阅读顺序输出纯 markdown。没有正文则返回 No text content."
                            ),
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        return response.choices[0].message.content or ""
