import base64
import uuid
from typing import Any, Dict, List

import google.generativeai as genai
from llama_parse import LlamaParse

from backend.app.config import get_settings

settings = get_settings()

CAPTION_PROMPT = """
You are analyzing a chart/image from a financial report.
Please provide:
1. A concise caption (1-2 sentences) describing what this chart shows
2. Key data points visible in the chart (numbers, trends, percentages)
3. Chart type (bar_chart, line_chart, pie_chart, table_image, diagram, other)

Respond in this JSON format:
{
    "caption": "Bar chart showing quarterly revenue...",
    "key_data": "Q1 2023: 14,500B VND, Q4 2025: 17,045B VND. Trend: consistent growth",
    "chart_type": "bar_chart"
}

If this is not a chart/graph (e.g. logo, signature, photo), respond:
{"caption": null, "key_data": null, "chart_type": "non_chart"}
"""


class ImageProcessor:
    """
    Extract images từ PDF qua LlamaParse → dùng Gemini 2.0 Flash caption.
    Chỉ lưu chunks cho ảnh có ý nghĩa (charts/graphs), bỏ qua logo/ảnh trang trí.
    """

    def __init__(self):
        self.parser = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="markdown",
            language="vi",
            verbose=False,
        )
        genai.configure(api_key=settings.google_api_key)
        self.vision_model = genai.GenerativeModel("gemini-2.0-flash-exp")

    async def process(
        self,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract images và generate captions.

        Returns:
            List of image chunks với caption làm content:
            [
                {
                    "id": "uuid",
                    "content": "Bar chart showing quarterly revenue from Q1 2023 to Q4 2025...",
                    "metadata": {
                        "chunk_type": "image_caption",
                        "page_num": 15,
                        "chart_type": "bar_chart",
                        "image_path": "uploads/doc_id_page15_img0.png"
                    }
                }
            ]
        """
        # LlamaParse trả về images kèm theo documents
        json_result = await self.parser.aget_json(file_path)
        chunks = []

        for page_data in json_result:
            page_num = page_data.get("page", 1)
            images = page_data.get("images", [])

            for img_idx, img_data in enumerate(images):
                # img_data chứa base64 image
                img_b64 = img_data.get("data", "")
                if not img_b64:
                    continue

                caption_data = self._caption_image(img_b64)

                # Bỏ qua ảnh không phải chart
                if not caption_data.get("caption"):
                    continue

                content = f"{caption_data['caption']}\n\nKey data: {caption_data.get('key_data', '')}"
                img_path = f"uploads/{metadata.get('doc_id', 'unknown')}_page{page_num}_img{img_idx}.png"

                chunk = {
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "metadata": {
                        **metadata,
                        "page_num": page_num,
                        "chunk_type": "image_caption",
                        "chart_type": caption_data.get("chart_type", "other"),
                        "image_path": img_path,
                    },
                }
                chunks.append(chunk)

        print(f"[ImageProcessor] Captioned {len(chunks)} image chunks")
        return chunks

    def _caption_image(self, image_b64: str) -> Dict:
        """Gọi Gemini để caption ảnh."""
        import json

        try:
            image_bytes = base64.b64decode(image_b64)
            response = self.vision_model.generate_content(
                [
                    CAPTION_PROMPT,
                    {"mime_type": "image/png", "data": image_bytes},
                ]
            )
            text = response.text.strip()
            # Bỏ markdown code block nếu có
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            print(f"[ImageProcessor] Caption failed: {e}")
            return {"caption": None, "key_data": None, "chart_type": "non_chart"}