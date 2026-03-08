import os
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
            extract_images=True,
        )
        genai.configure(api_key=settings.google_api_key)
        self.vision_model = genai.GenerativeModel("gemini-2.0-flash-exp")

    async def process(
        self,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract images via LlamaParse → caption with Gemini.

        LlamaParse flow:
        1. aget_json() → parse result with image names/references
        2. get_images(json_result, download_path) → download images to disk
        3. Read each image file → send to Gemini for captioning
        """
        import tempfile

        # Step 1: parse
        json_result = await self.parser.aget_json(file_path)

        # DEBUG: inspect json_result structure
        print(f"[ImageProcessor] json_result type={type(json_result)}, len={len(json_result) if json_result else 0}")
        if json_result:
            first = json_result[0]
            print(f"[ImageProcessor] first keys={list(first.keys()) if isinstance(first, dict) else type(first)}")
            if isinstance(first, dict):
                has_job_id = "job_id" in first
                has_pages = "pages" in first
                print(f"[ImageProcessor] has job_id={has_job_id}, has pages={has_pages}")
                if has_pages:
                    pages = first.get("pages", [])
                    print(f"[ImageProcessor] page count={len(pages)}")
                    if pages:
                        page0 = pages[0]
                        print(f"[ImageProcessor] page[0] keys={list(page0.keys())}")
                        print(f"[ImageProcessor] page[0] images={page0.get('images', [])}")

        # Step 2: download images to a temp directory
        doc_id = metadata.get("doc_id", "unknown")
        download_path = os.path.join(tempfile.gettempdir(), f"hyena_images_{doc_id}")

        # get_images is sync — call directly (Celery tasks run in subprocess, asyncio.to_thread unreliable)
        images = self.parser.get_images(json_result, download_path)

        print(f"[ImageProcessor] Downloaded {len(images)} images from LlamaParse")

        chunks = []
        for img_data in images:
            img_path = img_data.get("path", "")
            page_num = img_data.get("page_number", 1)
            print(f"[ImageProcessor] Processing image: path={img_path}, page={page_num}, exists={os.path.exists(img_path) if img_path else False}")

            if not img_path or not os.path.exists(img_path):
                print(f"[ImageProcessor] Skipping — file not found: {img_path}")
                continue

            # Step 3: read file bytes and send to Gemini
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            # detect mime type from extension
            mime_type = "image/jpeg" if img_path.lower().endswith(".jpg") else "image/png"
            caption_data = self._caption_image_bytes(img_bytes, mime_type)
            print(f"[ImageProcessor] Caption result: {caption_data}")

            # Skip non-chart images (logos etc.)
            if not caption_data.get("caption"):
                print(f"[ImageProcessor] Skipping — non_chart or caption failed")
                continue

            content = f"{caption_data['caption']}\n\nKey data: {caption_data.get('key_data', '')}"

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

    def _caption_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> Dict:
        """Call Gemini to caption an image given raw bytes."""
        import json

        try:
            response = self.vision_model.generate_content(
                [
                    CAPTION_PROMPT,
                    {"mime_type": mime_type, "data": image_bytes},
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