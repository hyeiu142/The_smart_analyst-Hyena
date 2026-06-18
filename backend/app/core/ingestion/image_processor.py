import os
import uuid
from typing import Any, Dict, List

from openai import OpenAI
from llama_parse import LlamaParse

from backend.app.config import get_settings

settings = get_settings()

CAPTION_PROMPT = """
You are acting as an expert financial data analyst. You are analyzing a chart, graph, or image from a financial report.
Please extract the maximum amount of detail possible to ensure accurate semantic search later.

Provide the following:
1. "caption": A highly detailed paragraph (not just 1-2 sentences) describing exactly what the chart illustrates. Include the axes, units, timeframes, and the overall context (e.g., "This dual-axis chart illustrates the ICT Revenue in billions USD alongside the YoY growth percentage from 2018 to 2025E...").
2. "key_data": Extract ALL visible data points, numbers, and categories into a well-formatted Markdown Table. After the table, list 2-3 key analytical insights (peaks, major drops, notable trends).
3. "chart_type": (bar_chart, line_chart, pie_chart, table_image, diagram, other).

Respond EXACTLY in this JSON format:
{
    "caption": "Detailed description here...",
    "key_data": "Markdown table here... \n\nInsights: ...",
    "chart_type": "bar_chart"
}

If this is completely NOT a chart/graph/data table (e.g. company logo, decorative photo, signature), respond:
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
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

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

        doc_id = metadata.get("doc_id", "unknown")
        upload_dir = getattr(settings, 'UPLOAD_DIR', 'uploads')
        download_path = os.path.join(upload_dir, 'images', doc_id)
        os.makedirs(download_path, exist_ok=True)

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

            file_size_kb = os.path.getsize(img_path) / 1024
            if file_size_kb < 15: 
                print(f"[ImageProcessor] Skipping tiny image ({file_size_kb:.1f}KB): {img_path}")
                os.remove(img_path)
                continue

            relative_image_url = f"/uploads/images/{doc_id}/{os.path.basename(img_path)}"
            
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
                    "image_path": relative_image_url,
                },
            }
            chunks.append(chunk)

        print(f"[ImageProcessor] Captioned {len(chunks)} image chunks")
        return chunks

    def _caption_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> Dict:
        """Call OpenAI gpt-4o-mini to caption an image given raw bytes."""
        import json
        import base64

        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAPTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
            
            text = response.choices[0].message.content.strip()
            # Bỏ markdown code block nếu có
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            print(f"[ImageProcessor] Caption failed: {e}")
            return {"caption": None, "key_data": None, "chart_type": "non_chart"}