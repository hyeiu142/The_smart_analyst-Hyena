import os
import re
import uuid
import mimetypes
from typing import Any, Dict, List

import httpx
from openai import OpenAI

from backend.app.config import get_settings
from backend.app.core.ingestion.doclayout_detector import DocLayoutFigureDetector

settings = get_settings()

CAPTION_PROMPT = """
You are acting as an expert financial data analyst. You are analyzing a chart, graph, or image from a financial report.
Please extract the maximum amount of detail possible to ensure accurate semantic search later.

Provide the following:
1. "caption": A highly detailed paragraph (not just 1-2 sentences) describing exactly what the chart illustrates. Include the original visible title, axes, units, timeframes, legend labels, and the overall context. Preserve Vietnamese labels exactly when visible, then add English explanations if useful.
2. "key_data": Extract ALL visible data points, numbers, and categories into a well-formatted Markdown Table. After the table, list 2-3 key analytical insights (peaks, major drops, notable trends).
3. "chart_type": (bar_chart, line_chart, pie_chart, table_image, diagram, other).

Important accuracy rules:
- Do NOT infer or invent values. If a value is not visible, write "not visible" instead of 0.
- For stacked bar charts, map each colored segment using the legend exactly. Read the segment label printed inside that color block.
- Pay special attention to the latest year/rightmost bar and preserve decimal separators exactly (e.g. 23.1% vs 23,1%).
- If the chart title or legend is in Vietnamese, include those Vietnamese words in the caption/key_data so Vietnamese retrieval queries can match the chart.
- Never shift values between legend categories. For example, if red is "Mỹ", the red label belongs to "Mỹ" only.

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
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.figure_detector = DocLayoutFigureDetector()

    async def process(
        self,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect chart regions with DocLayout-YOLO, caption each crop, and create
        searchable image chunks during ingestion.
        """
        doc_id = metadata.get("doc_id", "unknown")
        upload_dir = settings.upload_dir
        download_path = os.path.join(upload_dir, 'images', doc_id)
        os.makedirs(download_path, exist_ok=True)

        images = self.figure_detector.extract_figures(
            pdf_path=file_path,
            doc_id=doc_id,
            output_dir=download_path,
        )
        print(f"[ImageProcessor] DocLayout-YOLO found {len(images)} figure crops")

        chunks = []
        for img_data in images:
            img_path = img_data.get("path", "")
            page_num = int(img_data.get("page_number") or 1)
            print(f"[ImageProcessor] Processing image: path={img_path}, page={page_num}, exists={os.path.exists(img_path) if img_path else False}")

            if not img_path or not os.path.exists(img_path):
                print(f"[ImageProcessor] Skipping - file not found: {img_path}")
                continue

            if not self._is_valid_image_file(img_path):
                print(f"[ImageProcessor] Skipping - invalid image file: {img_path}")
                continue

            file_size_kb = os.path.getsize(img_path) / 1024
            if file_size_kb < 15: 
                print(f"[ImageProcessor] Skipping tiny image ({file_size_kb:.1f}KB): {img_path}")
                continue

            relative_image_url = f"/uploads/images/{doc_id}/{os.path.basename(img_path)}"

            with open(img_path, "rb") as image_file:
                image_bytes = image_file.read()
            mime_type = mimetypes.guess_type(img_path)[0] or "image/png"
            caption_data = self._caption_image_bytes(image_bytes, mime_type)
            if not caption_data.get("caption"):
                print(f"[ImageProcessor] Skipping non-chart/unreadable crop: {img_path}")
                continue

            chart_type = caption_data.get("chart_type", "other")
            content = (
                f"{caption_data['caption']}\n\n"
                f"Key data: {caption_data.get('key_data') or ''}"
            )
            chunk = {
                "id": str(uuid.uuid4()),
                "content": content,
                "metadata": {
                    **metadata,
                    "page_num": page_num,
                    "chunk_type": "image_caption",
                    "chart_type": chart_type,
                    "image_status": "described",
                    "image_path": relative_image_url,
                    "local_image_path": img_path,
                    "bbox": img_data.get("bbox"),
                    "detection_label": img_data.get("label"),
                    "detection_confidence": img_data.get("confidence"),
                },
            }
            chunks.append(chunk)

        print(f"[ImageProcessor] Created {len(chunks)} described YOLO image chunks")
        return chunks

    def describe_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any] | None:
        """Caption one pending image chunk on demand."""
        metadata = dict(chunk.get("metadata") or {})
        if metadata.get("image_status") == "described":
            return None

        img_path = metadata.get("local_image_path")
        if not img_path and metadata.get("image_path"):
            relative_path = str(metadata["image_path"]).lstrip("/")
            if relative_path.startswith("uploads/"):
                relative_path = relative_path[len("uploads/") :]
            img_path = os.path.join(settings.upload_dir, relative_path)

        if not img_path or not os.path.exists(img_path) or not self._is_valid_image_file(img_path):
            print(f"[ImageProcessor] Lazy describe skipped - missing image: {img_path}")
            return None

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        mime_type = mimetypes.guess_type(img_path)[0] or "image/png"
        caption_data = self._caption_image_bytes(img_bytes, mime_type)
        if not caption_data.get("caption"):
            metadata.update({
                "chunk_type": "image_caption",
                "chart_type": "non_chart",
                "image_status": "described",
            })
            content = "Non-chart image. No financial chart data was extracted."
        else:
            metadata.update({
                "chunk_type": "image_caption",
                "chart_type": caption_data.get("chart_type", "other"),
                "image_status": "described",
            })
            content = f"{caption_data['caption']}\n\nKey data: {caption_data.get('key_data', '')}"

        return {
            "id": chunk["id"],
            "content": content,
            "metadata": metadata,
        }

    def _download_images_from_json(self, json_result: List[dict], download_path: str) -> List[Dict[str, Any]]:
        """Download LlamaParse image artifacts with HTTP/file validation."""
        if not json_result:
            return []

        headers = {"Authorization": f"Bearer {settings.llama_cloud_api_key}"}
        images: List[Dict[str, Any]] = []

        with httpx.Client(timeout=60.0) as client:
            for result in json_result:
                job_id = result.get("job_id")
                pages = result.get("pages") or []
                if not job_id:
                    print("[ImageProcessor] Skipping result without job_id")
                    continue

                for page in pages:
                    page_number = page.get("page", page.get("page_number", 1))
                    for image in page.get("images") or []:
                        image_name = image.get("name") or image.get("image_name")
                        if not image_name:
                            continue

                        filename = f"{job_id}-{image_name}"
                        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                            filename += ".png"

                        image_path = os.path.join(download_path, filename)
                        image_url = f"{self.parser.base_url}/api/parsing/job/{job_id}/result/image/{image_name}"

                        try:
                            response = client.get(image_url, headers=headers)
                            response.raise_for_status()
                        except Exception as exc:
                            print(f"[ImageProcessor] Download failed for {image_name}: {exc}")
                            continue

                        content_type = response.headers.get("content-type", "")
                        if "image" not in content_type and not self._looks_like_image(response.content):
                            print(f"[ImageProcessor] Download skipped - non-image response for {image_name}: {content_type}")
                            continue

                        with open(image_path, "wb") as f:
                            f.write(response.content)

                        images.append({
                            **image,
                            "path": image_path,
                            "job_id": job_id,
                            "page_number": page_number,
                            "original_pdf_path": result.get("file_path"),
                        })

        return images

    def _collect_local_images(self, download_path: str) -> List[Dict[str, Any]]:
        """Fallback for reruns: reuse valid images already present on disk."""
        if not os.path.isdir(download_path):
            return []

        images = []
        for filename in sorted(os.listdir(download_path)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            path = os.path.join(download_path, filename)
            if self._is_valid_image_file(path):
                images.append({"path": path, "page_number": self._infer_page_number({"path": path})})

        if images:
            print(f"[ImageProcessor] Reusing {len(images)} local images from {download_path}")
        return images

    def _infer_page_number(self, image_data: Dict[str, Any]) -> int:
        explicit = image_data.get("page_number") or image_data.get("page")
        if explicit:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                pass

        path = image_data.get("path", "")
        filename = os.path.basename(path)
        match = re.search(r"(?:page_|img_p)(\d+)", filename)
        if not match:
            return 1

        page = int(match.group(1))
        return page + 1 if "img_p" in filename else page

    def _is_valid_image_file(self, image_path: str) -> bool:
        try:
            with open(image_path, "rb") as f:
                return self._looks_like_image(f.read(16))
        except OSError:
            return False

    def _looks_like_image(self, data: bytes) -> bool:
        return (
            data.startswith(b"\x89PNG\r\n\x1a\n")
            or data.startswith(b"\xff\xd8\xff")
            or data.startswith(b"RIFF") and data[8:12] == b"WEBP"
        )

    def _caption_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> Dict:
        """Call OpenAI gpt-4o-mini to caption an image given raw bytes."""
        import json
        import base64

        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model=settings.vision_model,
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
                temperature=0,
                max_tokens=1200
            )
            
            text = response.choices[0].message.content.strip()
            # Bỏ markdown code block nếu có
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            else:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    text = match.group(0)
            return json.loads(text)
        except Exception as e:
            print(f"[ImageProcessor] Caption failed: {e}")
            return {"caption": None, "key_data": None, "chart_type": "non_chart"}
