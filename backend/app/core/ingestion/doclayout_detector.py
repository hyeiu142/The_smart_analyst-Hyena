import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz
from PIL import Image

from backend.app.config import get_settings


os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

settings = get_settings()


@dataclass
class DetectedFigure:
    path: str
    page_number: int
    label: str
    confidence: float
    bbox: list[int]


class DocLayoutFigureDetector:
    """Render PDF pages and crop DocLayout-YOLO figure regions."""

    def __init__(self):
        self.model_path = settings.doclayout_model_path
        self.device = settings.doclayout_device
        self.conf = settings.doclayout_conf
        self.dpi = settings.doclayout_dpi
        self.imgsz = settings.doclayout_imgsz
        self._model = None

    def _load_model(self):
        if self._model is None:
            from doclayout_yolo import YOLOv10

            self._model = YOLOv10(self.model_path)
        return self._model

    def extract_figures(
        self,
        pdf_path: str,
        doc_id: str,
        output_dir: str,
    ) -> List[dict]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        model = self._load_model()
        figures: List[dict] = []
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_index in range(len(doc)):
            page_num = page_index + 1
            pix = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
            page_image_path = os.path.join(output_dir, f"page_{page_num}_render.png")
            pix.save(page_image_path)

            with Image.open(page_image_path) as image:
                image = image.convert("RGB")
                page_width, page_height = image.size

                results = model.predict(
                    page_image_path,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    device=self.device,
                    verbose=False,
                )
                if not results:
                    continue

                names = getattr(results[0], "names", None) or getattr(model, "names", {}) or {}
                boxes = getattr(results[0], "boxes", None)
                if boxes is None:
                    continue

                page_figures = 0
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    label = str(dict(names).get(class_id, class_id))
                    if self._normalize_label(label) != "figure":
                        continue

                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = self._clamp_box(
                        box.xyxy[0].tolist(),
                        page_width=page_width,
                        page_height=page_height,
                        padding=8,
                    )
                    if self._should_skip_box(x1, y1, x2, y2, page_width, page_height):
                        continue

                    page_figures += 1
                    crop_path = os.path.join(
                        output_dir,
                        f"page_{page_num}_{page_figures:02d}_figure.png",
                    )
                    image.crop((x1, y1, x2, y2)).save(crop_path)

                    figures.append(
                        DetectedFigure(
                            path=crop_path,
                            page_number=page_num,
                            label=label,
                            confidence=round(confidence, 4),
                            bbox=[x1, y1, x2, y2],
                        ).__dict__
                    )

            try:
                os.remove(page_image_path)
            except OSError:
                pass

        return figures

    def _clamp_box(
        self,
        xyxy: list[float],
        page_width: int,
        page_height: int,
        padding: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = xyxy
        return (
            max(0, round(x1) - padding),
            max(0, round(y1) - padding),
            min(page_width, round(x2) + padding),
            min(page_height, round(y2) + padding),
        )

    def _should_skip_box(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        page_width: int,
        page_height: int,
    ) -> bool:
        width = x2 - x1
        height = y2 - y1
        if width < 120 or height < 80:
            return True

        area_ratio = (width * height) / (page_width * page_height)
        return area_ratio > 0.45

    def _normalize_label(self, label: str) -> str:
        return label.lower().replace(" ", "_").replace("-", "_")
