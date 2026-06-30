import os
from pathlib import Path

import pytest
from PIL import Image

from backend.app.core.ingestion.doclayout_detector import DocLayoutFigureDetector
from backend.app.core.ingestion.image_processor import ImageProcessor


class FakeFigureDetector:
    def __init__(self, crop_path: Path):
        self.crop_path = crop_path

    def extract_figures(self, pdf_path: str, doc_id: str, output_dir: str) -> list[dict]:
        return [
            {
                "path": str(self.crop_path),
                "page_number": 4,
                "label": "figure",
                "confidence": 0.91,
                "bbox": [10, 20, 300, 220],
            }
        ]


@pytest.mark.asyncio
async def test_process_creates_described_chunk_from_yolo_crop(tmp_path, monkeypatch):
    crop_path = tmp_path / "page_4_01_figure.png"
    Image.frombytes("RGB", (800, 600), os.urandom(800 * 600 * 3)).save(crop_path)

    processor = ImageProcessor.__new__(ImageProcessor)
    processor.figure_detector = FakeFigureDetector(crop_path)
    processor._caption_image_bytes = lambda _data, _mime: {
        "caption": "Biểu đồ cơ cấu doanh thu theo thị trường.",
        "key_data": "Nhật Bản 43,7%; Mỹ 23,1%; APAC 23,1%.",
        "chart_type": "stacked_bar",
    }
    monkeypatch.setattr(
        "backend.app.core.ingestion.image_processor.settings.upload_dir",
        str(tmp_path),
    )

    chunks = await processor.process(
        file_path="report.pdf",
        metadata={"doc_id": "doc-1", "company": "FPT", "year": 2025},
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "cơ cấu doanh thu theo thị trường" in chunk["content"]
    assert chunk["metadata"]["page_num"] == 4
    assert chunk["metadata"]["image_status"] == "described"
    assert chunk["metadata"]["chunk_type"] == "image_caption"
    assert chunk["metadata"]["bbox"] == [10, 20, 300, 220]


def test_detector_keeps_extra_space_above_figure_title():
    detector = DocLayoutFigureDetector.__new__(DocLayoutFigureDetector)

    box = detector._clamp_box(
        [100, 100, 300, 300],
        page_width=500,
        page_height=500,
        padding=8,
    )

    assert box == (92, 44, 308, 308)
