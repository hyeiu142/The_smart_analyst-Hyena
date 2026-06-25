#!/usr/bin/env python3
"""
DocLayout-YOLO experiment for local layout detection.

Flow:
    PDF -> render page images -> DocLayout-YOLO layout boxes -> debug overlays -> crops

This script is intentionally kept under research/image_extraction and is not
wired into the main ingestion pipeline.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import fitz
from PIL import Image, ImageDraw, ImageFont
from doclayout_yolo import YOLOv10


@dataclass
class Detection:
    page: int
    index: int
    label: str
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    crop_path: str | None = None


def render_pdf_pages(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int,
    first_page: int | None,
    last_page: int | None,
) -> list[tuple[int, Path]]:
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    start = max(1, first_page or 1)
    end = min(len(doc), last_page or len(doc))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    paths: list[tuple[int, Path]] = []
    for page_num in range(start, end + 1):
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out_path = pages_dir / f"page_{page_num}.png"
        pix.save(out_path)
        paths.append((page_num, out_path))

    return paths


def normalize_label(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_")


def should_crop(label: str, crop_labels: set[str]) -> bool:
    if not crop_labels:
        return True
    return normalize_label(label) in crop_labels


def should_keep_box(
    label: str,
    confidence: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    page_width: int,
    page_height: int,
    max_area_ratio: float,
    min_width: int,
    min_height: int,
) -> bool:
    width = x2 - x1
    height = y2 - y1
    if width < min_width or height < min_height:
        return False

    area_ratio = (width * height) / (page_width * page_height)
    if area_ratio > max_area_ratio:
        return False

    normalized = normalize_label(label)
    if normalized in {"figure_caption", "table_caption", "table_footnote", "abandon"}:
        return False

    return True


def clamp_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: int,
    top_padding: int | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    effective_top_padding = padding if top_padding is None else top_padding
    return (
        max(0, round(x1) - padding),
        max(0, round(y1) - effective_top_padding),
        min(width, round(x2) + padding),
        min(height, round(y2) + padding),
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in normalize_label(value)).strip("_")


def draw_overlay(image: Image.Image, detections: list[Detection], out_path: Path) -> None:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for det in detections:
        draw.rectangle((det.x1, det.y1, det.x2, det.y2), outline="red", width=4)
        text = f"{det.index} {det.label} {det.confidence:.2f}"
        text_bbox = draw.textbbox((det.x1 + 4, det.y1 + 4), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((det.x1 + 4, det.y1 + 4), text, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def result_names(result: Any, model: YOLOv10) -> dict[int, str]:
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}
    return {int(key): str(value) for key, value in dict(names).items()}


def detect_page(
    model: YOLOv10,
    image_path: Path,
    page_num: int,
    out_dir: Path,
    debug_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
    padding: int,
    top_padding: int,
    crop_labels: set[str],
    max_area_ratio: float,
    min_width: int,
    min_height: int,
) -> list[Detection]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size

        results = model.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)
        if not results:
            draw_overlay(image, [], debug_dir / f"page_{page_num}_debug.png")
            return []

        result = results[0]
        names = result_names(result, model)
        detections: list[Detection] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            draw_overlay(image, [], debug_dir / f"page_{page_num}_debug.png")
            return []

        for raw_index, box in enumerate(boxes, start=1):
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = names.get(class_id, str(class_id))
            x1, y1, x2, y2 = clamp_box(tuple(xyxy), width, height, padding, top_padding)

            if not should_keep_box(
                label=label,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                page_width=width,
                page_height=height,
                max_area_ratio=max_area_ratio,
                min_width=min_width,
                min_height=min_height,
            ):
                continue

            det = Detection(
                page=page_num,
                index=len(detections) + 1,
                label=label,
                class_id=class_id,
                confidence=round(confidence, 4),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )

            if should_crop(label, crop_labels):
                crop_name = f"page_{page_num}_{det.index:02d}_{safe_name(label)}.png"
                crop_path = out_dir / crop_name
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop((x1, y1, x2, y2)).save(crop_path)
                det.crop_path = str(crop_path)

            detections.append(det)

        draw_overlay(image, detections, debug_dir / f"page_{page_num}_debug.png")
        return detections


def parse_labels(value: str) -> set[str]:
    if not value:
        return set()
    return {normalize_label(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DocLayout-YOLO layout detection on PDF pages.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("research/image_extraction/data/models/doclayout_yolo_docstructbench_imgsz1024.pt"))
    parser.add_argument("--pages-dir", type=Path, default=Path("research/image_extraction/data/doclayout_pages"))
    parser.add_argument("--out-dir", type=Path, default=Path("research/image_extraction/data/doclayout_outputs"))
    parser.add_argument("--debug-dir", type=Path, default=Path("research/image_extraction/data/doclayout_debug"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--first-page", type=int, default=None)
    parser.add_argument("--last-page", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument(
        "--top-padding",
        type=int,
        default=56,
        help="Extra upward crop padding to keep chart titles that sit just above detected figures.",
    )
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--min-width", type=int, default=120)
    parser.add_argument("--min-height", type=int, default=80)
    parser.add_argument(
        "--crop-labels",
        default="figure,table,picture",
        help="Comma-separated labels to crop. Empty string crops all labels.",
    )

    args = parser.parse_args()

    crop_labels = parse_labels(args.crop_labels)
    pages = render_pdf_pages(args.pdf, args.pages_dir, args.dpi, args.first_page, args.last_page)

    model = YOLOv10(str(args.model))

    all_detections: list[Detection] = []
    for page_num, image_path in pages:
        detections = detect_page(
            model=model,
            image_path=image_path,
            page_num=page_num,
            out_dir=args.out_dir,
            debug_dir=args.debug_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            padding=args.padding,
            top_padding=args.top_padding,
            crop_labels=crop_labels,
            max_area_ratio=args.max_area_ratio,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        all_detections.extend(detections)
        cropped = sum(1 for det in detections if det.crop_path)
        print(f"page={page_num} detections={len(detections)} crops={cropped}")

    manifest_path = args.manifest or args.out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(det) for det in all_detections], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Detections: {len(all_detections)}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
