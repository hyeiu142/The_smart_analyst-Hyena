#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


SKIP_TYPES = {"full_page_screenshot", "page_screenshot"}


def find_pages(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("pages"), list):
        return data["pages"]

    if isinstance(data.get("documents"), list):
        for document in data["documents"]:
            if isinstance(document, dict) and isinstance(document.get("pages"), list):
                return document["pages"]

    return []


def page_number(page: dict[str, Any], fallback: int) -> int:
    for key in ("page", "page_number", "page_num"):
        value = page.get(key)
        if isinstance(value, int):
            return value
    return fallback


def collect_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for key in ("images", "charts", "figures"):
        value = page.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def read_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if all(key in item for key in ("x", "y", "width", "height")):
        return float(item["x"]), float(item["y"]), float(item["width"]), float(item["height"])

    bbox = item.get("bbox") or item.get("bounding_box")

    if isinstance(bbox, dict):
        if all(key in bbox for key in ("x", "y", "width", "height")):
            return float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"])
        if all(key in bbox for key in ("x1", "y1", "x2", "y2")):
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
            return x1, y1, x2 - x1, y2 - y1

    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return x1, y1, x2 - x1, y2 - y1

    return None


def scale_bbox(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    scale_x = image_width / page_width
    scale_y = image_height / page_height

    return (
        round(x * scale_x),
        round(y * scale_y),
        round((x + width) * scale_x),
        round((y + height) * scale_y),
    )


def add_padding(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding_x_ratio: float,
    padding_y_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    crop_width = x2 - x1
    crop_height = y2 - y1
    padding_x = round(crop_width * padding_x_ratio)
    padding_y = round(crop_height * padding_y_ratio)

    return (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(image_width, x2 + padding_x),
        min(image_height, y2 + padding_y),
    )


def should_skip(
    item: dict[str, Any],
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    min_width: int,
    min_height: int,
    max_area_ratio: float,
) -> bool:
    item_type = str(item.get("type", "")).lower()
    if item_type in SKIP_TYPES:
        return True

    x1, y1, x2, y2 = box
    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width < min_width or crop_height < min_height:
        return True

    area_ratio = (crop_width * crop_height) / (image_width * image_height)
    return area_ratio >= max_area_ratio


def crop_from_llama_bboxes(
    llama_json_path: Path,
    pages_dir: Path,
    out_dir: Path,
    page_template: str,
    padding_x_ratio: float,
    padding_y_ratio: float,
    min_width: int,
    min_height: int,
    max_area_ratio: float,
) -> int:
    data = json.loads(llama_json_path.read_text(encoding="utf-8"))
    pages = find_pages(data)

    if not pages:
        raise ValueError("No pages found in LlamaParse JSON.")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for index, page in enumerate(pages, start=1):
        num = page_number(page, index)
        page_image_path = pages_dir / page_template.format(page=num, page_num=num)

        if not page_image_path.exists():
            print(f"Skip page {num}: missing image {page_image_path}")
            continue

        page_width = float(page.get("width") or page.get("page_width") or 0)
        page_height = float(page.get("height") or page.get("page_height") or 0)
        if page_width <= 0 or page_height <= 0:
            print(f"Skip page {num}: missing page width/height in JSON")
            continue

        image = Image.open(page_image_path)
        image_width, image_height = image.size

        for item_index, item in enumerate(collect_items(page), start=1):
            bbox = read_bbox(item)
            if bbox is None:
                continue

            pixel_box = scale_bbox(bbox, page_width, page_height, image_width, image_height)
            padded_box = add_padding(pixel_box, image_width, image_height, padding_x_ratio, padding_y_ratio)

            if should_skip(item, padded_box, image_width, image_height, min_width, min_height, max_area_ratio):
                continue

            item_type = str(item.get("type") or "image").lower().replace(" ", "_")
            out_path = out_dir / f"page_{num}_{item_index:02d}_{item_type}.png"
            image.crop(padded_box).save(out_path)

            saved_count += 1
            print(f"Saved {out_path}")

    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop charts/images from rendered pages using LlamaParse bboxes.")
    parser.add_argument("llama_json", type=Path)
    parser.add_argument("--pages-dir", type=Path, default=Path("pages"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--page-template", default="page_{page}.png")
    parser.add_argument("--padding-x", type=float, default=0.15)
    parser.add_argument("--padding-y", type=float, default=0.25)
    parser.add_argument("--min-width", type=int, default=120)
    parser.add_argument("--min-height", type=int, default=80)
    parser.add_argument("--max-area-ratio", type=float, default=0.92)

    args = parser.parse_args()

    saved_count = crop_from_llama_bboxes(
        llama_json_path=args.llama_json,
        pages_dir=args.pages_dir,
        out_dir=args.out_dir,
        page_template=args.page_template,
        padding_x_ratio=args.padding_x,
        padding_y_ratio=args.padding_y,
        min_width=args.min_width,
        min_height=args.min_height,
        max_area_ratio=args.max_area_ratio,
    )
    print(f"Done. Saved {saved_count} crops.")


if __name__ == "__main__":
    main()
