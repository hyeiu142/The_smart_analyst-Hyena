#!/usr/bin/env python3
"""
Crop chart/image regions from rendered PDF pages using LlamaParse bboxes.

Responsibility:
    Pillow crops final images. This script does not call LlamaParse and does not
    render PDF pages.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SKIP_TYPES = {
    "full_page_screenshot",
    "page_screenshot",
    "screenshot",
}

ITEM_LIST_KEYS = (
    "charts",
    "images",
    "figures",
    "items",
    "layout",
    "blocks",
)


@dataclass(frozen=True)
class CropCandidate:
    page: int
    index: int
    item_type: str
    bbox: tuple[float, float, float, float]
    raw: dict[str, Any]


def find_pages(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("pages"), list):
        return data["pages"]

    documents = data.get("documents")
    if isinstance(documents, list):
        pages: list[dict[str, Any]] = []
        for document in documents:
            if isinstance(document, dict) and isinstance(document.get("pages"), list):
                pages.extend(document["pages"])
        return pages

    return []


def page_number(page: dict[str, Any], fallback: int) -> int:
    for key in ("page", "page_number", "page_num", "page_index"):
        value = page.get(key)
        if isinstance(value, int):
            return value + 1 if key == "page_index" else value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return fallback


def page_size(page: dict[str, Any]) -> tuple[float, float] | None:
    width = page.get("width") or page.get("page_width")
    height = page.get("height") or page.get("page_height")

    if width and height:
        return float(width), float(height)

    bbox = read_bbox(page)
    if bbox:
        _, _, bbox_width, bbox_height = bbox
        if bbox_width > 0 and bbox_height > 0:
            return bbox_width, bbox_height

    return None


def iter_dict_items(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for key in ITEM_LIST_KEYS:
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    yield from iter_dict_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_dict_items(item)


def collect_page_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    items: list[dict[str, Any]] = []

    for key in ITEM_LIST_KEYS:
        value = page.get(key)
        if not isinstance(value, list):
            continue

        for item in iter_dict_items(value):
            marker = id(item)
            if marker not in seen:
                seen.add(marker)
                items.append(item)

    return items


def read_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if all(key in item for key in ("x", "y", "width", "height")):
        return (
            float(item["x"]),
            float(item["y"]),
            float(item["width"]),
            float(item["height"]),
        )

    bbox = (
        item.get("bbox")
        or item.get("bounding_box")
        or item.get("bounds")
        or item.get("box")
    )

    if isinstance(bbox, dict):
        if all(key in bbox for key in ("x", "y", "width", "height")):
            return (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["width"]),
                float(bbox["height"]),
            )

        if all(key in bbox for key in ("left", "top", "width", "height")):
            return (
                float(bbox["left"]),
                float(bbox["top"]),
                float(bbox["width"]),
                float(bbox["height"]),
            )

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


def item_type(item: dict[str, Any]) -> str:
    value = item.get("type") or item.get("category") or item.get("label") or "image"
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip().lower())
    return normalized.strip("_") or "image"


def build_candidates(page: dict[str, Any], fallback_page_num: int) -> list[CropCandidate]:
    num = page_number(page, fallback_page_num)
    candidates: list[CropCandidate] = []

    for item in collect_page_items(page):
        bbox = read_bbox(item)
        if not bbox:
            continue

        candidates.append(
            CropCandidate(
                page=num,
                index=len(candidates) + 1,
                item_type=item_type(item),
                bbox=bbox,
                raw=item,
            )
        )

    return candidates


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
    crop_width = max(0, x2 - x1)
    crop_height = max(0, y2 - y1)

    padding_x = round(crop_width * padding_x_ratio)
    padding_y = round(crop_height * padding_y_ratio)

    return (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(image_width, x2 + padding_x),
        min(image_height, y2 + padding_y),
    )


def should_skip(
    candidate: CropCandidate,
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    min_width: int,
    min_height: int,
    max_area_ratio: float,
) -> bool:
    if candidate.item_type in SKIP_TYPES:
        return True

    x1, y1, x2, y2 = box
    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width <= 0 or crop_height <= 0:
        return True

    if crop_width < min_width or crop_height < min_height:
        return True

    page_area = image_width * image_height
    crop_area = crop_width * crop_height
    if page_area <= 0:
        return True

    return (crop_area / page_area) >= max_area_ratio


def crop_from_llama_bboxes(
    llama_json_path: Path,
    pages_dir: Path,
    out_dir: Path,
    page_template: str = "page_{page}.png",
    padding_x_ratio: float = 0.03,
    padding_y_ratio: float = 0.04,
    min_width: int = 120,
    min_height: int = 80,
    max_area_ratio: float = 0.92,
) -> int:
    data = json.loads(llama_json_path.read_text(encoding="utf-8"))
    pages = find_pages(data)

    if not pages:
        raise ValueError("No pages found in LlamaParse JSON.")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for fallback_page_num, page in enumerate(pages, start=1):
        num = page_number(page, fallback_page_num)
        size = page_size(page)
        if not size:
            print(f"Skip page {num}: missing page width/height in JSON")
            continue

        page_width, page_height = size
        page_image_path = pages_dir / page_template.format(page=num, page_num=num)
        if not page_image_path.exists():
            print(f"Skip page {num}: missing image {page_image_path}")
            continue

        with Image.open(page_image_path) as image:
            image_width, image_height = image.size

            for candidate in build_candidates(page, num):
                pixel_box = scale_bbox(
                    bbox=candidate.bbox,
                    page_width=page_width,
                    page_height=page_height,
                    image_width=image_width,
                    image_height=image_height,
                )
                padded_box = add_padding(
                    box=pixel_box,
                    image_width=image_width,
                    image_height=image_height,
                    padding_x_ratio=padding_x_ratio,
                    padding_y_ratio=padding_y_ratio,
                )

                if should_skip(
                    candidate=candidate,
                    box=padded_box,
                    image_width=image_width,
                    image_height=image_height,
                    min_width=min_width,
                    min_height=min_height,
                    max_area_ratio=max_area_ratio,
                ):
                    continue

                out_path = out_dir / f"page_{num}_{candidate.index:02d}_{candidate.item_type}.png"
                image.crop(padded_box).save(out_path)
                saved_count += 1
                print(f"Saved {out_path}")

    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop charts/images from rendered pages using LlamaParse bboxes.")
    parser.add_argument("llama_json", type=Path, help="LlamaParse JSON path.")
    parser.add_argument("--pages-dir", type=Path, default=Path("pages"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--page-template", default="page_{page}.png")
    parser.add_argument("--padding-x", type=float, default=0.03)
    parser.add_argument("--padding-y", type=float, default=0.04)
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
