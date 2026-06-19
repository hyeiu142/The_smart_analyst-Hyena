#!/usr/bin/env python3
"""
Fast local chart cropper for research reports.

This is an experiment-only script. It avoids LlamaParse/vision APIs and uses
PyMuPDF + Pillow + NumPy so image extraction can stay fast during ingestion.

Flow:
    PDF or page PNGs -> rendered page images -> local region detection -> crops
"""

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Region:
    page: int
    index: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    reason: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


def render_pdf_pages(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int,
    first_page: int | None,
    last_page: int | None,
) -> list[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    start = max(1, first_page or 1)
    end = min(len(doc), last_page or len(doc))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    paths: list[Path] = []
    for page_num in range(start, end + 1):
        out_path = pages_dir / f"page_{page_num}.png"
        pix = doc[page_num - 1].get_pixmap(matrix=matrix, alpha=False)
        pix.save(out_path)
        paths.append(out_path)

    return paths


def page_paths_from_dir(pages_dir: Path) -> list[Path]:
    def page_sort_key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else 0, path.name)

    return sorted(
        [
            path
            for path in pages_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ],
        key=page_sort_key,
    )


def infer_page_num(path: Path, fallback: int) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else fallback


def tile_activity_mask(
    image: Image.Image,
    tile: int,
    ink_threshold: float,
    color_threshold: float,
) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))

    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = max_channel - min_channel

    non_white = min_channel < 245
    colored = (saturation > 24) & (max_channel < 250)
    dark = max_channel < 170

    h, w = non_white.shape
    tile_rows = int(np.ceil(h / tile))
    tile_cols = int(np.ceil(w / tile))
    mask = np.zeros((tile_rows, tile_cols), dtype=bool)

    for row in range(tile_rows):
        y1 = row * tile
        y2 = min(h, y1 + tile)
        for col in range(tile_cols):
            x1 = col * tile
            x2 = min(w, x1 + tile)

            area = (y2 - y1) * (x2 - x1)
            if area <= 0:
                continue

            ink_ratio = float(non_white[y1:y2, x1:x2].sum()) / area
            color_ratio = float(colored[y1:y2, x1:x2].sum()) / area
            dark_ratio = float(dark[y1:y2, x1:x2].sum()) / area

            # Charts normally have colored marks spread across a large area.
            # Dark ratio keeps black/blue axis labels and table-like chart text.
            mask[row, col] = (
                color_ratio >= color_threshold
                or ink_ratio >= ink_threshold
                or (color_ratio >= color_threshold * 0.45 and dark_ratio >= 0.015)
            )

    return mask


def dilate(mask: np.ndarray, radius_x: int, radius_y: int) -> np.ndarray:
    rows, cols = mask.shape
    out = np.zeros_like(mask)
    active = np.argwhere(mask)

    for row, col in active:
        r1 = max(0, row - radius_y)
        r2 = min(rows, row + radius_y + 1)
        c1 = max(0, col - radius_x)
        c2 = min(cols, col + radius_x + 1)
        out[r1:r2, c1:c2] = True

    return out


def connected_components(mask: np.ndarray) -> Iterable[tuple[int, int, int, int, int]]:
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    for start_row in range(rows):
        for start_col in range(cols):
            if not mask[start_row, start_col] or visited[start_row, start_col]:
                continue

            q = deque([(start_row, start_col)])
            visited[start_row, start_col] = True
            min_row = max_row = start_row
            min_col = max_col = start_col
            count = 0

            while q:
                row, col = q.popleft()
                count += 1
                min_row = min(min_row, row)
                max_row = max(max_row, row)
                min_col = min(min_col, col)
                max_col = max(max_col, col)

                for next_row in (row - 1, row, row + 1):
                    for next_col in (col - 1, col, col + 1):
                        if next_row == row and next_col == col:
                            continue
                        if not (0 <= next_row < rows and 0 <= next_col < cols):
                            continue
                        if visited[next_row, next_col] or not mask[next_row, next_col]:
                            continue

                        visited[next_row, next_col] = True
                        q.append((next_row, next_col))

            yield min_row, min_col, max_row, max_col, count


def expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def boxes_overlap_or_close(
    a: Region,
    b: Region,
    gap_x: int,
    gap_y: int,
) -> bool:
    return not (
        a.x2 + gap_x < b.x1
        or b.x2 + gap_x < a.x1
        or a.y2 + gap_y < b.y1
        or b.y2 + gap_y < a.y1
    )


def merge_regions(regions: list[Region], gap_x: int, gap_y: int) -> list[Region]:
    if gap_x <= 0 and gap_y <= 0:
        return regions

    merged = regions[:]
    changed = True

    while changed:
        changed = False
        next_regions: list[Region] = []
        used = [False] * len(merged)

        for i, region in enumerate(merged):
            if used[i]:
                continue

            current = region
            used[i] = True

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                other = merged[j]
                if not boxes_overlap_or_close(current, other, gap_x, gap_y):
                    continue

                current = Region(
                    page=current.page,
                    index=current.index,
                    x1=min(current.x1, other.x1),
                    y1=min(current.y1, other.y1),
                    x2=max(current.x2, other.x2),
                    y2=max(current.y2, other.y2),
                    score=max(current.score, other.score),
                    reason="merged",
                )
                used[j] = True
                changed = True

            next_regions.append(current)

        merged = next_regions

    return merged


def low_ink_runs(values: np.ndarray, threshold: float, min_run: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(values):
        if value <= threshold:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= min_run:
                runs.append((start, index))
            start = None

    if start is not None and len(values) - start >= min_run:
        runs.append((start, len(values)))

    return runs


def choose_center_run(
    runs: list[tuple[int, int]],
    length: int,
    min_margin_ratio: float = 0.18,
) -> tuple[int, int] | None:
    if not runs:
        return None

    min_margin = int(length * min_margin_ratio)
    center = length / 2
    candidates = [
        run
        for run in runs
        if run[0] > min_margin and run[1] < length - min_margin
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda run: (abs(((run[0] + run[1]) / 2) - center), -(run[1] - run[0])),
    )


def split_region_once(image: Image.Image, region: Region) -> list[Region] | None:
    crop = np.asarray(image.crop((region.x1, region.y1, region.x2, region.y2)).convert("RGB"))
    if crop.size == 0:
        return None

    max_channel = crop.max(axis=2)
    min_channel = crop.min(axis=2)
    saturation = max_channel - min_channel
    ink = (min_channel < 245) | ((saturation > 24) & (max_channel < 250))

    height, width = ink.shape

    if width >= 520:
        col_ratio = ink.mean(axis=0)
        runs = low_ink_runs(col_ratio, threshold=0.012, min_run=max(18, width // 45))
        run = choose_center_run(runs, width)
        if run:
            split = (run[0] + run[1]) // 2
            left_width = split
            right_width = width - split
            if left_width >= 180 and right_width >= 180:
                return [
                    Region(region.page, region.index, region.x1, region.y1, region.x1 + split, region.y2, region.score, "split_x"),
                    Region(region.page, region.index, region.x1 + split, region.y1, region.x2, region.y2, region.score, "split_x"),
                ]

    if height >= 420:
        row_ratio = ink.mean(axis=1)
        runs = low_ink_runs(row_ratio, threshold=0.012, min_run=max(16, height // 45))
        run = choose_center_run(runs, height, min_margin_ratio=0.14)
        if run:
            split = (run[0] + run[1]) // 2
            top_height = split
            bottom_height = height - split
            if top_height >= 130 and bottom_height >= 130:
                return [
                    Region(region.page, region.index, region.x1, region.y1, region.x2, region.y1 + split, region.score, "split_y"),
                    Region(region.page, region.index, region.x1, region.y1 + split, region.x2, region.y2, region.score, "split_y"),
                ]

    return None


def split_large_regions(
    image: Image.Image,
    regions: list[Region],
    image_width: int,
    image_height: int,
    min_width: int,
    min_height: int,
) -> list[Region]:
    output: list[Region] = []
    queue = deque(regions)

    while queue:
        region = queue.popleft()
        area_ratio = region.area / (image_width * image_height)
        should_try_split = (
            area_ratio > 0.16
            or region.width > image_width * 0.62
            or region.height > image_height * 0.34
        )

        if should_try_split:
            parts = split_region_once(image, region)
            if parts:
                for part in parts:
                    if part.width >= min_width and part.height >= min_height:
                        queue.append(part)
                continue

        output.append(region)

    return output


def region_score(image: Image.Image, region: Region) -> float:
    crop = np.asarray(image.crop((region.x1, region.y1, region.x2, region.y2)).convert("RGB"))
    if crop.size == 0:
        return 0.0

    max_channel = crop.max(axis=2)
    min_channel = crop.min(axis=2)
    saturation = max_channel - min_channel
    non_white_ratio = float((min_channel < 245).sum()) / (region.area or 1)
    color_ratio = float(((saturation > 24) & (max_channel < 250)).sum()) / (region.area or 1)
    aspect_bonus = min(region.width / max(region.height, 1), region.height / max(region.width, 1))

    return round((non_white_ratio * 1.2) + (color_ratio * 3.0) + (aspect_bonus * 0.15), 4)


def longest_true_run(mask: np.ndarray, axis: int) -> int:
    best = 0
    lines = mask if axis == 1 else mask.T

    for line in lines:
        current = 0
        for value in line:
            if value:
                current += 1
                best = max(best, current)
            else:
                current = 0

    return best


def graphic_evidence(image: Image.Image, region: Region) -> tuple[float, float, float, float]:
    crop = np.asarray(image.crop((region.x1, region.y1, region.x2, region.y2)).convert("RGB"))
    if crop.size == 0 or region.area <= 0:
        return 0.0, 0.0, 0.0, 0.0

    max_channel = crop.max(axis=2)
    min_channel = crop.min(axis=2)
    saturation = max_channel - min_channel
    height, width = max_channel.shape

    colored = (saturation > 24) & (max_channel < 250)
    grid = (saturation < 12) & (min_channel > 170) & (min_channel < 238)
    very_colored = (saturation > 55) & (max_channel < 250)
    graphic_mask = very_colored | grid

    colored_ratio = float(colored.sum()) / region.area
    grid_ratio = float(grid.sum()) / region.area
    very_colored_ratio = float(very_colored.sum()) / region.area
    row_run_ratio = longest_true_run(graphic_mask, axis=1) / max(width, 1)
    col_run_ratio = longest_true_run(graphic_mask, axis=0) / max(height, 1)
    long_run_ratio = max(row_run_ratio, col_run_ratio)

    return colored_ratio, grid_ratio, very_colored_ratio, long_run_ratio


def looks_like_header(region: Region, image_width: int, image_height: int) -> bool:
    return (
        region.y1 < image_height * 0.07
        and region.width > image_width * 0.72
        and region.height < image_height * 0.24
    )


def looks_like_table_band(
    region: Region,
    colored_ratio: float,
    grid_ratio: float,
    image_width: int,
) -> bool:
    return (
        region.height < 170
        and region.width > image_width * 0.42
        and grid_ratio > 0.12
        and colored_ratio < 0.015
    )


def detect_regions(
    image: Image.Image,
    page_num: int,
    tile: int,
    ink_threshold: float,
    color_threshold: float,
    dilate_x: int,
    dilate_y: int,
    padding_x: int,
    padding_y: int,
    min_width: int,
    min_height: int,
    max_area_ratio: float,
    ignore_top_ratio: float,
    ignore_bottom_ratio: float,
    merge_gap_x: int,
    merge_gap_y: int,
    max_regions: int,
    min_score: float,
    min_low_color_score: float,
    min_colored_ratio: float,
    min_grid_ratio: float,
    min_very_colored_ratio: float,
    min_graphic_run_ratio: float,
) -> list[Region]:
    image_width, image_height = image.size
    mask = tile_activity_mask(image, tile, ink_threshold, color_threshold)
    mask = dilate(mask, dilate_x, dilate_y)

    regions: list[Region] = []
    ignore_top = int(image_height * ignore_top_ratio)
    ignore_bottom = int(image_height * (1.0 - ignore_bottom_ratio))

    for row1, col1, row2, col2, tile_count in connected_components(mask):
        x1 = col1 * tile
        y1 = row1 * tile
        x2 = min(image_width, (col2 + 1) * tile)
        y2 = min(image_height, (row2 + 1) * tile)
        x1, y1, x2, y2 = expand_box((x1, y1, x2, y2), image_width, image_height, padding_x, padding_y)

        width = x2 - x1
        height = y2 - y1
        area_ratio = (width * height) / (image_width * image_height)

        if width < min_width or height < min_height:
            continue
        if area_ratio >= max_area_ratio:
            continue
        if y2 <= ignore_top or y1 >= ignore_bottom:
            continue
        if tile_count < 3:
            continue

        regions.append(
            Region(
                page=page_num,
                index=len(regions) + 1,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=0.0,
                reason="component",
            )
        )

    regions = merge_regions(regions, merge_gap_x, merge_gap_y)
    regions = split_large_regions(
        image=image,
        regions=regions,
        image_width=image_width,
        image_height=image_height,
        min_width=min_width,
        min_height=min_height,
    )

    rescored: list[Region] = []
    for region in regions:
        area_ratio = region.area / (image_width * image_height)
        if area_ratio >= max_area_ratio:
            continue
        if looks_like_header(region, image_width, image_height):
            continue

        colored_ratio, grid_ratio, very_colored_ratio, long_run_ratio = graphic_evidence(image, region)
        if looks_like_table_band(region, colored_ratio, grid_ratio, image_width):
            continue

        has_graphics = (
            colored_ratio >= min_colored_ratio
            or grid_ratio >= min_grid_ratio
            or very_colored_ratio >= min_very_colored_ratio
        )
        if not has_graphics:
            continue
        if long_run_ratio < min_graphic_run_ratio:
            continue

        score = region_score(image, region)
        low_color_region = colored_ratio < 0.01 and very_colored_ratio < 0.005
        lower_column_low_color_chart = (
            low_color_region
            and region.y1 > image_height * 0.55
            and region.width < image_width * 0.60
            and region.height >= 180
        )
        if lower_column_low_color_chart:
            required_score = min_score * 0.50
        elif low_color_region and region.width > image_width * 0.72:
            required_score = min_low_color_score
        elif low_color_region:
            required_score = min_score
        else:
            required_score = min_score
        if score < required_score:
            continue

        output_region = region
        if lower_column_low_color_chart and region.height < 280:
            expanded_y1 = max(0, region.y1 - int(region.height * 1.45))
            output_region = Region(
                page=region.page,
                index=region.index,
                x1=region.x1,
                y1=expanded_y1,
                x2=region.x2,
                y2=region.y2,
                score=region.score,
                reason=f"{region.reason}_expand_up",
            )

        rescored.append(
            Region(
                page=output_region.page,
                index=output_region.index,
                x1=output_region.x1,
                y1=output_region.y1,
                x2=output_region.x2,
                y2=output_region.y2,
                score=score,
                reason=output_region.reason,
            )
        )

    rescored.sort(key=lambda item: (item.y1, item.x1))
    rescored = rescored[:max_regions]

    return [
        Region(
            page=region.page,
            index=index,
            x1=region.x1,
            y1=region.y1,
            x2=region.x2,
            y2=region.y2,
            score=region.score,
            reason=region.reason,
        )
        for index, region in enumerate(rescored, start=1)
    ]


def save_debug_overlay(image: Image.Image, regions: list[Region], out_path: Path) -> None:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)

    for region in regions:
        draw.rectangle((region.x1, region.y1, region.x2, region.y2), outline="red", width=4)
        draw.text((region.x1 + 6, region.y1 + 6), f"{region.index}: {region.score}", fill="red")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def crop_page(
    page_path: Path,
    page_num: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[Region]:
    with Image.open(page_path) as image:
        image = image.convert("RGB")
        regions = detect_regions(
            image=image,
            page_num=page_num,
            tile=args.tile,
            ink_threshold=args.ink_threshold,
            color_threshold=args.color_threshold,
            dilate_x=args.dilate_x,
            dilate_y=args.dilate_y,
            padding_x=args.padding_x,
            padding_y=args.padding_y,
            min_width=args.min_width,
            min_height=args.min_height,
            max_area_ratio=args.max_area_ratio,
            ignore_top_ratio=args.ignore_top,
            ignore_bottom_ratio=args.ignore_bottom,
            merge_gap_x=args.merge_gap_x,
            merge_gap_y=args.merge_gap_y,
            max_regions=args.max_regions,
            min_score=args.min_score,
            min_low_color_score=args.min_low_color_score,
            min_colored_ratio=args.min_colored_ratio,
            min_grid_ratio=args.min_grid_ratio,
            min_very_colored_ratio=args.min_very_colored_ratio,
            min_graphic_run_ratio=args.min_graphic_run_ratio,
        )

        for region in regions:
            crop_path = out_dir / f"page_{page_num}_{region.index:02d}_chart.png"
            image.crop((region.x1, region.y1, region.x2, region.y2)).save(crop_path)
            print(f"Saved {crop_path} score={region.score}")

        if args.debug_dir:
            save_debug_overlay(
                image=image,
                regions=regions,
                out_path=args.debug_dir / f"page_{page_num}_debug.png",
            )

    return regions


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast local chart cropper using rendered page images.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path, help="Input PDF. Pages will be rendered first.")
    source.add_argument("--pages-dir", type=Path, help="Directory containing page_*.png files.")

    parser.add_argument("--render-dir", type=Path, default=Path("research/image_extraction/fast_pages"))
    parser.add_argument("--out-dir", type=Path, default=Path("research/image_extraction/fast_outputs"))
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--first-page", type=int, default=None)
    parser.add_argument("--last-page", type=int, default=None)

    parser.add_argument("--tile", type=int, default=18)
    parser.add_argument("--ink-threshold", type=float, default=0.08)
    parser.add_argument("--color-threshold", type=float, default=0.02)
    parser.add_argument("--dilate-x", type=int, default=0)
    parser.add_argument("--dilate-y", type=int, default=0)
    parser.add_argument("--merge-gap-x", type=int, default=0)
    parser.add_argument("--merge-gap-y", type=int, default=0)
    parser.add_argument("--padding-x", type=int, default=16)
    parser.add_argument("--padding-y", type=int, default=14)
    parser.add_argument("--min-width", type=int, default=120)
    parser.add_argument("--min-height", type=int, default=90)
    parser.add_argument("--max-area-ratio", type=float, default=0.25)
    parser.add_argument("--min-score", type=float, default=0.28)
    parser.add_argument("--min-low-color-score", type=float, default=0.25)
    parser.add_argument("--min-colored-ratio", type=float, default=0.13)
    parser.add_argument("--min-grid-ratio", type=float, default=0.018)
    parser.add_argument("--min-very-colored-ratio", type=float, default=0.08)
    parser.add_argument("--min-graphic-run-ratio", type=float, default=0.12)
    parser.add_argument("--ignore-top", type=float, default=0.055)
    parser.add_argument("--ignore-bottom", type=float, default=0.035)
    parser.add_argument("--max-regions", type=int, default=10)

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        page_paths = render_pdf_pages(
            pdf_path=args.pdf,
            pages_dir=args.render_dir,
            dpi=args.dpi,
            first_page=args.first_page,
            last_page=args.last_page,
        )
    else:
        page_paths = page_paths_from_dir(args.pages_dir)

    all_regions: list[Region] = []
    for fallback_index, page_path in enumerate(page_paths, start=1):
        page_num = infer_page_num(page_path, fallback_index)
        all_regions.extend(crop_page(page_path, page_num, args.out_dir, args))

    manifest_path = args.manifest or args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(region) for region in all_regions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Saved {len(all_regions)} crops.")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
