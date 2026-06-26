#!/usr/bin/env python3
"""
Describe YOLO crop images into structured chart/table metadata.

Input:
    manifest.json from doclayout_yolo_extract.py

Output:
    chart_metadata.jsonl, one JSON object per crop

Run from project root.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


CHART_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "visual_type": {
            "type": "string",
            "enum": ["chart", "table", "figure", "other", "unreadable"],
        },
        "chart_type": {
            "type": "string",
            "enum": [
                "bar",
                "line",
                "pie",
                "donut",
                "stacked_bar",
                "combo",
                "table",
                "diagram",
                "other",
                "unknown",
            ],
        },
        "title": {"type": "string"},
        "unit": {"type": "string"},
        "x_axis": {
            "type": "array",
            "items": {"type": "string"},
        },
        "series": {
            "type": "array",
            "items": {"type": "string"},
        },
        "data_points": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "period": {"type": "string"},
                    "series": {"type": "string"},
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["period", "series", "label", "value", "unit"],
            },
        },
        "key_facts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "caption_structured": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "visual_type",
        "chart_type",
        "title",
        "unit",
        "x_axis",
        "series",
        "data_points",
        "key_facts",
        "caption_structured",
        "confidence",
        "warnings",
    ],
}


PROMPT = """
You are extracting structured data from a cropped chart/table image in a Vietnamese financial report.

Return only facts visible in the image. Do not invent missing values.

Focus on:
- Vietnamese chart/table title.
- Chart type.
- Units.
- Legend/series names.
- Years or categories.
- Explicit numeric values visible in the image.
- Key facts useful for financial question answering.

Rules:
- Preserve Vietnamese labels when visible.
- Preserve number formatting as seen, but use comma decimal style in the Vietnamese caption if natural.
- Add data_points only for numbers directly printed in the crop, such as table cells,
  bar labels, pie/donut labels, callouts, or visible point labels.
- Do not estimate values from axes, line positions, bar heights, or visual proportions.
- Do not interpolate missing values.
- For line charts without printed point labels, keep data_points empty or include only
  explicitly labeled values. Summarize visible trends in key_facts instead.
- If a value is not directly visible, omit it from data_points and mention it in warnings.
- caption_structured must be Vietnamese and searchable. Include title, page context if possible, and the most important data points.
"""


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_pages(value: str | None) -> set[int] | None:
    if not value:
        return None

    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))

    return pages


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def iter_crop_rows(
    manifest: list[dict[str, Any]],
    pages: set[int] | None,
    labels: set[str],
) -> list[dict[str, Any]]:
    rows = []

    for item in manifest:
        crop_path = item.get("crop_path")
        label = str(item.get("label") or "").lower().strip()
        page = item.get("page")

        if not crop_path:
            continue
        if labels and label not in labels:
            continue
        if pages is not None and int(page) not in pages:
            continue

        rows.append(item)

    return rows


def describe_crop(
    client: OpenAI,
    model: str,
    crop_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "chart_metadata",
                "strict": True,
                "schema": CHART_METADATA_SCHEMA,
            },
        },
        messages=[
            {
                "role": "system",
                "content": PROMPT.strip(),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract structured metadata from this crop.\n"
                            f"Context: {json.dumps(context, ensure_ascii=False)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(crop_path),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def build_output_row(
    document: str,
    manifest_row: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    crop_path = manifest_row.get("crop_path")

    return {
        "document": document,
        "page": manifest_row.get("page"),
        "index": manifest_row.get("index"),
        "label": manifest_row.get("label"),
        "class_id": manifest_row.get("class_id"),
        "confidence_detection": manifest_row.get("confidence"),
        "bbox": [
            manifest_row.get("x1"),
            manifest_row.get("y1"),
            manifest_row.get("x2"),
            manifest_row.get("y2"),
        ],
        "crop_path": crop_path,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe image crops into structured JSONL metadata.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/image_extraction/test_outputs/FPT_2025_7_titlepad/manifest.json"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--document", default="FPT_2025_7.pdf")
    parser.add_argument("--model", default=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"))
    parser.add_argument("--pages", default=None, help="Examples: 4 or 1,4,5 or 1-4")
    parser.add_argument("--labels", default="figure,picture")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    out_path = args.out or args.manifest.parent / "chart_metadata.jsonl"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite to replace it.")

    pages = parse_pages(args.pages)
    labels = {item.strip().lower() for item in args.labels.split(",") if item.strip()}

    manifest = load_manifest(args.manifest)
    crop_rows = iter_crop_rows(manifest, pages=pages, labels=labels)
    if args.limit is not None:
        crop_rows = crop_rows[: args.limit]

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running describe_crops.py, "
            "or run the script in the same environment as the backend."
        )

    client = OpenAI()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        for position, row in enumerate(crop_rows, start=1):
            crop_path = Path(row["crop_path"])
            if not crop_path.exists():
                result = {
                    "document": args.document,
                    "page": row.get("page"),
                    "index": row.get("index"),
                    "label": row.get("label"),
                    "crop_path": str(crop_path),
                    "error": f"Crop not found: {crop_path}",
                }
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[{position}/{len(crop_rows)}] missing {crop_path}")
                continue

            context = {
                "document": args.document,
                "page": row.get("page"),
                "index": row.get("index"),
                "label": row.get("label"),
                "bbox": [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")],
                "crop_path": str(crop_path),
            }

            print(f"[{position}/{len(crop_rows)}] describing {crop_path}")
            try:
                metadata = describe_crop(
                    client=client,
                    model=args.model,
                    crop_path=crop_path,
                    context=context,
                )
                result = build_output_row(args.document, row, metadata)
            except Exception as exc:
                result = {
                    "document": args.document,
                    "page": row.get("page"),
                    "index": row.get("index"),
                    "label": row.get("label"),
                    "crop_path": str(crop_path),
                    "error": str(exc),
                }

            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote {len(crop_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
