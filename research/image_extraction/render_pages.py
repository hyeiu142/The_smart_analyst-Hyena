#!/usr/bin/env python3
"""
Render PDF pages to PNG images.

Responsibility:
    PyMuPDF renders PDF pages. This script does not detect charts and does not
    crop anything.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import fitz


def render_pages(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 160,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    start = max(1, first_page or 1)
    end = min(len(doc), last_page or len(doc))

    if start > end:
        raise ValueError(f"Invalid page range: first_page={start}, last_page={end}")

    manifest: list[dict[str, Any]] = []

    for page_num in range(start, end + 1):
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image_path = out_dir / f"page_{page_num}.png"
        pix.save(image_path)

        manifest.append(
            {
                "page": page_num,
                "image_path": str(image_path),
                "pdf_width": page.rect.width,
                "pdf_height": page.rect.height,
                "image_width": pix.width,
                "image_height": pix.height,
                "dpi": dpi,
            }
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PDF pages to PNG images.")
    parser.add_argument("pdf", type=Path, help="Input PDF path.")
    parser.add_argument("--out-dir", type=Path, default=Path("pages"))
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--first-page", type=int, default=None)
    parser.add_argument("--last-page", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)

    args = parser.parse_args()

    manifest = render_pages(
        pdf_path=args.pdf,
        out_dir=args.out_dir,
        dpi=args.dpi,
        first_page=args.first_page,
        last_page=args.last_page,
    )

    manifest_path = args.manifest or args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Rendered {len(manifest)} pages to {args.out_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
