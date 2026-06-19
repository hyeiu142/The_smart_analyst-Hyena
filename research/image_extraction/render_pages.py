#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import fitz


def render_pages(pdf_path: Path, out_dir: Path, dpi: int, first_page: int | None, last_page: int | None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    start = 1 if first_page is None else max(1, first_page)
    end = len(doc) if last_page is None else min(len(doc), last_page)

    manifest = []

    for page_num in range(start, end + 1):
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out_path = out_dir / f"page_{page_num}.png"
        pix.save(out_path)

        manifest.append(
            {
                "page": page_num,
                "image_path": str(out_path),
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
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("pages"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--first-page", type=int, default=None)
    parser.add_argument("--last-page", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)

    args = parser.parse_args()

    manifest = render_pages(args.pdf, args.out_dir, args.dpi, args.first_page, args.last_page)
    manifest_path = args.manifest or args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rendered {len(manifest)} pages to {args.out_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
