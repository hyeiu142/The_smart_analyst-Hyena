from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.ingestion.image_processor import ImageProcessor
from backend.app.core.retrieval.embedder import Embedder
from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper


def scroll_image_points(qdrant: QdrantClientWrapper, doc_id: str) -> list[Any]:
    points, _next_offset = qdrant.client.scroll(
        collection_name=qdrant.IMAGE_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
        limit=200,
        with_payload=True,
        with_vectors=False,
    )
    return list(points)


def build_content(caption_result: dict[str, Any]) -> str:
    return "\n\n".join(
        str(part).strip()
        for part in [
            caption_result.get("caption"),
            caption_result.get("key_data"),
        ]
        if part
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recaption all image chunks for one document and update Qdrant."
    )
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Caption images but do not upsert updated chunks.",
    )
    args = parser.parse_args()

    qdrant = QdrantClientWrapper()
    image_processor = ImageProcessor()
    embedder = Embedder()
    points = scroll_image_points(qdrant, args.doc_id)
    if args.limit is not None:
        points = points[: args.limit]

    print(f"Found {len(points)} image chunks for doc_id={args.doc_id}")
    updated_count = 0
    skipped_count = 0

    for index, point in enumerate(points, start=1):
        payload = point.payload or {}
        metadata = dict(payload.get("metadata") or {})
        image_path = metadata.get("local_image_path") or metadata.get("image_path")
        if image_path and image_path.startswith("/uploads/"):
            image_path = "/project" + image_path

        print(f"[{index}/{len(points)}] point_id={point.id} path={image_path}")
        if not image_path or not Path(image_path).exists():
            print("  skipped: image file not found")
            skipped_count += 1
            continue

        caption_result = image_processor._caption_image_bytes(Path(image_path).read_bytes())
        content = build_content(caption_result)
        if not content:
            print("  skipped: caption result is empty/non-chart")
            skipped_count += 1
            continue

        metadata["chart_type"] = caption_result.get("chart_type") or metadata.get("chart_type")
        metadata["image_status"] = "described"

        if not args.dry_run:
            qdrant.upsert_chunks(
                qdrant.IMAGE_COLLECTION,
                [
                    {
                        "id": str(point.id),
                        "vector": embedder.embed_documents(content),
                        "payload": {
                            "content": content,
                            "metadata": metadata,
                        },
                    }
                ],
            )

        print(f"  updated: {content[:180].replace(chr(10), ' ')}")
        updated_count += 1

    print(
        f"Done. updated={updated_count}, skipped={skipped_count}, dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
