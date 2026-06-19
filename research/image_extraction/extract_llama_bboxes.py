#!/usr/bin/env python3
"""
Extract LlamaParse layout JSON for image/chart bounding boxes.

Responsibility:
    LlamaParse detects page layout and bbox metadata only. This script does not
    render pages and does not crop final images.
"""

import argparse
import asyncio
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_api_key(explicit_key: str | None = None) -> str:
    if explicit_key:
        return explicit_key

    load_dotenv(PROJECT_ROOT / ".env")

    env_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if env_key:
        return env_key

    try:
        from backend.app.config import get_settings

        return get_settings().llama_cloud_api_key
    except Exception:
        return ""


def load_llama_parse_class():
    try:
        from llama_cloud_services import LlamaParse

        return LlamaParse
    except ImportError:
        from llama_parse import LlamaParse

        return LlamaParse


def normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result

    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict):
            return result[0]
        return {"documents": result}

    return {"raw": result}


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def extract_llama_json(pdf_path: Path, api_key: str) -> dict[str, Any]:
    LlamaParse = load_llama_parse_class()
    parser = LlamaParse(
        api_key=api_key,
        result_type="json",
        language="vi",
        verbose=True,
        extract_images=True,
    )

    if hasattr(parser, "aget_json"):
        result = await maybe_await(parser.aget_json(str(pdf_path)))
    elif hasattr(parser, "get_json_result"):
        result = await maybe_await(parser.get_json_result(str(pdf_path)))
    elif hasattr(parser, "load_data"):
        result = await maybe_await(parser.load_data(str(pdf_path)))
    else:
        raise RuntimeError("Unsupported LlamaParse client: no JSON extraction method found.")

    return normalize_result(result)


def count_pages(data: dict[str, Any]) -> int:
    if isinstance(data.get("pages"), list):
        return len(data["pages"])

    documents = data.get("documents")
    if isinstance(documents, list):
        return sum(
            len(document.get("pages", []))
            for document in documents
            if isinstance(document, dict)
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LlamaParse JSON with page/image bbox metadata.")
    parser.add_argument("pdf", type=Path, help="Input PDF path.")
    parser.add_argument("--out-json", type=Path, default=Path("llama_parse.json"))
    parser.add_argument("--api-key", default=None, help="Optional override. Defaults to LLAMA_CLOUD_API_KEY from .env.")

    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    if not api_key:
        raise SystemExit("Missing LLAMA_CLOUD_API_KEY. Add it to .env or pass --api-key.")

    data = asyncio.run(extract_llama_json(args.pdf, api_key))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote LlamaParse JSON to {args.out_json}")
    print(f"Pages found: {count_pages(data)}")


if __name__ == "__main__":
    main()
