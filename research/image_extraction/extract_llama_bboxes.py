import argparse
import json
import os
from pathlib import Path


def load_llama_parse_class():
    try:
        from llama_cloud_services import LlamaParse

        return LlamaParse
    except ImportError:
        from llama_parse import LlamaParse

        return LlamaParse


def extract_llama_json(pdf_path: Path, api_key: str) -> dict:
    LlamaParse = load_llama_parse_class()

    parser = LlamaParse(
        api_key=api_key,
        result_type="json",
        verbose=True,
    )

    if hasattr(parser, "get_json_result"):
        result = parser.get_json_result(str(pdf_path))
    else:
        result = parser.load_data(str(pdf_path))

    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict):
            return result[0]

        return {"documents": result}

    if isinstance(result, dict):
        return result

    return {"raw": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LlamaParse JSON with bbox data.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("llama_parse.json"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLAMA_CLOUD_API_KEY"),
    )

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set LLAMA_CLOUD_API_KEY or pass --api-key.")

    data = extract_llama_json(args.pdf, args.api_key)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pages = data.get("pages", [])
    print(f"Wrote LlamaParse JSON to {args.out_json}")
    print(f"Pages found: {len(pages)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any


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


def extract_llama_json(pdf_path: Path, api_key: str) -> dict[str, Any]:
    LlamaParse = load_llama_parse_class()

    parser = LlamaParse(
        api_key=api_key,
        result_type="json",
        verbose=True,
    )

    if hasattr(parser, "get_json_result"):
        result = parser.get_json_result(str(pdf_path))
    else:
        result = parser.load_data(str(pdf_path))

    return normalize_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LlamaParse JSON with bbox data.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out-json", type=Path, default=Path("llama_parse.json"))
    parser.add_argument("--api-key", default=os.getenv("LLAMA_CLOUD_API_KEY"))

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set LLAMA_CLOUD_API_KEY or pass --api-key.")

    data = extract_llama_json(args.pdf, args.api_key)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote LlamaParse JSON to {args.out_json}")
    print(f"Pages found: {len(data.get('pages', []))}")


if __name__ == "__main__":
    main()
