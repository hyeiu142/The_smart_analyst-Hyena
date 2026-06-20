from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to {url}: {exc.reason}") from exc


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_number(value: str) -> str:
    value = normalize_text(value)
    value = value.replace("%", "")
    value = value.replace(",", ".")
    return value.strip()


def number_variants(value: str) -> set[str]:
    raw = normalize_text(value)
    normalized = normalize_number(value)
    variants = {raw, normalized}

    if "." in normalized:
        variants.add(normalized.replace(".", ","))
        variants.add(normalized.replace(".", ""))
    if "," in raw:
        variants.add(raw.replace(",", "."))
    if "." in raw:
        variants.add(raw.replace(".", ","))

    return {variant for variant in variants if variant}


def get_chunk_page(chunk: dict[str, Any]) -> int | None:
    metadata = chunk.get("metadata") or {}
    page = metadata.get("page")
    if page is None:
        page = metadata.get("page_num")
    if page is None:
        return None

    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def get_chunk_type(chunk: dict[str, Any]) -> str | None:
    metadata = chunk.get("metadata") or {}
    source_collection = chunk.get("source_collection")
    chunk_type = metadata.get("chunk_type")

    if source_collection:
        return normalize_text(source_collection)
    if chunk_type == "image_caption":
        return "image"
    return normalize_text(chunk_type)


def contains_expected_number(content: str, expected_number: str) -> bool:
    normalized_content = normalize_text(content)
    normalized_content_number = normalize_number(content)

    for variant in number_variants(expected_number):
        if variant in normalized_content or variant in normalized_content_number:
            return True
    return False


def evaluate_case(case: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_pages = set(case.get("expected_pages") or [])
    expected_types = set(case.get("expected_chunk_types") or [])
    expected_numbers = case.get("expected_numbers") or []
    expected_image_hint = case.get("expected_image_hint")

    hit_pages = []
    hit_types = []
    hit_numbers = []
    hit_image_hint = False

    for rank, chunk in enumerate(results, start=1):
        content = chunk.get("content") or ""
        page = get_chunk_page(chunk)
        chunk_type = get_chunk_type(chunk)

        if page in expected_pages:
            hit_pages.append({"rank": rank, "page": page})
        if chunk_type in expected_types:
            hit_types.append({"rank": rank, "type": chunk_type})

        for number in expected_numbers:
            if contains_expected_number(content, number):
                hit_numbers.append({"rank": rank, "number": number})

        if expected_image_hint and normalize_text(expected_image_hint) in normalize_text(content):
            hit_image_hint = True

    found_numbers = {item["number"] for item in hit_numbers}
    page_hit = not expected_pages or bool(hit_pages)
    type_hit = not expected_types or bool(hit_types)
    number_hit = all(number in found_numbers for number in expected_numbers)
    image_hint_hit = expected_image_hint is None or hit_image_hint
    passed = page_hit and type_hit and number_hit and image_hint_hit

    return {
        "id": case["id"],
        "category": case.get("category"),
        "question": case["question"],
        "passed": passed,
        "checks": {
            "page_hit": page_hit,
            "type_hit": type_hit,
            "number_hit": number_hit,
            "image_hint_hit": image_hint_hit,
        },
        "expected": {
            "pages": sorted(expected_pages),
            "chunk_types": sorted(expected_types),
            "numbers": expected_numbers,
            "image_hint": expected_image_hint,
        },
        "hits": {
            "pages": hit_pages,
            "types": hit_types,
            "numbers": hit_numbers,
            "image_hint": hit_image_hint,
        },
        "top_results": [
            {
                "rank": index + 1,
                "score": chunk.get("score"),
                "source_collection": chunk.get("source_collection"),
                "chunk_type": get_chunk_type(chunk),
                "page": get_chunk_page(chunk),
                "preview": normalize_text(chunk.get("content", ""))[:250],
            }
            for index, chunk in enumerate(results)
        ],
    }


def summarize(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(evaluations)
    passed = sum(1 for item in evaluations if item["passed"])

    by_category: dict[str, dict[str, Any]] = {}
    for item in evaluations:
        category = item.get("category") or "unknown"
        bucket = by_category.setdefault(category, {"total": 0, "passed": 0, "pass_rate": 0.0})
        bucket["total"] += 1
        if item["passed"]:
            bucket["passed"] += 1

    for bucket in by_category.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0.0

    check_rates = {}
    for check_name in ["page_hit", "type_hit", "number_hit", "image_hint_hit"]:
        count = sum(1 for item in evaluations if item["checks"][check_name])
        check_rates[check_name] = round(count / total, 4) if total else 0.0

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_category": by_category,
        "check_rates": check_rates,
    }


def write_outputs(
    report_dir: Path,
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = report_dir / f"retrieval_eval_{timestamp}.json"
    failures_path = report_dir / f"retrieval_failures_{timestamp}.jsonl"
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "results": evaluations,
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with failures_path.open("w", encoding="utf-8") as file:
        for item in evaluations:
            if not item["passed"]:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

    return report_path, failures_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against Hyena /query/similar API.")
    parser.add_argument("--test-set", type=Path, default=Path("eval/test_sets/fpt_2025_qa.jsonl"))
    parser.add_argument("--api-base", default="http://localhost:8001/api/v1")
    parser.add_argument("--reports-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    cases = load_jsonl(args.test_set)
    similar_url = args.api_base.rstrip("/") + "/query/similar"

    evaluations = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}")
        payload = {
            "question": case["question"],
            "top_k": args.top_k,
            "company": case.get("company"),
            "year": case.get("year"),
            "quarter": case.get("quarter"),
        }

        response = post_json(similar_url, payload)
        results = response.get("results") or []
        evaluations.append(evaluate_case(case, results))

    summary = summarize(evaluations)
    report_path, failures_path = write_outputs(args.reports_dir, summary, evaluations)

    print("\nRetrieval evaluation complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")
    print(f"Failures: {failures_path}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
