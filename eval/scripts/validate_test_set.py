from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_CATEGORY_COUNTS = {
    "text": 25,
    "table": 25,
    "image": 25,
    "mixed": 15,
    "unanswerable": 10,
}

EXPECTED_DEV_COUNTS = {
    "text": 18,
    "table": 18,
    "image": 18,
    "mixed": 10,
    "unanswerable": 6,
}

EXPECTED_TEST_COUNTS = {
    "text": 7,
    "table": 7,
    "image": 7,
    "mixed": 5,
    "unanswerable": 4,
}

REQUIRED_FIELDS = {
    "id",
    "question",
    "ground_truth_answer",
    "evidence",
    "expected_pages",
    "expected_chunk_types",
    "company",
    "year",
    "category",
    "difficulty",
    "answerable",
}


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    if not path.exists():
        return rows, [f"File not found: {path}"]

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                continue

            if not isinstance(row, dict):
                errors.append(f"{path}:{line_number}: each row must be a JSON object")
                continue

            rows.append(row)

    return rows, errors


def validate_case(case: dict[str, Any], location: str) -> list[str]:
    errors: list[str] = []
    missing_fields = sorted(REQUIRED_FIELDS - case.keys())
    if missing_fields:
        errors.append(f"{location}: missing fields: {', '.join(missing_fields)}")

    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append(f"{location}: id must be a non-empty string")

    for field in ("question", "ground_truth_answer", "evidence", "company"):
        value = case.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{location}: {field} must be a non-empty string")

    category = case.get("category")
    if category not in EXPECTED_CATEGORY_COUNTS:
        errors.append(f"{location}: invalid category: {category!r}")

    if not isinstance(case.get("answerable"), bool):
        errors.append(f"{location}: answerable must be a boolean")

    pages = case.get("expected_pages")
    if not isinstance(pages, list) or any(
        not isinstance(page, int) or isinstance(page, bool) or page < 1
        for page in pages
    ):
        errors.append(f"{location}: expected_pages must contain positive integers")

    chunk_types = case.get("expected_chunk_types")
    if not isinstance(chunk_types, list) or any(
        not isinstance(chunk_type, str) or not chunk_type.strip()
        for chunk_type in chunk_types
    ):
        errors.append(
            f"{location}: expected_chunk_types must contain non-empty strings"
        )

    expected_evidence = case.get("expected_evidence")
    if expected_evidence is not None:
        if not isinstance(expected_evidence, list):
            errors.append(f"{location}: expected_evidence must be a list")
        else:
            if case.get("answerable") is True and not expected_evidence:
                errors.append(
                    f"{location}: answerable case must have expected_evidence"
                )
            if case.get("answerable") is False and expected_evidence:
                errors.append(
                    f"{location}: unanswerable case must have empty expected_evidence"
                )

            for index, item in enumerate(expected_evidence, start=1):
                item_location = f"{location}:expected_evidence[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{item_location}: must be an object")
                    continue

                evidence_pages = item.get("pages")
                if not isinstance(evidence_pages, list) or any(
                    not isinstance(page, int)
                    or isinstance(page, bool)
                    or page < 1
                    for page in evidence_pages
                ):
                    errors.append(
                        f"{item_location}: pages must contain positive integers"
                    )

                evidence_types = item.get("chunk_types")
                if not isinstance(evidence_types, list) or any(
                    not isinstance(chunk_type, str) or not chunk_type.strip()
                    for chunk_type in evidence_types
                ):
                    errors.append(
                        f"{item_location}: chunk_types must contain non-empty strings"
                    )
                elif "image" in evidence_types and len(evidence_types) > 1:
                    errors.append(
                        f"{item_location}: image evidence must be a separate unit"
                    )

                required_terms = item.get("required_terms")
                if not isinstance(required_terms, list) or any(
                    not isinstance(term, str) or not term.strip()
                    for term in required_terms
                ):
                    errors.append(
                        f"{item_location}: required_terms must contain non-empty strings"
                    )
                elif evidence_types and "image" not in evidence_types and not required_terms:
                    errors.append(
                        f"{item_location}: non-image evidence needs required_terms"
                    )

    return errors


def duplicate_ids(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(row.get("id") for row in rows)
    return sorted(
        case_id
        for case_id, count in counts.items()
        if isinstance(case_id, str) and count > 1
    )


def category_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("category")) for row in rows)


def validate_expected_counts(
    label: str,
    rows: list[dict[str, Any]],
    expected_total: int,
    expected_categories: dict[str, int],
) -> list[str]:
    errors: list[str] = []
    if len(rows) != expected_total:
        errors.append(f"{label}: expected {expected_total} cases, found {len(rows)}")

    actual_categories = category_counts(rows)
    for category, expected_count in expected_categories.items():
        actual_count = actual_categories.get(category, 0)
        if actual_count != expected_count:
            errors.append(
                f"{label}: category {category!r} expected "
                f"{expected_count}, found {actual_count}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Hyena evaluation dataset and dev/test split."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("eval/test_sets/fpt_2025_qa_100.jsonl"),
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("eval/test_sets/fpt_2025_dev.jsonl"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("eval/test_sets/fpt_2025_test.jsonl"),
    )
    args = parser.parse_args()

    source, source_load_errors = load_jsonl(args.source)
    dev, dev_load_errors = load_jsonl(args.dev)
    test, test_load_errors = load_jsonl(args.test)
    errors = source_load_errors + dev_load_errors + test_load_errors

    for label, rows in (("source", source), ("dev", dev), ("test", test)):
        for index, case in enumerate(rows, start=1):
            errors.extend(validate_case(case, f"{label}:row {index}"))

        duplicates = duplicate_ids(rows)
        if duplicates:
            errors.append(f"{label}: duplicate IDs: {', '.join(duplicates)}")

    errors.extend(
        validate_expected_counts(
            "source",
            source,
            expected_total=100,
            expected_categories=EXPECTED_CATEGORY_COUNTS,
        )
    )
    errors.extend(
        validate_expected_counts(
            "dev",
            dev,
            expected_total=70,
            expected_categories=EXPECTED_DEV_COUNTS,
        )
    )
    errors.extend(
        validate_expected_counts(
            "test",
            test,
            expected_total=30,
            expected_categories=EXPECTED_TEST_COUNTS,
        )
    )

    source_ids = {row.get("id") for row in source}
    dev_ids = {row.get("id") for row in dev}
    test_ids = {row.get("id") for row in test}

    overlap = sorted(
        case_id
        for case_id in dev_ids & test_ids
        if isinstance(case_id, str)
    )
    if overlap:
        errors.append(f"dev/test overlap: {', '.join(overlap)}")

    combined_ids = dev_ids | test_ids
    missing_from_split = sorted(
        case_id
        for case_id in source_ids - combined_ids
        if isinstance(case_id, str)
    )
    extra_in_split = sorted(
        case_id
        for case_id in combined_ids - source_ids
        if isinstance(case_id, str)
    )
    if missing_from_split:
        errors.append(f"source IDs missing from split: {', '.join(missing_from_split)}")
    if extra_in_split:
        errors.append(f"split IDs missing from source: {', '.join(extra_in_split)}")

    missing_expected_evidence = [
        str(case.get("id"))
        for case in source
        if "expected_evidence" not in case
    ]

    print(f"Dataset: {len(source)} cases")
    print(f"Dev: {len(dev)}")
    print(f"Test: {len(test)}")
    print(f"Duplicate source IDs: {len(duplicate_ids(source))}")
    print(f"Dev/test overlap: {len(overlap)}")
    print(f"Invalid cases: {len(errors)}")
    print(f"Missing expected_evidence: {len(missing_expected_evidence)}")
    print("Category:")
    counts = category_counts(source)
    for category in EXPECTED_CATEGORY_COUNTS:
        print(f"  {category}: {counts.get(category, 0)}")

    if missing_expected_evidence:
        print("\nCases missing expected_evidence:")
        for case_id in missing_expected_evidence:
            print(f"  {case_id}")

    if errors:
        print("\nValidation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
