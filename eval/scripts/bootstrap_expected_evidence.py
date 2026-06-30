from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NUMBER_PATTERN = re.compile(
    r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?:\s*%|\s*(?:tỷ|triệu|nghìn)\s*(?:đồng|usd|vnd)?|\s*lần)?",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"^(?:19|20)\d{2}$")

EXPECTED_EVIDENCE_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "fpt_text_013": [
        {
            "pages": [2],
            "chunk_types": ["text"],
            "required_terms": [
                "hạ tầng",
                "ai",
                "phần mềm",
                "phần cứng",
                "rủi ro về thuế",
            ],
        }
    ],
    "fpt_text_024": [
        {
            "pages": [3],
            "chunk_types": ["text"],
            "required_terms": ["ngoại hạng anh", "2025-2026", "2030-2031"],
        }
    ],
    "fpt_mixed_001": [
        {"pages": [3], "chunk_types": ["table"], "required_terms": ["44.475"]},
        {"pages": [4], "chunk_types": ["image"], "required_terms": []},
    ],
    "fpt_mixed_002": [
        {"pages": [3], "chunk_types": ["table"], "required_terms": ["19.508"]},
        {"pages": [4], "chunk_types": ["image"], "required_terms": []},
    ],
    "fpt_mixed_003": [
        {"pages": [3], "chunk_types": ["table"], "required_terms": ["6.132"]},
        {"pages": [1], "chunk_types": ["text"], "required_terms": ["24%"]},
    ],
    "fpt_mixed_004": [
        {"pages": [3], "chunk_types": ["table"], "required_terms": ["13.039"]},
        {"pages": [4], "chunk_types": ["image"], "required_terms": []},
    ],
    "fpt_mixed_005": [
        {"pages": [4], "chunk_types": ["image"], "required_terms": []},
        {
            "pages": [3],
            "chunk_types": ["text"],
            "required_terms": ["12", "26"],
        },
    ],
    "fpt_mixed_006": [
        {"pages": [2], "chunk_types": ["text"], "required_terms": ["198"]},
        {"pages": [1], "chunk_types": ["text"], "required_terms": ["70.113"]},
    ],
    "fpt_mixed_007": [
        {"pages": [2], "chunk_types": ["text"], "required_terms": ["26%"]},
        {"pages": [3], "chunk_types": ["table"], "required_terms": ["11,6%"]},
    ],
    "fpt_mixed_008": [
        {"pages": [3], "chunk_types": ["text"], "required_terms": ["0,8%"]},
        {"pages": [4], "chunk_types": ["image"], "required_terms": []},
    ],
    "fpt_mixed_009": [
        {"pages": [3], "chunk_types": ["text"], "required_terms": ["25,4%"]},
        {"pages": [4], "chunk_types": ["image"], "required_terms": []},
    ],
    "fpt_mixed_010": [
        {"pages": [3], "chunk_types": ["table"], "required_terms": ["16,0%"]},
        {"pages": [5], "chunk_types": ["table"], "required_terms": ["16.0%"]},
    ],
    "fpt_mixed_011": [
        {"pages": [5], "chunk_types": ["table"], "required_terms": ["5.211"]},
        {
            "pages": [3],
            "chunk_types": ["table"],
            "required_terms": ["5.211", "21,4%"],
        },
    ],
    "fpt_mixed_012": [
        {"pages": [5], "chunk_types": ["table"], "required_terms": ["14.912"]},
        {"pages": [3], "chunk_types": ["table"], "required_terms": ["14.911"]},
    ],
    "fpt_mixed_013": [
        {"pages": [1], "chunk_types": ["text"], "required_terms": ["11.226"]},
        {"pages": [5], "chunk_types": ["table"], "required_terms": ["11.226"]},
    ],
    "fpt_mixed_014": [
        {"pages": [1], "chunk_types": ["text"], "required_terms": ["102.100"]},
        {"pages": [1], "chunk_types": ["table"], "required_terms": ["19,6"]},
    ],
    "fpt_mixed_015": [
        {
            "pages": [5],
            "chunk_types": ["table"],
            "required_terms": ["10.189", "11.660"],
        }
    ],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: expected an object"
                )
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_term(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    value = re.sub(r"\s+%", "%", value)
    return value


def extract_numeric_terms(answer: str) -> list[str]:
    terms: list[str] = []
    for match in NUMBER_PATTERN.finditer(answer):
        term = normalize_term(match.group(0))
        number_match = re.match(r"[-+]?\d+(?:[.,]\d+)*", term)
        if not number_match:
            continue
        number = number_match.group(0)
        term = f"{number}%" if "%" in term else number
        bare_number = re.sub(r"[^\d]", "", term)

        # A standalone year is usually too broad to identify the right evidence.
        if YEAR_PATTERN.fullmatch(term):
            continue
        if not bare_number or term in terms:
            continue
        terms.append(term)
    return terms


def normalize_chunk_types(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        chunk_type = str(value).strip().lower()
        if not chunk_type:
            continue
        if chunk_type == "chart":
            chunk_type = "image"
        if chunk_type not in normalized:
            normalized.append(chunk_type)
    return normalized


def build_suggestion(case: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    case_id = str(case.get("id") or "")
    if case_id in EXPECTED_EVIDENCE_OVERRIDES:
        return EXPECTED_EVIDENCE_OVERRIDES[case_id], []

    if case.get("answerable") is False:
        return [], []

    pages = list(case.get("expected_pages") or [])
    chunk_types = normalize_chunk_types(case.get("expected_chunk_types") or [])
    required_terms = extract_numeric_terms(
        str(case.get("ground_truth_answer") or "")
    )

    reasons: list[str] = []
    category = case.get("category")
    if category == "image":
        required_terms = []
    if not pages:
        reasons.append("missing_pages")
    if not chunk_types:
        reasons.append("missing_chunk_types")
    if category != "image" and not required_terms:
        reasons.append("add_discriminative_required_terms")

    suggestion = [
        {
            "pages": pages,
            "chunk_types": chunk_types,
            "required_terms": required_terms,
        }
    ]
    return suggestion, reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a review copy of the evaluation dataset with suggested "
            "expected_evidence. The source file is never modified."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval/test_sets/fpt_2025_qa_100.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/test_sets/fpt_2025_qa_100_review.jsonl"),
    )
    parser.add_argument(
        "--review-report",
        type=Path,
        default=Path("eval/reports/expected_evidence_review.jsonl"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    for output_path in (args.output, args.review_report):
        if output_path.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}. Use --force if intended."
            )

    rows = load_jsonl(args.input)
    output_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for case in rows:
        output_case = dict(case)
        suggestion, reasons = build_suggestion(case)
        output_case["expected_evidence"] = suggestion
        output_rows.append(output_case)

        if reasons:
            review_rows.append(
                {
                    "id": case.get("id"),
                    "category": case.get("category"),
                    "question": case.get("question"),
                    "ground_truth_answer": case.get("ground_truth_answer"),
                    "evidence": case.get("evidence"),
                    "suggested_expected_evidence": suggestion,
                    "review_reasons": reasons,
                }
            )

    write_jsonl(args.output, output_rows)
    write_jsonl(args.review_report, review_rows)

    print(f"Cases processed: {len(output_rows)}")
    print(f"Cases requiring review: {len(review_rows)}")
    print(f"Review dataset: {args.output}")
    print(f"Review report: {args.review_report}")
    if args.output.resolve() == args.input.resolve():
        print("Source dataset was updated.")
    else:
        print("Source dataset was not modified.")


if __name__ == "__main__":
    main()
