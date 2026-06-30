from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REFUSAL_PATTERNS = [
    "không có thông tin",
    "không được cung cấp",
    "không tìm thấy",
    "không thể",
    "chưa có dữ liệu",
    "not provided",
    "not found",
    "cannot determine",
]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int = 180,
    max_rate_limit_retries: int = 3,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_rate_limit_retries + 1):
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
            if exc.code != 429 or attempt >= max_rate_limit_retries:
                raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc

            try:
                error_payload = json.loads(body)
            except json.JSONDecodeError:
                error_payload = {}

            retry_after = int(error_payload.get("retry_after") or 60)
            wait_seconds = retry_after + 1
            print(
                f"Rate limit reached. Waiting {wait_seconds}s "
                f"before retry {attempt + 1}/{max_rate_limit_retries}..."
            )
            time.sleep(wait_seconds)
        except URLError as exc:
            raise RuntimeError(f"Cannot connect to {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Request to {url} timed out after {timeout}s"
            ) from exc

    raise RuntimeError(f"Rate-limit retries exhausted for {url}")


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


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?%?", normalize_text(text))


def contains_number(text: str, expected_number: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_number_text = normalize_number(text)
    return any(
        variant in normalized_text or variant in normalized_number_text
        for variant in number_variants(expected_number)
    )


def contains_term(text: str, expected_term: str) -> bool:
    if any(character.isdigit() for character in expected_term):
        return contains_number(text, expected_term)
    return normalize_text(expected_term) in normalize_text(text)


def get_expected_terms(case: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    # Prefer the final gold answer. Evidence often contains page numbers
    # ("trang 4") that should not be treated as required answer values.
    for number in extract_numbers(str(case.get("ground_truth_answer") or "")):
        if number not in terms:
            terms.append(number)

    for evidence in case.get("expected_evidence") or []:
        for term in evidence.get("required_terms") or []:
            if term not in terms:
                terms.append(term)

    if terms:
        return terms

    for number in extract_numbers(str(case.get("evidence") or "")):
        if number not in terms:
            terms.append(number)

    if terms:
        return terms

    # Fallback for non-numeric answers: use meaningful answer words.
    stopwords = {
        "và",
        "là",
        "có",
        "theo",
        "năm",
        "trong",
        "của",
        "với",
        "được",
        "bao",
        "nhiêu",
    }
    words = re.findall(r"[\wÀ-ỹ]+", normalize_text(case.get("ground_truth_answer")))
    return [word for word in words if len(word) >= 4 and word not in stopwords][:5]


def source_page(source: dict[str, Any]) -> int | None:
    try:
        return int(source.get("page"))
    except (TypeError, ValueError):
        return None


def source_type(source: dict[str, Any]) -> str:
    value = normalize_text(source.get("type"))
    if value in {"chart", "figure", "image_caption"}:
        return "image"
    return value


def source_evidence_hit(
    case: dict[str, Any],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_pages = {int(page) for page in case.get("expected_pages") or []}
    expected_types = {
        "image" if normalize_text(chunk_type) in {"chart", "figure"} else normalize_text(chunk_type)
        for chunk_type in case.get("expected_chunk_types") or []
    }
    source_pages = {page for page in (source_page(source) for source in sources) if page}
    source_types = {source_type(source) for source in sources}

    page_hit = not expected_pages or bool(expected_pages & source_pages)
    type_hit = not expected_types or bool(expected_types & source_types)

    return {
        "page_hit": page_hit,
        "type_hit": type_hit,
        "source_pages": sorted(source_pages),
        "source_types": sorted(source_types),
        "expected_pages": sorted(expected_pages),
        "expected_types": sorted(expected_types),
    }


def answer_has_citation(answer: str, sources: list[dict[str, Any]]) -> bool:
    if re.search(r"\[source\s*#?\d+\]", normalize_text(answer)):
        return True
    return bool(sources)


def is_refusal(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def deterministic_judge(
    case: dict[str, Any],
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    if not case.get("answerable", True):
        refusal = is_refusal(answer)
        question_numbers = set(extract_numbers(case.get("question", "")))
        answer_numbers = set(extract_numbers(answer))
        hallucinated_numbers = bool(answer_numbers - question_numbers)
        passed = refusal and not hallucinated_numbers
        return {
            "mode": "deterministic",
            "passed": passed,
            "scores": {
                "answer_correctness": 1.0 if passed else 0.0,
                "faithfulness": 1.0 if refusal else 0.0,
                "citation_accuracy": 1.0,
                "unanswerable_handling": 1.0 if passed else 0.0,
            },
            "reason": (
                "unanswerable_refusal"
                if passed
                else "unanswerable_not_refused_or_hallucinated"
            ),
            "matched_terms": [],
            "missing_terms": [],
            "source_evidence": source_evidence_hit(case, sources),
        }

    expected_terms = get_expected_terms(case)
    matched_terms = [term for term in expected_terms if contains_term(answer, term)]
    missing_terms = [term for term in expected_terms if term not in matched_terms]
    source_hit = source_evidence_hit(case, sources)
    citation_ok = answer_has_citation(answer, sources)

    if expected_terms:
        correctness = len(matched_terms) / len(expected_terms)
    else:
        correctness = 0.0

    citation_accuracy = 1.0 if citation_ok and source_hit["page_hit"] else 0.0
    faithfulness = 1.0 if citation_ok and source_hit["page_hit"] and source_hit["type_hit"] else 0.0
    passed = correctness >= 0.8 and faithfulness >= 1.0 and citation_accuracy >= 1.0

    if passed:
        reason = "passed"
    elif correctness < 0.8:
        reason = "answer_term_mismatch"
    elif not source_hit["page_hit"]:
        reason = "source_page_mismatch"
    elif not source_hit["type_hit"]:
        reason = "source_type_mismatch"
    else:
        reason = "missing_citation"

    return {
        "mode": "deterministic",
        "passed": passed,
        "scores": {
            "answer_correctness": round(correctness, 4),
            "faithfulness": faithfulness,
            "citation_accuracy": citation_accuracy,
            "unanswerable_handling": 1.0,
        },
        "reason": reason,
        "expected_terms": expected_terms,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "source_evidence": source_hit,
    }


def evaluate_case(
    case: dict[str, Any],
    response: dict[str, Any],
    latency_ms: float,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    answer = response.get("answer") or ""
    sources = response.get("sources") or []
    judge = deterministic_judge(case, answer, sources)

    return {
        "id": case["id"],
        "category": case.get("category"),
        "question": case["question"],
        "answerable": case.get("answerable", True),
        "ground_truth_answer": case.get("ground_truth_answer"),
        "evidence": case.get("evidence"),
        "answer": answer,
        "sources": sources,
        "latency_ms": round(latency_ms, 2),
        "request_payload": request_payload,
        "passed": judge["passed"],
        "judge": judge,
    }


def average(items: list[float]) -> float:
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "answer_correctness": 0.0,
            "faithfulness": 0.0,
            "citation_accuracy": 0.0,
            "unanswerable_handling": 0.0,
            "avg_latency_ms": 0.0,
        }

    total = len(items)
    passed = sum(1 for item in items if item["passed"])

    def score(name: str) -> float:
        return average([float(item["judge"]["scores"][name]) for item in items])

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4),
        "answer_correctness": score("answer_correctness"),
        "faithfulness": score("faithfulness"),
        "citation_accuracy": score("citation_accuracy"),
        "unanswerable_handling": score("unanswerable_handling"),
        "avg_latency_ms": round(
            sum(float(item["latency_ms"]) for item in items) / total,
            2,
        ),
    }


def summarize(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({item.get("category") or "unknown" for item in evaluations})
    failed = [item for item in evaluations if not item["passed"]]
    reasons = sorted({item["judge"]["reason"] for item in failed})

    return {
        **aggregate(evaluations),
        "failure_reasons": {
            reason: sum(1 for item in failed if item["judge"]["reason"] == reason)
            for reason in reasons
        },
        "by_category": {
            category: aggregate(
                [
                    item
                    for item in evaluations
                    if (item.get("category") or "unknown") == category
                ]
            )
            for category in categories
        },
    }


def write_outputs(
    reports_dir: Path,
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    run_id = datetime.now().strftime("generation_%Y%m%d_%H%M%S")
    run_dir = reports_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "cases.jsonl", evaluations)
    write_jsonl(
        run_dir / "failures.jsonl",
        [item for item in evaluations if not item["passed"]],
    )
    return run_dir


def build_payload(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": case["question"],
        "top_k": args.top_k,
        "company": case.get("company"),
        "year": case.get("year"),
        "quarter": case.get("quarter"),
    }
    if args.top_k_text is not None:
        payload["top_k_text"] = args.top_k_text
    if args.top_k_table is not None:
        payload["top_k_table"] = args.top_k_table
    if args.top_k_image is not None:
        payload["top_k_image"] = args.top_k_image
    if args.reranker:
        payload["reranker"] = args.reranker
    if args.reranker_model:
        payload["reranker_model"] = args.reranker_model
    payload["cross_encoder_top_n"] = args.cross_encoder_top_n
    return {key: value for key, value in payload.items() if value is not None}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run answer generation evaluation against Hyena /query API."
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("eval/test_sets/fpt_2025_dev.jsonl"),
    )
    parser.add_argument("--api-base", default="http://localhost:8001/api/v1")
    parser.add_argument("--reports-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-k-text", type=int)
    parser.add_argument("--top-k-table", type=int)
    parser.add_argument("--top-k-image", type=int)
    parser.add_argument(
        "--reranker",
        choices=["vector", "heuristic", "cross_encoder"],
        default="heuristic",
    )
    parser.add_argument("--reranker-model")
    parser.add_argument("--cross-encoder-top-n", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N cases after loading the test set.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help=(
            "Kept for workflow clarity. Current script uses deterministic judging "
            "only, so this flag avoids accidental paid LLM judging later."
        ),
    )
    args = parser.parse_args()

    cases = load_jsonl(args.test_set)
    if args.limit is not None:
        cases = cases[: args.limit]

    query_url = args.api_base.rstrip("/") + "/query/"
    evaluations = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}")
        payload = build_payload(case, args)
        started_at = time.perf_counter()
        response = post_json(query_url, payload, timeout=args.timeout)
        latency_ms = (time.perf_counter() - started_at) * 1000
        evaluations.append(
            evaluate_case(
                case,
                response=response,
                latency_ms=latency_ms,
                request_payload=payload,
            )
        )

    summary = summarize(evaluations)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_set": str(args.test_set),
        "api_base": args.api_base,
        "top_k": args.top_k,
        "top_k_text": args.top_k_text,
        "top_k_table": args.top_k_table,
        "top_k_image": args.top_k_image,
        "reranker": args.reranker,
        "reranker_model": args.reranker_model,
        "cross_encoder_top_n": args.cross_encoder_top_n,
        "timeout": args.timeout,
        "case_count": len(cases),
        "limit": args.limit,
        "judge": "deterministic",
        "skip_judge": args.skip_judge,
        "note": (
            "/query/ currently evaluates production generation behavior. "
            "If backend does not pass modality/reranker fields into RAGEngine, "
            "those fields are recorded but may not affect generation retrieval."
        ),
    }
    run_dir = write_outputs(args.reports_dir, summary, evaluations, config)

    print("\nGeneration evaluation complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRun directory: {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
