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


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int = 120,
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

    return {v for v in variants if v}


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
    chunk_type = metadata.get("chunk_type")
    source_collection = chunk.get("source_collection")

    if source_collection:
        return normalize_text(source_collection)

    if chunk_type == "image_caption":
        return "image"

    return normalize_text(chunk_type)


def contains_expected_term(content: str, expected_term: str) -> bool:
    normalized_content = normalize_text(content)
    normalized_content_number = normalize_number(content)

    if not any(character.isdigit() for character in expected_term):
        return normalize_text(expected_term) in normalized_content

    for variant in number_variants(expected_term):
        if variant in normalized_content or variant in normalized_content_number:
            return True

    return False


def match_evidence(
    evidence: dict[str, Any],
    chunk: dict[str, Any],
) -> dict[str, Any]:
    pages = set(evidence.get("pages") or [])
    chunk_types = {
        normalize_text(chunk_type)
        for chunk_type in evidence.get("chunk_types") or []
    }
    required_terms = evidence.get("required_terms") or []

    page = get_chunk_page(chunk)
    chunk_type = get_chunk_type(chunk)
    content = chunk.get("content") or ""

    page_match = not pages or page in pages
    type_match = not chunk_types or chunk_type in chunk_types
    matched_terms = [
        term for term in required_terms if contains_expected_term(content, term)
    ]
    terms_match = len(matched_terms) == len(required_terms)

    return {
        "matched": page_match and type_match and terms_match,
        "page_match": page_match,
        "type_match": type_match,
        "terms_match": terms_match,
        "matched_terms": matched_terms,
    }


def classify_missing_evidence(
    evidence: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    pages = set(evidence.get("pages") or [])
    chunk_types = {
        normalize_text(chunk_type)
        for chunk_type in evidence.get("chunk_types") or []
    }
    required_terms = evidence.get("required_terms") or []

    page_matches = []
    type_matches = []
    page_type_matches = []
    partial_term_matches = []

    for rank, chunk in enumerate(results, start=1):
        match = match_evidence(evidence, chunk)
        chunk_summary = {
            "rank": rank,
            "chunk_id": chunk.get("id"),
            "page": get_chunk_page(chunk),
            "chunk_type": get_chunk_type(chunk),
            "matched_terms": match["matched_terms"],
        }

        if match["page_match"]:
            page_matches.append(chunk_summary)
        if match["type_match"]:
            type_matches.append(chunk_summary)
        if match["page_match"] and match["type_match"]:
            page_type_matches.append(chunk_summary)
            if required_terms and match["matched_terms"]:
                partial_term_matches.append(chunk_summary)

    expected_image = "image" in chunk_types
    if page_type_matches and required_terms:
        reason = "evidence_term_mismatch"
    elif page_type_matches:
        reason = "unexpected_unmatched_evidence"
    elif expected_image and type_matches:
        reason = "image_wrong_page"
    elif expected_image:
        reason = "image_not_retrieved"
    elif type_matches and not page_matches:
        reason = "wrong_page"
    elif page_matches and not type_matches:
        reason = "wrong_modality"
    elif page_matches and type_matches:
        reason = "wrong_chunk_alignment"
    else:
        reason = "not_retrieved"

    return {
        "reason": reason,
        "expected_pages": sorted(pages),
        "expected_chunk_types": sorted(chunk_types),
        "required_terms": required_terms,
        "page_match_count": len(page_matches),
        "type_match_count": len(type_matches),
        "page_type_match_count": len(page_type_matches),
        "partial_term_matches": partial_term_matches[:3],
    }


def classify_failure(
    case: dict[str, Any],
    results: list[dict[str, Any]],
    expected_evidence: list[dict[str, Any]],
    evidence_hits: list[dict[str, Any]],
    missing_evidence_indexes: list[int],
) -> dict[str, Any] | None:
    if not missing_evidence_indexes:
        return None

    missing_details = [
        {
            "evidence_index": evidence_index,
            **classify_missing_evidence(
                expected_evidence[evidence_index],
                results,
            ),
        }
        for evidence_index in missing_evidence_indexes
    ]
    missing_reasons = [detail["reason"] for detail in missing_details]
    unique_reasons = sorted(set(missing_reasons))

    if evidence_hits:
        primary_reason = "partial_evidence_retrieved"
    elif len(unique_reasons) == 1:
        primary_reason = unique_reasons[0]
    else:
        primary_reason = "multiple_failure_modes"

    return {
        "reason": primary_reason,
        "category": case.get("category"),
        "matched_evidence_count": len(
            {hit["evidence_index"] for hit in evidence_hits}
        ),
        "missing_evidence_count": len(missing_evidence_indexes),
        "missing_details": missing_details,
    }


def evaluate_case(
    case: dict[str, Any],
    results: list[dict[str, Any]],
    latency_ms: float,
    retrieval_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_evidence = case.get("expected_evidence") or []
    if not case.get("answerable", True):
        return {
            "id": case["id"],
            "category": case.get("category"),
            "question": case["question"],
            "answerable": False,
            "excluded_from_retrieval_metrics": True,
            "latency_ms": round(latency_ms, 2),
            "retrieval_config": retrieval_config or {},
        }

    evidence_hits: list[dict[str, Any]] = []
    matched_evidence_indexes: set[int] = set()
    matched_ranks: set[int] = set()

    for evidence_index, evidence in enumerate(expected_evidence):
        for rank, chunk in enumerate(results, start=1):
            match = match_evidence(evidence, chunk)
            if not match["matched"]:
                continue

            matched_evidence_indexes.add(evidence_index)
            matched_ranks.add(rank)
            evidence_hits.append(
                {
                    "evidence_index": evidence_index,
                    "rank": rank,
                    "chunk_id": chunk.get("id"),
                    "page": get_chunk_page(chunk),
                    "chunk_type": get_chunk_type(chunk),
                    "matched_terms": match["matched_terms"],
                }
            )
            break

    evidence_count = len(expected_evidence)

    def matched_within(k: int) -> set[int]:
        return {
            hit["evidence_index"]
            for hit in evidence_hits
            if hit["rank"] <= k
        }

    matched_at_5 = matched_within(5)
    matched_at_10 = matched_within(10)
    first_relevant_rank = min(matched_ranks) if matched_ranks else None
    recall_at_5 = len(matched_at_5) / evidence_count if evidence_count else 0.0
    recall_at_10 = len(matched_at_10) / evidence_count if evidence_count else 0.0

    expected_pages = {
        page
        for evidence in expected_evidence
        for page in evidence.get("pages") or []
    }
    expected_types = {
        normalize_text(chunk_type)
        for evidence in expected_evidence
        for chunk_type in evidence.get("chunk_types") or []
    }
    result_pages = {get_chunk_page(chunk) for chunk in results}
    result_types = {get_chunk_type(chunk) for chunk in results}
    missing_evidence_indexes = sorted(
        set(range(evidence_count)) - matched_evidence_indexes
    )
    failure = classify_failure(
        case,
        results,
        expected_evidence,
        evidence_hits,
        missing_evidence_indexes,
    )

    evaluation = {
        "id": case["id"],
        "category": case.get("category"),
        "question": case["question"],
        "answerable": True,
        "excluded_from_retrieval_metrics": False,
        "passed": recall_at_10 == 1.0,
        "metrics": {
            "hit_at_5": bool(matched_at_5),
            "hit_at_10": bool(matched_at_10),
            "recall_at_5": round(recall_at_5, 4),
            "recall_at_10": round(recall_at_10, 4),
            "reciprocal_rank": (
                round(1 / first_relevant_rank, 4)
                if first_relevant_rank is not None
                else 0.0
            ),
            "first_relevant_rank": first_relevant_rank,
            "page_hit": bool(expected_pages & result_pages),
            "type_hit": bool(expected_types & result_types),
        },
        "latency_ms": round(latency_ms, 2),
        "retrieval_config": retrieval_config or {},
        "expected_evidence": expected_evidence,
        "evidence_hits": evidence_hits,
        "missing_evidence_indexes": missing_evidence_indexes,
        "top_results": [
            {
                "rank": index + 1,
                "id": chunk.get("id"),
                "score": chunk.get("score"),
                "rerank_score": chunk.get("rerank_score"),
                "reranker_score": chunk.get("reranker_score"),
                "source_collection": chunk.get("source_collection"),
                "chunk_type": get_chunk_type(chunk),
                "page": get_chunk_page(chunk),
                "preview": normalize_text(chunk.get("content", ""))[:250],
            }
            for index, chunk in enumerate(results)
        ],
    }
    if failure:
        evaluation["failure"] = failure

    return evaluation


def aggregate_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "hit_at_5": 0.0,
            "hit_at_10": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "page_accuracy": 0.0,
            "chunk_type_accuracy": 0.0,
            "avg_latency_ms": 0.0,
        }

    total = len(items)

    def average(metric_name: str) -> float:
        return round(
            sum(float(item["metrics"][metric_name]) for item in items) / total,
            4,
        )

    passed = sum(1 for item in items if item["passed"])
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "hit_at_5": average("hit_at_5"),
        "hit_at_10": average("hit_at_10"),
        "recall_at_5": average("recall_at_5"),
        "recall_at_10": average("recall_at_10"),
        "mrr": average("reciprocal_rank"),
        "page_accuracy": average("page_hit"),
        "chunk_type_accuracy": average("type_hit"),
        "avg_latency_ms": round(
            sum(float(item["latency_ms"]) for item in items) / total,
            2,
        ),
    }


def summarize(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    included = [
        item
        for item in evaluations
        if not item["excluded_from_retrieval_metrics"]
    ]
    excluded = len(evaluations) - len(included)
    categories = sorted({item.get("category") or "unknown" for item in included})

    failed = [item for item in included if not item["passed"]]
    failure_reasons = sorted(
        {
            (item.get("failure") or {}).get("reason")
            for item in failed
            if (item.get("failure") or {}).get("reason")
        }
    )

    return {
        **aggregate_metrics(included),
        "excluded_unanswerable": excluded,
        "failure_reasons": {
            reason: sum(
                1
                for item in failed
                if (item.get("failure") or {}).get("reason") == reason
            )
            for reason in failure_reasons
        },
        "by_category": {
            category: aggregate_metrics(
                [
                    item
                    for item in included
                    if (item.get("category") or "unknown") == category
                ]
            )
            for category in categories
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    reports_dir: Path,
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    run_id = datetime.now().strftime("retrieval_%Y%m%d_%H%M%S")
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
        [
            item
            for item in evaluations
            if not item["excluded_from_retrieval_metrics"] and not item["passed"]
        ],
    )
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation against Hyena /query/similar API."
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
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--reranker",
        choices=["vector", "heuristic", "cross_encoder"],
        default="heuristic",
    )
    parser.add_argument("--reranker-model")
    parser.add_argument("--cross-encoder-top-n", type=int, default=12)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N cases after loading the test set.",
    )
    args = parser.parse_args()

    if args.top_k < 10:
        parser.error("--top-k must be at least 10 to calculate Hit@10 and Recall@10")

    cases = load_jsonl(args.test_set)
    if args.limit is not None:
        cases = cases[: args.limit]
    similar_url = args.api_base.rstrip("/") + "/query/similar"

    evaluations = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}")

        if not case.get("answerable", True):
            evaluations.append(evaluate_case(case, [], latency_ms=0.0))
            continue

        payload = {
            "question": case["question"],
            "top_k": args.top_k,
            "company": case.get("company"),
            "year": case.get("year"),
            "quarter": case.get("quarter"),
            "reranker": args.reranker,
            "cross_encoder_top_n": args.cross_encoder_top_n,
        }
        if args.reranker_model:
            payload["reranker_model"] = args.reranker_model
        if args.top_k_text is not None:
            payload["top_k_text"] = args.top_k_text
        if args.top_k_table is not None:
            payload["top_k_table"] = args.top_k_table
        if args.top_k_image is not None:
            payload["top_k_image"] = args.top_k_image

        started_at = time.perf_counter()
        response = post_json(similar_url, payload, timeout=args.timeout)
        latency_ms = (time.perf_counter() - started_at) * 1000
        results = response.get("results") or []
        evaluations.append(
            evaluate_case(
                case,
                results,
                latency_ms=latency_ms,
                retrieval_config=response.get("retrieval_config"),
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
    }
    run_dir = write_outputs(args.reports_dir, summary, evaluations, config)

    print("\nRetrieval evaluation complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRun directory: {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
