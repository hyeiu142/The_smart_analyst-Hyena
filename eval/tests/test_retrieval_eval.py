from io import BytesIO
from urllib.error import HTTPError

from eval.scripts import run_retrieval_eval
from eval.scripts.run_retrieval_eval import (
    evaluate_case,
    classify_missing_evidence,
    match_evidence,
    summarize,
)


def make_chunk(
    *,
    content: str,
    page: int,
    chunk_type: str,
    score: float = 0.8,
) -> dict:
    return {
        "id": f"{chunk_type}-{page}",
        "score": score,
        "content": content,
        "metadata": {"page": page, "chunk_type": chunk_type},
        "source_collection": chunk_type,
    }


def test_match_evidence_normalizes_decimal_separator() -> None:
    evidence = {
        "pages": [3],
        "chunk_types": ["text"],
        "required_terms": ["25,4%"],
    }
    chunk = make_chunk(
        content="Nhật Bản tăng trưởng 25.4% trong năm.",
        page=3,
        chunk_type="text",
    )

    assert match_evidence(evidence, chunk)["matched"] is True


def test_evaluate_mixed_case_requires_all_evidence_for_pass() -> None:
    case = {
        "id": "mixed",
        "category": "mixed",
        "question": "Example",
        "answerable": True,
        "expected_evidence": [
            {
                "pages": [3],
                "chunk_types": ["text"],
                "required_terms": ["25,4%"],
            },
            {
                "pages": [4],
                "chunk_types": ["image"],
                "required_terms": [],
            },
        ],
    }
    results = [
        make_chunk(content="Nhật Bản +25,4%", page=3, chunk_type="text"),
        make_chunk(content="Pending image", page=4, chunk_type="image"),
    ]

    evaluation = evaluate_case(case, results, latency_ms=12.5)

    assert evaluation["passed"] is True
    assert evaluation["metrics"]["hit_at_5"] is True
    assert evaluation["metrics"]["recall_at_5"] == 1.0
    assert evaluation["metrics"]["reciprocal_rank"] == 1.0


def test_unanswerable_case_is_excluded_from_retrieval_metrics() -> None:
    case = {
        "id": "unanswerable",
        "category": "unanswerable",
        "question": "Unknown",
        "answerable": False,
        "expected_evidence": [],
    }

    evaluation = evaluate_case(case, [], latency_ms=0.0)
    summary = summarize([evaluation])

    assert evaluation["excluded_from_retrieval_metrics"] is True
    assert summary["total"] == 0
    assert summary["excluded_unanswerable"] == 1


def test_failed_case_has_failure_reason_summary() -> None:
    case = {
        "id": "table",
        "category": "table",
        "question": "Example",
        "answerable": True,
        "expected_evidence": [
            {
                "pages": [5],
                "chunk_types": ["table"],
                "required_terms": ["88.089"],
            },
        ],
    }
    results = [
        make_chunk(content="Wrong table value 10.000", page=5, chunk_type="table")
    ]

    evaluation = evaluate_case(case, results, latency_ms=1.0)
    summary = summarize([evaluation])

    assert evaluation["passed"] is False
    assert evaluation["failure"]["reason"] == "evidence_term_mismatch"
    assert summary["failure_reasons"] == {"evidence_term_mismatch": 1}


def test_classify_missing_image_evidence_wrong_page() -> None:
    evidence = {
        "pages": [4],
        "chunk_types": ["image"],
        "required_terms": [],
    }
    results = [make_chunk(content="image pending", page=5, chunk_type="image")]

    classification = classify_missing_evidence(evidence, results)

    assert classification["reason"] == "image_wrong_page"


def test_post_json_retries_rate_limit(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b'{"results": []}'

    responses = iter(
        [
            HTTPError(
                url="http://example.test",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=BytesIO(b'{"retry_after": 1}'),
            ),
            Response(),
        ]
    )
    waits: list[int] = []

    def fake_urlopen(request, timeout):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        run_retrieval_eval,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(run_retrieval_eval.time, "sleep", waits.append)

    result = run_retrieval_eval.post_json(
        "http://example.test",
        {"question": "test"},
    )

    assert result == {"results": []}
    assert waits == [2]
