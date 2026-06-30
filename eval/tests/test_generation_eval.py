from eval.scripts.run_generation_eval import (
    contains_number,
    deterministic_judge,
    get_expected_terms,
    summarize,
)


def test_expected_terms_extracts_numbers_from_ground_truth_and_evidence() -> None:
    case = {
        "ground_truth_answer": "Nhật Bản tăng 25,4% và chiếm 43,7% doanh thu.",
        "evidence": "Text trang 3 ghi +25.4%; biểu đồ trang 4 ghi 43,7%.",
        "expected_evidence": [],
    }

    assert get_expected_terms(case) == ["25,4%", "43,7%"]


def test_deterministic_judge_accepts_decimal_separator_variants() -> None:
    case = {
        "answerable": True,
        "ground_truth_answer": "23,1%.",
        "evidence": "Biểu đồ trang 4 ghi Mỹ 23,1%.",
        "expected_pages": [4],
        "expected_chunk_types": ["image", "chart"],
    }
    answer = "Thị trường Mỹ chiếm 23.1% doanh thu [Source #1]."
    sources = [{"page": 4, "type": "image", "preview": "Mỹ 23,1%"}]

    result = deterministic_judge(case, answer, sources)

    assert result["passed"] is True
    assert result["scores"]["answer_correctness"] == 1.0


def test_deterministic_judge_fails_wrong_number() -> None:
    case = {
        "answerable": True,
        "ground_truth_answer": "23,1%.",
        "evidence": "Biểu đồ trang 4 ghi Mỹ 23,1%.",
        "expected_pages": [4],
        "expected_chunk_types": ["image"],
    }
    answer = "Thị trường Mỹ chiếm 30% doanh thu [Source #1]."
    sources = [{"page": 4, "type": "image", "preview": "Mỹ 23,1%"}]

    result = deterministic_judge(case, answer, sources)

    assert result["passed"] is False
    assert result["reason"] == "answer_term_mismatch"


def test_contains_number_allows_small_ocr_rounding_difference() -> None:
    assert contains_number("Doanh thu 19,508 tỷ VND", "19.507")
    assert contains_number("Tiền mặt 10,541 tỷ VND", "10.540")
    assert not contains_number("Doanh thu 30% doanh thu", "28%")


def test_unanswerable_refusal_passes_without_hallucinated_number() -> None:
    case = {
        "answerable": False,
        "question": "Kế hoạch lợi nhuận trước thuế năm 2026 là bao nhiêu?",
        "ground_truth_answer": "Không có thông tin.",
        "expected_pages": [],
        "expected_chunk_types": [],
    }
    answer = "Không có thông tin về kế hoạch lợi nhuận trước thuế năm 2026 trong tài liệu đã cho."

    result = deterministic_judge(case, answer, [])

    assert result["passed"] is True


def test_unanswerable_refusal_allows_contextual_dates() -> None:
    case = {
        "answerable": False,
        "question": "FPT đã trả bao nhiêu tiền để sở hữu bản quyền Ngoại hạng Anh?",
        "ground_truth_answer": "Không công bố giá trị mua bản quyền.",
        "expected_pages": [],
        "expected_chunk_types": [],
    }
    answer = (
        "Tài liệu không cung cấp số tiền FPT đã trả. Tài liệu chỉ nêu "
        "giai đoạn phát sóng 2025-2026 đến 2030-2031."
    )

    result = deterministic_judge(case, answer, [])

    assert result["passed"] is True
    assert result["reason"] == "unanswerable_refusal"


def test_summarize_groups_failures_by_category_and_reason() -> None:
    evaluations = [
        {
            "category": "image",
            "passed": True,
            "latency_ms": 100,
            "judge": {
                "reason": "passed",
                "scores": {
                    "answer_correctness": 1.0,
                    "faithfulness": 1.0,
                    "citation_accuracy": 1.0,
                    "unanswerable_handling": 1.0,
                },
            },
        },
        {
            "category": "image",
            "passed": False,
            "latency_ms": 200,
            "judge": {
                "reason": "answer_term_mismatch",
                "scores": {
                    "answer_correctness": 0.0,
                    "faithfulness": 1.0,
                    "citation_accuracy": 1.0,
                    "unanswerable_handling": 1.0,
                },
            },
        },
    ]

    summary = summarize(evaluations)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["failure_reasons"] == {"answer_term_mismatch": 1}
    assert summary["by_category"]["image"]["total"] == 2
