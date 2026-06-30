from backend.app.core.retrieval.reranker import (
    CrossEncoderReranker,
    get_cross_encoder_reranker,
    rerank_chunks,
)


def make_chunk(
    *,
    score: float,
    content: str,
    page: int,
    chunk_type: str,
) -> dict:
    return {
        "id": f"{chunk_type}-{page}",
        "score": score,
        "content": content,
        "metadata": {"page": page, "chunk_type": chunk_type},
        "source_collection": chunk_type,
    }


def test_reranker_uses_page_context_for_pending_images() -> None:
    chunks = [
        make_chunk(
            score=0.72,
            content="Cơ cấu doanh thu theo thị trường Mỹ Nhật Bản APAC",
            page=4,
            chunk_type="text",
        ),
        make_chunk(
            score=0.16,
            content="pending chart/image crop from financial report. document page 5.",
            page=5,
            chunk_type="image",
        ),
        make_chunk(
            score=0.12,
            content="pending chart/image crop from financial report. document page 4.",
            page=4,
            chunk_type="image",
        ),
    ]

    results = rerank_chunks(
        "Biểu đồ cơ cấu doanh thu theo thị trường năm 2025 là gì?",
        chunks,
    )

    assert results[0]["id"] == "image-4"


def test_reranker_penalizes_images_for_non_image_questions() -> None:
    chunks = [
        make_chunk(
            score=0.55,
            content="Doanh thu năm 2025 là 70.113 tỷ đồng",
            page=1,
            chunk_type="table",
        ),
        make_chunk(
            score=0.54,
            content="pending chart/image crop from financial report. document page 1.",
            page=1,
            chunk_type="image",
        ),
    ]

    results = rerank_chunks("Doanh thu năm 2025 là bao nhiêu?", chunks)

    assert results[0]["id"] == "table-1"


def test_cross_encoder_can_force_rerank_when_candidates_fit_top_n(monkeypatch) -> None:
    class FakeModel:
        def predict(self, pairs):
            return [0.1, 0.9]

    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_load_model", lambda: None)
    reranker._model = FakeModel()

    chunks = [
        make_chunk(score=0.9, content="less relevant", page=1, chunk_type="text"),
        make_chunk(score=0.8, content="more relevant", page=2, chunk_type="text"),
    ]

    results = reranker.rerank("query", chunks, top_n=2, force=True)

    assert results[0]["id"] == "text-2"
    assert results[0]["reranker_score"] == 0.9


def test_cross_encoder_reranker_is_cached_by_model_name() -> None:
    get_cross_encoder_reranker.cache_clear()

    first = get_cross_encoder_reranker("model-a")
    second = get_cross_encoder_reranker("model-a")
    other = get_cross_encoder_reranker("model-b")

    assert first is second
    assert first is not other
