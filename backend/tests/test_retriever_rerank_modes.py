from backend.app.core.retrieval.retriever import MultiCollectionRetriever


def make_chunk(score: float, content: str, page: int, chunk_type: str) -> dict:
    return {
        "id": f"{chunk_type}-{page}-{score}",
        "score": score,
        "content": content,
        "metadata": {"page": page, "chunk_type": chunk_type},
    }


def test_cross_encoder_mode_preranks_before_model_scoring(monkeypatch) -> None:
    retriever = MultiCollectionRetriever()
    monkeypatch.setattr(retriever.embedder, "embed_documents", lambda question: [0.1])
    monkeypatch.setattr(retriever, "_build_filter", lambda filters: None)

    def fake_search(collection_name, query_vector, limit, filters):
        chunk_type = collection_name.split("_")[0]
        return [
            make_chunk(
                score=0.5 + index / 100,
                content=f"{chunk_type} doanh thu 2025 {index}",
                page=index,
                chunk_type=chunk_type,
            )
            for index in range(limit)
        ]

    class FakeModel:
        def rerank(self, query, chunks, top_n, force=False):
            assert len(chunks) == 3
            assert top_n == 3
            for chunk in chunks:
                chunk["reranker_score"] = 1.0
            return list(reversed(chunks))

    monkeypatch.setattr(retriever.qdrant, "search", fake_search)
    monkeypatch.setattr(
        "backend.app.core.retrieval.retriever.get_cross_encoder_reranker",
        lambda model_name: FakeModel(),
    )

    results = retriever.retrieve(
        "Doanh thu năm 2025?",
        top_k_text=4,
        top_k_table=4,
        top_k_image=0,
        reranker="cross_encoder",
        cross_encoder_top_n=3,
    )

    assert len(results) == 8
    assert results[0]["reranker_score"] == 1.0
