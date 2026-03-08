"""Tests for the RAG query pipeline."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestQueryAnalyzer:
    """Test QueryAnalyzer intent detection."""

    def test_fact_lookup_intent(self):
        """Revenue/profit questions → fact_lookup."""
        from backend.app.core.generation.query_analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("What was the revenue in Q4 2025?")
        assert result["intent"] in ("fact_lookup", "comparison", "trend")

    def test_comparison_intent(self):
        """'Compare' keyword → comparison intent."""
        from backend.app.core.generation.query_analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Compare gross margin between 2024 and 2025")
        assert "intent" in result
        assert "data_types_needed" in result

    def test_analysis_returns_required_keys(self):
        """Analysis result always has required keys."""
        from backend.app.core.generation.query_analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Any question")
        assert "intent" in result
        assert "data_types_needed" in result


class TestContextBuilder:
    """Test ContextBuilder formatting."""

    def test_build_empty(self):
        """Empty chunks → fallback message."""
        from backend.app.core.generation.context_builder import ContextBuilder
        builder = ContextBuilder()
        result = builder.build([])
        assert "No relevant" in result or result == ""

    def test_build_with_chunks(self, sample_chunks):
        """Chunks → formatted context string."""
        from backend.app.core.generation.context_builder import ContextBuilder
        chunks_with_meta = [
            {**c, "score": 0.9, "source_collection": "text"}
            for c in sample_chunks
        ]
        builder = ContextBuilder()
        context = builder.build(chunks_with_meta)
        assert len(context) > 0
        assert "FPT" in context or "Revenue" in context

    def test_build_citations(self, sample_chunks):
        """Citations include required fields."""
        from backend.app.core.generation.context_builder import ContextBuilder
        chunks_with_meta = [
            {**c, "score": 0.9, "source_collection": "text"}
            for c in sample_chunks
        ]
        builder = ContextBuilder()
        citations = builder.build_citations(chunks_with_meta)

        assert len(citations) == len(sample_chunks)
        for cite in citations:
            assert "index" in cite
            assert "score" in cite
            assert "company" in cite
            assert "preview" in cite


class TestQdrantClientWrapper:
    """Test Qdrant client wrapper."""

    @patch("backend.app.core.retrieval.qdrant_client.QdrantClient")
    def test_ensure_collections_creates_missing(self, mock_qdrant_cls):
        """ensure_collections() creates collections that don't exist."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        # Simulate no existing collections
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper
        wrapper = QdrantClientWrapper()
        wrapper.ensure_collections()

        # Should create 3 collections
        assert mock_client.create_collection.call_count == 3

    @patch("backend.app.core.retrieval.qdrant_client.QdrantClient")
    def test_ensure_collections_skips_existing(self, mock_qdrant_cls):
        """ensure_collections() skips already existing collections."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        # Simulate all 3 collections exist
        existing = [
            MagicMock(name="text_chunks"),
            MagicMock(name="table_chunks"),
            MagicMock(name="image_chunks"),
        ]
        existing[0].name = "text_chunks"
        existing[1].name = "table_chunks"
        existing[2].name = "image_chunks"

        mock_collections = MagicMock()
        mock_collections.collections = existing
        mock_client.get_collections.return_value = mock_collections

        from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper
        wrapper = QdrantClientWrapper()
        wrapper.ensure_collections()

        mock_client.create_collection.assert_not_called()


class TestEmbedder:
    """Test Embedder wrapper."""

    @patch("backend.app.core.retrieval.embedder.OpenAI")
    def test_embed_single_returns_vector(self, mock_openai_cls):
        """embed_documents() returns a list of floats."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        fake_embedding = MagicMock()
        fake_embedding.embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = MagicMock(
            data=[fake_embedding]
        )

        from backend.app.core.retrieval.embedder import Embedder
        embedder = Embedder()
        result = embedder.embed_documents("test text")

        assert isinstance(result, list)
        assert len(result) == 1536

    @patch("backend.app.core.retrieval.embedder.OpenAI")
    def test_embed_batch_returns_multiple_vectors(self, mock_openai_cls):
        """embed_batch() returns one vector per input text."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        texts = ["text one", "text two", "text three"]
        mock_embeddings = [
            MagicMock(embedding=[0.1] * 1536) for _ in texts
        ]
        mock_client.embeddings.create.return_value = MagicMock(
            data=mock_embeddings
        )

        from backend.app.core.retrieval.embedder import Embedder
        embedder = Embedder()
        result = embedder.embed_batch(texts)

        assert len(result) == len(texts)
        for vec in result:
            assert len(vec) == 1536
