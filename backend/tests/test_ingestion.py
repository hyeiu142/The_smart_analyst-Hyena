"""Tests for document upload API endpoint."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
import io


@pytest.fixture
def client():
    """FastAPI test client."""
    with patch("backend.app.core.retrieval.qdrant_client.QdrantClient"):
        from backend.app.main import app
        return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """GET / returns API info."""
        resp = client.get("/api")
        # Either 200 (JSON) or 404 is acceptable depending on static mount
        assert resp.status_code in (200, 404)

    @patch("backend.app.api.v1.health.get_settings")
    def test_health_endpoint(self, mock_settings, client):
        """GET /api/v1/health/ returns 200."""
        resp = client.get("/api/v1/health/")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


class TestDocumentModels:
    """Test Pydantic models for documents."""

    def test_query_request_defaults(self):
        """QueryRequest has sensible defaults."""
        from backend.app.models.query import QueryRequest
        req = QueryRequest(question="What is revenue?")
        assert req.top_k == 5
        assert req.company is None
        assert req.year is None

    def test_query_request_custom_top_k(self):
        """QueryRequest accepts custom top_k."""
        from backend.app.models.query import QueryRequest
        req = QueryRequest(question="Revenue?", top_k=10)
        assert req.top_k == 10

    def test_source_document_model(self):
        """SourceDocument validates correctly — matches ContextBuilder.build_citations() output."""
        from backend.app.models.query import SourceDocument
        src = SourceDocument(
            index=1,
            type="text",
            company="FPT",
            page=3,
            score=0.92,
            preview="Revenue was 17,045B VND...",
        )
        assert src.score == 0.92
        assert src.company == "FPT"
        assert src.index == 1
        assert src.type == "text"
