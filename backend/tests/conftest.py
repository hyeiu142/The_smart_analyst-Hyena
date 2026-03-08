import pytest
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_settings():
    """Mock settings to avoid requiring actual API keys in tests."""
    settings = MagicMock()
    settings.openai_api_key = "sk-test-key"
    settings.llama_cloud_api_key = "llx-test-key"
    settings.google_api_key = "test-google-key"
    settings.qdrant_host = "localhost"
    settings.qdrant_port = 6333
    settings.redis_url = "redis://localhost:6379/0"
    settings.embedding_model = "text-embedding-3-small"
    settings.llm_model = "gpt-4o-mini"
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    return settings


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "id": "test-uuid-1",
            "content": "Revenue in Q4 2025 was 17,045 billion VND, up 14.9% YoY.",
            "metadata": {
                "doc_id": "doc-uuid-1",
                "company": "FPT",
                "year": 2025,
                "quarter": "Q4",
                "page": 1,
                "chunk_type": "text",
            },
        },
        {
            "id": "test-uuid-2",
            "content": "| Metric | 2024 | 2025 | %YoY |\n| Revenue | 14,833 | 17,045 | +14.9% |",
            "metadata": {
                "doc_id": "doc-uuid-1",
                "company": "FPT",
                "year": 2025,
                "quarter": "Q4",
                "page": 2,
                "chunk_type": "table",
            },
        },
    ]
