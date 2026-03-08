# 🐾 Hyena — Financial Document Intelligence

An enterprise-grade **Multimodal RAG** system that lets you upload financial reports (PDF) and ask questions in natural language. Powered by LlamaParse, OpenAI, and Qdrant.

## ✨ Features

- **📄 Multimodal ingestion** — extracts text, tables, and images from PDFs
- **🔍 Semantic search** — multi-collection Qdrant vector search
- **🧠 Agentic RAG** — query analyzer detects intent and routes to the right data
- **⚡ Streaming answers** — token-by-token responses via SSE
- **📊 Citation tracking** — every answer links back to source pages
- **🐳 Docker-ready** — one command to deploy everything

## 🏗 Architecture

```
PDF Upload
    │
    ▼
LlamaParse ──→ TextProcessor ──→ text_chunks (Qdrant)
               TableProcessor ─→ table_chunks (Qdrant)
               ImageProcessor ─→ image_chunks (Qdrant)
    │
    ▼ (Celery background job)

User Query
    │
    ├──→ QueryAnalyzer (intent + entities)
    ├──→ MultiCollectionRetriever (search all 3 collections)
    ├──→ ContextBuilder (format context + citations)
    └──→ LLM (GPT-4o-mini) ──→ Answer + Sources
```

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose
- API keys: OpenAI, LlamaCloud, Google

### 1. Clone & configure
```bash
git clone https://github.com/PivePipioipia/The_smart_analyst-Hyena.git
cd The_smart_analyst-Hyena

cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Start everything
```bash
make up
```

That's it! Open:
- **Frontend:** http://localhost
- **API docs:** http://localhost:8001/docs

### Stop
```bash
make down
```

---

## 💻 Local Development

```bash
# Start infrastructure (Qdrant + Redis)
make infra

# Start backend + Celery worker together
make dev-all

# Frontend is served by FastAPI at http://localhost:8000
```

### Environment (local dev)
```bash
REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
```

---

## 🗂 Project Structure

```
Hyena/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # FastAPI endpoints
│   │   │   ├── documents.py # Upload, list, delete, status
│   │   │   ├── query.py     # RAG query + streaming
│   │   │   └── health.py    # Health checks
│   │   ├── core/
│   │   │   ├── ingestion/   # PDF processing pipeline
│   │   │   ├── retrieval/   # Qdrant + embeddings
│   │   │   └── generation/  # RAG engine + context builder
│   │   ├── models/          # Pydantic schemas
│   │   ├── workers/         # Celery tasks
│   │   └── config.py        # Settings (pydantic-settings)
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
├── scripts/                 # Utility scripts
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload PDF |
| `GET`  | `/api/v1/documents/` | List all documents |
| `GET`  | `/api/v1/documents/{id}/status` | Check processing status |
| `DELETE` | `/api/v1/documents/{id}` | Delete document + chunks |
| `POST` | `/api/v1/query/` | RAG query |
| `POST` | `/api/v1/query/stream` | Streaming RAG query (SSE) |
| `GET`  | `/api/v1/health/qdrant` | Qdrant health |
| `GET`  | `/api/v1/health/redis` | Redis health |

Full interactive docs: http://localhost:8001/docs

---

## 🧪 Testing

```bash
# Unit + integration tests
make test

# Test ingestion pipeline manually
uv run python scripts/test_ingestion.py

# Test RAG query pipeline manually
uv run python scripts/test_query.py
```

---

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `hyena-frontend` | 80 | Nginx serving UI |
| `hyena-backend` | 8001 | FastAPI server |
| `hyena-worker` | — | Celery worker |
| `hyena-redis` | 6379 | Message broker + job store |
| `hyena-qdrant` | 6333 | Vector database |

---

## 🛠 Makefile Commands

```bash
make up        # Start all services (Docker)
make down      # Stop all services
make build     # Rebuild Docker images
make logs      # Follow all logs
make infra     # Start Qdrant + Redis only
make dev       # Run backend locally
make worker    # Run Celery worker locally
make dev-all   # Run backend + worker together
make test      # Run tests
make clean     # Remove all Docker volumes
```

---

## 📝 License

MIT
