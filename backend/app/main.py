from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import logging

# Configure logging so RAG / Reranker logs appear in uvicorn terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from backend.app.api.v1.router import api_router
from backend.app.config import get_settings
from backend.app.middleware.rate_limiter import RateLimiterMiddleware

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Hyena API...")
    yield
    print("Shutting down Hyena API...")

app = FastAPI(
    title="Hyena Multimodal RAG API",
    description="Enterprise Document Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimiterMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/api")
async def root():
    return {
        "name": "Hyena API",
        "version": "1.0.0",
        "status": "running"
    }

# Serve frontend — must be LAST
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")