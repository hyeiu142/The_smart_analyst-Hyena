from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.api.v1.router import api_router
from backend.app.config import get_settings

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "name": "Hyena API", 
        "version": "1.0.0",
        "status": "running"
    }