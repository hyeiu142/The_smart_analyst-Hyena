from fastapi import APIRouter
from qdrant_client import QdrantClient
import redis as redis_lib

from backend.app.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/")
async def health_check():
    return {"status": "healthy", "service": "Hyena API"}


@router.get("/qdrant")
async def qdrant_health():
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        collections = client.get_collections()
        return {
            "status": "connected",
            "collections": [c.name for c in collections.collections],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/redis")
async def redis_health():
    try:
        r = redis_lib.from_url(settings.redis_url)
        r.ping()
        return {"status": "connected", "url": settings.redis_url}
    except Exception as e:
        return {"status": "error", "message": str(e)}