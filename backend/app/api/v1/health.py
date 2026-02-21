from fastapi import APIRouter
from qdrant_client import QdrantClient  

from backend.app.config import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/")
async def health_check():
    return {"status":"healthy"}

@router.get("/qdrant")
async def qdrant_health():
    try: 
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        collections = client.get_collections()
        return {
            "status":"conneted", 
            "collections":[c.name for c in collections.collections]
        }
    except Exception as e:
        return {
            "status":"error", 
            "message":str(e)
        }