from fastapi import APIRouter, HTTPException

from backend.app.models.query import QueryRequest, QueryResponse
from backend.app.core.generation.rag_engine import RAGEngine

router = APIRouter()

rag_engine = RAGEngine()

@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG Query endpoint.

    Take question, search in Qdrant, generate answer with LLM.
    """
    try: 
        filters = None
        result = await rag_engine.query(
            question=request.question, 
            top_k=request.top_k, 
            filters=filters
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            question=request.question
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))