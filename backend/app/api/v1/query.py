from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from backend.app.models.query import QueryRequest, QueryResponse
from backend.app.core.generation.rag_engine import RAGEngine

router = APIRouter()

rag_engine = RAGEngine()


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG Query endpoint — multi-collection với auto filter từ company/year/quarter.
    """
    try:
        # Build filters từ request nếu có
        filters = {}
        if request.company:
            filters["company"] = request.company
        if request.year:
            filters["year"] = request.year
        if request.quarter:
            filters["quarter"] = request.quarter
        filters = filters if filters else None

        result = await rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            filters=filters,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            question=request.question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming RAG response (Server-Sent Events).
    Frontend nhận từng token như ChatGPT.
    """
    async def event_generator():
        try:
            filters = {}
            if request.company:
                filters["company"] = request.company
            if request.year:
                filters["year"] = request.year
            filters = filters if filters else None

            async for token in rag_engine.stream_query(
                question=request.question,
                top_k=request.top_k,
                filters=filters,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/similar")
async def find_similar(request: QueryRequest):
    """Tìm chunks tương tự, không generate answer."""
    try:
        from backend.app.core.retrieval.retriever import MultiCollectionRetriever
        retriever = MultiCollectionRetriever()
        chunks = retriever.retrieve(question=request.question, top_k_text=3, top_k_table=5, top_k_image=2)
        return {"question": request.question, "results": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
