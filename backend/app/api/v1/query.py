from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from backend.app.models.query import QueryRequest, QueryResponse
from backend.app.core.generation.rag_engine import RAGEngine

router = APIRouter()

rag_engine = RAGEngine()


def build_retrieval_allocation(request: QueryRequest) -> tuple[int, int, int, int]:
    top_k = max(1, request.top_k)

    if (
        request.top_k_text is not None
        or request.top_k_table is not None
        or request.top_k_image is not None
    ):
        # Cho phép evaluator thử nhiều retrieval allocation trên DEV.
        # Field nào không truyền thì giữ 0 để test đúng cấu hình người gọi.
        return (
            top_k,
            max(0, request.top_k_text or 0),
            max(0, request.top_k_table or 0),
            max(0, request.top_k_image or 0),
        )

    # Giữ tỷ lệ retrieval mặc định hiện tại: 30% text, 50% table, 20% image.
    top_k_text = int(top_k * 0.3)
    top_k_table = int(top_k * 0.5)
    top_k_image = top_k - top_k_text - top_k_table
    return top_k, top_k_text, top_k_table, top_k_image


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
    print(f"\nĐÃ NHẬN ĐƯỢC CÂU HỎI: '{request.question}' TỪ FRONTEND!\n")
    async def event_generator():
        try:
            filters = {}
            if request.company:
                filters["company"] = request.company
            if request.year:
                filters["year"] = request.year
            filters = filters if filters else None

            async for event in rag_engine.stream_query(
                question=request.question,
                top_k=request.top_k,
                filters=filters,
            ):
                payload = event if isinstance(event, dict) else {"token": event}
                yield f"data: {json.dumps(payload)}\n\n"

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

        filters = {}
        if request.company:
            filters["company"] = request.company
        if request.year:
            filters["year"] = request.year
        if request.quarter:
            filters["quarter"] = request.quarter

        top_k, top_k_text, top_k_table, top_k_image = build_retrieval_allocation(
            request
        )

        retriever = MultiCollectionRetriever()
        chunks = retriever.retrieve(
            question=request.question,
            top_k_text=top_k_text,
            top_k_table=top_k_table,
            top_k_image=top_k_image,
            filters=filters or None,
            reranker=request.reranker,
            reranker_model=request.reranker_model,
            cross_encoder_top_n=request.cross_encoder_top_n,
        )

        chunks = chunks[:top_k]

        return {
            "question": request.question,
            "retrieval_config": {
                "top_k": top_k,
                "top_k_text": top_k_text,
                "top_k_table": top_k_table,
                "top_k_image": top_k_image,
                "reranker": request.reranker,
                "reranker_model": request.reranker_model,
                "cross_encoder_top_n": request.cross_encoder_top_n,
                "filters": filters,
            },
            "results": chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def cache_stats():
    """Return semantic cache statistics."""
    try:
        stats = rag_engine.cache.stats() if rag_engine.cache else {"status": "disabled"}
        return {"cache": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
