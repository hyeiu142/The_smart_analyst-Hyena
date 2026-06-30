from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str
    top_k: int = 5
    top_k_text: Optional[int] = None
    top_k_table: Optional[int] = None
    top_k_image: Optional[int] = None
    reranker: str = "heuristic"
    reranker_model: Optional[str] = None
    cross_encoder_top_n: int = 12

    company: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[str] = None

class SourceDocument(BaseModel):
    """1 source citation trả về frontend — khớp với ContextBuilder.build_citations()"""
    index: int
    type: str          # text | table | image
    company: str
    page: Any          # int hoặc "?"
    score: float
    preview: str

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    sources: List[SourceDocument]
    question: str
