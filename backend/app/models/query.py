from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str
    top_k: int = 5

    company: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[str] = None

class SourceDocument(BaseModel):
    """1 source document in result"""
    content: str
    metadata: Dict[str, Any]
    score: float

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    sources: List[SourceDocument]
    question: str 