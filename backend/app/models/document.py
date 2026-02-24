from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentUploadRequest(BaseModel):
    company: str = Field(..., example="Vinamilk")
    year: int = Field(..., example=2025)
    quarter: Optional[str] = Field(None, example="Q4")


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    company: str
    year: int
    quarter: Optional[str] = None
    status: str  # pending | processing | completed | failed
    created_at: datetime
    message: str


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    company: str
    year: int
    quarter: Optional[str] = None
    status: str
    total_chunks: Optional[int] = 0
    text_chunks: Optional[int] = 0
    table_chunks: Optional[int] = 0
    image_chunks: Optional[int] = 0
    created_at: datetime


class DocumentStatusResponse(BaseModel):
    doc_id: str
    status: str
    progress: Optional[str] = None
    total_chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    error: Optional[str] = None