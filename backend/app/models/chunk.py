from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class ChunkMetadata(BaseModel):
    doc_id: str
    company: str
    year: int
    quarter: Optional[str] = None
    page_num: int = 1
    chunk_type: str
    section_path: Optional[str] = None
    heading_h1: Optional[str] = None
    heading_h2: Optional[str] = None

class TextChunkMetadata(ChunkMetadata):
    chunk_type: str = "text"

class TableChunkMetadata(ChunkMetadata):
    chunk_type: str = "table"
    table_id: Optional[str] = None
    table_title: Optional[str] = None
    row_count: Optional[int] = None
    col_count: Optional[int] = None
    headers: Optional[List[str]] = None

class ImageChunkMetadata(ChunkMetadata):
    chunk_type: str = "image_caption"
    image_path: Optional[str] = None
    chart_type: Optional[str] = None
    data_points: Optional[Dict[str, Any]] = None

class Chunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None