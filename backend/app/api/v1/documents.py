import os
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import aiofiles

from backend.app.models.document import(
    DocumentListItem, 
    DocumentResponse, 
    DocumentStatusResponse, 
)
from backend.app.workers.tasks import process_document_task

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_doc_store: dict = {}

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    company: str = Form(...),
    year: int = Form(...),
    quarter: str = Form(None),
):
    """Upload PDF and kick off Celery ingestion task."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    metadata = {
        "company": company, 
        "year": year, 
        "quarter": quarter,
        "original_filename": file.filename,
        "file_path": file_path,
    }

    _doc_store[doc_id] = {
        "doc_id": doc_id, 
        "filename": file.filename,
        "company": company,
        "year": year,
        "quarter": quarter,
        "status": "pending",
        "total_chunks": 0,
        "text_chunks": 0,
        "table_chunks": 0,
        "image_chunks": 0,
        "created_at": datetime.utcnow(),
        "error": None,
    }

    process_document_task.delay(doc_id, file_path, metadata)

    return DocumentResponse(
        doc_id=doc_id,
        filename=file.filename,
        company=company,
        year=year,
        quarter=quarter,
        status="pending",
        created_at=_doc_store[doc_id]["created_at"],
        message=f"Uploaded. Processing started. Track with doc_id: {doc_id}",
    )

@router.get("/", response_model=List[DocumentListItem])
async def list_documents():
    return [DocumentListItem(**doc) for doc in _doc_store.values()]

@router.get("/{doc_id}", response_model=DocumentListItem)
async def get_document(doc_id: str):
    doc = _doc_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentListItem(**doc)

@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(doc_id: str):
    doc = _doc_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        doc_id=doc_id,
        status=doc["status"],
        total_chunks=doc["total_chunks"],
        text_chunks=doc["text_chunks"],
        table_chunks=doc["table_chunks"],
        image_chunks=doc["image_chunks"],
        error=doc.get("error"),
    )

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    doc = _doc_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper
    qdrant = QdrantClientWrapper()
    qdrant.delete_by_doc_id(doc_id)

    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{doc['filename']}")
    if os.path.exists(file_path):
        os.remove(file_path)

    del _doc_store[doc_id]
    return {"message": f"Document {doc_id} deleted successfully"}

def update_doc_status(doc_id: str, status: str, result: dict = None, error: str = None):
    """Call from Celery task to update document status."""
    if doc_id not in _doc_store:
        return
    _doc_store[doc_id]["status"] = status
    if error:
        _doc_store[doc_id]["error"] = error
    if result:
        _doc_store[doc_id]["total_chunks"] = result.get("total_chunks", 0)
        _doc_store[doc_id]["text_chunks"] = result.get("text_chunks", 0)
        _doc_store[doc_id]["table_chunks"] = result.get("table_chunks", 0)
        _doc_store[doc_id]["image_chunks"] = result.get("image_chunks", 0)