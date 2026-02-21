from typing import Dict, Any
import uuid

from backend.app.core.ingestion.text_processor import TextProcessor
from backend.app.core.retrieval.embedder import Embedder
from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper  

class IngestionPipeline: 
    """
    Main orchestrator cho document ingestion.
    
    Flow:
    1. Parse PDF → Text chunks
    2. Embed chunks
    3. Store vào Qdrant
    """

    def __init__(self):
        self.text_processor=TextProcessor()
        self.embedder = Embedder()
        self.qdrant=QdrantClientWrapper()
    async def process_document(
        self, 
        file_path: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process 1 document completely.
        
        Args:
            file_path: Path to PDF
            doc_id: Unique document ID
            metadata: {
                "company": "Vinamilk",
                "year": 2025,
                "quarter": "Q4",
                "original_filename": "vnm_q4_2025.pdf"
            }
        
        Returns:
            {
                "doc_id": "uuid-abc",
                "total_chunks": 45,
                "text_chunks": 45,
                "status": "completed"
            }
        """
        print(f"\nStarting ingestion for doc_id: {doc_id}")
        metadata["doc_id"] = doc_id

        text_chunks = self.text_processor.process(file_path, metadata)

        print(f"Embedding {len(text_chunks)} text chunks...")
        contents = [chunk["content"] for chunk in text_chunks]
        embeddings = self.embedder.embed_batch(contents)

        for chunk, embedding in zip(text_chunks, embeddings): 
            chunk["embedding"] = embedding
            chunk["payload"] = {
                "content": chunk["content"],
                "metadata":chunk["metadata"]
            }
        print (f"Storing chunks in Qdrant...")
        self.qdrant.upsert_chunks(
            collection_name=self.qdrant.TEXT_COLLECTION,
            chunks=text_chunks
        )

        print(f"Ingestion completed for doc_id: {doc_id}")

        return {
            "doc_id": doc_id, 
            "total_chunks": len(text_chunks),
            "text_chunks": len(text_chunks),
            "table_chunks": 0, # TODO: later
            "image_chunks": 0, # TODO: later
            "status": "completed"
        }
