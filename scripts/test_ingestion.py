import asyncio
import sys
sys.path.append("/home/yennguyen/Hyena")

from backend.app.core.ingestion.pipeline import IngestionPipeline
from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper
import uuid

async def main():
    print("Initializing Qdrant collections...")
    qdrant = QdrantClientWrapper()
    qdrant.ensure_collections()

    pipeline = IngestionPipeline()

    doc_id = str(uuid.uuid4())
    result = await pipeline.process_document(
        file_path="/home/yennguyen/Hyena/Docs/20260130_VNM_IR_Newsletter_Q4_2025_68b565f20d.pdf",
        doc_id = doc_id,
        metadata={
            "company": "Vinamilk",
            "year": 2025,
            "quarter": "Q4",
            "original_filename": "Docs/20260130_VNM_IR_Newsletter_Q4_2025_68b565f20d.pdf"
        }
    )
    print("\nResult:")
    print(result)

    info = qdrant.get_collection_info("text_chunks")
    print(f"\nQdrant text_chunks: {info['points_count']} points")

if __name__ == "__main__":
    asyncio.run(main())