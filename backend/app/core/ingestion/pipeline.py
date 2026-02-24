from typing import Any, Dict

from backend.app.core.ingestion.text_processor import TextProcessor
from backend.app.core.ingestion.table_processor import TableProcessor
from backend.app.core.ingestion.image_processor import ImageProcessor
from backend.app.core.retrieval.embedder import Embedder
from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper


class IngestionPipeline:
    """
    Orchestrator cho toàn bộ ingestion pipeline.

    Flow:
    1. Parse PDF → text chunks + table chunks + image chunks (song song)
    2. Embed tất cả chunks
    3. Upsert vào các Qdrant collections tương ứng
    """

    def __init__(self):
        self.text_processor = TextProcessor()
        self.table_processor = TableProcessor()
        self.image_processor = ImageProcessor()
        self.embedder = Embedder()
        self.qdrant = QdrantClientWrapper()

        # Đảm bảo collections tồn tại
        self.qdrant.ensure_collections()

    async def process_document(
        self,
        file_path: str,
        doc_id: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process 1 document hoàn chỉnh qua 3 pipelines.

        Args:
            file_path: Đường dẫn file PDF
            doc_id: UUID document
            metadata: {company, year, quarter, original_filename}

        Returns:
            {
                "doc_id": "uuid",
                "total_chunks": 80,
                "text_chunks": 45,
                "table_chunks": 20,
                "image_chunks": 15,
                "status": "completed"
            }
        """
        metadata["doc_id"] = doc_id
        print(f"\n[Pipeline] Starting ingestion for doc_id={doc_id}")

        # ── 1. Extract ──────────────────────────────────────────────
        print("[Pipeline] Step 1/3: Extracting content...")
        text_chunks = await self.text_processor.process(file_path, metadata)
        table_chunks = await self.table_processor.process(file_path, metadata)
        image_chunks = await self.image_processor.process(file_path, metadata)

        print(f"[Pipeline] Extracted: {len(text_chunks)} text, {len(table_chunks)} tables, {len(image_chunks)} images")

        # ── 2. Embed ────────────────────────────────────────────────
        print("[Pipeline] Step 2/3: Embedding chunks...")

        def _embed_and_attach(chunks):
            if not chunks:
                return
            contents = [c["content"] for c in chunks]
            embeddings = self.embedder.embed_batch(contents)
            for chunk, vector in zip(chunks, embeddings):
                chunk["vector"] = vector
                chunk["payload"] = {
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                }

        _embed_and_attach(text_chunks)
        _embed_and_attach(table_chunks)
        _embed_and_attach(image_chunks)

        # ── 3. Store ────────────────────────────────────────────────
        print("[Pipeline] Step 3/3: Storing to Qdrant...")
        if text_chunks:
            self.qdrant.upsert_chunks(self.qdrant.TEXT_COLLECTION, text_chunks)
        if table_chunks:
            self.qdrant.upsert_chunks(self.qdrant.TABLE_COLLECTION, table_chunks)
        if image_chunks:
            self.qdrant.upsert_chunks(self.qdrant.IMAGE_COLLECTION, image_chunks)

        total = len(text_chunks) + len(table_chunks) + len(image_chunks)
        print(f"[Pipeline] Completed! Total chunks stored: {total}")

        return {
            "doc_id": doc_id,
            "total_chunks": total,
            "text_chunks": len(text_chunks),
            "table_chunks": len(table_chunks),
            "image_chunks": len(image_chunks),
            "status": "completed",
        }