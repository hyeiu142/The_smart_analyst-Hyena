from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from typing import List, Dict, Any
import uuid
from backend.app.config import get_settings

settings = get_settings()

class TextProcessor: 
    """
    Text processor from PDF: 
    1. Parse PDF with LlamaParse
    2. Chunk text with hierarchy
    3. 
    """
    def __init__(self):
        self.parser = LlamaParse(
            api_key=settings.llama_cloud_api_key, 
            result_type="markdown",
            language="en",
            verbose=True
        )
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

    async def process(
        self,
        file_path: str, 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process PDF to text chunk

        Args: 
            file_path: Path to PDF file
            metadata: Dict content company, year, etc
        Returns: 
            List of text chunks (without embedding)
            [
                {
                    "id": "uuid-123",
                    "content": "Revenue in Q4...",
                    "metadata": {
                        "doc_id": metadata["doc_id"],
                        "company": "Vinamilk",
                        "year": 2025,
                        "page": 1,
                        "chunk_type": "text"
                    }
                }
            ]
        """
        documents = await self.parser.aload_data(file_path)

        chunks = []

        for doc in documents: 
            nodes = self.splitter.get_nodes_from_documents([doc])
            for node in nodes: 
                chunk = {
                    "id": str(uuid.uuid4()), 
                    "content":node.text,
                    "metadata": {
                        **metadata, #doc_id, company, year, quarter
                        "page": node.metadata.get("page",1), 
                        "chunk_type": "text"
                    }
                }
                chunks.append(chunk)
        print(f"Processed {len(chunks)} text chunks")
        return chunks