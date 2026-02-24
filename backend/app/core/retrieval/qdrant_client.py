from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import uuid
from backend.app.config import get_settings

settings = get_settings()

class QdrantClientWrapper:
    """Qdrant client wrapper, manages 3 collections: text_chunks, table_chunks, image_chunks"""
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.embedding_dim = 1536

        self.TEXT_COLLECTION = "text_chunks"
        self.TABLE_COLLECTION = "table_chunks"
        self.IMAGE_COLLECTION = "image_chunks"

    def ensure_collections(self):
        collection = [
            self.TEXT_COLLECTION,
            self.TABLE_COLLECTION,
            self.IMAGE_COLLECTION
        ]

        existing = {col.name for col in self.client.get_collections().collections}
        for collection_name in collection: 
            if collection_name not in existing:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)

                )
                print(f"created collection: {collection_name}")
            else: 
                print(f"Collection already exists: {collection_name}")
    
    def upsert_chunks(
        self, 
        collection_name: str,
        chunks: List[Dict[str, Any]]
    ):
        """
        Upsert chunks into Qdrant

        Args:
            collection_name: "text_chunks", "table_chunks", or "image_chunks"
            chunks: List of dicts, for each dict: 
                id: str(uuid)
                vector: List[float] (1526 dims)
                payload: Dict (content+metadata)

        Example: 
            chunks = [
                {
                    "id": "uuid=123",
                    "vector": [0.1, 0.2, ...],
                    "payload": {
                        "content": "Revenue in Q4...",
                        "metadata": {
                            "doc_id": "uuid-abc",
                            "company": "Vinamilk",
                            "year": 2025
                        }
                    }
                }
            ]
        """
        points = [
            PointStruct(
                id=chunk["id"],
                vector=chunk["vector"],
                payload=chunk["payload"]
            )
            for chunk in chunks
        ]
        self.client.upsert(
            collection_name=collection_name,
            points = points
        )

        print(f"Upserted {len(chunks)} chunks to {collection_name}")
    
    def search(
        self, 
        collection_name: str,
        query_vector: List[float], 
        limit: int=5, 
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search chunks trong collection.

        Args: 
            collection_name: Collection for search
            query_vector: Query vector (1526 dims)
            limit: Number of results to return
            filters: Qdrant filter (optional)
                Example: {"company": "Vinamilk", "year": 2025}
        
        Returns: 
            List of dicts: score + payload
        """
        results = self.client.query_points(
            collection_name=collection_name, 
            query=query_vector, 
            limit=limit, 
            query_filter=filters
        ).points

        return [
            {
                "id": hit.id, 
                "score": hit.score, 
                "content":hit.payload.get("content"), 
                "metadata":hit.payload.get("metadata")
            }
            for hit in results
        ]

    def delete_by_doc_id(self, doc_id: str):
        """
        Delete all chunks belong to a document

        Args: 
            doc_id: Document ID to delete
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )

        for collection in [self.TEXT_COLLECTION, self.TABLE_COLLECTION, self.IMAGE_COLLECTION]:
            self.client.delete(
                collection_name=collection,
                points_selector=filter_condition
            )

        print(f"Deleted all chunks for doc_id: {doc_id}")
    def get_collection_info(self, collection_name: str) -> Dict:
        """ 
        Get collection info

        """
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count
        }


        
