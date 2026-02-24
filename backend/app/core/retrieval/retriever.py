from typing import Any, Dict, List, Optional
from backend.app.core.retrieval.embedder import Embedder
from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper

class MultiCollectionRetriever: 
    """
    Search at the same time 3 collections: text, table, image. 
    Return all results labeled with chunk_type.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.qdrant = QdrantClientWrapper()

    def retrieve(
        self, 
        question: str, 
        top_k_text: int = 3, 
        top_k_table: int = 5,
        top_k_image: int = 2, 
        filters: Optional[Dict] = None, 

    ) -> List[Dict[str, Any]]:
        """
        Search 3 collections at the same time, merge all results
        
        Args: 
            question: user question
            top_k_text: number of text chunks to return
            top_k_table: number of table chunks to return
            top_k_image: number of image chunks to return
            filters: optional filters for metadata (e.g. company, year, quarter)

        Returns: 
            List chunks sorted by score descending.
            Each chunk has field "chunk_type" to indicate.
        """

        query_vector = self.embedder.embed_documents(question)
        qdrant_filters = self._build_filter(filters) if filters else None

        text_results = self.qdrant.search(
            collection_name=self.qdrant.TEXT_COLLECTION,
            query_vector=query_vector,
            limit=top_k_text,
            filters=qdrant_filters,
        )
        table_results = self.qdrant.search(
            collection_name=self.qdrant.TABLE_COLLECTION,
            query_vector=query_vector,
            limit=top_k_table,
            filters=qdrant_filters,
        )
        image_results = self.qdrant.search(
            collection_name=self.qdrant.IMAGE_COLLECTION,
            query_vector=query_vector,
            limit=top_k_image,
            filters=qdrant_filters,
        )

        for r in text_results:
            r["source_collection"] = "text"
        for r in table_results:
            r["source_collection"] = "table"
        for r in image_results:
            r["source_collection"] = "image"

        all_results = text_results + table_results + image_results
        all_results.sort(key=lambda x: x["score"], reverse=True)

        return all_results
    
    def _build_filter(self, filters: Dict) -> Dict:
        """
        Convert dict đơn giản sang Qdrant Filter object.

        Input:  {"company": "Vinamilk", "year": 2025}
        Output: Qdrant Filter với FieldCondition
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                )
            )
        return Filter(must=conditions)


