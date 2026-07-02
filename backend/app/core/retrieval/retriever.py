from typing import Any, Dict, List, Optional
try:
    from langfuse import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator
from backend.app.core.retrieval.embedder import Embedder
from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper
from backend.app.core.retrieval.reranker import (
    get_cross_encoder_reranker,
    rerank_chunks,
)

class MultiCollectionRetriever: 
    """
    Search at the same time 3 collections: text, table, image. 
    Return all results labeled with chunk_type.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.qdrant = QdrantClientWrapper()

    @observe()
    def retrieve(
        self, 
        question: str, 
        top_k_text: int = 3, 
        top_k_table: int = 5,
        top_k_image: int = 2, 
        filters: Optional[Dict] = None, 
        reranker: str = "vector",
        reranker_model: Optional[str] = None,
        cross_encoder_top_n: int = 12,
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

        text_results = self._search_collection(
            collection_name=self.qdrant.TEXT_COLLECTION,
            query_vector=query_vector,
            limit=top_k_text,
            filters=qdrant_filters,
        )
        table_results = self._search_collection(
            collection_name=self.qdrant.TABLE_COLLECTION,
            query_vector=query_vector,
            limit=top_k_table,
            filters=qdrant_filters,
        )
        image_results = self._search_collection(
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

        if reranker == "heuristic":
            return rerank_chunks(question, all_results)

        if reranker == "cross_encoder":
            model_name = reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            preranked = rerank_chunks(question, all_results)
            model_candidates = preranked[:cross_encoder_top_n]
            remaining = preranked[cross_encoder_top_n:]
            reranked = get_cross_encoder_reranker(model_name).rerank(
                question,
                model_candidates,
                top_n=len(model_candidates),
                force=True,
            )
            return reranked + remaining

        return all_results

    def _search_collection(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        filters: Any,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            filters=filters,
        )
    
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
