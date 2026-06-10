"""
Cross-Encoder Reranker for Advanced RAG.

Flow:
  Qdrant (Bi-Encoder, top_k=20)
      └─> Reranker (Cross-Encoder, pick top_n=5)
               └─> LLM

Model: BAAI/bge-reranker-v2-m3 (state-of-the-art, multilingual, ~1.1GB)
- Supports Vietnamese, English financial text
- First run: auto-downloads from HuggingFace (~2 min)
- Subsequent runs: loads from local cache instantly
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a Cross-Encoder model.

    Differences from Bi-Encoder (embedding):
    - Bi-Encoder:   embeds query & doc SEPARATELY → fast, but less precise
    - Cross-Encoder: sees query + doc TOGETHER  → slow, but much more precise

    Use Cross-Encoder only on the short-list (top 20) to keep latency low.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        # ↑ ~80MB, ~1-2s on CPU — fast enough for production without GPU
        # For higher accuracy with GPU: use "BAAI/bge-reranker-v2-m3" (~1.1GB)
    ):
        self.model_name = model_name
        self._model = None  # lazy load

    def _load_model(self):
        """Lazy-load model on first use (avoid startup cost)."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"[Reranker] Loading {self.model_name}...")
                self._model = CrossEncoder(self.model_name)
                logger.info("[Reranker] Model ready.")
            except Exception as e:
                logger.error(f"[Reranker] Failed to load model: {e}")
                raise

    def _deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks by content hash (keeps first/highest-score occurrence)."""
        seen = set()
        unique = []
        for chunk in chunks:
            content = chunk.get("content", "")
            # Use first 200 chars as fingerprint (avoids minor whitespace differences)
            fingerprint = content[:200].strip()
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(chunk)
        return unique

    def _normalize_scores(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize reranker scores to [0, 1] range for consistent display."""
        scores = [c["reranker_score"] for c in chunks]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            for c in chunks:
                c["score"] = 1.0
        else:
            for c in chunks:
                c["score"] = round((c["reranker_score"] - min_s) / (max_s - min_s), 4)
        return chunks

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks by cross-encoder score, return top_n best.

        Args:
            query:  The user's question
            chunks: List of chunk dicts from Qdrant (must have 'content' key)
            top_n:  How many top chunks to return after reranking

        Returns:
            List of top_n chunks, sorted by reranker score (descending).
            Scores are normalized to [0,1] for consistent display.
        """
        if not chunks:
            return []

        # Step 1: Deduplicate (same chunk may come from text + table collection)
        chunks = self._deduplicate(chunks)

        if len(chunks) <= top_n:
            return chunks

        self._load_model()

        # Step 2: Cross-encoder scoring
        pairs = [(query, chunk.get("content", "")) for chunk in chunks]
        scores = self._model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)
            chunk["qdrant_score"] = chunk.get("score", 0.0)

        # Step 3: Sort and take top_n
        reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)
        top = reranked[:top_n]

        # Step 4: Normalize scores to [0,1] for display
        top = self._normalize_scores(top)

        logger.info(
            f"[Reranker] {len(chunks)} → {len(top)} chunks (after dedup). "
            f"Best score: {top[0]['score']:.4f}"
        )
        return top
