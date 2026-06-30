"""
Rerankers for Hyena retrieval.

The RAG answer path keeps the original CrossEncoderReranker. The evaluation
`/query/similar` path also uses a lightweight deterministic reranker so we can
run retrieval experiments without adding another model dependency.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a Cross-Encoder model.

    Differences from Bi-Encoder (embedding):
    - Bi-Encoder:   embeds query & doc SEPARATELY → fast, but less precise
    - Cross-Encoder: sees query + doc TOGETHER  → slow, but much more precise

    Use Cross-Encoder only on the short-list to keep latency low.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
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
        """Remove duplicate chunks by content fingerprint."""
        seen = set()
        unique = []
        for chunk in chunks:
            content = chunk.get("content", "")
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
                c["score"] = round(
                    (c["reranker_score"] - min_s) / (max_s - min_s),
                    4,
                )
        return chunks

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 5,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks by cross-encoder score, return top_n best.
        """
        if not chunks:
            return []

        chunks = self._deduplicate(chunks)
        if len(chunks) <= top_n and not force:
            return chunks

        self._load_model()

        pairs = [(query, chunk.get("content", "")) for chunk in chunks]
        scores = self._model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)
            chunk["qdrant_score"] = chunk.get("score", 0.0)

        reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)
        top = self._normalize_scores(reranked[:top_n])

        logger.info(
            f"[Reranker] {len(chunks)} → {len(top)} chunks (after dedup). "
            f"Best score: {top[0]['score']:.4f}"
        )
        return top


@lru_cache(maxsize=4)
def get_cross_encoder_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)


STOPWORDS = {
    "bao",
    "các",
    "của",
    "cho",
    "đến",
    "được",
    "gì",
    "là",
    "năm",
    "những",
    "theo",
    "trong",
    "và",
    "về",
}

TABLE_INTENT_TERMS = {
    "doanh thu",
    "lợi nhuận",
    "lntt",
    "lnst",
    "tài sản",
    "vcsh",
    "vốn chủ",
    "biên",
    "giá vốn",
    "dòng tiền",
}

IMAGE_INTENT_TERMS = {
    "biểu đồ",
    "chart",
    "hình",
    "ảnh",
    "cơ cấu",
    "tỷ trọng",
    "thị trường",
}

NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*%?")
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_tokens(value: Any) -> set[str]:
    tokens = set()
    for token in TOKEN_PATTERN.findall(normalize_text(value)):
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.add(token)
    return tokens


def normalize_number(value: str) -> str:
    return normalize_text(value).replace(",", ".").replace("%", "")


def extract_numbers(value: Any) -> set[str]:
    numbers = set()
    for number in NUMBER_PATTERN.findall(normalize_text(value)):
        normalized = normalize_number(number)
        if normalized:
            numbers.add(normalized)
    return numbers


def get_chunk_type(chunk: dict[str, Any]) -> str:
    source_collection = chunk.get("source_collection")
    if source_collection:
        return normalize_text(source_collection)

    metadata = chunk.get("metadata") or {}
    chunk_type = metadata.get("chunk_type")
    if chunk_type == "image_caption":
        return "image"
    return normalize_text(chunk_type)


def get_chunk_page(chunk: dict[str, Any]) -> int | None:
    metadata = chunk.get("metadata") or {}
    page = metadata.get("page")
    if page is None:
        page = metadata.get("page_num")
    if page is None:
        return None

    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def has_any_phrase(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def score_chunk(
    chunk: dict[str, Any],
    *,
    query_text: str,
    query_tokens: set[str],
    query_numbers: set[str],
    wants_table: bool,
    wants_image: bool,
    page_context_scores: dict[int, float],
) -> float:
    chunk_type = get_chunk_type(chunk)
    page = get_chunk_page(chunk)
    content = normalize_text(chunk.get("content") or "")
    score = float(chunk.get("score") or 0.0)

    content_tokens = extract_tokens(content)
    content_numbers = extract_numbers(content)

    token_overlap = query_tokens & content_tokens
    number_overlap = query_numbers & content_numbers

    rerank_score = score
    rerank_score += min(len(token_overlap), 8) * 0.025
    rerank_score += min(len(number_overlap), 4) * 0.08

    if chunk_type == "table" and wants_table:
        rerank_score += 0.18
    if chunk_type == "image" and wants_image:
        rerank_score += 0.35
    if chunk_type == "image" and not wants_image:
        rerank_score -= 0.12

    # Image chunks are currently generic "pending image" captions. Use strong
    # text/table candidates on the same page as a proxy signal for image rank.
    if chunk_type == "image" and page in page_context_scores:
        rerank_score += page_context_scores[page] * 0.55

    if chunk_type == "table" and wants_image:
        rerank_score += 0.05

    return rerank_score


def rerank_chunks(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_text = normalize_text(question)
    query_tokens = extract_tokens(query_text)
    query_numbers = extract_numbers(query_text)
    wants_table = has_any_phrase(query_text, TABLE_INTENT_TERMS) or bool(query_numbers)
    wants_image = has_any_phrase(query_text, IMAGE_INTENT_TERMS)

    page_context_scores: dict[int, float] = {}
    for chunk in chunks:
        chunk_type = get_chunk_type(chunk)
        if chunk_type == "image":
            continue

        page = get_chunk_page(chunk)
        if page is None:
            continue

        page_context_scores[page] = max(
            page_context_scores.get(page, 0.0),
            float(chunk.get("score") or 0.0),
        )

    reranked = []
    for index, chunk in enumerate(chunks):
        rerank_score = score_chunk(
            chunk,
            query_text=query_text,
            query_tokens=query_tokens,
            query_numbers=query_numbers,
            wants_table=wants_table,
            wants_image=wants_image,
            page_context_scores=page_context_scores,
        )
        updated = dict(chunk)
        updated["rerank_score"] = round(rerank_score, 6)
        updated["_original_rank"] = index
        reranked.append(updated)

    reranked.sort(
        key=lambda item: (
            item["rerank_score"],
            float(item.get("score") or 0.0),
            -int(item.get("_original_rank") or 0),
        ),
        reverse=True,
    )

    for item in reranked:
        item.pop("_original_rank", None)

    return reranked
