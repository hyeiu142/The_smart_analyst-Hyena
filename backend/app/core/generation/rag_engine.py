import asyncio
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from backend.app.config import get_settings
from backend.app.core.retrieval.retriever import MultiCollectionRetriever
from backend.app.core.retrieval.reranker import CrossEncoderReranker
from backend.app.core.retrieval.embedder import Embedder
from backend.app.core.cache.semantic_cache import SemanticCache
from backend.app.core.generation.query_analyzer import QueryAnalyzer
from backend.app.core.generation.context_builder import ContextBuilder
from backend.app.core.ingestion.image_processor import ImageProcessor

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """
You are an expert financial analyst assistant.
Answer questions based ONLY on the provided context from financial documents.
Always cite your sources using [Source #N] format.
If the context doesn't contain enough information, say so clearly.
Be precise with numbers and percentages.
Respond in the same language as the user's question.
"""


class RAGEngine:
    """
    Agentic RAG Engine — multi-collection, with query analysis.

    Flow:
    1. QueryAnalyzer: detect intent, extract entities
    2. MultiCollectionRetriever: search text + table + image collections
    3. ContextBuilder: format context
    4. LLM: synthesize answer with citations
    """

    def __init__(self):
        self.llm = OpenAI(api_key=settings.openai_api_key)
        self.retriever = MultiCollectionRetriever()
        self.reranker = CrossEncoderReranker()  # lazy-loads on first query
        self.embedder = Embedder()
        self.analyzer = QueryAnalyzer()
        self.context_builder = ContextBuilder()
        self.image_processor = ImageProcessor()
        self.cache = self._init_cache()

    def _init_cache(self) -> Optional[SemanticCache]:
        """Initialize Redis-backed semantic cache. Returns None if Redis not available."""
        try:
            import redis
            r = redis.from_url(settings.redis_url, decode_responses=True)
            r.ping()  # Test connection
            cache = SemanticCache(r, self.embedder)
            logger.info("[RAGEngine] Semantic cache enabled.")
            return cache
        except Exception as e:
            logger.warning(f"[RAGEngine] Cache disabled (Redis unavailable): {e}")
            return None

    async def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Main RAG query với multi-collection support.

        Args:
            question: Câu hỏi user
            top_k: Tổng số chunks lấy (phân bổ: text=3, table=top_k, image=2)
            filters: Override filters (nếu None, tự động detect từ query)

        Returns:
            {
                "answer": "...",
                "sources": [...],
                "analysis": {...}  # query analysis result
            }
        """
        # 0. Check semantic cache first
        if self.cache:
            cached = self.cache.get(question)
            if cached:
                return cached

        # 1. Analyze query
        analysis = self.analyzer.analyze(question)
        logger.info(f"[RAG] Intent: {analysis.get('intent')}, Types: {analysis.get('data_types_needed')}")

        # 2. Build filters
        if filters is None:
            filters = self.analyzer.build_filters(analysis) or None

        # 3. Wide retrieval: cast a bigger net (top_k * 4)
        wide_k = max(top_k * 4, 20)
        chunks = self.retriever.retrieve(
            question=question,
            top_k_text=wide_k // 3,
            top_k_table=wide_k // 2,
            top_k_image=wide_k // 6,
            filters=filters,
        )

        if self._needs_image_analysis(analysis, question):
            described_count = await asyncio.to_thread(self._describe_pending_images, chunks, 2)
            if described_count:
                logger.info(f"[RAG] Lazily described {described_count} image chunks; retrieving again")
                chunks = self.retriever.retrieve(
                    question=question,
                    top_k_text=wide_k // 3,
                    top_k_table=wide_k // 2,
                    top_k_image=wide_k // 3,
                    filters=filters,
                )

        if not chunks:
            return {
                "answer": "Không tìm thấy thông tin liên quan trong tài liệu.",
                "sources": [],
                "analysis": analysis,
            }

        # 4. Rerank: cross-encoder picks the best top_k from the wide set
        # Run in thread to avoid blocking the async event loop (model.predict is CPU-bound)
        chunks = await asyncio.to_thread(self.reranker.rerank, question, chunks, top_k)
        logger.info(f"[RAG] After rerank: {len(chunks)} chunks")

        # 5. Build context
        context = self.context_builder.build(chunks)
        citations = self.context_builder.build_citations(chunks)

        # 6. LLM synthesis
        answer = self._synthesize(question, context)

        result = {
            "answer": answer,
            "sources": citations,
            "analysis": analysis,
        }

        # 7. Store in semantic cache for future queries
        if self.cache:
            self.cache.set(question, result)

        return result

    def _needs_image_analysis(self, analysis: Dict[str, Any], question: str) -> bool:
        data_types = analysis.get("data_types_needed") or []
        if "image" in {str(item).lower() for item in data_types}:
            return True

        normalized = question.lower()
        image_keywords = [
            "biểu đồ",
            "chart",
            "hình",
            "figure",
            "cơ cấu",
            "yoy",
            "theo khối",
            "thị trường",
        ]
        return any(keyword in normalized for keyword in image_keywords)

    def _describe_pending_images(self, chunks: List[Dict[str, Any]], max_images: int = 2) -> int:
        pending_images = [
            chunk
            for chunk in chunks
            if chunk.get("source_collection") == "image"
            and (chunk.get("metadata") or {}).get("image_status") == "pending"
        ]
        if not pending_images:
            return 0

        updated_chunks: List[Dict[str, Any]] = []
        for chunk in pending_images[:max_images]:
            described = self.image_processor.describe_chunk(chunk)
            if not described:
                continue
            described["vector"] = self.embedder.embed_documents(described["content"])
            described["payload"] = {
                "content": described["content"],
                "metadata": described["metadata"],
            }
            updated_chunks.append(described)

        if updated_chunks:
            self.retriever.qdrant.upsert_chunks(self.retriever.qdrant.IMAGE_COLLECTION, updated_chunks)

        return len(updated_chunks)

    def _synthesize(self, question: str, context: str) -> str:
        """Gọi LLM để generate answer từ context."""
        user_message = f"""Context from financial documents:
{context}

---
Question: {question}

Please answer based on the context above. Cite sources using [Source #N] format."""

        response = self.llm.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    def _build_context(self, chunks: List[Dict]) -> str:
        """Legacy method — giữ để backward compat."""
        return self.context_builder.build(chunks)

    async def stream_query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ):
        """
        Streaming version — yield từng token để dùng với SSE.
        """
        from typing import AsyncGenerator

        analysis = self.analyzer.analyze(question)
        if filters is None:
            filters = self.analyzer.build_filters(analysis) or None

        # Wide retrieval
        wide_k = max(top_k * 4, 20)
        chunks = self.retriever.retrieve(
            question=question,
            top_k_text=wide_k // 3,
            top_k_table=wide_k // 2,
            top_k_image=wide_k // 6,
            filters=filters,
        )

        if self._needs_image_analysis(analysis, question):
            described_count = await asyncio.to_thread(self._describe_pending_images, chunks, 2)
            if described_count:
                logger.info(f"[RAG] Lazily described {described_count} image chunks; retrieving again")
                chunks = self.retriever.retrieve(
                    question=question,
                    top_k_text=wide_k // 3,
                    top_k_table=wide_k // 2,
                    top_k_image=wide_k // 3,
                    filters=filters,
                )

        if not chunks:
            yield "Không tìm thấy thông tin liên quan trong tài liệu."
            return

        # Rerank (run in thread to avoid blocking event loop)
        chunks = await asyncio.to_thread(self.reranker.rerank, question, chunks, top_k)

        context = self.context_builder.build(chunks)
        user_message = f"""Context from financial documents:
{context}

---
Question: {question}

Please answer based on the context above. Cite sources using [Source #N] format."""

        stream = self.llm.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1500,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
