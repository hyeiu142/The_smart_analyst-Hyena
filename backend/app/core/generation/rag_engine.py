import asyncio
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from backend.app.config import get_settings
from backend.app.core.retrieval.retriever import MultiCollectionRetriever
from backend.app.core.retrieval.embedder import Embedder
from backend.app.core.cache.semantic_cache import SemanticCache
from backend.app.core.generation.query_analyzer import QueryAnalyzer
from backend.app.core.generation.context_builder import ContextBuilder
from backend.app.core.ingestion.image_processor import ImageProcessor
from backend.app.core.observability.costs import estimate_openai_cost_usd
from backend.app.core.observability.logger import JsonlTraceLogger
from backend.app.core.observability.token_usage import openai_usage_to_dict
from backend.app.core.observability.trace import RAGTrace

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
        self.embedder = Embedder()
        self.analyzer = QueryAnalyzer()
        self.context_builder = ContextBuilder()
        self.image_processor = ImageProcessor()
        self.trace_logger = JsonlTraceLogger()
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
        top_k_text: Optional[int] = None,
        top_k_table: Optional[int] = None,
        top_k_image: Optional[int] = None,
        reranker: str = "cross_encoder",
        reranker_model: Optional[str] = None,
        cross_encoder_top_n: int = 12,
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
        trace = RAGTrace(question=question, mode="query")
        try:
            skip_cache = self._question_has_image_keywords(question)
            trace.set_metric("cache_skipped", skip_cache)

            # 0. Check semantic cache first
            with trace.step("cache_lookup"):
                if self.cache and not skip_cache:
                    cached = self.cache.get(question)
                    if cached:
                        trace.set_metric("cache_hit", True)
                        trace.finish("success")
                        return cached
                trace.set_metric("cache_hit", False)

            # 1. Analyze query
            with trace.step("analysis"):
                analysis = self.analyzer.analyze(question)
            logger.info(f"[RAG] Intent: {analysis.get('intent')}, Types: {analysis.get('data_types_needed')}")
            trace.set_metric("intent", analysis.get("intent"))
            trace.set_metric("data_types_needed", analysis.get("data_types_needed"))

            # 2. Build filters
            if filters is None:
                with trace.step("filter_build"):
                    filters = self.analyzer.build_filters(analysis) or None
            trace.set_metric("filters", filters)

            retrieval_top_k_text, retrieval_top_k_table, retrieval_top_k_image = (
                self._build_retrieval_allocation(
                    top_k=top_k,
                    top_k_text=top_k_text,
                    top_k_table=top_k_table,
                    top_k_image=top_k_image,
                )
            )
            trace.set_metric(
                "retrieval_config",
                {
                    "top_k": top_k,
                    "top_k_text": retrieval_top_k_text,
                    "top_k_table": retrieval_top_k_table,
                    "top_k_image": retrieval_top_k_image,
                    "reranker": reranker,
                    "reranker_model": reranker_model,
                    "cross_encoder_top_n": cross_encoder_top_n,
                },
            )

            # 3. Retrieval: explicit allocation is used for eval/experiments;
            # otherwise keep the previous wide retrieval behavior.
            with trace.step("retrieval"):
                chunks = self.retriever.retrieve(
                    question=question,
                    top_k_text=retrieval_top_k_text,
                    top_k_table=retrieval_top_k_table,
                    top_k_image=retrieval_top_k_image,
                    filters=filters,
                    reranker=reranker,
                    reranker_model=reranker_model,
                    cross_encoder_top_n=cross_encoder_top_n,
                )
            self._record_retrieval_metrics(trace, chunks, "retrieval")

            image_lazy_triggered = self._needs_image_analysis(analysis, question)
            trace.set_metric("image_lazy_triggered", image_lazy_triggered)
            if image_lazy_triggered:
                with trace.step("lazy_image"):
                    described_count = await asyncio.to_thread(self._describe_pending_images, chunks, 2)
                trace.set_metric("images_described", described_count)
                if described_count:
                    logger.info(f"[RAG] Lazily described {described_count} image chunks; retrieving again")
                    with trace.step("retrieval_after_image"):
                        chunks = self.retriever.retrieve(
                            question=question,
                            top_k_text=retrieval_top_k_text,
                            top_k_table=retrieval_top_k_table,
                            top_k_image=max(retrieval_top_k_image, top_k),
                            filters=filters,
                            reranker=reranker,
                            reranker_model=reranker_model,
                            cross_encoder_top_n=cross_encoder_top_n,
                        )
                    self._record_retrieval_metrics(trace, chunks, "retrieval_after_image")
            else:
                trace.set_metric("images_described", 0)

            if not chunks:
                trace.finish("success")
                return {
                    "answer": "Không tìm thấy thông tin liên quan trong tài liệu.",
                    "sources": [],
                    "analysis": analysis,
                }

            image_candidates = self._get_usable_image_candidates(chunks)

            # 4. Select final context chunks. retrieve() already applies the
            # requested reranker mode, so avoid a second rerank here.
            with trace.step("rerank"):
                chunks = chunks[:top_k]
            logger.info(f"[RAG] After context selection: {len(chunks)} chunks")
            self._record_retrieval_metrics(trace, chunks, "rerank")
            chunks = self._ensure_image_context(chunks, image_candidates, image_lazy_triggered, trace)
            self._record_retrieval_metrics(trace, chunks, "context_selection")
            self._record_selected_image_metrics(trace, chunks)

            # 5. Build context
            with trace.step("context_build"):
                context = self.context_builder.build(chunks)
                citations = self.context_builder.build_citations(chunks)
            trace.set_metric("context_chars", len(context))
            trace.set_metric("citations_count", len(citations))

            # 6. LLM synthesis
            with trace.step("generation"):
                answer, usage = self._synthesize_with_usage(question, context)
            trace.add_tokens("generation", usage)
            estimated_cost = estimate_openai_cost_usd(
                settings.llm_model,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            trace.add_cost("estimated_usd", estimated_cost)

            result = {
                "answer": answer,
                "sources": citations,
                "analysis": analysis,
            }

            # 7. Store in semantic cache for future queries
            if self.cache and not skip_cache:
                with trace.step("cache_set"):
                    self.cache.set(question, result)

            trace.finish("success")
            return result
        except Exception as exc:
            trace.finish("error", str(exc))
            raise
        finally:
            self.trace_logger.write(trace.to_dict())

    def _needs_image_analysis(self, analysis: Dict[str, Any], question: str) -> bool:
        data_types = analysis.get("data_types_needed") or []
        if "image" in {str(item).lower() for item in data_types}:
            return True

        return self._question_has_image_keywords(question)

    def _question_has_image_keywords(self, question: str) -> bool:
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
            "diễn biến giá cổ phiếu",
            "vn-index",
        ]
        return any(keyword in normalized for keyword in image_keywords)

    def _build_retrieval_allocation(
        self,
        *,
        top_k: int,
        top_k_text: Optional[int],
        top_k_table: Optional[int],
        top_k_image: Optional[int],
    ) -> tuple[int, int, int]:
        if (
            top_k_text is not None
            or top_k_table is not None
            or top_k_image is not None
        ):
            return (
                max(0, top_k_text or 0),
                max(0, top_k_table or 0),
                max(0, top_k_image or 0),
            )

        wide_k = max(top_k * 4, 20)
        return wide_k // 3, wide_k // 2, wide_k // 6

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

    def _record_retrieval_metrics(self, trace: RAGTrace, chunks: List[Dict[str, Any]], prefix: str) -> None:
        source_counts = {"text": 0, "table": 0, "image": 0}
        top_scores = []
        for chunk in chunks:
            source = chunk.get("source_collection", "text")
            if source in source_counts:
                source_counts[source] += 1
            if "score" in chunk:
                top_scores.append(round(float(chunk["score"]), 4))

        trace.set_metric(f"{prefix}_chunks", len(chunks))
        trace.set_metric(f"{prefix}_text_hits", source_counts["text"])
        trace.set_metric(f"{prefix}_table_hits", source_counts["table"])
        trace.set_metric(f"{prefix}_image_hits", source_counts["image"])
        trace.set_metric(f"{prefix}_top_scores", top_scores[:5])

    def _get_usable_image_candidates(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = [
            chunk
            for chunk in chunks
            if chunk.get("source_collection") == "image"
            and self._has_image_caption(chunk)
        ]
        return sorted(candidates, key=lambda chunk: float(chunk.get("score") or 0), reverse=True)

    def _has_image_caption(self, chunk: Dict[str, Any]) -> bool:
        metadata = chunk.get("metadata") or {}
        content = (chunk.get("content") or "").strip()
        if not content:
            return False
        if metadata.get("image_status") == "pending" or metadata.get("chunk_type") == "image_pending":
            return False
        if content.lower().startswith("pending chart/image crop"):
            return False
        return True

    def _ensure_image_context(
        self,
        chunks: List[Dict[str, Any]],
        image_candidates: List[Dict[str, Any]],
        needs_image: bool,
        trace: RAGTrace,
    ) -> List[Dict[str, Any]]:
        if not needs_image:
            trace.set_metric("image_forced_into_context", False)
            return chunks

        selected_images = [
            chunk
            for chunk in chunks
            if chunk.get("source_collection") == "image"
            and self._has_image_caption(chunk)
        ]
        selected_ids = {chunk.get("id") for chunk in selected_images}

        if not image_candidates:
            trace.set_metric("image_forced_into_context", False)
            trace.set_metric("image_force_reason", "no_usable_image_caption")
            return chunks

        added_images = []
        for candidate in image_candidates:
            candidate_id = candidate.get("id")
            if candidate_id in selected_ids:
                continue
            selected_images.append(candidate)
            added_images.append(candidate)
            selected_ids.add(candidate_id)
            if len(selected_images) >= 2:
                break

        trace.set_metric("image_forced_into_context", bool(added_images))
        if added_images:
            trace.set_metric(
                "forced_image_scores",
                [round(float(chunk.get("score") or 0), 4) for chunk in added_images],
            )

        non_image_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("source_collection") != "image"
        ]
        return selected_images + non_image_chunks

    def _record_selected_image_metrics(self, trace: RAGTrace, chunks: List[Dict[str, Any]]) -> None:
        selected_images = [
            chunk
            for chunk in chunks
            if chunk.get("source_collection") == "image"
        ]
        trace.set_metric("selected_image_count", len(selected_images))
        trace.set_metric(
            "selected_image_paths",
            [self._image_path_for_log(chunk) for chunk in selected_images],
        )

    def _image_path_for_log(self, chunk: Dict[str, Any]) -> str | None:
        metadata = chunk.get("metadata") or {}
        return metadata.get("image_path") or metadata.get("local_image_path")

    def _synthesize(self, question: str, context: str) -> str:
        """Gọi LLM để generate answer từ context."""
        answer, _usage = self._synthesize_with_usage(question, context)
        return answer

    def _synthesize_with_usage(self, question: str, context: str) -> tuple[str, Dict[str, int]]:
        """Gọi LLM để generate answer từ context, kèm token usage."""
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
        return response.choices[0].message.content, openai_usage_to_dict(response)

    def _build_context(self, chunks: List[Dict]) -> str:
        """Legacy method — giữ để backward compat."""
        return self.context_builder.build(chunks)

    async def stream_query(
        self,
        question: str,
        top_k: int = 5,
        top_k_text: Optional[int] = None,
        top_k_table: Optional[int] = None,
        top_k_image: Optional[int] = None,
        reranker: str = "cross_encoder",
        reranker_model: Optional[str] = None,
        cross_encoder_top_n: int = 12,
        filters: Optional[Dict] = None,
    ):
        """
        Streaming version — yield từng token để dùng với SSE.
        """
        from typing import AsyncGenerator

        trace = RAGTrace(question=question, mode="query_stream")
        try:
            with trace.step("analysis"):
                analysis = self.analyzer.analyze(question)
            trace.set_metric("intent", analysis.get("intent"))
            trace.set_metric("data_types_needed", analysis.get("data_types_needed"))

            if filters is None:
                with trace.step("filter_build"):
                    filters = self.analyzer.build_filters(analysis) or None
            trace.set_metric("filters", filters)

            retrieval_top_k_text, retrieval_top_k_table, retrieval_top_k_image = (
                self._build_retrieval_allocation(
                    top_k=top_k,
                    top_k_text=top_k_text,
                    top_k_table=top_k_table,
                    top_k_image=top_k_image,
                )
            )
            trace.set_metric(
                "retrieval_config",
                {
                    "top_k": top_k,
                    "top_k_text": retrieval_top_k_text,
                    "top_k_table": retrieval_top_k_table,
                    "top_k_image": retrieval_top_k_image,
                    "reranker": reranker,
                    "reranker_model": reranker_model,
                    "cross_encoder_top_n": cross_encoder_top_n,
                },
            )

            with trace.step("retrieval"):
                chunks = self.retriever.retrieve(
                    question=question,
                    top_k_text=retrieval_top_k_text,
                    top_k_table=retrieval_top_k_table,
                    top_k_image=retrieval_top_k_image,
                    filters=filters,
                    reranker=reranker,
                    reranker_model=reranker_model,
                    cross_encoder_top_n=cross_encoder_top_n,
                )
            self._record_retrieval_metrics(trace, chunks, "retrieval")

            image_lazy_triggered = self._needs_image_analysis(analysis, question)
            trace.set_metric("image_lazy_triggered", image_lazy_triggered)
            if image_lazy_triggered:
                with trace.step("lazy_image"):
                    described_count = await asyncio.to_thread(self._describe_pending_images, chunks, 2)
                trace.set_metric("images_described", described_count)
                if described_count:
                    logger.info(f"[RAG] Lazily described {described_count} image chunks; retrieving again")
                    with trace.step("retrieval_after_image"):
                        chunks = self.retriever.retrieve(
                            question=question,
                            top_k_text=retrieval_top_k_text,
                            top_k_table=retrieval_top_k_table,
                            top_k_image=max(retrieval_top_k_image, top_k),
                            filters=filters,
                            reranker=reranker,
                            reranker_model=reranker_model,
                            cross_encoder_top_n=cross_encoder_top_n,
                        )
                    self._record_retrieval_metrics(trace, chunks, "retrieval_after_image")
            else:
                trace.set_metric("images_described", 0)

            if not chunks:
                trace.finish("success")
                yield "Không tìm thấy thông tin liên quan trong tài liệu."
                return

            image_candidates = self._get_usable_image_candidates(chunks)

            with trace.step("rerank"):
                chunks = chunks[:top_k]
            self._record_retrieval_metrics(trace, chunks, "rerank")
            chunks = self._ensure_image_context(chunks, image_candidates, image_lazy_triggered, trace)
            self._record_retrieval_metrics(trace, chunks, "context_selection")
            self._record_selected_image_metrics(trace, chunks)

            with trace.step("context_build"):
                context = self.context_builder.build(chunks)
                citations = self.context_builder.build_citations(chunks)
            trace.set_metric("context_chars", len(context))
            trace.set_metric("citations_count", len(citations))

            user_message = f"""Context from financial documents:
{context}

---
Question: {question}

Please answer based on the context above. Cite sources using [Source #N] format."""

            with trace.step("generation_stream"):
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
                        yield {"token": delta}
            yield {"sources": citations}
            trace.finish("success")
        except Exception as exc:
            trace.finish("error", str(exc))
            raise
        finally:
            self.trace_logger.write(trace.to_dict())
