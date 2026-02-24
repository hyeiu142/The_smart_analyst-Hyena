from typing import Any, Dict, List, Optional

from openai import OpenAI

from backend.app.config import get_settings
from backend.app.core.retrieval.retriever import MultiCollectionRetriever
from backend.app.core.generation.query_analyzer import QueryAnalyzer
from backend.app.core.generation.context_builder import ContextBuilder

settings = get_settings()

SYSTEM_PROMPT = """You are an expert financial analyst assistant.
Answer questions based ONLY on the provided context from financial documents.
Always cite your sources using [Source #N] format.
If the context doesn't contain enough information, say so clearly.
Be precise with numbers and percentages.
Respond in the same language as the user's question."""


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
        self.analyzer = QueryAnalyzer()
        self.context_builder = ContextBuilder()

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
        # 1. Analyze query
        analysis = self.analyzer.analyze(question)
        print(f"[RAG] Intent: {analysis.get('intent')}, Types needed: {analysis.get('data_types_needed')}")

        # 2. Build filters (từ analysis nếu không có override)
        if filters is None:
            filters = self.analyzer.build_filters(analysis) or None

        # 3. Multi-collection retrieval
        chunks = self.retriever.retrieve(
            question=question,
            top_k_text=3,
            top_k_table=top_k,
            top_k_image=2,
            filters=filters,
        )

        if not chunks:
            return {
                "answer": "Không tìm thấy thông tin liên quan trong tài liệu.",
                "sources": [],
                "analysis": analysis,
            }

        # 4. Build context
        context = self.context_builder.build(chunks)
        citations = self.context_builder.build_citations(chunks)

        # 5. LLM synthesis
        answer = self._synthesize(question, context)

        return {
            "answer": answer,
            "sources": citations,
            "analysis": analysis,
        }

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

        chunks = self.retriever.retrieve(
            question=question, top_k_text=3, top_k_table=top_k, top_k_image=2, filters=filters
        )

        if not chunks:
            yield "Không tìm thấy thông tin liên quan trong tài liệu."
            return

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
