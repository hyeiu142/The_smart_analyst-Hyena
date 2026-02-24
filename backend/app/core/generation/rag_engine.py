from openai import OpenAI
from typing import List, Dict, Any

from backend.app.config import get_settings
from backend.app.core.retrieval.embedder import Embedder
from backend.app.core.retrieval.qdrant_client import QdrantClientWrapper

settings = get_settings()

class RAGEngine: 
    """ 
    RAG Engine for document retrieval and generation

    FLow: 
    1. Embed query
    2. Search Qdrant
    3. Build context from retrieved chunks
    4. Call LLM to generate answer
    """

    def __init__(self):
        self.llm = OpenAI(api_key=settings.openai_api_key)
        self.embedder = Embedder()
        self.qdrant = QdrantClientWrapper()
    async def query(
        self, 
        question: str, 
        top_k: int = 5, 
        filters: Dict = None
    ) -> Dict[str, Any]:
        """
        Main RAG query
        Args: 
            question: User question
            top_k: Number of retrieved chunks for context
            filters: Filters for metadata (company, year, ect.)
                Example: {"company": "Vinamilk", "year": 2025}
            
        Returns:
            {
                "answer": "Doanh thu Q4 2025 là 17,045 tỷ VND...",
                "sources": [
                    {
                        "content": "Revenue in Q4 2025...",
                        "metadata": {"page": 1, "company": "Vinamilk"},
                        "score": 0.89
                    }
                ]
            }

        """

        # 1. Embed query
        query_vector = self.embedder.embed_documents(question)

        # 2. Search Qdrant
        results = self.qdrant.search(
            collection_name=self.qdrant.TEXT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            filters=filters
        )

        if not results: 
            return {
                "answer": "Could not find relevant information",
                "sources": []
            }
        
        # 3. Build context from retrieved chunks
        context = self._build_context(results)

        # 4. Call LLM 
        answer = self._synthesize(question, context)

        return {
            "answer": answer, 
            "sources": [
                {
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "score": round(r["score"], 4)
                }
                for r in results
            ]
        }
    def _build_context(self, chunks: List[Dict]) -> str: 
        """
        Combine retrieved chunks into a context string to provide to LLM

        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1): 
            meta = chunk.get("metadata", {})
            company = meta.get("company", "Unknown")
            page = meta.get("page", "Unknown")
            page = meta.get("page", "?")

            context_parts.append(f"[Source {i} - {company}, Page {page}]\n{chunk['content']}")
        return "\n\n---\n\n".join(context_parts)

    def _synthesize(self, question: str, context: str) -> str: 
        """
        Call LLM to synthesize answer from context
        """
        system_prompt = """
        You are a financial analyst assistant specializing in analyzing corporate financial reports.

        Your task is to answer questions based ONLY on the provided context.

        Rules: 
        - Answer in the same language as the question (Vietnamese or English)
        - Be concise and precise with numbers
        - Always cite source you're referring to (e.g., [Sourse 1])
        - If the answer is not in the context, say so clearly 
        - Do NOT make up number or facts
        """

        user_prompt = f"""Context from financial documents: {context}
        Question: {question}
        Answer based on the context above: 
        """
        response = self.llm.chat.completions.create(
            model=settings.llm_model, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], 
            temperature=0.1, 
            max_tokens=1024

        )

        return response.choices[0].message.content


