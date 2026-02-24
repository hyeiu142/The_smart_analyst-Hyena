from typing import Any, Dict, List


class ContextBuilder:
    """
    Format danh sách chunks thành context string đẹp để đưa vào LLM.
    Phân biệt rõ nguồn: [TEXT], [TABLE], [CHART].
    """

    def build(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Args:
            chunks: List từ MultiCollectionRetriever.retrieve()

        Returns:
            Context string dạng:
            ---
            [TABLE - Vinamilk, Page 12, Score: 0.92]
            | Metric | 2024 | 2025 |
            ...
            ---
            [TEXT - Vinamilk, Page 24, Score: 0.87]
            Gross profit margin expanded...
            ---
        """
        if not chunks:
            return "No relevant information found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = self._format_header(i, chunk)
            parts.append(f"{header}\n{chunk['content']}")

        return "\n\n---\n\n".join(parts)

    def _format_header(self, index: int, chunk: Dict) -> str:
        meta = chunk.get("metadata") or {}
        source_type = chunk.get("source_collection", "text").upper()
        company = meta.get("company", "Unknown")
        page = meta.get("page_num", meta.get("page", "?"))
        score = round(chunk.get("score", 0), 3)

        label_map = {
            "TEXT": "TEXT",
            "TABLE": "TABLE",
            "IMAGE": "CHART/IMAGE",
        }
        label = label_map.get(source_type, source_type)

        return f"[{label} #{index} - {company}, Page {page}, Score: {score}]"

    def build_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict]:
        """
        Tạo danh sách citations để trả về frontend.

        Returns:
            [
                {
                    "index": 1,
                    "type": "table",
                    "company": "Vinamilk",
                    "page": 12,
                    "score": 0.92,
                    "preview": "| Metric | 2024 ..."  (50 chars)
                }
            ]
        """
        citations = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata") or {}
            citations.append({
                "index": i,
                "type": chunk.get("source_collection", "text"),
                "company": meta.get("company", "Unknown"),
                "page": meta.get("page_num", meta.get("page", "?")),
                "score": round(chunk.get("score", 0), 4),
                "preview": chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"],
            })
        return citations