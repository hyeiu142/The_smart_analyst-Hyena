import re
import uuid
from typing import Any, Dict, List

from llama_parse import LlamaParse

from backend.app.config import get_settings

settings = get_settings()


class TableProcessor:
    """
    Extract tables từ PDF qua LlamaParse → lưu dưới dạng Markdown.

    LlamaParse trả về Markdown, trong đó tables được render thành
    chuẩn Markdown table (| col1 | col2 |).
    Ta tách riêng từng bảng và tạo chunk độc lập.
    """

    def __init__(self):
        self.parser = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="markdown",
            language="vi",       # Hỗ trợ tiếng Việt
            verbose=False,
            # Yêu cầu LlamaParse giữ nguyên cấu trúc bảng
            extract_tables=True,
        )

    async def process(
        self,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract tables và tạo chunks.

        Returns:
            List of table chunks:
            [
                {
                    "id": "uuid",
                    "content": "| Metric | 2024 | 2025 |\\n|---|---|---|\\n| Revenue | 58,000 | 63,724 |",
                    "metadata": {
                        "doc_id": "...",
                        "company": "Vinamilk",
                        "year": 2025,
                        "page_num": 12,
                        "chunk_type": "table",
                        "table_title": "Key Financial Metrics",
                        "row_count": 5,
                        "col_count": 3,
                        "headers": ["Metric", "2024", "2025"]
                    }
                }
            ]
        """
        documents = await self.parser.aload_data(file_path)
        chunks = []

        for doc in documents:
            page_num = doc.metadata.get("page", 1)
            tables = self._extract_tables_from_markdown(doc.text)

            for table_idx, table_md in enumerate(tables):
                headers, row_count, col_count = self._parse_table_info(table_md)
                title = self._infer_title(table_md, doc.text)

                chunk = {
                    "id": str(uuid.uuid4()),
                    "content": table_md,
                    "metadata": {
                        **metadata,
                        "page_num": page_num,
                        "chunk_type": "table",
                        "table_id": f"table_{page_num}_{table_idx}",
                        "table_title": title,
                        "row_count": row_count,
                        "col_count": col_count,
                        "headers": headers,
                    },
                }
                chunks.append(chunk)

        print(f"[TableProcessor] Extracted {len(chunks)} table chunks")
        return chunks

    def _extract_tables_from_markdown(self, text: str) -> List[str]:
        """Tách các Markdown tables từ text."""
        # Pattern: dòng bắt đầu bằng |, có ít nhất 2 dòng
        pattern = r"(\|.+\|[\n\r]+(?:\|[-:| ]+\|[\n\r]+)(?:\|.+\|[\n\r]*)+)"
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches if m.count("\n") >= 2]

    def _parse_table_info(self, table_md: str):
        """Extract headers, row_count, col_count từ markdown table."""
        lines = [l for l in table_md.strip().split("\n") if l.strip()]
        if not lines:
            return [], 0, 0

        # Header là dòng đầu tiên
        header_line = lines[0]
        headers = [h.strip() for h in header_line.split("|") if h.strip()]
        col_count = len(headers)

        # Bỏ dòng separator (dòng 2 chứa ---)
        data_rows = [l for l in lines[2:] if l.strip() and "|" in l]
        row_count = len(data_rows)

        return headers, row_count, col_count

    def _infer_title(self, table_md: str, full_text: str) -> str:
        """
        Tìm tiêu đề bảng từ dòng văn bản ngay trước bảng.
        """
        idx = full_text.find(table_md[:50])
        if idx == -1:
            return "Untitled Table"

        # Lấy 200 ký tự trước bảng
        before = full_text[max(0, idx - 200):idx]
        lines_before = [l.strip() for l in before.split("\n") if l.strip()]

        if lines_before:
            # Dòng cuối cùng trước bảng thường là tiêu đề
            candidate = lines_before[-1]
            # Bỏ markdown heading ký tự
            candidate = re.sub(r"^#+\s*", "", candidate)
            if len(candidate) < 100:
                return candidate

        return "Untitled Table"