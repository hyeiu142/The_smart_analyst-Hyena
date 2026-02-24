import json
from typing import Any, Dict, List

from openai import OpenAI

from backend.app.config import get_settings

settings = get_settings()

ANALYZER_PROMPT = """
You are a query analyzer for a financial document RAG system.
Given a user question, extract structured information.

Return JSON with this exact structure:
{
    "intent": "fact_lookup | trend_analysis | comparison | summary | other",
    "entities": {
        "company": "company name or null",
        "year": 2025 or null,
        "years": [2023, 2024, 2025] or null,
        "quarter": "Q4" or null,
        "metric": "gross profit margin" or null
    },
    "data_types_needed": ["text", "table", "image"],
    "sub_questions": ["sub question 1", "sub question 2"]
}

Rules:
- data_types_needed: always include "text". Add "table" if numbers/metrics needed. Add "image" if charts needed.
- If question spans multiple years → set intent="trend_analysis", fill years array.
- sub_questions: break complex question into 1-3 simpler ones for retrieval.
"""


class QueryAnalyzer:
    """
    Analyze user question to extract intent, entities, data types needed, and sub-questions for retrieval.
    Use GPT-4o-mini
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Analyze user question.

        Args:
            question: User's question

        Returns:
            {
                "intent": "trend_analysis",
                "entities": {"company": "Vinamilk", "years": [2023,2024,2025], "metric": "gross profit margin"},
                "data_types_needed": ["text", "table", "image"],
                "sub_questions": ["gross profit margin Vinamilk 2023", ...]
            }
        """
        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": ANALYZER_PROMPT},
                    {"role": "user", "content": question},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            
            print(f"[QueryAnalyzer] Failed: {e}, using fallback")
            return {
                "intent": "other",
                "entities": {},
                "data_types_needed": ["text", "table"],
                "sub_questions": [question],
            }

    def build_filters(self, analysis: Dict) -> Dict:
        """
        Convert entities to Qdrant filter dict.

        Returns none dict if no filters needed.
        """
        entities = analysis.get("entities", {})
        filters = {}

        if entities.get("company"):
            filters["company"] = entities["company"]

        if entities.get("year") and not entities.get("years"):
            filters["year"] = entities["year"]

        return filters