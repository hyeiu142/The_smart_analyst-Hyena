# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ragas>=0.4.0",
#     "langchain-openai>=0.3.0",
#     "langchain-community>=0.3.0",
#     "datasets",
#     "python-dotenv",
#     "pandas"
# ]
# ///
"""
Run RAGAS evaluation on a generation report (cases.jsonl).

Usage:
    uv run eval/scripts/run_ragas_eval.py --report-dir eval/reports/generation_XXXXXXXX
    uv run eval/scripts/run_ragas_eval.py --report-dir eval/reports/generation_XXXXXXXX --limit 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Compatibility patch: RAGAS 0.4.x tries to import ChatVertexAI which was
# removed from langchain-community >= 0.3.0. Mock it so the import succeeds.
# ---------------------------------------------------------------------------
_mock_vertexai = MagicMock()
_mock_vertexai.ChatVertexAI = MagicMock
sys.modules.setdefault("langchain_community.chat_models.vertexai", _mock_vertexai)

from datasets import Dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import AnswerCorrectness, Faithfulness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ragas").setLevel(logging.WARNING)

load_dotenv()


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    cases = []
    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def build_ragas_dataset(cases: list[dict[str, Any]]) -> Dataset:
    data: dict[str, list] = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    for case in cases:
        data["user_input"].append(case["question"])
        data["response"].append(case.get("answer", ""))
        data["reference"].append(case.get("ground_truth_answer", ""))

        contexts = [
            source.get("preview") or source.get("text", "")
            for source in case.get("sources", [])
            if (source.get("preview") or source.get("text", "")).strip()
        ]
        if not contexts:
            contexts = ["No context retrieved."]
        data["retrieved_contexts"].append(contexts)

    return Dataset.from_dict(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on generation report.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases.")
    args = parser.parse_args()

    cases_path = args.report_dir / "cases.jsonl"
    if not cases_path.exists():
        raise FileNotFoundError(f"cases.jsonl not found in {args.report_dir}")

    cases = load_cases(cases_path)
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"Loaded {len(cases)} cases from {cases_path.name}")

    dataset = build_ragas_dataset(cases)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_retries=5, timeout=120.0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", max_retries=5, timeout=120.0)

    metrics = [
        AnswerCorrectness(llm=llm),
        Faithfulness(llm=llm),
    ]

    print("Adapting internal RAGAS prompts to Vietnamese (this takes a few seconds)...")
    for metric in metrics:
        if hasattr(metric, "adapt_prompts"):
            metric.adapt_prompts("vietnamese", llm=llm)

    print("Running RAGAS evaluation (uses OpenAI API)...")
    run_config = RunConfig(max_workers=8, timeout=120)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    print("\n=== RAGAS Results ===")

    df = result.to_pandas()
    numeric_cols = df.select_dtypes(include="number").columns
    summary = {k: round(float(df[k].mean()), 4) for k in numeric_cols}
    print(json.dumps(summary, indent=2))

    csv_path = args.report_dir / "ragas_report.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_path = args.report_dir / "ragas_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nCSV report  → {csv_path}")
    print(f"JSON summary → {summary_path}")


if __name__ == "__main__":
    main()
