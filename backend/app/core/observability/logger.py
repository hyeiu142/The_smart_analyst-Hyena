from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonlTraceLogger:
    def __init__(
        self,
        trace_path: str = "logs/rag_traces.jsonl",
        summary_path: str = "logs/rag_summary.jsonl",
        latest_path: str = "logs/rag_latest.md",
    ):
        self.trace_path = Path(trace_path)
        self.summary_path = Path(summary_path)
        self.latest_path = Path(latest_path)

    def write(self, trace: Dict[str, Any]) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(trace, ensure_ascii=False) + "\n")

        summary = self._build_summary(trace)
        with self.summary_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary, ensure_ascii=False) + "\n")

        self.latest_path.write_text(self._build_markdown(trace, summary), encoding="utf-8")

    def _build_summary(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        metrics = trace.get("metrics") or {}
        steps = trace.get("steps") or {}
        tokens = trace.get("tokens") or {}
        cost = trace.get("cost") or {}

        slowest_step = None
        if steps:
            slowest_step = max(steps.items(), key=lambda item: float(item[1] or 0))

        return {
            "request_id": trace.get("request_id"),
            "started_at": trace.get("started_at"),
            "mode": trace.get("mode"),
            "status": trace.get("status"),
            "question": self._shorten(trace.get("question") or "", 160),
            "total_latency_ms": metrics.get("total_latency_ms"),
            "slowest_step": slowest_step[0] if slowest_step else None,
            "slowest_step_ms": slowest_step[1] if slowest_step else None,
            "cache_skipped": metrics.get("cache_skipped"),
            "cache_hit": metrics.get("cache_hit"),
            "intent": metrics.get("intent"),
            "filters": metrics.get("filters"),
            "retrieval_chunks": metrics.get("retrieval_chunks"),
            "rerank_chunks": metrics.get("rerank_chunks"),
            "context_selection_chunks": metrics.get("context_selection_chunks"),
            "context_selection_image_hits": metrics.get("context_selection_image_hits"),
            "citations_count": metrics.get("citations_count"),
            "image_lazy_triggered": metrics.get("image_lazy_triggered"),
            "images_described": metrics.get("images_described"),
            "image_forced_into_context": metrics.get("image_forced_into_context"),
            "forced_image_scores": metrics.get("forced_image_scores"),
            "selected_image_count": metrics.get("selected_image_count"),
            "selected_image_paths": metrics.get("selected_image_paths"),
            "generation_total_tokens": tokens.get("generation_total_tokens"),
            "estimated_usd": cost.get("estimated_usd"),
            "error": trace.get("error"),
        }

    def _build_markdown(self, trace: Dict[str, Any], summary: Dict[str, Any]) -> str:
        metrics = trace.get("metrics") or {}
        steps = trace.get("steps") or {}
        tokens = trace.get("tokens") or {}
        cost = trace.get("cost") or {}

        lines = [
            "# Latest RAG Trace",
            "",
            "## Request",
            f"- request_id: `{summary.get('request_id')}`",
            f"- started_at: `{summary.get('started_at')}`",
            f"- mode: `{summary.get('mode')}`",
            f"- status: `{summary.get('status')}`",
            f"- total_latency_ms: `{summary.get('total_latency_ms')}`",
            f"- question: {trace.get('question') or ''}",
            "",
            "## Retrieval",
            f"- cache_skipped: `{metrics.get('cache_skipped')}`",
            f"- cache_hit: `{metrics.get('cache_hit')}`",
            f"- intent: `{metrics.get('intent')}`",
            f"- data_types_needed: `{metrics.get('data_types_needed')}`",
            f"- filters: `{metrics.get('filters')}`",
            f"- retrieval_chunks: `{metrics.get('retrieval_chunks')}`",
            f"- retrieval_hits: text=`{metrics.get('retrieval_text_hits')}`, table=`{metrics.get('retrieval_table_hits')}`, image=`{metrics.get('retrieval_image_hits')}`",
            f"- rerank_chunks: `{metrics.get('rerank_chunks')}`",
            f"- rerank_hits: text=`{metrics.get('rerank_text_hits')}`, table=`{metrics.get('rerank_table_hits')}`, image=`{metrics.get('rerank_image_hits')}`",
            f"- context_selection_chunks: `{metrics.get('context_selection_chunks')}`",
            f"- context_selection_hits: text=`{metrics.get('context_selection_text_hits')}`, table=`{metrics.get('context_selection_table_hits')}`, image=`{metrics.get('context_selection_image_hits')}`",
            f"- citations_count: `{metrics.get('citations_count')}`",
            "",
            "## Image",
            f"- image_lazy_triggered: `{metrics.get('image_lazy_triggered')}`",
            f"- images_described: `{metrics.get('images_described')}`",
            f"- image_forced_into_context: `{metrics.get('image_forced_into_context')}`",
            f"- forced_image_score: `{metrics.get('forced_image_score')}`",
            f"- forced_image_scores: `{metrics.get('forced_image_scores')}`",
            f"- selected_image_count: `{metrics.get('selected_image_count')}`",
            f"- selected_image_paths: `{metrics.get('selected_image_paths')}`",
            "",
            "## Timing",
        ]

        for name, elapsed_ms in sorted(steps.items(), key=lambda item: float(item[1] or 0), reverse=True):
            lines.append(f"- {name}: `{elapsed_ms}` ms")

        lines.extend([
            "",
            "## Tokens And Cost",
            f"- tokens: `{tokens}`",
            f"- cost: `{cost}`",
        ])

        if trace.get("error"):
            lines.extend(["", "## Error", str(trace["error"])])

        lines.append("")
        return "\n".join(lines)

    def _shorten(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."
