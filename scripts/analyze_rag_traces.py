#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List


def load_traces(path: Path) -> List[Dict[str, Any]]:
    traces = []
    if not path.exists():
        return traces

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(trace, dict):
                traces.append(trace)

    return traces


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * p))
    return round(sorted_values[index], 2)


def summarize(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [
        float((trace.get("metrics") or {}).get("total_latency_ms", 0) or 0)
        for trace in traces
    ]
    successes = [trace for trace in traces if trace.get("status") == "success"]
    errors = [trace for trace in traces if trace.get("status") == "error"]
    cache_hits = [
        trace for trace in traces
        if (trace.get("metrics") or {}).get("cache_hit") is True
    ]
    image_triggers = [
        trace for trace in traces
        if (trace.get("metrics") or {}).get("image_lazy_triggered") is True
    ]
    images_described = sum(
        int((trace.get("metrics") or {}).get("images_described", 0) or 0)
        for trace in traces
    )
    estimated_cost = sum(
        float((trace.get("cost") or {}).get("estimated_usd", 0) or 0)
        for trace in traces
    )

    return {
        "total_requests": len(traces),
        "successes": len(successes),
        "errors": len(errors),
        "cache_hits": len(cache_hits),
        "image_lazy_triggers": len(image_triggers),
        "images_described": images_described,
        "avg_latency_ms": round(mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "estimated_cost_usd": round(estimated_cost, 6),
    }


def print_human_summary(summary: Dict[str, Any]) -> None:
    print("RAG Observability Summary")
    print("=========================")
    print(f"Requests        : {summary['total_requests']}")
    print(f"Success / Errors: {summary['successes']} / {summary['errors']}")
    print(f"Cache hits      : {summary['cache_hits']}")
    print(f"Image triggers  : {summary['image_lazy_triggers']}")
    print(f"Images described: {summary['images_described']}")
    print(f"Latency avg/p50/p95 ms: {summary['avg_latency_ms']} / {summary['p50_latency_ms']} / {summary['p95_latency_ms']}")
    print(f"Estimated cost  : ${summary['estimated_cost_usd']}")


def print_recent(summaries: List[Dict[str, Any]], limit: int) -> None:
    if not summaries or limit <= 0:
        return

    print()
    print(f"Recent Requests ({min(limit, len(summaries))})")
    print("================")
    for item in summaries[-limit:]:
        print(
            f"- {item.get('started_at')} | {item.get('status')} | {item.get('mode')} | "
            f"{item.get('total_latency_ms')} ms | slowest={item.get('slowest_step')} "
            f"({item.get('slowest_step_ms')} ms)"
        )
        print(f"  id={item.get('request_id')}")
        print(f"  q={item.get('question')}")
        if item.get("error"):
            print(f"  error={item.get('error')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Hyena RAG trace JSONL logs.")
    parser.add_argument("--log", type=Path, default=Path("logs/rag_traces.jsonl"))
    parser.add_argument("--summary-log", type=Path, default=Path("logs/rag_summary.jsonl"))
    parser.add_argument("--recent", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()

    traces = load_traces(args.log)
    summary = summarize(traces)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print_human_summary(summary)
    print_recent(load_traces(args.summary_log), args.recent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
