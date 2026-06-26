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
            traces.append(json.loads(line))

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Hyena RAG trace JSONL logs.")
    parser.add_argument("--log", type=Path, default=Path("logs/rag_traces.jsonl"))
    args = parser.parse_args()

    traces = load_traces(args.log)
    summary = summarize(traces)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
