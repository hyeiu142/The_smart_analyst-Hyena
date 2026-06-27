from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter()

SUMMARY_LOG = Path("logs/rag_summary.jsonl")
TRACE_LOG = Path("logs/rag_traces.jsonl")
LATEST_MD = Path("logs/rag_latest.md")


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)

    return rows[-limit:] if limit else rows


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * p))
    return round(sorted_values[index], 2)


def _avg(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def _rate(count: int, total: int) -> float:
    return round((count / total) * 100, 2) if total else 0.0


def _numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _step_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces = _load_jsonl(TRACE_LOG)
    if rows:
        request_ids = {row.get("request_id") for row in rows}
        traces = [trace for trace in traces if trace.get("request_id") in request_ids]

    totals: dict[str, list[float]] = {}
    for trace in traces:
        steps = trace.get("steps") or {}
        if not isinstance(steps, dict):
            continue
        for name, value in steps.items():
            try:
                totals.setdefault(name, []).append(float(value or 0))
            except (TypeError, ValueError):
                continue

    breakdown = [
        {
            "step": name,
            "avg_ms": _avg(values),
            "p95_ms": _percentile(values, 0.95),
            "count": len(values),
        }
        for name, values in totals.items()
    ]
    return sorted(breakdown, key=lambda item: item["avg_ms"], reverse=True)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    latencies = _numeric(rows, "total_latency_ms")
    costs = _numeric(rows, "estimated_usd")
    tokens = _numeric(rows, "generation_total_tokens")
    success_count = sum(1 for row in rows if row.get("status") == "success")
    error_count = sum(1 for row in rows if row.get("status") == "error")
    cache_hits = sum(1 for row in rows if row.get("cache_hit") is True)
    cache_observed = sum(1 for row in rows if row.get("cache_hit") is not None)
    image_grounded = sum(1 for row in rows if int(row.get("selected_image_count") or 0) > 0)
    image_triggered = sum(1 for row in rows if row.get("image_lazy_triggered") is True)

    return {
        "total_requests": total,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": _rate(success_count, total),
        "error_rate": _rate(error_count, total),
        "avg_latency_ms": _avg(latencies),
        "p50_latency_ms": round(median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "avg_cost_usd": round(mean(costs), 6) if costs else 0.0,
        "total_cost_usd": round(sum(costs), 6),
        "avg_tokens": _avg(tokens),
        "cache_hit_rate": _rate(cache_hits, cache_observed),
        "image_grounding_rate": _rate(image_grounded, image_triggered),
        "image_triggered_count": image_triggered,
        "image_grounded_count": image_grounded,
        "query_stream_count": sum(1 for row in rows if row.get("mode") == "query_stream"),
        "query_count": sum(1 for row in rows if row.get("mode") == "query"),
    }


@router.get("/summary")
async def get_summary(limit: int = Query(default=200, ge=1, le=5000)):
    rows = _load_jsonl(SUMMARY_LOG, limit=limit)
    return {
        "summary": _summarize(rows),
        "step_breakdown": _step_breakdown(rows),
    }


@router.get("/recent")
async def get_recent(limit: int = Query(default=50, ge=1, le=500)):
    return {"requests": list(reversed(_load_jsonl(SUMMARY_LOG, limit=limit)))}


@router.get("/latest")
async def get_latest():
    latest = _load_jsonl(SUMMARY_LOG, limit=1)
    return {
        "summary": latest[0] if latest else None,
        "markdown": LATEST_MD.read_text(encoding="utf-8") if LATEST_MD.exists() else "",
    }
