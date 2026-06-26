from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class RAGTrace:
    question: str
    mode: str = "query"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    status: str = "running"
    error: str | None = None

    steps: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)
    cost: Dict[str, float] = field(default_factory=dict)

    _start_time: float = field(default_factory=time.perf_counter, repr=False)

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.steps[f"{name}_ms"] = round(elapsed_ms, 2)

    def set_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def add_tokens(self, prefix: str, usage: Dict[str, int]) -> None:
        for key, value in usage.items():
            self.tokens[f"{prefix}_{key}"] = int(value or 0)

    def add_cost(self, key: str, value: float) -> None:
        self.cost[key] = round(float(value or 0), 6)

    def finish(self, status: str = "success", error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.ended_at = utc_now_iso()
        self.metrics["total_latency_ms"] = round((time.perf_counter() - self._start_time) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "mode": self.mode,
            "question": self.question,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "error": self.error,
            "steps": self.steps,
            "metrics": self.metrics,
            "tokens": self.tokens,
            "cost": self.cost,
        }
