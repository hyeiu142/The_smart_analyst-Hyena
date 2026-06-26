from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonlTraceLogger:
    def __init__(self, path: str = "logs/rag_traces.jsonl"):
        self.path = Path(path)

    def write(self, trace: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(trace, ensure_ascii=False) + "\n")