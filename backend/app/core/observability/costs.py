from __future__ import annotations

from typing import Dict


MODEL_PRICES_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "text-embedding-3-small": {
        "input": 0.02,
        "output": 0.0,
    },
}


def estimate_openai_cost_usd(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    prices = MODEL_PRICES_PER_1M.get(model)
    if not prices:
        return 0.0

    input_cost = prompt_tokens / 1_000_000 * prices["input"]
    output_cost = completion_tokens / 1_000_000 * prices["output"]
    return round(input_cost + output_cost, 6)