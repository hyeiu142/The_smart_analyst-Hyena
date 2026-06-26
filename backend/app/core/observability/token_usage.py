from __future__ import annotations

from typing import Any, Dict


def openai_usage_to_dict(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {}

    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }