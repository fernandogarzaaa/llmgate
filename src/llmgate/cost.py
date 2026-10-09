"""Token estimation and cost accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class Price:
    """USD per one million tokens."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0

    def __post_init__(self) -> None:
        if self.input_per_1m < 0 or self.output_per_1m < 0:
            raise ValueError("prices must be >= 0")


def cost_usd(prompt_tokens: int, completion_tokens: int, price: Price) -> float:
    """Cost of one call: tokens x price per million tokens."""
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts must be >= 0")
    return (prompt_tokens * price.input_per_1m + completion_tokens * price.output_per_1m) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token for English text).

    Used only when an upstream does not report usage, and for pre-request rate
    limiting. Empty text is 0 tokens; any non-empty text is at least 1.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def message_text(messages: Iterable[Mapping[str, Any]]) -> str:
    """All textual content of a chat ``messages`` list, newline-joined.

    Handles both string content and the list-of-parts form
    (``[{"type": "text", "text": ...}, ...]``); non-text parts are ignored.
    """
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, Mapping) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
    return "\n".join(parts)


def usage_from_response(
    payload: Mapping[str, Any], prompt_text: str, completion_text: str
) -> Dict[str, Any]:
    """Prompt/completion token counts from an OpenAI ``usage`` object, or estimates.

    Returns ``{"prompt_tokens", "completion_tokens", "estimated"}``. A usage
    object counts only when both counts are present non-negative integers.
    """
    usage: Optional[Mapping[str, Any]] = payload.get("usage") if isinstance(payload, Mapping) else None
    if isinstance(usage, Mapping):
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(pt, int) and isinstance(ct, int) and pt >= 0 and ct >= 0:
            return {"prompt_tokens": pt, "completion_tokens": ct, "estimated": False}
    return {
        "prompt_tokens": estimate_tokens(prompt_text),
        "completion_tokens": estimate_tokens(completion_text),
        "estimated": True,
    }
