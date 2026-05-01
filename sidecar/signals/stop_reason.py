"""Tier 0 signal: stop/finish_reason → confidence factor."""
from __future__ import annotations

_SCORES: dict[str | None, float] = {
    "stop": 1.0,
    "tool_calls": 0.9,
    "function_call": 0.9,
    "length": 0.65,
    "content_filter": 0.5,
    None: 0.8,
}


def compute_stop_reason_signal(finish_reason: str | None) -> float:
    return _SCORES.get(finish_reason, 0.8)
