"""Weighted fusion of available signals into a single confidence scalar."""
from __future__ import annotations

from sidecar.config import settings

_TIER0_WEIGHTS: dict[str, float] = {
    "logprob_entropy": 0.85,
    "stop_reason": 0.15,
}

_TIER1_WEIGHTS: dict[str, float] = {
    "logprob_entropy": 0.50,
    "semantic_entropy": 0.40,
    "stop_reason": 0.10,
}


def combine_signals(signals: dict[str, float | None], tier: int) -> float:
    """Weighted mean of available (non-None) signals.

    Missing signals are excluded and the remaining weights renormalized.
    Returns a value in [0, 1]; falls back to 0.5 if no signals available.
    """
    weights = _TIER0_WEIGHTS if tier == 0 else _TIER1_WEIGHTS
    weighted_sum = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        value = signals.get(name)
        if value is not None:
            weighted_sum += weight * value
            total_weight += weight
    if total_weight == 0.0:
        return 0.5
    raw = weighted_sum / total_weight
    return max(0.0, min(1.0, raw))


def classify_confidence_tier(confidence: float) -> int:
    """Map scalar confidence to ordinal tier label (0=low, 1=mid, 2=high)."""
    if confidence < settings.tier1_confidence_band_low:
        return 0
    if confidence > settings.tier1_confidence_band_high:
        return 2
    return 1
