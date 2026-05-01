"""Tier 0 signal: per-token logprob entropy → scalar confidence."""
from __future__ import annotations

import math
from typing import Any


def compute_logprob_entropy(choices: list[dict[str, Any]]) -> float | None:
    """Compute mean normalized Shannon entropy over all output token positions.

    Args:
        choices: The ``choices`` list from an OpenAI response or accumulated
                 streaming chunks.  We read ``choices[0].logprobs.content``.

    Returns:
        Mean normalized entropy in [0, 1] where 0 = fully certain and
        1 = maximum uncertainty, or ``None`` if no logprob data is present.
    """
    if not choices:
        return None

    logprobs_obj = choices[0].get("logprobs") or {}
    content_tokens: list[dict[str, Any]] = logprobs_obj.get("content") or []

    if not content_tokens:
        return None

    per_token_entropies: list[float] = []

    for token_entry in content_tokens:
        top_k: list[dict[str, Any]] = token_entry.get("top_logprobs") or []
        if not top_k:
            continue

        # Filter out numerically extreme values (−∞ / underflow)
        log_probs = [e["logprob"] for e in top_k if e.get("logprob", -1000) > -100]
        if not log_probs:
            continue

        probs = [math.exp(lp) for lp in log_probs]
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]

        k = len(probs)
        if k < 2:
            # Single alternative → entropy = 0 → confidence = 1
            per_token_entropies.append(0.0)
            continue

        h = -sum(p * math.log(p) for p in probs if p > 1e-12)
        h_max = math.log(k)
        per_token_entropies.append(h / h_max)

    if not per_token_entropies:
        return None

    return sum(per_token_entropies) / len(per_token_entropies)


def entropy_to_confidence(entropy: float) -> float:
    """Map normalized entropy [0=certain, 1=uniform] to confidence [0, 1]."""
    return 1.0 - entropy
