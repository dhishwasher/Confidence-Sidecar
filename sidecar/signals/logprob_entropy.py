"""Tier 0 signal: per-token logprob entropy → scalar generation certainty.

IMPORTANT — what this measures
-------------------------------
This signal measures *token-distribution confidence*: how peaked the model's
next-token distribution is at each output position.  High score means the
model's own distribution was sharply concentrated; low score means it was
spread across many alternatives.

This is NOT a measure of factual correctness.  A model can be token-certain
while producing a confident hallucination.  Label this signal "generation
certainty" or "token-distribution confidence" in any user-facing copy.

Residual-mass correction
------------------------
OpenAI returns at most top_logprobs_count alternatives per position.  The
omitted tail tokens carry real probability mass.  Renormalising only the
returned K values inflates apparent certainty.  Instead, we treat the total
missing mass as a single "other" bucket and include it in the entropy
calculation.  This makes the score conservative (closer to 0.5) when the
tail is significant.
"""
from __future__ import annotations

import math
from typing import Any


def compute_logprob_entropy(choices: list[dict[str, Any]]) -> float | None:
    """Compute mean normalised Shannon entropy over all output token positions.

    Args:
        choices: The ``choices`` list from an OpenAI response or an accumulated
                 streaming snapshot.  Reads ``choices[0].logprobs.content``.

    Returns:
        Mean normalised entropy in [0, 1] where 0 = fully certain and
        1 = maximum uncertainty.  Returns ``None`` if no logprob data is
        present (e.g. Anthropic provider).
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

        # Filter extreme underflow values (logprob < -100 ≈ prob < 3e-44)
        log_probs = [e["logprob"] for e in top_k if e.get("logprob", -1000) > -100]
        if not log_probs:
            continue

        probs = [math.exp(lp) for lp in log_probs]
        total_returned = sum(probs)

        # Residual-mass correction: treat all unobserved tail tokens as one
        # aggregate "other" bucket.  This avoids over-confident scores when
        # the top-K slice covers only a fraction of the true distribution.
        residual = max(0.0, 1.0 - total_returned)
        if residual > 1e-6:
            probs.append(residual)

        k = len(probs)
        if k < 2:
            per_token_entropies.append(0.0)
            continue

        h = -sum(p * math.log(p) for p in probs if p > 1e-12)
        h_max = math.log(k)
        per_token_entropies.append(h / h_max)

    if not per_token_entropies:
        return None

    return sum(per_token_entropies) / len(per_token_entropies)


def entropy_to_confidence(entropy: float) -> float:
    """Map normalised entropy [0=certain, 1=uniform] to confidence [0, 1]."""
    return 1.0 - entropy
