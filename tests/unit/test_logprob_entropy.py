"""Unit tests for logprob entropy computation (residual-mass corrected)."""
import math

import pytest

from sidecar.signals.logprob_entropy import compute_logprob_entropy, entropy_to_confidence


def _choices(top_logprobs_per_token: list[list[tuple[str, float]]]) -> list[dict]:
    """Build a minimal choices structure."""
    content = []
    for alternatives in top_logprobs_per_token:
        content.append({
            "token": alternatives[0][0],
            "logprob": alternatives[0][1],
            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alternatives],
        })
    return [{"logprobs": {"content": content}, "finish_reason": "stop"}]


def test_entropy_uniform_five_tokens():
    """5 equal-probability tokens that together sum to 1 → max entropy."""
    lp = math.log(1 / 5)
    choices = _choices([[("a", lp), ("b", lp), ("c", lp), ("d", lp), ("e", lp)]])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    # Probabilities sum to 1 → no residual → entropy = log(5)/log(5) = 1.0
    assert math.isclose(entropy, 1.0, abs_tol=1e-5)
    assert math.isclose(entropy_to_confidence(entropy), 0.0, abs_tol=1e-5)


def test_entropy_dominant_token():
    """One token with prob ~1, others negligible → confidence near 1."""
    choices = _choices([[
        ("Paris", -0.001),
        ("London", -10.0),
        ("Berlin", -11.0),
        ("Rome", -12.0),
        ("Madrid", -13.0),
    ]])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    assert entropy < 0.05
    assert entropy_to_confidence(entropy) > 0.95


def test_entropy_single_alternative():
    """Only one top_logprob entry with prob ≈ 1 → entropy = 0 → confidence = 1."""
    choices = _choices([[("yes", -0.0001)]])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    # Residual = 1 - exp(-0.0001) ≈ 0.0001, so k=2
    # entropy is tiny but > 0
    assert entropy < 0.01
    assert entropy_to_confidence(entropy) > 0.99


def test_residual_mass_raises_entropy():
    """When returned probs sum to only 0.5, residual adds a 0.5 'other' bucket.

    Without correction: entropy over one token with prob 1.0 → 0 (spuriously certain).
    With correction: entropy over {0.5, 0.5} → 1.0 (maximally uncertain).
    """
    # Single top_logprob with logprob = log(0.5)
    lp_half = math.log(0.5)
    choices = _choices([[("yes", lp_half)]])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    # probs = [0.5, residual=0.5] → H = log(2)/log(2) = 1.0
    assert math.isclose(entropy, 1.0, abs_tol=1e-5)


def test_entropy_empty_choices():
    assert compute_logprob_entropy([]) is None


def test_entropy_no_logprobs_field():
    assert compute_logprob_entropy([{"finish_reason": "stop"}]) is None


def test_entropy_empty_content():
    assert compute_logprob_entropy([{"logprobs": {"content": []}, "finish_reason": "stop"}]) is None


def test_entropy_neginf_filtered():
    """Logprob < -100 filtered without NaN."""
    choices = _choices([[
        ("yes", -0.1),
        ("no", -999.0),
    ]])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    assert not math.isnan(entropy)


def test_entropy_multi_token_mean():
    """Multi-token sequence: mean taken across positions."""
    # Position 0: single alternative with prob 1 → H ≈ 0
    # Position 1: uniform 4 alternatives summing to 1 → H = 1
    pos0 = [("a", -1e-9)]  # prob ≈ 1, residual ≈ 0, k=1 → entropy=0
    lp4 = math.log(0.25)
    pos1 = [("w", lp4), ("x", lp4), ("y", lp4), ("z", lp4)]

    choices = _choices([pos0, pos1])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    # pos0: k=1 (residual < 1e-6) → 0.0
    # pos1: k=4, probs sum to 1, no residual → H = 1.0
    # mean = 0.5
    assert math.isclose(entropy, 0.5, abs_tol=0.01)
