"""Unit tests for logprob entropy computation."""
import math

import pytest

from sidecar.signals.logprob_entropy import compute_logprob_entropy, entropy_to_confidence


def _choices(top_logprobs_per_token: list[list[tuple[str, float]]]) -> list[dict]:
    """Helper: build a minimal choices structure."""
    content = []
    for alternatives in top_logprobs_per_token:
        content.append({
            "token": alternatives[0][0],
            "logprob": alternatives[0][1],
            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alternatives],
        })
    return [{"logprobs": {"content": content}, "finish_reason": "stop"}]


def test_entropy_uniform_five_tokens():
    """5 equal-probability tokens → max entropy → confidence = 0."""
    lp = math.log(1 / 5)
    choices = _choices([[("a", lp), ("b", lp), ("c", lp), ("d", lp), ("e", lp)]])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    assert math.isclose(entropy, 1.0, abs_tol=1e-6)
    assert math.isclose(entropy_to_confidence(entropy), 0.0, abs_tol=1e-6)


def test_entropy_dominant_token():
    """One token with prob ~1, others negligible → low entropy → confidence near 1."""
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
    """Only one top_logprob → entropy = 0 → confidence = 1."""
    choices = _choices([[("yes", -0.01)]])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    assert math.isclose(entropy, 0.0, abs_tol=1e-9)
    assert math.isclose(entropy_to_confidence(entropy), 1.0, abs_tol=1e-9)


def test_entropy_empty_choices():
    assert compute_logprob_entropy([]) is None


def test_entropy_no_logprobs_field():
    choices = [{"finish_reason": "stop"}]
    assert compute_logprob_entropy(choices) is None


def test_entropy_empty_content():
    choices = [{"logprobs": {"content": []}, "finish_reason": "stop"}]
    assert compute_logprob_entropy(choices) is None


def test_entropy_neginf_filtered():
    """Tokens with logprob < -100 should be filtered without NaN."""
    choices = _choices([[
        ("yes", -0.1),
        ("no", -999.0),   # underflow
    ]])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    assert not math.isnan(entropy)


def test_entropy_multi_token_mean():
    """Multi-token sequence: verify mean is taken across positions."""
    # Position 0: single alternative → H=0
    # Position 1: uniform 4 alternatives → H=1
    pos0 = [("a", 0.0)]
    lp4 = math.log(0.25)
    pos1 = [("w", lp4), ("x", lp4), ("y", lp4), ("z", lp4)]

    choices = _choices([pos0, pos1])
    entropy = compute_logprob_entropy(choices)
    assert entropy is not None
    # Expected: mean(0.0, 1.0) = 0.5
    assert math.isclose(entropy, 0.5, abs_tol=1e-6)
