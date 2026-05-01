import pytest
from sidecar.signals.combiner import classify_confidence_tier, combine_signals


def test_tier0_full_signals():
    signals = {"logprob_entropy": 0.9, "stop_reason": 0.8}
    result = combine_signals(signals, tier=0)
    expected = 0.85 * 0.9 + 0.15 * 0.8
    assert abs(result - expected) < 1e-9


def test_missing_logprob_fallback_to_stop_reason():
    signals = {"logprob_entropy": None, "stop_reason": 0.65}
    result = combine_signals(signals, tier=0)
    # Only stop_reason contributes (weight renormalizes to 1.0)
    assert abs(result - 0.65) < 1e-9


def test_all_missing_returns_neutral():
    assert combine_signals({}, tier=0) == 0.5


def test_clamps_above_one():
    # Shouldn't happen naturally, but combiner must clamp
    signals = {"logprob_entropy": 1.5, "stop_reason": 1.5}
    assert combine_signals(signals, tier=0) == 1.0


def test_clamps_below_zero():
    signals = {"logprob_entropy": -0.5, "stop_reason": -0.5}
    assert combine_signals(signals, tier=0) == 0.0


def test_classify_low_confidence():
    assert classify_confidence_tier(0.1) == 0


def test_classify_mid_confidence():
    assert classify_confidence_tier(0.5) == 1


def test_classify_high_confidence():
    assert classify_confidence_tier(0.9) == 2


def test_classify_boundary_low():
    # Exactly at low boundary goes to mid
    assert classify_confidence_tier(0.3) == 1


def test_classify_boundary_high():
    # Exactly at high boundary goes to mid (not >)
    assert classify_confidence_tier(0.7) == 1
