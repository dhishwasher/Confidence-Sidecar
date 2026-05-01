from sidecar.signals.stop_reason import compute_stop_reason_signal


def test_stop_gives_highest():
    assert compute_stop_reason_signal("stop") == 1.0


def test_tool_calls():
    assert compute_stop_reason_signal("tool_calls") == 0.9


def test_length_truncation_penalized():
    assert compute_stop_reason_signal("length") == 0.65


def test_content_filter_penalized():
    assert compute_stop_reason_signal("content_filter") == 0.5


def test_none_reason():
    assert compute_stop_reason_signal(None) == 0.8


def test_unknown_reason_fallback():
    assert compute_stop_reason_signal("weird_new_reason") == 0.8
