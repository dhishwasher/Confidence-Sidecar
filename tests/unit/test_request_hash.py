"""Unit tests for request hash width."""
from sidecar.models.openai import ChatCompletionRequest
from sidecar.routers.proxy import _make_request_hash

_BASE = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
}


def _hash(**overrides) -> str:
    req = ChatCompletionRequest.model_validate({**_BASE, **overrides})
    return _make_request_hash(req)


def test_same_request_same_hash():
    assert _hash() == _hash()


def test_different_model():
    assert _hash(model="gpt-4o") != _hash(model="gpt-4o-mini")


def test_different_temperature():
    assert _hash(temperature=0.0) != _hash(temperature=1.0)


def test_different_seed():
    assert _hash(seed=1) != _hash(seed=2)


def test_different_max_tokens():
    assert _hash(max_tokens=100) != _hash(max_tokens=200)


def test_different_tools():
    tool_a = [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
    tool_b = [{"type": "function", "function": {"name": "bar", "parameters": {}}}]
    assert _hash(tools=tool_a) != _hash(tools=tool_b)


def test_different_messages():
    req_a = ChatCompletionRequest.model_validate({"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]})
    req_b = ChatCompletionRequest.model_validate({"model": "gpt-4o", "messages": [{"role": "user", "content": "Bye"}]})
    assert _make_request_hash(req_a) != _make_request_hash(req_b)
