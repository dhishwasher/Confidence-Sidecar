"""Unit tests for SSEAccumulator."""
import orjson
import pytest

from sidecar.streaming.sse_parser import SSEAccumulator


CHUNK_WITH_LOGPROBS = (
    b'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
    b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null,'
    b'"logprobs":{"content":[{"token":"Hi","logprob":-0.1,'
    b'"top_logprobs":[{"token":"Hi","logprob":-0.1},{"token":"Hey","logprob":-2.0}]}]}}]}\n\n'
)

CHUNK_NO_LOGPROBS = (
    b'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
    b'"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null,"logprobs":null}]}\n\n'
)

CHUNK_FINISH = (
    b'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}]}\n\n'
)

DONE_LINE = b"data: [DONE]\n\n"


def test_simple_chunk_forwarded():
    acc = SSEAccumulator(customer_wants_logprobs=True)
    results = acc.feed(CHUNK_WITH_LOGPROBS)
    assert len(results) == 1
    assert results[0] is not None
    assert b"data:" in results[0]


def test_logprob_accumulated():
    acc = SSEAccumulator(customer_wants_logprobs=True)
    acc.feed(CHUNK_WITH_LOGPROBS)
    assert len(acc.logprob_tokens) == 1
    assert acc.logprob_tokens[0]["token"] == "Hi"


def test_logprobs_stripped_when_not_wanted():
    acc = SSEAccumulator(customer_wants_logprobs=False)
    results = acc.feed(CHUNK_WITH_LOGPROBS)
    assert len(results) == 1
    payload = orjson.loads(results[0][len(b"data: "):].strip())
    assert payload["choices"][0]["logprobs"] is None


def test_logprobs_preserved_when_wanted():
    acc = SSEAccumulator(customer_wants_logprobs=True)
    results = acc.feed(CHUNK_WITH_LOGPROBS)
    payload = orjson.loads(results[0][len(b"data: "):].strip())
    assert payload["choices"][0]["logprobs"] is not None


def test_done_returns_none():
    acc = SSEAccumulator(customer_wants_logprobs=False)
    results = acc.feed(DONE_LINE)
    assert len(results) == 1
    assert results[0] is None
    assert acc.done is True


def test_finish_reason_captured():
    acc = SSEAccumulator(customer_wants_logprobs=False)
    acc.feed(CHUNK_FINISH)
    assert acc.finish_reason == "stop"


def test_buffered_split_across_feeds():
    """Simulate bytes arriving split across two feed() calls."""
    acc = SSEAccumulator(customer_wants_logprobs=True)
    half = len(CHUNK_WITH_LOGPROBS) // 2
    r1 = acc.feed(CHUNK_WITH_LOGPROBS[:half])
    r2 = acc.feed(CHUNK_WITH_LOGPROBS[half:])
    all_results = [x for x in r1 + r2 if x is not None]
    assert len(all_results) == 1


def test_multiple_chunks_accumulate_logprobs():
    acc = SSEAccumulator(customer_wants_logprobs=True)
    acc.feed(CHUNK_WITH_LOGPROBS)
    acc.feed(CHUNK_WITH_LOGPROBS)
    assert len(acc.logprob_tokens) == 2


def test_build_choices_snapshot():
    acc = SSEAccumulator(customer_wants_logprobs=True)
    acc.feed(CHUNK_WITH_LOGPROBS)
    acc.feed(CHUNK_FINISH)
    snapshot = acc.build_choices_snapshot()
    assert snapshot[0]["finish_reason"] == "stop"
    assert len(snapshot[0]["logprobs"]["content"]) == 1
