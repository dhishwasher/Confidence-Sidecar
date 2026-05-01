"""Integration tests for the streaming proxy path."""
import orjson
import pytest


MINIMAL_REQUEST = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": True,
}


@pytest.mark.asyncio
async def test_trace_id_in_initial_headers(client, mock_openai_stream):
    async with client.stream("POST", "/v1/chat/completions", json=MINIMAL_REQUEST) as resp:
        assert resp.status_code == 200
        assert resp.headers.get("x-trace-id", "").startswith("tr_")
        # consume body to avoid connection errors
        async for _ in resp.aiter_bytes():
            pass


@pytest.mark.asyncio
async def test_confidence_chunk_emitted_before_done(client, mock_openai_stream):
    lines = []
    async with client.stream("POST", "/v1/chat/completions", json=MINIMAL_REQUEST) as resp:
        async for line in resp.aiter_lines():
            line = line.strip()
            if line:
                lines.append(line)

    # Find confidence chunk
    confidence_lines = [l for l in lines if b"chat.completion.confidence" in l.encode()]
    assert len(confidence_lines) == 1, f"Expected 1 confidence chunk, got: {confidence_lines}"

    payload = orjson.loads(confidence_lines[0].removeprefix("data: "))
    assert "confidence" in payload
    assert 0.0 <= payload["confidence"] <= 1.0
    assert "trace_id" in payload
    assert payload["trace_id"].startswith("tr_")
    assert "confidence_tier" in payload
    assert "signals" in payload


@pytest.mark.asyncio
async def test_done_is_last_event(client, mock_openai_stream):
    lines = []
    async with client.stream("POST", "/v1/chat/completions", json=MINIMAL_REQUEST) as resp:
        async for line in resp.aiter_lines():
            line = line.strip()
            if line:
                lines.append(line)

    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_content_type_is_sse(client, mock_openai_stream):
    async with client.stream("POST", "/v1/chat/completions", json=MINIMAL_REQUEST) as resp:
        assert "text/event-stream" in resp.headers.get("content-type", "")
        async for _ in resp.aiter_bytes():
            pass
