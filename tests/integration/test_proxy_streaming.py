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
        async for _ in resp.aiter_bytes():
            pass


@pytest.mark.asyncio
async def test_confidence_chunk_has_required_fields(client, mock_openai_stream):
    lines = []
    async with client.stream("POST", "/v1/chat/completions", json=MINIMAL_REQUEST) as resp:
        async for line in resp.aiter_lines():
            line = line.strip()
            if line:
                lines.append(line)

    confidence_lines = [l for l in lines if "chat.completion.confidence" in l]
    assert len(confidence_lines) == 1

    payload = orjson.loads(confidence_lines[0].removeprefix("data: "))
    assert 0.0 <= payload["confidence"] <= 1.0
    assert 0.0 <= payload["confidence_raw"] <= 1.0
    assert payload["trace_id"].startswith("tr_")
    assert payload["confidence_tier"] in (0, 1, 2)
    assert payload["confidence_method"] == "tier0_logprob_stop_v1"
    assert payload["calibration_status"] == "uncalibrated"
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


@pytest.mark.asyncio
async def test_stream_mode_store_only(client, mock_openai_stream, monkeypatch):
    """In store_only mode, no confidence SSE chunk is injected."""
    from sidecar import config
    monkeypatch.setattr(config.settings, "confidence_stream_mode", "store_only")

    lines = []
    async with client.stream("POST", "/v1/chat/completions", json=MINIMAL_REQUEST) as resp:
        async for line in resp.aiter_lines():
            line = line.strip()
            if line:
                lines.append(line)

    confidence_lines = [l for l in lines if "chat.completion.confidence" in l]
    assert len(confidence_lines) == 0
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_stream_mode_disabled_passes_through(client, mock_openai_stream, monkeypatch):
    """In disabled mode, the stream is forwarded completely unchanged."""
    from sidecar import config
    monkeypatch.setattr(config.settings, "confidence_stream_mode", "disabled")

    lines = []
    async with client.stream("POST", "/v1/chat/completions", json=MINIMAL_REQUEST) as resp:
        async for line in resp.aiter_lines():
            line = line.strip()
            if line:
                lines.append(line)

    confidence_lines = [l for l in lines if "chat.completion.confidence" in l]
    assert len(confidence_lines) == 0


@pytest.mark.asyncio
async def test_disabled_mode_does_not_inject_logprobs_upstream(
    client, stream_bytes, monkeypatch
):
    """Disabled mode must not add logprobs=True to the upstream request."""
    import json
    import httpx
    import respx

    from sidecar import config
    monkeypatch.setattr(config.settings, "confidence_stream_mode", "disabled")

    captured = {}

    def interceptor(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=stream_bytes,
            headers={"content-type": "text/event-stream"},
        )

    with respx.mock(base_url="https://api.openai.com", assert_all_called=False) as mock:
        mock.post("/v1/chat/completions").mock(side_effect=interceptor)
        async with client.stream(
            "POST", "/v1/chat/completions",
            json={**MINIMAL_REQUEST, "logprobs": False},
        ) as resp:
            async for _ in resp.aiter_bytes():
                pass

    # In disabled mode the customer's original logprobs=False must be forwarded
    assert captured["body"].get("logprobs") is not True
