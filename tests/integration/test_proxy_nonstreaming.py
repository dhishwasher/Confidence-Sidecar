"""Integration tests for the non-streaming proxy path."""
import json

import httpx
import pytest
import respx

MINIMAL_REQUEST = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}


@pytest.mark.asyncio
async def test_confidence_headers_present(client, mock_openai):
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    assert resp.status_code == 200
    assert "x-confidence" in resp.headers
    conf = float(resp.headers["x-confidence"])
    assert 0.0 <= conf <= 1.0


@pytest.mark.asyncio
async def test_trace_id_header_present(client, mock_openai):
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    assert resp.status_code == 200
    assert resp.headers.get("x-trace-id", "").startswith("tr_")


@pytest.mark.asyncio
async def test_confidence_tier_header(client, mock_openai):
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    assert resp.status_code == 200
    tier = int(resp.headers["x-confidence-tier"])
    assert tier in (0, 1, 2)


@pytest.mark.asyncio
async def test_logprobs_stripped_when_not_requested(client, mock_openai):
    resp = await client.post("/v1/chat/completions", json={**MINIMAL_REQUEST, "logprobs": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["logprobs"] is None


@pytest.mark.asyncio
async def test_logprobs_preserved_when_requested(client, mock_openai):
    resp = await client.post(
        "/v1/chat/completions",
        json={**MINIMAL_REQUEST, "logprobs": True, "top_logprobs": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["logprobs"] is not None


@pytest.mark.asyncio
async def test_upstream_receives_injected_logprobs(client, openai_response_json):
    """Verify the upstream call always has logprobs=True regardless of customer request."""
    captured = {}

    def interceptor(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=openai_response_json)

    with respx.mock(base_url="https://api.openai.com", assert_all_called=False) as mock:
        mock.post("/v1/chat/completions").mock(side_effect=interceptor)
        await client.post("/v1/chat/completions", json={**MINIMAL_REQUEST, "logprobs": False})

    assert captured["body"]["logprobs"] is True
    assert captured["body"]["top_logprobs"] >= 5


@pytest.mark.asyncio
async def test_response_shape_passthrough(client, mock_openai, openai_response_json):
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    body = resp.json()
    assert body["id"] == openai_response_json["id"]
    assert body["object"] == "chat.completion"
    assert "choices" in body
    assert body["choices"][0]["message"]["content"] == "Paris"


@pytest.mark.asyncio
async def test_trace_saved_to_db(client, mock_openai, db):
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    trace_id = resp.headers.get("x-trace-id")
    assert trace_id is not None

    # Give the background task a moment to complete
    import asyncio
    await asyncio.sleep(0.05)

    trace_resp = await client.get(f"/v1/traces/{trace_id}")
    assert trace_resp.status_code == 200
    trace = trace_resp.json()
    assert trace["id"] == trace_id
    assert trace["provider"] == "openai"
