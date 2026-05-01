"""Integration tests for GET /v1/traces/{id}."""
import asyncio

import pytest

MINIMAL_REQUEST = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
}


@pytest.mark.asyncio
async def test_trace_not_found(client, db):
    resp = await client.get("/v1/traces/tr_nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_trace_retrievable_after_request(client, mock_openai, db):
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    assert resp.status_code == 200
    trace_id = resp.headers["x-trace-id"]

    await asyncio.sleep(0.05)  # let background save complete

    trace_resp = await client.get(f"/v1/traces/{trace_id}")
    assert trace_resp.status_code == 200
    body = trace_resp.json()
    assert body["id"] == trace_id
    assert body["model"] == "gpt-4o"
    assert body["provider"] == "openai"
    assert body["tier"] == 0
    assert body["confidence"] is not None
    assert isinstance(body["signals"], list)
    assert len(body["signals"]) >= 1


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_confidence_endpoint_not_found(client, db):
    resp = await client.get("/v1/traces/tr_nonexistent/confidence")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_confidence_endpoint_returns_fields(client, mock_openai, db):
    """GET /v1/traces/{id}/confidence returns the lightweight confidence payload."""
    import asyncio
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    trace_id = resp.headers["x-trace-id"]
    await asyncio.sleep(0.05)

    conf_resp = await client.get(f"/v1/traces/{trace_id}/confidence")
    assert conf_resp.status_code == 200
    body = conf_resp.json()
    assert body["trace_id"] == trace_id
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["confidence_raw"] <= 1.0
    assert body["confidence_method"] == "tier0_logprob_stop_v1"
    assert body["calibration_status"] == "uncalibrated"
    assert isinstance(body["signals"], dict)
    assert body["confidence_tier"] in (0, 1, 2)
