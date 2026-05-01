"""Integration tests for GET /v1/calibration/curve."""
import asyncio

import pytest

MINIMAL_REQUEST = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
}


@pytest.mark.asyncio
async def test_curve_no_data(client, db):
    resp = await client.get("/v1/calibration/curve")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_data"
    assert body["n_samples"] == 0
    assert body["buckets"] == []
    assert body["metrics"] is None


@pytest.mark.asyncio
async def test_curve_with_feedback(client, mock_openai, db):
    """After labelling a trace the curve endpoint returns real bucket data."""
    # Create a trace
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    assert resp.status_code == 200
    trace_id = resp.headers["x-trace-id"]
    await asyncio.sleep(0.05)

    # Label it
    fb = await client.post(f"/v1/feedback/{trace_id}", json={"label": "correct"})
    assert fb.status_code == 200

    # Curve should now have data
    curve = await client.get("/v1/calibration/curve")
    assert curve.status_code == 200
    body = curve.json()
    assert body["n_samples"] == 1
    assert body["status"] in ("ok", "low_sample_count")
    assert body["metrics"] is not None
    assert "ece" in body["metrics"]
    assert "brier_score" in body["metrics"]

    # At least one bucket should be non-empty
    non_empty = [b for b in body["buckets"] if b["count"] > 0]
    assert len(non_empty) >= 1
    bucket = non_empty[0]
    assert bucket["fraction_correct"] is not None
    assert 0.0 <= bucket["fraction_correct"] <= 1.0
    assert bucket["mean_confidence"] is not None


@pytest.mark.asyncio
async def test_curve_buckets_structure(client, mock_openai, db):
    """Curve always returns exactly 10 buckets covering [0, 1]."""
    resp = await client.post("/v1/chat/completions", json=MINIMAL_REQUEST)
    trace_id = resp.headers["x-trace-id"]
    await asyncio.sleep(0.05)
    await client.post(f"/v1/feedback/{trace_id}", json={"label": "incorrect"})

    body = (await client.get("/v1/calibration/curve")).json()
    assert len(body["buckets"]) == 10
    assert body["buckets"][0]["bin_lower"] == 0.0
    assert body["buckets"][-1]["bin_upper"] == 1.0
