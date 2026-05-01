"""POST /v1/chat/completions — the core proxy endpoint."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sidecar.middleware.auth import get_customer_id
from sidecar.models.openai import ChatCompletionRequest
from sidecar.models.trace import SignalRecord, TraceRecord
from sidecar.providers.openai import get_openai_provider, inject_logprobs, strip_logprobs_from_response
from sidecar.signals.combiner import classify_confidence_tier, combine_signals
from sidecar.signals.logprob_entropy import compute_logprob_entropy, entropy_to_confidence
from sidecar.signals.stop_reason import compute_stop_reason_signal
from sidecar.storage.trace_repo import save_trace
from sidecar.streaming.sse_emitter import emit_streaming_response

logger = logging.getLogger(__name__)
router = APIRouter()


def _make_request_hash(messages: list) -> str:
    payload = json.dumps([m if isinstance(m, dict) else m.model_dump() for m in messages], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


async def _save_trace_bg(trace: TraceRecord) -> None:
    try:
        await save_trace(trace)
    except Exception:
        logger.exception("Failed to save trace %s", trace.id)


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    customer_id: str = Depends(get_customer_id),
):
    body = await request.json()
    try:
        parsed = ChatCompletionRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    customer_wants_logprobs = bool(parsed.logprobs)
    trace_id = "tr_" + uuid4().hex
    request_hash = _make_request_hash(parsed.messages)
    started_at = time.time()

    upstream_body, _ = inject_logprobs(body)
    provider = get_openai_provider()
    inbound_headers = dict(request.headers)

    if parsed.stream:
        upstream_started_at = time.time()

        async def byte_stream():
            async for chunk in provider.stream(upstream_body, inbound_headers):
                yield chunk

        return StreamingResponse(
            emit_streaming_response(
                upstream_bytes=byte_stream(),
                trace_id=trace_id,
                customer_id=customer_id,
                customer_wants_logprobs=customer_wants_logprobs,
                request_model=parsed.model,
                request_hash=request_hash,
                started_at=started_at,
                upstream_started_at=upstream_started_at,
                save_trace_fn=save_trace,
            ),
            media_type="text/event-stream",
            headers={
                "X-Trace-Id": trace_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming path
    upstream_started_at = time.time()
    try:
        resp = await provider.complete(upstream_body, inbound_headers)
    except Exception as exc:
        logger.exception("Upstream error")
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")
    upstream_latency_ms = int((time.time() - upstream_started_at) * 1000)

    data = resp.json()
    choices = data.get("choices", [])
    finish_reason = choices[0].get("finish_reason") if choices else None

    # Compute Tier 0 signals
    entropy = compute_logprob_entropy(choices)
    lp_conf = entropy_to_confidence(entropy) if entropy is not None else None
    sr_conf = compute_stop_reason_signal(finish_reason)

    signals_map: dict[str, float | None] = {
        "logprob_entropy": lp_conf,
        "stop_reason": sr_conf,
    }
    confidence = combine_signals(signals_map, tier=0)
    conf_tier = classify_confidence_tier(confidence)

    if not customer_wants_logprobs:
        strip_logprobs_from_response(data)

    now = time.time()
    signal_records = [
        SignalRecord(signal_name="stop_reason", signal_value=sr_conf, computed_at=now),
    ]
    if lp_conf is not None:
        signal_records.append(
            SignalRecord(signal_name="logprob_entropy", signal_value=lp_conf, computed_at=now)
        )

    usage = data.get("usage") or {}
    trace = TraceRecord(
        id=trace_id,
        customer_id=customer_id,
        created_at=started_at,
        model=parsed.model,
        provider="openai",
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        tier=0,
        confidence=round(confidence, 4),
        confidence_raw=round(confidence, 4),
        confidence_tier=conf_tier,
        stop_reason=finish_reason,
        request_hash=request_hash,
        streaming=False,
        latency_ms=int((now - started_at) * 1000),
        upstream_latency_ms=upstream_latency_ms,
        signals=signal_records,
    )
    asyncio.create_task(_save_trace_bg(trace))

    return JSONResponse(
        content=data,
        headers={
            "X-Confidence": f"{confidence:.4f}",
            "X-Confidence-Tier": str(conf_tier),
            "X-Signal-Logprob-Entropy": f"{entropy:.4f}" if entropy is not None else "n/a",
            "X-Trace-Id": trace_id,
        },
    )
