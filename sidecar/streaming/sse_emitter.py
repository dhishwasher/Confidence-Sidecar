"""Streaming response emitter with confidence chunk injection."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

import orjson

from sidecar.models.openai import ConfidenceChunk
from sidecar.models.trace import SignalRecord, TraceRecord
from sidecar.signals.combiner import classify_confidence_tier, combine_signals
from sidecar.signals.logprob_entropy import compute_logprob_entropy, entropy_to_confidence
from sidecar.signals.stop_reason import compute_stop_reason_signal
from sidecar.streaming.sse_parser import SSEAccumulator

logger = logging.getLogger(__name__)


async def emit_streaming_response(
    upstream_bytes: AsyncIterator[bytes],
    trace_id: str,
    customer_id: str,
    customer_wants_logprobs: bool,
    request_model: str,
    request_hash: str,
    started_at: float,
    upstream_started_at: float,
    save_trace_fn,  # coroutine: async (TraceRecord) -> None
) -> AsyncIterator[bytes]:
    """Async generator for the streaming proxy response.

    Forwards upstream SSE chunks, intercepts [DONE], injects a confidence
    SSE chunk, then re-emits [DONE].  Fires trace save as a background task.
    """
    acc = SSEAccumulator(customer_wants_logprobs=customer_wants_logprobs)

    async for raw in upstream_bytes:
        for item in acc.feed(raw):
            if item is None:
                async for chunk in _emit_confidence_and_done(
                    acc, trace_id, customer_id, request_model,
                    request_hash, started_at, upstream_started_at, save_trace_fn
                ):
                    yield chunk
                return
            yield item

    # Defensive: flush any bytes still buffered (upstream omitted trailing \n\n)
    for item in acc.flush():
        if item is None:
            async for chunk in _emit_confidence_and_done(
                acc, trace_id, customer_id, request_model,
                request_hash, started_at, upstream_started_at, save_trace_fn
            ):
                yield chunk
            return
        yield item


async def _emit_confidence_and_done(
    acc: SSEAccumulator,
    trace_id: str,
    customer_id: str,
    request_model: str,
    request_hash: str,
    started_at: float,
    upstream_started_at: float,
    save_trace_fn,
) -> AsyncIterator[bytes]:
    choices_snapshot = acc.build_choices_snapshot()
    entropy = compute_logprob_entropy(choices_snapshot)
    lp_conf = entropy_to_confidence(entropy) if entropy is not None else None
    sr_conf = compute_stop_reason_signal(acc.finish_reason)

    signals_map: dict[str, float | None] = {
        "logprob_entropy": lp_conf,
        "stop_reason": sr_conf,
    }
    confidence = combine_signals(signals_map, tier=0)
    conf_tier = classify_confidence_tier(confidence)

    now = time.time()
    signal_records = [
        SignalRecord(signal_name="stop_reason", signal_value=sr_conf, computed_at=now),
    ]
    if lp_conf is not None:
        signal_records.append(
            SignalRecord(signal_name="logprob_entropy", signal_value=lp_conf, computed_at=now)
        )

    conf_chunk = ConfidenceChunk(
        trace_id=trace_id,
        confidence=round(confidence, 4),
        confidence_tier=conf_tier,
        signals={k: round(v, 4) for k, v in signals_map.items() if v is not None},
    )
    yield b"data: " + orjson.dumps(conf_chunk.model_dump()) + b"\n\n"
    yield b"data: [DONE]\n\n"

    trace = TraceRecord(
        id=trace_id,
        customer_id=customer_id,
        created_at=started_at,
        model=request_model,
        provider="openai",
        tier=0,
        confidence=round(confidence, 4),
        confidence_raw=round(confidence, 4),
        confidence_tier=conf_tier,
        stop_reason=acc.finish_reason,
        request_hash=request_hash,
        streaming=True,
        latency_ms=int((now - started_at) * 1000),
        upstream_latency_ms=int((now - upstream_started_at) * 1000),
        signals=signal_records,
    )
    asyncio.create_task(_safe_save(save_trace_fn, trace))


async def _safe_save(save_fn, trace: TraceRecord) -> None:
    try:
        await save_fn(trace)
    except Exception:
        logger.exception("Failed to save trace %s", trace.id)
