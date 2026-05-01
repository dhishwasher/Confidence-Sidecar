"""CRUD for traces and signals."""
from __future__ import annotations

import time

import orjson

from sidecar.models.trace import SignalRecord, TraceRecord
from sidecar.storage.database import get_db


async def save_trace(trace: TraceRecord) -> None:
    db = await get_db()
    await db.execute(
        """
        INSERT INTO traces (
            id, customer_id, created_at, model, provider,
            prompt_tokens, completion_tokens, tier,
            confidence, confidence_raw, confidence_tier,
            confidence_method, calibration_status,
            stop_reason, request_hash, streaming,
            latency_ms, upstream_latency_ms
        ) VALUES (
            :id, :customer_id, :created_at, :model, :provider,
            :prompt_tokens, :completion_tokens, :tier,
            :confidence, :confidence_raw, :confidence_tier,
            :confidence_method, :calibration_status,
            :stop_reason, :request_hash, :streaming,
            :latency_ms, :upstream_latency_ms
        )
        """,
        {
            "id": trace.id,
            "customer_id": trace.customer_id,
            "created_at": trace.created_at,
            "model": trace.model,
            "provider": trace.provider,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "tier": trace.tier,
            "confidence": trace.confidence,
            "confidence_raw": trace.confidence_raw,
            "confidence_tier": trace.confidence_tier,
            "confidence_method": trace.confidence_method,
            "calibration_status": trace.calibration_status,
            "stop_reason": trace.stop_reason,
            "request_hash": trace.request_hash,
            "streaming": int(trace.streaming),
            "latency_ms": trace.latency_ms,
            "upstream_latency_ms": trace.upstream_latency_ms,
        },
    )
    now = time.time()
    for sig in trace.signals:
        await db.execute(
            """
            INSERT INTO signals (trace_id, signal_name, signal_value, signal_metadata, computed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                trace.id,
                sig.signal_name,
                sig.signal_value,
                orjson.dumps(sig.signal_metadata).decode() if sig.signal_metadata else None,
                sig.computed_at if sig.computed_at else now,
            ),
        )
    await db.commit()


async def get_trace(trace_id: str) -> TraceRecord | None:
    db = await get_db()
    async with db.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)) as cur:
        row = await cur.fetchone()
    if row is None:
        return None

    trace_data = dict(row)
    trace_data["streaming"] = bool(trace_data["streaming"])
    # Provide defaults for older rows that predate migration 003
    trace_data.setdefault("confidence_method", "tier0_logprob_stop_v1")
    trace_data.setdefault("calibration_status", "uncalibrated")

    async with db.execute(
        "SELECT * FROM signals WHERE trace_id = ? ORDER BY computed_at", (trace_id,)
    ) as cur:
        signal_rows = await cur.fetchall()

    signals = [
        SignalRecord(
            signal_name=r["signal_name"],
            signal_value=r["signal_value"],
            signal_metadata=orjson.loads(r["signal_metadata"]) if r["signal_metadata"] else None,
            computed_at=r["computed_at"],
        )
        for r in signal_rows
    ]
    trace_data["signals"] = signals
    return TraceRecord.model_validate(trace_data)
