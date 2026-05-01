from __future__ import annotations

from pydantic import BaseModel


class SignalRecord(BaseModel):
    signal_name: str
    signal_value: float
    signal_metadata: dict | None = None
    computed_at: float


class TraceRecord(BaseModel):
    id: str
    customer_id: str
    created_at: float
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tier: int = 0
    confidence: float | None = None
    confidence_raw: float | None = None
    confidence_tier: int | None = None
    stop_reason: str | None = None
    request_hash: str
    streaming: bool = False
    latency_ms: int | None = None
    upstream_latency_ms: int | None = None
    signals: list[SignalRecord] = []
