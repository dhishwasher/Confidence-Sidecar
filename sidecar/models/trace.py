from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

CalibrationStatus = Literal["uncalibrated", "customer_calibrated", "global_calibrated"]


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

    # Raw score: weighted fusion of available signals, before calibration
    confidence_raw: float | None = None
    # Calibrated score: identity until enough feedback trains a curve
    confidence: float | None = None
    confidence_tier: int | None = None   # 0=low, 1=mid, 2=high

    # Calibration provenance
    confidence_method: str = "tier0_logprob_stop_v1"
    calibration_status: CalibrationStatus = "uncalibrated"

    stop_reason: str | None = None
    request_hash: str
    streaming: bool = False
    latency_ms: int | None = None
    upstream_latency_ms: int | None = None
    signals: list[SignalRecord] = []
