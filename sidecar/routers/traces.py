from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from sidecar.models.trace import TraceRecord
from sidecar.storage.trace_repo import get_trace

router = APIRouter()


@router.get("/traces/{trace_id}", response_model=TraceRecord)
async def get_trace_endpoint(trace_id: str) -> TraceRecord:
    trace = await get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@router.get("/traces/{trace_id}/confidence")
async def get_trace_confidence(trace_id: str) -> JSONResponse:
    """Lightweight confidence fetch for streaming clients using store_only mode.

    After a streamed response completes, clients that did not receive an inline
    confidence chunk can poll this endpoint with the X-Trace-Id they received
    in the initial response headers.

    Returns as soon as the background trace save has committed.  Clients
    should retry with a short backoff if they receive 404 immediately after
    stream completion (the save is fire-and-forget and may take ~50 ms).
    """
    trace = await get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return JSONResponse({
        "trace_id": trace.id,
        "confidence": trace.confidence,
        "confidence_raw": trace.confidence_raw,
        "confidence_tier": trace.confidence_tier,
        "confidence_method": trace.confidence_method,
        "calibration_status": trace.calibration_status,
        "signals": {s.signal_name: s.signal_value for s in trace.signals},
    })
