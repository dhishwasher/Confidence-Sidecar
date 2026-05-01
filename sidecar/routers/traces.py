from fastapi import APIRouter, HTTPException
from sidecar.models.trace import TraceRecord
from sidecar.storage.trace_repo import get_trace

router = APIRouter()


@router.get("/traces/{trace_id}", response_model=TraceRecord)
async def get_trace_endpoint(trace_id: str) -> TraceRecord:
    trace = await get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
