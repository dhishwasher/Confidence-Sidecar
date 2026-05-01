from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from sidecar.models.feedback import FeedbackRequest
from sidecar.storage.feedback_repo import save_feedback
from sidecar.storage.trace_repo import get_trace

router = APIRouter()


@router.post("/feedback/{trace_id}")
async def submit_feedback(trace_id: str, body: FeedbackRequest) -> JSONResponse:
    trace = await get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    await save_feedback(trace_id, body)
    return JSONResponse({"status": "accepted", "trace_id": trace_id})
