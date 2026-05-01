"""GET /v1/calibration/curve — stub for Weeks 9-10 dashboard."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/calibration/curve")
async def calibration_curve() -> JSONResponse:
    return JSONResponse(
        {
            "status": "not_enough_data",
            "message": "Calibration curve requires feedback data. Submit labels via POST /v1/feedback/{trace_id}.",
            "buckets": [],
        }
    )
