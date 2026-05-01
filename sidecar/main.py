import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sidecar.config import settings
from sidecar.storage.database import close_db, init_db

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uses settings.database_url by default (no argument needed)
    await init_db()
    logger.info("Confidence Sidecar started")
    yield
    await close_db()
    logger.info("Confidence Sidecar stopped")


app = FastAPI(title="Confidence Sidecar", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


from sidecar.routers import calibration, feedback, proxy, traces  # noqa: E402

app.include_router(proxy.router, prefix="/v1")
app.include_router(traces.router, prefix="/v1")
app.include_router(feedback.router, prefix="/v1")
app.include_router(calibration.router, prefix="/v1")
