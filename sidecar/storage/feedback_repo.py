"""CRUD for feedback records and calibration refit triggering."""
from __future__ import annotations

import asyncio
import logging
import time

import orjson

from sidecar.models.feedback import FeedbackRequest
from sidecar.storage.database import get_db

logger = logging.getLogger(__name__)


async def save_feedback(trace_id: str, customer_id: str, req: FeedbackRequest) -> None:
    """Persist a feedback label and fire-and-forget calibration refit if ready."""
    db = await get_db()
    await db.execute(
        """
        INSERT INTO feedback (trace_id, label, score, metadata, created_at, source)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            req.label,
            req.score,
            orjson.dumps(req.metadata).decode() if req.metadata else None,
            time.time(),
            req.source,
        ),
    )
    await db.commit()

    # Non-blocking: trigger calibration check in the background
    from sidecar.calibration.engine import maybe_retrain
    asyncio.create_task(_safe_retrain(customer_id))


async def _safe_retrain(customer_id: str) -> None:
    try:
        from sidecar.calibration.engine import maybe_retrain
        await maybe_retrain(customer_id)
    except Exception:
        logger.exception("Calibration refit task failed for %s", customer_id)
