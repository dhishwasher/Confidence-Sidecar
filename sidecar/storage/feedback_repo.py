"""CRUD for feedback records."""
from __future__ import annotations

import time

import orjson

from sidecar.models.feedback import FeedbackRequest
from sidecar.storage.database import get_db


async def save_feedback(trace_id: str, req: FeedbackRequest) -> None:
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
