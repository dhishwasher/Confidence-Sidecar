"""Calibration engine: look up per-customer Platt scaling params and apply them.

Day-1 behaviour (no feedback yet): identity mapping, status = "uncalibrated".
After CALIBRATION_TRIGGER_SAMPLES feedback labels: per-customer logistic
regression is fitted and applied, status = "customer_calibrated".
A global model (customer_id = "__global__") can be pre-seeded from public
benchmarks to give new customers something better than identity from day 1.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time

import orjson

from sidecar.config import settings

logger = logging.getLogger(__name__)

_GLOBAL_CUSTOMER = "__global__"


async def calibrate(
    raw_confidence: float,
    customer_id: str,
) -> tuple[float, str]:
    """Return *(calibrated_confidence, calibration_status)*.

    Precedence:
    1. Customer-specific Platt scaling (if trained)
    2. Global Platt scaling (if seeded)
    3. Identity — returns raw_confidence unchanged
    """
    # Lazy import to avoid circular dependency at module load
    from sidecar.storage.database import get_db

    db = await get_db()

    # Try customer-specific params first
    params = await _load_params(db, customer_id)
    if params:
        return _apply_platt(raw_confidence, params), "customer_calibrated"

    # Fall back to global
    global_params = await _load_params(db, _GLOBAL_CUSTOMER)
    if global_params:
        return _apply_platt(raw_confidence, global_params), "global_calibrated"

    return raw_confidence, "uncalibrated"


async def maybe_retrain(customer_id: str) -> None:
    """Trigger calibration refit if enough feedback has accumulated.

    Called fire-and-forget from the feedback router.
    """
    from sidecar.storage.database import get_db

    db = await get_db()

    # Count labelled feedback rows that have a matching raw confidence value
    async with db.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM feedback f
        JOIN traces t ON f.trace_id = t.id
        WHERE t.customer_id = ?
          AND f.label IN ('correct', 'incorrect')
          AND t.confidence_raw IS NOT NULL
        """,
        (customer_id,),
    ) as cur:
        row = await cur.fetchone()

    count = row["cnt"] if row else 0
    threshold = settings.calibration_trigger_samples

    if count < threshold:
        return

    # Only retrain every 10 new samples past the threshold to avoid thrashing
    if count % 10 != 0:
        return

    logger.info("Triggering calibration refit for %s (%d samples)", customer_id, count)

    async with db.execute(
        """
        SELECT t.confidence_raw AS raw_score, f.label
        FROM feedback f
        JOIN traces t ON f.trace_id = t.id
        WHERE t.customer_id = ?
          AND f.label IN ('correct', 'incorrect')
          AND t.confidence_raw IS NOT NULL
        """,
        (customer_id,),
    ) as cur:
        rows = await cur.fetchall()

    raw_scores = [r["raw_score"] for r in rows]
    labels = [1 if r["label"] == "correct" else 0 for r in rows]

    # Guard: logistic regression requires at least one sample from each class
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        only_class = "correct" if 1 in unique_labels else "incorrect"
        logger.warning(
            "Skipping calibration refit for %s: all %d labels are '%s'. "
            "Need both correct and incorrect examples.",
            customer_id, len(labels), only_class,
        )
        return

    loop = asyncio.get_event_loop()
    try:
        from sidecar.calibration.platt import _fit_platt
        params = await loop.run_in_executor(None, _fit_platt, raw_scores, labels)
    except ImportError:
        logger.warning("scikit-learn not installed; skipping calibration refit")
        return
    except Exception:
        logger.exception("Calibration refit failed for %s", customer_id)
        return

    await db.execute(
        """
        INSERT INTO calibration_params (customer_id, model_type, params, trained_at, n_samples)
        VALUES (?, 'platt', ?, ?, ?)
        """,
        (customer_id, orjson.dumps(params).decode(), time.time(), len(rows)),
    )
    await db.commit()
    logger.info("Calibration refit complete for %s: %s", customer_id, params)


# ── helpers ────────────────────────────────────────────────────────────────────

async def _load_params(db, customer_id: str) -> dict | None:
    async with db.execute(
        "SELECT params FROM calibration_params WHERE customer_id = ? ORDER BY trained_at DESC LIMIT 1",
        (customer_id,),
    ) as cur:
        row = await cur.fetchone()
    return orjson.loads(row["params"]) if row else None


def _apply_platt(raw: float, params: dict) -> float:
    a = params.get("a", 1.0)
    b = params.get("b", 0.0)
    return 1.0 / (1.0 + math.exp(-(a * raw + b)))
