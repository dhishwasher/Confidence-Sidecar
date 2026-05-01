"""GET /v1/calibration/curve — per-customer reliability diagram."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from sidecar.middleware.auth import get_customer_id
from sidecar.storage.database import get_db

router = APIRouter()

_N_BINS = 10
_MIN_SAMPLES_FOR_CURVE = 10  # below this we return data but flag low-confidence


@router.get("/calibration/curve")
async def calibration_curve(
    customer_id: str = Depends(get_customer_id),
) -> JSONResponse:
    """Reliability diagram for the authenticated customer.

    Bins ``confidence_raw`` scores into 10 equal-width buckets and computes
    the fraction of responses actually labelled correct in each bucket.
    A well-calibrated model has ``fraction_correct ≈ mean_confidence`` in
    every bucket (points close to the diagonal).

    Also returns Expected Calibration Error (ECE) and Brier score so
    callers can track calibration quality over time.
    """
    db = await get_db()

    async with db.execute(
        """
        SELECT t.confidence_raw, f.label
        FROM traces t
        JOIN feedback f ON f.trace_id = t.id
        WHERE t.customer_id = ?
          AND f.label IN ('correct', 'incorrect')
          AND t.confidence_raw IS NOT NULL
        ORDER BY t.confidence_raw
        """,
        (customer_id,),
    ) as cur:
        rows = await cur.fetchall()

    n = len(rows)

    if n == 0:
        return JSONResponse(
            {
                "status": "no_data",
                "message": (
                    "No labelled traces found for this customer. "
                    "Submit labels via POST /v1/feedback/{trace_id}."
                ),
                "n_samples": 0,
                "metrics": None,
                "buckets": [],
            }
        )

    scores = [float(r["confidence_raw"]) for r in rows]
    labels = [1 if r["label"] == "correct" else 0 for r in rows]

    # ── Reliability diagram buckets ───────────────────────────────────────────
    bin_counts = [0] * _N_BINS
    bin_correct = [0] * _N_BINS
    bin_conf_sum = [0.0] * _N_BINS

    for score, label in zip(scores, labels):
        idx = min(int(score * _N_BINS), _N_BINS - 1)
        bin_counts[idx] += 1
        bin_correct[idx] += label
        bin_conf_sum[idx] += score

    buckets = []
    for i in range(_N_BINS):
        count = bin_counts[i]
        mean_conf = bin_conf_sum[i] / count if count > 0 else None
        frac_correct = bin_correct[i] / count if count > 0 else None
        buckets.append(
            {
                "bin_lower": round(i / _N_BINS, 2),
                "bin_upper": round((i + 1) / _N_BINS, 2),
                "count": count,
                "mean_confidence": round(mean_conf, 4) if mean_conf is not None else None,
                "fraction_correct": round(frac_correct, 4) if frac_correct is not None else None,
            }
        )

    # ── Scalar metrics ────────────────────────────────────────────────────────
    ece = sum(
        (bin_counts[i] / n) * abs((bin_correct[i] / bin_counts[i]) - (bin_conf_sum[i] / bin_counts[i]))
        for i in range(_N_BINS)
        if bin_counts[i] > 0
    )
    brier = sum((s - l) ** 2 for s, l in zip(scores, labels)) / n

    # ── Calibration model status ──────────────────────────────────────────────
    async with db.execute(
        """
        SELECT model_type, trained_at, n_samples, brier_score, ece
        FROM calibration_params
        WHERE customer_id = ?
        ORDER BY trained_at DESC
        LIMIT 1
        """,
        (customer_id,),
    ) as cur:
        cal_row = await cur.fetchone()

    calibration_model = (
        {
            "model_type": cal_row["model_type"],
            "trained_at": cal_row["trained_at"],
            "n_samples": cal_row["n_samples"],
            "brier_score": cal_row["brier_score"],
            "ece": cal_row["ece"],
        }
        if cal_row
        else None
    )

    status = "ok" if n >= _MIN_SAMPLES_FOR_CURVE else "low_sample_count"

    return JSONResponse(
        {
            "status": status,
            "n_samples": n,
            "metrics": {
                "ece": round(ece, 4),
                "brier_score": round(brier, 4),
            },
            "calibration_model": calibration_model,
            "buckets": buckets,
        }
    )
