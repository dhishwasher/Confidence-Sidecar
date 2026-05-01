"""Platt scaling calibration trainer (stub for Weeks 7-8).

Requires the 'calibration' extras: scikit-learn, numpy.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class InsufficientLabelDiversityError(ValueError):
    """Raised when all feedback labels belong to a single class.

    Logistic regression requires at least one positive and one negative
    example.  Callers should record 'insufficient_label_diversity' as the
    calibration_status rather than crashing or silently training on bad data.
    """


async def train_calibration(
    customer_id: str,
    raw_scores: list[float],
    labels: list[int],  # 1=correct, 0=incorrect
) -> dict:
    """Fit Platt scaling (logistic regression) on raw_scores → labels.

    Runs in a threadpool executor to avoid blocking the event loop.
    Returns a dict with 'a', 'b' params for sigmoid(a*x + b) mapping.

    Raises:
        InsufficientLabelDiversityError: if all labels are the same class.
    """
    if len(set(labels)) < 2:
        raise InsufficientLabelDiversityError(
            f"All {len(labels)} feedback labels are '{labels[0]}'. "
            "Need at least one 'correct' and one 'incorrect' sample to fit calibration."
        )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fit_platt, raw_scores, labels)


def _fit_platt(raw_scores: list[float], labels: list[int]) -> dict:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "scikit-learn and numpy required for calibration. "
            "Install with: pip install 'confidence-sidecar[calibration]'"
        ) from exc

    X = np.array(raw_scores).reshape(-1, 1)
    y = np.array(labels)
    clf = LogisticRegression(C=1e10)
    clf.fit(X, y)
    a = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    return {"a": a, "b": b}
