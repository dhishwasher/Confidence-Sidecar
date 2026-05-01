from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    label: Literal["correct", "incorrect", "partial"]
    score: float | None = None
    metadata: dict | None = None
    source: str = "human"
