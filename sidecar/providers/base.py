from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx


class AbstractProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        request_body: dict[str, Any],
        upstream_headers: dict[str, str],
    ) -> httpx.Response:
        """Send a non-streaming completion request. Returns the raw httpx response."""

    @abstractmethod
    async def stream(
        self,
        request_body: dict[str, Any],
        upstream_headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        """Send a streaming completion request. Yields raw SSE bytes."""
