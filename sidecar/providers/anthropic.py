"""Anthropic provider stub (Weeks 5-6).

Anthropic does not expose raw token logprobs via their API, so the
logprob_entropy signal will always be None for Anthropic traffic.
Tier 0 confidence falls back to stop_reason only (15% weight → weaker signal).
Tier 1 semantic entropy carries the load once it is wired up.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from sidecar.config import settings
from sidecar.providers.base import AbstractProvider


class AnthropicProvider(AbstractProvider):
    """Anthropic Messages API → OpenAI-compatible response adapter.

    Not yet implemented. Placeholder raises NotImplementedError.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.upstream_anthropic_base_url).rstrip("/")

    async def complete(
        self,
        request_body: dict[str, Any],
        upstream_headers: dict[str, str],
    ) -> httpx.Response:
        raise NotImplementedError("Anthropic provider not yet implemented (Weeks 5-6)")

    async def stream(
        self,
        request_body: dict[str, Any],
        upstream_headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("Anthropic provider not yet implemented (Weeks 5-6)")
        yield  # make this an async generator
