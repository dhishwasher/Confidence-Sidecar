from __future__ import annotations

import copy
from typing import Any, AsyncIterator

import httpx

from sidecar.config import settings
from sidecar.providers.base import AbstractProvider

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    return _client


def inject_logprobs(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Ensure logprobs=True and top_logprobs >= settings.top_logprobs_count.

    Returns the modified body and a flag indicating whether the customer
    originally requested logprobs.
    """
    customer_wanted = bool(body.get("logprobs"))
    body = copy.deepcopy(body)
    body["logprobs"] = True
    body["top_logprobs"] = max(
        settings.top_logprobs_count,
        body.get("top_logprobs") or 0,
    )
    return body, customer_wanted


def strip_logprobs_from_response(data: dict[str, Any]) -> None:
    """Remove logprob data from a non-streaming response dict in-place."""
    for choice in data.get("choices", []):
        choice["logprobs"] = None


class OpenAIProvider(AbstractProvider):
    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.upstream_openai_base_url).rstrip("/")

    def _upstream_headers(self, customer_headers: dict[str, str]) -> dict[str, str]:
        headers = {
            k: v
            for k, v in customer_headers.items()
            if k.lower() not in {"host", "content-length", "authorization"}
        }
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    async def complete(
        self,
        request_body: dict[str, Any],
        upstream_headers: dict[str, str],
    ) -> httpx.Response:
        url = f"{self._base_url}/v1/chat/completions"
        resp = await get_client().post(
            url,
            json=request_body,
            headers=self._upstream_headers(upstream_headers),
        )
        resp.raise_for_status()
        return resp

    async def stream(
        self,
        request_body: dict[str, Any],
        upstream_headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        url = f"{self._base_url}/v1/chat/completions"
        async with get_client().stream(
            "POST",
            url,
            json=request_body,
            headers=self._upstream_headers(upstream_headers),
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk


_provider: OpenAIProvider | None = None


def get_openai_provider() -> OpenAIProvider:
    global _provider
    if _provider is None:
        _provider = OpenAIProvider()
    return _provider
