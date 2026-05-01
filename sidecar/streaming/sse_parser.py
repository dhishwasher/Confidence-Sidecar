"""SSE stream parser that accumulates logprobs from OpenAI streaming responses."""
from __future__ import annotations

from typing import Any

import orjson


class SSEAccumulator:
    """Parses raw SSE bytes, accumulates logprob data, optionally strips logprobs.

    Usage::

        acc = SSEAccumulator(customer_wants_logprobs=False)
        async for raw_bytes in upstream:
            for line in acc.feed(raw_bytes):
                # line is already-forwarded SSE payload (bytes), or None for [DONE]
                if line is not None:
                    yield line

        entropy = compute_logprob_entropy(acc.build_choices_snapshot())
    """

    def __init__(self, customer_wants_logprobs: bool) -> None:
        self._customer_wants_logprobs = customer_wants_logprobs
        self._buffer = b""

        # Accumulated data for signal computation
        self.logprob_tokens: list[dict[str, Any]] = []  # list of LogprobToken dicts
        self.finish_reason: str | None = None
        self.completion_id: str | None = None
        self.model: str | None = None
        self.done: bool = False

    def feed(self, raw: bytes) -> list[bytes | None]:
        """Feed raw bytes from the upstream SSE stream.

        Returns a list of forwarded SSE lines (as bytes, already formatted as
        ``b"data: ...\\n\\n"``).  A ``None`` entry means the ``[DONE]`` sentinel
        was encountered — callers must NOT forward it; they should inject the
        confidence chunk first.
        """
        self._buffer += raw
        results: list[bytes | None] = []

        while True:
            # SSE events are separated by double newlines
            for sep in (b"\r\n\r\n", b"\n\n"):
                idx = self._buffer.find(sep)
                if idx != -1:
                    event_bytes = self._buffer[: idx + len(sep)]
                    self._buffer = self._buffer[idx + len(sep):]
                    out = self._process_event(event_bytes)
                    results.append(out)  # None signals [DONE] to callers
                    break
            else:
                break  # no complete event found; keep buffering

        return results

    def _process_event(self, event: bytes) -> bytes | None:
        """Parse one complete SSE event.

        Returns:
            - Modified event bytes to forward to the client, or
            - ``None`` to signal [DONE] interception (do not forward).
        """
        lines = event.split(b"\n")
        data_line: bytes | None = None
        for line in lines:
            if line.startswith(b"data:"):
                data_line = line[5:].strip()
                break

        if data_line is None:
            # Keep-alive / comment lines — forward as-is
            return event

        if data_line == b"[DONE]":
            self.done = True
            return None  # caller will inject confidence chunk then re-emit [DONE]

        try:
            payload: dict[str, Any] = orjson.loads(data_line)
        except Exception:
            return event  # malformed; forward unchanged

        # Capture metadata from first chunk
        if self.completion_id is None:
            self.completion_id = payload.get("id")
            self.model = payload.get("model")

        choices: list[dict[str, Any]] = payload.get("choices") or []
        for choice in choices:
            # Accumulate logprobs
            logprobs_obj = choice.get("logprobs") or {}
            for token_entry in logprobs_obj.get("content") or []:
                self.logprob_tokens.append(token_entry)

            # Capture finish reason
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

            # Strip logprobs from forwarded payload if customer didn't want them
            if not self._customer_wants_logprobs:
                choice["logprobs"] = None

        forwarded = orjson.dumps(payload)
        return b"data: " + forwarded + b"\n\n"

    def flush(self) -> list[bytes | None]:
        """Process any bytes remaining in the buffer after the upstream stream closes.

        Real OpenAI always terminates with ``\\n\\n``, but defensive flushing
        handles proxies or tests that omit the trailing separator.
        """
        if self._buffer.strip():
            return self.feed(b"\n\n")
        return []

    def build_choices_snapshot(self) -> list[dict[str, Any]]:
        """Return a minimal choices structure compatible with logprob_entropy computation."""
        return [
            {
                "logprobs": {"content": self.logprob_tokens},
                "finish_reason": self.finish_reason,
            }
        ]
