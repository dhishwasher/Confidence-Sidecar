from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobToken(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] = []


class ChoiceLogprobs(BaseModel):
    content: list[LogprobToken] | None = None
    refusal: list[LogprobToken] | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None


class UsageStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageStats | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


# --- Streaming models ---

class DeltaMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    refusal: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    system_fingerprint: str | None = None
    service_tier: str | None = None
    usage: UsageStats | None = None


class ConfidenceChunk(BaseModel):
    """Extra SSE object injected by the proxy before [DONE].

    Standard OpenAI SDKs skip unknown object types, so this is invisible to
    unmodified client code.  Set CONFIDENCE_STREAM_MODE=store_only or
    disabled if your client is known to choke on extra SSE objects.

    Semantics note: ``confidence`` is the *token-distribution confidence*
    (generation certainty), not a claim about factual correctness.
    """

    object: Literal["chat.completion.confidence"] = "chat.completion.confidence"
    trace_id: str

    # Calibrated score (identity until a calibration curve is fitted)
    confidence: float
    # Raw signal fusion score before calibration
    confidence_raw: float
    confidence_tier: int  # 0=low(<0.3), 1=mid(0.3-0.7), 2=high(>0.7)

    confidence_method: str
    calibration_status: str

    # Per-signal breakdown
    signals: dict[str, float]
