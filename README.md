# Confidence Sidecar

A drop-in OpenAI-compatible proxy that adds **calibrated generation-confidence telemetry** to LLM responses — without requiring any application code changes.

Point your existing OpenAI client at the sidecar instead of `api.openai.com`. You get the same response shape back, plus confidence metadata in headers and an optional streaming chunk.

---

## What it does

Every request through the sidecar gets a **token-distribution confidence score**: a measure of how peaked the model's own next-token distribution was across the response. High score means the model committed strongly to each token. Low score means it was hedging.

**This is not hallucination detection and it is not factual correctness.** A model can be token-certain while producing a confident hallucination. The score is labelled `confidence_method: tier0_logprob_stop_v1` so consumers know exactly what they are reading.

What it is useful for:
- Flagging responses that warrant human review or a follow-up call
- Routing uncertain responses to a stronger model
- Building a per-customer reliability curve over time using your own feedback labels
- Catching responses that were cut off (`finish_reason: length`) before full completion

---

## Quick start

```bash
git clone https://github.com/dhishwasher/Confidence-Sidecar
cd Confidence-Sidecar
pip install -e .
cp .env.example .env          # add your OPENAI_API_KEY
uvicorn sidecar.main:app --reload
```

Then point your client at `http://localhost:8000` instead of `https://api.openai.com`.

---

## Demo

### High-confidence response

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer test" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }' -i
```

```
HTTP/1.1 200 OK
X-Confidence: 0.9731
X-Confidence-Raw: 0.9731
X-Confidence-Tier: 2
X-Confidence-Method: tier0_logprob_stop_v1
X-Calibration-Status: uncalibrated
X-Signal-Logprob-Entropy: 0.0269
X-Trace-Id: tr_a3f2...

{"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop", ...}]}
```

### Low-confidence response

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer test" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What will the stock market do tomorrow?"}]
  }' -i
```

```
X-Confidence: 0.4102
X-Confidence-Raw: 0.4102
X-Confidence-Tier: 1
X-Confidence-Method: tier0_logprob_stop_v1
X-Calibration-Status: uncalibrated
X-Signal-Logprob-Entropy: 0.5898
X-Trace-Id: tr_9c1d...
```

---

## Response headers

| Header | Description |
|---|---|
| `X-Confidence` | Calibrated score in [0, 1]. Identity until a per-customer curve is fitted. |
| `X-Confidence-Raw` | Raw signal-fusion score before calibration. |
| `X-Confidence-Tier` | `0` = low (< 0.3), `1` = mid (0.3–0.7), `2` = high (> 0.7) |
| `X-Confidence-Method` | Signal method identifier. Currently `tier0_logprob_stop_v1`. |
| `X-Calibration-Status` | `uncalibrated` / `customer_calibrated` / `global_calibrated` |
| `X-Signal-Logprob-Entropy` | Raw mean normalised token entropy (lower = more certain) |
| `X-Trace-Id` | Opaque trace identifier for fetching full signal breakdown |

---

## Streaming

By default the sidecar injects a `chat.completion.confidence` SSE object immediately before `[DONE]`. Standard OpenAI SDKs ignore unknown object types, so this is invisible to unmodified client code.

```
data: {"object":"chat.completion.confidence","trace_id":"tr_...","confidence":0.91,...}

data: [DONE]
```

### Stream modes

Set `CONFIDENCE_STREAM_MODE` in your `.env`:

| Mode | Behaviour |
|---|---|
| `chunk` (default) | Inject confidence SSE chunk before `[DONE]` |
| `store_only` | Compute and store confidence but do not inject into the SSE body. Fetch the score afterwards via `GET /v1/traces/{trace_id}/confidence` using the `X-Trace-Id` header. Safe for clients that reject unknown SSE object types. |
| `disabled` | Pure transparent pass-through. No logprob injection upstream, no parsing, no tracing, no SSE modification. |

---

## Endpoints

```
POST /v1/chat/completions                      OpenAI-compatible proxy
GET  /v1/traces/{trace_id}                     Full signal breakdown for a trace
GET  /v1/traces/{trace_id}/confidence          Lightweight confidence fetch (for store_only clients)
POST /v1/feedback/{trace_id}                   Submit a ground-truth label
GET  /v1/calibration/curve                     Per-customer reliability diagram (coming soon)
GET  /health                                   Liveness probe
```

### Fetching confidence after a stored stream

```bash
# 1. Open the stream — capture X-Trace-Id from response headers
TRACE_ID=$(curl -si ... | grep -i x-trace-id | awk '{print $2}' | tr -d '\r')

# 2. After the stream closes, fetch the score
curl http://localhost:8000/v1/traces/$TRACE_ID/confidence
```

```json
{
  "trace_id": "tr_9c1d...",
  "confidence": 0.91,
  "confidence_raw": 0.87,
  "confidence_tier": 2,
  "confidence_method": "tier0_logprob_stop_v1",
  "calibration_status": "uncalibrated",
  "signals": {"logprob_entropy": 0.89, "stop_reason": 1.0}
}
```

### Submitting feedback

```bash
curl -X POST http://localhost:8000/v1/feedback/tr_9c1d... \
  -H "Content-Type: application/json" \
  -d '{"label": "correct"}'         # "correct" | "incorrect" | "partial"
```

After 50 correct/incorrect labels accumulate for a customer, the sidecar automatically fits a Platt scaling curve and `X-Calibration-Status` transitions from `uncalibrated` to `customer_calibrated`.

---

## Environment variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
SIDECAR_API_KEY=              # empty = dev mode, any bearer token accepted
DATABASE_URL=sqlite+aiosqlite:///./traces.db
CONFIDENCE_STREAM_MODE=chunk  # chunk | store_only | disabled
TOP_LOGPROBS_COUNT=5          # top-K alternatives to request per token
CALIBRATION_TRIGGER_SAMPLES=50
LOG_LEVEL=INFO
```

---

## How the score is computed (Tier 0)

1. **Logprob entropy** — the sidecar always requests `top_logprobs=5` from OpenAI (stripped from the response if you did not ask for them). For each output token it computes normalised Shannon entropy over the top-K distribution, adding the residual probability mass outside top-K as a single "other" bucket. This makes the score conservative: it cannot look more certain than the model actually was.

2. **Stop-reason signal** — `stop` = 1.0, `tool_calls` = 0.9, `length` = 0.65, `content_filter` = 0.5.

3. **Weighted fusion** — `0.85 × logprob_confidence + 0.15 × stop_reason`.

4. **Calibration** — identity mapping until per-customer Platt scaling is fitted from feedback labels.

---

## Limitations

- **Not factual correctness.** The score reflects the model's own token-distribution certainty. A model that hallucinations confidently will score high.
- **Anthropic provider not yet wired.** Anthropic does not expose raw logprobs, so Tier 0 falls back to stop-reason only. Tier 1 semantic entropy (Weeks 3–4) will carry the load for Anthropic traffic.
- **Single-tenant auth.** With `SIDECAR_API_KEY` set, all traffic is treated as one tenant. Multi-tenant key registration is not yet implemented.
- **Semantic entropy (Tier 1) not active.** The module exists but requires `pip install 'confidence-sidecar[tier1]'` and explicit wiring. The score today is Tier 0 only.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
