# Serving allpaka models

`allpaka serve` runs one inference owner with bounded admission, model-aware
scheduling, a byte-budgeted prefix cache, and one or more resident GGUF models.

## Reproducible startup

```sh
allpaka serve \
  --model models/qwen3-30b-a3b-Q4_K_M.gguf \
  --bind 127.0.0.1:8099 \
  --max-queued 256 \
  --max-batch 16 \
  --batch-context-tokens 32768 \
  --model-budget-gib 0 \
  --prefix-cache-mib 512
```

Repeat `--model` to make several models available. The model stem is the API
model id exposed by `GET /v1/models`. Startup rejects duplicate ids and rejects
the complete model set before Metal attachment when `--model-budget-gib` is
non-zero and too small.

The typed runtime profile comes from `allpaka.toml`. Without an explicit
configuration, allpaka applies a compatible cached autotune result for the
model and Metal device. An incompatible cache entry is ignored rather than
partially applied.

## Scheduling contract

`--max-queued` bounds accepted work. `--max-batch` and
`--batch-context-tokens` bound each model-aware scheduling decision. The
current batching mode is `model-aware-admission`: requests are grouped fairly
by model, then executed serially by the inference owner. It is not yet a
multi-session fused GPU kernel, and `GET /stats` reports
`"kernel_batching": false` explicitly.

## Prefix-cache contract

`--prefix-cache-mib` sets one global byte budget. Cache keys are namespaced by
model, so equal token ids from different models cannot alias. Entries contain
the prompt-boundary state required by the architecture and are evicted within
the byte budget. A zero budget disables retention.

`GET /stats` reports `prefix_cache_entries` and `prefix_cache_bytes`. These are
residency measurements, not a claim that every request was a cache hit.

## Acceleration and fallback

Normal serving follows the selected runtime fallback policy. Benchmark mode is
fail-closed: a requested GPU benchmark is invalid if Metal is unavailable or a
measured phase declines to CPU. Use `allpaka explain --model <model.gguf>` to
see model requirements, backend capabilities, tensor coverage, selected
profile, and exact acceleration decline reasons before serving.

The main introspection endpoints are:

- `GET /health`
- `GET /v1/models`
- `GET /stats`
- `POST /v1/chat/completions`

## Request lifetime and sessions

HTTP admission continues while inference runs. Readers are limited to 16, with
five-second socket timeouts and a 4 MiB body limit. Queue exhaustion returns
HTTP 429. Inference remains owned by one thread.

Chat requests may specify `request_id` (ASCII letters, digits, `-` or `_`, at
most 128 bytes), `timeout_ms` (1..600000, default 120000), and `session_id`
(non-empty string of at most 128 bytes). Active request ids must be unique.

`POST /v1/requests/{request_id}/cancel` cancels a queued or running request.
Cancellation and deadlines are checked between prefill chunks and decode
steps. An already submitted GPU command is allowed to finish. Before SSE
starts, interruption returns HTTP 408; during SSE it emits an error event and
`[DONE]`. Completion removes the request registration.

A model retains at most four named sessions. Idle sessions expire after ten
minutes and are collected on the next model request. A request without
`session_id` has independent KV state. Named sessions are scoped to the model
and reuse their own state. The service has no tenant authentication; expose it
only to trusted clients or place an authenticating proxy in front of it.

The combined prompt and generation reservation must fit 16384 tokens.
`usage.prompt_tokens_details.cached_tokens` reports reused prompt tokens.
Run `python3 scripts/test-serving-controls.py` with the local small Qwen model
to exercise isolation, reuse, cancellation, deadlines and recovery.
