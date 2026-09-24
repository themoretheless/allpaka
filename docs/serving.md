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

## Watching a live server

`allpaka watch` polls `/health`, `/v1/models`, `/stats` and `/resources` and
prints the transitions between them: a one-word `phase` plus the conditions that
moved. It reads only those endpoints, changes nothing, and stops with the
process.

```
15:09:27  phase=Ready       health=0ms models=0ms stats=0ms resources=0ms context=0/1 mem=1.2 GiB/uncapped
15:09:28  phase=Generating  Ready -> Generating  +Generating (model lock held 300ms without answering)  health=0ms ...
15:09:29  phase=Ready       Generating -> Ready  -Generating  health=0ms ...
```

The conditions are `Reachable`, `ModelsAdmitted`, `EngineResponsive`,
`ResourcesSampled`, `Generating`, `MemoryHeadroom`, `PrefixCacheResident`. Two
rules decide what they can claim:

* **Which probes can see an outage.** `/health` and `/resources` are answered
  before the model lock, so only they can tell a dead server from a busy one.
  `/v1/models` and `/stats` are dispatched after it: an unanswered probe there
  while the lock is held is the generation, so the condition stays held and
  says "held by the model lock". `--interval-ms` bounds each probe, which is
  what makes a held probe observable at all.
* **Residency is not reuse.** `PrefixCacheResident` reports entries and bytes
  from `/stats`, which is what A4 of the roadmap calls residency - not hit
  rate. `MemoryHeadroom` compares `reserved_bytes` to `limit_bytes`, and
  reports `no cap set` when the limit is the `u64::MAX` sentinel `serve` uses
  for an unbudgeted process.

`Generating` comes from two measurements rather than a threshold on absolute
latency: `/stats` waiting several times longer than `/resources` in the same
round (same connection style, same machine, so the ratio is contention), or the
context captured at the end of the last generation advancing between polls. The
measured numbers ride on every printed line so a reader can disagree with the
ratio without reading the source. `--json` emits one object per transition with
the drift lines kept, which the human line omits.

## Images

There is no vision projector in the engine yet, so image parts are refused
rather than ignored. A request whose `content` carries `image_url` (or
`input_image` / `image`) parts gets HTTP 400 with `"code":
"images_unsupported"` before any prefill runs, and no session state is touched.
Text parts of a mixed message are kept: they used to be flattened away together
with the picture, which made a vision message silently arrive empty. Mixed
`content` arrays are parsed for every message, so history that still contains an
image keeps the request rejected until the image is removed.

This is why the body limit above is 16 MiB instead of a fixed 4 MiB - Studio
allows four attachments, about 6 MB before base64 encoding.

The census the vision increments are planned from is:
`allpaka inspect <mmproj>.gguf --mmproj`. It prints the `clip.*` metadata fields
and the tensors grouped by top-level name prefix, and claims nothing beyond what
the file contains.

## Request lifetime and sessions

HTTP admission continues while inference runs. Readers are limited to 16, with
five-second socket timeouts and a 16 MiB body limit (`ALLPAKA_MAX_BODY_MIB`,
1..1024, default 16). Queue exhaustion returns
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

## Shared memory admission

`--memory-budget-mib N` limits the combined reservations for mapped model
weights, the complete configured prefix-cache capacity, and live session
KV/SSM storage plus a conservative RoPE growth allowance. Zero means unlimited.
The existing `--model-budget-gib` and `--prefix-cache-mib` limits still apply.
For example, use `--memory-budget-mib 32768 --prefix-cache-mib 256` to set a
32 GiB admission limit with a 256 MiB cache reservation.

Reservations precede session allocation, remain charged while named sessions
are retained, and account for old/new buffer overlap during replacement.
Insufficient request capacity returns HTTP 429 before streaming; startup fails
if model weights and cache capacity alone exceed the limit. `/stats` exposes
`memory_admission.limit_bytes`, `reserved_bytes`, and `peak_reserved_bytes`.
These are reservation counters, not measured RSS. GPU scratch, model-side
auxiliary allocations, HTTP buffers, and other process overhead are not yet
covered; this is not a hard process memory cap.

Validation: `python3 scripts/test-serving-memory.py` exercises actual memory
rejection, reservation release, and a successful request after rejection.
