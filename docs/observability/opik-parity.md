# Opik capabilities in native Allpaka Studio

The user explicitly requested Opik features in addition to jcode features.
This remains part of the same implementation objective. Native Studio capabilities
must be exercised, not inferred from existing inference telemetry.

Source reviewed on 2026-10-07: https://github.com/comet-ml/opik#-what-is-opik
Documentation/source audit and the detailed capability inventory remain open.
The source README identifies these capability groups:

| Requirement | Acceptance evidence needed | Current checkout |
| --- | --- | --- |
| Hierarchical agent/LLM/tool traces | Persisted parent/child spans, durations, status, cancellation/error receipts and runtime tool-loop evidence | Added native turn/model/tool spans, local Swarm member/model/tool/synthesis/critic trees, manual/automatic compaction attempts and experiment/item/model trees; remote dispatch summaries recorded, worker-internal cross-host tree stitching still open |
| Conversation/thread observability | Trace-to-session/project links, session aggregation and filters | Added project/session/time-filtered conversation aggregates in the existing summary API and Studio: traces/statuses, model-leaf usage, unknown counters, currency-separated reported costs and explicit external call counts. Groups use project+session identity and provide paginated conversations (up to 100 per page) while global totals include all scanned records. Richer thread analytics remain open |
| Token and cost accounting | Reported usage, cached tokens, separate reported/estimated prices; unknown cost remains unknown | Reported numeric usage/cached tokens/cost whitelisted per model span. Summary totals preserve unknown values and separate currencies; model/source groups add calls/statuses/tokens/cost and nearest-rank whole-span latency P50/P95 with unknown-duration counts. Native and external provenance remain distinct. Native generation spans now capture configured provider IDs across chat/Swarm/experiments/compaction/judges/memory extraction. Summary groups separate provider/model/source; legacy unknown providers remain null. External SDK blocks/decorators accept explicit provider metadata. TTFT and price estimates remain open |
| Trace browsing and monitoring | Search/filter, detail tree, aggregates and time-series UI with bounded APIs | Added filtered/paginated trace API and per-session metadata panel; dashboards and richer filters missing |
| Human feedback | Trace/span scores, comments, correction and annotation workflows | Added trace/span numerical and categorical scores, reviewer names, comments, corrected answers, per-target aggregates, immutable revisions, conflict protection, recoverable removal, version browser and JSON export. Reviewer names are self-reported; score definitions, annotation queues and team authentication remain open |
| Dataset management | Versioned samples, reference answers, metadata, immutable experiment snapshots and import/export | Added immutable SHA-256 verified snapshots, revision conflicts, Studio editor and JSON import/export; Added recoverable server/SDK dataset archival, separate active/archive catalogs, optimistic lifecycle revisions, blocked archived editing and retained immutable version reads. Added Studio active/archive browsing and explicit archive/restore controls with version pins, single mutation admission and stale project/catalog guards. Hard deletion/retention and large-scale storage remain open |
| Experiments | Dataset runner, per-item results/errors, bounded concurrency, cancellation, replay and comparisons | Added native no-tool provider runs on pinned snapshots, item output/status/usage, bounded concurrency/timeouts, cancellation, replay through the API, run browser and comparison receipts; startup hard-crash recovery added; Added Python SDK evaluation of synchronous local callbacks on pinned datasets, preflight references/thresholds/baselines, once-only input execution, immutable offline scoring/gates and partial-output exception evidence. Async SDK callbacks now share preflight, execute with 1–8 bounded workers, per-item timeout/cancellation evidence, failure admission stop and awaited worker cleanup, offload Studio HTTP from the event loop and retain the same offline gates. Both callback APIs now explicitly accept frozen dataset contexts through with_contexts=True while excluding reference answers and metadata; native contracts verify callback transport and no scoring provider calls. Callback evaluators also accept an explicit active trace with metadata-only per-sample children, inherited async parents and bounded span admission; SDK/native contracts verify privacy and persisted hierarchy. CI task mode now explicitly creates project-scoped traces and includes delivery evidence in JSON/JUnit; callback admission rejects foreign-project traces and preserves scored evidence on delivery failure. Saved offline/Python scores now have a bounded project/dataset-filtered API/SDK catalog with pagination, verified recalculated summary admission, invalid counts and explicit scan truncation; output text is excluded. Studio now browses offline score summaries and per-sample outputs with pagination and stale-project checks; handler tests and browser catalog/detail/21-record forward-back pagination verification pass without console errors. Studio additionally selects baseline/candidate offline receipts and displays native verified per-sample comparisons with project/selection epoch checks; handler tests and browser baseline/candidate selection, F1 1 → 0.5 regression display and persisted-receipt verification pass without console errors. Studio offline score catalog also explicitly filters by selected dataset across versions, resets comparison selection on scope changes and discards stale dataset responses; handler tests and browser empty-selection, populated dataset and empty-other-dataset qualification pass. Browser verification exposed and fixed the placeholder option lacking an explicit empty value; a regression test models native option-value fallback. Durable partial task runs, resumable partial runs and richer analysis remain open |
| Evaluation metrics | Deterministic metrics, configurable LLM judges, hallucination/moderation and RAG relevance/precision evaluations | Added exact match, reference containment, valid JSON, structural JSON equality and whitespace-token multiset F1 and Unicode-character-bigram multiset F1 with paired regression checks and CI thresholds; Added explicit single-answer model judge with a user rubric, score/reason receipt and native trace. Added pinned dataset judge plans, native cancellable batches, experiment-derived plans with source hashes, SDK/CI thresholds and verified paired judge comparisons. Studio browses saved runs and starts reviewed plans by ID. Criteria now pin verified versions from the immutable instruction library, retain full snapshots and reject mismatched criterion revisions in paired gates. Saved offline/Python answers can now create native judge plans with server-verified exact output/dataset coverage and pinned source receipt ID/hash, without task reruns or planning inference. SDK combined gates now require both verified saved deterministic thresholds/paired comparison and native model-judge thresholds/paired comparison, retain both results and preserve deterministic evidence on judge errors; SDK/native contracts verify failures are not masked. Studio additionally creates reviewed judge plans from expanded offline-score answers with shared preset/custom criterion controls and stale-source guards; handler tests and browser criterion/save/review verification pass, with exact source/output pins, no judge execution and no console errors. CI combined gates now explicitly judge saved scores with pinned presets, separate deterministic/judge baselines/thresholds, JSON evidence and separate JUnit cases; native contracts verify one gate cannot mask the other. Native combined experiment persistence and Studio combined gates, dedicated judge-library UI/custom verdict protocols, Three pinned model rubrics now cover answer relevance, reference correctness and source faithfulness, with mandatory evidence preflight; their real-model quality remains unqualified. Moderation, richer RAG metrics and calibrated hallucination detection remain open |
| Evaluation regression gate | Same dataset/metric/project snapshots, complete paired results, strict-quality checks; any paired decrease blocks promotion | Added conservative observational comparison: validates frozen snapshot and recomputes each receipt; failed, partial, truncated, mismatched and tampered results rejected; any paired decline blocks eligibility even if the mean improves. No automatic promotion |
| Prompt management/playground | Versioned prompts, variants, model comparisons and reproducible prompt-to-run links | Added immutable SHA-256 verified text-template/system-instruction versions, optimistic conflicts, explicit version loading, paginated version history, variant copies in Studio and pinned full snapshots in experiment receipts. Reuses dataset runner/model settings and paired comparisons. Variant ancestry pins parent ID/version/hash and is immutable across later revisions. Added Python SDK matrices of 2–16 frozen prompt/model variants on identical dataset/project/metric pins, sequential persistent native runs and paired comparisons against the first variant, partial-failure evidence and no automatic promotion. SDK and local native contracts verify regression rejection despite improved means. Added Studio matrix-report viewing with authoritative run reloads, project/hash/metric validation, recomputed paired comparisons and experiment navigation; local browser flow verified. Added native immutable matrix summary persistence, reverified reads, SDK save/read and Studio save/open of reviewed imported run matrices. Added bounded project matrix catalog and Studio pagination/open routing. Added direct Studio draft/edit/start/stop of 2-16 prompt/model configurations, immutable request capture, sequential native experiments and saved matrix evidence. Added server-owned matrix jobs with durable reservations/admission flags, restart interruption, explicit continuation/retry lineage and reuse of verified completed variants; Studio launch/catalog/open and native restart/retry fixtures are qualified. Added pinned single-input preview and native Playground launches through the existing experiment manager, with saved samples/results, native tracing/cancellation/export, optional metrics and unscored responses excluded from quality gates. Added bounded versioned User/Assistant few-shot message templates, pair editing, full preview and native Playground/dataset execution. Browser message-editor save/preview and unscored Playground/result/trace flow verified against a local stub. Browser stop/resume/retry, scored Playground interaction and automatic scheduling remain open |
| Online evaluation | Configurable sampling/rules, persisted judgments and failure handling | Missing |
| Guardrails | Configurable input/output checks, explicit actions and receipts | Added explicit local Python input/output policies with bounded UTF-8 length, literal forbidden-fragment and strict JSON checks, frozen configuration hashes, observe/block actions and metadata-only receipts. Sync/async task wrappers block inputs before execution, withhold blocked outputs, preserve business exceptions/cancellation and never retry tasks. Explicit active SDK trace integration now persists metadata-only stage/action/pass-fail/policy-hash spans; local native verification confirms blocked output status and content privacy. Native policy persistence, automatic runtime integration, UI, model-based safety checks and full server-side rule receipts remain open |
| Prompt/agent optimization | Candidate generation, evaluation and conservative promotion based on evidence | Missing |
| SDK/framework integrations | Native API/SDK, trace ingestion and external integration contracts | Added dependency-free Python evaluation client for pinned native runs, wait/cancel and paired gates. Added immutable external metadata-tree ingestion through API/Python SDK, parent/timing/type bounds, explicit external provenance, native browsing/feedback and token/cost summary with external model counts. Python context blocks now collect nested/async-sibling parent links, monotonic durations, explicit usage and exception/cancellation status without capturing arguments, outputs or exception text. Tests verify private-field rejection, preservation of business exceptions, unfinished-child interruption and no model invocation. Explicit project-scoped idempotency keys now deduplicate identical normalized requests, reject changed content with 409 and remain atomic across concurrent senders; raw keys are not retained. Context traces accept a key before execution and freeze the completed snapshot; explicit export retries reuse identical metadata/timestamps/key without rerunning functions, reject unkeyed retries and cache successful receipts. SDK tests cover failed delivery and local post-close mutation. Added explicit sync/async Python adapters for OpenAI Chat Completions, Responses and their foreground streams, with terminal outcome checks, connection cleanup and metadata-only reported usage. Added sync/async non-streaming and raw-event streaming Anthropic Messages adapters with conservative cache-inclusive input totals. SDK and native persisted-trace contracts verify unchanged results, privacy and provider/model summaries using local fixtures. Streaming ingestion, background delivery, additional framework instrumentation and Opik protocol compatibility remain open |
| CI evaluation | Automated dataset runs and enforceable regression exits, with retained artifacts | Added executable native evaluation client with fail-closed exits, no-paired-regression default, optional strict-improvement gate, explicit mean-score thresholds, timeout/interrupt cancellation, new JSON evidence and JUnit reports; pytest usage documented. Cloud/real-model qualification and hosted workflow execution remain unverified |
| Trace lifecycle/privacy | Recoverable removal, retention, export and redaction; no credential or system-prompt persistence | Added startup orphan recovery under an exclusive data-directory lock; completed metadata preserved, unknown durations remain unknown. Added recoverable trace removal/restoration with bounded persistent markers, exclusion from active listing/direct reads/feedback/summary, a paginated removed catalog, running-span rejection and preservation of original metadata/feedback. Keyed external retries cannot resurrect a removed trace. Added project trash browsing with pagination, completed-trace removal, restoration/undo and stale-project checks; handler tests and browser remove/trash/restore flow pass; exact feedback receipt and summary membership survive the cycle. Added versioned bounded per-trace JSON export through API/SDK/UI, preserving metadata and optionally including human feedback only by explicit selection. Running/removed exports reject; API contracts verify exact snapshots and zero provider calls. Browser download verification passes: explicit feedback selection, exact downloaded trace/feedback snapshots, zero provider calls and no console errors. Added bounded selected-trace bulk export through API/SDK with project/terminal/visibility checks, preserved order, explicit feedback selection and all-or-error responses; native contracts cover selected packets and rejection. Added a selected-trace file export CLI with explicit feedback selection, exclusive atomic file creation and overwrite/racing-writer protection; local native contracts verify full saved packets. Added Studio paginated selected-trace export with selection preserved across pages, running-trace exclusion, explicit feedback opt-in, a 100-trace limit and stale-project/selection response rejection. Handler tests pass; browser qualification downloaded two selected traces from different pages and verified exact native snapshots, default feedback exclusion and zero provider calls, without console errors. Whole-project export, retention and redaction remain open |

Implementation order: trace/span storage and native execution instrumentation;
trace browser + usage/cost; datasets and experiments; metrics and paired gate;
prompts/playground and feedback; online evaluation and guardrails; optimizer,
SDK/CI, lifecycle and integration coverage. This order does not reduce the scope.

Earlier notes referenced observability/evaluation files from an older local
checkout. Those files are absent in the recovered GitHub copy. Their existence
in memory is not current completion evidence; recover their actual code if a
newer working copy becomes available, then audit and test it before reuse.

Verified increment: `cargo test -p allpaka-chat` passes 64 tests (one native
credential-store test ignored); `python3 scripts/test-studio.py` passes with a
local streaming mock. The harness verifies model/tool parent links, durations,
usage, metadata privacy, cancellation status and trace persistence across server
restart. Dataset/experiment contracts cover immutable revisions, integrity, cancellation, partial outputs, tamper detection and paired comparisons. Browser verification covered editing, saving revisions and loading the preserved original revision. Live cloud-provider parity remains unverified.

Storage is `<studio-data>/observability/traces`. Content capture is disabled.
The API retains a provider's raw reported `cost`, without inferring currency or
substituting zero for missing data. Local Swarm records participant phases with model/tool children and separate synthesis/critic phases. Manual compaction has its own trace; automatic compaction is a child of the ordinary turn, with a model span per attempted segment. Experiment runs persist `trace_id` and link their per-item/model trees. Remote worker dispatch is summarized; model/tool spans inside the worker and cross-host links remain open. Startup now marks records left running by process termination as interrupted while holding an exclusive data-directory lock. Completed spans and records remain intact, unknown durations stay unknown, and a recovery timestamp is separate from execution timing. Trace retention remains required work.

Explicit dataset and experiment artifacts are stored under `<studio-data>/evaluation`. Unlike metadata-only traces, these intentionally contain supplied sample inputs, reference answers, contexts, prompt templates and model outputs. Provider credentials are not part of the artifacts. See `docs/studio.md` for API limits and usage.


Human feedback source audit: https://www.comet.com/docs/opik/tracing/advanced/annotate_traces/
Native feedback is separate from runtime trace metadata in
`<studio-data>/observability/feedback/<trace-id>/<version>.json`. Explicit human
comments and corrections are intentionally persisted there. They do not turn on
prompt/tool content capture or enter model context automatically. Each version
contains a bounded snapshot, with a server timestamp; concurrent stale writes
return 409. Reviewer identity is a local label, not verified authentication.
Both API and UI support whole-trace and span annotations; deleted entries remain
restorable and historical versions are read-only in the UI. Contract tests cover
invalid targets, conflicts, numeric/category aggregates, content isolation and
restart. Eight independent store instances verify exclusive revision commits. Browser verification covers saving a score/comment/correction, viewing historical versions read-only, and removal/restoration; no browser errors were recorded.


2026-10-07 instrumentation increment: native Studio contracts verify complete
parent/child topology and terminal states for local Swarm models/tools, synthesis,
critic, failed members and cancellation; manual and automatic compaction,
identical-input retries, token limits, failures and cancellation; experiment
trace linkage, failure/cancellation and restart. Usage lives on model leaves
(or the remote-dispatch summary), not on local wrapper phases, avoiding duplicate
usage in tree aggregation. Prompts, summaries, tool arguments/results and model
content remain absent from trace records. Studio renders nesting depth and can
open the experiment trace from its result. Browser verification covered the
Swarm tree and the experiment/item/model tree with no recorded console errors.
Swarm members that exhaust their step budget while requesting another tool are
marked incomplete instead of successful; the final incomplete tool request is
not executed. Retention and distributed trace correlation remain required work. Startup orphan recovery is implemented in the following increment.


2026-10-07 recovery increment: Studio holds an advisory OS file lock for the
entire runtime lifetime, preventing a second compliant Studio process from
writing or recovering the same data directory. The lock file remains in place;
its presence is not ownership, and OS release after process death permits restart.
At startup, bounded trace and experiment scans mark abandoned running records
interrupted, preserve completed spans/items and reported usage, clear partial-run
promotion/mean-score eligibility, and persist a distinct `recovered_ms`. Unknown
durations are never derived from restart time. No model, command or experiment
is automatically replayed. SIGKILL contract tests exercise an active model and
partially completed dataset run, verify retained outputs and completed trace
bytes, no provider replay, and idempotence on a second restart. A second server
using the same data directory is rejected before recovery. Unix lock tests also
verify release while a forked child still holds an inherited descriptor.
Browser verification shows interrupted traces and experiment items, recovery
explanations and unknown durations without console errors. Windows lock/runtime
behavior has not been executed on Windows in this checkout. Recovery uses the
existing bounded catalog budgets (10K traces/64 MiB, 1K runs/128 MiB); scalable
indexed catalogs and retention remain open.

Prompt versions are explicit artifacts under `<studio-data>/evaluation/prompts`.
Runs with `prompt_ref: {id, version}` retain the full verified snapshot; future
edits do not change their instructions. Inline `prompt_template` remains supported
with the standard system instruction. Supplying both is rejected. The Studio
version selector shows the latest catalog revision while the version field and
receipt state identify the revision actually loaded for a run.

Prompt increment validation: 64 Rust unit tests and 2 integration tests pass (one
native OS vault roundtrip intentionally ignored); CLI build with `airbug` and
full mock Studio contracts pass. Contracts verify pinned prompt receipts after a
new library revision, custom system instruction delivery, stale/ambiguous request
rejection and hash tampering. UI creation of two versions and loading the first
version were checked in the browser without console errors.

Native evaluation client / CI instructions and limits: [evaluation-ci.md](evaluation-ci.md).

### Explicit Python Chat Completions adapter

`Studio.track_openai_chat(create, provider_id=..., model_id=...)` wraps a sync
or async OpenAI-compatible Chat Completions `create` callable without importing
or patching the provider SDK. Calls inside this client's trace add a model span;
other calls retain their original result. The request model must match the pinned
technical model ID. The adapter reads only reported prompt/completion token
counts and cached prompt tokens, preserves exceptions/cancellation and does not
add retries. Prompts, choices, response IDs, keys and error text are not retained.
Background Responses and other framework adapters remain open.

```python
create = studio.track_openai_chat(
    openai_client.chat.completions.create,
    provider_id="openai", model_id="gpt-4.1-mini",
)
with studio.trace("default", "request-123"):
    answer = create(model="gpt-4.1-mini", messages=messages)
```

Async clients use the same wrapper with `await create(...)`. Model IDs in this
example are labels, not a model recommendation. The adapter accepts only
non-streaming requests. Missing or malformed usage remains unknown; cost is not
estimated. Field mapping follows the upstream
[Chat Completions usage schema](https://github.com/openai/openai-python/blob/main/src/openai/types/completion_usage.py).

### Responses API adapter

`Studio.track_openai_responses(create, provider_id=..., model_id=...)` provides
the same explicit sync/async metadata tracing for `client.responses.create`.
It maps `input_tokens`, `output_tokens` and
`input_tokens_details.cached_tokens` according to the upstream
[Responses usage schema](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_usage.py).

```python
create_response = studio.track_openai_responses(
    openai_client.responses.create,
    provider_id="openai", model_id="gpt-4.1-mini",
)
with studio.trace("default", "response-123"):
    response = create_response(model="gpt-4.1-mini", input=user_input)
```

No response text, instructions, tool payloads, provider response IDs or exception
bodies are captured. Usage counters do not estimate costs. The model must match
the pinned adapter ID. Streaming and background responses are rejected before
the provider callable is invoked; background polling and stream consumption
require separate lifecycle support and remain open. Span completion records
completion of the API call, not independent verification of answer quality.

### Chat Completions streams

`Studio.track_openai_chat_stream(create, provider_id=..., model_id=...)` wraps
a sync or async streaming callable. Pass `stream=True` and consume the stream
inside the active trace, preferably with a context manager:

```python
create_stream = studio.track_openai_chat_stream(
    openai_client.chat.completions.create,
    provider_id="openai", model_id="gpt-4.1-mini",
)
with studio.trace("default", "stream-123"):
    with create_stream(model="gpt-4.1-mini", messages=messages, stream=True,
                       stream_options={"include_usage": True}) as stream:
        for chunk in stream:
            display(chunk)
```

Chunks retain their original identity and are not buffered. Usage is copied only
from reported chunk counters; stream options are forwarded unchanged. Exhaustion
completes the model span and closes the underlying stream. Early explicit/context
close interrupts it. Read errors preserve the original exception and close the
stream. Other calls between chunks remain siblings rather than accidental child
spans. Leaving a stream active when its trace ends retains interrupted evidence
and raises the existing unfinished-child error. Streams are single-consumer;
Background Responses remain open. No TTFT claim is made.

The same Chat Completions stream adapter now accepts asynchronous `create`
callables. Await stream creation and use an async context manager:

```python
create_stream = studio.track_openai_chat_stream(
    async_openai_client.chat.completions.create,
    provider_id="openai", model_id="gpt-4.1-mini",
)
with studio.trace("default", "async-stream-123"):
    async with await create_stream(model="gpt-4.1-mini", messages=messages,
                                   stream=True,
                                   stream_options={"include_usage": True}) as stream:
        async for chunk in stream:
            await display(chunk)
```

Async exhaustion closes the provider stream and completes its span. Early
`await stream.close()` / `await stream.aclose()` interrupts it. Cancellation or
read errors attempt connection cleanup and preserve the original exception;
a second cancellation can interrupt asynchronous cleanup. Parallel readers are
rejected; each read restores its parent context before returning a chunk.
No chunk text is retained. Use/close the stream before the trace context exits.

### Responses event streams

`Studio.track_openai_responses_stream(create, provider_id=..., model_id=...)`
now traces sync and async foreground `responses.create(..., stream=True)` event
streams, using the same consumption/close lifecycle as Chat Completions streams.
Events are yielded unchanged and never buffered. Only terminal event type,
response status and reported usage counters are inspected.

```python
create_events = studio.track_openai_responses_stream(
    async_openai_client.responses.create,
    provider_id="openai", model_id="gpt-4.1-mini",
)
with studio.trace("default", "events-123"):
    async with await create_events(model="gpt-4.1-mini", input=user_input,
                                   stream=True) as events:
        async for event in events:
            await display(event)
```

A valid `response.completed` event whose nested status is `completed` allows a
completed span after exhaustion. `response.failed` and `response.incomplete`
with matching nested statuses produce failed spans. Exhaustion without a valid
terminal event produces interrupted evidence. Events and original exceptions
remain caller-owned; the adapter does not synthesize provider errors or retain
output/error bodies. Failure remains sticky if multiple valid terminal events
are seen. Early close/cancellation interrupts the stream. Background calls are
still rejected. This integrates foreground SSE events; WebSocket multi-response
sessions and background polling remain open. Event mapping follows the upstream
[completed](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_completed_event.py),
[failed](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_failed_event.py)
and [incomplete](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_incomplete_event.py)
schemas.

### Anthropic Messages adapter

`Studio.track_anthropic_messages(create, provider_id=..., model_id=...)` wraps
sync/async non-streaming `client.messages.create` calls with the same explicit
model pin, original result/error preservation and metadata-only trace lifecycle.
No Anthropic SDK dependency, global monkey patch or automatic retry is added.

```python
create_message = studio.track_anthropic_messages(
    anthropic_client.messages.create,
    provider_id="anthropic", model_id="claude-configured-model",
)
with studio.trace("default", "message-123"):
    message = create_message(model="claude-configured-model",
                             max_tokens=1024, messages=messages)
```

For async clients use `await create_message(...)`. Model IDs above are placeholders,
not model recommendations. Reported output tokens are kept directly; total input
is the sum of reported uncached input, cache-creation input and cache-read input,
as described by [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
All three counters must be present, bounded nonnegative integers and their sum
must fit u64. Otherwise total input/cache-read usage stays unknown rather than
assuming absent cache counters are zero. Cache writes contribute to total input;
cache creation is now retained separately in the native trace schema. Costs are
not estimated. System instructions, content, IDs, tool payloads and errors are
not saved. Other Anthropic APIs remain open.

### Anthropic Messages raw event streams

`Studio.track_anthropic_messages_stream(create, provider_id=..., model_id=...)`
wraps sync/async `client.messages.create(..., stream=True)` raw event iterators.
Use the same sync or async context-manager consumption pattern as above.
The high-level `client.messages.stream` manager is a different interface and is
not supported by this adapter.

A `message_start` with a Message object establishes initial reported input,
cache-creation, cache-read and output counters. `message_delta` usage updates
replace reported counters instead of adding cumulative totals; missing fields
retain the prior reported values. Total input is recalculated from the three
known input components. Output text and content-block deltas are never inspected.
A subsequent `message_stop` permits completion after exhaustion; a missing stop
or start produces interrupted evidence. An error event records failure without
capturing its body; original iterator errors remain unchanged. Repeated starts
are conservatively marked failed. Early close/cancellation uses the existing
stream cleanup behavior. Events remain caller-owned and are not buffered.

The event mapping follows the upstream
[start](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_start_event.py),
[delta](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_delta_event.py)
and [stop](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_stop_event.py)
schemas. Real-provider/model qualification remains unverified; contract tests use
local fixtures and do not send paid model requests.

Native external trace ingestion now accepts `cache_creation_input_tokens` as an
optional bounded counter. Metadata contexts and Anthropic adapters retain it.
Model groups and overall summaries expose separate reported cache-creation/read
totals, with unknown-call counts rather than assuming absent counters are zero.
Cache counters are components of input usage and are not added again to total
input. Older receipts remain readable; missing counters are unknown.

### Selected trace batch export

`POST /api/observability/trace-exports` accepts `project_id`, 1–100 unique
`trace_ids` and optional `include_feedback` (false by default). Every trace must
be terminal, visible and in the selected project. One invalid entry rejects the
entire response; no partial batch is returned and no model is called. Each entry
uses the existing versioned single-trace export format. Per-trace size remains
3 MiB; the full batch is limited to 16 MiB and preserves selection order.

```python
packet = studio.export_traces("default", [first_trace_id, second_trace_id])
# Feedback, which can contain manually entered text, requires explicit selection:
reviewed_packet = studio.export_traces(
    "default", [first_trace_id], include_feedback=True,
)
```

The result has kind `trace_export_batch`, schema version 1, project ID,
trace count, packets in `traces`, privacy flags and `provider_calls: 0`.
This is a read/export operation, not trace import or server-side archival.
Each feedback snapshot has its own immutable revision; the batch does not
promise a transaction across simultaneous reviewer changes. Studio selection UI is implemented and browser-qualified.
Whole-project pagination, retention and redaction remain open.

The file launcher saves one full selected batch without exposing feedback bodies
in its stdout summary:

```sh
python3 scripts/export-studio-traces.py --base-url http://127.0.0.1:8100 \
  --project-id default --trace-id FIRST_TRACE_ID --trace-id SECOND_TRACE_ID \
  --output selected-traces.json
```

Repeat `--trace-id` for 1–100 unique traces. Add `--include-feedback` only when
manually entered feedback should be included. Exit 0 confirms a saved file;
exit 2 indicates invalid/unavailable evidence or a file error, and 130 indicates
interruption. Existing files and symlinks are rejected before the API call.
The final write uses an exclusive link from a flushed temporary file in the
same directory, preserving files created by racing writers. No partial export
file is published on failure. This selected export adds no provider calls.

The selected-trace launcher also accepts `--format jsonl`. Each UTF-8 line is
one complete native `trace_export` packet, preserving selected order, privacy
flags and explicitly included feedback. Embedded newlines are JSON-escaped;
there is no batch-header line. Default `json` retains the full batch envelope.
Both formats validate the whole response before publishing a new file and
retain exclusive file creation. Tests cover Unicode/multiline feedback,
packet order, invalid response rejection and exact native-snapshot conversion.

### Explicit Python guardrails

`allpaka_guardrails` is dependency-free and runs local deterministic checks.
The caller explicitly chooses `observe` or `block`; there is no implicit policy.

```python
from allpaka_guardrails import guard_task, GuardrailBlocked

checked = guard_task(
    my_text_task,
    input_rules=[{"id": "input-limit", "kind": "max_bytes", "value": 8000}],
    output_rules=[{"id": "json-answer", "kind": "json_valid", "value": True}],
    action="block",
)
try:
    result = checked("question")  # await checked(...) for an async task
    answer = result["output"]
    receipts = result["guardrails"]
except GuardrailBlocked as error:
    receipt = error.receipt
```

Each policy has 1–32 uniquely identified rules. Supported kinds are `min_bytes`,
`max_bytes`, `forbidden_substrings` (case-sensitive literals) and `json_valid`.
Input/output text is bounded at 64000 UTF-8 bytes. JSON validation rejects
duplicate keys at any depth, nonfinite values including overflowing exponents,
and excessive parser nesting. `check_guardrails` runs the same checks directly.

Receipts contain only rule IDs/kinds/results, stage, action, a frozen policy
SHA-256 and `provider_calls=0` for the checks themselves; wrapped tasks may
independently call providers. Rule values and input/output text are excluded.
A blocked input never calls the task. A blocked output raises without returning
the output and attaches the successful input receipt. Observe mode returns
the original output with both receipts even when checks fail. Original task
exceptions/cancellation propagate unchanged. Policies are validated and frozen
at wrapper construction, preventing later rule-list mutation from bypassing
checks. Local receipts are returned to the caller, not persisted by Studio.

These deterministic rules do not establish semantic safety, moderation or
prompt-injection protection. Native/server policies and their integration
remain part of the full backlog. Verification: four focused guardrail tests
plus all 49 existing SDK tests pass, with no provider/network calls.

Guardrail calls and wrappers additionally accept `trace=active_trace`, where
`active_trace` is an entered `Studio.trace(...)` context. Each completed check
creates a native-compatible tool span named
`guardrail.STAGE.ACTION.OUTCOME.POLICY_SHA256`. Observe failures complete
normally with a `fail` name; blocking failures have failed span status and
raise `GuardrailBlocked`. No input, output, rule values or per-rule text is
added to the trace. Existing context parent links and trace export/idempotency
semantics apply. A closed/inactive trace rejects before task execution; trace
admission failures are not silently ignored. This opt-in linkage is separate
from native policy storage or full server-side rule receipts. Five focused
tests plus all 49 SDK tests pass. A live local Studio ingestion/readback check
confirmed both stage spans, blocked output status, matching policy fingerprint
and absence of private output text.

The trace catalog now accepts `guardrail=any|failed|blocked`, also available as
`Studio.traces(project_id=..., guardrail='blocked', offset=0, limit=50)`.
`any` selects recognized reported guardrail spans; `failed` selects failed check
outcomes in either observe or block mode; `blocked` requires block/fail metadata
and failed span status. Filtering precedes total-count calculation and pagination
and composes with project/session and removed-trace scope. Unknown filter values
reject. Only tool/external-tool spans with exact stage/action/outcome fields and
64 lowercase hexadecimal fingerprint characters qualify. These are reported SDK
check outcomes, not authenticated safety certificates or proof of native policy
execution. A native store regression verifies malformed fingerprints, observed
violations, blocking outcomes, project exclusion and filtered pagination; all
50 Python SDK tests and eight guardrail tests pass. Studio filter controls and guardrail summary rendering are implemented and browser-qualified. Native policy storage and full server-side rule receipts remain open.

`GET /api/observability/summary` now includes a `guardrails` object within the
same active-trace/project/session/time scope as the existing usage summary.
It reports checks, passed checks, violations, blocked violations,
`incomplete_or_inconsistent`, and input/output check counts. A pass requires
completed status; an observed violation requires completed status; a blocked
violation requires failed status. Running/interrupted or contradictory status
metadata are counted separately, never treated as successful checks. Only
recognized metadata fingerprints qualify, using the same parser as the catalog
filter. These are reported check-span counts, not unique task/sample counts,
moderation scores or authenticated policy certificates. Guardrails do not add
model calls, tokens or costs. Native regression tests cover outcome accounting,
malformed fingerprints and project-scoped persisted summary integration.

The guardrail summary additionally exposes `policies`: per-policy SHA-256
counters with the same pass/violation/block/incomplete and stage semantics.
Groups are bounded to the first 100 fingerprints in ascending lexical order,
independent of filesystem scan order, with `policy_groups_limit`,
`policy_group_order` and `policy_groups_truncated` metadata. Retained group
counts are complete for the current scoped scan; global counters include all
recognized checks even when groups truncate. An omitted policy under truncation
is unknown, not zero. Policy values and input/output text remain excluded.
A regression feeds 105 policies repeatedly in opposite orders, verifies identical
receipts, complete retained counts and unchanged global totals. Full chat-crate
Rust tests, 50 SDK tests and ten guardrail tests pass.

Trace browsing additionally accepts `guardrail_policy_sha256=HASH` (exactly 64
lowercase hexadecimal characters), either alone or together with
`guardrail=any|failed|blocked`. Python usage:

```python
studio.traces(project_id="default", guardrail="blocked",
              guardrail_policy_sha256=policy_hash, offset=0, limit=50)
```

The policy and outcome predicates must match the same recognized tool span.
A trace containing an observed failure for policy A and a blocking failure for
policy B therefore does not match blocked/A. Both filters apply before totals
and pagination, within the existing visibility/project/session scope. A native
regression verifies same-span conjunction, policy-only selection, blocking/B
selection, and invalid uppercase/length hashes. Full chat-crate Rust tests,
50 SDK tests and ten guardrail tests pass.

Browser qualification on 2026-10-07 verified blocked-only and exact-policy filtering, malformed/unknown hashes, general trace pagination, exact API-matched guardrail summary counts and policy groups, without console warnings/errors. Screenshot evidence: `/tmp/allpaka-guardrail-summary-ui-20261007.png`.

Python streaming adapters now optionally record `first_text_ms` on model-span
usage: monotonic milliseconds from adapter request start to the first nonempty
text event observed by the caller. OpenAI Chat uses choice delta content;
Responses uses `response.output_text.delta`; Anthropic raw streams use
`content_block_delta` with a `text_delta`. Empty/role/control/tool-only events
do not populate the field. Later text events never overwrite it. The adapters
still return original chunks and capture no text. Sync/async wrappers share this
logic; tests use deterministic clocks and verify exact first-event timing,
unknown values before text, unchanged chunk identity and content privacy.

This is client-observed first-text latency, including connection setup and any
consumer scheduling/read delays. It is not server-side first-token generation
time, token throughput or a latency estimate for non-text/tool-only responses.
Missing observations remain absent. External ingestion accepts only unsigned
integer milliseconds and rejects a value exceeding a known span duration;
metadata-only export retains the field. Aggregate percentiles and UI rendering
remain open. Source event schemas:
https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_text_delta_event.py
https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/text_delta.py

The summary API now returns `first_text` both globally and per provider/model/
source group, containing known/unknown calls, min/max and nearest-rank P50/P95
milliseconds. Unknown and inconsistent observations do not enter quantiles;
known zero remains zero; empty distributions return null. Whole-model-span
latency remains separate. The scope is `client_observed_first_text` with
`server_token_time=false`, including consumer delays and connection setup.
Failed/interrupted calls with valid observed text timings still contribute.
Missing observations in native/non-streaming calls remain unknown. Regression
coverage verifies zero, quantiles, missing/negative/inconsistent values and
empty distributions. Full chat-crate Rust tests, 52 SDK tests and ten guardrail
tests pass. Studio now renders first-text distributions globally and per model, with known/unknown counts and client-delay scope explanation. Handler tests and browser verification of the persisted synthetic distribution pass without console errors.

Native API qualification used explicitly reported local fixture timings
0/10/100 ms plus one unknown value: persisted group summary retained known zero,
returned P50=10 and P95=100 with three known/one unknown calls, and global known
count increased by three. An observation exceeding the model-span duration was
rejected. These are synthetic accounting fixtures, not provider performance
measurements; no inference was invoked by ingestion/summary reads.

The trace catalog supports exact root `status` plus inclusive `since_ms` /
`until_ms` filters on trace start time. They compose with project/session,
visibility and guardrail/policy filters before total-count calculation and
pagination. Reversed periods reject. Status identifiers are bounded to 1–40
lowercase ASCII letters/underscores, supporting runtime states such as failed,
interrupted, timeout or no_progress without matching individual child statuses.
Python usage:

```python
studio.traces(project_id="default", status="failed", since_ms=start_ms,
              until_ms=end_ms, guardrail="blocked", offset=0, limit=20)
```

SDK time bounds require unsigned 64-bit integer milliseconds (not booleans).
Native tests verify inclusive equal bounds, filtered pagination ordering,
project exclusion and malformed ranges/statuses. Full chat-crate Rust tests,
53 SDK tests and ten guardrail tests pass. Studio period/status controls are implemented; handler tests verify time conversion, reversed ranges and stale dates. Browser verification confirms status/day filtering, empty future periods and invalid range rejection without console errors. The existing summary period controls are independent.

### Trace telemetry time series

`GET /api/observability/time-series` requires inclusive `since_ms`, `until_ms`
and `bucket_ms` (1–86400000 ms), with optional project/session scope. It returns
`trace_time_series` schema 1 with at most 500 ascending, zero-filled intervals,
trace/root-state counts, model call counts, reported model usage/currency costs,
whole-span and first-text distributions, and guardrail counters. Each bucket
has `start_ms` and `end_exclusive_ms`; the final interval is clipped at until+1.
Reversed ranges, zero intervals, unsigned-time overflow and excessive bucket
counts reject. Existing bounded active-trace scan/visibility rules apply.

```python
studio.trace_time_series(project_id="default", since_ms=start_ms,
                         until_ms=end_ms, bucket_ms=3600000)
```

All spans and current outcomes of a trace are assigned to its start interval,
even if execution continued into later intervals. This is a reported telemetry
histogram, not a stream of completion events, wall-time cost accrual or real-time
snapshot transaction across files. No content is captured or inference invoked;
`provider_calls=0`. Empty latency distributions remain null. Model counters do
not invent a combined provider/model identity; currencies stay separate. Native
regression checks inclusive boundary assignment, clipped final intervals,
filtered project scope, empty intervals, reported tokens/latency and bounds.
Full chat-crate Rust tests, 54 SDK tests and ten guardrail tests pass. Interactive Studio charts for trace/model counts, failed root states and guardrail violations are implemented. Browser qualification verifies hourly and 15-minute buckets, exact API-matched values, empty intervals and selected-bucket details without console errors. Hosted monitoring remains open.

The full native Studio contract suite now includes the time-series HTTP route:
three persisted external fixture traces at 10/19/30 ms produce two records in
the first interval, an empty middle interval and an inclusive final record.
The contract verifies separate USD/EUR reported costs, known-zero first-text
quantiles, guardrail counts, absent-project scope, malformed/unknown query
rejection and unchanged provider request count. The complete suite passes,
including persistence/restart and hard-crash recovery, using a local mock
provider without cloud calls. Evidence log:
`/tmp/allpaka-time-series-native-contract.log`.

### Portable local guardrail policy manifests

The Python guardrail module now exposes `policy_manifest(rules)` and
`rules_from_manifest(manifest)`. A manifest includes the exact validated rules,
versioned schema and the same SHA-256 fingerprint used by execution receipts.
Loading rejects unknown fields, boolean schema versions, invalid rules and
fingerprint mismatches. Returned rules are independent copies. Canonical policy
payloads are bounded to 128 KiB. Eleven guardrail tests pass, including Unicode-
compatible serialization, tampering and mutation isolation.

Unlike metadata-only receipts, manifests intentionally contain rule values,
including literal fragments. They are explicit configuration artifacts and must
not be included automatically in trace telemetry. This is the portable policy
representation; native persistent policy storage and Studio management remain
open.

### Explicit policy file persistence in Python

`save_policy(path, rules)` validates and writes a private temporary file, flushes
it, then publishes with an exclusive hard link. Existing paths are never
replaced, including concurrent writers. `load_policy(path)` reads at most
128 KiB, requires a regular file, rejects duplicate JSON keys/nonfinite numbers,
and validates the manifest schema and execution fingerprint before returning
rules. On platforms exposing O_NOFOLLOW the final path cannot be a symlink.
The Unix implementation is qualified; Windows behavior is not yet qualified.

```python
from allpaka_guardrails import save_policy, load_policy, guard_task
save_policy("response-policy.json", [
    {"id": "json", "kind": "json_valid", "value": True}
])
output_rules = load_policy("response-policy.json")
checked_task = guard_task(task, input_rules=input_rules,
                          output_rules=output_rules, action="block")
```

Twelve guardrail tests cover file round-trip, exact execution fingerprints,
private permissions, overwrite rejection, two racing writers, temporary-file
cleanup, oversized files, duplicate keys and symlinks. All 54 SDK tests pass.
This implements explicit local configuration persistence; native server-side
policy management and Studio authoring remain open. Policy values are not
attached to traces automatically.

Native policy schema validation now lives in `guardrail_policies.rs`: bounded
unique technical rule IDs, supported rule/value types and canonical ordered
rule hashing match the Python manifest. A committed Python-generated fixture
with Unicode and a literal newline validates with the same fingerprint in
Rust. Two focused native tests pass, including tampering and strict schema
rejection. This module is registered but not yet routed or persisted; native
policy management is not claimed complete.

### Native policy API and SDK

`POST /api/guardrail-policies` explicitly stores a complete validated manifest;
`GET /api/guardrail-policies/{sha256}` reads it after revalidating stored contents.
Both return `policy` and `provider_calls=0`. POST parses JSON with duplicate-key
rejection and a 128 KiB payload cap. Content-addressed files are atomically
published without replacement; corrupt existing records fail closed. Policies
are installation-local explicit configuration, not project-authorized resources
or automatically enforced runtime rules.

Python exposes `Studio.save_guardrail_policy(rules)` and
`Studio.guardrail_policy(fingerprint)`. SDK validation checks returned policy
fingerprints and zero provider calls. Three storage/schema Rust tests,
14 guardrail tests and 54 existing SDK tests pass; native Studio builds.
HTTP runtime qualification and Studio authoring remain pending.

HTTP runtime qualification now confirms native save/read and repeated save via
the Python SDK, exact execution fingerprints, rejection of changed rules,
unknown hashes, invalid schema and duplicate JSON keys, with provider request
count unchanged. The runtime check found and corrected the read route syntax
for the installed Axum version. These checks are part of `test-studio.py`;
the broader contract run is still in progress at this checkpoint.

The complete Studio HTTP contract passed after native policy routing was added
(mock provider only). `Studio.guard_task_with_policies(task,
input_policy_sha256=..., output_policy_sha256=..., action='block')` now resolves
both immutable policy manifests and constructs the existing sync/async guardrail
wrapper. Rules are frozen before task execution; individual calls do not refetch
or silently change policy. Fifteen guardrail tests pass, including mutation
isolation after resolution and input blocking before callback execution.
The contract additionally asserts stored policy retrieval after Studio restart;
that new restart assertion is being qualified in the next full run.

Stored policies now have a paginated metadata catalog:
`GET /api/guardrail-policies?offset=0&limit=20`, also exposed as
`Studio.guardrail_policies`. Pages contain fingerprints, schema versions and
rule counts, without rule values. Ordering is ascending fingerprint, page size
is 1-100, offset 0-10000 and the directory scan is bounded to 10000 entries.
Selected records are integrity-checked before returning metadata. Unknown query
fields and malformed bounds reject. Focused native storage tests and 15 Python
guardrail tests pass. The full earlier policy contract including retrieval
after restart passed; HTTP catalog qualification is still pending.

Native HTTP catalog qualification passed for 21 policies: exact ascending
fingerprints, pages 20+1, empty later pages, metadata-only rows, malformed bounds
and unknown query rejection with zero provider calls. SDK catalog responses
additionally validate pagination, exact metadata schema, fingerprints, rule
counts and unique ascending ordering. Fifteen guardrail tests and 54 SDK tests
pass after these checks. The full contract run remains in progress at this
checkpoint; browser qualification is still pending.

`POST /api/guardrail-policies/create` accepts exactly `{rules:[...]}` and derives
the canonical manifest/fingerprint natively before storing it. Explicit import
still requires a complete fingerprint-verified manifest. The create endpoint
uses strict duplicate-key parsing, bounded body/rules and the same exclusive
content-addressed store. Python `Studio.create_guardrail_policy(rules)` checks
the native result against its independently computed manifest. Native HTTP
qualification confirms exact SDK agreement, idempotence and rejection of empty
rules, unknown fields and boolean byte limits with zero inference. Fifteen
Python guardrail tests and 54 SDK tests pass; the full latest contract run is
still executing at this checkpoint. Studio form authoring remains pending.

Native policy execution now supports the same four deterministic rule kinds
as Python. `Manifest::check` validates the policy first, enforces UTF-8 byte
bounds, uses strict duplicate-key JSON parsing and returns content-free rule
outcomes with the pinned fingerprint. Block/observe and input/output semantics
match SDK receipts. Four native policy tests pass, including Unicode boundaries,
nonfinite/duplicate JSON rejection and private literal exclusion from receipts.
Runtime invocation and enforcement are not yet wired to chat turns; this is the
native checker, not a claim of automatic chat protection. UI tests additionally
verify retaining edits made to a draft while its older snapshot is saved.

Native checks are now callable explicitly through
`POST /api/guardrail-policies/{sha256}/check` with text, stage and action.
`Studio.check_guardrail_policy(hash,text,stage='input',action='observe')`
validates bounded input and returned rule outcomes. Text is processed in-memory;
this endpoint does not persist it, append traces or invoke a model. A blocked
receipt reports what enforcement should do; the endpoint itself does not execute
or wrap another task. Automatic native chat enforcement remains open.
HTTP qualification matches Python receipts exactly for safe/failing text across
both stages and both actions, and rejects invalid stages/actions and oversized
text. Sixteen guardrail tests and 54 SDK tests pass. Latest full HTTP contract
run is still executing at this checkpoint.

Native execution preparation now resolves both input/output fingerprints before
execution and freezes the validated policies for the whole prepared run. Later
file changes cannot silently change an in-flight policy; a fresh preparation
rejects corrupt storage. Explicit action determines whether output must be
buffered before publication. Five focused native tests pass, including mutation
of persisted files after preparation. The full explicit-check HTTP contract
also passed. Chat actor invocation and buffered publication remain unfinished;
no automatic chat enforcement is claimed yet.

Native Chat settings now accept explicit `guardrails` containing input and output
policy fingerprints plus action. Policies are loaded/frozen before model calls;
input blocking prevents inference, output blocking suppresses streamed text and
reasoning until the completed assistant message passes. Failed blocked output
is not written as assistant content. Observe mode preserves delivery and records
violations. Metadata-only tool spans use the existing fingerprint/outcome naming
contract. New steering input is checked before subsequent model calls.

HTTP qualification passes input blocking with zero provider calls, output
blocking after one mock call without assistant text, and observe delivery with
violation spans. Studio builds; latest full regression run is still executing.
Currently selection is supported in Chat only; other modes explicitly reject
it rather than silently ignoring checks. UI selection and Swarm/Goal coverage
remain unfinished. User messages retain ordinary conversation persistence;
this is not a conversation-content redaction feature.

Guardrail selection now supports Plan, Auto and Goal as well as Chat. Native
HTTP checks qualify input blocking without inference and buffered output
blocking after one mock request for each added mode. Swarm remains explicitly
rejected until its independent participant/synthesis paths are covered. Checks
apply to assistant textual content before tool batch execution; they do not
validate tool arguments, image contents, semantic truth or remote side effects.
The UI describes the supported modes. Native builds, UI handler checks and
16 Python guardrail tests pass; full latest HTTP regression is still running.

Swarm integration preparation now carries the frozen validated policy pair in
an in-memory-only Settings field, preserving it across internal member/merge
settings clones. It is skipped in serialization/deserialization so full policy
values cannot appear in persisted public session settings. Actual participant,
remote worker and merge enforcement remain pending, and Swarm selection is
still rejected. Full Rust tests and the Plan/Auto/Goal HTTP contract passed.

Local-provider Swarm now uses the prepared policy pair in each participant and
merge/critic invocation. Input prompts are checked before generation, blocked
participant output is buffered before report publication, and accepted output
is published after validation. Merge output is likewise buffered and checked.
Remote worker selection is explicitly rejected when guardrails are enabled;
remote policy transport remains open. Native HTTP tests confirm empty blocked
reports and successful accepted reports plus synthesis. Build passes; full
latest regression remains running. Critic, retries and streaming interruption
need additional guardrail-specific qualification.

Guardrail-specific HTTP qualification now covers Swarm critic and member retry:
accepted critic runs have at least nine completed guardrail spans, retried
member/synthesis/critic runs have at least seven, with a new trace and accepted
answer publication. The preceding full Swarm guardrail regression passed.
Blocked ordinary chat output now also preserves reported provider usage in the
session state, in addition to model trace usage; the new HTTP usage assertion is
pending the next run. Latest full critic/retry regression is still executing.

Remote-worker guardrail rejection now also occurs immediately before dispatch,
covering retries that retain an old worker URL absent from the current roster.
Full Rust tests pass. Native blocked-output HTTP qualification confirms reported
3/4 input/output token usage remains in session state. Broader HTTP regressions
exposed an old absolute retry-request counter after adding new retry scenarios;
the counter now measures the current scenario delta and a fresh full run is
executing. One earlier run also failed a preserved-report assertion; its next
failure now includes exact report evidence for diagnosis. Full success is not
claimed for these latest runs.

The retry scenario now passes after changing its global counter to a scenario
baseline. The critic scenario had the same old absolute-count assumption and
now also measures its own delta. The latest full contract additionally checks
cancelled buffered generation: no private stream text in session state,
interrupted trace without running spans and no text after restart. This latest
run is active; cancellation/restart completion is not yet claimed.

The full guardrail cancellation/restart HTTP contract now passes after fixing
scenario-specific retry/critic counters. Native guardrail spans additionally
persist `usage.guardrail_receipt`: stage/action, policy fingerprint and each
technical rule ID/kind/outcome. The whitelist discards extra text/rule values
and rejects inconsistent aggregate/block outcomes, duplicate IDs and invalid
metadata. A Rust privacy test verifies literal exclusion and consistency;
full chat-crate Rust tests pass. External receipt transport and Studio rendering
of per-rule outcomes remain open, and native HTTP persistence of this new field
is not yet qualified.

Python traced checks now attach frozen per-rule receipts to tool spans through
`set_guardrail_receipt`. The SDK rejects extra/private fields, inconsistent
outcomes and mismatched tool name/kind. External ingestion accepts the new
bounded receipt field only when its exact whitelisted contents, span name,
kind and terminal status agree; values/text are rejected, not silently stored.
Seventeen guardrail tests, 54 SDK tests and full Rust tests pass. The preceding
native per-rule HTTP persistence/restart contract passed. New external receipt
transport still needs native HTTP qualification.

External Python per-rule receipt transport is now qualified against the native
HTTP server: observe and block outcomes persist exactly, injected text and
mismatched span names are rejected, and per-rule metadata survives restart.
The complete Studio HTTP contract passes with a mock provider (no live cloud
requests). In the browser, creating and saving a max-bytes policy, checking both
passing and failing text, and selecting the saved policy for the chat output
were verified. Selection leaves enforcement disabled until explicitly enabled.
Evidence: `/tmp/allpaka-external-rule-http.log` and
`/tmp/allpaka-policy-author-check-20261007.png`. Browser execution of an enabled
chat policy and rendering per-rule trace details still require qualification.

Further browser qualification found that the automation locator interactions
did not deliver the chat send action to the live page. A native accessibility
field edit and button click did send successfully and produced the expected
mock answer (`/tmp/allpaka-native-send-20261007.png`). This does not qualify
enabled chat enforcement; repeat that scenario with native interactions.

Native dataset experiments now enforce the explicitly selected frozen input
and output policies. Input blocking performs no inference; output blocking
withholds the answer and scores while retaining reported usage; observe mode
records per-rule violations and keeps the output. The full mock-provider HTTP
contract, full chat-crate Rust tests, 54 SDK tests and 17 Python guardrail tests
pass. Evidence: `/tmp/allpaka-eval-guard-http.log`,
`/tmp/allpaka-eval-guard-rust.log`, `/tmp/allpaka-eval-guard-sdk.log` and
`/tmp/allpaka-eval-guard-python.log`. Rubric judge and compaction execution paths
still need policy enforcement and separate qualification.

Rubric judge execution now checks the rendered rubric/input/answer/reference
payload before inference and the raw generated verdict before parsing or saving
a judge receipt. It uses the explicitly selected saved input/output policy pair;
block rejects the request and observe retains the verdict with per-rule trace
metadata. Batch judge runs share this execution path. Direct judge block/observe
HTTP scenarios and full chat-crate Rust tests pass; full HTTP regression is
still running. Compaction remains a separate unfinished enforcement path.

The full HTTP regression subsequently passed, including restart and crash
recovery (`/tmp/allpaka-judge-guard-http.log`). No live cloud calls were made.

Compaction now freezes the selected saved policy pair before processing history.
Each summary/transcript prompt is checked before inference and every generated
summary is checked before accepting it or retrying truncated output. Blocking
leaves original history and the previous compaction unchanged. Observe records
violations and permits an otherwise valid summary. Guarded transcript segments
reserve space for the previous summary within the 64000-byte check bound.
New HTTP rollback/no-inference/observe scenarios are running; no qualification
claim is made until that run finishes.

Those compaction scenarios and the full HTTP regression subsequently passed
(`/tmp/allpaka-compact-guard-http.log`), as did full chat-crate Rust tests
(`/tmp/allpaka-compact-guard-rust.log`). Evidence uses a mock provider only.

Memory extraction now freezes and enforces the selected saved policy pair:
source conversation payload before inference and generated notes JSON before
parsing or proposal persistence. Blocked results create no proposal; observe
retains an unaccepted proposal. Native HTTP scenarios verify input no-inference,
output rejection without a catalog entry, and observe without automatic note
acceptance. Full Rust tests pass. The first HTTP run reached these passing cases
then failed an old catalog equality assumption after the new observed proposal;
the assertion now compares SDK and current server catalogs and rerun is active.

The corrected full HTTP rerun passed (`/tmp/allpaka-memory-guard-http.log`),
including restart and crash recovery. Rust evidence is
`/tmp/allpaka-memory-guard-rust.log`. Live cloud models remain unqualified.

Guardrails additionally support `required_substrings`: all 1–100 bounded literal
fragments must occur in the text, with exact case and Unicode matching. Python
manifests/checks/trace receipts, native policies/receipt ingestion and Studio
policy authoring/trace rendering accept the same rule kind. New Python tests
cover all-fragment matching, missing fragments, case and manifest roundtrips;
native HTTP checks compare Python and Rust results on the same Unicode samples.
Eighteen Python guardrail tests, full Rust tests and policy/receipt UI handler
checks pass; full HTTP regression is running.

The full HTTP regression passed (`/tmp/allpaka-required-http.log`), including
Python/native required-fragment agreement. Rust and Python evidence is in
`/tmp/allpaka-required-rust.log` and `/tmp/allpaka-required-python.log`. Browser
qualification of selecting this new rule kind remains open.

Required-fragment coverage now also includes a native Rust regression checking
all-fragment semantics, exact Unicode case, invalid rule values and metadata-only
receipt whitelisting. Studio handler tests exercise adding/removing this rule
kind and rendering its outcome. These focused checks pass
(`/tmp/allpaka-required-focused.log`); an executable Python usage example is in
`docs/observability/evaluation-ci.md`. Full browser qualification remains open.

Experiment export qualification (2026-10-07): native `GET /api/evaluation/experiments/:id/export?include_outputs=false` produces schema-1 JSON with dataset revision/hash, run status/quality, aggregate scores and per-example status/scores/duration/error presence/truncation. It preserves failed/incomplete outcomes, omits prompt/settings/raw usage/error text, and includes model answers only with explicit `include_outputs=true`. Python `Studio.export_experiment` validates the opt-in boolean and ID. Experiment detail UI downloads this packet with an unchecked answer checkbox. This is an observation export, not a replay/import or promotion artifact. SDK/UI handler checks and native HTTP completed/failed fixtures cover the export; real browser download remains unqualified.

Tabular experiment export (2026-10-07): Python `Studio.export_experiment_csv` converts the same native snapshot to CSV, one row per example, with pinned dataset identity, run/sample status, quality flag, timing, error/truncation flags and separate selected metric columns. Missing failed/pending scores and unknown durations stay blank. Answers remain opt-in. CSV quoting preserves commas, Unicode, quotes and multiline answers; potentially executable spreadsheet text cells receive an apostrophe prefix, so JSON is the exact-text export. Invalid/nonfinite/out-of-range scores or malformed packet metadata reject the export. SDK tests cover failed rows, privacy defaults, Unicode round-trip and formula neutralization.

CSV UI qualification (2026-10-07): experiment export now offers JSON/CSV in Studio, using the same native receipt and explicit answer opt-in. CSV retains incomplete rows, blank unknown scores/timing, quoted multiline Unicode and spreadsheet formula neutralization. Handler tests verify both downloaded filenames/content and special text cells. Native CLI build passes; real browser file-download qualification remains open.

Experiment sample exploration (2026-10-07): Studio run detail offers status filters (all/problems/completed/pending/running), Unicode case-insensitive literal search over sample ID/output/error, filtered/total counts and a clear empty result state. Problems include failed/cancelled/interrupted/timed-out rows, explicit errors and truncated outputs. Filter controls and expanded sample state persist within a refreshing run and reset when opening another run. This filters local displayed receipts only: it does not alter aggregate scores, paired gates, exports or model calls. Handler tests verify status/search combinations, literal HTML-like output, Unicode, expanded state, refresh and run isolation; real browser interaction remains unqualified.

Experiment sample ranking (2026-10-07): Studio sample exploration supports original dataset order, slowest/fastest first, and ascending score for each selected deterministic metric. Unknown/nonfinite values sort last, zeros remain valid, and ties preserve receipt order. Display sorting never mutates saved items, exports, aggregates or gates. Known durations are displayed explicitly; unknown timings are omitted. Handler qualification covers zero values, missing scores/timings, stable ties, run-refresh sort retention and unchanged source arrays. Browser interaction remains unqualified.

Export interaction continuity (2026-10-07): an experiment's export panel is reused during refresh, retaining JSON/CSV selection, answer opt-in and pending-button state. Opening another run resets to JSON without answers. Duplicate activation while a request is pending does not issue another download. The format selector has an accessible label. Handler tests cover refresh identity, run isolation and pending activation; CLI build passes.

Paired answer exploration (2026-10-07): Studio comparison supports all/regression/improvement/unchanged metric-pair filters and defaults to regressions when any exist. Each pair can explicitly load the baseline/candidate saved answers, with run/project/dataset revision/hash identity checks, literal text rendering, shared request caching and retry on transport failure. Changed selection/project discards late comparison responses. This UI does not change the conservative native promotion gate. Handler tests verify filter classification/counts, no automatic answer reads, paired answer rendering, request reuse and retry. CLI build passes; live browser interaction remains unqualified.

Saved comparison reopening (2026-10-07): native `GET /api/evaluation/comparisons/:id` reads bounded regular JSON receipts, checks path/ID, revalidates both strict-quality experiment outputs against the pinned dataset and recomputes deterministic pairs/gate metadata; any receipt difference is rejected. It makes no provider calls and writes no new comparison. `Studio.comparison(id)` exposes it. Studio populates the saved comparison ID after creation and opens it explicitly, enforcing current-project identity and discarding stale selection/project responses. SDK and UI tests cover routes/bounds, reopen/project isolation/duplicate suppression; native HTTP fixtures cover exact successful/regression receipts, tampered pair rejection and restart persistence. Browser reopening remains unqualified.

Comparison history catalog (2026-10-07): native project-scoped comparison metadata catalog supports offset/limit (1–100), descending ID order and has-more, with bounded scan (2000 regular JSON receipts, 16 MiB total, 4 MiB each), strict JSON/path identity checks. Summaries explicitly report `summaries_verified:false` and omit eligibility/pairs; opening uses the separately reverified receipt endpoint. Python `Studio.comparisons` validates page bounds. Studio history lists dataset revision/receipt ID, offers previous/next pages and routes Open through project-checked verification. Handler qualification covers paging, open routing and stale project responses; live browser history remains unqualified.

Comparison evidence export (2026-10-07): Studio comparison detail downloads the complete saved comparison JSON (all metric pairs, pinned dataset hash/revision and gate outcome) after a fresh call to the reverified receipt endpoint and matching receipt/run IDs. Current display filters do not limit exported evidence. Answers/prompts/settings are not included in the native comparison receipt. Handler qualification verifies full receipt download, filename, explicit fresh verification and button restoration; CLI build passes. Real browser download remains unqualified.

Comparison CSV evidence (2026-10-07): Python `Studio.export_comparison_csv` loads the native reverified comparison and exports every metric pair with pinned run/project/dataset identities, version/hash, baseline/candidate/delta/change and repeated conservative gate evidence. It rejects nonfinite/out-of-range scores, inconsistent deltas, duplicate pairs and mismatched regression/improvement counts, eligibility or reason. Spreadsheet formula-like text is neutralized; numeric negative deltas remain numeric. SDK qualification covers Unicode/formula cells, mixed gains/losses and forged gate/delta rejection. HTTP qualification checks a real native paired regression receipt. No answers/settings/prompts are included.

Comparison CSV UI (2026-10-07): Studio comparison export offers JSON/CSV and fetches the reverified native receipt for either choice. CSV includes all metric pairs regardless of display filter, fixed run/dataset identity and quality evidence, numeric deltas and change classification. Formula-like text cells are neutralized while negative numeric deltas remain numeric. Handler tests verify both download types, complete gain/loss/tie rows and formula/negative-value behavior; CLI build passes. Browser download qualification remains open.

Native persistent experiment matrices (2026-10-07): `POST /api/evaluation/matrices` accepts 2–16 uniquely labelled distinct completed strict-quality run IDs, baseline first, within one project and identical pinned dataset/hash/metric configuration. It revalidates outputs/scores, computes every baseline pair, records all variants/provider/model/means and conservative `no_paired_regression` result; any decrease blocks the overall matrix, regardless of mean gains. `automatic_promotion:false`. Immutable schema-1 receipts live in `evaluation/matrices`, bounded to 4 MiB; `GET /api/evaluation/matrices/:id` bounds/parses storage and reconstructs all evidence, rejecting any mismatch. No model calls occur. Python save/read methods validate labels/IDs/cardinality. Studio can save a complete reviewed SDK matrix report and reopen the native receipt by ID. Rust tests (102 + 2 integration, one credential test ignored), 63 SDK tests and full native mock HTTP pass; fixtures verify mixed regressions/gains, tampered success rejection and exact persistence after restart. UI handler qualification covers save payload/open and existing foreign/duplicate/stale rejection; browser save/open remains unqualified. Matrix catalog and direct variant authoring/start remain required work.

Saved matrix catalog (2026-10-07): native `GET /api/evaluation/matrices?project_id=...&offset=...&limit=...` lists bounded metadata (ID, dataset revision, baseline and variant count), with descending ID order and pagination. It scans at most 2000 regular receipts, 16 MiB total/4 MiB each, validates schema/file identity and declares `summaries_verified:false`; it omits quality conclusions/pair data until reverified opening. Python `experiment_matrices` validates page bounds. Studio lists project matrices with next/previous pages and opens via the verified native endpoint. SDK/UI handlers cover paging, open routing and stale project discard; native HTTP covers saved positive/regression matrices, page separation and non-authoritative catalog fields. Direct matrix variant authoring/start and browser catalog qualification remain open.

Direct Studio matrix authoring/execution (2026-10-07): draft controls copy the current saved dataset/selected metrics/settings/prompt reference or inline template, expose unique variant labels, editable model and inline template, protected saved-template preview and removal (2–16 variants, baseline first). Run freezes all request objects and checks shared project/dataset revision/hash/metrics/read-only Chat before model requests. It executes native experiments sequentially, reports current IDs, waits boundedly (2 h per variant), stops new admission on failure/explicit stop, requests cancellation of its own active run and retains IDs/statuses of partial results. Complete runs produce the reverified native persistent matrix; any paired loss keeps overall failure. No settings promotion occurs. Native orchestration executes through the actual JS helper against mock-provider HTTP and verifies three native runs plus saved mixed gain/regression evidence; full Studio regression passes. Draft/run handler tests cover copy/edit/remove, frozen snapshots, incompatible preflight without requests, stop-before-wait cancellation, failed partial evidence and project guards. CLI build passes. This UI sequence lives in the current browser page and is not a durable server job: closing/restarting does not replay/resume the sequence; already started native experiments remain server-owned. Live browser clicks and durable orchestration remain required qualification/work.

Matrix source preflight (2026-10-07): direct Studio orchestration reads and checks the pinned dataset identity/project/version/hash and resolves every saved prompt version before any experiment is admitted. Drafts retain prompt hash pins; mismatched IDs/version/project/hash reject. Prompt reads are cached by immutable ID/version. All resolved/inline templates require `{{input}}` and at most 16000 UTF-8 bytes before launching the first variant. Handler regressions verify that an invalid later prompt/project or inline template causes zero model requests; existing frozen request/stop/partial evidence qualification still passes. Full native mock HTTP exercises the actual helper with this dataset preflight; real browser and durable orchestration remain open.

Saved matrix drill-down (2026-10-07): each saved variant now exposes all/regression/improvement/unchanged metric-pair filters, 20-row pagination and explicit experiment navigation. Regressions are selected initially when present. Navigation fetches the native run and matches run/project/dataset version/hash to the verified matrix before rendering; stale foreign-project controls make no request. Handler tests qualify 45-pair page transitions/reset/empty filters, pinned navigation and rejection of changed hashes. Existing matrix import/save/open and CLI build pass. Browser drill-down remains unqualified.

SDK execution-to-persistence integration (2026-10-07): `Studio.evaluate_matrix(..., persist=True)` explicitly saves complete native run variants and attaches the returned immutable `native_matrix` receipt to the client report. Default behavior stays client-only. Persistence occurs once after all variants terminate; save failures retain every completed result via `matrix_results` and set `matrix_persistence_failed`, without re-executing variants. Client thresholds and strict-improvement results remain in each variant result and are not replaced by the server's separate no-paired-regression matrix gate. SDK tests verify opt-in validation, one save, retained evidence on save failure and that a native pass cannot mask a failed client gate. Native HTTP verifies saved/reopened regression matrix from actual SDK execution.

CI matrix persistence (2026-10-07): evaluation CLI exposes explicit `--persist-matrix` (matrix-file only), delegates to SDK persistence and reports `native_matrix_id` while retaining full evidence in JSON reports. Per-variant client quality gates remain authoritative for CI exits. Native HTTP fixture saves a paired-regression CI matrix and verifies exit 1 plus re-opened native failure. Invalid standalone flag rejects in argument preflight. Hosted CI and cloud-model qualification remain open.

Aggregate receipt integrity (2026-10-07): comparison and native matrix reconstruction validate metric configuration and recompute mean scores from every verified sample using the execution accumulation order. Modified aggregate means are rejected even when sample scores remain intact. Native HTTP fixtures qualify rejection through both comparison and matrix saving; the full mock HTTP suite, 102 Rust unit tests and two integration tests pass. This qualification uses local mock providers.

Matrix save recovery (2026-10-07): a failed final save retains all completed variant IDs and exposes an explicit save-only retry in Studio. Recovery rereads each native experiment and checks project, dataset identity/version/hash and complete strict-quality status before saving. It never launches or cancels models. Repeated storage failures remain retryable; project changes or a newer matrix run prevent stale controls from replacing the current result. Helper and actual UI-handler fixtures qualify save failure, repeated failure, successful recovery, changed dataset rejection and project isolation. Browser interaction and durable orchestration remain unqualified.

Server-owned matrix jobs (2026-10-07): native `/api/evaluation/matrix-jobs` accepts frozen labelled experiment requests and pinned dataset hashes; every variant is validated before any model call. At most two sequences run, each sequentially using the existing bounded native experiment manager. Job and run reservations are private, synced to storage before inference. A separate durable admission flag distinguishes an unsent reservation from a missing attempted-run receipt; the latter is rejected rather than replayed. Provider identity/endpoint/protocol hashes are pinned without credential values, and changed routing blocks resume. GET detail/catalog and explicit cancel/resume retain owned IDs and final immutable matrix receipt. Disconnecting the API caller does not interrupt the worker. Startup marks active sequences interrupted without model replay; explicit resume accepts only verified completed existing runs (or a provably unsent reservation with no run file), and rejects interrupted/failed inference rather than replaying it. Final-save failure can resume without running models again, using the same reserved matrix ID. Python SDK exposes start/read/list/cancel/resume. This first backend milestone left browser migration and explicit recovery of interrupted inference for subsequent work; browser migration is qualified below.

Server matrix qualification: 105 Rust unit tests and two integration tests pass (one credential test ignored), along with 66 SDK protocol tests and the full native/mock HTTP contract suite. Native fixtures cover all-variant preflight before inference, mixed paired regressions/gains, final-save failure/retry with unchanged provider call count, changed routing and missing attempted-run rejection, cancellation of only the owned experiment, completed-job persistence, actual SIGKILL interruption with no replay, and an explicit deterministic between-variant journal fixture that reuses a completed run plus an unsent reservation after restart. No live cloud requests or new browser job flows were qualified.


Server matrix UI integration (2026-10-07): the draft launch now sends one native matrix-job request instead of sequencing experiment requests in the page. Frozen dataset/prompt hash pins reach server preflight (optional `prompt_sha256` is checked against the immutable prompt snapshot). Progress retains the job ID; transport failure or a project change detaches observation without cancelling/restarting the worker. Explicit Stop uses only the job cancellation endpoint. A bounded paginated job catalog and ID-based opening expose refreshed state, explicit resume of completed/unsent variants, cancellation and opening of the reverified saved matrix with exact variant IDs/labels. Superseded page-owned execution/save-recovery helpers were removed. Stale panels and foreign-project responses cannot replace newer state or repeat actions. Six UI helper/handler suites, 105 Rust unit tests + two integration tests, 66 SDK tests and the full native/mock HTTP suite pass. The actual new UI orchestrator runs against the native provider fixture with exactly one job POST and no experiment/matrix POST from the page. Live browser clicks qualify draft copy, launch, saved job/catalog opening and final matrix display; screenshot `/tmp/allpaka-server-matrix-ui-20261007.jpg`. In-browser stop/resume and closing a page during an unfinished model call remain separately unqualified; native HTTP qualifies cancellation, restart/no-replay and explicit between-variant continuation. No live cloud quality was tested.


Explicit matrix retry (2026-10-07): native `POST /api/evaluation/matrix-jobs/:id/retry` creates a distinct job/matrix ID for interrupted, failed or cancelled sequences. It preserves the source file/runs, pins unchanged provider routing and revalidates every available original run configuration. Verified completed strict-quality runs are reused; failed/interrupted/missing/unsent variants receive new reservations and explicitly requested model calls. Active/completed source jobs and active/ambiguous-running native experiments reject. New metadata contains `retry_of` and `previous_run_ids`; journal reads bound/validate lineage. Python SDK exposes `retry_matrix_job`. Studio explains the new calls, submits only on the explicit retry action, opens the new job, and provides parent navigation; stale controls/project changes and substituted lineage cannot replace current state or resubmit. 106 Rust unit + two integration tests, 66 SDK tests, six UI handler/helper suites, CLI build and full native/mock HTTP pass. An actual SIGKILL fixture leaves a completed baseline, interrupted second run and unsent third; retry reuses the baseline, makes exactly two new provider calls, keeps the original bytes/IDs unchanged and preserves paired regression blocking. Browser retry clicks and live cloud qualification remain unverified.


Goal completion telemetry (2026-10-07): native Goal turns that stop with an empty/incomplete reported milestone plan receive `goal_incomplete` instead of `completed`. Studio localizes this state; the session pauses and exposes the saved progress. A bounded continuation may complete the remaining plan within the existing turn step budget, while repeated unsupported completion at the same revision pauses without unbounded calls. This status distinguishes incomplete declared milestones from tool failure and independently verified fulfillment; it does not certify the original objective or model quality.


Conversation observability: `/api/observability/summary` and Python `trace_summary` now return groups keyed by project/session, with first/last trace start times, status counts, model-leaf usage/latency/cache summaries and external model call counts. Time/project/session filters apply before grouping; removed traces remain excluded. Results list the latest 100 conversations by last trace start with deterministic project/session ties and explicit truncation; global totals count all scanned records. Studio displays per-conversation reported token/currency totals and unknown values. Existing summary JSON export includes these groups. The bounds of the existing native scan remain unchanged; this is a trace-derived view, not an inventory of chats with no traces.

Conversation summary qualification: 113 Rust unit + two integration tests pass (one credential test ignored), 67 SDK tests, CLI build, dedicated UI rendering tests and the full native/mock HTTP suite pass. Native/unit fixtures verify per-session status/usage unknowns, leaf-only totals, currency exclusion, distinct projects sharing a session ID and 102 groups with only 100 displayed while all traces remain in global totals. UI tests qualify currency-separated reported values, external call counts, unknown counters and truncation explanation. Browser clicks for this new summary section and live cloud analytics remain unqualified.


Conversation pagination: summary queries accept `conversation_offset` (0–10000) and `conversation_limit` (1–100, default 100). Receipts include the offset, limit, total group count and `conversation_has_more`; deterministic last-start/project/session order is preserved. Global summary totals remain unchanged across pages. Python `trace_summary` exposes the arguments and rejects invalid/bool bounds before HTTP. Studio provides previous/next pages and resets to the first page on Refresh; project/date/scope changes invalidate old controls. The actual async loader ignores stale responses and retains the current request's busy state. Paging reads the current trace set rather than a frozen multi-request snapshot.

Pagination qualification: 113 Rust unit + two integration tests pass (one credential test ignored), 68 SDK tests, CLI build, conversation rendering/navigation tests, actual async loader stale-response tests and the full native/mock HTTP suite pass. A 102-group fixture traverses all groups without duplicates, keeps whole-selection totals across pages and verifies project separation; API/SDK reject invalid page bounds. Native SDK HTTP qualifies empty later pages with unchanged totals and no inference calls. Browser clicks for pagination remain unqualified.


Conversation trace drill-down: Studio's per-conversation summary now opens the existing native trace tree browser inline, with project/session and summary time filters pinned. Lists page 20 traces; stale parent/page responses, repeated old navigation actions and substituted project/session/time data cannot replace current content. Failed reads expose an explicit retry without inference. Existing native trace metadata/details and privacy boundaries are reused.

Drill-down qualification: CLI build, conversation rendering/page-loader tests, the actual drill-down helper tests and full native/mock HTTP suite pass. UI fixtures verify session/project/date pins, 20+1 trace navigation, duplicate stale-action rejection, foreign-scope rejection, explicit read retry and ignored stale parent results. Native SDK/HTTP qualifies summary-to-trace project/session/time filtering with unchanged provider call count. Browser clicks for this new inline drill-down remain unqualified. Rust production sources were unchanged in this increment.


Summary trace-status filtering: `/api/observability/summary` and SDK `trace_summary(status=...)` now filter root trace state before grouping and totals, with bounded lowercase/underscore status validation. Studio exposes completed/error/interrupted/running/no-progress/incomplete-goal/token-limit selection, invalidates stale pages on change and carries the selected state into conversation trace drill-down. Drill-down rejects substituted states as well as substituted project/session/time. Exported summary receipts retain `status`. This filter describes the root trace outcome; model/span status counts within matching traces remain their own recorded outcomes.

Status-filter qualification: 113 Rust unit + two integration tests pass (one credential test ignored), 68 SDK tests, CLI build, summary loader/rendering/drill-down tests and full native/mock HTTP suite pass. Fixtures verify matching and empty status selections, total/group scoping, API/SDK invalid status rejection, filter propagation across summary pages and rejection of a substituted trace status in drill-down. No new browser-click or live cloud qualification.


Time-series status scope: native `/api/observability/time-series` and SDK `trace_time_series(status=...)` now share the summary's bounded root-trace status filter. Filtering occurs before interval assignment and usage/guardrail aggregation; empty intervals remain present. Receipts and chart exports retain the selected state. Studio sends the summary state to the activity chart, checks the returned scope and preserves it in interval drill-down. Error drill-down additionally requires failed root traces; contradictory completed/running/etc. selection cannot open errors from a different selection. Model/span statuses remain independently recorded within matching traces.

Time-series filter qualification: 113 Rust unit + two integration tests pass (one credential test ignored), 68 SDK tests, CLI build, existing time-series UI tests extended with filter propagation/state substitution/contradictory error selection, related summary/drill-down tests and full native/mock HTTP suite pass. Native fixtures select one failed trace from three and verify exact token/currency values plus retained empty intervals without additional provider calls. API/SDK reject malformed status filters. New browser clicks and live cloud monitoring remain unqualified.

Interval trace navigation admission: previous/next controls are now bound to the exact loaded page epoch as well as the selected interval. Once a read starts, old controls cannot admit another request. Explicit interval reload supersedes a pending page, whose late result cannot replace the newer view. Disabled boundary controls do not request data. Returned trace pages additionally require a nonnegative safe total and at most 20 rows. Time-series UI fixtures cover repeated old controls, explicit reload races, malformed totals and oversize pages; related conversation/summary UI tests, syntax/diff checks and CLI build pass. Rust production sources were unchanged. Browser clicks remain unqualified.


Dataset archive foundation: `POST /api/evaluation/datasets/:id/lifecycle` accepts project ID, pinned latest `base_version`, current `base_revision` and boolean `archived`. A bounded private atomically replaced/synced lifecycle marker changes catalog visibility without modifying snapshot bytes/hashes. Stale lifecycle or dataset versions reject with 409; foreign projects reject. `GET /api/evaluation/datasets?project_id=...&archived=true` lists archived datasets; the default catalog excludes them. Archived datasets cannot gain new versions until explicit restoration. Immutable version reads and explicitly pinned evaluation references remain available, including reads used by existing experiments/judges. This is recoverable archival, not physical deletion or invalidation of pinned evaluation sources. Python SDK exposes `datasets` and `dataset_lifecycle`. Studio archive UI and retention remain unfinished.

Dataset archive qualification: 114 Rust unit + two integration tests pass (one credential test ignored), 69 SDK tests, CLI build and full native/mock HTTP suite pass. Fixtures verify archive/recovery catalog membership, retained immutable snapshot/hash, rejected archived writes, lifecycle conflict protection, reload persistence and SDK fail-before-request bounds. HTTP archive/restore makes no provider calls. UI controls and hard deletion/retention remain open.


Dataset archive UI: the existing evaluation dialog now lists active or archived dataset metadata and explicitly archives/restores a selected latest version with its lifecycle revision. Confirmed mutations refresh active choices and the current lifecycle list without clearing the sample editor. Existing valid picker selection survives catalog refresh; removed archived choices disappear. Catalog requests and lifecycle actions are guarded against stale epochs/project/dialog changes. Mutations admit once per button, and confirmed successful mutation remains disabled if subsequent catalog refresh fails. No model call is initiated. Dedicated actual-helper tests qualify archive/restore pins, duplicate/stale/foreign action rejection, picker preservation and stale catalog ordering; related matrix UI suites, syntax/diff checks and CLI build pass. Browser clicks for archive controls remain unqualified. Rust production sources were unchanged in this increment.

Browser archive qualification (2026-10-07): actual Studio clicks archive the dedicated `Архив UI 2026-10-07` fixture, remove it from active choices, show it in the archive, restore it and show it again in the active catalog. The unsaved editor name survives both actions. Native API reads verify exact immutable snapshot/hash and lifecycle revisions 0→1→2; model span count stays 195 throughout the workflow. Browser console has no warnings/errors. Screenshots: `/tmp/allpaka-dataset-archive-browser-20261007.jpg` and `/tmp/allpaka-dataset-restored-browser-20261007.jpg`. Stale/concurrent client scenarios remain helper/native-fixture evidence rather than browser-click qualification. No live cloud requests.


Dataset catalog pagination: optional `offset` (0–2000) and `limit` (1–100) query parameters page active or archived metadata with total/offset/limit/has-more receipts. Equal names are ordered by dataset ID for stable ties. Requests without pagination retain the previous full bounded catalog for existing clients and the editor picker. Python `datasets` exposes and validates optional bounds. Studio archive management pages 20 records, preserves archive-state selection and pinned lifecycle actions across navigation, rejects malformed page metadata and invalidates old buttons when another page starts loading. Physical snapshot retention/deletion is unchanged.

Catalog pagination qualification: 115 Rust unit + two integration tests pass (one credential test ignored), 69 SDK tests, CLI build, lifecycle/picker UI tests and full native/mock HTTP suite pass. Fixtures page 25 equal-name datasets without duplicate/missing IDs, retain unpaged compatibility, validate bounded queries and exact native first-page membership, and navigate a 20+1 UI list with stale control rejection. Browser archive/restore was previously qualified; new multi-page clicks remain separately unqualified. No live cloud requests.


Dataset catalog search: optional bounded `q` (200 Unicode characters / 800 UTF-8 bytes) matches a substring of the latest dataset name or ID after trimming and lowercase normalization. Project/archive filters and search apply before totals/pagination. Python SDK exposes `datasets(search=...)` with pre-request validation. Studio adds a search field and explicit Find button, retaining the active/archive scope across searches; query edits clear the old view and invalidate old lifecycle actions. Pagination and mutation refresh retain the current query. This is literal normalized substring search, not semantic retrieval or accent folding.

Search qualification: 115 Rust unit + two integration tests pass (one credential test ignored), 69 SDK tests, CLI build, lifecycle/picker UI helper tests and full native/mock HTTP suite pass. Fixtures verify ASCII/Cyrillic case normalization, exact ID matching, empty selection, search length rejection, native filtered totals/page membership and query-edit invalidation of old lifecycle controls. Browser archive/restore was previously qualified; search and new page clicks remain separately unqualified. No live cloud requests.

Dataset version history: GET `/api/evaluation/datasets/:id/versions` verifies immutable snapshots and returns newest-first metadata (version, name, sample count, SHA-256), project identity, lifecycle state and bounded pagination (default 20, max 100; offset max 1000). Archived snapshots remain readable. Corrupt or missing selected revisions fail the page rather than supplying unverified metadata. Python `dataset_versions` validates pagination before HTTP. Studio presents 20 revisions per page and loads a selected snapshot only when its identity, project, version and hash match the selected row; stale responses cannot replace a subsequently created editor draft. History performs no inference and does not delete snapshots.

Version-history qualification: 116 Rust unit and two integration tests pass (one credential test ignored), 70 Python SDK tests and focused UI helper tests pass. The UI tests cover snapshot pinning, malformed metadata, duplicate loads, hash substitution and draft replacement during requests. Browser clicks for these new history controls remain unqualified; live cloud qualification remains open.

The rebuilt Studio CLI and full native HTTP contract suite also pass with the local mock provider, including version-history pagination and revision hashes. No live cloud requests were used.

Selected-version loading now also rejects late responses after changing the selected version, dataset, project, editor snapshot or closing the dialog. Responses must match the requested dataset/project/version and provide a valid SHA-256. Failures from superseded requests are ignored. `scripts/test-studio-dataset-load-ui.js` executes the actual UI helper and checks these cases; the version-history helper regression also passes. Existing native optimistic-write tests continue to define old-version saves as conflicts, preventing overwriting newer revisions. This change does not implement branching from historical dataset versions.

Dataset variants now have immutable, content-hashed ancestry: optional `origin` contains the source dataset ID, revision and SHA-256. New variants verify the source snapshot and same-project ownership before writing. Updates preserve ancestry and reject replacing it. Existing snapshots without ancestry retain their original hash algorithm and serialized shape. Source revisions can be historical or archived and are never rewritten. Studio stages an editable variant from the loaded snapshot, clears the selected original, invalidates pending loads and saves under a new ID; source identity/hash is displayed. Python `fork_dataset` validates the source receipt and creates an exact copy with ancestry through the native dataset save endpoint, without inference. Generic JSON import intentionally remains independent.

Variant checks: 117 Rust unit plus two integration tests pass (one credential test ignored), 71 SDK tests pass, and actual UI helper checks cover staged source pins, no save/inference on fork and invalidation of pending loads. Native HTTP checks confirm revision-1 forks after source revision 2, retained ancestry after edits, hash rejection and unchanged original snapshots with zero additional provider calls. New variant buttons are not yet browser-qualified; this is explicit branching, not dataset merge/conflict resolution.

The full native/mock HTTP contract suite passed for dataset variants. An additional focused Rust check verifies that tampering with stored ancestry invalidates the snapshot hash; the final CLI build and JavaScript syntax checks also passed. No live cloud requests were used.

Dataset revision comparison: GET `/api/evaluation/datasets/:id/compare` accepts project ID, two explicit revisions and bounded pagination (100 per page max; offset max 4000). Both snapshots are integrity-verified and must belong to the supplied project. Samples are matched by stable ID; added/removed/changed/unchanged counts cover the entire comparison, while changes are sorted by sample ID and paginated. Changed-field flags cover input, expected output (including null versus empty), ordered contexts and JSON metadata. Name changes are reported separately. Both SHA-256 pins are returned; sample text is not copied into comparison receipts and no inference occurs. Reordering the sample list alone is not a semantic sample change. This compares two revisions of one dataset, not different variant IDs or merge resolution.

Python `compare_dataset_versions` validates bounds before requesting. Studio renders counts, field labels, snapshot hashes and bounded pages, rejects inconsistent page receipts and ignores superseded project/version responses; old page controls cannot launch requests after a newer comparison. 118 Rust unit and two integration tests pass (one credential test ignored), 72 SDK tests and focused UI helper suites pass. Native tests cover all four sample fields, additions/removals, unchanged rows, reverse comparison, identical revisions, project rejection, pagination and hash pins. New comparison controls remain unqualified in the browser.

Comparison qualification also includes a successful CLI build and the full native HTTP contract suite using a local mock provider. HTTP checks verify changed fields and exact snapshot pins, reject excessive page limits and confirm zero additional provider calls. No live cloud requests were used.

Server-owned matrix job admission now honors the catalogue's existing 1000-job and 16 MiB storage limits before spawning execution. Capacity checks run under the shared persistence mutex, so simultaneous admissions cannot independently consume the same remaining space. Each admitted job reserves its immutable serialized fields plus 16 KiB for bounded mutable progress (up to 16 run IDs, status, attempted flags and a 1000-character escaped error). New jobs whose reservation would exceed the per-job 2 MiB or global bound are rejected without starting inference. Retry creates a new job and follows the same admission path; updating/resuming an existing job retains its reserved capacity and remains possible at the count limit. Nothing is automatically removed from history. Existing installations already beyond the reservation budget reject new jobs while retaining existing records; physical cleanup/retention policy remains open.

Capacity Rust fixtures fill all 1000 catalogue slots and verify admission rejection plus an existing-job update. A separate byte-budget fixture leaves fewer than 1000 jobs but exhausts reserved bytes, verifies rejection and verifies progress/error serialization fits the reservation. 120 Rust unit tests plus two integration tests pass (one credential test ignored).

Final capacity qualification: 121 Rust unit plus two integration tests pass (one credential test ignored). A concurrent admission fixture gives two writers room for only one new reservation and verifies exactly one succeeds. CLI build and full native/mock HTTP contracts pass. The HTTP fixture temporarily fills 1000 persisted jobs, confirms admission fails with no new file or provider call, and confirms catalogue/detail reads still work; only fixture-owned temporary records are removed afterward. The byte reservation can reject admission before the count limit, depending on job size. No live cloud requests were used.

Python dataset comparison CSV export now gathers the complete bounded list (up to 4000 changed IDs, 100 per request). Every page must preserve dataset/project/revision identity, both SHA-256 pins, total and counts. Pages must have exactly the expected length and continuation flag, IDs must strictly increase across boundaries, field flags must be known and unique, and observed change counts must equal the receipt totals before any CSV is returned. Invalid/substituted/incomplete pages fail the export. Columns include dataset/project, both revisions/hashes, sample ID, change type and changed field names; no questions, reference answers or context bodies are copied. Spreadsheet-formula prefixes are escaped. Empty comparisons return a header-only CSV. This is a Python SDK feature; there is no new download button in Studio yet.

SDK export qualification: 73 tests pass, including a 101-change two-page fixture, snapshot hash substitution, oversized totals, malformed fields, duplicate boundary IDs, count mismatch, foreign project and spreadsheet escaping. Native HTTP export verification checks the exact changed fields and both immutable source hashes with no additional inference calls.

The complete native/mock HTTP contract suite passed with the comparison CSV export check, and Python compilation and whitespace checks passed. No live cloud requests were used.

Studio now also downloads complete dataset comparison CSVs from the reviewed comparison, including unvisited pages. The collector reuses the reviewed page, fetches remaining pages under its captured dataset/project/revision scope, verifies the original hashes/totals/counts on every page, requires strictly increasing unique IDs across page boundaries, and checks full observed counts before creating any file. Page validation is shared with comparison rendering and additionally rejects invalid IDs, unknown/duplicate field flags and inconsistent change types. Old controls, changed scope or a closed dialog cannot download; repeated clicks while collection is in progress do not issue duplicate requests. CSV cells are quoted and formula-prefixed strings escaped; downloads have UTF-8 BOM. No sample text or model call is introduced.

The actual JavaScript helper/button suite verifies complete 101-change export over two pages, pinned hashes, duplicate-click exclusion, substituted second page rejection, scope-change cancellation, duplicate boundary IDs and spreadsheet escaping. Dataset load/history/variant helper regressions and syntax/whitespace checks also pass. Browser download interaction remains unqualified; these are helper/runtime tests, not browser-click evidence.

Pinned single-input prompt preview is now available through POST `/api/evaluation/prompts/:id/versions/:version/preview`, Python `preview_prompt` and Studio. It verifies immutable prompt identity/hash and same-project ownership, accepts a bounded question/context list, renders the existing text-template semantics and returns the exact stored system instruction plus rendered user message, source pins, zero provider calls and `saved:false`. Literal placeholders inside supplied question/context text are not expanded again; the system instruction is not interpolated, matching the dataset runner. Rendering is incrementally bounded to 1 MiB before appending, with question/context input limits; this avoids unbounded repeated-placeholder expansion. Preview neither writes a dataset/run nor performs inference. It is the preview stage of Playground: standalone single-input inference and multi-message prompt templates remain open.

Preview qualification: 122 Rust unit tests plus two integration tests pass (one credential test ignored), 74 SDK tests pass, and actual UI-helper checks verify message roles/hash/version pins and stale-request, edited-input, changed-prompt and closed-dialog exclusion. Rust fixtures test literal-marker preservation, context joining, system semantics, project rejection, hash tampering and expansion limits. Native HTTP verifies pinned messages and no extra provider calls, and rejects oversized input. Browser interaction for the new preview remains unqualified.

The rebuilt Studio CLI and full native/mock HTTP contract suite passed for prompt preview. No live cloud requests were used. The final frontend build includes preview invalidation when creating a prompt variant.

Single-input Playground now has a native launch path: POST `/api/evaluation/playground` accepts Chat settings without workspace writes, an explicit saved prompt reference and reviewed SHA-256, one question/context list, optional reference/metrics and a bounded timeout. Prompt/project/hash, settings, reference-dependent metric requirements and preview expansion bounds are checked before persistence or inference. It uses the existing experiment manager, trace tree, guardrail enforcement, concurrency admission, output/usage persistence, cancellation, restart interruption and export path. The one-sample dataset is committed only after acquiring an existing experiment slot; the resulting run retains the complete pinned prompt snapshot and dataset hash. No new provider runner is introduced. A disk failure after saving the sample can leave a retained sample without a successful run; there is no cross-file transactional cleanup.

Playground runs are explicitly tagged in receipts/catalog/export. Metrics may be empty only for internally admitted single-input Playground requests; ordinary experiments still require metrics. An unscored completed response has `strict_quality:false` and empty score summaries and cannot be admitted to quality comparisons. SDK `start_playground` freezes caller settings/contexts and supports explicitly chosen metrics/reference; Studio uses unscored mode, captures prompt/question/model configuration, excludes duplicate launch clicks and routes accepted results to the existing history/results. A response arriving after edits or a project/prompt change does not replace the current result view. Closing the dialog does not cancel a durable accepted run.

Local qualification so far: 123 Rust unit plus two integration tests pass (one credential test ignored), 75 SDK tests and actual Playground/preview UI helper tests pass. The launch helper verifies fixed source pins, captured question/settings, Chat/no-write mode, duplicate exclusion, stale-view exclusion and wrong-receipt/project rejection. Rust verifies ordinary experiments cannot skip metrics and Playground admits exactly one sample. Standalone browser clicks and multi-message prompt templates remain open.

Full native/mock HTTP qualification passed for single-input Playground: one question produces exactly one model call and a completed unscored receipt with pinned prompt/dataset hashes and a saved sample; wrong reviewed hashes and write-enabled settings fail without creating dataset/run files or provider calls. Input blocking rules stop the request before inference. Export identifies Playground receipts and unscored quality comparisons are rejected. The final CLI build, JavaScript syntax and whitespace checks passed. No live cloud requests were used.

Versioned chat-message templates now support an optional immutable `messages` array of alternating User/Assistant templates, beginning and ending with User (up to seven few-shot pairs plus the final question). The final User requires `{{input}}`; inline text `template` must be empty for this format. System instructions remain separate and literal. Each content is at most 16000 UTF-8 bytes and all message templates total at most 64000; rendered history is incrementally limited to 1 MiB. Content/roles are hashed together with prompt metadata and ancestry. Snapshots without messages preserve their pre-change serialized shape and SHA-256 algorithm.

Save/load/variants/history retain message templates; Python `save_chat_prompt` freezes and validates the structure before requesting. Studio provides format selection and a non-JSON pair editor that inserts/removes whole User/Assistant pairs while retaining the final question. Native preview returns all rendered messages. Playground and ordinary scored dataset experiments send the complete message history through the existing provider adapter; input guardrails inspect the joined history so a private marker in the final User cannot bypass checks against a benign first example. Literal markers inside supplied question/context text remain literal. This provides bounded few-shot conversational templates; arbitrary system/tool-message templates and provider tool-call examples are not implemented.

Current local checks: 124 Rust unit tests plus two integration tests pass (one credential test ignored), 76 SDK tests and chat editor/preview/Playground UI helper checks pass. Fixtures cover hash changes on message edits, bad role ordering, conflicting formats, literal rendering, role-pinned previews, whole-pair editing and the 15-message bound. New browser interaction remains unqualified.

Full native/mock HTTP contracts passed for message templates. The fixture verifies the four-message preview exactly equals the provider request, preserves full message templates in run receipts, executes the same chat template in Playground and scored dataset mode, blocks a forbidden final User before inference despite benign earlier examples, and rejects invalid Assistant-first templates. CLI build, frontend syntax and whitespace checks passed. Legacy prompt revision/hash/variant/recovery contracts also continue to pass. No live cloud requests were used.

Browser qualification (2026-10-07): in an isolated native Studio data directory with a local model stub, the browser created a conversational prompt, selected message format, added one User/Assistant example, edited all three messages and the system instruction, saved version 1 and opened its single-question preview. API and stub evidence confirm save/preview caused zero model calls. The browser then launched a question, refreshed the run, opened its completed answer and trace tree. The provider received exactly the four messages shown by preview, once; the run retained the exact saved prompt snapshot/hash, an unscored completed response and native experiment/item/model trace with reported 9 input / 5 output tokens. Browser console warnings/errors were empty. The result screenshot and evidence packet are `/tmp/allpaka-playground-browser-20261007.jpg` and `/tmp/allpaka-chat-prompts-browser-evidence.json`. The temporary tab and owned preview processes were closed; fixture files were retained. This qualifies save/preview/unscored launch/result/trace interactions, not cancellation, reload during a call, scored browser execution or live cloud compatibility.

Experiment catalog now supports optional bounded offset/limit (max offset 1000, max page 100) and filters for status, provider ID, dataset ID, case-normalized literal model substring and Playground boolean. Project and all filters apply before total/pagination, with deterministic ID-descending order. Queries without pagination preserve the legacy full bounded catalogue. Receipt metadata includes project identity, totals, page continuation, zero provider calls and `summaries_verified:false`; stored quality flags are observational summaries, not reverified promotion evidence. Existing full-scan bounds remain 1000 artifacts / 128 MiB and fail rather than truncate silently.

Python `experiments` validates bounds/filter types before requesting. Studio browses 20 rows per page with status/model/type/current-dataset/current-provider filters; query edits invalidate old lists/actions, and old-project responses/errors cannot replace the current view. Detail opening pins project/run/dataset hash. Page receipts reject wrong project, metadata shape, order or filter membership. Legacy full catalogue is independently retained for baseline/candidate choices so browsing a filtered page does not remove otherwise valid comparison selections. Initial dialog opening resets filters and new dataset selections invalidate a dataset-scoped list.

Qualification: 125 Rust unit tests plus two integration tests pass (one credential test ignored), 77 SDK tests, actual catalog/Playground UI helpers and full native/mock HTTP contracts pass. Rust fixture checks 25 rows across two pages, case normalization, intersected filters, foreign-project exclusion and legacy full listing. Native HTTP checks exact filtered membership/totals, distinct pages, Playground-only listing and rejected bounds/status with no extra provider calls. UI fixtures check 20+1 paging, full comparison choices across pages, stale controls/details/responses/errors, absent dataset scope and mismatched filtered responses. New catalog filter/page browser interaction remains unqualified.

Studio Playground now exposes the existing optional native metrics/reference support. Six deterministic metrics are selectable independently; only JSON-validity scoring can run without a reference. The launch captures the metric set and explicit reference value (null versus an intentionally supplied empty string), and receipt admission checks the exact selected metric set alongside source prompt pins. Editing metrics/reference while launch is pending cannot replace the current view with a response for the previous settings. Loading/saving a prompt resets the metric/reference controls to unscored defaults. No metric is selected implicitly.

A read-only Playground result source panel fetches the saved one-sample dataset, verifies run/project/version/hash/sample identity and content types, and shows the actual question, reference and contexts used for that run. It ignores stale selections and prevents duplicate source requests. This makes the score's basis reviewable even after current editor fields change.

Qualification: 77 SDK tests pass, including selected-metric/reference capture; Playground launch and saved-source UI helper suites verify reference requirements, frozen reference/metric sets, receipt-set substitution rejection, old-form edit exclusion, source hash checks and stale source responses. CLI build, JavaScript syntax and whitespace checks pass. The full native/mock HTTP suite verifies a scored Playground run with exact-match and token-F1 both 1.0, retained reference in its immutable dataset and one model call; missing references are rejected before writing a dataset or invoking a model. This does not qualify the new scored browser interaction yet; earlier browser qualification covered unscored launch only. No live cloud requests were used.

Experiment storage admission now checks the existing 1000-record and 128 MiB catalog bounds under the persistence mutex before spawning inference. Running records reserve immutable serialized fields plus bounded per-item output/progress space; terminal records account for actual bytes plus progress headroom. Updates to existing records remain possible at the record limit. Playground commits its immutable input sample only after capacity admission. No history is deleted automatically; physical retention management remains open.

Saved experiment output now obeys both the 65536-byte UTF-8 limit and a 128 KiB + 16-byte JSON serialization bound, preventing control-character escaping from exhausting a reserved run. Truncated replies retain a readable bounded output but cannot receive metric scores or strict-quality status.

Qualification: 128 Rust unit tests and two integration tests pass (one credential test ignored), CLI build and native/mock HTTP contracts pass. Fixtures verify that full-catalog rejection invokes neither the Playground sample write nor the provider, existing records remain updateable, and competing reservations admit only one writer when one slot of byte capacity remains. UTF-8 and JSON escaping limits are covered locally. No live cloud requests were used.

Additional native/mock HTTP qualification sends 65536 NUL characters through the real provider/experiment pipeline and confirms bounded persisted JSON, retained truncated output, failed item/run status, empty scores and strict_quality=false. The full HTTP suite passes after this fixture, including persistence/restart and crash-recovery contracts.

Browser qualification of scored Playground: using an isolated data directory and local mock provider, Studio saved a prompt version, selected exact-match and word-F1 metrics with an explicit reference, and launched one input. Both metrics display 1.0; persisted strict-quality status and the pinned source dataset were independently checked. The saved-source panel retains the original question, reference and context after changing editor fields and reopening the source. Exactly one provider request was recorded; browser warning/error logs were empty. Evidence: `/tmp/allpaka-playground-scored-browser-evidence.json` and `/tmp/allpaka-playground-scored-browser.jpg`. Preview and temporary tab were closed. This covers scored launch and source inspection, not all catalog/export/matrix browser workflows or live cloud integration.

Experiment JSON exports now retain provider/model, trace ID, concurrency/timeout and immutable prompt identity/version/hash (null for inline templates) while continuing to omit full settings and prompt bodies. Answer inclusion remains explicit. The Python CSV exporter now accepts empty metrics only for an explicitly identified unscored Playground receipt; ordinary metric-free experiment receipts and strict-quality claims without metrics remain rejected. A focused SDK fixture covers unscored output opt-in, absent score columns and both rejection cases. Native/mock HTTP fixtures cover prompt/source provenance, metadata-only truncated output export, explicit output inclusion and comparison rejection without another model call.

Export qualification: 128 Rust unit plus two integration tests, 78 Python SDK tests, CLI build and the complete native/mock HTTP suite pass. JSON provenance fields and retained incomplete-output evidence were verified through the server export endpoint; no live cloud calls were made.

Studio and Python CSV experiment exports now flatten provider/model, trace ID and saved prompt ID/version/hash into named columns. Missing optional provenance (including legacy receipts and inline prompts) becomes empty cells. Text provenance uses the same spreadsheet-formula escaping as sample IDs and outputs; Python rejects malformed supplied prompt references, including boolean versions. SDK and UI helper fixtures cover these fields, absent source values and formula escaping. Actual browser CSV download remains unqualified.

CSV provenance qualification: 78 Python SDK tests, Studio export helper checks, JavaScript syntax, CLI build and full native/mock HTTP contracts pass. The HTTP fixture verifies CSV model/provider/trace/prompt version/hash against the persisted run while omitting outputs and incomplete-answer scores. No live cloud calls were used.

Real browser CSV download qualification: a saved scored Playground fixture was opened in Studio and exported twice through the CSV control, first with answers disabled and then explicitly enabled. Actual downloaded files in Downloads were parsed independently: provider/model, trace ID, prompt ID/version/hash and exact-match score match the saved run; output is absent in the first file and exact in the second. Export adds no provider calls (one fixture inference total), browser warning/error logs are empty. The browser download-event observer timed out, but both actual downloaded files proved completion; no repeated metadata export was performed. Evidence: `/tmp/allpaka-csv-browser-fixture.json`, `/tmp/allpaka-csv-browser-evidence.json`, `/tmp/allpaka-csv-browser.jpg`. Temporary browser tab and preview process were closed; downloaded evidence was retained.

Human feedback version catalog foundation: native GET `/observability/traces/:id/feedback/versions` and Python `feedback_versions` expose newest-first metadata pages (offset 0–2000, limit 1–100, default 20). Rows include revision/saved time and total/active annotation counts; comments, corrections and reviewer names are omitted. The trace target is validated; full selected revision reads use the existing endpoint. This adds history discovery, not annotation queues or verified reviewer authentication. Studio history-page UI remains open.

Feedback history qualification: 128 Rust unit plus two integration tests, 79 SDK tests, CLI build and full native/mock HTTP contracts pass. Fixtures verify newest-first pages, removal/restoration active counts, empty history, page bounds, source revision persistence across Store reload and metadata-only responses with no provider calls. No new history-list UI or live cloud behavior is claimed.

Studio feedback history-list UI now uses the native metadata catalog, displays 20 newest-first revisions per page with saved time/active counts, and opens the selected full revision after checking trace/version identity. Old page actions, duplicate pending requests and responses/errors from a replaced or closed review panel are excluded using a view/page epoch. The existing version picker and read-only historical view remain supported. The helper regression fixture covers a 20+1 page boundary, revision selection, malformed receipt rejection and stale action/response/error suppression. Browser history-list interaction remains unqualified; annotation queues remain open.

Feedback selection race fix: latest refresh, explicit version reads and review mutations now share a request epoch and verify trace/version identity before replacing the selected receipt. A slow save/remove/restore response cannot replace a newer selected view; persisted user-authorized writes remain stored even if their response is no longer current. Closed panels ignore stale responses/errors. Helper fixtures reproduce reverse-order version reads, substituted trace IDs, closure during refresh and a late save after opening an historical revision. Browser coverage of this new race handling remains open.

Feedback close/reopen qualification: closing a review panel now invalidates both pending request and selected-view epochs; reopening redraws the retained revision with fresh controls. A response initiated before closure cannot replace the reopened panel even when it arrives after reopening. The selection regression fixture explicitly covers this timing in addition to reverse-order reads and late writes. Studio browser interaction for this case remains unqualified.

Real browser feedback history qualification: an isolated mock-provider trace with 21 review revisions was opened in Studio. The catalog displayed 20+1 versions across pages; opening version 1 showed one annotation and no editing form. Closing/reopening retained that historical selection; selecting latest version 21 restored the editing form. A browser-authored score/comment created version 22, independently verified through the API while version 1 remained immutable. History/review actions added no model calls (one fixture inference total). Browser warning/error logs were empty before the final save. Evidence: `/tmp/allpaka-feedback-browser-fixture.json`, `/tmp/allpaka-feedback-browser-evidence.json`, `/tmp/allpaka-feedback-history-browser.jpg`. Temporary tab and preview were closed. Deliberately delayed response timing remains covered by helper tests, not this browser fixture.

Studio human-review filtering: local case-insensitive substring search over explicit annotation author/metric/category/comment/correction and all/active/deleted filtering now hide unmatched rows without rebuilding the editing form. A matching/total counter refers to filtered annotations; aggregate summaries are explicitly labeled as the whole revision. Filter values persist across revision redraws. Helper fixtures verify case folding, deleted membership, no-match counts and stable form identity. Browser interaction for these new filters remains unqualified.

Python human-review workflow support: `feedback` reads current or explicitly pinned revisions; `save_feedback` captures a JSON snapshot of supplied annotations and appends through the native optimistic base-version endpoint. Known fields, author/ID/span types, UTF-8 field bounds, finite bounded scores, mutually exclusive score kinds and explicit deletion flags are validated before HTTP. Existing returned IDs support edit/remove/restore; server revision conflicts are not retried automatically. Reviewer identity remains self-reported; queues and authenticated teams remain open. SDK fixtures cover immutable request capture, version zero reads, invalid/oversize annotations and no-request rejection.

SDK review qualification: 80 SDK tests and the full native/mock HTTP suite pass. The server fixture reads the initial empty review through SDK, saves numerical/comment/correction feedback through SDK, and reads the immutable first revision after native updates/removal/restoration. No live cloud calls were used.

Durable review queue source foundation: POST `/observability/review-queues` creates a named project queue with explicit instructions and 1–200 unique trace/span references; GET `/observability/review-queues/:id` reads the saved source manifest. Admission verifies existing non-removed traces, existing spans and matching project ownership before writing. A private atomic synced manifest survives Store reload. Storage bounds are 1000 queues, 128 MiB catalog and 1 MiB per manifest. No raw model content is captured and no provider is called. This is source admission/storage only: catalogue UI, reviewer assignment/claims, item status/completion linked to feedback, queue revisions/export and authenticated teams remain unfinished.

Review queue source qualification: 129 Rust unit plus two integration tests, CLI build and the full native/mock HTTP suite pass. Fixtures cover Store reload preserving instructions/trace/span references, duplicate source rejection, traversal rejection, HTTP create/detail equality and missing trace/span rejection with no inference calls. Cross-project admission and capacity-edge behavior need dedicated fixtures before broader qualification claims.

Review queue catalog and Python access: native GET `/observability/review-queues?project_id=...&offset=0&limit=20` pages bounded project-filtered metadata in descending queue-ID order (offset 0–1000, limit 1–100). Rows omit instructions/targets and include name/version/target count. Scans honor 1000 records/128 MiB and reject non-file entries. Python creates captured unique trace/span sources, reads manifests and validates page bounds before requests. This still does not implement review assignments, completion or queue UI.

Review queue catalog qualification: 129 Rust unit plus two integration tests, 81 SDK tests, CLI build and full native/mock HTTP contracts pass. Additional Store fixture checks a 20+2 page boundary and strict descending cross-page IDs. Native HTTP uses SDK create/detail/catalog, validates metadata omission, creates a valid second project, rejects its attempt to include another project's trace, and confirms its empty catalog without provider calls. The initial fixture used an invalid project ID/empty roots and was corrected before qualification; it did not establish cross-project behavior. Capacity-edge fixtures and full queue workflows remain open.

Review queue assignment foundation: POST `/:id/assignments` and Python `assign_review_queue` set or clear a bounded self-reported reviewer on an existing source index, with optimistic base-version protection (409 on stale version). Mutable queue manifests retain source references/instructions and atomically sync the new version/assignment map; legacy source manifests default to empty assignments. Updates respect per-file/catalog byte and count limits and never remove queues automatically. Assignment revision history, authenticated ownership, completion feedback linkage and queue UI remain open.

Review assignment qualification: 129 Rust unit plus two integration tests, 82 SDK tests, CLI build and full native/mock HTTP contracts pass. Store fixtures verify assignment persistence, clearing, stale source revisions, invalid target indices and two simultaneous writers admitting exactly one update. HTTP uses SDK assignment/detail, verifies 409 as SDK `EvaluationError(reason="http_409")`, clears the assignment and retains original queue sources without inference. The initial HTTP fixture incorrectly caught the raw urllib error instead of the SDK error and was repaired before final qualification. Browser assignment UI and authenticated reviewer ownership remain unimplemented.

Review queue completion foundation: native POST `/:id/completion` completes a source only after checking an explicit saved feedback version/annotation ID against the exact trace/span and its assigned self-reported reviewer. The annotation must be non-deleted in that immutable version. Queue evidence retains reviewer/version/annotation identity; completed sources cannot be reassigned until explicit reopen, which clears completion while retaining assignment. Source/queue base-version conflicts and byte limits apply to mutations. Catalog metadata adds assigned/completed counts. Later feedback revisions do not silently rewrite pinned historical evidence. Completion remains a human review marker, not verified identity or automatic quality promotion. Completion SDK/UI and assignment revision audit history remain open.

Completion qualification: 129 Rust unit plus two integration tests, CLI build and full native/mock HTTP contracts pass. The fixture rejects evidence from a different assigned reviewer, completes from the matching saved span annotation, verifies stored evidence, rejects reassignment/duplicate completion, reopens explicitly and completes again without provider calls. Completion persistence/restart, later deletion semantics, stale completion conflicts and wrong-span/unknown-annotation evidence still need dedicated fixtures.

Python completion/reopen workflow: SDK methods validate source index, queue base version and positive saved feedback version before sending the native completion/reopen actions. Reopen sends no substitute evidence; conflicts are surfaced without retry. SDK fixtures cover exact action bodies and fail-before-request bounds. Extended HTTP fixtures cover deleted-in-selected-version evidence, wrong-span evidence, unknown annotation IDs, stale completion versions, explicit completion from a retained earlier non-deleted version and queue evidence retention across restart. Queue UI and authenticated reviewer identity remain unfinished.

Completion SDK/evidence qualification: 83 SDK tests and full native/mock HTTP contracts pass. SDK completion/reopen calls reach the native workflow; wrong reviewer/span, unknown/deleted selected annotations and stale queue versions reject. Explicit earlier-version evidence remains usable after later removal; the completed queue manifest, assignments and annotation/version pin compare exactly after Studio restart. No live cloud calls were used.

Studio queue UI foundation: context panel opens a project catalog with 20-record pages, queue selection, instructions and per-source assignment controls. Source opening verifies trace/project/span identity and defaults the feedback form to the selected span. Trace details can create a named/instructed queue for the current whole trace or selected step without manual IDs. Catalog and selected-queue epochs are separate, so queue selection retains usable catalog controls; old project/page/selection responses are excluded and duplicate mutations are blocked. Helper fixtures cover pages, repeated selection, assignment version payloads, substituted source rejection, frozen create sources and removed-panel responses. Multi-source creation remains SDK-only; completion/reopen controls and actual queue browser qualification remain open.

### Review queue completion UI (2026-10-07)

Studio queue details now load an explicit feedback revision (or the current revision), filter active annotations by the assigned reviewer and exact trace/span source, and show the selected saved annotation before completion. Completion captures the feedback version and annotation ID; the native server validates those pins. Completed targets can be reopened while retaining their reviewer assignment. Opening a queue source prefills the manual feedback reviewer and span.

Local UI fixtures cover reviewer/source/deletion filtering, captured completion evidence, substituted evidence rejection, duplicate write exclusion, reopening, and stale version/error responses. Queue catalog and feedback selection fixtures pass. This UI increment has not been qualified in a real browser; reviewer names remain self-reported and queue assignment/completion revision history is not yet implemented.

Queue creation UI now accepts multiple selected spans of the current trace (1..200 references) and checks UTF-8 byte limits for the name/instructions before submitting. UI fixtures cover exact two-span source capture and empty selection rejection. Cross-trace batch creation remains SDK-only; this change is locally tested, not browser-qualified.

Review queue catalog filtering is now native and available in Studio and SDK: pending (any incomplete target), completed (all targets complete), unassigned (any target without assignment), and exact self-reported reviewer name. Filters apply before totals/pagination; receipts echo filters and the UI rejects substituted filters. Reviewer matching includes completed assignments. No authenticated ownership is implied.

Review queue CSV export now exists in Studio (displayed saved snapshot) and SDK (one current detail request). It includes queue version, trace/span source, assignment, completion status, and saved feedback version/annotation reference, without loading feedback/trace content or invoking models. Local fixtures verify row contents, missing span handling, spreadsheet formula escaping, duplicate target rejection, foreign queue identity and mismatched evidence reviewer rejection. Browser download qualification for queue CSV remains open.

### Real browser queue qualification (2026-10-07)

On an isolated local Studio with a mock provider, browser tab 44 assigned a reviewer, selected the saved feedback preview, completed the target with revision 1 evidence, downloaded actual CSV, combined completed/reviewer filters, and reopened the target. Native API verified queue versions 3 and 4, exact annotation pins, retained assignment and cleared completion. Parsed Downloads CSV matched the persisted completed queue and evidence. Browser warning/error logs were empty; exactly one fixture model call occurred (source Playground), with no calls for review actions. Evidence: `/tmp/allpaka-queue-browser-evidence.json`, screenshot `/tmp/allpaka-review-queue-browser.png`. Multi-source creation and delayed-response races remain qualified by local fixtures only.

Queue catalog now supports case-insensitive substring search on queue names through native API `name`, SDK and Studio. Nonempty search is bounded to 200 UTF-8 bytes, applies before pagination, and is echoed in filter receipts. Fixtures cover case-insensitive match, no match, invalid queries and UI/SDK request capture. Browser qualification of name search remains open. Explicit HTML labels were added for catalog search/status/reviewer controls after browser locator inspection.

Review queues now have reversible archive/restore lifecycle with optimistic version checks, native durable writes and `archived` catalog filter, Studio controls, SDK method and CSV archive state. Missing archive fields in existing records default to false; omitted API/SDK filter preserves all-queue listing. Archived queues remain readable/exportable, but target assignment/completion/reopening is rejected until restoration. Unit fixtures verify persistence across store reopening, stale lifecycle conflict, archive filtering and preserved source/assignment. UI fixtures verify archive/restore controls and disabled assignment. SDK validates boolean state/filter before requests. No immutable lifecycle audit history or archive-based capacity reclamation is claimed; browser lifecycle qualification remains open.

HTTP/SDK lifecycle qualification now runs in `scripts/test-studio.py`: completed queue archive v10→11, combined archived/completed/reviewer/name filtering, cross-project isolation, assignment/reopen rejection while archived, duplicate-state rejection, stale lifecycle 409, CSV preserved annotation/version pins, restore v11→12 and rearchive v12→13. Provider call count remains unchanged throughout these actions. Restart assertions require exact archived queue v13 with retained evidence. Full native test suite passes (129 unit tests, 2 integration tests, one credential test ignored).

Review queue changes now append local history entries in the same atomically replaced queue artifact: create, assignment (including clear), complete (exact feedback version/annotation), reopen, archive and restore. GET `/api/observability/review-queues/:id/history` and SDK `review_queue_history` provide version-descending pages (1..100, offset0..2000), total and `history_complete`. Existing records without history remain readable; their new entries do not fabricate earlier changes. Each queue is bounded to 2000 entries and the existing 1MiB artifact limit; archive does not free capacity. This is append-only application behavior, not cryptographic or authenticated audit evidence. Native fixtures cover recorded ordering, lifecycle entries and legacy partial history; 87 SDK tests pass. UI history browsing is not yet implemented; HTTP/restart qualification was added to the contract fixture and still needs a run on the new binary.

The full mock-provider Studio contract suite passed on the new history binary. Queue history pages returned versions13/12 and the create tail, completion entry retained revision1 and its original annotation ID, and the exact latest history page survived server restart. No live cloud requests were made.

Studio queue history browsing is now implemented with 20-entry pages, localized actions, assignment removal, completion reviewer/revision/annotation evidence and explicit partial-history message. Receipt validation pins queue/project/current displayed version, rejects malformed ordering/evidence and ignores stale responses/errors after panel closing or queue replacement. Local UI fixtures cover 20+1 pagination, partial history, snapshot version substitution, duplicate entry versions and stale closure/view errors. Real-browser history qualification remains open.

Explicit SDK online callback evaluation foundation: `@studio.track('task', evaluate=callback)` invokes the evaluator only for successful function results inside this client's active trace. Sync functions require sync evaluators; async functions accept sync/async evaluators. Numeric score mappings (1..20 technical metric names, finite0..1) are saved in a child tool span's `usage.evaluation_scores`, native validated and exported as reported external metadata. Task arguments/results and evaluator exception text are not persisted. Evaluator errors mark the child span failed and remain locally accessible via `trace.evaluation_errors` while preserving a successful task result; cancellation propagates. These are caller-reported assessments, not independently verified native metrics or human reviews. No sampling rules, background evaluator service, prompt/version provenance or automatic judge invocation are implied. Local SDK tests cover opt-in execution, sync/async results, privacy and failure isolation; native tests cover metadata shape/bounds. HTTP persistence qualification remains open.

HTTP callback evaluation qualification now verifies saved child-span numeric scores, exact parent source, successful task preserved when evaluator fails, private output absent from trace/export, and native rejection of empty/boolean/out-of-range/nontechnical-name scores or scores on model/failed spans. Provider call count stays unchanged. SDK fixtures also prove evaluator is skipped on business failure and business exception text is omitted from metadata. Full Rust suite passes (131 unit tests,2 integration tests,one credential test ignored) and89 SDK tests pass. Restart fixture compares the full retained online trace.

Studio trace detail now renders numeric callback scores inside the corresponding span with an explicit caller-reported assessment label. Invalid score objects are rejected in rendering. Actual traceView fixtures exposed undefined reviewSpan/reviewerName parameters introduced by the earlier queue integration; fixed by preserving existing relatedDepth/removed arguments and adding explicit fourth/fifth review scope arguments, and updating queue source calls. Fixtures now execute the actual traceView for normal, queue-scoped and removed traces, plus score ordering/malformed input. Browser qualification of score rendering and corrected queue source opening remains open.

Explicit SDK callback sampling now supports `evaluation_sample_rate` in [0,1], default1. Rate0 skips all evaluations; interior rates use a versioned SHA-256 key over project ID, caller correlation ID, tracked technical name and parent span index. Selection is stable for matching identities/trace structure; it is a probabilistic rate over identities, not an exact quota. Skipped calls emit no evaluation span or pass score. Sampling requires an explicit evaluator; invalid/bool/nonfinite rates are rejected before execution. Sync/async rate0 and100 repeated identities at0.5 are covered in90 SDK tests. This is SDK caller configuration, not a persisted server rule, budget cap or authenticated provenance; exported scores alone do not describe the sampling denominator.

Sampling provenance now persists on each successful evaluator-configured task span as `usage.evaluation_sampling` (method sha256_v1, sample_rate, selected), including skipped tasks. Native ingestion recomputes the versioned project/correlation/name/span-index hash decision, rejects selected-flag substitutions and unknown receipt fields, and preserves receipt metadata in exports. Native fixtures share an SDK-generated0.5 golden decision and reject inverted choices for rates0/0.5/1. Task failures have no successful-result sampling receipt; the denominator is successful decorated calls only. Receipt presence proves the chosen sampling decision, not evaluator execution success or independently verified quality. HTTP persistence and UI receipt presentation remain open.

Studio now presents sampling receipt selection and configured percentage on the original successful task span, explicitly separating skipped quality from evaluation success. Selected tasks point to the child assessment outcome. Actual trace UI fixtures reject unknown receipt fields/methods and boolean rates and verify skipped/selected wording. HTTP fixture adds selected and skipped native receipt persistence/export, no callback for rate0, substituted selected-flag rejection and exact trace survival across restart.

Real browser tab46 qualification on isolated Studio verified queue source opening, auto-selected span1 and assigned reviewer in manual feedback, callback quality0.75, selected100% and skipped0% labels, history create/assignment entries, archive disabled assignment and restoration preserving assignment, then history restore/archive/assignment/create ordering. Native API verified queuev4, exact retained scores/sampling and history. Private result absent from trace, zero model calls, browser warning/error logs empty. Evidence `/tmp/allpaka-online-browser-evidence.json`, screenshots `/tmp/allpaka-online-browser.png` and `/tmp/allpaka-online-browser-history.png`. Multi-page and partial-history UI remain local-fixture qualified only.

Explicit callback identity now accepts paired `evaluator_id`/`evaluator_version` in SDK track. Each attempted assessment child stores `usage.evaluator_ref` before execution, including failed evaluators. Native ingestion validates technical ASCII ID, positive u64 version and tool kind; public export retains only the exact id/version shape. Studio displays caller-reported evaluator identity/version. Callback source/code is not hashed or independently verified, and skipped task sampling does not imply assessment execution or evaluator provenance. SDK and native shape/bounds tests and actual trace UI identity fixtures pass; HTTP/restart identity qualification remains open.

HTTP identity fixture now verifies caller-reported evaluator ID/version on both completed and failed assessment spans, retained identity in export, and rejection of boolean/zero versions, nontechnical names, additional private fields and evaluator references on model spans. No model calls occur. Full native suite passes (132 unit tests,2 integration tests,one credential test ignored). Full trace equality on restart includes the evaluator references.

Native callback evaluation summary foundation: GET `/api/observability/evaluations/summary?project_id=...` and SDK `callback_evaluation_summary` scan persisted active project traces with optional inclusive since/until boundaries. Metric groups separate evaluator ID/version and unlabelled legacy assessments, returning count/mean/min/max plus selected/skipped task and completed/failed assessment counts. Assessment counts use external tool spans labelled evaluation or carrying assessment metadata. Existing trace scan limits apply, groups bounded4000; no truncation or automatic promotion. Local native fixtures verify evaluator version separation, project isolation, mean/count, failed evaluator count and time exclusion; SDK validates request bounds. Studio summary UI and HTTP/restart summary qualification remain open.

Studio callback summary UI now has project-scoped time filters, selected/skipped and completed/failed counts, metric table separating evaluator versions and anonymous groups, and local50-row pages. Receipt identity/time/source and numeric bounds/group uniqueness are validated; filter edit/dialog close/project changes invalidate old results/errors. UI fixtures cover51-group paging and stale/duplicate group rejection. Native mean is clamped to observed min/max to avoid floating accumulation crossing its range for repeated identical scores. Native focused tests and CLI embedding build were run; real HTTP/restart summary and browser summary qualification remain open.

HTTP summary qualification now verifies exact evaluator identity/metric mean/min/max/count, selected2/skipped1 and completed1/failed1 counts, project/time exclusion, invalid time/unknown parameter rejection, no private output and unchanged provider call count. Restart fixture compares the summary at fixed time boundaries with the original report. Native rounding fixture repeats0.1 twenty times and verifies mean remains exactly within min/max. Full Rust suite passes:133 unit tests and2 integration tests,one credential test ignored.


### Callback summary exports

Studio exports the displayed callback summary as JSON or CSV. Exports include all metric groups, including groups beyond the visible page, and preserve project/time filters, caller-reported provenance and global counters. CSV repeats global counters on each metric row; an empty summary includes a context-only row. Text cells escape spreadsheet formulas. Changing filters, closing the dialog or changing project invalidates old download controls. Local UI fixtures cover pagination, all-group export, empty metrics, formula escaping and stale controls; browser download qualification remains pending.

Python SDK now provides `export_callback_evaluation_summary(project_id, format="json"|"csv", since_ms=..., until_ms=...)`. It reads one scoped summary, validates counters, metric bounds, duplicate groups, evaluator references and provenance before exporting. CSV preserves counters even without metric rows and escapes formula text. Local SDK fixtures cover 51 groups, empty results, filters, malformed receipts and no-request rejection of unsupported formats.

HTTP qualification: SDK JSON export matches the native summary exactly; parsed CSV retains metric values, evaluator version, global counts and exact time boundaries. A foreign empty project produces a context-only CSV row. Provider request count stays unchanged. The full Studio contract suite passed with the local mock provider, including summary persistence after restart; live cloud and browser download checks remain unqualified.

Browser qualification: a real Studio summary displays caller-reported evaluator browser-check version 2, quality 0.75, one selected and one skipped call. Both CSV and JSON buttons create files in Downloads; parsed files match the native fixture and preserve counters and identity. No private outputs, provider calls or browser warnings/errors were observed. Browser download-event notification timed out although the CSV file was successfully saved; qualification used the saved files. Pagination and stale controls remain covered by local UI fixtures.


### Explicit callback summary CI thresholds

Python SDK `check_callback_evaluation_summary(project_id, requirements, since_ms=None, until_ms=None, max_failed_assessments=None)` validates version-pinned requirements before requesting one summary. Each requirement specifies `evaluator_id`, `evaluator_version`, `metric`, positive `min_count`, and `min_mean` and/or `min_score`. Missing groups, insufficient samples, means below thresholds, or observed minima below thresholds fail; equality passes. Versions and unidentified legacy groups cannot substitute for the requested identity. The optional failure limit covers the entire scoped summary, not individual evaluator versions, since failed assessments have no scored groups.

The returned receipt contains `passed`, per-requirement reasons and observations, the exact scoped summary, and the optional global failure check. Caller-reported provenance is preserved; this does not certify independent quality or promote a model automatically. Local SDK fixtures cover ties, failures, missing versions, low sample counts, individual score regressions and invalid preflight without HTTP requests. Native HTTP qualification of this gate is pending.

```python
receipt = studio.check_callback_evaluation_summary(
    'default',
    [dict(evaluator_id='exact-check', evaluator_version=2, metric='exact',
          min_count=20, min_mean=0.95, min_score=0.8)],
    since_ms=window_start_ms, until_ms=window_end_ms,
    max_failed_assessments=0,
)
if not receipt['passed']:
    raise SystemExit(1)
```

Callback gate HTTP qualification passed: the exact version and score thresholds pass on equality; a zero global failure limit fails for the retained failed assessment; requesting another evaluator version fails with `missing_metric_group`. All checks retain the exact time-scoped summary and make no additional model calls. The full Studio contract suite passed against the local mock provider, including persistence and process-recovery scenarios. SDK fixtures also reject excessively large integer scores/thresholds without numeric-conversion overflow. Live cloud qualification remains open.


### Callback CI command

The existing `scripts/evaluate-studio.py` entry point accepts `--callback-requirements-file` with a JSON array of SDK requirements. Optional `--callback-since-ms`, `--callback-until-ms` and `--callback-max-failed-assessments` scope the check. This mode rejects mixed model/dataset/task options and accepts the existing `--report` and `--junit` destinations. Exit codes: 0 pass, 1 failed thresholds, 2 request/receipt/artifact error, 130 interruption. Reports are created without overwriting existing evidence. Requirements are bounded to 128 KiB and duplicate JSON keys are rejected before network requests.

```sh
python3 scripts/evaluate-studio.py --base-url http://127.0.0.1:7431 \
  --project-id default --callback-requirements-file callback-requirements.json \
  --callback-since-ms 1791324000000 --callback-until-ms 1791410400000 \
  --callback-max-failed-assessments 0 --report callback-result.json \
  --junit callback-result.xml
```

JUnit contains one case per metric requirement and one for the optional global failure limit, preserving failure reasons and observations. Local command fixtures exercise pass/fail/error codes, exact query bounds, JSON/JUnit evidence, invalid/oversized/duplicate-key configurations, existing artifact preservation and mixed-mode rejection without network. SDK's underlying gate is HTTP-qualified; command-level HTTP qualification remains pending.

Callback CI command HTTP qualification: subprocess runs against native Studio return 0 for exact thresholds with one allowed failed assessment and 1 for a zero failure limit. Saved JSON equals SDK gate receipts exactly; JUnit cases/failure counts match these decisions. Provider request count is unchanged. JUnit properties additionally identify the project, time boundaries, caller-reported source and no automatic promotion; these properties are covered by local command fixtures.

Full Studio contracts passed after placing callback CLI fixtures/reports in a separate artifact directory outside conversation history. The initial fixture placement under the history root caused restart admission to reject report JSON as malformed conversations; the test was corrected, with no runtime format changes. Final run includes callback CLI HTTP checks and all existing restart/recovery checks against the local mock provider.


### Per-evaluator callback attempt counters

Native callback summaries now include `evaluators`, bounded to 4000 distinct evaluator ID/version pairs. Each group counts completed and failed assessment spans once, independent of the number of scored metrics. Failed assessments remain visible without metric scores; unlabelled legacy assessments form a distinct null-ID/null-version group. Project/time scope and caller-reported provenance are preserved. SDK JSON export validates identity uniqueness, exact counter shapes and agreement with global completed/failed totals when these groups are present; older summary receipts remain compatible. CSV retains metric rows/global counts and does not yet export per-evaluator attempt groups. Focused native and SDK tests pass for version separation, failed-only evaluators, unlabelled failure, duplicate groups and inconsistent totals. UI display, version-specific failure thresholds and HTTP qualification remain pending.


### Version-specific callback failure limits

Each SDK/CLI metric requirement now optionally accepts `max_failed_assessments` for its exact evaluator ID/version. This counts assessment attempts rather than metric rows. Other evaluator versions cannot cause or mask this failure check. An absent per-evaluator attempt group, including old servers without these counters, fails with `evaluator_attempts_unavailable`; it is not treated as zero failures. Threshold equality passes. The existing top-level limit still applies to the entire scoped summary. Gate receipts and JUnit preserve the observed per-evaluator attempts and failure reasons. Local SDK tests cover version isolation, failure, equality, old-server missing data and boolean rejection; native HTTP fixtures have been added but have not yet run against the rebuilt server.

Studio now displays per-evaluator completed/failed attempt counters in a separate section paginated by 50, including failed-only evaluator versions. Receipts are validated for unique identity/version groups, exact shapes, bounded counters, aggregate totals and metric identities before display or download. Older servers explicitly show unavailable counters. Local UI fixtures cover 51 evaluator groups and a failed-only final page, and reject mismatched totals. Browser qualification for this additional section remains pending. The full Rust suite passed (133 unit tests, two integration tests, one credential-dependent ignored test); HTTP assertions confirm exact-check v2 has zero failures while invalid-check v3 has one, so version-specific limit zero passes and global limit zero fails.

The full updated HTTP Studio suite passed with the local mock provider, including evaluator counters/gates and exact callback summary persistence after restart. No live cloud requests were made.

SDK `export_callback_evaluator_attempts_csv(project_id, since_ms=None, until_ms=None)` exports completed/failed attempt counts independently of scored metric rows. Failed-only and unidentified groups are retained, with exact version identity and scope/provenance columns. Empty groups yield one context-only row with blank attempt counts; older servers without per-evaluator counters fail explicitly. The method obtains and validates one summary, escapes spreadsheet formulas and makes no model calls. SDK fixtures qualify failed-only/unidentified rows, empty results and unavailable counters; HTTP and browser qualification of this separate CSV remains pending.

Studio now provides “Скачать попытки оценщиков CSV” inside the per-version attempt section. It exports every evaluator group from the validated displayed receipt, including groups outside the visible page and failed-only versions, without requesting fresh data. CSV retains scope/provenance and escapes formula text; empty summaries retain a context-only row. Old controls cannot download after dialog close, filter or project change. Local UI fixtures verify all 51 groups, failed-only final row and closed-dialog exclusion. SDK HTTP attempt-export assertions are added; browser download qualification remains pending.

Browser qualification of per-evaluator attempts and CSV passed: expanded Studio section shows failed-check v3 with zero completed/one failed attempt and good-check v2 with one completed/zero failed. Clicking the attempts CSV button creates a real Downloads file whose parsed rows equal the HTTP-qualified SDK export exactly. No private outputs, provider calls or browser warnings/errors were observed. Pagination beyond 50 groups remains locally fixture-qualified.

### Saved consolidation proposal review in Studio

The memory dialog now opens retained consolidation proposals in pages of 20 for the selected project or global scope. Opening a proposal loads its candidate into the editable memory form with both proposal provenance and exact source revision pins. It makes no generation request and does not save a note. Explicit save still validates current source revisions on the server. UI contract checks cover duplicate request admission, scope/editor stale response exclusion, malformed or duplicate pins, and preservation of both origins. The 100 Python SDK tests pass. This increment has not been visually qualified in a live browser; provider/model quality remains unqualified.

Saved consolidation catalog browser qualification: a retained mock-generated proposal opened into editable fields, with no new provider request and no note created until explicit save. Saving edited text retained both proposal and source revision provenance; the two source notes and original proposal stayed unchanged. Browser warnings/errors were empty. Evidence: `/tmp/allpaka-catalog-browser-evidence.json`, screenshot `/tmp/allpaka-consolidation-catalog-browser.png`. This verifies browser behavior against a local mock, not real-model semantic quality.

Proposal storage HTTP qualification now passes for both count and byte exhaustion. Conversation extraction and consolidation return rejection with zero additional mock-provider requests, unchanged memory notes, and no added retained proposals. Temporary quota fixtures are removed before continuing, and ordinary generation plus restart persistence still pass. The full Studio contract suite completed successfully; evidence log `/tmp/allpaka-proposal-capacity-http.log`. Real-cloud/model quality qualification remains separate.

Server-owned online evaluation rule groundwork: native `online_evaluation::Rule` validates project/rule/evaluator identities, pinned positive evaluator version, enabled state and finite sample rates in 0..1. SHA-256 selection keys include project, rule, evaluator identity/version and trace identity, giving stable selection independent of ordering/restart and distinct samples for evaluator revisions. Two native tests pass across 1,000 traces, restored definitions, reverse scan order, project isolation, disabled/zero/full rates, invalid identities/nonfinite rates and unknown fields. This definition is not yet wired to persisted rules, trace admission or an execution queue; automatic server evaluation remains incomplete. Evidence `/tmp/allpaka-online-rule-tests.log`.

Immutable online evaluation rule persistence: POST `/api/observability/online-rules` saves a strict native rule definition after checking the project; GET `/api/observability/online-rules/:hash` reads a verified SHA-256-addressed snapshot. Schema-1 snapshots preserve rule/project/evaluator/version/sample rate/enabled state, reject unknown/duplicate JSON fields, nonregular files, escaped directories, oversized reads, identity/hash mismatch and invalid values. Save is idempotent, bounded to 1,000 definitions and uses atomic no-overwrite commits under a write lock. Native tests verify reopen, evaluator revision separation and tamper rejection. Three online rule tests and 102 SDK tests pass. HTTP qualification, catalog/lifecycle, active rule selection, execution queue and runtime evaluation are still open; saving an enabled definition does not yet activate automatic evaluation. Logs `/tmp/allpaka-online-rule-storage-tests.log`, `/tmp/allpaka-online-rule-sdk.log`.

Online rule snapshot HTTP qualification passed: idempotent saves and exact reads, new immutable hash for evaluator revision 2 with revision 1 retained, rejection of unknown projects/fields and invalid sampling/version/enabled types, tampered snapshot read/save rejection, and exact two-version persistence after Studio restart. Rule API requests invoked no provider. The full Studio contract suite passes (`/tmp/allpaka-online-rule-http.log`). Rule catalog/lifecycle, active bindings and evaluator execution remain required work.

Online rule catalog: GET `/api/observability/online-rules?project_id=...&offset=...&limit=...` and SDK `online_evaluation_rules` browse immutable verified definitions, default page size 20 and maximum 100. The bounded 1,000-file scan validates every snapshot before filtering by project, sorts by rule hash, and returns filtered totals/has_more with provider_calls zero and automatic_execution false. A corrupt snapshot rejects the entire catalog rather than returning partial results. Native storage tests cover two pages, empty foreign project, invalid limits and tampered catalog rejection; three online tests and 103 SDK tests pass. HTTP catalog qualification, Studio UI, active bindings and execution queue remain pending. Evidence `/tmp/allpaka-online-rule-catalog-tests.log`, `/tmp/allpaka-online-rule-catalog-sdk.log`.

Online rule catalog HTTP qualification passes: two evaluator revisions appear on separate limit-1 pages with stable ascending hashes and filtered totals, global scope is empty, malformed query/oversized limit are rejected, tampered storage rejects the whole catalog, and both page packets survive restart exactly. Catalog requests invoke no provider and explicitly report automatic_execution false. Full Studio contract suite passed (`/tmp/allpaka-online-catalog-http.log`). Studio catalog UI, active rule binding/lifecycle and execution remain required work.

Online rule binding persistence: GET/POST `/api/observability/online-rule-bindings` and SDK read/bind methods store immutable binding revisions for a project/rule identity, pinning a saved rule hash and active state. Updates require exact base_version, retain prior revisions and stop at 1,000. Reads verify current revision/hash, contiguous history filenames, pinned snapshot identity and enabled state for active bindings. Binding identity directories use SHA-256 rather than user path segments; nonexistent reads do not create directories. Native tests cover activation, stale rejection, evaluator snapshot switch, deactivation, reopen, disabled activation rejection and current binding tamper rejection. Four online tests and 104 SDK tests pass. HTTP binding qualification, UI, durable execution admission and worker integration remain open: persisted active state does not yet run evaluations. Evidence `/tmp/allpaka-online-binding-tests.log`, `/tmp/allpaka-online-binding-sdk.log`.

Online binding HTTP qualification passed: initially absent binding, activation revision 1, stale-write rejection, evaluator snapshot switch revision 2, deactivation revision 3, foreign scope absent, invalid active type rejection, tampered binding read/write rejection, and exact current status persistence after restart. Binding requests caused no provider calls. The full Studio contract suite passes (`/tmp/allpaka-online-binding-http.log`). Automatic runtime selection/execution and Studio controls remain incomplete.

Durable online selection admission: POST `/api/observability/online-selections` and SDK `select_online_evaluations(project_id, trace_id)` admit only existing, nonremoved, completed native traces in the specified project. Active binding decisions pin full binding revision/hash and rule snapshot, are sampled deterministically, and persist immutable receipts keyed by project/trace. Repeating admission retains original decisions after binding changes/restart; new traces reflect current active bindings. Reads validate receipt digest, identities, ordering, pinned binding digest and saved rule sampling. Storage is bounded to 1,000 selections and 1 MiB per receipt; provider_calls zero and automatic_execution false remain explicit. Five native online tests and 105 SDK tests pass. HTTP trace/selection qualification, automatic trace-triggered admission, queue workers, evaluators and trace-content evidence pins remain incomplete. Selection freezes rule decisions, not mutable trace contents. Evidence `/tmp/allpaka-online-selection-tests.log`, `/tmp/allpaka-online-selection-sdk.log`.

Online selection HTTP qualification passed on existing completed external native trace records: active binding revision 4 is pinned with a deterministic boolean decision; deactivation does not change the retained receipt; a different trace after deactivation has no decisions. Missing/foreign-project traces are rejected, no provider calls occur, and repeating selection after restart returns the exact retained packet. Full Studio contract suite passes (`/tmp/allpaka-online-selection-http.log`). Running/removed target HTTP rejection, receipt file tamper/limits, frozen trace evidence, automatic admission and evaluator worker execution remain required qualification or implementation.

Online selection rejection qualification now passes through HTTP for removed and currently running native traces, tampered persisted decision, receipt larger than 1 MiB, and full 1,000-file catalog. Existing valid receipt remains readable at capacity; rejected new admission creates no receipt and provider-call count remains unchanged. Temporary quota/tamper fixtures are restored before persistence checks. Full Studio contract suite passed (`/tmp/allpaka-online-selection-rejections.log`). Frozen trace evidence, automatic trace admission and evaluator worker execution remain incomplete.

Online selection trace evidence pin: schema-2 receipts now include SHA-256 of the completed native metadata trace record, and bind that fingerprint into the selection digest. Repeat admission rejects changed trace metadata rather than silently treating it as the original source. Native fingerprint tests cover running/foreign rejection, stability across store reopen and change detection when a completed child span is added. Five online selection tests, one trace-fingerprint test and 105 SDK tests pass. This pins metadata only: original prompts/answers are not captured. Older schema-1 selection files are retained and rejected for execution/admission because they lack evidence pins; migration/read-only legacy review remains pending. HTTP fingerprint/tamper/restart qualification, automatic trigger, worker queue and evaluators remain incomplete. Logs `/tmp/allpaka-online-trace-pin-tests.log`, `/tmp/allpaka-trace-fingerprint-tests.log`.

Schema-2 selection trace fingerprint HTTP qualification passed: receipt contains a 64-character trace SHA-256; modifying a completed trace child-span name rejects repeated selection, restoring the original trace restores exact receipt replay without provider calls, and unchanged fingerprint/selection persist across restart. Full Studio contract suite passed (`/tmp/allpaka-online-trace-pin-http.log`). This is a fingerprint of known native metadata, not prompt/answer evidence. Legacy schema-1 read-only review/migration, automatic admission and evaluator execution remain incomplete.

Read-only online selection archive: GET `/api/observability/online-selections?project_id=...&trace_id=...` and SDK `online_evaluation_selection_archive` read verified schema-1/schema-2 receipts without migration, trace mutation or provider calls. Archive envelope explicitly reports trace_evidence_pinned and execution_eligible false; schema-1 remains ineligible for repeat admission. Reads bound paths/bytes, verify original legacy digest and common pinned decision semantics, preserve the original receipt and never add a trace hash to legacy data. Native tests prove exact legacy/current readback and unchanged legacy file bytes, while repeat admission of legacy receipt still fails. Five online tests and 106 SDK tests pass. HTTP archive qualification, reviewed migration, automatic admission and evaluator execution remain pending. Evidence `/tmp/allpaka-online-archive-tests.log`, `/tmp/allpaka-online-archive-sdk.log`.

Online archive HTTP qualification passed for schema-2 exact readback and schema-1 legacy fixture. Legacy digest is generated independently in Python; archive returns exact legacy receipt, explicitly reports no trace evidence pin and execution_eligible false, preserves file bytes and blocks repeat admission. Restoring original schema-2 receipt restores exact archive readback; that packet persists across restart. Provider count remains unchanged and full Studio contract suite passes (`/tmp/allpaka-online-archive-http.log`). Reviewed legacy migration and automatic evaluator execution remain incomplete.

Native online assessment execution foundation: POST `/api/observability/online-assessments` and SDK `assess_online_trace` execute selected pinned definitions against completed native trace metadata. Builtin evaluator `trace_health` version 1 reports `span_success_rate` as completed spans divided by all spans. Other evaluator IDs/versions produce retained failed assessments with evaluator_unavailable and no invented score. Immutable aggregate receipts pin selection/trace/rule/binding/evaluator identities, are bounded to 1 MiB and 1,000 files, and reverified on repeat admission. This currently uses explicit synchronous local execution, not automatic triggers/background workers. Native tests cover scores, reopen/idempotence, changed evidence rejection and unavailable evaluator failure. Six online tests and 107 SDK tests pass. HTTP qualification, metadata score-count coverage, automatic admission, durable worker queue, model/rubric evaluators and semantic quality remain incomplete. Evidence `/tmp/allpaka-online-assessment-tests.log`, `/tmp/allpaka-online-assessment-sdk.log`.

Native metadata assessment qualification: a native trace with completed root/model and failed tool scores exactly 2/3, remains stable after reopen, rejects a running child, foreign project and removed target. HTTP execution verifies trace_health/version1 score1 on completed SDK trace metadata and evaluator_unavailable failure with no scores for a selected unsupported evaluator; exact repeated results and both packets survive restart without provider calls. Catalog/binding fixtures are re-snapshotted after adding evaluator definitions. Full Studio contract suite passes (`/tmp/allpaka-online-assessment-http.log`), plus focused native test (`/tmp/allpaka-trace-health-tests.log`). This is explicit metadata evaluation, not answer quality or automatic production evaluation. Automatic trigger/queue, result UI and model evaluators remain incomplete.

Automatic native trace admission: a completed root span now attempts durable online job admission after trace flush. Projects without active bindings create no selection/job; active selected decisions pin schema-2 selection/trace evidence into immutable pending jobs. Jobs are idempotent by selection hash and bounded to 1,000 files/8 KiB reads. Failed/interrupted roots do not enqueue. Errors are reported locally and do not change the primary completed trace. Native test verifies automatic pending job creation, repeated enqueue deduplication, failed-root exclusion, and full-queue failure isolation with the primary trace remaining completed. Seven online tests and 107 SDK tests pass (`/tmp/allpaka-online-auto-admission-tests.log`). This hook currently covers native Guard root completion only; external trace ingestion, HTTP qualification, startup pending-job recovery, background workers and automatic evaluator execution remain incomplete.

Automatic admission HTTP qualification passes via ordinary chat completion: one model request produces completed native trace and immutable pending job pinned to active binding revision7 and schema-2 selection/trace hashes. At full 1,000-job capacity, explicit admission failure is observed in server logs while the second chat answer/trace remain completed; the queue gains no job. Pending job and selection survive restart exactly. Test fixture uses a distinct online_auto_trace variable to avoid later compaction-test shadowing. Full Studio contract suite passes (`/tmp/allpaka-online-auto-admission-http-recheck.log`). Pending jobs still have no background consumer/startup recovery, and external trace ingestion is not yet hooked.

Durable job processing foundation: POST `/api/observability/online-jobs/drain` and SDK `drain_online_evaluation_jobs(limit=20)` explicitly process batches of 1–100 jobs under a single worker lock. Immutable pending job files remain intact; separate immutable job-result receipts pin job/selection and assessment hashes. Changed/unavailable trace evidence becomes a retained failed result without an assessment, unsupported evaluator attempts become evaluator_failed results, and already finished jobs are skipped only after result/linked assessment integrity verification. Native tests verify success, repeat/reopen behavior, linked assessment tamper rejection, changed-evidence failure with no assessment, and no replay of terminal failures. Eight online tests and 108 SDK tests pass (`/tmp/allpaka-online-job-processing-tests.log`, `/tmp/allpaka-online-job-processing-sdk.log`). HTTP batch qualification, asynchronous consumer/startup recovery, result UI and model evaluators remain incomplete. Current job results explicitly report automatic_execution false because batches are invoked explicitly.

Online batch HTTP qualification passes: explicit batch completes one automatically admitted native job, repeat skips its terminal result, and a second job whose trace metadata changes after admission becomes a terminal failed result. Restoring source data does not replay that failure. Exactly three deliberate chat fixture model calls are counted; evaluation batches add none. Both immutable job-result file bytes and terminal batch counts survive restart exactly. Invalid batch limit is rejected. Full Studio contract suite passed (`/tmp/allpaka-online-batch-http.log`). Asynchronous default consumer/startup processing, external-ingest admission, result UI and model evaluators remain incomplete.

Online queue background consumer: Studio now starts an owned asynchronous worker after native recovery, immediately checks retained jobs, and polls every two seconds in bounded batches of 20. Blocking storage work runs outside the async executor and shares the manual drain mutex. Dropping the Studio worker guard aborts future polling; an already admitted bounded storage batch may finish. Repeated identical worker errors are logged once until the error changes or recovery succeeds. ALLPAKA_ONLINE_WORKER=0 explicitly disables this worker (manual HTTP contract fixtures use this setting). New assessments and job results preserve automatic_execution=true for worker execution; verified existing records retain their original execution origin when later read through manual APIs. Nine native online tests pass, including retained pending-job execution, worker recreation without replay, immutable terminal bytes and explicit assessment reuse of automatic provenance. Studio binary builds. These tests qualify local trace-health execution, not model answer quality; default-worker HTTP lifecycle qualification remains open.
The full Studio HTTP contract suite also passes with explicit manual-worker isolation (`/tmp/allpaka-worker-http.log`, mock provider, no live cloud requests). This verifies compatibility of the existing rule/binding/admission/drain/restart APIs, not automatic HTTP worker lifecycle.

External trace online admission: successful HTTP external ingestion now admits completed traces with no running spans to the existing pinned selection/job queue, including idempotent retries. Failed roots are excluded. Admission errors are logged without rejecting the successfully stored primary trace. The default-worker HTTP lifecycle test scripts/test-studio-online-worker.py passes with no model provider configured: a pending external job created in manual mode is processed after Studio restart in default mode; a newly ingested external trace is processed automatically; a duplicate creates no extra job; a failed trace creates none; a second restart retains exact terminal result bytes and manual drain skips both results. Queue-capacity failure also preserves successful external ingestion. This proves metadata trace-health execution, not external answer quality or trust in caller-reported telemetry.
Post-admission compatibility qualification: full Studio HTTP contracts pass with the mock provider (`/tmp/allpaka-external-worker-http.log`), and all 108 Python SDK tests pass. No live cloud calls were made.

Online job catalog: GET /api/observability/online-jobs?project_id=...&offset=...&limit=... and SDK online_evaluation_jobs return pending/completed/failed rows, immutable job definitions and verified terminal result packets. Pages default to 20, accept 1..100 rows and offset up to 1000, sort by selection SHA-256, and filter project before totals. The bounded 1000-job scan validates jobs and existing result digests/linked assessment evidence before filtering; corrupt records reject the catalog. Reads hold the processing mutex but never execute evaluators or create result directories. Terminal packet shape and outcome/error consistency are now validated on repeated processing as well. Nine online native tests pass, including read-only pending catalog, completed evidence, foreign project, empty page, invalid limit and result tamper rejection; 109 SDK tests pass. The dedicated default-worker HTTP test qualifies pending reads without result-directory mutation, two result pages, automatic provenance and restart alongside existing worker lifecycle checks. Studio binary builds. Job catalog Studio UI remains unfinished; native metadata health does not qualify answer quality.

Studio online queue UI: “Очередь оценок” opens the current project's retained job catalog, with pages of 20, pending/completed/failed states, automatic/manual execution origin, failure explanations and expandable saved job/result evidence. Refresh returns to the first page; closing the dialog invalidates pending responses and old page controls, and changed project excludes old replies. The UI checks catalog scope, exact pagination, ordered unique selection IDs, job/result identity, terminal shape/outcome and no-provider metadata before rendering. It displays metadata health as distinct from answer quality. Focused UI fixtures pass for pagination, automatic origin, substituted project/result rejection and stale/closed-dialog exclusion; existing callback summary UI tests also pass. Real-browser visual qualification remains pending.

Studio online rule management: “Правила оценки” browses saved project rule snapshots in pages of 20, reads current bindings, distinguishes the active snapshot from other saved versions, and explicitly activates/deactivates a selected version with optimistic base_version. Saving a trace_health v1 rule accepts a technical rule name and 0..100% sampling, retains an immutable snapshot and does not automatically activate it. Unsupported saved evaluator IDs remain visible with their identity; the composer only offers the currently supported native health evaluator. Duplicate mutation clicks are excluded, disabled snapshots cannot activate, closed dialogs/changed projects exclude stale replies, and conflicts require refreshing bindings. Local fixtures qualify successful save, sampling input validation, activation/deactivation revision pins, double-click exclusion, stale mutation response and foreign snapshot rejection; existing queue UI fixtures pass. Browser interaction/visual qualification remains open; real answer-quality evaluators remain incomplete.

Browser qualification for online rules and queue: an isolated Studio instance with no model provider was operated through the real browser UI to save browser-health (trace_health v1, 100%), activate binding revision 1, display one automatically evaluated external trace, reopen after server restart and deactivate through revision 2. A subsequent external trace remains stored but receives no job. API/storage evidence corroborates the UI actions in /tmp/allpaka-online-browser-evidence.json. The initial expanded raw evidence overflowed the modal; evidence is now a separately collapsed disclosure with wrapped lines and bounded modal width. Corrected rendering was inspected and saved at /tmp/allpaka-online-queue-browser.png. Rule and queue UI fixtures and Studio build pass after the fix. This qualification covers desktop browser interactions, not multi-page browser behavior or real model quality. The temporary browser tab and test server were closed; isolated evidence data is retained.

Trace-linked model judge: the existing explicit evaluation/judge endpoint accepts optional trace_ref {trace_id,trace_sha256}. It verifies exact completed, nonremoved project trace metadata before provider admission and again before retaining a verdict. The immutable judge receipt pins this reference and explicitly records answer_source=caller_supplied and trace_content_verified=false; trace metadata does not establish that submitted text was the actual traced answer. SDK judge_trace exposes the same explicit text/rubric/settings flow. Existing judge timeouts, output bounds, no-tools behavior, guardrails and observational verdict semantics apply. Two focused native judge tests pass, covering incomplete/foreign/changed trace rejection alongside verdict validation; 110 SDK tests pass. This is explicit model evaluation with trace provenance, not automatic online model execution. Content admission/retention, model-evaluator configuration and queue integration remain necessary for that full feature; real-model semantic qualification remains open.
Trace-linked judge HTTP qualification passes in the full Studio suite (`/tmp/allpaka-trace-judge-http.log`): explicit SDK scoring invokes the mock provider once, persists exact trace pins/caller-supplied provenance, reads the identical immutable receipt, and rejects a substituted source hash without another provider call. No live cloud requests were made. Concurrent trace mutation during model execution is native revalidated by implementation but does not yet have HTTP timing qualification.

Explicit online quality text sources: POST /api/observability/online-quality-sources stores an immutable caller-supplied {project_id,trace_id,trace_sha256,input,output,reference} snapshot independently of metadata-only traces. The exact completed/nonremoved project trace fingerprint is checked before storage and again before commit. SHA-256 binds the complete typed source; identical saves reuse verified retained bytes, changed content creates a distinct snapshot. Source text is not independently certified as the traced answer (answer_source=caller_supplied, trace_content_verified=false). Input/output/reference limits are 16/64/64 KiB, serialized requests/receipts at most 160,000 bytes, storage at most 1000 JSON snapshots/16 MiB. Reads via GET /api/observability/online-quality-sources/:hash?project_id=... verify strict JSON, shape, source hash and scope; historical source reads remain available after trace metadata changes, while new admission rejects changed metadata. SDK save_online_quality_source and online_quality_source expose explicit capture and reading; trace ingestion does not capture content automatically. One native source test and 111 SDK tests pass. The dedicated HTTP worker suite passes idempotent source save/read, substituted content rejection and exact source receipt/bytes after restart with no model provider configured. Model worker configuration, source-to-job binding and automatic quality judging remain unfinished.

Pinned quality-source judge execution: POST /api/observability/online-quality-sources/:hash/judge accepts only settings/rubric, loads the exact retained source server-side, revalidates its project/completed-trace fingerprint, and delegates to the existing bounded model judge. SDK judge_online_quality_source exposes it. Judge receipts retain quality_source_sha256 alongside exact trace_ref and caller-supplied answer provenance. The general judge endpoint's optional quality_source_ref also requires input/output/reference and trace_ref to match that saved source exactly before provider admission and again before storing a verdict; source substitution cannot gain the pinned provenance label. Native source tests reject execution after metadata change while preserving historical reads; two judge tests and 111 SDK tests pass. This completes explicit scoring of pinned retained content, while durable background model jobs/rule configuration remain unfinished. No live-model quality claim is made.
Pinned-source judge full HTTP qualification passes with the mock provider (`/tmp/allpaka-pinned-source-judge-http.log`): source capture followed by SDK model scoring retains exact input/output, source/trace hashes and caller-supplied provenance; a forged input on the general judge endpoint is rejected before another provider call. The retained verdict is read identically. No live cloud requests were made.

Durable background model quality jobs: POST /api/observability/online-quality-jobs accepts a saved source hash, explicit settings and bounded rubric; GET /api/observability/online-quality-jobs/:id returns pending/running/completed/failed/interrupted state, immutable plan and terminal result. The deterministic job ID hashes the normalized request, so identical submissions reuse the same job and never create an implicit retry. Admission verifies source/project/completed trace and Chat-without-writes settings before saving; 1000 job directories and 64 KiB packet reads bound storage scans. Studio starts an owned serial worker that polls every two seconds, commits a no-overwrite claim before invoking the existing source judge, and retains the judge ID/receipt hash on completion. Terminal reads reverify claim, result digest and linked saved verdict identity/hash/source/rubric/project. Startup turns claimed but unterminated requests into interrupted results without replay; unclaimed pending requests remain eligible. Dropping the worker cancels ongoing async work; a provider request may already have happened, hence claimed work is never automatically retried. One native job test passes for pending retention, one-time claim, interrupted recovery, stable terminal bytes and result tampering; 112 SDK tests pass. SDK submit_online_quality_job and online_quality_job expose submission/status. This is explicit queued model scoring, not automatic model-rule admission, model-job Studio UI or real-model semantic qualification. Multiple Studio processes sharing a data directory and hardware power-loss durability are not qualified.
Background quality-job full HTTP qualification passes (`/tmp/allpaka-quality-jobs-http.log`, mock provider): explicit queued source scoring invokes the provider once; duplicate submissions reuse the exact job/result; linked judge evidence agrees with retained source and receipt hashes; completed job status survives restart; a stopped-server fixture with a retained claim and no result recovers as interrupted without another provider call. The complete Studio suite passes. This recovery fixture qualifies retained-claim handling, not a live mid-request SIGKILL timing case for this new worker.

Model quality job catalog and Studio viewer: GET /api/observability/online-quality-jobs?project_id=...&offset=...&limit=... and SDK online_quality_jobs browse pending/running/completed/failed/interrupted jobs, source hash, provider/model and verified terminal references. Scan remains bounded to 1000 job directories; project filtering precedes totals and pages (default20, max100), sorted by deterministic job ID. Catalog/detail reads now validate a running claim as well as completed result/linked verdict integrity. Metadata pages omit rubric and submitted text and do not invoke providers. Studio “Модельные оценки” displays state/provider/model and fetches completed verdict score/reason only on explicit click, checking project, source/job receipt linkage and caller-supplied provenance. Interrupted rows explain why automatic replay is absent. Pages and verdict requests ignore replies after dialog/project changes; duplicate verdict clicks are excluded. One focused native job/catalog test, 113 SDK tests and three online/model UI fixture scripts pass. Viewer browser qualification and automatic model-rule scheduling remain open; job submission still uses API/SDK.
Model catalog full HTTP qualification passes (`/tmp/allpaka-model-catalog-http.log`, mock provider): after restart, completed and interrupted jobs occupy two exact pages, the foreign project is empty, catalog reads report no execution and issue no provider requests, and the complete Studio suite passes. No live cloud requests were made.

Automatic model-rule source admission: immutable model evaluator definitions pin normalized provider/model/settings and bounded rubric, exposed by POST/GET /api/observability/online-model-evaluators[/:hash?project_id=...]. Definitions are SHA-256 addressed (model_quality.<hash>, evaluator_version1), strict/verified on read, bounded to 64 KiB and 1000 files, and do not activate rules or invoke a provider. Saving a rule using this evaluator requires the existing configuration's matching project/version. Rules retain existing deterministic sampling and immutable optimistic binding behavior. Explicit quality-source save now admits selected model rules using the trace's retained selection (or creates a pinned selection if none exists), and queues one model job per source/rule/configuration pin. Job plans retain selection/binding/rule/evaluator hashes; admission and worker execution reverify these pins and exact settings/rubric. Repeated source saves reuse the same job, including completed/interrupted history, and never imply an automatic retry. Metadata job admission/assessment excludes model_quality rules so mixed rule selection does not produce an unavailable metadata evaluator error. No automatic content capture occurs; rule activation alone does not send trace metadata or existing histories to models. Source admission failures are logged while preserving the captured source; repeat explicit save can retry admission. Rule scheduling after source save is not transactionally coupled to source persistence, and startup does not scan previously captured sources to repair missing admission. One immutable-model-config test, nine metadata online tests and 114 SDK tests pass; source-save HTTP model execution assertions have passed within the still-running full suite. Model-rule configuration/source capture composers and real-provider semantic qualification remain unfinished.
Automatic model-rule full HTTP qualification passes (`/tmp/allpaka-model-rule-http.log`, mock provider): save/read config and activate a matching rule without provider calls; completed external trace selection followed by explicit source capture invokes the model once in the background; the job plan pins exact configuration and selection hashes; repeated source save after deactivation retains the same source/job without another call. Completed explicit/automatic jobs and interrupted history survive restart and remain catalog-readable. The full Studio contract suite passes, with no live cloud requests.

Studio model rule composer: the rule dialog now offers metadata execution health or model answer quality. Model mode displays the currently selected provider/model and accepts a bounded rubric; it saves immutable evaluator configuration first, verifies returned configuration/settings, and then saves a rule referencing its exact model_quality hash. Both steps are save-only; activation remains a separate explicit action in the rule list. Model settings are Chat without writes and output limit at most4096; configured chat guardrails and other settings are retained. The UI explains that explicitly captured text selected by active rules will be sent to the configured provider under its pricing/terms. Model-rule rows use a readable label instead of the full hash. Changes to model/settings/project/rubric/kind/dialog during saving exclude stale follow-up writes; duplicate saves are blocked. If only configuration save finishes, that immutable snapshot remains but no rule is activated. Extended rule UI fixtures pass for config→rule save ordering, exact evaluator reference, Chat/write/token settings, duplicate clicks, substituted model and changed rubric; existing metadata queue and model-result UI fixtures pass. Browser model-composer qualification remains open; source-capture UI is still unfinished.

Studio explicit quality-source capture: completed, nonremoved traces without running spans now offer “Передать текст для оценки”. Opening pins/validates the completed trace through the existing selection endpoint without a provider call. The dialog explicitly discloses separate text retention and possible transmission/cost under selected model rules; question/answer/optional reference are manually supplied and bounded by UTF-8 bytes. Save validates exact returned source content/identity/hash and caller-supplied provenance, excludes duplicate saves and stale trace/session/project/dialog/draft responses, and links to the model job viewer. Confirmation says text was saved and rules may create jobs, rather than claiming admission succeeded. Closing clears the text draft. Capture UI fixtures pass for exact trace pin, explicit source, byte limits, duplicate saves, substituted text and stale dialog/session handling. Existing trace evaluation, trash, guardrail, review-queue, conversation, bulk and time-series UI fixtures pass; legacy receipt/trash harness boundaries and assertions were updated for additional trace controls. Studio builds. Real-browser capture/model-composer qualification remains open; native source admission/queue execution was previously HTTP-qualified.

Real-browser model-rule/capture/viewer qualification: isolated Studio was operated through UI with a local streaming mock. A local/mock chat was selected; the model rule composer saved and activated browser-model-quality with a typed rubric and normalized Chat/no-write settings. One primary chat turn created a completed native trace. Its capture dialog explicitly stored question/answer text; the automatic model job completed, and the viewer displayed score0.75/reason. Native storage evidence confirms the captured text, exact source→rule/config/selection job pins and job→judge receipt hashes; exactly two provider requests occurred (one primary, one judge), with the explicitly supplied answer present in the judge request. Evidence is retained at /tmp/allpaka-model-browser-evidence.json and screenshot /tmp/allpaka-model-quality-browser.png. Desktop rendering was inspected. Test browser tab and Studio/mock processes were closed, fixture data retained. This qualifies the UI and local provider contract, not semantic quality of a real model or live cloud integration.
