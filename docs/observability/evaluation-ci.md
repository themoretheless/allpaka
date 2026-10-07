# Native Studio evaluation client and CI gate

The Python client calls the existing Studio dataset runner and comparison gate.
It does not implement a second provider runner, auto-promote settings, or start a
Studio server. Use Python 3.9+ and a running Studio with its providers configured.
A model evaluation can consume tokens; execution happens only when invoked.

Create a dataset/prompt and record the explicit versions in Studio first. A CI
run should use the same Studio data directory that contains its baseline and
snapshots. Never copy provider credentials into a report or command argument.

```sh
python3 scripts/evaluate-studio.py \
  --base-url http://127.0.0.1:8100 \
  --dataset-id DATASET_ID --dataset-version 1 \
  --provider local --model MODEL_ID \
  --prompt-id PROMPT_ID --prompt-version 2 \
  --metric exact_match --min-score exact_match=1 --baseline-id BASELINE_RUN_ID \
  --report artifacts/evaluation-RUN.json \
  --junit artifacts/evaluation-RUN.xml
```

Replace uppercase placeholders with IDs from your Studio. Use unique report paths
for each run. `--prompt-template '{{input}}'` supports existing inline evaluation
instructions; it is mutually exclusive with a saved prompt. `--metric` can be
repeated. Settings always use Chat with writes disabled and no tools.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Complete strict-quality run; all requested score thresholds met; with a baseline, no paired score declined |
| 1 | Requested score threshold missed, paired regression, or no improvement when `--require-improvement` is used |
| 2 | Invalid configuration, unavailable service, failed/incomplete evaluation, wait timeout, comparison rejection or report failure |
| 130 | Interrupted; cancellation of the newly created run was attempted |

`--min-score METRIC=NUMBER` requires a finite mean score in [0,1] for a selected
metric; repeat it for multiple metrics. It works without a baseline. Without
thresholds or a baseline, success means the run completed with valid receipts,
not that its answers reached a quality target.

The default regression policy permits ties. The server's `eligible` field still
requires a strict improvement; permitting a tie in CI never promotes settings.
Add `--require-improvement` for candidate promotion checks. Any single sample /
metric decrease fails both policies, even when the aggregate mean rises.
Comparison rejects different dataset hashes/versions, metrics or projects and
recomputes the persisted scores. Incomplete/truncated/provider-error results do
not pass. The CLI validates baseline project/version/metrics before launching.

`--timeout` bounds waiting after run creation (default 300 seconds). Request
timeouts are at most 10 seconds; `--item-timeout` is the native per-item bound
(default 120), and `--concurrency` is the native bound (default 4). A wait/network
error or interrupt attempts cancellation of only the new run. Failure to cancel
is reported explicitly with its known run ID. A failed creation response can have
an unknown outcome: the client does not retry POST or guess a run ID. Inspect
Studio's run catalog if creation loses its response.

JSON reports include full run evidence: the prompt template, system instruction
for saved prompts, model outputs, usage, pinned dataset and prompt hashes,
threshold results and comparison receipt. The dataset snapshot remains in Studio;
the run report identifies it rather than duplicating every input/reference. Protect these artifacts as evaluation
data. Console/JUnit summaries omit prompt and answer content. Reports use new
files with restrictive permissions; existing report paths are rejected before
launch and are never overwritten. Artifact failure fails CI even if scores pass.
HTTP redirects are rejected; server error bodies are not echoed into logs.

## Python / pytest use

The client has no third-party runtime dependencies. Add `sdk/python` to your
Python import path (for example `PYTHONPATH=sdk/python`).

```python
from allpaka_studio import Studio


def test_answer_quality():
    result = Studio("http://127.0.0.1:8100").evaluate(
        {
            "dataset_id": "DATASET_ID", "dataset_version": 1,
            "settings": {"project_id": "default", "provider": "local",
                         "model": "MODEL_ID", "mode": "chat", "allow_writes": False},
            "prompt_ref": {"id": "PROMPT_ID", "version": 2},
            "metrics": ["exact_match"],
        },
        baseline_id="BASELINE_RUN_ID",
        timeout=300, min_scores={"exact_match": 1},
    )
    assert result["passed"], result["comparison"]
```

`Studio.wait(run_id)` reads an existing run without cancelling it on timeout;
`Studio.cancel(run_id)` is explicit. `Studio.evaluate(...)` owns its new run and
attempts cancellation on wait failure. Persist `result` if you need CI evidence.
This is an evaluation client, not an Opik-compatible SDK or external trace
instrumentation. Framework wrappers and distributed trace ingestion remain open.

Validation: `python3 scripts/test-studio-sdk.py` checks protocol/privacy bounds,
fail-closed responses and cancellation ownership. `scripts/test-studio.py` runs
actual Studio against a local streaming fixture and verifies CLI pass/regression/
tie/strict-improvement exits, JSON/JUnit artifacts, preservation of an existing
report and timeout cancellation. No cloud model is invoked by those tests.

The pytest example is documented integration usage; pytest is not installed in
the verified local environment. The SDK tests use standard-library unittest.

`--metric whitespace_token_f1` measures case-sensitive multiset overlap of Unicode-whitespace-separated tokens: `2 * overlap / (reference_tokens + output_tokens)`. Repeated tokens count, order does not, and punctuation is preserved. Every sample must have a reference containing at least one token. This is a lexical score, not a semantic or LLM judge. It supports the same mean thresholds and conservative paired comparison as other native metrics.

Native contract coverage verifies token F1 0.8 → 1 as an eligible paired improvement and the reverse as a blocked regression. CI with observed 0.8 passes a 0.8 minimum and fails a 0.9 minimum.

`--metric json_equals` compares parsed JSON values against a valid JSON reference. Object key order and formatting are ignored; array order and value types remain significant. Invalid model JSON scores zero; invalid reference JSON rejects the run before provider invocation. Parsing and number equality follow `serde_json::Value` semantics; duplicate object keys are rejected at every nesting level, including equivalent escaped keys.

`GET /api/evaluation/metrics` (Python SDK `Studio.metrics()`) discovers native metric IDs, descriptions, reference requirements, score ranges, direction and whether scoring invokes a provider. The current catalog contains six deterministic metrics and permits up to six unique metrics per run.

`POST /api/evaluation/score` scores supplied outputs without provider invocation. Pass `dataset_id`, explicit positive `dataset_version`, selected `metrics`, and `outputs` mapping every sample ID to its answer. Up to 200 samples and 64000 bytes per output are supported, subject to the normal HTTP request bound. Unknown/missing sample IDs reject the request. Returns an immutable receipt with dataset hash, per-sample scores and means; stored under `evaluation/offline_scores`. Python SDK: `Studio.score_outputs(dataset_id, dataset_version, outputs, metrics)`. Offline receipts are distinct from provider experiment runs; use the offline comparison endpoint described below.

`GET /api/evaluation/score/:id` (SDK `scored_outputs(receipt_id)`) reopens an offline receipt with a 16 MiB read bound. It verifies the pinned dataset hash/project and exact sample coverage, then recomputes per-row and mean scores. Missing, malformed or modified score receipts fail explicitly. This consistency verification is not a cryptographic signature of supplied output authenticity.

`POST /api/evaluation/score/compare` accepts `baseline_id` and `candidate_id` for offline receipts (SDK `compare_scored_outputs`). Both receipts are independently verified against their pinned dataset before comparison. Projects, dataset snapshots and metric lists must match. Any per-sample metric decrease blocks eligibility; ties alone are not improvements. The immutable result is stored under `evaluation/offline_comparisons` and remains observational, without automatic promotion or provider calls.

Python `evaluate_outputs(dataset_id, dataset_version, outputs, metrics, min_scores=..., baseline_id=..., require_improvement=False)` saves and verifies an offline receipt and returns `passed`, receipt, thresholds and optional comparison. Default paired policy allows ties but blocks declines; strict mode requires a baseline and genuine paired improvement. Invalid thresholds fail before any write. Errors after receipt creation preserve its ID through `EvaluationError.run_id`. This SDK gate does not automatically promote outputs.

CLI offline mode: `python3 scripts/evaluate-studio.py --base-url http://127.0.0.1:8100 --dataset-id ID --dataset-version 1 --outputs-file answers.json --metric whitespace_token_f1 --min-score whitespace_token_f1=0.8`. The file maps each sample ID to a string answer. Provider/model/prompt options are excluded. Duplicate keys, invalid value types and input over 16 MiB reject before scoring. Baseline, strict improvement, exclusive JSON/JUnit artifact paths and exit codes 0 (pass), 1 (quality failure), 2 (invalid/error) apply as for provider evaluations. Offline reports contain the verified receipt rather than a provider run.

`POST /api/evaluation/judge` explicitly invokes the selected Chat model (writes disabled) with `settings`, `rubric`, `input`, `output` and optional `reference`. Limits: rubric/input 16 KB each, output/reference 64 KB each, two concurrent judges, 60 seconds and 4096 output tokens. Requires complete strict JSON verdict `{score, reason}`, finite score 0–1 and nonempty reason up to 8 KB. Immutable receipts retain supplied text, settings, verdict, usage and source hash under `evaluation/judges`; native traces contain metadata only. Rubric-based model scores are observational and do not automatically promote anything. Pinned dataset judge plans and batches are available below; combined experiment metrics and versioned judge libraries remain open.

Python `Studio.judge(settings, rubric, input_text, output, reference=None)` explicitly invokes the judge; `Studio.judgment(receipt_id)` reopens the saved result. New receipts include a SHA-256 of the complete receipt payload. Read validation uses a 1 MiB bound, rejects duplicate keys, checks identity/hash/verdict bounds, and makes no provider calls. The hash detects changes but is not a signed proof of model authenticity; receipts from the initial un-hashed prototype are rejected by this read endpoint.

Judge requests can include `dataset_ref: {dataset_id, dataset_version, sample_id}`. Studio loads the exact positive version, verifies project/question/reference, supplies its bounded contexts as untrusted source material, and records the dataset SHA-256 and sample ID in the judgment. SDK `judge_sample(dataset_id, dataset_version, sample_id, settings, rubric, output)` reads that pinned sample and submits the judge request. Later dataset revisions do not change the judged source. Batch orchestration and experiment-row judge integration remain open.

SDK `judge_outputs(dataset_id, dataset_version, outputs, settings, rubric, min_score=None)` sequentially evaluates 1–200 frozen outputs using a pinned dataset. All sample coverage, source/output/context bounds, project/mode and threshold checks precede the first judge invocation. Each judgment is independently persisted and reopened with hash validation. Returned summary includes per-sample receipts, mean score and threshold result. Errors/interruption carry `receipt_ids`, `completed` and `sample_id`; completed calls are never automatically retried. The aggregate is client-side, not a durable native batch job; native orchestration, cancellation/recovery and experiment-row integration remain open.

`POST /api/evaluation/judge-plans` freezes a batch before model execution: settings, rubric, dataset ID/version, and exact output mapping. It prevalidates all 1–200 sources/answers/contexts, project/provider/mode, clamps output tokens to 4096 and saves immutable full sample inputs plus dataset/plan hashes. Total plan bound is 16 MiB. `GET /api/evaluation/judge-plans/:id` verifies identity, plan hash and pinned dataset hash. SDK: `plan_judges(...)` and `judge_plan(id)`. Creation/read never invoke a model. This is preparation for a durable native runner; creating a plan currently does not launch or schedule it.

`POST /api/evaluation/judge-runs` launches a saved `plan_id` in the background (up to two live batches). `GET /api/evaluation/judge-runs/:id` verifies the plan and individual verdict receipts; `/cancel` requests cancellation of the active provider future and stops later samples. SDK: `start_judges`, `judge_run`, `cancel_judges`. Individual results and terminal state are immutable events; restart recovery marks unfinished jobs interrupted and never replays model calls. Partial scores remain available; an overall mean is published only for a fully completed batch. Canceling a local future does not guarantee that the remote provider stops billing. Batch tracing now records an orchestration root and per-sample spans linked to individual judge traces. Usage remains on those model traces to avoid counting it twice. The Studio project catalog now opens saved results and traces and can cancel active batches. Studio can also preview a saved plan by ID and explicitly start its batch. Creating plans from the UI and richer recovery tooling remain open.

SDK `wait_judges(run_id, timeout=300, poll_interval=0.25)` observes an existing batch without canceling it. `evaluate_judge_plan(plan_id, min_score=None, timeout=300, poll_interval=0.25)` starts a new batch, verifies its plan/hash and completed mean score, and applies the threshold. Wait errors/timeouts or keyboard interruption request cancellation only of the batch created by that call; `run_id` and `cancellation_failed` preserve follow-up evidence. Quality failures return `passed=false` while keeping the completed run. No model retry or automatic promotion occurs.

CLI model-judge mode: `python3 scripts/evaluate-studio.py --base-url http://127.0.0.1:8100 --judge-plan-id ID --judge-min-score 0.8 --report judge.json --junit judge.xml`. The plan already pins model, source and rubric, so source/provider/prompt/metric flags cannot be mixed into this mode. Completed threshold failure exits 1; execution failure/timeout exits 2 and keeps run ID/cancellation outcome. JSON report retains the run and plan hash; JUnit reports quality failures separately from errors. Timeout invokes SDK cancellation of that newly created run, without retries.

`GET /api/evaluation/judge-runs?project_id=...&offset=0&limit=20` lists metadata for saved runs, with total count and pagination (limit 1–20, offset up to 1000). It scans at most 1000 directories and verifies selected runs against their pinned plans and verdict receipts before returning scores. Invalid storage causes an explicit error. It does not invoke providers. Python SDK: `judge_runs(project_id, offset=0, limit=20)`.

SDK `plan_experiment_judges(run_id, settings, rubric)` freezes the answers of a
completed native experiment into a judge plan without provider calls. The server
verifies the supplied `experiment_id` against the pinned dataset and original
per-sample outputs, rejecting partial/truncated/duplicate/substituted answers.
The plan retains `experiment_source.id` and a SHA-256 of the experiment snapshot
inside its own integrity hash. The judge model may differ from the generating
model, but must use the same project. Run this plan explicitly with
`start_judges` or `evaluate_judge_plan`; creating it does not start evaluation.

`POST /api/evaluation/judge-runs/compare` takes `baseline_id` and `candidate_id`.
Both batches must be fully completed and use identical pinned dataset, rubric
and all judge settings. The server reopens and verifies the plans and per-sample
verdict receipts. It persists an immutable paired comparison under
`evaluation/judge_comparisons`, including receipt IDs, score differences and
plan hashes. Any per-sample decrease blocks `eligible` even when the mean rises;
a tie is not an improvement. These are observed model scores, not statistical
proof of superiority. Comparison invokes no provider and promotes nothing.
Python SDK: `compare_judges(baseline_id, candidate_id)`.

Judge-plan CI also accepts `--baseline-id JUDGE_RUN_ID` and optional
`--require-improvement`. The baseline must be completed with an identical
pinned dataset, rubric and judge settings; incompatibility is rejected before
starting a new model batch. Default comparison allows ties but blocks any paired
regression. Strict mode also requires at least one per-sample gain. Combine
these with `--judge-min-score` to enforce an absolute threshold as well.
JSON/JUnit retain the comparison and quality result; completed gate failures exit
1. SDK: `evaluate_judge_plan(plan_id, baseline_id=baseline_id,
require_improvement=True, min_score=0.8)`. Provider calls occur only in the new
batch, never in preflight or comparison. These remain observational gates.

Versioned criteria reuse the immutable instruction library. SDK
`save_rubric(project_id, name, rubric, rubric_id=None, base_version=0)` stores
criterion text in the library's `system` field with template `{{input}}`.
Edits use the exact prior version; conflicts are rejected. A judge plan accepts
`rubric_ref: {id, version}` instead of inline `rubric`, and stores the full
SHA-256 verified library snapshot as `rubric_snapshot`. The judge reads the
snapshot's system text as its criterion; its fixed verdict protocol remains
unchanged. The plan never substitutes a later criterion revision. References
must have a positive version and belong to the dataset project. Reopening plans
verifies the retained library revision and full snapshot.

SDK `plan_judges(..., rubric_ref={"id": rubric_id, "version": 1})` and
`plan_experiment_judges(..., rubric_ref=...)` support this route. Paired judge
comparisons also require identical criterion snapshots, so inline and library
criteria or different pinned revisions cannot be silently mixed. This reuses
the existing library/history/export UI; dedicated criterion controls and
versioned custom verdict protocols remain open.

`GET /api/evaluation/judge-presets` (SDK `judge_presets()`) exposes three explicit
version-1 model rubrics: `answer_relevance`, `reference_correctness` and
`source_faithfulness`. Higher scores are better, from 0 to 1. Correctness requires
a nonblank reference for every sample; faithfulness requires nonblank contexts
or a reference; relevance requires a question. Plans reject missing evidence
before provider invocation. Discovery and plan creation invoke no provider.

Use `plan_judges(dataset_id, version, outputs, settings,
judge_preset={"id": "source_faithfulness", "version": 1})`, or the equivalent
`plan_experiment_judges` argument, then explicitly start/evaluate that plan.
The full preset definition is retained inside the plan integrity hash and
verified on read. Presets are exclusive with inline/versioned custom criteria;
paired comparisons require identical preset snapshots. Contexts and references
are delivered as untrusted evidence using the existing bounded judge protocol.
These are model rubric scores, not deterministic truth/hallucination detectors;
the mock tests prove delivery, persistence and preflight, not evaluator quality.
Studio now creates preset/custom judge plans from completed experiment answers and opens their preview for explicit execution. Real-model calibration, saved criterion-version selection in UI and richer RAG metrics remain open.
# Comparing a matrix of models and prompts

`Studio.evaluate_matrix(requests, labels=None, timeout=300, poll_interval=0.25,
min_scores=None, require_improvement=False)` runs 2–16 native experiment requests
sequentially. Requests may use different model settings and inline templates or
pinned prompt references; dataset ID/version, project and metrics must match.
The client freezes nested requests before execution and validates these common
pins, prompt bounds and thresholds before creating the first experiment. Native
validation still checks provider configuration, stored artifacts and samples.

The first variant becomes the paired baseline for every subsequent variant.
Thresholds apply to every variant. A higher mean cannot override any paired
regression. The return value has `kind="client_experiment_matrix"`, `baseline_id`
and labelled `variants`, each retaining the ordinary `evaluate` result including
the persistent native run and comparison receipt. The matrix summary itself is
client-owned; save it as JSON if needed. This API does not select or promote a
winner. A failed quality gate is retained and later variants still run; an
execution error or incomplete run stops the matrix. Exceptions retain
`matrix_results`, `variant_index` and `variant_label`, plus the ordinary failing
run/cancellation evidence. No variant is retried. Timeout is per variant.

The existing CI command accepts `--matrix-file variants.json`. Its strict JSON
object contains `requests` (the native experiment request array) and optional
`labels` (one unique name per request). Duplicate keys, nonfinite values, unknown
top-level fields and files over 1 MiB are rejected. Dataset/model/prompt options
and an external baseline cannot be combined with this mode: the first request
is the baseline. `--min-score METRIC=NUMBER` applies to all variants;
`--require-improvement` applies to candidates. For example:

```sh
python3 scripts/evaluate-studio.py --base-url http://127.0.0.1:8100 \
  --matrix-file variants.json --require-improvement \
  --report matrix-report.json --junit matrix.xml
```

Exit 0 means every variant passed, 1 means a completed quality gate failed,
2 means an execution/request/artifact error, and 130 means interruption.
JSON reports retain full native run/comparison receipts and partial results on
failure. JUnit has one test per completed variant and an additional error test
when execution stops. Report files must be new; existing evidence is preserved.
Model settings inside requests may contain custom system instructions, so the
matrix input belongs with evaluation artifacts rather than trace metadata.

Studio's evaluation dialog includes a matrix-report JSON importer (32 MiB,
1–16 retained variants, including partial error reports). It validates unique
labels/run IDs, reopens each native experiment in the selected project and
requires identical dataset hashes/versions/metrics. Displayed means come from
the native receipts, not the imported report. Completed strict-quality candidates
are compared again through the verified native comparison endpoint; this creates
comparison evidence without invoking a model. The panel shows per-example
changes and opens individual experiments. Imported CI thresholds are not treated
as trusted native gate results. Project changes/reopening the dialog discard
stale import responses. `node scripts/test-studio-matrix-ui.js` exercises actual
handler behavior with API fixtures. Browser verification on 2026-10-07 loaded a two-variant SDK report, showed native metric values and per-example paired changes, and opened the corresponding experiment. The browser console reported no warnings/errors. Fixture models are local mocks; real-model quality remains unqualified.

`character_bigram_f1` compares multisets of adjacent Unicode scalar pairs with
2 × shared pairs / total pairs. Pair multiplicity is bounded by each side's
actual count. Case, whitespace and punctuation are preserved; strings shorter
than two scalars use exact equality. A reference is required. The score is in
[0,1], higher is better, and no scoring provider is called. This measures local
text overlap, not semantic correctness or global sequence equality. Native,
offline, SDK matrix and CI threshold/paired gate paths support this metric.

### Evaluate a local Python task

`Studio.evaluate_task(dataset_id, version, task, metrics, min_scores=...,
baseline_id=..., require_improvement=False)` runs a synchronous callback once
for each input in a pinned dataset (1–200 samples). By default the callback receives only
input text; contexts require explicit selection. It never receives reference
answers or sample metadata, and must return at most 64000 UTF-8 bytes
of UTF-8 text. Dataset/metric/reference/threshold and baseline compatibility
checks precede execution. Outputs use the existing immutable offline receipts,
verified means, thresholds and conservative paired gates. Callback execution
can incur application-defined costs or side effects; there are no retries or
automatic promotion. On callback failure the original exception is re-raised
with `completed_outputs`, `sample_id`, `dataset_id`, and `dataset_version`;
no incomplete score receipt is fabricated. Durable partial task runs remain open.

`await Studio.evaluate_task_async(...)` accepts an awaitable task and shares the
same pinned-dataset/metric/reference/baseline preflight. Samples run with bounded `concurrency` (1–8, default 1) and a configurable `item_timeout` (default 60 seconds, above zero through 600).
Studio HTTP operations run in worker threads, keeping the event loop available.
Timeouts cancel the current awaitable and raise `asyncio.TimeoutError`; caller
cancellation propagates `asyncio.CancelledError`. Both attach partial outputs
and sample/dataset identity, stop further task execution, and do not write a
partial score receipt. Cooperative task cancellation is required: Python cannot
force-stop callbacks that suppress cancellation. Cancelling during the final
HTTP write can leave a committed receipt with a lost response; no automatic
retry or rollback is performed. SDK tests cover timeout cleanup, cancellation,
once-only calls and worker-thread HTTP execution; native contracts cover a
complete async result and rejection of its paired decline.

Both task methods accept explicit `with_contexts=True`. In that mode the callback
receives `(input_text, contexts_tuple)` from the pinned dataset. Context strings
are copied and frozen before any callback executes; reference answers and sample
metadata are never passed. Default callbacks still receive input text only.
The SDK validates all context lists before executing tasks (at most 50 strings,
64 KiB each). This supports supplying RAG sources already stored in the dataset;
it does not execute retrieval or infer contexts from application outputs.

Async workers stop admitting new samples on the first observed failure. Remaining
workers are cancelled and awaited before the exception returns. Failure evidence
adds `started_sample_ids` and all completed outputs in dataset order; no score
receipt is created for an incomplete set. Successful output dictionaries retain
dataset order regardless of completion order. SDK synchronization tests verify
the configured concurrency bound and cleanup of all active workers; a native
contract evaluates four callback outputs with two workers and persists a complete
zero-provider-call score receipt. Callback side effects already performed are
not rolled back by cancellation.

### Python tasks in CI

The existing `scripts/evaluate-studio.py` also accepts `--task-file PATH`
(mutually exclusive with provider prompts, ready outputs, matrices and judge
plans). It executes a trusted local Python file and selects `--task-function`
(default `task`). Top-level module code runs during loading; this is explicit
local code execution, not a sandbox. Dataset ID/version, metrics, thresholds,
paired baseline and report/JUnit options remain the same. `--with-contexts`
passes frozen dataset sources. `--task-async` enables awaitable callbacks,
`--concurrency` (default 4) and `--item-timeout` (default 120 seconds). Sync tasks
have no forced timeout; `--timeout` concerns native provider/judge modes.
Successful tasks retain full immutable offline receipts in JSON; completed
quality failures exit 1, execution errors exit 2 and interruption exits 130.
Failures retain partial outputs and sample identity in the explicitly requested
JSON report and emit a JUnit error. Exception text is not included in the
summary. Existing artifact paths are never overwritten. Callback outputs can
contain application data; save reports to an appropriate evaluation directory.
Native contracts verify sync success, async quality failure, partial callback
failure and JSON/JUnit evidence without scoring-provider calls.

### Trace Python task evaluation

Both callback evaluation methods optionally accept `trace=trace`, an entered
`Studio.trace(...)` context owned by the same client. Each executed sample gets
a child tool span named `sample.<sample_id>` with duration and terminal status.
Input text, references, contexts, outputs and exception messages are excluded
from tracing. Async workers inherit the parent context, so concurrently executing
samples remain siblings. The caller owns trace closure and delivery; optional
idempotency/retry behavior remains the existing trace context behavior. Preflight
rejects a closed/foreign trace or insufficient span capacity before executing
callbacks (200 spans including the root; instrumented nested functions consume
additional capacity). A caught callback error retains its failed sample span;
root status reflects whether the error leaves the trace context. SDK tests verify
metadata privacy and failure statuses; native contracts verify the persisted
four-sample tree. Native automatic score-receipt links remain open.

CI task mode accepts `--trace-correlation TECHNICAL_ID` and optional
`--trace-key TECHNICAL_KEY`. The trace belongs to `--project-id`; callback
preflight rejects a dataset from another project before any callback runs.
JSON/JUnit summaries include `trace_id` and `trace_export_failed`. When scoring
succeeds but trace delivery fails, execution exits 2 and the full scoring result
is preserved inside `evaluation_result` rather than discarded. A new CLI
invocation executes the callback again; an idempotency key does not make task
execution idempotent and must not be reused across different execution snapshots.
Native contracts verify the saved trace ID and completed per-sample spans.

### Browse saved offline scores

`GET /api/evaluation/score?project_id=...&dataset_id=...&offset=0&limit=20`
(Python `Studio.list_scored_outputs`) returns verified summaries of persisted
ready-answer and Python task scores. Dataset filtering is optional; pages contain
1–100 records, offsets 0–1000. Summaries include dataset pins/hash, metrics,
means and sample count, never answer text. Every admitted receipt passes the
existing immutable dataset/coverage/score verification. Invalid receipts are
excluded and counted. IDs sort descending (not a claimed creation timestamp).
Reads are bounded to 1000 receipts/64 MiB from at most 10000 directory entries;
`truncated=true` distinguishes a bounded partial catalog from a complete one.
`total` counts matching verified records within that scan, not a global count
when truncated. No provider is invoked. Studio now provides project-scoped paginated browsing and verified per-item detail views; handler tests cover stale/foreign response protection and text rendering. Browser qualification passes for catalog loading, verified per-item output/metrics and forward/back pagination of 21 synthetic receipts, without console errors. An indexed large-scale catalog remains open.

### Judge saved callback or ready-answer outputs

`Studio.plan_scored_output_judges(receipt_id, settings, rubric=..., rubric_ref=...,
judge_preset=...)` reads a verified offline score receipt and freezes its answers
into the existing explicit native judge plan. Select exactly one criterion source.
`POST /api/evaluation/judge-plans` now optionally accepts `offline_score_id`,
exclusive with `experiment_id`. The server reloads and verifies saved scores,
requires identical dataset/project/version/hash and complete exact output coverage,
then retains `offline_score_source.id` and the full source-receipt SHA-256 inside
the immutable plan hash. It does not rerun a Python task or call a model while
planning. Running the reviewed plan remains a separate explicit action and can
incur judge provider usage. Changing supplied answers rejects before plan commit.
The source hash is consistency evidence, not a digital signature. SDK/native
contracts verify provenance, project mismatch and substituted-answer rejection.
Studio now creates reviewed judge plans from expanded offline scores using the shared experiment criterion form; source/stale-handler tests and browser criterion/save/review verification pass; exact saved output/source pins and absence of judge execution were verified, with no console errors. SDK combined gates are available as described below.

### Combined saved-answer quality gate

`Studio.evaluate_scored_output_judges(...)` combines the verified saved score
receipt with an explicit native model-judge plan/run. `min_scores` and optional
`baseline_score_id`/`require_improvement` gate deterministic metrics;
`min_judge_score` and optional `baseline_judge_id`/`require_judge_improvement`
gate the model criterion. Both must pass. A completed deterministic quality
failure still runs the explicitly requested judge for diagnostic evidence;
request/receipt/preflight errors stop before judging. Callback execution is
never repeated. The returned `client_combined_evaluation` contains both full
results and the frozen native plan, with `automatic_promotion=false`. It is a
client summary, not a new native experiment artifact. Judge execution errors
preserve the original exception and attach `deterministic_result` and
`judge_plan_id`. Native judge timeout/cancellation ownership remains unchanged.
SDK tests cover each gate failing independently, both passing and partial error
evidence; native contracts prove that a passing judge does not override a failed
exact-match threshold on saved callback/ready answers. Combined Studio gates and native combined artifact persistence remain open; the CI mode below uses the SDK summary.

### Combined gate in CI

The existing evaluation script accepts `--scored-judge-id SCORE_ID`,
`--provider`, `--model`, `--judge-preset` and a positive
`--judge-preset-version`. The saved score already pins dataset and deterministic
metrics, so dataset/metric/prompt flags are rejected. `--min-score` and
`--baseline-id`/`--require-improvement` gate deterministic evidence;
`--judge-min-score`, `--judge-baseline-id` and `--require-judge-improvement`
gate model judging. This mode explicitly invokes the chosen judge provider on
saved answers; it never executes the original Python task. Exit 0 requires both
gates; completed quality failures exit 1 and execution/request errors exit 2.
JSON retains both full results; JUnit contains separate deterministic and judge
cases plus an execution error case for incomplete runs. Retained deterministic
results and plan ID survive judge execution errors. Artifact paths are exclusive.
Native contracts verify a passing judge alongside a failed exact-match gate,
exit 1, both JSON components and two JUnit cases with one quality failure.

Combined judge failures also retain a read-only `judge_run_receipt` snapshot when
a native run ID exists. Reading this evidence does not invoke a model. If the
read fails, `judge_evidence_unavailable=true` distinguishes missing evidence from
an empty completed run and the original execution error is preserved. The
snapshot can still be cancelling/running; it is evidence at read time, not proof
of terminal completion. CI JSON retains it alongside the deterministic receipt
and judge plan ID. JUnit emits an execution error, not a fabricated passing judge
case. Native timeout contracts verify retained deterministic/run evidence and
exit 2; SDK tests verify the failure receipt and original exception identity.

Saved combined results can be reviewed without running tasks or models again:

```python
review = studio.review_scored_output_judges(
    score_id, completed_judge_run_id,
    min_scores={"exact_match": 0.9}, min_judge_score=0.8,
)
assert review["passed"]
```

This uses only verified receipt/plan GET endpoints, requires a completed judge run
bound to the same saved answers and pinned dataset, and returns both threshold
checks with `provider_calls: 0`. Changing thresholds does not change the saved
scores. The returned combined summary is client-side; native persistent combined
artifacts and combined review UI remain unfinished. Paired baseline comparisons
remain separate explicit operations.

The CI launcher supports the same GET-only review with separate deterministic
and judge JUnit cases:

```sh
python3 scripts/evaluate-studio.py --base-url http://127.0.0.1:8100 \
  --review-score-id SCORE_ID --review-judge-run-id COMPLETED_JUDGE_RUN_ID \
  --min-score exact_match=0.9 --judge-min-score 0.8 \
  --report review.json --junit review.xml
```

Exit codes are 0 when both gates pass, 1 for completed quality failures and 2
for invalid arguments or unavailable/mismatched evidence. Model, dataset,
preset and baseline flags are rejected in this mode; the receipts pin the
sources. Report files must be new paths.

If task callbacks finish but score delivery or verification fails, both Python
helpers preserve all completed answers on the original exception as
`completed_outputs`, with dataset pins and `evaluation_phase="score_delivery"`.
The async helper also preserves `started_sample_ids`. No task is retried.
The server may already have saved the receipt if its response was lost; these
answers are recovery evidence, not proof that delivery did not occur.

Python task JUnit reports now include one diagnostic case per completed sample
and a separate aggregate `quality_gate` case. Sample scores are evidence; mean
thresholds are applied only to the aggregate case. If score delivery fails,
completed callbacks are marked skipped with `callback_completed_score_unavailable`
rather than claiming successful evaluation. Execution errors identify the failed
sample when known. A trace-export failure preserves the saved score cases and adds
an execution error. JUnit includes IDs and scores, but no answer text or exception
body; the explicitly selected full JSON report retains answer evidence.

### Guarded Python tasks

For `Studio.evaluate_task` / `evaluate_task_async` or CI `--task-file`, export
an explicitly guarded text callback using `guard_task(..., return_receipts=False)`.
The default wrapper result includes receipts and is intentionally not a text
answer; this explicit option returns the original successful answer for scoring.
The same option works for sync/async tasks and optional frozen reference contexts.

A blocked input does not invoke the callback. A blocked output is not scored or
returned. Existing task failure behavior preserves the blocked sample ID and
previous completed outputs (plus started sample IDs for async evaluation), without
submitting an incomplete scoring receipt or retrying execution. The original
`GuardrailBlocked` exception retains the local check receipt.

CI exits 2 with `guardrail_blocked`. Its partial JSON and execution-error JUnit
include only the check stage, block action, pass/block booleans and verified
64-character policy SHA-256; private rule values, input/output, exception text
and arbitrary additional receipt fields are excluded from this guardrail evidence.
Successful prior task outputs remain in the existing JSON partial-output evidence,
so those report files retain the same explicit answer-capture semantics as before.
Guardrail-check evidence is not an aggregate quality pass/fail receipt and does
not promote a task. Seven focused tests and all 49 SDK tests pass, including
sync/async partial-output preservation and CI/JUnit privacy checks.

CI also accepts `--guardrails-file rules.json` with `--task-file`. This applies
an explicit policy to the selected callback without changing its source:

```json
{
  "schema_version": 1,
  "action": "block",
  "input_rules": [{"id": "input-limit", "kind": "max_bytes", "value": 8000}],
  "output_rules": [{"id": "json-answer", "kind": "json_valid", "value": true}]
}
```

```sh
python3 scripts/evaluate-studio.py --base-url http://127.0.0.1:8100 \
  --dataset-id DATASET --dataset-version 1 --metric json_valid \
  --task-file task.py --guardrails-file rules.json \
  --trace-correlation guarded-ci-run --report evidence.json --junit evidence.xml
```

Configuration is limited to 128 KiB, rejects duplicate/unknown fields and
unsupported rules/actions, and is validated before loading the trusted task
source or executing its callback. Only task mode accepts this flag. The trace
option explicitly links guardrail spans under the sample spans. `observe` is
also supported: failed checks do not block task execution or override dataset
quality gates. Configuration is not uploaded or copied into reports; only
outcome/policy fingerprints appear in guardrail evidence. Eight focused tests
and all 49 SDK tests pass. A native local CI subprocess verified blocked second
sample, preserved first output, JSON/JUnit error evidence, persisted hierarchy,
zero model spans and absence of private input text.

Guarded evaluations now preflight known trace span requirements before callback
execution: one sample span plus two checks per guard wrapper attached to the same
active trace. Nested wrappers accumulate their check budget. With only the root
span present, 66 single-wrapper samples fit the 200-span limit; 67 reject before
any callback runs. Both sync and async evaluators enforce this admission check.
User-created spans inside callbacks or concurrent work are not predictable and
still remain subject to the normal runtime span limit. A new subprocess test
also verifies that malformed guardrail configuration rejects before executing
module-level code from the trusted task file. Ten focused guardrail tests and
all 50 SDK tests pass.

### Fingerprint-verified guardrail policy files

Task evaluation accepts a saved input/output policy pair instead of the combined
inline rule configuration:

```sh
python3 scripts/evaluate-studio.py --base-url http://127.0.0.1:18805 \
  --dataset-id DATASET --dataset-version 1 --metric json_valid \
  --task-file task.py --input-policy-file input-policy.json \
  --output-policy-file output-policy.json --policy-action block
```

Create both files with `allpaka_guardrails.save_policy`. All three policy options
are required together; they cannot be mixed with `--guardrails-file`. The action
must explicitly be `block` or `observe`. Both manifests and their fingerprints
are validated before loading Python task code. Existing task guardrail trace,
partial-output and failure evidence behavior applies. Tests verify policy
round-trip through the CI loader, execution and rejection of altered policies.
Native server-side policy storage remains separate unfinished work.

Native server-side policy storage and dataset experiment enforcement have since
been implemented. Supply `settings.guardrails` with
`input_policy_sha256`, `output_policy_sha256` and `action` (`block` or `observe`).
Both saved policies are validated and frozen before starting the experiment.
Each rendered sample prompt is checked before inference; each generated answer
is checked before persistence and scoring. A blocked item has
`error: "guardrail_blocked"`, no output and no scores. Output blocking retains
reported model usage. Observe mode keeps the answer and records violations in
the experiment trace, including per-rule outcomes without text or rule values.
This applies to native dataset experiments; rubric judges and compaction are
separate execution paths and are not qualified by this change.

Subprocess qualification of the policy-file CLI confirms that tampering with
either input or output policy stops with exit 2 and an
`invalid_request_or_receipt` report before task module import. A task whose
module would create a marker file never executes. Missing policy action,
action without files and mixing manifests with the combined configuration are
rejected by argument validation. Thirteen guardrail tests pass; these checks
use an unreachable local endpoint and require no inference.

To require literal answer fragments, use `required_substrings`. Every supplied
fragment must occur, with exact case; this is a deterministic text check and
makes no semantic claim. For example:

```python
from allpaka_guardrails import check_guardrails
rules = [{"id": "answer_sections", "kind": "required_substrings",
          "value": ["Result:", "Evidence:"]}]
receipt = check_guardrails("Result: done\nEvidence: checks passed", rules,
                           stage="output", action="block")
assert receipt["passed"]
```

The same rule can be saved with `Studio.save_guardrail_policy(rules)` and selected
as the chat or native experiment output policy. Rule lists contain 1–100
nonempty fragments of at most 1000 UTF-8 bytes each. Receipts expose rule IDs,
kinds and outcomes without copying the required literals or checked text.

Matrix CI persistence: add `--persist-matrix` together with `--matrix-file` to save the completed native matrix in Studio. JSON stdout includes `native_matrix_id`; the full report retains `native_matrix` evidence. CI exit codes still use each client's configured thresholds/strict-improvement/paired gates, so saving a failing matrix never makes the job succeed. Using the persistence flag without a matrix file rejects before any model request.

Server-owned matrices can also be launched independently of a long-lived Python
client. `Studio.start_matrix_job(project_id, variants)` accepts 2–16 dictionaries
with `label`, pinned dataset `sha256`, and a native experiment `request` (including
the same project/dataset version/metrics and read-only Chat settings). Poll
`matrix_job(id)` or list `matrix_jobs(project_id)`; a completed job returns
`matrix_id`, which `experiment_matrix(matrix_id)` revalidates. Disconnecting the
client leaves the native worker running. `cancel_matrix_job(id)` stops admission
and cancels only its current experiment. After a Studio restart, jobs become
`interrupted`; `resume_matrix_job(id)` explicitly continues verified completed
runs and provably unsent reservations. Changed provider routing or a missing
already-attempted run receipt blocks continuation. It rejects interrupted inference and never silently
replays provider calls. A failed final save can resume with the original IDs and
no additional model calls. These jobs use the native no-paired-regression matrix
gate; client-specific thresholds still require separate checking.


`Studio.retry_matrix_job(id)` explicitly forks an interrupted/failed/cancelled
sequence into a new job. This is different from resume: it authorizes new model
calls for failed/interrupted/missing/unsent variants while reusing verified
completed strict-quality runs. The original job remains intact; the returned
job includes `retry_of` and the original `previous_run_ids`. Active or completed
source jobs reject. Provider routing and original available run configurations
must still match the pinned sequence. The resulting matrix remains subject to
the native no-paired-regression gate and never promotes settings automatically.
