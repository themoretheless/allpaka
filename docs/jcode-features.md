# Jcode capability integration

Scope: implement the complete capability inventory requested in this chat, using
Allpaka's existing Studio and native Rust runtime. This file tracks unfinished
work; passing a subset of tests does not prove overall feature parity.

Upstream inspected: 1jehuang/jcode, a0c41dc2f30a936b7944e282fbd2b11d81683dfb.
Integration checkout: allpaka-jcode, based on Allpaka 7b29a54. The original
allpaka directory lacked sources and a usable Git repository. Any newer local
work must be reconciled if its location becomes available.

| Capability group | Current evidence / remaining work |
| --- | --- |
| Memory | Added explicit project/global immutable notes, SHA-256 integrity, revision conflicts, removal/restore revisions, expiry exclusion and native lexical memory_recall tool. Recall groups exact-content duplicates with bounded pinned source receipts, without modifying stored notes. Added Studio editor with project/global browsing, text filtering, expiry calendar, soft removal/restore, read-only historical revisions and JSON export. Added explicit bounded model extraction into persisted proposals with source hash and traces; review/edit/accept UI and Python SDK preserve verified proposal provenance. Automatic extraction remains open. Semantic recall, consolidation, contradiction handling and automatic staleness handling still needed. Upstream's current recall uses Jev Decisions; README embeddings description differs. |
| History/context | Existing Studio history and compaction. Added current-transcript search over original messages, with Unicode-safe excerpts and compacted markers. Added named message bookmarks, history filter/label search, navigation, branch-prefix copying and validated inert import/export. External harness import still needed. |
| Swarm | Existing read-only member debate/synthesis. Added source search for members. Messaging, edit notifications, dependency graph, worker pool, task artifacts and completion gates still needed. |
| Long-running work | Added session-owned background commands, bounded output, event-driven multi-task waiting, cancellation/timeouts and Studio controls. Added durable milestone IDs, revision checks, criteria/reported evidence and bounded checkpoints with explicit restart continuation. Added bounded in-turn continuation and a reported-plan completion gate. Independent goal verification, cross-turn automation and overnight reports remain open. |
| Interaction | Existing actor has send/send_now/steer and queue actions; semantics need runtime audit. Added bounded identical-failure handling: three consecutive calls with the same canonical arguments/error pause the turn, cancel the remaining batch calls with protocol receipts and mark the root trace no_progress; successful tools, changed failures and steering reset the counter. Resume retries in a fresh turn. General unfinished-task continuation and broader progress detection remain open. |
| Code tools | Added literal agentgrep with preceding declaration hints and bounded scanning; hidden paths, symlinks, binary files and conversation storage excluded. Added shared per-path mutation locks with a 32-thread regression test. AST context, adaptive repeat suppression, batch execution and remote compilation still needed. |
| UI | Auxiliary panels, Markdown/PDF, Mermaid, widgets and interactive applets still needed. |
| Extensions | Lifecycle hooks, input transformation/policy hooks, MCP discovery/management, skills and instruction overlays require implementation or audit. |
| Providers | Existing adapters and custom providers. Added explicit catalog-only provider doctor API/SDK: validates endpoint, credential availability, bounded strict catalog, HTTP/auth/rate-limit/redirect/network diagnoses and selected-model presence, with a 10-second overall deadline and no inference/body disclosure. Added Studio connection diagnostics with provider/model selection, localized repair hints and stale-response guards; browser verified catalog readiness and missing-model paths. Added native project/session/time-filtered trace usage summary with model-leaf-only token subtotals, explicit unknown counts and reported costs separated by declared currency; unlabelled costs are not added to monetary totals. Added Studio summary panel with date/scope filters, unknown counters, linked trace details and JSON export; browser verified filtering and export on synthetic data. OAuth coverage, live model reload, durable coverage/spend ledger and cache diagnostics require implementation or audit. |
| Remote/resilience | Existing shared Studio service. Added exclusive history-directory ownership and restart recovery for traces/experiments without provider replay. Reconnect/reload persistence, shared MCP lifecycle, safe updating and native SSH require implementation or audit. Live cross-host migration is an upstream proposal, not an implemented source feature. |
| Automation/integrations | Browser/computer control, mail/calendar, scheduled cycles, ambient work, SDK and harness API still needed. |
| Self-development/reports | Build/reload workflow, desktop support and productivity reports still needed. |

Verification so far: `cargo test -p allpaka-chat` passes 79 tests with one native
credential-store test ignored. New searches are registered in the agent execution
path; source search, compacted transcript retrieval and background command tool-call roundtrips are verified through the local mock-provider Studio contract harness. Live cloud providers are not verified.

Additional required source: [Opik capabilities](observability/opik-parity.md).
This inventory remains part of the same objective, not a separate optional task.

Imported conversations now also clear saved guardrail selection and internal
prepared policies. Import remains inert: original messages/bookmarks are copied,
but enforcement must be selected explicitly for the new conversation. A native
HTTP scenario injects an enabled policy into the exported settings and verifies
that the imported copy has no policy and makes no provider call. Full regression
qualification of this change is running.

The full mock-provider HTTP regression passed, including import and restart
(`/tmp/allpaka-import-policy-http.log`); full chat-crate Rust tests also passed
(`/tmp/allpaka-import-policy-rust.log`).

Background task snapshots and catalogs now expose wall-clock `started_ms` and
`finished_ms`, plus monotonic `duration_ms`. Finish/duration remain null while
running; terminal duration includes process-group shutdown and output draining.
Studio displays reported terminal duration beside the task status. Six native
background tests pass, including running unknown timing, terminal measurement
and catalog agreement (`/tmp/allpaka-background-timing.log`). JavaScript syntax
and diff whitespace checks pass. This does not make tasks persistent: restart
receipts, durable goals and automatic wakeup remain unfinished.

Background commands now save bounded versioned start/terminal receipts under
`<studio-data>/background-tasks`. Completed stdout/stderr, exit code and timing
restore in the owning conversation. Start-only receipts recover as interrupted
with unknown exit/duration; commands are not replayed or reattached. Command
text is not persisted, but explicitly captured output is. Receipt commits are
exclusive/atomic, with Unix 0600 files. Cleanup removes retained terminal
receipts as well as registry entries. The native persistence regression verifies
finished output, orphan recovery, ownership, cleanup and private file mode.
Automatic agent wakeup and durable milestone goals remain unfinished.

Seven focused background tests pass (`/tmp/allpaka-bg-persist-focused.log`),
including Unix private file mode. Full chat-crate Rust tests and the full native
mock-provider HTTP regression pass (`/tmp/allpaka-bg-persist-rust.log`,
`/tmp/allpaka-bg-persist-http.log`). Dedicated server-kill/process-ownership and
browser recovery scenarios still need qualification.

A dedicated native HTTP crash scenario now starts a test-only command that
writes a launch marker and records its process-group ID. It kills Studio,
restarts against the same data, checks interrupted/unknown timing, unchanged
single-launch marker and retained completed output, then explicitly kills the
fixture group and cleans task receipts. The graceful restart path also checks
completed output. This qualifies receipt recovery rather than reattachment or
continued supervision of an orphan process; the full run is active.

The first server-kill scenario exposed a cleanup bug: durable files disappeared,
but registry retention recomputed terminal state after their deletion. Cleanup
now freezes the admitted terminal IDs before deleting files and removes exactly
those IDs from memory, avoiding both stale registry entries and races with newly
finishing tasks. The persistence unit test checks immediate empty registry as
well as empty state after reopening. The corrected full crash run is active.

The corrected full HTTP crash/restart regression passed
(`/tmp/allpaka-bg-crash-http.log`), including completed-output restoration,
interrupted unknown timing, single-launch marker and immediate receipt cleanup.
Seven focused native tests pass (`/tmp/allpaka-bg-cleanup-focused.log`). Browser
recovery and automatic agent wakeup remain open.

Background recovery additionally validates receipt consistency: nonzero start
metadata; start records contain no invented output/exit/finish timing; terminal
records require timing and consistent success/exit status; start/terminal pairs
must agree on owner and start time. Corrupted owner, start time, missing duration
and false success are rejected. This is structural validation, not cryptographic
authentication. Seven focused background tests pass, covering both valid restart
and malformed receipt rejection (`/tmp/allpaka-bg-receipt-validation.log`).

The Python SDK now manages existing Studio background tasks through
`start_background`, `background_tasks`, `background_output`, `wait_background`,
`cancel_background` and explicit `cleanup_background`. Command/wait bounds are
validated before requests, duplicate task IDs reject, and the network timeout
accommodates the 60-second server wait. Fifty-five SDK tests pass
(`/tmp/allpaka-bg-sdk.log`). Native HTTP qualification exercises SDK wait,
output and persistent catalog; full regression is running. Usage is documented
in `docs/studio.md`.

The full SDK/native HTTP regression subsequently passed
(`/tmp/allpaka-bg-sdk-http.log`), including background wait/output/catalog and
server crash recovery. This does not implement automatic wakeup or durable goals.

Native HTTP qualification now also exercises SDK start, actual cancellation,
terminal wait and cleanup, plus SDK rejection for foreign task access and Chat
command execution. Tests assert the standard redacted `EvaluationError` codes.
The full run passes (`/tmp/allpaka-bg-sdk-lifecycle-http.log`), including server
crash recovery and persisted output. No automatic continuation is claimed.

Studio background handler qualification now verifies terminal duration display,
restart interruption with unknown duration, no cancellation button on recovered
terminal tasks, literal stdout rendering, explicit cancel/cleanup routing and
hidden empty catalog. `node scripts/test-studio-background-ui.js` passes. This
is handler-level evidence; real-browser recovery qualification remains open.

Explicit background continuation is now implemented with optional `follow_up`
on native tool/API and SDK starts. Terminal completion sends one internal actor
event, queues task ID/status and the supplied bounded instruction under current
session settings, and does not include command output. Paused/stopped chats,
non-active folders and modes outside Auto/Goal suppress continuation. Internal
events cannot be submitted as public action kinds. Follow-ups are not replayed
on restart. Full Rust tests, 55 SDK tests and native HTTP regression pass
(`/tmp/allpaka-bg-wake-rust.log`, `/tmp/allpaka-bg-wake-sdk.log`,
`/tmp/allpaka-bg-wake-http.log`). Native scenarios verify one successful Auto
continuation and no inference/messages after Stop. Browser controls, restart
scheduling and durable milestone goals remain unfinished.

Background continuation now captures a session-local authorization generation.
Stop/Send-now cancellation changes that generation; later completion from an old
command cannot enqueue its follow-up after the user starts another turn. The
internal generation is not exported/imported or accepted in public actions.
Full Rust and first full HTTP regressions pass
(`/tmp/allpaka-bg-wake-epoch-rust.log`, `/tmp/allpaka-bg-wake-epoch-http.log`).
A stronger scenario holds command completion behind a test file until after
Stop and a new task; its full HTTP rerun is active.

The deterministic full rerun passed
(`/tmp/allpaka-bg-wake-epoch-deterministic-http.log`): completion was released
only after Stop and a new user turn, and no stale instruction entered messages
or queue. This does not add cross-restart follow-up replay.

Studio now has an explicit background launch form in the existing context panel:
command, bounded seconds timeout and optional follow-up. It appears for active
Auto/Goal conversations, clears drafts when switching chats, validates UTF-8
limits, prevents duplicate submits and discards stale replies. No command starts
without the user's launch button. Panel handler and launch-handler tests pass
(`node scripts/test-studio-background-ui.js`,
`node scripts/test-studio-background-launch-ui.js`); full browser qualification
of command launch/recovery remains open.

Real-browser launch qualification passed using native accessibility controls:
opened the existing Auto fixture, entered `printf ui-background-proof` with a
10-second limit, explicitly launched it, observed completed status/duration and
opened the exact stdout. Screenshot:
`/tmp/allpaka-background-launch-ui-20261007.png`. Browser page-reload recovery
remains unqualified: after reload the automation click helper could not resolve
its target; this is not evidence of an application recovery failure.

History refresh qualification (2026-10-07): unchanged automatic session-list responses retain existing DOM elements, preserving focus and selection. The cache includes filters, pagination request, selected session and complete returned rows; append and failed requests invalidate it. `scripts/test-studio-history-refresh-ui.js` verifies element identity, changed status/selection/filter updates, pagination, empty results and error recovery. Browser reload recovery remains unqualified.

Background output continuity (2026-10-07): explicitly opened captured output survives task-catalog updates, including another command finishing. The UI keeps output only for currently listed task IDs in the same conversation, clears it on conversation changes/removal, and discards late responses for removed tasks. It remains an explicit snapshot rather than automatic live output polling. `scripts/test-studio-background-ui.js` verifies persistence, conversation isolation and late-response rejection in addition to existing lifecycle controls.

Named background tasks (2026-10-07): native background start accepts an optional human-readable `name` (nonblank, at most 200 UTF-8 bytes), persisted in both start/completion receipts and exposed in catalog/status/output. Receipt pair names must agree; old receipts lacking the field load as unnamed. Python start and Studio launch form expose the label; UI displays it as literal text beside the technical ID and preserves name draft changes during pending launches. Rust qualification covers durable Unicode labels, oversized receipt rejection and legacy receipt loading; SDK/UI tests cover bounds, launch transport and literal labels. Native HTTP fixtures verify tool-start label persistence through SIGKILL recovery.

Selective background cleanup (2026-10-07): native `background` cleanup accepts optional `task_id` to remove only that conversation-owned terminal receipt; running, missing and foreign tasks are rejected. Omitted ID retains bulk terminal cleanup. Python `cleanup_background(session_id, task_id)` and each terminal Studio task row expose the selection. Rust qualification covers preservation of another retained receipt after reopening, foreign ownership and running rejection; SDK/UI tests cover selective payloads. Output caches are cleared for removed rows.

Background lifecycle diagnostics (2026-10-07): task cards display known integer exit codes (including zero), known valid start timestamps in the viewer's locale, truncated-output flags and explicit terminal receipt storage failure. Failed tasks with no code say the code is unknown; interrupted tasks do not invent an exit code or timestamp. Handler tests cover success/nonzero/unknown codes, missing start time, storage failure and truncation. CLI build and existing launch controls pass; browser qualification remains open.

Background history export (2026-10-07): native background `export` creates schema-1 session-owned lifecycle JSON, with names/status/timing/exit code/error presence/output byte counts/truncation. Captured stdout/stderr require boolean `include_outputs:true`; command text, follow-up instructions and raw error text are omitted. Export is bounded to 8 MiB and records `automatic_replay:false`; it is an observation snapshot, not a runnable import. Python `export_background` and separate Studio metadata/output export buttons expose the choice. Rust tests cover ownership, default output omission, explicit output and malformed opt-in rejection; SDK/UI tests cover routes/choice and downloadable JSON. Live browser download remains unqualified.

Background output request ordering (2026-10-07): explicit output reloads use per-task request tokens. Late earlier responses cannot overwrite a newer captured output, and obsolete response errors are ignored after conversation changes, removal or superseding reads. Reload labels distinguish opened output from first-time reads. Handler regression reverses two completion responses and verifies the newer output survives subsequent task updates. CLI build and existing lifecycle tests pass.

Background activity totals (2026-10-07): native session-owned task catalog includes summary status counts, saturating sum of known elapsed durations, explicit unknown-duration count and `duration_semantics:sum_task_elapsed`. Running/interrupted receipts without terminal timing remain unknown. Studio displays task/running counts and known-duration totals with a warning that parallel task durations add together. These are retained-task totals, not project productivity, wall-clock activity or model-cost reports. Rust tests cover running unknowns, terminal timing and owner isolation; UI tests verify duration/count semantics; HTTP fixture checks exact catalog timing against the completed receipt.

Background export response checks (2026-10-07): Studio verifies bounded task arrays and rejects command/follow-up/raw-error fields, plus stdout/stderr in metadata-only exports, before constructing a download. Switching conversations discards late export responses and errors. Handler regression confirms unexpected private output never creates a Blob/download and a late old-chat export is discarded. CLI build and background controls pass.

Unix background termination signals (2026-10-07): native lifecycle receipts persist optional signal number separately from integer exit code, and expose it in catalog/output/exports. Unix child exit status supplies the signal; unknown values remain null on other platforms/legacy receipts. Reader validation rejects invalid signals, exit-code/signal conflicts and signals in start receipts. Studio displays known signals instead of an invented code. Rust qualification launches self-TERM (15), verifies failed/unknown code, durable exact reopen and legacy-field compatibility; UI test verifies signal rendering. HTTP qualification exercises the same Unix termination through SDK wait and selective cleanup. Windows signal behavior remains unqualified.


Durable milestone foundation (2026-10-07): the existing session plan now retains stable IDs, acceptance criteria, reported completion evidence, optimistic revisions and bounded persisted checkpoints. Goal updates cannot silently remove or alter completed milestones; a human may explicitly reopen them. Legacy plans retain their marked progress and can acquire criteria incrementally. Every Goal model step receives the current revision and IDs, including explicit continuation after restart. Checkpoints are synced with the session before acknowledging changes; startup pauses interrupted work without automatic model replay. API and Python SDK expose reading and revision-checked updates. Studio provides staged milestone creation, criteria/evidence editing, explicit reopening and checkpoint history. Evidence remains a reported claim, not independently certified. Full goal objective/status management, independent completion verification and automatic unfinished-task continuation remain open. Browser qualification of the new plan controls remains open.

Qualification: 109 Rust unit tests and two integration tests pass (one credential test ignored), 67 SDK tests pass, focused plan UI tests and CLI build pass. The full native/mock HTTP suite passes, including actual SIGKILL recovery: the saved plan/checkpoints survive, startup makes no model calls, and explicit continuation receives unchanged completed milestone IDs plus the current revision. No live cloud requests or browser verification of the new plan controls were performed.


Reported Goal completion gate (2026-10-07): a text-only model finish cannot close an empty/incomplete plan or a plan lacking criteria/reported evidence. It receives a continuation within the existing step budget. Another unsupported finish at the same plan revision pauses with `goal_incomplete`, as does exhausting steps while the plan remains incomplete. No new runner, unbounded retry or restart replay is introduced. `reported_completion_ready` exposes the exact reported-data predicate. This does not prove fulfillment of the original objective, associate a new user objective with a new plan, or independently execute/verify acceptance criteria; those requirements remain open.

Gate qualification: 110 Rust unit tests + two integration tests pass (one credential test ignored), CLI build and focused plan UI checks pass. Full native/mock HTTP checks pass: incomplete-plan finish automatically continues to a complete reported plan in four calls without changing the completed milestone; unsupported completion with no plan pauses after two calls; repaired tools resume but cannot close an empty Goal plan; actual SIGKILL recovery and matrix retry remain verified without restart inference replay. No live cloud or new browser flow qualification.


Goal objective binding (2026-10-07): new Send/SendNow work in Goal mode receives a distinct persisted goal ID and original user-message index. Prior nonempty plans remain in bounded checkpoints with their goal ID, while the current plan clears before execution. Resume, Steer and background continuation retain the goal. Thus an already completed prior plan cannot make a new objective immediately ready. Legacy chats remain compatible. Studio distinguishes checkpoints from previous/current goals. This is objective provenance and plan isolation; independent criterion verification remains unfinished.

Objective-binding qualification: 111 Rust unit + two integration tests pass (one credential test ignored), CLI build, focused plan UI and full native/mock HTTP suite pass. The native fixture finishes one reported plan, sends a distinct new objective in the same session, confirms an empty not-ready current plan and retained old-goal checkpoints, then explicitly resumes without replacing the new goal ID. Actual SIGKILL restart retains the goal metadata and plan packet without inference replay. No live cloud or browser qualification of the new goal checkpoint labels.


Goal provenance integrity: restored nonempty plans require the last checkpoint to carry the current goal ID; malformed checkpoint goal IDs and invalid original-message references reject. Legacy goal-less plans remain compatible. Every Goal model call includes the persisted origin and original user objective as well as the fresh plan revision and milestones, preserving the objective through history compaction. Unit fixtures reject cross-goal substitution and invalid origins.

Provenance integrity qualification: 112 Rust unit + two integration tests pass (one credential test ignored), CLI build, focused plan UI and full native/mock HTTP contract suite pass, including distinct goals, explicit resume, completed checkpoint preservation and actual SIGKILL restart. No live cloud qualification.


Historical Goal objectives: new checkpoints retain `goal_origin` (goal ID and original user-message index), while `goal_id` remains compatible with existing clients/checkpoints. Studio displays a bounded original-objective excerpt beside each checkpoint. Restoration rejects invalid message references, goal-ID mismatch and inconsistent message origins for the same goal, including a conflict with the current goal. Legacy checkpoints lacking the origin remain readable without inventing historical provenance.

Historical objective qualification: 112 Rust unit + two integration tests pass (one credential test ignored), CLI build and plan UI tests pass. Full native/mock HTTP suite verifies exact old/new checkpoint origins, explicit continuation and packet persistence through SIGKILL. UI helper tests verify rendering the original objective excerpt; browser clicks for this new presentation remain unqualified. No live cloud requests.

Declared memory expiry review (2026-10-07): native `/api/memory/expiry` and Python SDK `memory_expiry` report expired and expiring notes with pinned latest ID/version/SHA-256, metadata only. Horizon is 0..365 days (default30), global scope is explicit, removed notes are counted but omitted from review rows. Exact deadline is expired; exact horizon is expiring. Notes without deadlines are counted separately and are not inferred stale by age. This is a read-only review foundation, not semantic freshness detection, automatic renewal or removal. UI and HTTP qualification remain open; native fixtures cover boundaries, foreign-project isolation, global deduplication and unchanged source versions.

Studio declared-expiry review now shows state counts, selected-scope expired/expiring metadata and readonly pinned historical content with ID/project/version/hash/expiry receipt checks. Input/scope changes and panel closing invalidate old replies. Local UI fixtures cover captured snapshot, cross-project substitution, invalid horizon, duplicate read exclusion and stale input/errors. Browser and real HTTP qualification remain open.

HTTP/SDK declared-expiry fixtures now verify exact project state counts, explicit global scope without double counting, zero-day horizon, missing project/unknown parameter/oversize horizon rejection, metadata SHA pins and unchanged note snapshots/provider call count. Full native suite passes:130 unit tests,2 integration tests,one credential test ignored. The restart fixture additionally verifies the original expiry note revision/hash remains readable and visible in the expiry report.

Real browser expiry qualification (tab45, isolated local Studio) verified project expired/expiring counts, readonly saved content, shared scope replacement and correct datetime editor population. Browser review exposed a duplicate `memory-expiry` ID between the new container and existing datetime input; fixed to `memory-expiry-review` and added a regression check. Rebuilt/restarted with the same fixtures, verified the editor is an INPUT datetime-local with saved expiry, and unchanged native note revisions. Browser warning/error logs empty, zero provider calls. Evidence `/tmp/allpaka-memory-expiry-browser-evidence.json`, screenshot `/tmp/allpaka-memory-expiry-browser.png`.


Explicit reviewed memory consolidation: the existing memory save API and SDK `consolidate_memory(project_id, sources, name=..., content=...)` can create a new note carrying 2..20 source ID/version/SHA-256 pins. Native admission holds the memory write lock while verifying same-project, nonremoved, hash-matching, latest source revisions and unique IDs. It saves only the new note, preserving all source versions. Source provenance participates in the snapshot digest and cannot change on later revisions; legacy snapshots without it retain their original digest. Content is explicitly reviewed/caller-authored; semantic synthesis, automatic consolidation, contradiction resolution, source expiry interpretation and consolidation UI remain open. Focused native tests and SDK request/preflight tests pass; HTTP qualification is pending.

Reviewed memory consolidation HTTP qualification passed: SDK saves exact source pins; editing the consolidated note preserves origin; revising a source rejects reuse of stale pins; attempts to replace origin fail. Original source version 1 and consolidated versions 1/2 remain identical after Studio restart. Provider request count is unchanged. The full Studio contract suite passed against the local mock provider. Expanded native tests cover duplicate source IDs, substituted hashes and origin replacement; all five memory tests and 97 SDK tests pass. UI and semantic automatic consolidation remain open.

Studio memory editor now displays immutable consolidation source pins (ID, version, SHA-256). Each source opens as a read-only preview of that exact saved revision, with project/identity/version/hash verified before rendering; the edited note stays selected. Malformed or duplicate pins are rejected. Source reads are excluded after editor selection, scope/project changes or dialog close. Local UI fixtures qualify exact historical reads, hash substitution, duplicate pins and stale editor/dialog responses; browser qualification and source-selection/consolidation authoring UI remain pending.

Studio now includes a reviewed consolidation composer: select 2..20 active notes in the current memory scope, prepare an editable combined draft, and save through the existing memory API with exact ID/version/hash pins. Foreign and removed sources are excluded; scope is locked for the prepared draft. If combined text exceeds 16 KiB the content field stays empty for explicit authoring instead of truncating. New/source selection clears pending provenance; native admission remains authoritative for changed source versions. Local UI fixtures qualify selection counts, scope exclusion, version pins, editable draft and stale project/dialog guards. End-to-end browser authoring and duplicate-save/stale-save-response qualification remain pending; semantic synthesis remains open.

Memory editor save admission now permits only one mutation at a time, including new consolidated notes, preventing duplicate creates from repeated clicks. Editor selection, text/expiry edits, scope/project changes and dialog close invalidate pending save presentation. The saved server record is retained even if the UI changed; stale success/error responses do not replace the new draft, and conflicts are not retried automatically. Local UI fixtures qualify duplicate admission, pinned consolidation payload, stale draft/close exclusion and visible current conflicts. Browser authoring qualification remains pending.

Browser consolidation authoring passed: selected two source notes, prepared draft, edited title/content, saved exactly one new version-1 note, expanded immutable origin and read original source text. Native reads verify exact source pins and unchanged original snapshots. No model calls or browser warnings/errors occurred. Double-click/save-race cases remain fixture-qualified rather than browser-qualified; semantic synthesis is still open.


Native consolidated memory source status: GET `/api/memory/notes/:id/versions/:version/source-status` and SDK `memory_consolidation_source_status(note_id, version)` compare immutable source pins against verified latest revisions under the memory write lock. Metadata-only rows retain original version/hash and latest version/hash with current/changed/removed status. Historical source snapshots remain intact and are reverified. This reports revision status, not semantic truth or freshness, and never modifies notes or calls models. Ordinary notes return an empty source list. Native fixtures cover current, changed and removed source revisions; SDK tests cover exact route and preflight. HTTP and UI qualification remain pending.

Studio consolidation provenance now includes “Проверить изменения источников”. It reads the exact selected note revision, verifies its identity/hash/project and every pinned source against the status receipt, then displays unchanged/changed/removed with original and latest revision numbers. Duplicate clicks cannot issue parallel status reads. Incorrect receipts fail; changed editor/scope/project or closed dialog excludes stale replies. Wording distinguishes revision checks from semantic freshness. Local UI fixtures cover changed/removed rows and substituted note hash; native HTTP and browser qualification are pending.

Consolidation source-status HTTP qualification passed: current/current changes to changed/current after a source revision, then changed/removed after soft removal. Metadata retains original pin hashes and identifies latest versions/hashes without note content or model calls. Ordinary notes return no sources. The exact status receipt survives restart. Full Studio contract tests passed with the local mock provider; full Rust suite passes 134 unit and two integration tests (one credential-dependent ignored), plus 98 SDK tests. Browser status-button qualification remains pending.

Browser source-status qualification passed: selected a consolidated note and clicked its status check; Studio shows original v1 to latest v2 for both changed and removed source records. Native status and consolidated snapshot remain byte-equivalent in decoded JSON after the check. No model calls or browser warnings/errors occurred. Semantic freshness remains outside this revision-status feature.


Consolidation synthesis preparation: native POST `/api/memory/consolidation-input` accepts project and 2..20 exact source pins, loads the selected current revisions under the memory write lock, rejects duplicate/removed/foreign/changed/hash-substituted sources, and returns explicitly selected source name/content/expiry plus ID/version/hash. The serialized source array is capped at 24 KiB and fingerprinted with SHA-256. This preparation is read-only and makes no provider calls; model synthesis and durable synthesis proposals remain incomplete. Native fixtures qualify source preparation and rejection after source revision; HTTP qualification remains pending.


Model consolidation proposals: native POST `/api/memory/consolidation-proposals` and SDK `propose_memory_consolidation(settings,sources)` explicitly generate one candidate from verified pinned sources. Generation uses the existing extraction semaphore, Chat without writes, bounded source bytes, 4096-token output maximum, 60-second timeout, input/output guardrail policies and metadata-only generation traces. The prompt asks to retain qualifications and expose conflicts rather than silently choose a winner; model compliance still requires review. A successful bounded JSON candidate is saved in the existing immutable proposal store and read with its existing proposal endpoint. Notes are not automatically saved. SDK `accept_memory` retains both proposal origin and source pins; native admission rejects missing or substituted consolidation pins and source changes. Local SDK tests qualify request/preflight and paired origin fields; native admission tests cover joint origins and omitted-pin rejection. Provider execution, guards and restart HTTP qualification for this new mode are pending; UI and proposal catalog discovery remain open. Existing conversation proposal catalogs exclude proposals without session identity.

Model consolidation HTTP qualification passed against the local mock provider: exactly one generation receives the verified source rows; notes are unchanged before explicit acceptance; proposal and accepted note retain source pins and proposal origin. Missing pins, source changes before generation and invalid zero-candidate responses fail without saving a candidate. Native model traces exclude source/candidate text. The proposal and accepted snapshot survive restart unchanged. Full Studio contracts and full Rust tests pass, plus 99 SDK tests. Live model semantic quality, new-mode guardrail qualification, UI generation/acceptance and catalog discovery remain open.

Studio consolidation composer now offers “Подготовить черновик моделью”. The selected 2..20 pinned sources are sent explicitly with current provider/model in Chat without writes. The response must contain exactly one bounded candidate and matching project/source pins; it fills editable title/content and preserves proposal plus consolidation origins for the existing Save action. Generation creates no accepted memory note. Repeated generation clicks are guarded; changed draft, source selection, project/scope or closed dialog excludes stale responses. Local UI fixtures qualify single request admission, Chat/write settings, editable review-only candidate, paired origins, substituted source pins and stale draft/dialog responses. Browser model-generation qualification remains pending.

Consolidation guardrail HTTP qualification passed: input block rejects before generation, output block calls the mock once but stores no proposal, and observe mode retains one candidate plus failed guardrail evidence without accepting a note. Source/candidate content remains absent from traces. Native fixture rejects oversized combined source input and duplicate IDs while preserving original snapshot hashes. Full Studio contracts pass with the local mock provider; 99 SDK tests and synthesis UI fixtures pass. Live model and browser synthesis execution remain pending.

Browser model consolidation flow passed against the local streaming mock: selected two pins, generated one saved proposal with no new note, edited candidate title/content, explicitly saved exactly one note, and verified both proposal origin and immutable source pins. The original model candidate and original source snapshots remain unchanged. Exactly one provider request, no browser warnings/errors. This qualifies UI/runtime flow, not semantic quality of a real model.


Consolidation proposal discovery: GET `/api/memory/consolidation-proposals?project_id=...&offset=...&limit=...` and SDK `memory_consolidation_proposals` return saved candidate metadata without generation. The catalog filters exact project and consolidation kind before totals/pages, orders IDs descending, returns 1..100 rows with offset up to 1000, and excludes candidate/source text and settings. Shared proposal scans remain bounded to 1000 files, 128 KiB each and 16 MiB aggregate; failures reject the catalog rather than returning partial data. Ordinary conversation extraction proposals remain separate. SDK routing/preflight tests pass; Rust compilation/extraction tests pass. HTTP pagination/project/privacy/restart fixtures are added but pending execution. Studio saved-proposal browser remains open.

Consolidation catalog HTTP qualification passed: two saved model proposals paginate as separate one-row pages with correct totals/has_more; source counts and note counts match while text/settings remain absent. Global scope is empty and no provider requests occur. Invalid/unknown query parameters are rejected. The exact first-page receipt survives restart. Full Studio contracts pass with the local mock provider and all 100 SDK tests pass. Saved-proposal Studio catalog UI remains incomplete.

### Saved consolidation proposal review in Studio

The memory dialog now opens retained consolidation proposals in pages of 20 for the selected project or global scope. Opening a proposal loads its candidate into the editable memory form with both proposal provenance and exact source revision pins. It makes no generation request and does not save a note. Explicit save still validates current source revisions on the server. UI contract checks cover duplicate request admission, scope/editor stale response exclusion, malformed or duplicate pins, and preservation of both origins. The 100 Python SDK tests pass. This increment has not been visually qualified in a live browser; provider/model quality remains unqualified.

Saved consolidation catalog browser qualification: a retained mock-generated proposal opened into editable fields, with no new provider request and no note created until explicit save. Saving edited text retained both proposal and source revision provenance; the two source notes and original proposal stayed unchanged. Browser warnings/errors were empty. Evidence: `/tmp/allpaka-catalog-browser-evidence.json`, screenshot `/tmp/allpaka-consolidation-catalog-browser.png`. This verifies browser behavior against a local mock, not real-model semantic quality.

Proposal storage admission now runs before provider generation for both conversation extraction and consolidation. It rejects non-regular JSON entries, escaped storage directories, oversized existing receipts, and insufficient catalog capacity. Admission conservatively reserves two files of up to 128,000 bytes for the two concurrent proposal requests; a serialized receipt must also fit that per-file bound before commit. Native tests cover the 998/999-file admission boundary, per-file overflow, and total-byte exhaustion. Three focused memory extraction tests pass. HTTP rejection/provider-call-count qualification for this increment remains pending.

Proposal storage HTTP qualification now passes for both count and byte exhaustion. Conversation extraction and consolidation return rejection with zero additional mock-provider requests, unchanged memory notes, and no added retained proposals. Temporary quota fixtures are removed before continuing, and ordinary generation plus restart persistence still pass. The full Studio contract suite completed successfully; evidence log `/tmp/allpaka-proposal-capacity-http.log`. Real-cloud/model quality qualification remains separate.

Conversation extraction source status: GET `/api/memory/proposals/:id/source-status` and SDK `memory_extraction_source_status` compare the original bounded extraction prefix against its retained SHA-256. Status is current, changed, or unavailable; metadata includes original boundary, current message count and running state, with no conversation text, inference, or note mutation. Appended messages leave the original prefix unchanged. Four native extraction tests, 101 SDK tests, and the full Studio HTTP suite pass, including exact-prefix current status before/after restart and rejecting consolidation receipts on this endpoint. Changed/shortened prefix behavior is unit-tested; HTTP changed/unavailable cases and Studio UI remain pending. Evidence `/tmp/allpaka-extraction-status-http-recheck.log`.

Studio extraction source review: opened conversation proposal receipts now include an explicit source-check button. It displays current/changed/unavailable prefix status, distinguishes revision comparison from factual correctness, and reports running conversations. The UI validates receipt identity/project/boundary/hash, status consistency and no-inference/no-mutation metadata. Duplicate clicks and late replies after session/project/dialog/panel changes are excluded. UI contract tests for all three statuses and malformed/stale replies pass alongside existing memory save and consolidation catalog tests. Real-browser qualification and changed/unavailable HTTP cases remain pending.

Extraction source status HTTP qualification now covers all statuses: appended messages retain current status and the original hash; an isolated test history rewritten while Studio is stopped returns changed after restart; a test history moved outside the data directory returns unavailable. Retained proposals remain exactly readable, notes remain unchanged, and status requests make no additional mock-provider calls. The full Studio contract suite passes (`/tmp/allpaka-extraction-source-status-cases.log`). Source-check browser visual qualification remains pending.
