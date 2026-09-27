# Making a dead-GPU run loud

Parent investigation:
[`../2026-09-25-gpu-window-cost/235b-cliff.txt`](../2026-09-25-gpu-window-cost/235b-cliff.txt),
whose DoD item 1 was "read the driver's verdict after each wait". This is that
item, implemented and verified. Written 2026-09-25 19:52-20:09 UTC.

## The hole

`wait_until_completed` returns happily when the driver dropped the buffer. The
output arena is reused across dispatches, so the read-back of an aborted
command buffer returns the *previous* dispatch's values, which are finite,
which makes the non-finite logits check pass, and which keeps every other
counter in `metal.rs` - waits, dispatches, `attempts`, `successes`, `declines`,
`residency set=true` - saying full coverage. The cliff arms printed 400+ tok/s
of throughput for weights the GPU never touched, and exited 0.

## The change

Built on `8a16214` in a detached worktree (`/private/tmp/allpaka-guard`),
because `gpu/metal.rs` and `bench.rs` in the main tree are modified and
uncommitted by another session. Nothing here is merged yet.

`crates/allpaka-backend/src/gpu/metal.rs`

- `CB_ERRORS: AtomicU64` + `CB_ERROR_FIRST: Mutex<Option<String>>`.
- `note_cb_error(&CommandBufferRef)` - sends `error`, and on a non-nil result
  reads `code`, `domain` and `localizedDescription` and hands the formatted
  string to `record_cb_error`.
- `record_cb_error(String)` - counts every failure, keeps the first detail,
  logs at most three (a run that fails fails on every dispatch).
- `ns_to_str` - `UTF8String` through `CStr`, no new dependency.
- Called from `note_gpu_times` (so the 13 instrumented waits are covered for
  free) and added explicitly at the 5 waits that read their own timestamps or
  read no times at all: `cstamps_report`'s blit, the verify-debug readback, and
  the three `split_here!` macros. All 20 waits in the file are now covered.
- `pub fn cb_error_stats() -> (u64, Option<String>)`, reachable as
  `gpu::cb_error_stats()` through the existing `pub use metal::*`.
- One unit test: two recorded failures move the counter by two and leave a
  detail behind.

`crates/allpaka-cli/src/bench.rs`

- `guard_cb_errors(phase, before)` - `anyhow::ensure!(failed == 0, ...)`, plus
  a printed `gpu command buffers failed during <phase>: 0` line so a clean arm
  shows the check was live.
- Called twice: `warmup and prefill`, and `decode`. The baselines are taken
  before each phase's warmups, so a GPU that dies during warmup cannot be
  read as a clean measurement.

`gpu/stub.rs` and `gpu/cuda/mod.rs` carry a `(0, None)` accessor so a
cross-backend caller compiles; the CUDA doc comment states plainly that CUDA
has no command-buffer object and that its failures ride the launch and sync
return codes instead, which the call sites already check.

## Verification, both directions

Script: [`guard-run.sh`](guard-run.sh) (caps 32 and 6),
[`boundary-recheck.sh`](boundary-recheck.sh) (caps 12, 10, 8). Same binary,
235B, `PP=64 TG=8 SKIP_MTP=1`. Logs in `guard-cap*-log.txt`,
`recheck-cap*-log.txt`.

| cap | windows | span (GiB) | exit | prefill argmax | buffers failed |
|----:|--------:|-----------:|-----:|---------------:|---------------:|
| 32 | 3 | 82.67 | 0 | 291 | 0 |
| 12 | 9 | 98.67 | 0 | 291 | 0 |
| 10 | 11 | 102.67 | 0 | 291 | 0 |
| 8 | 14 | 108.67 | 1 | 151935 | 2 |
| 6 | 21 | 122.67 | 1 | 151935 | 2 |

`span` is the sum of window lengths as `../2026-09-25-gpu-window-cost/span-model.py`
computes it. Refused arms say:

```
allpaka: GPU command buffer failed: MTLCommandBufferErrorDomain: Insufficient
  Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory) (code 8)
Error: benchmark invalid: 2 GPU command buffer(s) failed during warmup and
  prefill: MTLCommandBufferErrorDomain: Insufficient Memory (code 8)
```

A refused arm also writes no report: `ALLPAKA_BENCH_REPORT` is opened after
the guards, so `guard-cap6-log.txt` has no companion JSON while the three
clean arms each have one. A dead run cannot leave an artifact behind.

Two facts worth keeping from the numbers. First, **two** dropped buffers are
enough to kill a run: the failing arms log 2 counted failures out of 941
prefill dispatches, and the clock line then reads `executing 0 ms, scheduling
31662 ms` against `executing 1993 ms, scheduling 0 ms` for cap 12 - the same
941-dispatch sequence either way. Second, the guard fires *before* the run is
reported, so no artifact can cite a refused arm by accident.

## The cliff is not memory pressure

The earlier boundary, `T in (104.67, 107.96] GiB` of mapped span, was measured
at 19:20 UTC with the machine at 84% free RAM. These arms ran at 20:05-20:08
UTC with `vm_stat` reporting 0-2 GiB free and another session holding ~76 GiB,
and the boundary is unchanged: cap 10 (102.67 GiB) clean, cap 8 (108.67 GiB)
refused. The new pair brackets `T` more loosely - `(102.67, 108.67]` - and
contains the published interval, so nothing is retracted. What is added is
that `T` did not move when host RAM did, which is what a device-side mapping
budget should do and what a pressure-induced failure should not.

## The read-back guard, and what it found on the first run

The parent file's other DoD item asked for the mapping itself to be verified
rather than only witnessed. `gpu::check_windows()` does that: one MSL kernel
bound to each window's own `bytesNoCopy` buffer, 32 threads x 8 samples per
window, each sample one byte at a page index the host recomputes with identical
wrapping `u32` arithmetic, summed per thread so 32 words come back per window.
The kernel never receives a host pointer, so what it reads is what the device's
page tables resolve for that buffer; the host then reads the same addresses
through its own mapping. A mismatch returns the window, its size, its address,
the thread and both sums.

The bench calls it right after attach, before any warmup, and refuses the run
on a mismatch (`weight read-back: 96 samples over 3 windows, all matching` on
the clean side).

The very first run of the check accused the **shipped** configuration: 235B at
cap 32, `window 0 (32 GiB at 0x7000000000) thread 0: the GPU read 941, the host
read 726`. That was the check's own bug, not the driver's - `page * 16384u` in
MSL is `uint` arithmetic, and a 32 GiB window has 2M pages, so every offset
above 4 GiB wrapped around inside the window. Widened to
`(uint64_t)page * 16384ull`, the clean arm matches (`readback-cap32-log.txt`)
and the accusation disappeared. Two things worth writing down: the 0.6B model
had passed the buggy build, because a 0.7 GiB window cannot overflow, so the
first arm the check ever examined was the first one big enough to break it -
which is the right order by luck, not by design; and a self-check that reports
a detailed, confident mismatch on a config known to be good must be believed
inverted before it is believed at all.

The check was then mutation-tested, the way the indexed-`_mv` parity test was:
the host comparison's address was moved forward by 16 bytes, and the run
against the 235B was refused - `the GPU read 726, the host read 783`, with the
GPU's own 726 unchanged from the clean run (`mutation-log.txt`). The comparison
has teeth. Reverted and rebuilt before the numbers above were taken.

On the cliff arm the two guards agree, and the read-back fires first:
`readback-cap8-log.txt` shows the refusal on the read-back itself - `window 0
(8 GiB at 0x7000000000) thread 0: the GPU read 0, the host read 965` - with two
`code 8` lines immediately above it, logged by the read-back's own command
buffer. At cap 8 the very first GPU work the process issues is dropped, so
nothing the model reports afterwards is about this run's weights. That the
failing arm reads zeros rather than wrong bytes is the evidence that the driver
*refuses* an over-wide span instead of resolving it badly; a genuinely silent
mis-mapping is still unobserved, and this check is what would catch it.

## Registry

`docs/decode-opts.md`'s knob registry named three unpriced Metal knobs; gap 3
(`ALLPAKA_GPU_WINDOW_GIB`, "no measurement at all") is closed by this
investigation and its own amendment was appended there at 20:45 UTC. Gaps 1 and
2 (`ALLPAKA_MM_K64`, `ALLPAKA_MM_LL_F32`, both prefill-side) remain unpriced.

## Not done

- Detection floor of the read-back: 256 page samples per window. It is a
  statement about whole windows, not about pages - one wrong page inside a
  4 GiB window escapes with probability 1 - 256/262144. Raising the depth is
  the knob, and the cost is real: the samples are chosen to land on distinct
  pages, so a deeper check first-touches that many 16 KiB pages of the model
  file on every run, outside the timed region but inside the process whose
  page cache the timing depends on.
- The `122.67 GiB` arm shows the count stops at 2 even though hundreds of
  buffers follow; whether the driver reports once and then stops, or whether
  later buffers complete empty, is untested. The bench refuses on either.
- Rates printed by these runs are not measurements. `TG=8` under another
  session's load; the throughput column is meaningless here by design.
- The code is not merged. It lives in the worktree and, so it cannot be lost
  with it, as [`dead-gpu-guard.patch`](dead-gpu-guard.patch): 121 added lines
  across four files, no deletions. `git apply --check` passes against the main
  tree as it stands with another session's uncommitted `metal.rs` and
  `bench.rs`, so landing is one command whenever those edits settle.

  It has deliberately not been applied, for two reasons. The first is
  attribution: a second session committing by pathspec would sweep these lines
  into its own message. The second is the instrument. The read-back dispatch
  first-touches up to 256 pages per window (84 MiB across the 235B's three
  windows), which changes the page-cache state a paired A/B run is measured
  against. Landing it mid-campaign would move a co-tenant's baseline under it
  without their knowing; land it between campaigns, and re-establish any
  paired-control bands the first time the guard is in the tree.
- Coverage was checked by script rather than by eye: 20 `wait_until_completed`
  sites in `metal.rs`, each with a `note_cb_error` or a `note_gpu_times` (which
  now calls it) within the next six lines; 0 uncovered.
