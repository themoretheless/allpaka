# Decode dispatch census, re-measured: 778/tok (GLM), 674/tok (30B)

Roadmap A9 item: the census in `docs/decode-opts.md` ("qwen3-30b ≈ 698
dispatches/tok; GLM-Air ≈ 803/tok") was declared unbacked — the 2026-09-12
sustained session left no per-run record ([glm45-air-m4-max-pending.md](../glm45-air-m4-max-pending.md)
says so itself, line "The sustained table is **not** reproducible"), and its
own guess was "dispatch density is structural and should not have drifted".
DoD: re-measure it, or drop the figure. Measured; the guess was wrong; the
drift has a mechanism.

## Why this one needed no quiet machine

The number is a **count, not a rate**. `allpaka bench` snapshots
`gpu::stats()` around the decode loop and prints
`gpu during decode: N waits, M dispatches`; M is set by the graph, not by how
busy the machine is. So these runs were taken at load average 8-10 on a
disturbed desktop and the census stands. The `tok/s` lines in the same logs are
disturbed (GLM 12-15 tok/s against a ~46 clean-band anchor) and are **not**
quoted anywhere here.

Regime for every run: `target/release/allpaka bench --engine <model>`,
`ALLPAKA_BENCH_PP=480`, `ALLPAKA_BENCH_TG=16` or `32`,
`ALLPAKA_BENCH_SKIP_MTP=1`, `ALLPAKA_PROFILE=max-performance`, default knobs
unless the file name says otherwise. Binary built 2026-09-24 16:51 from
`3ef9d9c` (the `q5_0_mv` merge) with this worktree's uncommitted changes on
top. MTP off, MEGA off (`mega_enabled()` is always false).

## Result

| arm | file | waits | dispatches | per token |
| --- | --- | ---: | ---: | ---: |
| GLM-4.5-Air Q4_K_M, tg32 | [glm-pp480-tg32.txt](glm-pp480-tg32.txt) | 32 | 24896 | **778.0** |
| GLM, tg16 | [glm-pp480-tg16.txt](glm-pp480-tg16.txt) | 16 | 12448 | **778.0** |
| GLM, `ALLPAKA_Q5_0_MV=0` | [glm-tg16-Q5_0_MV-off.txt](glm-tg16-Q5_0_MV-off.txt) | 16 | 12832 | **802.0** |
| GLM, `ALLPAKA_MV_ID=0` | [glm-tg16-MV_ID-off.txt](glm-tg16-MV_ID-off.txt) | 16 | 12448 | **778.0** |
| qwen3-30b-a3b Q4_K_M, tg16 | [qwen3-30b-pp480-tg16.txt](qwen3-30b-pp480-tg16.txt) | 16 | 10784 | **674.0** |
| 30B, `ALLPAKA_Q5_0_MV=0` | [qwen3-30b-tg16-Q5_0_MV-off.txt](qwen3-30b-tg16-Q5_0_MV-off.txt) | 16 | 10784 | **674.0** |

`waits` equals the token count in every arm: one command-buffer flush per
decode token, both models, both knob settings. That is the structural fact the
census was ever meant to support, and it is unchanged.

## The three findings

1. **Both old figures read high.** GLM 803 → **778** (−3.1%); 30B 698 → **674**
   (−3.4%). The GLM gap is not a rounding: the 2026-09-12 note records the raw
   total `25696` dispatches / 32 tok, and 25696/32 = 803.0 exactly, so the two
   readings genuinely differ by 800 dispatches over the same 32 tokens.
2. **Most of the GLM drift is the `q5_0_mv` port, measured not inferred.** With
   `ALLPAKA_Q5_0_MV=0` the same binary on the same model prints **802.0**, i.e.
   one dispatch/token above the 2026-09-12 figure, so the shipped scalar →
   indexed-`_mv` port removed **24 dispatches per token** and the arithmetic
   residual against the old census is 1/tok, unexplained and not chased. A
   second, independent consequence of that merge: on top of the clean-band
   **+16.05% decode** ([2026-09-24-q5-0-mv-e2e/paired-32.txt](../2026-09-24-q5-0-mv-e2e/paired-32.txt))
   it also thinned the dispatch stream by 3.0% (24 of 802). The share is too
   small to carry the +16%, so it does not rewrite that attribution — it just
   means part of the win is "fewer, bigger dispatches", which is the same
   story the kernel-level byte-rate numbers tell.
3. **The census is invariant to `MV_ID`, and the 30B gap is not the same
   mechanism.** `ALLPAKA_MV_ID=0` leaves GLM at 12448 (778.0) — the knob
   changes the grid, not the dispatch count — so `MV_ID`'s cost, whatever it
   turns out to be, cannot be a dispatch-count cost. That matters for A9's
   still-open `MV_ID` e2E: the hypothesis "it wins by merging dispatches" is
   now dead on a count, before any timing run. And on 30B,
   `ALLPAKA_Q5_0_MV=0` changes nothing at all (10784 either way), so 30B's
   −24 has no measured mechanism here; the figure is reported as re-measured
   and the drift left open.

## What this does not say

No rate claim, no llama comparison, and no per-layer decomposition of the 24
dispatches: 24 does not divide the 43 MoE layers of this 46-layer model, and
mapping it onto which `q5_0` rows are touched per token would need tensor
metadata work this artifact did not do. The honest statement is the knob delta:
one binary, one model, one env var, 802.0 vs 778.0.

## Free by-catch: how much of the waited time is NOT the GPU

The same lines print the CPU/GPU split, which is the instrument
[allpaka-vs-llama-metal-status.md](../allpaka-vs-llama-metal-status.md) gap 1
pointed at ("what is left is the time *between* kernels — the `WAIT_NS` minus
`GPU_BUSY_NS` bubble"). Default GLM
arms, inside one process:

| arm | encode | waited | executing | round trips + idle | idle share |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM tg32 default | 25 ms | 2055 ms | 2010 ms | 6 ms | **0.3%** |
| GLM tg16 default | 10 ms | 1303 ms | 1279 ms | 5 ms | **0.4%** |
| GLM tg16 `Q5_0_MV=0` | 16 ms | 1254 ms | 1225 ms | 4 ms | **0.3%** |
| GLM tg16 `MV_ID=0` | 36 ms | 1225 ms | 1010 ms | 163 ms | **13%** |
| 30B tg16 default | 7 ms | 282 ms | 270 ms | 2 ms | **0.7%** |

Only *within-process* ratios mean anything here — absolute ms across processes
started minutes apart on a loaded desktop are not comparable, and that is how
the `MV_ID=0` row has to be read. On that basis:

- On GLM's default decode path, ~99.6% of the waited window is the GPU
  executing, and host-side encode is 10-25 ms out of 1303. So "the CPU cannot
  keep up with 778 dispatches" is not the remaining GLM gap: the slack is
  0.19-0.31 ms of round trips + idle per token (6 ms / 32, 5 ms / 16), against
  a ~21 ms clean token - ~1-2% even if all of it were recoverable, and these
  are contended-window numbers. Whatever GLM still owes llama is inside
  kernels or inside GPU-side waiting, not between dispatches.
- That does **not** close the shared-expert barrier. A whole-`y_arena` drain
  shows up as threadgroups spinning *while the GPU is executing*, which this
  instrument counts as busy. Distinguishing those needs the GPU capture listed
  in the status doc, not these counters.
- The `MV_ID=0` arm is a lead worth a clean window, not a result: 13% of its
  waited time is round trips + idle against 0.4% for the default arm, at
  identical dispatch counts (12448 both). If that ratio survives a paired
  clean-band run, `MV_ID`'s whole effect is scheduling, and the "grid shape"
  wording in `docs/decode-opts.md` is finally pinned to a mechanism.

