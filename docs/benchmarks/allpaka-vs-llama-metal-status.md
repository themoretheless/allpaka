# allpaka vs llama.cpp on Metal — status (M4 Max)

Canonical cool matrix: [matrix-2026-09-12.md](matrix-2026-09-12.md)
(artifacts `.rbench/matrix-20260912-195149/`). MEGA off. `ALLPAKA_BENCH_SKIP_MTP=1`.

## Recapture 2026-11-06 (qwen3-30b Q4_K_M, pp480/tg32)

Artifacts: `.rbench/rerun-r2-on/`, `.rbench/rerun-r2-off/` (paired
`ALLPAKA_MMID_BN16` on/off — no effect, heuristic does not trigger at
avg 30 rows/expert; the 1136 outlier earlier that morning was page-cache
warm-up, not bn16). Machine warms through the morning: 1227 → 1393 → 1481.

- allpaka pp 1481 vs llama 1633 = **0.90×** (was 1.02× on 2026-09-12:
  allpaka flat 1480-1500, **llama.cpp updated +11%** 1465 → 1633)
- 2026-11-07, after the mmid tile-descriptor grid (+1.5%): pp **0.916×**
  (1462 vs 1596), tg **1.16×** (143 vs 124). Stage barriers re-measured
  post-tiles: ~2% (were 5.5% pre-tiles — the drain was straggler
  imbalance, the tiles fixed most of it).
- 2026-11-07, `attend_mm_fq` (qk_prep folded into attend_mm, producer
  groups publish K/V to a device counter, consumers spin; one dispatch +
  one barrier per layer-chunk removed): pp 1502.9 vs 1497.2 paired
  median = **+0.4%** (wins every pair). Kill switch `ALLPAKA_PF_FQK=0`.
  This was the last identified structural dispatch fusion; the residual
  pp gap (~8%) is diffuse ramp/bubble cost across ~10 dispatches/layer,
  with every individual mm kernel at parity ±3% against llama's.
- allpaka tg 147 vs llama 124 = **1.21×** (was 0.96×: allpaka 115 → 147)

So: no allpaka prefill regression, the gap is llama.cpp's own progress.
Decode on 30B is now a solid win.

### Where llama's +11% came from (source diff b ~Sep-12 → 0.9.0 cd959bae9)

- **mm kernels essentially unchanged** — mul_mm.metal diff is int64
  overflow fixes plus a two-pass `mul_mm_id_map0`. The new
  `mpp::tensor_ops::matmul2d` kernel is gated off on M4 Max
  ("tensor API disabled for pre-M5").
- The gain is **graph fusion** (ggml-metal-fusion.cpp, PR #28948 and on):
  topk_moe (SOFT_MAX+ARGSORT+GET_ROWS+norm+scale in one kernel),
  moe_reduce 2..8 (MUL+ADD chain), RMS_NORM+SCALE/MUL/ADD, ADD chains,
  MUL_MAT+ADD, and mul_mm_id amax-rescale removal (#30924).
- **allpaka already has equivalents** of every one of these (GPU route
  pick/scan/scatter, fused combine, fused residual-norm, fused
  swiglu-down) — this convergence is exactly why llama caught up from
  0.98× to 1.10× on pp.
- allpaka pp480 runs 481 dispatches in ONE command buffer, GPU busy
  321/322 ms — no CPU/launch overhead left. The residual ~10% on 30B is
  kernel-vs-kernel mm throughput.

### Next lever for pp on 30B

Head-to-head microbench DONE (2026-11-06): llama legacy
`kernel_mul_mm_q4_K_f32` (extracted standalone, MIT, from
kernels/mul_mm.metal @ cd959bae9) vs allpaka mm kernels, same data,
same grid, `MM_LLAMA=1` in `examples/mmbench.rs`:

| shape (m=480) | best allpaka | llama | Δ |
| --- | ---: | ---: | ---: |
| 2048→768 (30B expert) | 11.58 | 11.57 | 0% |
| 4096→1408 (GLM gate/up) | 12.25 | 12.00 | **+2% ours** |
| 1408→4096 (GLM down) | 12.42 | 12.50 | −1% |
| 2048→6144 (dense FFN) | 12.29 | 12.49 | −2% |

**The mm kernels tie (±2%).** Our dequant already uses llama's tricks
(d/16 + unshifted mask) plus a uint4 vector load. The residual pp gap on
30B is therefore NOT the plain-mm kernel. Candidates, in order:

1. mmid grouped regime: head-to-head DONE (2026-11-06): llama
   `kernel_mul_mm_id_q4_K_f32` (same extract file) vs our `mmllg_id_q4_k`
   at 128 experts, 30 rows/expert, 2048→768: **llama 11.42-11.45 vs ours
   11.10-11.13 TFLOPS-effective — llama +3%**, not the missing 10%.
   Their grid is (non-empty tile, col-tile) from the map0 prepass; ours is
   worst-case (col-tile, row-tile, expert) with early exit.
2. Attention path at pp480 (llama FA kernels vs ours) — measured:
   attention is 29% of layer GPU time (1.58 ms/layer), FFN 71%.

### Where the pp gap actually lives (2026-11-07 update, Qwen3-30B pp480)

Post-tiles accounting per pass (321 ms GPU busy): attention mm + FFN mm
261 ms, logits head ~25 ms (298 GFLOP, identical work for llama), norms /
rope / router / combine ~35 ms. Stage barriers now ~2% (probe
NO_STAGE_BARRIER: 1492 vs 1463). ALL barriers (racy probe): 1477 vs 1477
— nothing. The mm kernels tie llama's in microbench, but in-model the
mmid stages reach only ~8-10 TFLOPS vs 11.1 isolated: routing imbalance
plus cold weights, not fixed by tiles alone.

Direct probes on the onebuf path:
- **Stage barriers cost 5.5%**: `ALLPAKA_NO_STAGE_BARRIER=1` (racy, timing
  only) 1539 vs 1458 t/s = ~27 ms/pass of pipeline drain between FFN
  stages. Removing them safely needs per-expert producer-consumer sync
  (the route barrier is load-bearing — dropping it gives phantom 726 t/s).
- Router is already one fused kernel (`route_fused`, default on); combine
  is `combine_rows`; shared-expert swiglu is a standalone pass (~0.08
  ms/layer); `route_patch_shared_w` one more small dispatch.
- Fusible remainder (shared swiglu into shared down mm via a non-id
  SWIGLU variant, patch into combine) is ~6-10 ms/pass ≈ +2-3%.

Original note:

Per pass: attention 76 ms + FFN 185 ms = 261 ms, but GPU busy is 321 ms.
**~60 ms/pass (19%) is small-kernel time** — norms, rope, router, combine,
embed: ~10 dispatches per layer, each paying launch/occupancy ramp.
llama's total pass is 294 ms: if their fusion batch (topk_moe,
moe_reduce, RMS_NORM+SCALE, ADD chains) cuts their small-op time to
~30 ms, the numbers line up exactly with kernels tying. The next lever is
fusing our per-layer small kernels (residual_norm+rope, combine+residual,
router chain), not the mm or mmid kernels.

## 2026-11-07: combine+resnorm landed, norm+router falsified, route histogram

- `combine_rows_resnorm` (049171d): the MoE combine now also writes the next
  layer's attn-normed rows (`pf_hs`), saving 47 `norm_rows` dispatches per
  pp480 chunk. Verified active (188/188 layer-buffers probed) but **pp-neutral**
  (1458-1463 both ways) — the removed norm was ~0.2% of the pass. Kept as
  strictly less work; kill switch `ALLPAKA_PF_COMBINE_RESNORM=0`.
- `norm_rows_router` (FFN norm + router matmul per row, one dispatch):
  **1-4% SLOWER** — 480 row-threadgroups re-read the 128x2048 router weights
  instead of one tile-mm amortising them over all rows. Reverted; dead end.
- `ALLPAKA_ROWS_DEBUG` (5d8ba56) dumps the per-expert row histogram of the
  last fused layer. pp480 qwen3-30b: **65/128 experts hot, 3840 rows, 164
  tiles of 32 → 26.8% tile padding**. That exactly explains in-model mmid
  ~8.1 vs 11.1 TFLOPS microbench (11.1 × 0.732). llama's mm_id tiles the
  same way, so this is parity, not a gap — do not chase it via tile sorting.
- bench prints the Metal init error when the backend stays unattached
  (a kernel compile break previously surfaced as a silent 7x slowdown).
- 16-row mmid tiles (437b1de, `ALLPAKA_MMID_BN16=1`): slightly WORSE
  (1470-1485 vs 1490-1492). Decisive: **the FFN mmid stage is
  weight-bandwidth-bound, not compute-bound** — each hot expert's weights
  stream fully regardless of row count, so tile padding costs nothing.
  Side benefit: bn16 (avg_rows<=24) now uses the tight tile grid too.
- Attention decomposition (PF_SPLIT, pp480, per layer): qkv 0.67 ms,
  wo 0.53 ms (both ~15 TFLOPS, mm parity), attend 0.23 ms, prep 0.04 ms,
  fnorm+router 0.13 ms — no attention-side gap left to fuse.
- **Dense datapoint (Qwen3-0.6B Q8_0, 2026-11-07): pp 10749 vs llama
  10848 = 0.99x, tg 440 vs 332 = 1.32x.** Core mm/attention path is at
  parity; the 30B pp gap is MoE-path-specific (routing/mmid structure),
  not kernel quality: mmllp vs llama kernel_mul_mm_q4_K_f32 on all four
  30B shapes (5120/2048/768 out) is +/-3%. Cold-weight streaming
  (MM_WSPAN=400MiB) does NOT explain in-model mmid cost (11.1 TFLOPS
  warm and cold). gate+up combined mmid in-model ~15.6 TFLOPS-effective
  (FFN_SPLIT) — actually ABOVE microbench. The residual ~8% is diffuse:
  ~480 dispatches/chunk of ramp + stage bubbles, no single addressable
  kernel. buffer modes re-verified: onebuf 1464-1471 > defer 1427-1433 >
  wait 1405-1407.

## Verdict

**Metal GPU path works and is fast in absolute terms, but allpaka is not
faster than llama.cpp on every LLM.** Prefill often matches or beats llama;
decode wins on some MoE shapes and loses on others.

| Model | Prefill (allpaka/llama) | Decode (allpaka/llama) |
| --- | ---: | ---: |
| qwen3-0.6b Q8_0 | 0.99× (10749 vs 10848) | **1.32×** (440 vs 332) |
| qwen3-30b Q4_K_M | **1.02×** → 0.90× ¹ | 0.96× → **1.21×** ¹ |
| Qwen3.6-35B Q4_K_M | **1.03×** | **1.35×** |
| GLM-4.5-Air Q4_K_M | **~1.4×** | **0.86×** (39.0 vs 45.5) |
| qwen3-235b Q2_K_XL | ~1.0× | **0.72×** (20.0 vs 27.8) |

Unsupported on disk (skipped): dense Qwen3.8, Gemma4 `head_dim=512`.

¹ Recapture 2026-11-06 (see above): allpaka pp flat, llama.cpp updated.

## What already landed (absolute Metal wins)

Residency sets, concurrent encoders, whole-token decode, Q4/Q5/Q8 `_mv`,
SWFUSE, ATTN_MV, mm_ll — see metal-performance playbook. These moved allpaka
from CPU-class rates to competitive GPU rates; they do **not** imply
universal llama beat.

## Open gaps (priority)

1. **GLM decode (~0.86×)** — experts dominate (down ~33%, gate_up ~25%).
   Shared expert adds Q5+Q8 traffic every MoE layer. In-graph Q4/Q8 matvec BW
   ~0.85× llama is the remaining lever (not more dispatch folds).
2. **235B decode (~0.72×)** — Q2_K path; after GLM.

## Branch knobs pending cool verify (agent cannot measure)

Cursor agent seatbelt → `MTLCreateSystemDefaultDevice() == nil`. Fail-closed
harness: [scripts/glm-ab-stagger.sh](../../scripts/glm-ab-stagger.sh).

Unmeasured in-agent (opt-in / default until Terminal A/B):

- `ALLPAKA_MV_ID` default **ON** — llama `mul_mv_id` `grid.z=slots` for INDEXED Q4/Q8 `_mv`
- `ALLPAKA_SHARED_STAGGER=1` — shared Q5 after expert Q4 barrier
- Q4 `_mv` float4 activation loads (always on in tree)
- `ALLPAKA_MV_TG` ∈ {64,128,256}

```sh
# Run outside Cursor agent (Metal required):
scripts/glm-ab-stagger.sh
# A/B: MV_ID on/off, SHARED_STAGGER, llama. Ship if decode ≥ llama.
```

## Dead ends (do not retry)

RFUSE under normflag, SHARED_TAIL tok/s, NR0≠2, Q8 K-split+shmem, MEGA
device atomic spin, GUFUSE/CFUSE/RTOPK defaults — [decode-opts.md](../decode-opts.md).

## Next if GLM still <1.0× after stagger A/B

Metal System Trace / GPU capture on one decode token:

1. Expert Q8_0 down `[4096×1408]` indexed ×8 — effective GB/s vs llama `mul_mv_q8_0`
2. Expert Q4_K gate/up `[1408×4096]` indexed ×8 — vs llama `mul_mv_q4_K`
3. Barrier / encoder idle between stages (whole-`y_arena` drain)

Do **not** add fuse env knobs before those numbers.

## Interleaved / warm note

Hot or interleaved runs (e.g. `.rbench/beat-llama-20260913-091533`: allpaka
decode ~34 vs llama ~39) understate cool ceilings; prefer matrix cool ratios
for claims.

## 2026-11-08: Qwen3-8B-Q4_K_M (dense middle front)

Model: `~/.cache/allpaka/models/Qwen3-8B-Q4_K_M.gguf` (4.7 GiB, HF
`Qwen/Qwen3-8B-GGUF`). Fits in the 12 GiB free disk; the MoE flagships
(80B-A3B ~46 GiB, GLM-4.5-Air ~61 GiB) still do not.

Clean matrix (REPEATS=3, `.rbench/lock-8b-final`, machine cool, nothing
else running): **pp 836 vs 884 = 0.946×, tg 87 vs 84 = 1.039×**.

### Measurement-contamination lesson (do not re-learn)

Right after the download the same matrix showed tg 0.73×, then 1.38×,
then 0.64× across consecutive runs. Causes: (1) cold page cache — the
first bench process pays the SSD read of a just-written 4.7 GiB file;
(2) memory pressure (30 GiB in compressor, <100 MiB unused) — interleaved
engine processes evict each other's weights; (3) thermal drift after
hours of benching (±10%+ between rounds, late rounds faster as pressure
clears, then slower as the SoC heats). Standalone Q4_K matvec at dense
shapes: 298 GB/s (q8_0 control 487 GB/s) — kernels were never the
problem. Barrier probe `ALLPAKA_SKIP=bar/barn/barf/bara` on a cool
machine: decode 87.3 vs 87.9 t/s — barriers cost <1% on dense decode.

New harness: `crates/allpaka-backend/tests/gpu_densebench.rs`
(`--ignored`) measures flat Q4_K matvec bandwidth at dense 8B shapes
([12288×4096], [4096×12288], [6144×4096]) with a Q8_0 control.

### Standing on all fronts (clean matrices)

| front | pp | tg |
|---|---|---|
| Qwen3-0.6B Q8_0 | 1.013× | 1.350× |
| Qwen3-8B Q4_K_M | 0.946× | 1.039× |
| Qwen3-30B-A3B Q4_K_M | 0.928× | 1.164× |

Decode target (≥1.05×) holds on 0.6B and 30B; 8B decode is +4% (within
noise of the target). Prefill is uniformly −5..−7% on all three — the
diffuse dispatch-cost story above, not any single kernel.

## 2026-11-09 — Q6_K pipelining for the 8B dense front (commits 9ef6116, 6e955f9)

GGUF dump of Qwen3-8B-Q4_K_M showed mixed quant types: `attn_v.weight`
and `ffn_down.weight` are Q6_K on 18/36 layers (Q4_K elsewhere), the
output head is Q6_K. Three follow-on defects, all fixed:

1. `mm_kernel_for` mapped Q6K to the unpipelined `mmll_q6_k`; the
   12288-long K loop of ffn_down paid two threadgroup barriers per
   32 values. New `mmllp_q6_k` (double-buffered, one barrier per block).
2. The fused-QKV guard demanded equal types, so wq/wk Q4_K + wv Q6_K
   fell back to three separate mm dispatches per mixed layer. New
   `mmllp_qkv_q4k_q6k` (q/k staged as Q4_K, v tiles take the Q6_K stage
   path; the branch is threadgroup-uniform). Proven BIT-EXACT against
   three separate pipelined dispatches
   (`fused_qkv_mixed_q4k_q6k_matches_separate`).
3. Dense FFN rides `ffn_batch_grouped` as one group, so the Q6_K down
   bypassed `mm_kernel_for` entirely: `mmll_id_q6_k` unpipelined and
   without the swiglu fold. New `mmllps_id_q6_k` (pipelined + swiglu
   B-staging), wired into `mmid_swiglu_kernel_for`.

Selection verified end-to-end on the 8B model via `ALLPAKA_DEBUG_QKV=1`
markers (mixed qkv and both Q6_K kernels fire on the expected layers).
`ALLPAKA_MM_PIPE=0` reverts all three.

### New measurement hazard: prefill logits are NOT run-to-run bit-stable

An FNV-1a checksum over the raw logit bits (bench prints
`logits-fnv=...`) differs across IDENTICAL reruns of the same binary —
on pipe=0 and pipe=1 alike, so the race predates the Q6_K work and lives
in a pre-existing prefill path (concurrent dispatch overlap or an
unordered reduction). argmax is stable. Consequence: bitwise e2e A/B on
logits is impossible on this path; kernel changes are arbitrated by the
parity/tolerance tests plus bit-exact same-shape kernel comparisons
(fused vs separate), not by end-to-end hashes. Root cause not chased
(perf-neutral; quality-neutral at argmax level).

### Open: mmbench q6_k anomaly, needs a quiet machine

One noisy-machine run (searchpartyd/rustc storm) showed llama's
`kernel_mul_mm_q6_K` ahead of `mmll_q6_k` on 12288->4096 m=480 while
`mmllp_q6_k` landed behind both, and the q4_k control run inverted the
llama ordering minutes later — i.e. load-contaminated, not actionable.
Re-measure at top-process <60%: `MM_FMT=q6_k MM_NIN=12288 MM_NOUT=4096
MM_M=480 MM_LLAMA=1` with the q4_k same-shape control.

## 2026-11-09 (round 2): decode side of the Q6_K fold + fused decode QKV

Two decode-token gaps found in `decode_token` (m=1):

1. **SWIGLU_X missing on the Q6_K matvecs.** `sw_capable` listed every
   swiglu-folding matvec except `matvec_q6_k` / `matvec_q6_k_mv`, so the
   Q4_K_M dense down projection (Q6_K, 12288->4096) paid the standalone
   swiglu dispatch + one barrier per layer that Q4_K downs avoid. Both
   Q6_K matvecs now carry SWIGLU_X (buffer 8 = up rows, `sw4` on the
   activation loads, slot/row strides mirrored from x). Same fold the
   MoE routed/shared downs already had for other formats. Bit-exact
   vs standalone swiglu + plain matvec
   (`matvec_q6k_swiglu_fold_matches_separate`, both kernels).
   NOTE: the whole decode SWFUSE path stays opt-in
   (`ALLPAKA_SWFUSE=1`); flipping the default needs a cool-machine A/B.
2. **Decode QKV was three dispatches.** New `matvec_qkv_q4k_q6k` runs
   wq(Q4_K)+wk(Q4_K)+wv(Q6_K) as ONE matvec dispatch over n_q+n_k+n_v
   rows (NR0=2, simdgroup-uniform region branch, bodies verbatim from
   `matvec_q4_k_mv` / `matvec_q6_k_mv`, so bit-identical —
   `matvec_qkv_q4k_q6k_matches_separate`). llama.cpp runs three mul_mv
   here, so this is a candidate win over llama, not parity work.
   **Opt-in `ALLPAKA_QKV_MV=1`** for now: the gufuse measurement
   (qwen3-30b dual gate/up 105.0 vs 113.7 tok/s) showed fusion can lose
   to concurrent separate dispatches; cool A/B decides.

Selection proven end-to-end on Qwen3-8B (`ALLPAKA_DEBUG_QKV=1`:
`decode swiglu-fold on matvec_q6_k_mv`, `decode fused qkv
matvec_qkv_q4k_q6k`), declines=0. Telemetry note: the per-layer
dispatch counter is a static estimate (+8/layer), so fused variants do
not move the printed dispatch count.

Commit `def20ea`. Pending (quiet machine): the mmbench q6_k re-run
above; full 3-model matrix to re-baseline after the Q6_K rounds.

### Verdicts from interleaved A/B (contended machine, order-reversed controls)

- **`ALLPAKA_SWFUSE=1` on 8B decode: clear LOSS.** Forward pairs 1/6
  negative, reversed pairs 0/6 negative, median ≈ −20% tg. Physics:
  the fold doubles the down matvec's activation stream (gate+up instead
  of precomputed act) — ≈+400 MB of L2 reads per token on 8B shapes,
  matching the measured hit. Consistent with the 2026-09-26 default-off
  flip (235B q3_k −7.7 ms/token). The new Q6_K SWIGLU_X support stays
  available behind the flag; the default stays OFF. No flip.
- **`ALLPAKA_QKV_MV=1` on 8B decode: NEUTRAL.** In the quiet window
  (78-84 tok/s both arms) deltas were −1.3/+2.1/0.0/0.0/−0.1% — the
  concurrent encoder already overlaps the three dispatches, so fusion
  only removes launches, not bubbles. Stays opt-in, default OFF.
