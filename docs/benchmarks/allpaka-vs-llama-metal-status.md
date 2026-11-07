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

## Verdict

**Metal GPU path works and is fast in absolute terms, but allpaka is not
faster than llama.cpp on every LLM.** Prefill often matches or beats llama;
decode wins on some MoE shapes and loses on others.

| Model | Prefill (allpaka/llama) | Decode (allpaka/llama) |
| --- | ---: | ---: |
| qwen3-0.6b Q8_0 | 0.51× | **1.13×** |
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
