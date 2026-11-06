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

1. mmid grouped regime: our mmid loses ~7% vs plain at 30 rows/expert
   (10.8 vs 11.6 TFLOPS-effective, `MM_ID=1 MM_EXP=128`); llama's
   mm_id + map0 compaction may lose less.
2. Attention path at pp480 (llama FA kernels vs ours) — needs per-phase
   GPU timing to confirm.

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
