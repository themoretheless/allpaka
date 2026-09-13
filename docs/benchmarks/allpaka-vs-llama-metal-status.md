# allpaka vs llama.cpp on Metal — status (M4 Max)

Canonical cool matrix: [matrix-2026-09-12.md](matrix-2026-09-12.md)
(artifacts `.rbench/matrix-20260912-195149/`). MEGA off. `ALLPAKA_BENCH_SKIP_MTP=1`.

## Verdict

**Metal GPU path works and is fast in absolute terms, but allpaka is not
faster than llama.cpp on every LLM.** Prefill often matches or beats llama;
decode wins on some MoE shapes and loses on others.

| Model | Prefill (allpaka/llama) | Decode (allpaka/llama) |
| --- | ---: | ---: |
| qwen3-0.6b Q8_0 | 0.51× | **1.13×** |
| qwen3-30b Q4_K_M | **1.02×** | 0.96× |
| Qwen3.6-35B Q4_K_M | **1.03×** | **1.35×** |
| GLM-4.5-Air Q4_K_M | **~1.4×** | **0.86×** (39.0 vs 45.5) |
| qwen3-235b Q2_K_XL | ~1.0× | **0.72×** (20.0 vs 27.8) |

Unsupported on disk (skipped): dense Qwen3.8, Gemma4 `head_dim=512`.

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
