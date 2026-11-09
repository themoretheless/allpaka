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

## 2026-11-09: measurement-instrument hunt for the prefill gap

Target raised to ≥1.06× on every front. Before optimising, we tried to
localise the remaining ~5-7% prefill deficit with per-stage GPU timing.
Every instrument on this device fails or distorts:

- `ALLPAKA_PF_SPLIT=1` (commit + wait per stage): real per-buffer
  GPUStart/EndTime, but each wait idles the GPU and the clock sags, so
  stages read several × slower than inside a onebuf chunk. Sums came out
  ABOVE the whole-chunk GPU-busy time — unusable.
- `ALLPAKA_PF_CHAIN=1` (new, this session: per-stage commits, queued
  back-to-back, waited once per layer): the driver pre-assigns queued
  buffers, so GPUStartTime ≈ submission time and per-buffer spans absorb
  the queue wait — garbage. kernelStartTime/kernelEndTime are no better
  (a "gate" stage read 240 TFLOPS, physically impossible).
- `MTLCounterSampleBuffer` (`ALLPAKA_GPU_COUNTERS=1`, cstamp! hooks at
  every stage boundary): the device reports atDispatchBoundary sampling
  UNSUPPORTED (the check `gpu_counters_supported` gates it off); the
  atCommandBoundary-only mode cannot timestamp inside the onebuf chunk.
- The one trustworthy whole-chunk number: onebuf GPU busy == wall
  (637 ms for 480 tok on 8B, "executing 637, scheduling 0, idle 0") —
  the GPU never idles inside the chunk, so there are NO inter-dispatch
  gaps to reclaim. The deficit lives inside kernel execution time.

What survives (trustworthy):

- `mmbench` was silently poisoned: pseudo-random weight bytes leave f16
  scale fields denormal/NaN, dragging every kernel to 2-3 TFLOPS with
  4x run-to-run spread. New `MM_SANE=1` fills valid q4_k blocks
  (d=1.0, small sub-scales): our mmll_q4_k then reads ~12.6 TFLOPS on
  8B shapes and beats llama's kernel_mul_mm_q4_K_f32 at every dense
  shape tried (qkv 6144-out, gate/up 12288-out, down 12288-in).
  So the prefill deficit is NOT the mm kernels in isolation.
- New elimination probe `ALLPAKA_PF_SKIP=qkv,attend,wo,fnorm,gate_up,
  down,combine` drops a stage while keeping the command-buffer
  structure: wall(baseline) − wall(skip) prices the stage in real
  onebuf conditions. Bench aborts on non-finite logits, so the probe
  build (worktree .kimi-worktrees/perf-elim) downgrades that check to a
  warning when ALLPAKA_PF_SKIP is set.

External condition: a foreign agent build (another session) kept the
machine at 100-460% CPU for hours; every series above had to be taken
under DRAM/thermal contention and its absolute values are junk (same
config swung 123..751 tok/s). Elimination ranking across reps matches
the FLOP order (gate_up biggest, attention ~free) but magnitudes are
noise. Rerun the series when the machine is quiet; the script is
/tmp/elim8b4.sh (5 reps x 8 configs, interleaved, TG=1).

## 2026-11-09: llama pp kernel identified; grid-transpose experiment (negative)

Question carried over: which kernel does llama v0.9.0 actually run for
dense prefill mul_mat, and is it structurally better than mmllp_q4_k?

Findings (source: /tmp/llama-src @ cd959bae9 = homebrew v0.9.0):

- The `mul_mv_mma` simdgroup-MMA path
  (ggml-metal-ops.cpp ggml_metal_op_mul_mat_mma) is gated by
  `src1->ne[1] >= rows_min && <= GGML_METAL_MMA_ROWS_MAX` with
  MMA_ROWS_MAX = 16 (ggml-metal-common.cpp:53,143-154). It is a
  few-row kernel for decode/small batches only. pp480 never takes it.
- Dense prefill falls to `kernel_mul_mm` (use_mm path, ops.cpp:2853),
  non-tensor variant on M4 Max (tensor API is M5+/A19+ only;
  ggml-metal-device.m gates GGML_METAL_HAS_TENSOR).
- kernel_mul_mm (mul_mm.metal:155+) is STRUCTURALLY IDENTICAL to our
  mmll_q4_k: same 64x32 tile, NK=32, same sa/sb staging layout
  (`sa + 64*ib + 8*ly + lx`), same ma[4]/mb[2]/mc[8] sgitg-split MMA
  loop, same 2 threadgroup barriers per K-block. Our mmllp_* adds
  double-buffered staging (1 barrier/block) - strictly ahead on paper.
- llama's dispatch grid: x = token tiles (ne11/32), y = weight tiles
  (ne01/64). Ours was x = weight tiles, y = token tiles.

Experiment (grid transposition, patch archived at /tmp/mm-transpose.patch):
all dense mmll*/mmllp*/mmll_qkv*/matmul* kernels and their 7 dispatch
sites + tests + mmbench were transposed to llama's orientation (x =
token tiles) so concurrent threadgroups share one weight strip through
L2 instead of re-streaming the matrix per token tile.

- Correctness: full backend suite passes (gpu_parity incl.
  f32_router, gpu_moe_tail incl. fused-qkv bit-equality), e2e argmax
  unchanged (319).
- Performance: NEUTRAL. Interleaved kernel A/B in the same quiet
  window: 10.89 vs 10.98 TFLOPS (old vs new, <1%). e2e pp480 A/B under
  load: no consistent direction. Conclusion: the M4 Max memory system
  (SLC) already absorbs the per-token-tile weight re-reads; Metal's
  threadgroup rasterization order is not the pp bottleneck.
  NOT ADOPTED; reverted from the tree.

- test-backend-ops perf is not a usable production proxy for mm:
  m=4096,n=512,k=14336 q4_K reports 19.5 ms/run (~3 TFLOPS), which
  contradicts llama-bench's own pp throughput (~14.5 TFLOPS effective);
  per-run graph/alloc overhead dominates. Do not cite it for kernel
  quality.

Implication for the 6% pp gap on 8B (836 vs 884): it is NOT the dense
mm kernel structure and NOT grid rasterization. Our per-layer mm
throughput in production is ~11.6 TFLOPS vs ~12.3 for llama (same
kernel structure!) - the delta lives in per-layer non-mm losses
(stage boundaries, tails) or in measurement asymmetry. Next:
quiet-machine per-stage GPU timing, then attack the largest real
stage-boundary cost.

## 2026-11-09 (later): llama pp480 concurrency map + head parity verified

Source of truth: `GGML_METAL_GRAPH_DEBUG=1 llama-bench -v` (homebrew
v0.9.0, Qwen3-8B-Q4_K_M pp480) prints per-node concurrent/barrier
decisions and fused-op counts; log archived at
/tmp/llama-graph-debug.log (regenerable).

Per-layer node map (llama, dense 8B): RMS_NORM+MUL fused (1 dispatch),
then MUL_MAT Qcur (barrier), Vcur (barrier), Kcur (concurrent with V),
RMS_NORM q (barrier), RMS_NORM k (barrier), SET_ROWS cache_v
(concurrent), ROPE q (barrier), ROPE k (concurrent with rope q),
SET_ROWS cache_k, FLASH_ATTN_EXT, MUL_MAT wo, ADD ffn_inp, RMS_NORM+MUL
ffn, MUL_MAT gate (barrier), MUL_MAT up (CONCURRENT with gate), GLU
swiglu, MUL_MAT down, ADD l_out. = ~18 real dispatches, ~15 barrier
transitions per layer. Our fused path: ~8-10 dispatches, ~8-10
barriers per layer. We fuse every sibling pair they only overlap
(qkv, qk norm+rope+cache store in qk_prep, gate_up, resnorm).
Dispatch count and barrier count are NOT their edge.

Kernel selection at our shapes: plain `kernel_mul_mm_q4_K_f32` (+`_bci`
bounds-check variant only when K%32 != 0; 8B shapes all divide by 32).
No hidden faster variant. FA = `flash_attn_ext_f16_dk128_dv128`.
Norms = `kernel_rms_norm_mul_f32_4`. Adds = `kernel_bin_fuse_*` chains.

Head parity VERIFIED in source: llama-bench's `common_batch_get_one`
(common.cpp:2342) marks `output = (i == n_tokens-1)` - logits only for
the LAST token, so their lm_head is a 1-token mul_mv (Q6_K). Our bench
`forward_batch` (model.rs:1340) likewise heads only the last row
(`matmul_activation(&xs, last_offset, 1)`). The old "~25 ms head,
identical work" note referred to full-480 head math; current bench
paths on BOTH sides are last-token. No asymmetry to reclaim.

MUL_MAT+ADD fusion (`ggml_metal_fusion_check_mul_mat_add`) requires
`ggml_metal_op_mul_mat_may_use_mma` - mma is ne11<=16 only, so the
fusion NEVER fires in pp480. Residual adds are standalone dispatches
on both sides. Parity.

Net: on the dense path every structural lever checked is parity or in
our favour; the 8B pp gap (0.946x) remains empirically unexplained by
source diff. Next instrument: PF_SPLIT per-stage GPU times on a quiet
machine, stage-by-stage against llama's map (llama pp per-layer
15.1 ms vs ours 15.9 ms => ~0.8 ms/layer to find).
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

## 2026-11-09 (round 3): prefill race fixed in attend_mm_fq; A/B verdicts

**Race.** Prefill logits drifted run-to-run at pp480 (argmax stable).
Flag bisect: `ALLPAKA_ATTN_MM=0` and `ALLPAKA_PF_FQK=0` both stabilize;
`PF_ONEBUF=0`, `MM_PIPE=0`, `PF_SPLIT=1` do not. Root cause: attend_mm_fq's
producer/consumer sync was ONE global counter - a consumer's prefix wait
(`n_prod * (tg.y+1)`) could be satisfied by producers of LATER row blocks
finishing first, so early tiles read unwritten K/V. Fix (`819ac5a`):
per-rowblock counters, prefix wait per block, 256 B buffer, counters
zeroed by a tiny dispatch before every fused attention launch (the old
monotonic epoch could deadlock across chunks/sessions with differing
tile counts). Verified: pp480 fnv stable 5/5 and BIT-IDENTICAL to the
unfused `PF_FQK=0` reference; stable on 30B-A3B, 0.6B, two-chunk pp1200.

**A/B (interleaved pairs, contended machine, pp=128, Qwen3-8B):**
- `MM_PIPE` (the whole Q6_K pipelined round): mean +3.4%, 6/8 positive
  pairs. The 9ef6116/6e955f9 work is a real pp win.
- `PF_FQK` post-fix vs unfused: ~0% in stable windows - the correctness
  fix costs nothing measurable; the fusion keeps its original +0.4%.

E2e bitwise A/B of prefill changes is possible again (deterministic
logits), which hardens all future pp comparisons.

## 2026-11-09 (round 4): 30B pp diagnostics + FA inner-loop experiment (branch perf-fa-db)

**Where 30B pp time goes** (PF_SPLIT/FFN_SPLIT per-stage GPU ms shares,
pp480, shares only - split serialisation inflates absolutes ~2.7x):
attend 38%, expert gate+up 27%, expert down 16%, qkv 8%, wo 6%, the
rest (norms/router/swiglu/shared) ~5%. The per-row "ROWS matvec" MoE
path in metal.rs turned out to be the MTP verify path; real prefill MoE
runs `ffn_batch_grouped` with the tiled grouped-mm kernels
(mmll_id/mmllpg/mmllps). `MM_PIPE` (pipelined Q4_K gather + Q6_K
swiglu-fold down) measures +13-21% on 30B pp480 interleaved A/B - the
Q6_K round pays off on MoE too, not only on 8B dense.

**llama flash-attn reference** (ggml 0.29 homebrew, tensor API disabled
on M4 Max -> fa_f16 path): for dk128/f16 llama runs Q=8 query rows per
threadgroup, C=64 positions per tile, NSG=4 - the SAME macro-structure
as our attend_mm/attend_mm_fq. The remaining structural delta is
inner-loop pipelining: llama keeps two 8-dim blocks in flight in QK
(mq[2]/mk[2], 8 iterations of 2 MMAs) and two 8-position strips in
flight in PV (4 iterations of 2x4 MMAs); we ran single chains
(16x1 and 8x4).

**Experiment**: attend_mm + attend_mm_fq QK/PV loops restructured to
llama-style double-buffering. Accumulate order per accumulator
unchanged, so bit-identical - verified: 30B pp480 logits-fnv
`18259760d250fc77` before and after.

**A/B (interleaved pairs, heavily contended machine, top-CPU 150-250%)**:
- 30B pp480 window A: +6.8/+3.0/-3.5 (2/3 positive)
- 30B pp480 window B: -17.5/+2.8/-15.6/+39.9 (swings exceed the signal)
- 8B pp128: -12.8/+1.6/-3.0/-3.2/-11.1 (median ~ -2.5%)

Verdict: INCONCLUSIVE - plausible small win on 30B, plausible small loss
on 8B, noise swamps both. The change is pushed on branch `perf-fa-db`;
main keeps the single-chain loops until a quiet-machine (top-CPU < 60)
interleaved A/B via `scripts/quiet-round.sh` gives a real verdict.

**2026-11-10: perf stack merged into main.** The whole perf-elim stack
(Q6_K pipelining `mmllp_q6_k`/`mmllps_id_q6_k`, decode Q6K folds + fused
QKV opt-in, attend_mm_fq race fix, MM_PIPE default-on, bench PF_SKIP
instrumentation + logits-fnv) is merged into `main` at `d6f805f`, on top
of the upstream Metal module split (`gpu/metal/*`) and rustfmt wave.
Verified on the merge commit: release build OK, gpu_moe_tail 6/6,
gpu_parity 8/8, 30B Q4_K_M pp480 logits-fnv `18259760d250fc77`
(bit-identical). `perf-fa-db` (FA double-buffering) stays a branch pending
a quiet-machine A/B.

**llama v0.9.0 FA f16 kernel, detailed read** (source re-cloned to
`~/.cache/allpaka/llama-src`, ggml/src/ggml-metal/kernels/fa_common.metal):
- QK: NC=(C/8)/NSG=2 strips per simdgroup, inner 8 iterations of
  {2x simdgroup_load Q, 2x K (device mem, transpose), 2x MMA} - exactly the
  perf-fa-db restructure. No threadgroup staging of K in the f16 path.
- PV (DV>64): NC=(C/8)/2=4 iterations of {2 score-strip loads, then
  NO/2=2 rounds of FOUR V loads (mv[4]) + 4 MMAs}. Register accumulators
  lo[NO=4] stay live across the whole tile; O threadgroup round-trip once
  per tile (same as ours). perf-fa-db does 2-in-flight loads where llama
  does 4 - if fa-db validates, a mv[4]-style PV variant is the next step.
- Barriers per tile: 3 (post-QK, post-softmax, post-PV) - same as ours.
- Mask: precomputed per-tile skip table (blk); irrelevant for causal
  prefill - our per-position `t < valid` costs nothing extra there.
- v0.9.0 also has a Metal fusion PASS (ggml-metal-fusion.cpp):
  RMS_NORM+MUL(+ADD), MUL_MAT+ADD (residual into mm store), TOPK_MOE
  (softmax+argsort+gather router), MOE_REDUCE (weighted expert sum).
  Our decode already folds swiglu into down-matvec and gate+up into one
  dispatch; norms/router are candidates if the dispatch count gap shows.
