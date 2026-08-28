# MoE prefill optimizations (GLM-4.5-Air on Apple Silicon)

Benchmark setup: M4 Max 128 GB, `GLM-4.5-Air-Q4_K_M` (47 layers: 1 dense +
45 MoE+shared, 160 experts, top-8, hidden 4096, moe_ffn 1408), pp480.
Baseline `llama-bench -p 480` on the same machine: 361-376 tok/s
(High Power mode, watch `pmset -g | grep powermode`; thermal drift between
runs is 5-10%, always A/B back-to-back).

Result: prefill **286 → ~331-336 tok/s** (llama stays ~8-12% ahead; the
remaining gap is spread ~5-10% across all mm kernels, in-model mm achieves
~10.5 TFLOPS vs ~12.4-12.7 in a steady-state microbenchmark).

**Update 2026-08-28**: with the later kernel work in-tree (pipelined
K-loop, attend_mm) plus the per-chunk one-buffer segments (item 8), GLM
measures **362.8-368.4 tok/s pp480** - at llama parity (361-376 baseline),
the ~8-12% gap is closed. Decode 36.5-37.5 tok/s (the llama decode gap,
~41-43, remains a separate front).

## What landed (all verified: 199 tests green, greedy output identical)

1. **GPU-side MoE routing** (`route_pick` / `route_scan` / `route_scatter`
   kernels, `route_buf` in `Gpu`, `gpu.rs` ~1410-1530). Top-k selection,
   per-expert prefix sum and token scatter run on device; the router logits
   are rescued out of `y_arena` before arena reallocation
   (`PF_ROUTER_AT` static). `prefill_attn_block` returns an empty Vec when
   routing is on GPU; `model.rs` then builds a `GroupedRoute` and falls back
   (`return None`) for `n_used > 8`, and for Softmax gating unless it is the
   exact qwen3moe case (no selection bias, `norm_topk_prob` set).
   Env: `ALLPAKA_GPU_ROUTE=0` disables.

2. **SwiGLU folded into the down-projection mm** (`mmlls_id_q4_k`,
   `DEFINE_MM_ID_LL_NK` with `SWIGLU=1`). The kernel applies
   `silu(gate) * up` while staging B (reads raw gate via x, raw up via
   buffer(9), row stride via buffer(12)). Removes the standalone swiglu
   pass and one full-device barrier per MoE layer.

3. **Shared expert reordering**: its gate/up plain mms run in the same
   barrier-free window as the expert gate/up phase; its swiglu in the down
   phase; its down rows are appended right after the expert rows so the
   combine treats them as ordinary hit rows.

4. **Row-tile striding in mmid kernels** (`for r1 = tg.y*32; r1 += tgd.y*32`).
   Launch the worst-case grid (`ceil(m/32)`): empty threadgroups retire for
   free and this measured FASTER than a tight grid, which serializes hot
   experts through the loop. `ALLPAKA_MMID_RTILES` pins the grid for A/B.

5. **Vectorized q4_k dequant** (`dqll_q4_k` uses uint4 loads): +~1%.

6. **Debug tooling**: `examples/mmbench.rs` (`MM_ITERS`, `MM_WSPAN`),
   `ALLPAKA_FFN_TIME` (attn/FFN GPU ms per buffer), `ALLPAKA_FFN_SPLIT`
   (per-stage timing; warmup mixes stages, filter top-47 lines).

7. **Deferred prefill commits** (qwen3-30b work, verified there): the fused
   layer buffers chain through an MTLSharedEvent (`pf_ev`/`pf_ev_val`/
   `pf_pending` in `Gpu`, `commit_chained`/`prefill_drain`) instead of one
   CPU wait per layer; the CPU encodes layer N+1 while the GPU runs layer N.
   Measured 1092 -> 1107 tok/s pp480 (GPU bubbles ~16 -> ~0 ms; executing
   unchanged). Hazards handled: route_buf counters self-clean on the GPU
   (route_scan) so CPU staging happens once per alloc/bias pointer, the
   y_arena logits rescue drains first, `prefill_begin`/`prefill_end`/
   `prefill_abort` drain the chain (the abort covers a mid-chunk fallback).
   `ALLPAKA_PF_DEFER=0` reverts to per-layer waits. Also added
   `ALLPAKA_PF_SPLIT` (per-stage timing of the attention buffer, same
   serialising caveat as FFN_SPLIT).

8. **One-buffer prefill chunk** (`pf_obuf_cmd` in `Gpu`): the fused chunk
   encodes into ONE command buffer - a sequential encoder per stage,
   the existing (load-bearing) barriers order the stages, one commit+wait
   in `prefill_end`. llama.cpp runs its whole graph the same way. Measured
   qwen3-30b pp480: 1222-1230 vs 1196-1198 tok/s with the event chain
   (pp1200: 1124 vs 1095); GPU executing unchanged - this removes the
   per-buffer start overhead. The y_arena logits rescue becomes a GPU
   `copy_f32` encoded in place (nothing has committed for a CPU readback
   to see). `ALLPAKA_PF_ONEBUF=0` reverts to the event chain.
   **2026-08-28 rework: per-chunk SEGMENTS instead of all-or-nothing.**
   The shared buffer is armed lazily by the first eligible layer of the
   chunk (chained behind the last committed buffer through `pf_ev`) and
   sealed - committed chained - by the first layer that cannot join it;
   the next eligible layer re-arms. GLM's leading Dense layer now splits
   the chunk into segments (attn segment, dense FFN on its own chained
   buffer, one segment for the 45 MoE layers: 3 waits/chunk vs 93 on the
   chain) instead of disabling onebuf process-wide. A fused non-route
   buffer now also chains through the event (`defer` widened to
   `req.fused.is_some()`) so later segments stay ordered after it.
   Measured (pp480, back-to-back, powermode 2): GLM 362.8/364.4 vs
   368.4 chain (neutral within drift - but GLM now sits at llama parity:
   361-376 baseline); qwen3-30b 1523.6 vs 1484.8 (+2.6%, the onebuf win
   preserved); qwen35moe 1467.1 vs 1426.6 (+2.8%). `pf_obuf_disabled`
   is gone.

9. **Software-pipelined K-loop in the q4_k mm kernels** (`mmllp_q4_k`,
   `mmllpg_id_q4_k`, `mmllps_id_q4_k`, default ON, `ALLPAKA_MM_PIPE=0`
   reverts): double-buffered A/B staging - dequant/stage block i+1 into the
   shadow set while the MMA chews block i, one threadgroup barrier per
   block instead of two. mmbench on the 30B forms: plain 12.4 vs 11.4
   TFLOPS loop / 8.9 vs 8.0 one-shot (2048->2048), mmid gate 11.1 vs 9.2,
   down 10.0 vs 8.9 TFLOPS-effective. In-model pp480: 1428-1463 vs
   1335-1339 tok/s (paired, MM_PIPE on/off). Same-session reference:
   llama-bench 1354 - **allpaka +6.8%**. Greedy byte-parity verified.

## What was tried and falsified (do not retry without new evidence)

- **(qwen3-30b)** llama `kernel_mul_mm_id_q4_K` port: CLOSED BY INSPECTION +
  measurement. The kernel is structurally identical to our `mmllg_id_q4_k`
  (same 64x32 tiles, NK=32, 2 barriers per K-iter, dequant-to-smem, MMA
  from smem, worst-case grid with early exit) - ours additionally has the
  uint4-vectorized q4_k dequant and the direct simdgroup_store fast path
  (theirs always round-trips through smem + scalar store). No advantage to
  port. (The /tmp extraction harness for a head-to-head timing was
  abandoned: modern ggml-metal.metal template plumbing resists piecemeal
  extraction, and there was nothing left to find.)
- **(qwen3-30b, pp480)** K-split in mmid for occupancy: the in-model regime
  (~30 rows/expert, ALL 128 experts active) already measures 8.5-11
  TFLOPS-effective in mmbench (MM_ID=1, MM_ACT sweep 8/32/128 is flat) -
  the mmid grid is not occupancy-bound; no K-split port.
- **(qwen3-30b)** qkv fusion into one weight matrix: back-to-back
  dispatches already amortise the dispatch ramp (mmbench MM_PAR=3 in one
  buffer: 9.96 TFLOPS vs 7.8 single; loop 11.4) - one fused dispatch would
  recover ~1%, not worth the loader surgery.
- **(qwen3-30b)** `attend_mm` v1 (two-pass, threadgroup-staged K/V, 32
  rows/tg): tied t8 standalone, lost in-model. **v2 landed** (llama
  kernel_flash_attn_ext style: 8 q rows/tg, 64-pos tiles, K/V read directly
  by the MMAs from device memory, online softmax per row in registers, O in
  threadgroup with per-tile rescale): standalone 0.278 vs 0.742
  ms/dispatch at m=480, in-model pp480 1105-1113 -> 1181-1188 tok/s
  (executing -30 ms), pp1200 941 -> 1083 tok/s. Default ON,
  `ALLPAKA_ATTN_MM=0` reverts. Bugs caught by the MM_ATTN harness: softmax
  rescale missed `scale`, epilogue wrote 2 float4/lane over the next head,
  clamped K blocks aliased stale data into valid positions (only
  fully-masked blocks may clamp, and only into the chunk's own span).
- In-model mm penalty excluded causes (all measured, qwen3-30B shapes in
  mmbench): cold weights (16 GB window cycle, one-shot: same 7.9 TFLOPS),
  mmap-backed no-copy weights vs driver buffers (same), TLB, denormal
  activations, barriers (ALLPAKA_NO_BARRIER: +4% total). The attention
  buffer's mm+barrier skeleton reproduces standalone at 8.5 TFLOPS
  (MM_STRUCT=1). xctrace profiler session (Xcode-beta, Metal System Trace):
  the default template records per-command-buffer intervals only - the
  shader-timeline table stays empty (shaderprofiler flags are off in the
  template's XRRecordingOptions; flipping them in the keyed archive
  produced a malformed trace), and the counter table recorded only the RT
  counter. Per-dispatch GPU times stay unavailable from CLI; the earlier
  "in-model 2x" estimates were mostly PF_SPLIT/FFN_SPLIT serialization
  inflation (those modes serialize each stage into its own buffer).
  Barrier probes: ALLPAKA_NO_STAGE_BARRIER = noise, ALLPAKA_NO_BARRIER =
  +3.4% but phantom (the route barrier is load-bearing, mmid skips work).
  MTLCounterSampleBuffer probe (ALLPAKA_GPU_COUNTERS harness): this device
  exposes only the "timestamp" counter set and only atCommandBoundary
  sampling - per-dispatch compute-encoder samples assert "not supported on
  this device", so the mode is a silent no-op. Kernel ideas beyond the
  llama formula: 64-row tiles (`mmllr64_q4_k`, in the source for reference)
  measured SLOWER (10.8 vs 11.3-11.9 TFLOPS on 30B forms: fewer
  threadgroups loses more than the halved dequant/barriers win); K64 inner
  tiles previously falsified; simdgroup async copy has no header on this
  toolchain; fp16 accumulators untested (greedy byte-parity risk).
  Net: with onebuf + attend_mm the engine is at llama parity
  (~1231-1333 vs 1231-1360 tok/s pp480 across machine states); further
  gains need kernel-level work beyond the llama-structure or GUI
  Instruments (Metal Shader Timeline has no safe CLI enable).
- Barrier count / phase reordering: barriers cost ~0.5 ms/layer total, no win.
- `MTLDispatchType::Serial` instead of Concurrent+barriers: 0.
- Removing all barriers: phantom 726 tok/s - the route barrier is load
  bearing, mmid reads an empty table and exits without doing work.
- K64 inner tiles (`ALLPAKA_MMID_K64`): slower for both mm and mmid.
- Overlapping FFN stages without barriers: 0.8 ms, not worth the risk.
- Cold-weight microbench (2 GB window): 12.4 → 11.5 TFLOPS only; cold
  weights do not explain the gap.
- **Dual gate+up in one dispatch** (`mmllgd_id_q4_k`, `DUAL=1`: tiles with
  `r0 >= n_out/2` read up-weights from buffer(10); interleaved output with
  row stride `2*ffn`): measured SLOWER, 317 vs 331 tok/s. Code kept,
  enabled only via `ALLPAKA_DUAL=1`.

## Env switches (timing/debug)

| Var | Effect |
|---|---|
| `ALLPAKA_GPU_ROUTE=0` | CPU routing fallback |
| `ALLPAKA_PF_DEFER=0` | per-layer CPU waits instead of the event chain |
| `ALLPAKA_PF_ONEBUF=0` | event chain instead of one command buffer per chunk |
| `ALLPAKA_MM_PIPE=0` | two-barrier llama K-loop instead of the pipelined one |
| `ALLPAKA_PF_SPLIT` | per-stage timing of the attention buffer (serialises) |
| `ALLPAKA_DUAL=1` | one-dispatch gate+up (slower, reference) |
| `ALLPAKA_NO_BARRIER` / `ALLPAKA_NO_STAGE_BARRIER` | timing probes only - incorrect results |
| `ALLPAKA_SERIAL` | serial dispatch instead of barriers |
| `ALLPAKA_MMID_RTILES=n` | pin mmid grid y |
| `ALLPAKA_MMID_K64` | K64 inner tiles (slower) |
| `ALLPAKA_FFN_TIME` / `ALLPAKA_FFN_SPLIT` | GPU timing breakdowns |
| `ALLPAKA_BENCH_PP=n` | bench prefill length |
| `ALLPAKA_ATTN_MM=0` | revert to attend_rows_t8 prefill attention |
| `ALLPAKA_ATTN_T4=n` | attend_rows tile: 8 (default), 4, 0=row-per-tg |

mmbench modes: `MM_ID=1` (grouped mmid, `MM_ACT`/`MM_EXP`/`MM_USED`),
`MM_STRUCT=1` (attention-buffer mm+barrier skeleton), `MM_ATTN=1`
(attend_mm vs attend_rows_t8 parity + per-dispatch timing), `MM_MMAP=1`
(mmap no-copy weights), `MM_DENORM=1` (denormal activations).

## Remaining ideas (not done)

- Try llama.cpp's `kernel_mul_mm_q4_K` as the plain-mm kernel. Probably a
  dead end: our mmll/mm64ll measure 11.3-11.8 TFLOPS steady-state on the
  30B shapes (2048->2048/512/768, m=480), matching llama's microbench; the
  in-model loss (~5 TFLOPS effective) is not the kernel peak, not cold
  weights (MM_WSPAN 2 GB: same 11.3-11.8), and one-shot dispatch alone
  explains only 11.8 -> 8.5. The in-model mechanism is unidentified.
- Decode gap (35.8 vs llama ~41-43 tok/s) is a separate front.

# qwen35moe prefill (Qwen3.6-35B-A3B, same machine)

Hybrid arch: 30 gated-delta-net layers (conv 4-tap depthwise + deltanet
recurrence, 16 k-heads / 32 v-heads of 128) + 10 full-attention layers
(head_dim 256, 16 q / 2 kv, partial rope 64/256, sigmoid output gate fused
into wq), 256 experts top-8 (gate/up Q4_K, down Q5_K), gated shared expert.
Baseline `llama-bench -p 480`: 1305-1306 tok/s.

Result: prefill **1002 -> 1412 tok/s (108% of llama)**, decode 111-115
tok/s. Stages: GPU prefill landed at 947 (CPU 70), then:

1. **mmllp_q8_0** (pipelined K-loop for Q8_0, gpu.rs `mmllp_q8_0`): the
   double-buffered staging of mmllp_q4_k format-swapped (NL=2, 34-byte
   blocks). mmbench on the 35B forms: wqkv 2048->8192 12.06 -> 13.12
   TFLOPS (+8.8%), zgate 2048->4096 +9.0%, ssm_out 4096->2048 +11.2%.
   In-model 1002 -> 1283 (+28%): the GDN projections (wqkv/zgate/ssm_out)
   plus all attention wq/wk/wv/wo are Q8_0, so most projection FLOPs go
   through it. Side effect: 0.6B Q8_0 prefill 8784-8875 -> 9179.
   `ALLPAKA_MM_PIPE=0` reverts to mmll_q8_0.

2. **attend_mm256** (gpu.rs): the attend_mm structure at head_dim 256 -
   8 q-rows/threadgroup, K/V in 64-position tiles straight from device
   memory (simdgroup_load transpose), threadgroup scores, online softmax,
   O in threadgroup; 4 simdgroups own 64 dims each, 14 KB threadgroup
   memory (no simdgroup-count reduction needed, unlike attend_s32_256).
   In-model 1283 -> 1412 (+10%); the attend stage was 83 ms of ~500 ms
   prefill (4.2 ms/layer latency-bound attend_rows256) and is why the
   lever exceeded its ~4% FLOP share. `ALLPAKA_ATTN_MM=0` reverts to
   attend_rows256.

Negative result: **Q5_K pipe does not pay** on the down-expert forms
(512->2048: 9.65 -> 10.10 TFLOPS, +4.7%; 2048->512: +6.6%) - too few
K-blocks at n_in=512 to hide dequant, below the +8% porting criterion;
the grouped (mmid) pipelined variant was not attempted.

Verified: `cargo test -p allpaka-backend -p allpaka-model` green;
`verify` vs llama-server PASS at m=29 (GPU prefill) and m=10;
greedy 48 tokens byte-identical to llama, 3 repeated chat requests
byte-identical (SSM prompt-cache snapshots); qwen3-30b regression
prefill 1467 / decode 129, text unchanged.
