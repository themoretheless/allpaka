# Qwen3-30B-A3B Q4_K_M profiling, 2026-10-25

Status: numerical checks passed; isolated performance qualification is blocked by observed background load. Four-minute quiet-window admission completed with 0/6 benchmark blocks admitted. No benchmark worker remains running.

## Findings

- Allpaka's whole-buffer telemetry shows the prefill wait dominating CPU encoding (roughly 0.6 ms encode vs 318–394 ms wait in the sampled rounds). Decode CPU encoding was about 7.9–8.0 ms for 32 tokens, with 207–220 ms elapsed GPU command-buffer time. GPU timestamps include effects of contention; these are not exclusive hardware utilization counters.
- An intrusive prefill stage probe identified gate/up and expert down as the largest stages (roughly 68% of attributed stage time together). Gate/up run concurrently: the `ffn gate` interval includes both launches; the following `up` interval is mostly synchronization. Splitting stages disables normal batching and adds waits: these durations must not be sold as a production speed split.
- The first-token decode counter probe attributed about 42% to expert gate/up/down, 26% to QKV/output projections, and 12% to the router. It splits encoders and does not profile every token of the chained decode path.
- Native llama CPU/Metal traces were captured. The first 25-second launch trace ended during startup and has no useful llama GPU samples; it is rejected as a kernel profile. A second 3-second trace attached to the already running process captured llama activity. Neither exposes reliable individual shader durations. The optional libllama evaluation callback supplies an intrusive, synchronized per-operation wall profile instead. It disables graph fusion and is not directly comparable numerically to Allpaka's GPU stage timestamps.

## Changes

1. Remove an adjacent, redundant prefill down/combine arena barrier when there is no shared expert. The dependency barrier before combine stays. Shared-expert ordering is unchanged.
2. In expert matmul tail stores, use SIMD-group memory barriers for the private per-group scratch slices. Whole-threadgroup barriers remain at scratch entry/exit, protecting reuse by other MMA stages and row tiles. Whole-threadgroup tail barriers decrease from 16 to 2, replaced by 16 narrower SIMD-group barriers; full tiles keep their old path.
3. Extend the model benchmark with per-phase encode/wait/GPU timing and a `forced` full-logits mode. Add a native `scripts/llama-matched-bench.cpp` control using the installed libllama/ggml libraries and identical PP480, decode seed `[1]`, 32 forced input tokens, f16 KV and full-logits outputs. Normal throughput mode has no evaluation callback; set `LLAMA_MATCHED_PROFILE=1` only for diagnostics.
4. Allow `MM_SOURCE` in the existing kernel microbenchmark so a frozen reference shader can be compared with a candidate using one executable.

## Correctness

- Real-model `prefill_shared_parity`: passed after the barrier removal and again after the tail-store change (all public batch output paths).
- `gpu_moe_tail`: passed. Six quantization formats; empty and uneven expert groups; output-column tails; striding over multiple row tiles; gather and fused SwiGLU variants; repeated dispatches. Candidate and conservative-barrier outputs match bit for bit, with output canaries intact.
- Existing GPU parity suite: 8 passed.
- Forced before/after Allpaka logits match exactly (maximum difference 0), with 32/32 GPU decode steps and zero declines in every checked round.
- Matched Allpaka/libllama smoke: identical input tokens and matching final argmax (prefill 319, decode 271). Cross-engine logits are NOT bitwise equal: maximum absolute differences 0.0515 (prefill), 0.2490 (decode). This finite probe does not establish universal numeric equivalence or identical intermediate expert choices.

## Performance qualification

The preliminary paired kernel samples drifted substantially (roughly 1–2.4 ms); no gain is claimed. All smoke measurements and failed admission samples are retained. None is an isolated before/after benchmark.

The guarded runner requires three quiet two-second admission samples, then monitors the entire block. It rejects active compilers/renderers, more than one CPU core of background work, or more than 2% background GPU time outside WindowServer. This is a defined low-load desktop gate, not literal OS/GPU exclusivity. It runs all six before/after/libllama permutations and retains rejected blocks. Admission observed substantial Kimi Helper GPU activity and other CPU work; unrelated processes were not stopped.

Raw stdout is compressed under `raw/`; binary/model identities and external native trace paths are in `raw/metadata.json`. Source checkout had pre-existing modifications. Re-run qualification on a quiet machine before claiming acceleration or closing the llama performance gap.

Native control build used installed headers/libraries:

```sh
clang++ -std=c++17 -O3 scripts/llama-matched-bench.cpp \
  -I/opt/homebrew/Cellar/llama.cpp/0.7.0/include \
  -I/opt/homebrew/Cellar/ggml/0.27.0/include \
  -L/opt/homebrew/Cellar/llama.cpp/0.7.0/lib -lllama \
  -L/opt/homebrew/Cellar/ggml/0.27.0/lib -lggml -lggml-base \
  -o /tmp/allpaka-profile-20261025/llama-matched
```

Reference llama-bench uses random tokens: [pinned source](https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/llama-bench/llama-bench.cpp). Its rates must not be mixed with the matched full-logits control as if they were the same workload.
