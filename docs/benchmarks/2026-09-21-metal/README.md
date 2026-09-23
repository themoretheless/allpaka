# Allpaka vs llama.cpp — 2026-09-21

Apple M4 Max, 128 GiB. PP480/TG32; decode starts at one token, f16 KV. Five pairs per model, alternating order. Medians in tokens/s.

| Model | Phase | Allpaka median (min–max) | llama.cpp median (min–max) | Observed delta |
|---|---|---:|---:|---:|
| qwen3-0.6b | prefill | 9038.6 (7523.7–9230.4) | 8885.0 (8369.4–8992.7) | +1.7% |
| qwen3-0.6b | decode | 398.0 (390.2–401.5) | 332.8 (323.3–334.9) | +19.6% |
| qwen3-30b | prefill | 1437.2 (1435.0–1441.4) | 1533.7 (1531.6–1535.2) | -6.3% |
| qwen3-30b | decode | 137.5 (134.9–138.2) | 113.2 (110.9–115.3) | +21.5% |

## Reproduction

Allpaka `c1f56452f28ad70c73cbbb08d2cf171da73417da` plus `working-tree.patch`; release build from the working tree. llama.cpp `b29c606e2` (build 10964). Raw JSON, logs, model SHA-256 and per-pair verification are in the model subdirectories.

```sh
BENCH_OUTPUT_DIR=docs/benchmarks/2026-09-21-metal/qwen3-0.6b bash scripts/bench-compare.sh models/qwen3-0.6b-Q8_0.gguf 480 32 5
BENCH_OUTPUT_DIR=docs/benchmarks/2026-09-21-metal/qwen3-30b bash scripts/bench-compare.sh models/qwen3-30b-a3b-Q4_K_M.gguf 480 32 5
```

## Verification and limits

- All 10 allpaka runs used Metal; each decode had 32 successes, zero declines. Both engines used f16 KV.
- Token streams differ, including expert routing on MoE. `comparison_validated` remains false; deltas describe these runs only.
- Decode is short-context (1–33 tokens). This does not measure long-context generation or concurrent serving.
- Background CPU activity was observed before execution. No other user processes were stopped.
- The first 0.6B allpaka prefill was slower; all five samples remain included.
- The failed sandbox CPU attempt is excluded and preserved separately. Optional Airbug export failed; local benchmark artifacts are complete.

## Harness fix

The harness previously expected decode after the prefill context, while current allpaka starts decode from a one-token seed. Updated the guard, llama depth, selection filter and output metadata to one token. `bash -n` and both real five-pair runs passed.
