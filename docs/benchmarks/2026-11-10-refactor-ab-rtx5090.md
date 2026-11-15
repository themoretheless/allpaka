# Refactor A/B on RTX 5090 (CUDA), 2026-11-10

Checks invariant 4 of `docs/refactor/PLAN.md` (paired A/B and greedy parity
for hot-path edits) on the CUDA backend.

- A: `2e3dcd8` (before PLAN 1.5).
- B: `90a0f5e`, which adds 1.5 (CUDA/Metal switches read once through
  `RuntimePolicy`), 3.20–3.22 (`ffn_residual`, `AttnMode`, `QuantMat::desc`)
  and 3.23 (GPU eligibility decided at load).
- Model: Qwen2.5-7B-Instruct-Q4_K_M. `allpaka bench --engine`, pp512, tg128,
  MTP skipped. Release builds (`lto = "fat"`).
- Machine: Windows 11, RTX 5090 32 GB. Not quiet: desktop apps and parallel
  cargo builds ran during the measurement.
- Script: `scripts/ab-binaries.sh` (A B A B, 6 pairs after one warm-up each).

| pair | A pp | B pp | Δpp | A tg | B tg | Δtg | prefill logits |
|---|---|---|---|---|---|---|---|
| 1 | 266.5 | 268.6 | +0.79% | 172.2 | 158.3 | −8.07% | same |
| 2 | 254.6 | 265.4 | +4.24% | 148.8 | 159.6 | +7.26% | same |
| 3 | 257.0 | 262.2 | +2.02% | 179.5 | 177.8 | −0.95% | same |
| 4 | 259.1 | 259.2 | +0.04% | 167.9 | 156.6 | −6.73% | same |
| 5 | 269.1 | 244.9 | −8.99% | 154.7 | 174.6 | +12.86% | same |
| 6 | 246.2 | 249.7 | +1.42% | 179.9 | 179.9 | +0.00% | same |

Median Δ: prefill +0.9%, decode −0.5%. Pair-to-pair spread is ±9–13%, so
neither number is distinguishable from zero. The prefill logits hash
(`logits-fnv`) is identical in every run of both builds.

Conclusion: no regression visible; the expected gain from removing per-launch
`std::env::var` calls (PLAN 4.2) is below this machine's noise floor. A quiet
run would be needed to resolve effects under ~5%.
