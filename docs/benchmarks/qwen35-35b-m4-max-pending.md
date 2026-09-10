# Qwen3.5/3.6-35B-A3B Q4_K_M on Apple M4 Max — bench slot

Paired comparison template for the hybrid GDN + MoE model (`qwen35moe`).

```sh
scripts/bench-compare.sh models/qwen3.5-35b-a3b-Q4_K_M.gguf 480 32 5
scripts/bench-matrix.sh models/qwen3.5-35b-a3b-Q4_K_M.gguf
```

| Workload | llama.cpp | allpaka | allpaka / llama.cpp |
| --- | ---: | ---: | ---: |
| `pp480` prefill | _TBD_ (~1305 historical) | _TBD_ (~1412 historical, ~108%) | _TBD_ |
| `tg32` decode | _TBD_ | _TBD_ (~111–115 historical) | _TBD_ |

Historical note from `docs/moe-prefill.md`: prefill beat llama after Q8_0
pipe + `attend_mm256`. This branch routes Q8_0 decode through
`matvec_q8_0_mv` (default ON) and raises MEGA's expert cap to 256 for
opt-in `ALLPAKA_MEGA=1` experiments on this 256-expert model.

## Method

Same protocol as `docs/benchmarks/qwen3-30b-m4-max-2026-09-02.md`. Record
`ALLPAKA_Q8_MV`, `ALLPAKA_SWFUSE`, and `ALLPAKA_MEGA` in the artifact env.
