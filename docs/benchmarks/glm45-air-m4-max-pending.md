# GLM-4.5-Air Q4_K_M on Apple M4 Max — bench slot

Paired comparison template. Fill after running:

```sh
scripts/bench-compare.sh models/glm-4.5-air-Q4_K_M.gguf 480 32 5
# or the matrix helper:
scripts/bench-matrix.sh models/glm-4.5-air-Q4_K_M.gguf
```

| Workload | llama.cpp | allpaka | allpaka / llama.cpp |
| --- | ---: | ---: | ---: |
| `pp480` prefill | _TBD_ | _TBD_ | _TBD_ |
| `tg32` decode | _TBD_ (~41–43 historical) | _TBD_ (~36–37 historical) | _TBD_ |

Historical note from `docs/moe-prefill.md`: prefill reached llama parity
(~363–368 vs ~361–376); decode remained behind. This branch adds Q5_K/Q8_0
`_mv` matvecs, default SWFUSE / ATTN_MV, and MEGA `has_shared=2` (gated
shared expert) aimed at that decode gap — re-measure before claiming a win.

## Method

Same fail-closed GPU bench and alternating A/B protocol as
`docs/benchmarks/qwen3-30b-m4-max-2026-09-02.md`. Keep
`ALLPAKA_Q5_MV` / `ALLPAKA_Q8_MV` / `ALLPAKA_SWFUSE` / `ALLPAKA_ATTN_MV` /
`ALLPAKA_MEGA` recorded in the report env dump.
