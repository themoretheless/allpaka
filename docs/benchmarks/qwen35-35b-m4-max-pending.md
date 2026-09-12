# Qwen3.5/3.6-35B-A3B Q4_K_M on Apple M4 Max — bench slot

Measured 2026-09-12 on
`models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (`qwen35moe`). MEGA disabled.
`ALLPAKA_BENCH_SKIP_MTP=1`. No env overrides → Q5/Q8 `_mv`, SWFUSE, ATTN_MV
at defaults (ON).

## Sustained warm

Three fresh allpaka processes, then llama-bench `-r 3`. After the first
allpaka load the machine was warm; runs 2–3 are the useful band.

| Workload | llama.cpp | allpaka (best / warm) | allpaka / llama.cpp |
| --- | ---: | ---: | ---: |
| `pp480` prefill | 301.47 ± 22.06 | **1443** (warm; cold#1 541) | **~4.8×** |
| `tg32` decode | 26.38 ± 0.48 | **102** (warm; cold#1 40) | **~3.9×** |

Capability census (warm runs): Q4K / Q5K / Q8_0 / Q6K / F32 all
`matmul-kernel=yes`. Decode GPU path `32/32`, **23200** dispatches / 32 tok
≈ **725 / tok**.

Log: `.rbench/qwen36-35b-warm-20260912-173625.log`.

```sh
ALLPAKA_BENCH_SKIP_MTP=1 ALLPAKA_BENCH_PP=480 ALLPAKA_BENCH_TG=32 \
  target/release/allpaka bench --engine models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
llama-bench -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -p 480 -n 32 -r 3 -ngl 99 -ctk f16 -ctv f16
```

## Notes

- Historical cool-machine reference (`docs/moe-prefill.md`): allpaka ~1412 /
  111–115 vs llama ~1305. This session’s llama abs is soft (~301 / 26);
  allpaka warm abs is near the historical band. Prefer Δ% on a cool day
  before updating the Sep-02-style table.
- Do **not** enable MEGA; see `docs/decode-opts.md`.
- Confirms default Q5_K / Q8_0 matmul + decode path on this hybrid GDN+MoE
  model without overrides.
