# GLM-4.5-Air Q4_K_M on Apple M4 Max — bench slot

Measured 2026-09-12 on
`models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf` (+ shard 2). MEGA disabled.
`Q5_0` plain mm present (`matmul-kernel=yes` for 11.34 GiB).

## Sustained warm (preferred for absolute-ish rates)

Three fresh allpaka processes after a short cool-down, then llama-bench `-r 3`
in one process. Machine was thermally soft (both engines ~4× below the
historical ~360 / ~41–43 ceilings).

| Workload | llama.cpp | allpaka | allpaka / llama.cpp |
| --- | ---: | ---: | ---: |
| `pp480` prefill | 94.37 ± 9.38 | ~90.7–95.9 (best 95.9) | **~1.02×** (parity under thermal) |
| `tg32` decode | 11.43 ± 0.68 | ~8.9–9.4 (best 9.4) | **~0.82×** (−18%) |

Decode GPU path: `32/32` successes, `25696` dispatches / 32 tok ≈ **803 / tok**
(shared expert + MoE; no MEGA).

## Process-paired rbench (high variance)

`.rbench/glm45-air-20260912-172451` — warmup 1 + 5 AB/BA pairs, cooldown 2.5s.
Medians (Inconclusive): prefill llama 81.8 / allpaka 46.8 (−38%); decode
llama 9.8 / allpaka 6.0 (−40%). Fresh-process cold starts dominate; prefer the
sustained table above for the decode-gap read.

```sh
# Prefer sustained when machine is cool:
ALLPAKA_BENCH_SKIP_MTP=1 ALLPAKA_BENCH_PP=480 ALLPAKA_BENCH_TG=32 \
  target/release/allpaka bench --engine models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf
llama-bench -m models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf \
  -p 480 -n 32 -r 3 -ngl 99 -ctk f16 -ctv f16
# Paired (noisy when hot):
scripts/rbench-vs-llama.sh models/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf
```

## Notes

- Do **not** enable MEGA (can freeze the Mac); see `docs/decode-opts.md`.
- Env defaults: Q5/Q8 `_mv` ON, SWFUSE ON, ATTN_MV ON; no overrides in reports.
- Re-measure when cool before claiming a close of the historical decode gap.
