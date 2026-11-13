# CUDA prefill determinism and long-prompt prefill on RTX 5090, 2026-11-13

**Result:** the prefill logits hash changed from run to run because the f16
weight cache in `allpaka_ggml.dll` decided what to cache from the VRAM free at
that moment. With a budget fixed by the card and the model the hash is the same
in every run, at the same speed. Separately: on a 2 048-token prompt prefill is
about 98% of llama.cpp, not above it.

## Setup

- RTX 5090 (32 607 MiB), Windows 11, Qwen3-32B Q4_K_M, MTP skipped.
- allpaka: `allpaka bench --engine <model>`, one process per run, release
  build of the commit that adds this file.
- llama.cpp `97e4ca7` with `third_party/llama.cpp.patch`, CUDA build:
  `llama-bench -ngl 99 -r 1`, same pp and tg, run in the same loop.
- The card was checked idle (under 1.5 GiB used) before each allpaka run.
  Another process loaded the same model on the card twice during the day; the
  runs it overlapped ran at 10 to 55 tok/s and are not in the tables.

## Cause

Prefill GEMMs read a weight either as a cached f16 copy (cuBLASLt) or straight
from the quantised blocks (ggml). The bridge cached a weight when
`cudaMemGetInfo` reported `free >= size + 3 GiB`. The model's weights need
about 60 GiB as f16, so the cache always filled up, at about 8.4 GiB here, and
where it stopped depended on a few MiB of desktop VRAM. Runs that stopped one
10 MiB weight apart took different arithmetic for that weight and printed a
different hash.

Narrowing it down, 5 runs each, pp512, distinct `logits-fnv` values seen:

| Switch | distinct hashes in 5 runs |
|---|---|
| none (default) | 3 |
| `ALLPAKA_PF_GRAPH=0` | 3 |
| `ALLPAKA_PF_NORM_ROPE=0` | 2 |
| `ALLPAKA_PF_DEFER_ADD=0` | 2 |
| `ALLPAKA_LT=0` | 3 |
| `ALLPAKA_GGML=0` | 1 |
| `ALLPAKA_F16_W=0` (no f16 cache) | 1 |

## Fix

The backend passes the bridge a byte budget when weights are attached:
card size minus weight bytes minus a 6 GiB reserve, rounded down to 256 MiB.
Here the cache closed at 7 190 MiB in 55 weights. The bridge caches weights in
call order until the budget is spent and then closes the cache; it no longer
asks how much memory is free to decide. If less than `size + 1 GiB` is really
free it closes early and says so. `ALLPAKA_F16_W_MIB` overrides the budget.

## Before and after, pp512 / tg128

"old rule" is the previous binary with the rebuilt DLL (no budget set, so the
free-memory rule applies). "budget" is this commit.

| Round | old rule pp | budget pp | llama pp | old rule tg | budget tg | llama tg | old hash | budget hash |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 848 | 3 813 | 3 535 | 79.9 | 79.8 | 72.18 | `0c1c556d…` | `3848986d…` |
| 2 | 3 823 | 3 817 | 3 494 | 79.8 | 79.8 | 72.92 | `5833029f…` | `3848986d…` |
| 3 | 3 804 | 3 821 | 3 492 | 79.8 | 79.8 | 72.62 | `87818cbd…` | `3848986d…` |

- **Prefill:** 3 817 against 3 825 tok/s on average, inside the noise; 109% of
  llama.cpp.
- **Decode:** unchanged, 110% of llama.cpp. `decode tokens-fnv` was
  `95a4ac529cc87100` in all six allpaka runs.
- **Hash:** three values in three runs before; one value after.

## Repeatability after the fix

`logits-fnv` and argmax, 5 runs each unless noted:

| Mode | runs | distinct hashes | argmax |
|---|---|---|---|
| pp512, default | 12, and 4 more while another process shared the card | 1 (`3848986decfbbfc1`) | 201 |
| pp512, `ALLPAKA_PF_GRAPH=0` | 8 | 1 (same value) | 201 |
| pp128, default | 5 | 1 (`8f0f58…`) | 271 |
| pp128, `ALLPAKA_GGML=0` | 5 | 1 (`d83c9e…`) | 271 |
| pp2048, default | 3 | 1 (`b92fe84404b7fdf4`) | 261 |

`scripts/check-prefill-determinism.sh <model> [runs] [pp]` repeats this check
and fails when two runs disagree. The earlier report of pp128 without ggml
changing its argmax between runs did not reproduce.

## Prefill without graph replay, and on a long prompt

The prefill CUDA graph replays only when the chunk shape and KV base repeat.

pp512 with `ALLPAKA_PF_GRAPH=0`:

| Round | allpaka pp | llama pp | ratio |
|---|---|---|---|
| 1 | 3 748 | 3 358 | 111.6% |
| 2 | 3 733 | 3 410 | 109.5% |
| 3 | 3 724 | 3 508 | 106.1% |

pp2048, defaults (four 512-token chunks, each with a longer KV history):

| Round | allpaka pp | llama pp | ratio |
|---|---|---|---|
| 1 | 3 516 | 3 600 | 97.7% |
| 2 | 3 545 | 3 596 | 98.6% |
| 3 | 3 535 | 3 597 | 98.3% |

- **One 512-token chunk, no graph:** 106% to 112% of llama.cpp.
- **2 048-token prompt:** about 98% of llama.cpp. llama.cpp gets faster from
  pp512 to pp2048 (3 500 to 3 600); allpaka gets slower (3 820 to 3 530).

## Chunk size on the long prompt

`allpaka bench` feeds the prompt in chunks of `ALLPAKA_PREFILL_CHUNK` tokens
(512 by default). pp2048, three interleaved rounds, card idle before each run:

| Round | chunk 512 | chunk 1024 | chunk 2048 | llama pp |
|---|---|---|---|---|
| 1 | 3 528 | 3 653 | 3 713 | 3 599 |
| 2 | 3 539 | 3 651 | 3 721 | 3 601 |
| 3 | 3 541 | 3 651 | 3 725 | 3 601 |

- **Chunk 2048:** 103.3% of llama.cpp, against 98.2% at 512 and 101.4% at 1024.
- **Memory:** peak VRAM 28.7 GiB at chunk 2048, about 1.3 GiB more than at 512.

The default was not changed. `cuda_long_prefill`'s check that two identical
prefills in one process give identical logits failed once in seven runs at
chunk 2048 (one 1 097-token chunk after a 3-token prefix) and passed at chunk
2048 with `ALLPAKA_PF_GRAPH=0`, and four times out of four at chunk 512.
Another process was loading the same model on the card during part of that
series, so the failure is neither reproduced nor explained; the test now prints
the cosine and both argmaxes when it fails.
