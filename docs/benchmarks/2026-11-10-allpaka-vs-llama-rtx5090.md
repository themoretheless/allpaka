# allpaka vs llama.cpp on RTX 5090 (CUDA), 2026-11-10

Both engines run on the same machine in the same session.

- **Machine:** Windows 11, RTX 5090 32 GB, Ryzen 9 5950X.
- **allpaka:** `main` at `0640353`, release build. Command: `allpaka bench --engine`, with pp512, tg128 and MTP skipped. Each configuration ran 3 times.
- **llama.cpp:** `third_party/llama.cpp`, CUDA build (`build-cuda`), built here. Command: `llama-bench -p 512 -n 128 -ngl 99 -r 3`.

| Model | Metric | allpaka default | allpaka `GGML=1 CUDA_GRAPH=1` | llama.cpp CUDA | allpaka / llama |
|---|---|---|---|---|---|
| Qwen2.5-7B Q4_K_M | pp512 tok/s | 251–254 | 14 475–14 613 | 14 581 ± 1 259 | ~100% |
| Qwen2.5-7B Q4_K_M | tg128 tok/s | 162–179 | 256.6–257.0 | 267.7 ± 1.6 | 96% |
| Qwen3-32B Q4_K_M | pp512 tok/s | 70–84 | 3 409–3 427 | 3 531 ± 159 | 97% |
| Qwen3-32B Q4_K_M | tg128 tok/s | 57.1 | 66.8–66.9 | 69.9 ± 0.4 | 96% |

Notes:
- **Fast path:** prefill and decode only get fast with `ALLPAKA_GGML=1 ALLPAKA_CUDA_GRAPH=1`. With these the ggml MMQ, flash attention and the whole-token CUDA graph are used.
- **Default is ~58× slower in prefill:** by default the ggml library loads, but prefill uses the dequant+cuBLAS path.
- **Results:**
  - Prefill: parity on 7B, 97% of llama.cpp on 32B.
  - Decode: 96% of llama.cpp on both models.
- **Not a quiet machine:** desktop applications were running during the runs.

## Update: new defaults

ggml kernels and CUDA graphs are now on by default whenever
`allpaka_ggml.dll` is present; `ALLPAKA_GGML=0` and `ALLPAKA_CUDA_GRAPH=0`
turn them off. Re-measured with no environment variables set:

| Model | prefill tok/s | decode tok/s |
|---|---|---|
| Qwen2.5-7B Q4_K_M | 14 169–14 479 | 255.5–256.4 |
| Qwen3-32B Q4_K_M | 3 389–3 404 | 66.8 |

With both set to 0, 7B gives 273 prefill / 180 decode, the same as the old
default.

## Interleaved run with the new defaults

allpaka (`89f110d`, no environment variables set) and `llama-bench`
(CUDA) took turns on the same model, 3 rounds each, pp512/tg128. Medians:

| Model | allpaka pp | llama pp | ratio | allpaka tg | llama tg | ratio |
|---|---|---|---|---|---|---|
| Qwen2.5-7B Q4_K_M | 14 144 | 13 269 | 107% | 256.3 | 267.1 | 96% |
| Qwen3-32B Q4_K_M | 3 391 | 3 362 | 101% | 66.9 | 69.8 | 96% |

## Decode kernel work: RMSNorm, argmax, own decode attention

allpaka `main` at `5c1921f`, includes:
- the register-resident RMSNorm+Q8;
- the 1024-thread argmax;
- the multi-block RMSNorm;
- the split-KV decode attention;
- the unrolled 5120-input matvecs;
- the qwen2 Q8-cache fix.

allpaka and `llama-bench` (CUDA) took turns on the same model, 3 rounds, pp512/tg128, no environment variables set.

| Round | allpaka pp | llama pp | allpaka tg | llama tg | tg ratio |
|---|---|---|---|---|---|
| 1 | 3 412 | 3 380 | 76.5 | 69.21 | 110.5% |
| 2 | 3 400 | 3 369 | 76.4 | 69.66 | 109.7% |
| 3 | 3 404 | 3 370 | 76.4 | 69.59 | 109.8% |

Qwen3-32B Q4_K_M:
- **Decode:** about 110% of llama.cpp (was 96%).
- **Prefill:** about 101%.

**Known problem:** prompts of about 1 200 tokens still give incoherent output through `serve`. This happens with `ALLPAKA_ATTN_G8=0` too, where the server crashed during prefill. Short prompts are coherent.

## After the prefill correctness fixes

The numbers above came from prefill that read wrong data:
- the flash-attention output layout was scrambled;
- the queue wait between cudarc and ggml was missing.

Prefill argmax was 8 then; correct is 201. Commits `3815e23` and `4af983e` fix this, together with a rebuilt `allpaka_ggml.dll`. With them, a 1 220-token prompt answers coherently through `serve`. Same interleaved setup:

| Round | allpaka pp | llama pp | pp ratio | allpaka tg | llama tg | tg ratio |
|---|---|---|---|---|---|---|
| 1 | 3 162 | 3 385 | 93% | 76.5 | 69.54 | 110.0% |
| 2 | 3 159 | 3 372 | 94% | 76.5 | 69.54 | 110.0% |
| 3 | 3 145 | 3 392 | 93% | 76.5 | 69.42 | 110.2% |

- **Decode:** about 110% of llama.cpp.
- **Prefill:** 93% of llama.cpp. The always-on queue wait costs it; the earlier 101% was measured on wrong results.

## Shared-stream prefill (`98ee3e3`), 2026-11-11

ggml in prefill now runs on the cudarc stream. The GPU was idle; same interleaved setup.

| Round | allpaka pp | llama pp | pp ratio | allpaka tg | llama tg | tg ratio |
|---|---|---|---|---|---|---|
| 1 | 3 485 | 3 379 | 103.1% | 76.7 | 69.27 | 110.7% |
| 2 | 3 482 | 3 359 | 103.6% | 76.7 | 70.27 | 109.1% |
| 3 | 3 475 | 3 372 | 103.0% | 76.7 | 70.21 | 109.2% |

Prefill argmax was 201 in every round, and logits-fnv was the same in every round.

## Fused prefill kernels + prefill CUDA graph (`5ee6299`), 2026-11-11

This build includes:
- Q/K norm+RoPE+K store fusion and the residual add folded into RMSNorm (`e2b6fe9`);
- flash attention writing the o_proj input directly (`d5dea0f`; needs the rebuilt `allpaka_ggml.dll`);
- prefill CUDA graph replay, fixed and turned on (`5ee6299`).

Same interleaved setup, 3 rounds. This session ran faster overall than the previous one, for llama.cpp as well, so compare only within a round.

| Round | allpaka pp | no graph pp | llama pp | pp ratio (no graph) | allpaka tg | llama tg | tg ratio |
|---|---|---|---|---|---|---|---|
| 1 | 3 847 | 3 728 | 3 477 | 110.6% (107.2%) | 79.8 | 72.64 | 109.9% |
| 2 | 3 796 | 3 724 | 3 535 | 107.4% (105.3%) | 79.8 | 72.56 | 110.0% |
| 3 | 3 808 | 3 761 | 3 404 | 111.9% (110.5%) | 79.8 | 72.94 | 109.4% |

The graph only replays when the chunk shape and KV base repeat, as they do in the bench. A chunked long prompt never replays, so the "no graph" column is closer to real prompts.

## With working PDL (`ee2730f`), 2026-11-13

PDL is now really active (579 of 579 programmatic edges in the decode graph) and on by default. Same interleaved setup, GPU idle, 3 rounds.

| Model | Round | allpaka pp | llama pp | pp ratio | allpaka tg | llama tg | tg ratio |
|---|---|---|---|---|---|---|---|
| Qwen3-32B Q4_K_M | 1 | 3 819 | 3 517 | 108.6% | 80.5 | 72.23 | 111.5% |
| Qwen3-32B Q4_K_M | 2 | 3 785 | 3 518 | 107.6% | 80.5 | 72.32 | 111.3% |
| Qwen3-32B Q4_K_M | 3 | 3 801 | 3 567 | 106.5% | 80.5 | 72.51 | 111.0% |
| Qwen2.5-7B Q4_K_M | 1 | 17 914 | 13 661 | 131% | 305.3 | 274.19 | 111.3% |
| Qwen2.5-7B Q4_K_M | 2 | 17 935 | 13 829 | 130% | 305.2 | 275.24 | 110.9% |
| Qwen2.5-7B Q4_K_M | 3 | 17 523 | 13 659 | 128% | 305.2 | 277.24 | 110.1% |

- **32B:** prefill argmax was 201 in all rounds, and 12 of 12 separate runs gave the same `tokens-fnv`.
- **7B prefill is not trustworthy yet:** its argmax differed in every round (93256, 94030, 18351), so the 7B prefill ratio does not count until that is explained.

**Known problem in `serve`:** the second short request in a row on one server returns garbage on both models (greedy, temperature 0). The first short request, long prompts, and a short request after a long one answer correctly.
