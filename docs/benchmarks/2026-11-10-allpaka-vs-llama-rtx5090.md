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
