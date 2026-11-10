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
