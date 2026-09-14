# Qwen3-32B Q4_K_M on RTX 5090 (CUDA, fused decode + device prefill)

Measured on 2026-09-14 on Windows with an NVIDIA GeForce RTX 5090
(32 GB, compute capability 12.0). Same GGUF as the llama.cpp CUDA baseline.

Success bar (product): **>5% faster than llama.cpp** on every model and
metric (`pp*` and `tg*`). This run does **not** meet that bar yet.

| Workload | llama.cpp CUDA | allpaka latest |
| --- | ---: | ---: |
| `pp480` prefill | 2859.6 tok/s | **1407.0 tok/s** (target >3002) |
| `tg32` decode | 66.1 tok/s | **14.4 tok/s** (target >69.4) |

llama +5% targets: `pp480 > 3002.6`, `tg32 > 69.4`.

## What changed

- Dense `decode_token`: one CUDA stream, device-resident activations, **one**
  synchronize per token. Fail-closed: attempts=32, successes=32.
- Dense fused **prefill**: attn + FFN stay on device; sync mainly at
  `prefill_end`.
- Coalesced Q4_K / Q5_K / Q6_K dequant (warp-lane stores, 2D grid over
  superblocks) plus persistent RMSNorm scratch (no per-layer `cudaMalloc`).
  Prefill **542.7 → 1406.6 tok/s**.

Still ~2.1× short on prefill (need ggml-class MMQ / skip full f16 W) and
~4.5× on decode (matvec + launch tax). CUDA graphs remain opt-in
(`ALLPAKA_CUDA_GRAPH=1`); default is off after a capture regression.

## Method

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:PATH"
$env:ALLPAKA_BENCH_PP = "480"
$env:ALLPAKA_BENCH_TG = "32"
$env:ALLPAKA_BENCH_REPORT = "docs/benchmarks/qwen3-32b-rtx5090-allpaka-cuda-fused-2026-09-14.json"
cargo build --release -p allpaka-cli
allpaka bench --engine path\to\Qwen3-32B-Q4_K_M.gguf
```

Escape hatch: `ALLPAKA_CUDA_NO_FUSE=1` forces the old host-orchestrated decode.
