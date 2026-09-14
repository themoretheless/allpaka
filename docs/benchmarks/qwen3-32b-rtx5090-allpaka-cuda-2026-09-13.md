# Qwen3-32B Q4_K_M on RTX 5090 (CUDA)

Measured on 2026-09-13 on Windows with an NVIDIA GeForce RTX 5090
(32 GB, compute capability 12.0). Same GGUF file as the earlier
llama.cpp CUDA baseline on this machine.

| Workload | llama.cpp CUDA | allpaka CUDA | notes |
| --- | ---: | ---: | --- |
| `pp480` prefill | 2859.6 tok/s | 15.2 tok/s | allpaka path is host-orchestrated; many H2D/D2H syncs |
| `tg32` decode (ctx≈0) | 66.1 tok/s | 3.8 tok/s | fail-closed GPU coverage: 32/32 successes, 0 declines |

## Method

### allpaka

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:PATH"
$env:ALLPAKA_BENCH_PP = "480"
$env:ALLPAKA_BENCH_TG = "32"
$env:ALLPAKA_BENCH_REPORT = "docs/benchmarks/qwen3-32b-rtx5090-allpaka-cuda-2026-09-13.json"
cargo build --release -p allpaka-cli
allpaka bench --engine path\to\Qwen3-32B-Q4_K_M.gguf
```

Report metadata: `device=cuda`, decode fast-path attempts=successes=32.

### llama.cpp

Official `b10944` Windows CUDA 13.3 build, `-ngl 99 -r 5 -ctk f16 -ctv f16`:

See [qwen3-32b-rtx5090-cuda-2026-09-13.json](qwen3-32b-rtx5090-cuda-2026-09-13.json).

## Notes

- allpaka CUDA implements the full Metal public GPU API via NVRTC custom
  kernels plus cuBLASLt-backed prefill GEMM (dequant to f16 workspace).
- Weights are copied into VRAM at `attach` (18.4 GiB for this model).
- Current numbers are a correctness / fail-closed baseline, not a throughput
  target. Closing the gap means fewer host syncs (fused decode command stream
  like Metal) and less CPU-side orchestration in the token loop.
- Set `ALLPAKA_NO_GPU=1` to force the CPU path. CUDA Toolkit 13.x DLLs live
  under `CUDA_PATH\bin\x64` on this Windows layout.
