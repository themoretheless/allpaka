# Autotune by workload on RTX 5090, 2026-11-14

**Result:** `allpaka autotune --workload` measures, chooses and caches per
workload as designed, but on CUDA there is nothing for it to choose between:
the three profiles run at the same rates and hold the same memory. The
selections below are decided by noise.

## Setup

- RTX 5090, driver 621.01, Windows 11, Qwen2.5-7B-Instruct Q4_K_M.
- `allpaka autotune <gguf> --cache <fresh file> --workload W`, release build
  of the commit that adds this file. Each workload measures the three
  profiles in fresh processes: `allpaka bench --engine`, prompt 128 tokens,
  512 for `throughput`.
- The card was idle before each workload. No llama.cpp reference: the runs
  compare allpaka profiles with each other, not engines.

## Measurements

| Workload | Profile | decode tok/s | prefill tok/s | request tok/s | working memory |
|---|---|---|---|---|---|
| latency | balanced | 305.5 | 9 984 | 2 209.0 | 0.06 GiB |
| latency | max-performance | 306.2 | 9 876 | 2 207.9 | 0.06 GiB |
| latency | safe | 305.5 | 9 933 | 2 206.6 | 0.06 GiB |
| throughput | balanced | 306.0 | 17 850 | 2 422.0 | 0.21 GiB |
| throughput | max-performance | 305.8 | 17 322 | 2 411.9 | 0.21 GiB |
| throughput | safe | 306.0 | 17 632 | 2 418.1 | 0.21 GiB |
| memory | balanced | 306.1 | 9 951 | 2 210.8 | 0.06 GiB |
| memory | max-performance | 306.0 | 9 908 | 2 208.3 | 0.06 GiB |
| memory | safe | 306.2 | 9 942 | 2 210.8 | 0.06 GiB |

"request" is the rate over a 2 048-token prompt plus a 256-token answer at the
measured prefill and decode rates. "working memory" is the backend's scratch
arenas after the run.

## Selections

| Workload | Selected | Recorded objective | Why |
|---|---|---|---|
| latency | max-performance | 306.2 tok/s | highest decode, by 0.2% |
| throughput | balanced | 2 422.0 tok/s | highest request rate, by 0.2% |
| memory | balanced | 306.1 tok/s | all within 10% of the fastest, all 0.06 GiB; first measured wins the tie |

- **Mechanism:** the cache holds one entry per workload, keyed by
  `NVIDIA GeForce RTX 5090` and driver `621.01`; a second `--workload
  throughput` run is a cache hit.
- **Profiles on CUDA:** decode differs by at most 0.2% and prefill by at most
  3% between profiles, inside run-to-run noise. The profiles switch Metal
  kernel options; none of them changes a CUDA path.
- **Working memory:** was reported as 0 on CUDA before this commit
  (`gpu::arena_bytes` was a stub); it now counts the scratch arenas.
