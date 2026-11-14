# Model matrix on CUDA (RTX 5090)

Models that run on allpaka's CUDA path with greedy parity against llama.cpp,
each with the artifact that shows it. llama.cpp is `97e4ca7`, built by
`third_party/build_allpaka_ggml.bat`, measured in the same session.

| Model | Arch | Parity | Prefill vs llama.cpp | Decode vs llama.cpp | Artifacts |
|---|---|---|---|---|---|
| Qwen3-32B Q4_K_M | `qwen3`, dense | greedy stream identical, 64 tokens after a 709-token prompt, 3 rounds | 106% (replayed stream), 103–112% (pp512–pp2048) | 118% (replay), 110% (`bench`) | `2026-11-14-token-replay-qwen3-32b.json`, `2026-11-10-allpaka-vs-llama-rtx5090.md`, `2026-11-13-cuda-prefill-determinism-rtx5090.md` |
| Qwen2.5-7B-Instruct Q4_K_M | `qwen2`, dense, q/k/v bias | `verify --decode 32` passes against CUDA llama-server; replayed greedy stream identical for 4 tokens, then a near-tie split | 132% (replay) | 114% (replay), 104–128% through `serve` | `2026-11-14-token-replay-qwen25-7b.md`, `2026-11-14-serve-decode-7b-rtx5090.md` |

## On `allpaka verify` for 32B

`verify`'s prefill tolerance (0.15 in log-prob over the reference's top 10) is
tighter than llama.cpp agrees with itself on Qwen3-32B: its CPU and CUDA
builds differ by up to 0.24 on `The capital of France is`, and two CPU runs
gave 12095 at -1.21 and -1.11. allpaka's CPU reference differs from llama.cpp
CUDA by 0.24, inside that spread; argmax agrees in every combination. Greedy
parity for 32B is therefore taken from the token replay, which compares the
decoded tokens themselves.
