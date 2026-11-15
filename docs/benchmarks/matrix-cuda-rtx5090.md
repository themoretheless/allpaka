# Model matrix on CUDA (RTX 5090)

Models that run on allpaka's CUDA path with greedy parity against llama.cpp,
each with the artifact that shows it. llama.cpp is `97e4ca7`, built by
`third_party/build_allpaka_ggml.bat`, measured in the same session.

| Model | Arch | Parity | Prefill vs llama.cpp | Decode vs llama.cpp | Artifacts |
|---|---|---|---|---|---|
| Qwen3-32B Q4_K_M | `qwen3`, dense | greedy stream identical, 64 tokens after a 709-token prompt, 3 rounds | 106% (replayed stream), 103–112% (pp512–pp2048) | 118% (replay), 110% (`bench`) | `2026-11-14-token-replay-qwen3-32b.json`, `2026-11-10-allpaka-vs-llama-rtx5090.md`, `2026-11-13-cuda-prefill-determinism-rtx5090.md` |
| Qwen2.5-7B-Instruct Q4_K_M | `qwen2`, dense, q/k/v bias | `verify --decode 32` passes against CUDA llama-server; replayed greedy stream identical for 4 tokens, then a near-tie split | 132% (replay) | 114% (replay), 104–128% through `serve` | `2026-11-14-token-replay-qwen25-7b.md`, `2026-11-14-serve-decode-7b-rtx5090.md` |
| Llama-3.1-8B-Instruct Q4_K_M | `llama`, dense, NORM RoPE + `rope_freqs` | `verify --decode 32 --tol 0.3` passes on 3 prompts, argmax 96/96 | 110% (pp480 vs pp512) | 109% (`bench`) | `2026-11-14-verify-llama31-8b-rtx5090.md` |
| Gemma-3-12B-it Q4_K_M | `gemma3`, dense, GeGLU, sandwich norms, SWA 1024 | `verify --decode 32` passes on 3 short prompts and 1106/3011-token prompts past the window (argmax 160/160); 887-token prefill 0.53 (llama.cpp CUDA vs CPU: 0.44) | 90–95% (pp480 vs pp512), 107% (pp1232) | 103–105%, 102% at 1232 context | `2026-11-15-verify-gemma3-12b-rtx5090.md` |

## On `allpaka verify` for 32B

`verify`'s prefill tolerance (0.15 in log-prob over the reference's top 10) is
tighter than llama.cpp agrees with itself on Qwen3-32B: its CPU and CUDA
builds differ by up to 0.24 on `The capital of France is`, and two CPU runs
gave 12095 at -1.21 and -1.11. allpaka's CPU reference differs from llama.cpp
CUDA by 0.24, inside that spread; argmax agrees in every combination. Greedy
parity for 32B is therefore taken from the token replay, which compares the
decoded tokens themselves.

With the deeper-model default (0.3, `verify` since this note), Qwen3-32B passes
the prefill check. Its decode check does not: by the fourth step the largest
top-10 difference is 1.12 against a 0.48 decode tolerance, with argmax equal at
every step. llama.cpp's own CPU and CUDA builds differ by 0.78 at that step, on
a token near log-prob -7. Open as roadmap P8.
