# verify: Qwen3-32B, allpaka GPU path against llama.cpp CUDA, 2026-11-14

The two do not fit in 32 GiB side by side, so they ran one after the other:
`allpaka verify --record` against `llama-server -ngl 99` (llama.cpp at the
pinned revision, CUDA) wrote the reference
(`2026-11-14-verify-reference-qwen3-32b.json`: the default prompt, 5 tokens,
then 32 steps of the reference's own greedy continuation, top 10 log-probs at
each), and `allpaka verify --replay` scored the same contexts alone, on the
GPU and with `ALLPAKA_NO_GPU=1` on the CPU. Qwen3-32B Q4_K_M, RTX 5090.

| path | argmax agrees | median max-diff per position | max | positions over 0.48 |
|---|---|---|---|---|
| allpaka GPU | 33 / 33 | 0.164 | 0.850 (step 4) | 3 (steps 3, 4, 5) |
| allpaka CPU | 33 / 33 | 0.294 | 1.698 (step 4) | 6 |

Per position (max |log-prob diff| over the reference's top 10, the token it
falls on and that token's reference log-prob):

| pos | GPU | token | ref lp | CPU |
|---|---|---|---|---|
| 0 | 0.1238 | 458 | -3.892 | 0.2437 |
| 1 | 0.1689 | 58883 | -5.738 | 0.2597 |
| 2 | 0.2324 | 3555 | -1.591 | 0.4498 |
| 3 | 0.5326 | 653 | -5.730 | 1.3722 |
| 4 | 0.8498 | 856 | -8.348 | 1.6975 |
| 5 | 0.6026 | 9334 | -5.709 | 0.4773 |
| 6 | 0.3043 | 323 | -10.049 | 0.4492 |
| 7 | 0.3815 | 15948 | -3.849 | 0.4983 |
| 8 | 0.1638 | 6424 | -1.806 | 0.2311 |
| 9 | 0.2661 | 1298 | -8.592 | 0.3079 |
| 10 | 0.2245 | 323 | -6.966 | 0.2303 |
| 11 | 0.1591 | 220 | -4.375 | 0.1191 |
| 12 | 0.1640 | 1882 | -9.293 | 0.2275 |
| 13 | 0.1224 | 498 | -12.764 | 0.3919 |
| 14 | 0.0773 | 458 | -12.447 | 0.1614 |
| 15 | 0.1550 | 9625 | -13.161 | 0.4205 |
| 16 | 0.3940 | 3070 | -13.316 | 0.6262 |
| 17 | 0.1407 | 320 | -7.005 | 0.2869 |
| 18 | 0.0659 | 537 | -10.696 | 0.1809 |
| 19 | 0.1215 | 11 | -9.005 | 0.2041 |
| 20 | 0.0704 | 1096 | -3.714 | 0.2420 |
| 21 | 0.2695 | 17045 | -3.105 | 0.2830 |
| 22 | 0.1322 | 3881 | -4.730 | 0.0838 |
| 23 | 0.1613 | 304 | -9.591 | 0.2448 |
| 24 | 0.1528 | 264 | -4.497 | 0.2936 |
| 25 | 0.2407 | 10723 | -7.026 | 0.3955 |
| 26 | 0.1276 | 11 | -4.490 | 0.2060 |
| 27 | 0.1280 | 572 | -13.595 | 0.2659 |
| 28 | 0.1222 | 702 | -12.087 | 0.3458 |
| 29 | 0.2952 | 8426 | -15.270 | 0.7505 |
| 30 | 0.1077 | 315 | -14.024 | 0.3159 |
| 31 | 0.2092 | 13 | -10.576 | 0.3342 |
| 32 | 0.4180 | 19085 | -16.013 | 0.8078 |

- The GPU path agrees with llama.cpp CUDA about twice as closely as the CPU
  path does, and picks the same token at every position.
- Positions 3-5 stand out on both paths (GPU 0.53, 0.85, 0.60; CPU 1.37,
  1.70, 0.48), on tokens 5.7-8.3 nats below the top: the context there makes
  those tokens sensitive, the GPU path just less so. llama.cpp's own CPU and
  CUDA builds differ by 0.54 near this point (`docs/roadmap.md`, P8 history).
- With the default deep-dense decode tolerance (0.48) the replay fails at
  step 3; with 0.85 it passes. Not changing the default on one prompt.

## llama.cpp against itself, and the tolerance

llama-server reuses its cached prefix, so each recorded step is llama.cpp's
own decode path; with `ALLPAKA_VERIFY_NO_CACHE=1` the record recomputes the
whole context as one batch instead. On the same GPU and weights the two
llama.cpp paths disagree, and allpaka's GPU path sits inside that spread on
three prompts (33 positions each, argmax agreeing everywhere):

| prompt | llama.cpp decode vs batch (max / median) | allpaka GPU vs llama.cpp decode (max / median) |
|---|---|---|
| "The capital of France is" | 0.866 / 0.277 | 0.850 / 0.164 |
| "def fibonacci(n):" | 0.375 / 0.173 | 0.395 / 0.101 |
| "Photosynthesis is the process by which" | 0.292 / 0.196 | 0.375 / 0.140 |

At the end of the prose prompt llama.cpp's batch and token-by-token passes
differ by 0.32 (the code prompt: 0.06); allpaka's batch prefill there reads
0.375 against llama.cpp's batch, 0.287 when fed token by token.

So `verify` now defaults, for dense models deeper than 32 layers, to 0.4 at
the prefill position and 0.9 per decode step; all three recorded references
(`2026-11-14-verify-reference-qwen3-32b*.json`) pass `verify --replay` with
the defaults.
