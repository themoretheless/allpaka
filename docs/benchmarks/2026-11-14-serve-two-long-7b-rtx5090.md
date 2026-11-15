# serve: long requests taking turns, Qwen2.5-7B, 2026-11-14

`scripts/bench-two-long.py --addr 127.0.0.1:8099` (about 4 000 prompt tokens
and 512 new tokens each; B starts once A streams), RTX 5090, Q4_K_M.

| run | A first token | B first token | A / B median gap | A / B total |
|---|---|---|---|---|
| 1 | 846 ms | 736 ms | 8.07 / 8.07 ms | 5 668 / 4 857 ms |
| 2 | 21 ms | 735 ms | 8.12 / 8.13 ms | 4 862 / 4 873 ms |

- B no longer waits for all of A (about 4.9 s): its first token comes after
  its own prefill, 735 ms, while A keeps decoding.
- While they overlap each streams a token every 8.1 ms: the two take turns, one
  step each.

More at once (`--count N`, same server, same sizes):

| requests | median gap each | later first tokens |
|---|---|---|
| 1 | 3.96 ms | — |
| 3 | 12.24 ms | 769, 789 ms |
| 4 | 16.19 ms | 735, 829, 781 ms |
| 5 | 16.19 ms for four; the fifth 4.0 ms | the fifth 8 699 ms (queued: at most four take turns) |

N generations at N × 4.0–4.1 ms per token each: switching sessions every token
costs 2–3% against running them back to back. There is no batching yet, so
the total rate stays at that of a single request.
- `bench-concurrent-latency.py` (9 000-token heavy request, 1 024 new tokens):
  the chat's first token is 143–151 ms in both scenarios, largest gap 5–9 ms.

## With batched decode steps

The same benchmark after decode steps of concurrent generations were batched
into one pass over the weights (`Model::forward_multi`, captured CUDA graph).
The GPU carried a steady 9-10% load from elsewhere during these runs.

| requests | median gap each, taking turns | median gap each, batched |
|---|---|---|
| 1 | 3.96 ms | 4.42-4.80 ms |
| 2 | 8.1 ms | 7.84 ms |
| 4 | 16.19 ms | 10.8-12.2 ms |

Four requests finish in about two thirds of the time. The batched step alone
(`tests/cuda_decode_multi_parity.rs`, no HTTP) takes 7.2 ms for four sessions,
1.79 ms per token against 3.67 ms one at a time; the rest of the gap in
`serve` is per-token work outside the step (sampling, streaming, the turn
itself). Single-session decode is unchanged: `allpaka bench` 256.7 tok/s
against llama.cpp `tg128` 242.7 tok/s in the same session.
