# serve: two long requests taking turns, Qwen2.5-7B, 2026-11-14

`scripts/bench-two-long.py --addr 127.0.0.1:8099` (about 4 000 prompt tokens
and 512 new tokens each; B starts once A streams), RTX 5090, Q4_K_M.

| run | A first token | B first token | A / B median gap | A / B total |
|---|---|---|---|---|
| 1 | 846 ms | 736 ms | 8.07 / 8.07 ms | 5 668 / 4 857 ms |
| 2 | 21 ms | 735 ms | 8.12 / 8.13 ms | 4 862 / 4 873 ms |

- B no longer waits for all of A (about 4.9 s): its first token comes after
  its own prefill, 735 ms, while A keeps decoding.
- While they overlap each streams a token every 8.1 ms: the two take turns, one
  step each. A short chat alone streams at 3.5 ms per token, so two decoders at
  8.1 ms is some 15% under back-to-back throughput — the cost of switching
  sessions every token plus the longer context; no batching yet.
- `bench-concurrent-latency.py` (9 000-token heavy request, 1 024 new tokens):
  the chat's first token is 143–151 ms in both scenarios, largest gap 5–9 ms.
