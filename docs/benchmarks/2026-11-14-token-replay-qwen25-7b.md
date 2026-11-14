# Token replay: allpaka against llama.cpp on one stream, Qwen2.5-7B, 2026-11-14

`scripts/bench-token-replay.py` tokenizes the prompt once with llama-server
and feeds the same 709 ids to both engines: `allpaka bench --engine` through
`ALLPAKA_BENCH_TOKENS`, llama-server as a token array on `/completion`. Both
prefill into an empty cache and decode 128 tokens greedily from the end of the
prompt. Three rounds, engines alternating, a fresh process each time.
RTX 5090, llama.cpp `97e4ca7`. Full report:
`2026-11-14-token-replay-qwen25-7b.json` (schema `allpaka.token-replay/1`).

| Round | allpaka prefill | llama prefill | allpaka decode | llama decode |
|---|---|---|---|---|
| 1 | 15 688 | 11 784 | 256.5 | 224.4 |
| 2 | 15 156 | 12 599 | 254.5 | 238.9 |
| 3 | 15 575 | 11 670 | 257.2 | 219.4 |

- **Prefill:** median 132% of llama.cpp, on the identical token stream.
- **Decode:** median 114%, from the same 709-token context.
- **Comparability:** `prompt_identical: true`, `decode_identical: false`.
  The engines agree on the first four generated tokens and choose differently
  at the fifth (allpaka 18626, llama.cpp 3234), the same way in every round.
  From there they decode different text of the same length, so the decode
  rates measure similar but not identical work; the report says
  `comparable: false` rather than pretend otherwise.
- Qwen2.5-7B is dense, so the MoE routing caveat does not apply here; it is
  recorded in the report for runs on MoE models.
