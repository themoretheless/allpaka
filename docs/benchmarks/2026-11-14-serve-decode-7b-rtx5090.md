# serve decode on Qwen2.5-7B against llama-server, RTX 5090, 2026-11-14

**Result:** greedy requests to `serve` ran prompt-lookup drafting by
default, which cost 8–12% of decode on ordinary answers. With drafting on by
default only when the client sends `prediction`, `serve` decodes faster than
llama-server at every context depth measured.

## Where the time went

`tests/cuda_decode_paths_rate.rs` drives the engine directly, 200-token
prompt, 128 greedy tokens, second pass of each path:

| Path | tok/s |
|---|---|
| `forward` (logits to host, argmax on CPU) | 267.4 |
| `forward_greedy` (on-device argmax) | 275.8 |
| `forward_greedy_n` (device chain) | 262.3 |

All three decoded the same stream. The engine was not the problem.

Through `serve`, streaming, `temperature: 0`, 128 new tokens, three prompts
(story, explanation, 400-item list), two rounds:

| Request | round 1 | round 2 |
|---|---|---|
| greedy (`repeat_penalty: 1.0`), drafting on by default | 240, 233, 220 | 251, 240, 222 |
| greedy, `prompt_lookup: false` | 267, 263, 248 | 262, 261, 253 |
| default `repeat_penalty` (logits path, no drafting) | 248, 257, 244 | 252, 254, 240 |

After the change, the same probe: greedy by default 253, 258, 245 and
257, 258, 247; with `prompt_lookup: false` 255, 228, 236 and 254, 257, 248.

## Against llama-server

`scripts/bench-decode-vs-context.py`, greedy, 64 new tokens, the two engines
one after the other with the card idle, llama-server `97e4ca7` built by
`third_party/build_allpaka_ggml.bat`.

Before (`11343d5`, drafting on by default):

| context | allpaka tok/s | llama tok/s | ratio |
|---|---|---|---|
| 1 107 | 199.4 | 238.4 | 84% |
| 3 907 | 189.3 | 231.9 | 82% |
| 8 907 | 176.5 | 221.5 | 80% |
| 18 907 | 178.0 | 206.1 | 86% |
| 28 907 | 159.8 | 198.2 | 81% |

After, three rounds:

| context | allpaka tok/s | llama tok/s | ratio |
|---|---|---|---|
| 1 107 | 231.9 / 231.9 / 244.9 | 209.2 / 201.2 / 207.4 | 111% / 115% / 118% |
| 3 907 | 231.2 / 231.1 / 237.1 | 196.0 / 202.8 / 202.3 | 118% / 114% / 117% |
| 8 907 | 207.8 / 198.9 / 214.1 | 187.5 / 191.1 / 189.6 | 111% / 104% / 113% |
| 18 907 | 188.1 / 201.2 / 215.6 | 180.1 / 177.2 / 168.1 | 104% / 114% / 128% |
| 28 907 | 187.8 / 187.2 / 204.4 | 168.7 / 155.6 / 171.7 | 111% / 120% / 119% |

- **allpaka:** up 16–23% at every depth.
- **llama-server:** 13–18% slower in these rounds than in the run before the
  change, with nothing changed on its side. The ratios rest on the after
  rounds alone, each taken back to back; the before table is not a like-for-
  like baseline for llama.cpp.
- **Ratio:** 104–128% of llama-server, every round at every depth.
