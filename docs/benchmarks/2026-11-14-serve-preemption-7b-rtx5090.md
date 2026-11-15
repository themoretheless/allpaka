# serve: a short chat between the steps of a long request, Qwen2.5-7B, 2026-11-14

`scripts/bench-concurrent-latency.py` against `allpaka serve`, RTX 5090. The
heavy request is about 8 900 or 28 900 prompt tokens and is asked for a long
answer (1 024 new tokens, `--heavy-instruction`), so it keeps decoding; the
chat is short and streamed. "before" is the build without step preemption,
"after" the commit that adds this file.

| heavy prompt | scenario | before: chat first token | after: chat first token | before: heavy total | after: heavy total |
|---|---|---|---|---|---|
| 8 920 | heavy-first | 3 992 ms | 134 ms | 4 077 ms | 5 534 ms |
| 8 920 | heavy-first | 3 978 ms | 149 ms | 4 071 ms | 5 531 ms |
| 28 920 | heavy-first | 4 733 ms | 136 ms | 4 795 ms | 6 220 ms |
| 8 920 | chat-first | 146 / 20 ms | 145 / 134 ms | 6 216 / 6 086 ms | 6 271 / 6 119 ms |
| 28 920 | chat-first | 21 ms | 125 ms | 9 157 ms | 9 210 ms |

- **heavy-first:** the chat's first token waits for one step of the heavy
  request (a prefill chunk or a decode token) instead of all of it: 3.98–4.73 s
  down to 134–149 ms. The largest gap between chat tokens stays at 5–6 ms.
- **The heavy request** finishes later by about the chat's own run (1.4 s for
  365 chat tokens): it is paused, not slowed.
- **chat-first:** a long request arriving while a chat runs is not served in
  between (it is not four times shorter), so the chat streams undisturbed.
  The chat's first token is 125–145 ms instead of 20 ms in later rounds
  because the previous round's chat was served in a temporary session and left
  no prefix to reuse, not because of slower prefill.
- **Single requests:** decode rate unchanged, four paired rounds through
  `serve` (greedy 278–300 tok/s both builds; logits path 260–286).
