# Qwen3-30B-A3B Q4_K_M: Allpaka / llama.cpp, 2026-10-24

Six balanced copy/shared/llama blocks, three measured repetitions per process, PP480/TG32, decode context 1, f16 KV, Metal. Fresh release example frozen before execution; identities and full protocol in raw/metadata.json.

| tok/s, median of process medians | copy | shared | llama |
|---|---:|---:|---:|
| prefill | 766.7 | 827.9 | 987.1 |
| decode | 97.1 | 100.1 | 116.0 |

## Qualification

All six blocks completed and are retained. Exact Allpaka output tokens matched across runs; final logits differed by less than 1e-4. Every checked decode round completed all 32 GPU steps without fallback.

**Performance comparison is not qualified as an isolated speedup/regression.** Strong downward drift occurred in both engines during the series. Concurrent rustc processes were observed (one at 715.8% CPU); pmset reported no thermal/performance warnings. No street_walk process was observed at periodic checks, but this does not establish exclusive GPU use. No unrelated processes were stopped.

Paired shared/copy differences (median ratios per block, bootstrap 95% interval):
- prefill: -0.98% [-29.52%, +7.99%].
- decode: -3.55% [-17.07%, +7.63%].

No end-to-end zero-copy gain is established. Cross-engine comparison_validated=false also applies because llama uses a different token stream and MoE expert routing. Raw logs, samples, commands, output hashes and binary/model identities are preserved under raw/. Source checkout had pre-existing modifications; this is a measurement of that frozen build, not clean HEAD.
