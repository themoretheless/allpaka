# Decode optimization notes (Q5_K / Q8_0 / SWFUSE / MEGA)

Target: close the remaining GLM-4.5-Air decode gap (~36–37 vs llama ~41–43
tok/s) and keep Qwen MoE wins from regressing.

## Landed in this branch (needs Mac A/B)

| Lever | Default | Opt-out | Why |
| --- | --- | --- | --- |
| `matvec_q5_k_mv` | ON | `ALLPAKA_Q5_MV=0` | llama-structure Q5_K matvec + `SWIGLU_X`; GLM shared gate/up, qwen35 expert down |
| `matvec_q8_0_mv` | ON | `ALLPAKA_Q8_MV=0` | llama-structure Q8_0 matvec + `SWIGLU_X`; GLM down, qwen35 projections |
| decode SwiGLU fuse | ON | `ALLPAKA_SWFUSE=0` | drops standalone swiglu + one barrier per MoE layer (now covers Q5_K down) |
| decode `attend_mv` | ON | `ALLPAKA_ATTN_MV=0` | four-position-in-flight flash-attn vec path |
| plain `mmll_q5_0` | ON (via `MM_LL`) | — | GLM / dense Q5_0 prefill mm (indexed `mmll_id_q5_0` already existed) |
| `moe_ffn_mega` | **OFF (code)** | — | disabled; prior multi-TG atomic spin froze Macs — no env re-enable |

Microbench harnesses (Mac only):

```sh
cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture
cargo test -p allpaka-backend --test gpu_q35_mvbench -- --ignored --nocapture
# A/B matvec paths (do NOT set ALLPAKA_MEGA — it is disabled and unsafe to revive casually):
ALLPAKA_Q5_MV=0 ALLPAKA_Q8_MV=0 cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture
ALLPAKA_ATTN_MV=0 cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture
```

Paired end-to-end vs llama.cpp (rbench AB/BA process pairs):

```sh
cargo build --release -p allpaka-cli
./target/release/allpaka rbench models/glm-4.5-air-Q4_K_M.gguf --pp 480 --tg 32 --repeats 5
# or:
scripts/rbench-vs-llama.sh models/glm-4.5-air-Q4_K_M.gguf
scripts/bench-matrix.sh \
  models/qwen3-30b-a3b-Q4_K_M.gguf \
  models/qwen3.5-35b-a3b-Q4_K_M.gguf \
  models/glm-4.5-air-Q4_K_M.gguf
```

Artifacts land in `.rbench/llama-compare-*` (`run.json`, `comparison.json`, `raw/`).
Legacy shell A/B: `scripts/bench-compare.sh`.

rbench defaults: 1 discarded warmup pair, 1.5s cooldown between engines,
`ALLPAKA_BENCH_SKIP_MTP=1`, `ALLPAKA_PROFILE=max-performance` if unset.

## Still open

- In-model mm TFLOPS gap (peak matches llama; in-graph effective still lower).
- MTP economics on MoE (verify traffic scales with `m`).
- **MEGA / `moe_ffn_mega` — disabled in code until redesign.** Root cause
  (M4 Max): in-kernel `mega_sync` used a **device `atomic_uint` busy-wait**
  across threadgroups, soft-locking the GPU scheduler and freezing the Mac.
  The spin path is removed and `mega_enabled()` always returns false (env
  cannot turn it back on). Do not revive casually; need encoder-barrier
  stage splits or equivalent + a test that cannot wedge the GPU.
- **Decode without MEGA:** qwen3-30b ≈ 698 dispatches/tok; GLM-Air ≈ 803/tok
  (shared expert). Safe next levers are encoder-side only (already have
  `resnorm_router`, `combine_resnorm`, SWFUSE, indexed matvecs). Do **not**
  re-enable `ALLPAKA_DECODE_GUFUSE` / multi-TG device sync without a
  Mac-safe harness. Judge progress on cool-machine sustained A/B, not hot
  process-paired rbench.
- Absolute tok/s drift with thermal state (both engines); prefer Δ% after
  warmup. GLM 2026-09-12 warm: prefill ≈ llama parity, decode ~0.82×
  (`docs/benchmarks/glm45-air-m4-max-pending.md`).
- Confirm Q5/Q8 `_mv` + SWFUSE + ATTN_MV on qwen35: **done** (warm 2026-09-12,
  `docs/benchmarks/qwen35-35b-m4-max-pending.md` — no overrides, GPU 32/32).
