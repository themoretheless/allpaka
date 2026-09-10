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
| `moe_ffn_mega` | opt-in | `ALLPAKA_MEGA=1` | expert cap 256; `has_shared=2` unlocks GLM / qwen35 gated shared expert |

Microbench harnesses (Mac only):

```sh
cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture
cargo test -p allpaka-backend --test gpu_q35_mvbench -- --ignored --nocapture
# A/B the new kernels:
ALLPAKA_Q5_MV=0 ALLPAKA_Q8_MV=0 cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture
ALLPAKA_ATTN_MV=0 ALLPAKA_MEGA=1 cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture
```

Paired end-to-end vs llama.cpp:

```sh
scripts/bench-compare.sh models/glm-4.5-air-Q4_K_M.gguf
scripts/bench-matrix.sh \
  models/qwen3-30b-a3b-Q4_K_M.gguf \
  models/qwen3.5-35b-a3b-Q4_K_M.gguf \
  models/glm-4.5-air-Q4_K_M.gguf
```

## Still open

- In-model mm TFLOPS gap (peak matches llama; in-graph effective still lower).
- MTP economics on MoE (verify traffic scales with `m`).
- Confirm Q5/Q8 `_mv`, default SWFUSE, default ATTN_MV, and MEGA+shared_gate
  on M4 Max with greedy parity (`ALLPAKA_MEGA=1` for GLM / qwen35).
