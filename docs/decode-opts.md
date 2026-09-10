# Decode optimization notes (Q5_K / Q8_0 / SWFUSE / MEGA)

Target: close the remaining GLM-4.5-Air decode gap (~36–37 vs llama ~41–43
tok/s) and keep Qwen MoE wins from regressing.

## Landed in this branch (needs Mac A/B)

| Lever | Default | Opt-out | Why |
| --- | --- | --- | --- |
| `matvec_q5_k_mv` | ON | `ALLPAKA_Q5_MV=0` | llama-structure Q5_K matvec; GLM shared gate/up |
| `matvec_q8_0_mv` | ON | `ALLPAKA_Q8_MV=0` | llama-structure Q8_0 matvec + `SWIGLU_X`; GLM down, qwen35 projections |
| decode SwiGLU fuse | ON | `ALLPAKA_SWFUSE=0` | drops standalone swiglu + one barrier per MoE layer |
| `moe_ffn_mega` expert cap | 256 | still needs `ALLPAKA_MEGA=1` | unlocks qwen35moe (256 experts); GLM still blocked while `shared_gate` is set |

Microbench harnesses (Mac only):

```sh
cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture
cargo test -p allpaka-backend --test gpu_q35_mvbench -- --ignored --nocapture
# A/B the new kernels:
ALLPAKA_Q5_MV=0 ALLPAKA_Q8_MV=0 cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture
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
- MEGA + GLM shared-gate path (outer gate still requires `shared_gate.is_none()`).
- MTP economics on MoE (verify traffic scales with `m`).
- Confirm Q5/Q8 `_mv` and default SWFUSE on M4 Max with greedy parity.
