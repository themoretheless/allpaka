# ALLPAKA_* runtime switches

Generated from `crates/allpaka-backend/src/runtime.rs` by the
`env_reference_is_current` test; run it with `UPDATE_ENV_REFERENCE=1`
to rewrite this file. Every switch is read once per process.

| Variable | Policy field | Default | How it is read |
|---|---|---|---|
| `ALLPAKA_ATTN_G8` | `cuda_attn_g8` | on | on unless `=0` |
| `ALLPAKA_ATTN_MM` | `attend_mm` | on | on unless `=0` |
| `ALLPAKA_ATTN_ZERO` | `attn_zero` | off | on when set to anything |
| `ALLPAKA_CFUSE` | `cfuse` | off | off unless `=1` |
| `ALLPAKA_CUDA_GRAPH` | `cuda_graph` | on | on unless `=0` or `=false` |
| `ALLPAKA_CUDA_NO_FUSE` | `cuda_no_fuse` | off | on when set to anything |
| `ALLPAKA_DECODE_GUFUSE` | `gufuse` | off | off unless `=1` |
| `ALLPAKA_DECODE_SERIAL` | `decode_serial` | off | off unless `=1` |
| `ALLPAKA_DECODE_SPLIT` | `decode_split` | off | on when set to anything |
| `ALLPAKA_DUAL` | `metal_dual` | off | on when set to anything |
| `ALLPAKA_FA_INLINE` | `cuda_fa_inline` | on | on unless `=0` or `=false` |
| `ALLPAKA_FFN_SPLIT` | `metal_ffn_split` | off | on when set to anything |
| `ALLPAKA_FFN_TIME` | `metal_ffn_time` | off | on when set to anything |
| `ALLPAKA_FUSE_LOG` | `cuda_fuse_log` | off | off unless `=1` |
| `ALLPAKA_GDN_ZERO` | `gdn_zero` | off | on when set to anything |
| `ALLPAKA_GGML` | `cuda_ggml` | on | on unless `=0` or `=false` |
| `ALLPAKA_GGML_DECODE` | `cuda_ggml_decode` | off | off unless `=1` or `=true` |
| `ALLPAKA_GGML_GEMM` | `cuda_ggml_gemm` | on | on unless `=0` or `=false` |
| `ALLPAKA_GGML_MMVQ` | `cuda_ggml_mmvq` | off | off unless `=1` or `=true` |
| `ALLPAKA_GPU_COUNTERS` | `gpu_counters` | off | off unless `=1` |
| `ALLPAKA_GPU_EMBED` | `cuda_gpu_embed` | off | off unless `=1` or `=true` |
| `ALLPAKA_GPU_EMBED` | `metal_gpu_embed_off` | off | custom, see `runtime.rs` |
| `ALLPAKA_GPU_ROUTE` | `gpu_route` | on | on unless `=0` |
| `ALLPAKA_GU_N8` | `cuda_gu_n8` | off | off unless `=1` |
| `ALLPAKA_GU_Q8` | `cuda_gu_q8` | off | off unless `=1` |
| `ALLPAKA_HYBRID_REPLAY` | `cuda_hybrid_replay` | off | off unless `=1` or `=true` |
| `ALLPAKA_K5120` | `cuda_k5120` | on | on unless `=0` or `=false` |
| `ALLPAKA_K5120_GU` | `cuda_k5120_gu` | off | off unless `=1` or `=true` |
| `ALLPAKA_K8192` | `cuda_k8192` | on | on unless `=0` or `=false` |
| `ALLPAKA_MEGA_DEBUG` | `mega_debug` | off | off unless `=1` |
| `ALLPAKA_MMID_K64` | `mmid_k64` | off | on for any value except `0` |
| `ALLPAKA_MMID_TILES` | `mmid_tiles` | on | on unless `=0` |
| `ALLPAKA_MMVQ_ROWS` | `cuda_mmvq_rows2` | off | custom, see `runtime.rs` |
| `ALLPAKA_MM_K64` | `mm_k64` | off | on for any value except `0` |
| `ALLPAKA_MM_LL` | `mm_ll` | on | on unless `=0` |
| `ALLPAKA_MM_PIPE` | `mm_pipeline` | on | on unless `=0` |
| `ALLPAKA_MOE_ZERO` | `moe_zero` | off | on when set to anything |
| `ALLPAKA_MV_ID` | `mv_id_grid` | on | on unless `=0` |
| `ALLPAKA_NATIVE_ATTEND` | `cuda_native_attend` | off | off unless `=1` or `=true` |
| `ALLPAKA_NORMFLAG` | `normflag` | on | custom, see `runtime.rs` |
| `ALLPAKA_NO_BARRIER` | `no_barrier_probe` | off | on when set to anything |
| `ALLPAKA_NO_FA` | `cuda_no_fa` | off | off unless `=1` or `=true` |
| `ALLPAKA_NO_FA_GRAPH` | `cuda_no_fa_graph` | off | off unless `=1` or `=true` |
| `ALLPAKA_NO_ROUTE_FUSED` | `metal_no_route_fused` | off | on when set to anything |
| `ALLPAKA_NO_STAGE_BARRIER` | `no_stage_barrier` | off | on when set to anything |
| `ALLPAKA_OUT_N4` | `cuda_out_n4` | on | on unless `=0` or `=false` |
| `ALLPAKA_PF_CHAIN` | `metal_pf_chain` | off | on when set to anything |
| `ALLPAKA_PF_COMBINE_RESNORM` | `pf_combine_resnorm` | on | on unless `=0` |
| `ALLPAKA_PF_DEFER` | `prefill_defer` | on | on unless `=0` |
| `ALLPAKA_PF_FQK` | `pf_fused_qk` | on | on unless `=0` |
| `ALLPAKA_PF_GRAPH` | `cuda_pf_graph` | on | on unless `=0` or `=false` |
| `ALLPAKA_PF_OBUF_DENSE` | `dense_obuf` | on | on unless `=0` |
| `ALLPAKA_PF_ONEBUF` | `prefill_one_buffer` | on | on unless `=0` |
| `ALLPAKA_PF_SPLIT` | `metal_pf_split` | off | on when set to anything |
| `ALLPAKA_PREFILL_DUAL` | `cuda_prefill_dual` | on | on unless `=0` or `=false` |
| `ALLPAKA_PREFILL_ZERO_COPY` | `prefill_zero_copy` | on | on unless `=0` |
| `ALLPAKA_Q2_ILV` | `metal_q2_ilv` | off | on when set to anything |
| `ALLPAKA_Q2_MV` | `q2_mv` | off | off unless `=1` |
| `ALLPAKA_Q3_MV` | `q3_mv` | on | on unless `=0` |
| `ALLPAKA_Q4_MV` | `q4_mv` | on | on unless `=0` |
| `ALLPAKA_Q5_MV` | `q5_mv` | on | on unless `=0` |
| `ALLPAKA_Q6_F32` | `cuda_q6_f32` | off | off unless `=1` or `=true` |
| `ALLPAKA_Q6_K25600` | `cuda_q6_k25600` | off | off unless `=1` or `=true` |
| `ALLPAKA_Q6_MV` | `q6_mv` | on | on unless `=0` |
| `ALLPAKA_Q8` | `cuda_q8` | on | on unless `=0` or `=false` |
| `ALLPAKA_Q8_MV` | `q8_mv` | on | on unless `=0` |
| `ALLPAKA_QKV_MV` | `qkv_mv` | off | off unless `=1` |
| `ALLPAKA_QKV_SPLIT` | `cuda_qkv_split` | off | off unless `=1` or `=true` |
| `ALLPAKA_RFUSE` | `rfuse` | off | off unless `=1` |
| `ALLPAKA_RMS_MB` | `cuda_rms_mb` | on | on unless `=0` |
| `ALLPAKA_ROWS_DEBUG` | `metal_rows_debug` | off | on when set to anything |
| `ALLPAKA_RTOPK` | `rtopk_fused` | off | off unless `=1` |
| `ALLPAKA_SERIAL` | `serial` | off | on when set to anything |
| `ALLPAKA_SHARED_STAGGER` | `shared_stagger` | off | off unless `=1` |
| `ALLPAKA_SHARED_TAIL` | `shared_tail` | off | off unless `=1` |
| `ALLPAKA_TOKENBUF_DEBUG` | `tokenbuf_debug` | off | on when set to anything |
| `ALLPAKA_VERIFY_DEBUG` | `metal_verify_debug` | off | on when set to anything |
| `ALLPAKA_WO_FA_Q8` | `cuda_wo_fa_q8` | off | off unless `=1` or `=true` |

| Variable | Policy field | Meaning |
|---|---|---|
| `ALLPAKA_ATTN_SPLIT` | `attention_split` | split attention at n rows; unset = no split |
| `ALLPAKA_MV_TG` | `mv_tg` | matvec threadgroup size, one of 64/128/256; default 128 |
| `ALLPAKA_Q8_NR0` | `q8_nr0` | Q8 matvec rows per SIMD group, one of 1/2/4; default 2 |
| `ALLPAKA_NWARPS` | `cuda_nwarps` | force CUDA MMVQ warps to 2, 4 or 8; anything else = auto |
| `ALLPAKA_MMQ` | `cuda_mmq` | CUDA int8 MMQ prefill: 1 (NVRTC mma), tile, nvcc; default off |
| `ALLPAKA_MMID_RTILES` | `metal_mmid_rtiles` | pin the Metal grouped-FFN grid to n tiles |
