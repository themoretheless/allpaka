//! Immutable acceleration policy.
//!
//! Environment compatibility is confined to construction. Execution code
//! reads one policy snapshot, so effective settings cannot change depending
//! on which kernel happened to initialise its own `OnceLock` first.
//!
//! Flag conventions (keep the env spelling stable, it is user-facing):
//! - `on`    — default true, disabled by `=0`
//! - `off`   — default false, enabled by `=1`
//! - `set`   — default false, enabled by any value other than `=0`
//! - `present` — default false, enabled by mere presence
//! - `yes`   — default false, enabled by `=1` or `=true` (any case)
//! - `not_no` — default true, disabled by `=0` or `=false` (any case)

use std::sync::OnceLock;

#[inline]
fn on(_def: bool, name: &str) -> bool {
    std::env::var(name).map_or(true, |v| v != "0")
}
#[inline]
fn off(_def: bool, name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}
#[inline]
fn set(_def: bool, name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v != "0")
}
#[inline]
fn present(_def: bool, name: &str) -> bool {
    std::env::var_os(name).is_some()
}
#[inline]
fn yes(_def: bool, name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}
#[inline]
fn not_no(_def: bool, name: &str) -> bool {
    std::env::var(name).map_or(true, |v| !(v == "0" || v.eq_ignore_ascii_case("false")))
}

/// `ALLPAKA_MMQ`: opt-in int8 tensor-core MMQ variants for CUDA prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaMmq {
    Off,
    /// `=1`: NVRTC mma kernel.
    Mma,
    /// `=tile`: float shared-memory tile kernel.
    Tile,
    /// `=nvcc`: standalone PTX kernel.
    Nvcc,
}

macro_rules! flags {
    ($($field:ident : bool = $def:expr, env $env:literal, $parse:expr;)*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct RuntimePolicy {
            $(pub $field: bool,)*
            /// `ALLPAKA_ATTN_SPLIT=<n>`: split attention at n rows.
            pub attention_split: Option<usize>,
            /// `ALLPAKA_MV_TG` ∈ {64,128,256}, default 128.
            pub mv_tg: u64,
            /// `ALLPAKA_Q8_NR0` ∈ {1,2,4}, default 2.
            pub q8_nr0: usize,
            /// `ALLPAKA_NWARPS` ∈ {2,4,8}: force the CUDA MMVQ warp count.
            pub cuda_nwarps: Option<u8>,
            /// `ALLPAKA_MMQ`.
            pub cuda_mmq: CudaMmq,
            /// `ALLPAKA_MMID_RTILES=<n>`: pin the Metal grouped-FFN grid.
            pub metal_mmid_rtiles: Option<u64>,
        }

        impl Default for RuntimePolicy {
            #[inline]
            fn default() -> Self {
                Self {
                    $($field: $def,)*
                    attention_split: None,
                    mv_tg: 128,
                    q8_nr0: 2,
                    cuda_nwarps: None,
                    cuda_mmq: CudaMmq::Off,
                    metal_mmid_rtiles: None,
                }
            }
        }

        /// Every boolean switch: (variable, policy field, default, parse rule).
        pub const ENV_FLAGS: &[(&str, &str, bool, &str)] = &[
            $(($env, stringify!($field), $def, stringify!($parse)),)*
        ];

        impl RuntimePolicy {
            pub fn from_env() -> Self {
                Self {
                    $($field: ($parse)($def, $env),)*
                    attention_split: std::env::var("ALLPAKA_ATTN_SPLIT")
                        .ok()
                        .and_then(|v| v.parse().ok()),
                    mv_tg: std::env::var("ALLPAKA_MV_TG")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|n: &u64| matches!(n, 64 | 128 | 256))
                        .unwrap_or(128),
                    q8_nr0: std::env::var("ALLPAKA_Q8_NR0")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|n: &usize| [1, 2, 4].contains(n))
                        .unwrap_or(2),
                    cuda_nwarps: match std::env::var("ALLPAKA_NWARPS").as_deref() {
                        Ok("2") => Some(2),
                        Ok("4") => Some(4),
                        Ok("8") => Some(8),
                        _ => None,
                    },
                    cuda_mmq: match std::env::var("ALLPAKA_MMQ").as_deref() {
                        Ok("1") => CudaMmq::Mma,
                        Ok("tile") => CudaMmq::Tile,
                        Ok("nvcc") => CudaMmq::Nvcc,
                        _ => CudaMmq::Off,
                    },
                    metal_mmid_rtiles: std::env::var("ALLPAKA_MMID_RTILES")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok()),
                }
            }
        }
    };
}

flags! {
    normflag: bool = true, env "ALLPAKA_NORMFLAG", |_d, e| !std::env::var(e).is_ok_and(|v: String| v == "0");
    decode_serial: bool = false, env "ALLPAKA_DECODE_SERIAL", off;
    prefill_defer: bool = true, env "ALLPAKA_PF_DEFER", on;
    prefill_one_buffer: bool = true, env "ALLPAKA_PF_ONEBUF", on;
    gpu_route: bool = true, env "ALLPAKA_GPU_ROUTE", on;
    mm_pipeline: bool = true, env "ALLPAKA_MM_PIPE", on;
    dense_obuf: bool = true, env "ALLPAKA_PF_OBUF_DENSE", on;
    mv_id_grid: bool = true, env "ALLPAKA_MV_ID", on;
    q3_mv: bool = true, env "ALLPAKA_Q3_MV", on;
    q4_mv: bool = true, env "ALLPAKA_Q4_MV", on;
    q5_mv: bool = true, env "ALLPAKA_Q5_MV", on;
    q6_mv: bool = true, env "ALLPAKA_Q6_MV", on;
    q8_mv: bool = true, env "ALLPAKA_Q8_MV", on;
    qkv_mv: bool = false, env "ALLPAKA_QKV_MV", off;
    attend_mm: bool = true, env "ALLPAKA_ATTN_MM", on;
    mm_ll: bool = true, env "ALLPAKA_MM_LL", on;
    q2_mv: bool = false, env "ALLPAKA_Q2_MV", off;
    mega_debug: bool = false, env "ALLPAKA_MEGA_DEBUG", off;
    gpu_counters: bool = false, env "ALLPAKA_GPU_COUNTERS", off;
    cfuse: bool = false, env "ALLPAKA_CFUSE", off;
    gufuse: bool = false, env "ALLPAKA_DECODE_GUFUSE", off;
    rtopk_fused: bool = false, env "ALLPAKA_RTOPK", off;
    rfuse: bool = false, env "ALLPAKA_RFUSE", off;
    shared_tail: bool = false, env "ALLPAKA_SHARED_TAIL", off;
    shared_stagger: bool = false, env "ALLPAKA_SHARED_STAGGER", off;
    mm_k64: bool = false, env "ALLPAKA_MM_K64", set;
    mmid_k64: bool = false, env "ALLPAKA_MMID_K64", set;
    mmid_tiles: bool = true, env "ALLPAKA_MMID_TILES", on;
    pf_combine_resnorm: bool = true, env "ALLPAKA_PF_COMBINE_RESNORM", on;
    pf_fused_qk: bool = true, env "ALLPAKA_PF_FQK", on;
    no_stage_barrier: bool = false, env "ALLPAKA_NO_STAGE_BARRIER", present;
    no_barrier_probe: bool = false, env "ALLPAKA_NO_BARRIER", present;
    serial: bool = false, env "ALLPAKA_SERIAL", present;
    prefill_zero_copy: bool = true, env "ALLPAKA_PREFILL_ZERO_COPY", on;
    tokenbuf_debug: bool = false, env "ALLPAKA_TOKENBUF_DEBUG", present;
    decode_split: bool = false, env "ALLPAKA_DECODE_SPLIT", present;
    // On by default: measured on RTX 5090 (docs/benchmarks/2026-11-10-allpaka-vs-llama-rtx5090.md).
    cuda_graph: bool = true, env "ALLPAKA_CUDA_GRAPH", not_no;
    cuda_ggml: bool = true, env "ALLPAKA_GGML", not_no;
    cuda_no_fuse: bool = false, env "ALLPAKA_CUDA_NO_FUSE", present;
    // CUDA decode/prefill switches, formerly read from the environment on
    // every launch.
    cuda_q8: bool = true, env "ALLPAKA_Q8", not_no;
    cuda_mmvq_rows2: bool = false, env "ALLPAKA_MMVQ_ROWS", |_d, e| std::env::var(e).is_ok_and(|v: String| v == "2");
    cuda_out_n4: bool = true, env "ALLPAKA_OUT_N4", not_no;
    cuda_q6_f32: bool = false, env "ALLPAKA_Q6_F32", yes;
    cuda_k8192: bool = true, env "ALLPAKA_K8192", not_no;
    cuda_k5120: bool = false, env "ALLPAKA_K5120", yes;
    cuda_q6_k25600: bool = false, env "ALLPAKA_Q6_K25600", yes;
    cuda_ggml_mmvq: bool = false, env "ALLPAKA_GGML_MMVQ", yes;
    cuda_ggml_decode: bool = false, env "ALLPAKA_GGML_DECODE", yes;
    cuda_gu_q8: bool = false, env "ALLPAKA_GU_Q8", off;
    cuda_rms_mb: bool = true, env "ALLPAKA_RMS_MB", on;
    cuda_attn_g8: bool = true, env "ALLPAKA_ATTN_G8", on;
    cuda_gu_n8: bool = false, env "ALLPAKA_GU_N8", off;
    cuda_k5120_gu: bool = false, env "ALLPAKA_K5120_GU", yes;
    cuda_fuse_log: bool = false, env "ALLPAKA_FUSE_LOG", off;
    cuda_qkv_split: bool = false, env "ALLPAKA_QKV_SPLIT", yes;
    cuda_no_fa: bool = false, env "ALLPAKA_NO_FA", yes;
    cuda_fa_inline: bool = true, env "ALLPAKA_FA_INLINE", not_no;
    cuda_native_attend: bool = false, env "ALLPAKA_NATIVE_ATTEND", yes;
    cuda_hybrid_replay: bool = false, env "ALLPAKA_HYBRID_REPLAY", yes;
    cuda_wo_fa_q8: bool = false, env "ALLPAKA_WO_FA_Q8", yes;
    cuda_no_fa_graph: bool = false, env "ALLPAKA_NO_FA_GRAPH", yes;
    cuda_ggml_gemm: bool = true, env "ALLPAKA_GGML_GEMM", not_no;
    cuda_prefill_dual: bool = true, env "ALLPAKA_PREFILL_DUAL", not_no;
    cuda_pf_graph: bool = true, env "ALLPAKA_PF_GRAPH", not_no;
    cuda_gpu_embed: bool = false, env "ALLPAKA_GPU_EMBED", yes;
    gdn_zero: bool = false, env "ALLPAKA_GDN_ZERO", present;
    // Metal switches, formerly read from the environment on every call.
    // On Metal the GPU embed is on unless ALLPAKA_GPU_EMBED is 0/false.
    metal_gpu_embed_off: bool = false, env "ALLPAKA_GPU_EMBED", |_d, e| std::env::var(e).is_ok_and(|v: String| v == "0" || v.eq_ignore_ascii_case("false"));
    metal_dual: bool = false, env "ALLPAKA_DUAL", present;
    metal_no_route_fused: bool = false, env "ALLPAKA_NO_ROUTE_FUSED", present;
    metal_ffn_split: bool = false, env "ALLPAKA_FFN_SPLIT", present;
    metal_pf_split: bool = false, env "ALLPAKA_PF_SPLIT", present;
    metal_pf_chain: bool = false, env "ALLPAKA_PF_CHAIN", present;
    metal_ffn_time: bool = false, env "ALLPAKA_FFN_TIME", present;
    metal_q2_ilv: bool = false, env "ALLPAKA_Q2_ILV", present;
    metal_rows_debug: bool = false, env "ALLPAKA_ROWS_DEBUG", present;
    metal_verify_debug: bool = false, env "ALLPAKA_VERIFY_DEBUG", present;
    attn_zero: bool = false, env "ALLPAKA_ATTN_ZERO", present;
    moe_zero: bool = false, env "ALLPAKA_MOE_ZERO", present;
}

/// Switches that are not plain booleans: (variable, policy field, meaning).
pub const ENV_VALUES: &[(&str, &str, &str)] = &[
    (
        "ALLPAKA_ATTN_SPLIT",
        "attention_split",
        "split attention at n rows; unset = no split",
    ),
    (
        "ALLPAKA_MV_TG",
        "mv_tg",
        "matvec threadgroup size, one of 64/128/256; default 128",
    ),
    (
        "ALLPAKA_Q8_NR0",
        "q8_nr0",
        "Q8 matvec rows per SIMD group, one of 1/2/4; default 2",
    ),
    (
        "ALLPAKA_NWARPS",
        "cuda_nwarps",
        "force CUDA MMVQ warps to 2, 4 or 8; anything else = auto",
    ),
    (
        "ALLPAKA_MMQ",
        "cuda_mmq",
        "CUDA int8 MMQ prefill: 1 (NVRTC mma), tile, nvcc; default off",
    ),
    (
        "ALLPAKA_MMID_RTILES",
        "metal_mmid_rtiles",
        "pin the Metal grouped-FFN grid to n tiles",
    ),
];

fn parse_rule(rule: &str) -> &'static str {
    match rule {
        "on" => "on unless `=0`",
        "off" => "off unless `=1`",
        "set" => "on for any value except `0`",
        "present" => "on when set to anything",
        "yes" => "off unless `=1` or `=true`",
        "not_no" => "on unless `=0` or `=false`",
        _ => "custom, see `runtime.rs`",
    }
}

/// The reference table in docs/env-reference.md, generated from the policy.
pub fn env_reference_markdown() -> String {
    let mut out = String::from(
        "# ALLPAKA_* runtime switches\n\n\
         Generated from `crates/allpaka-backend/src/runtime.rs` by the\n\
         `env_reference_is_current` test; run it with `UPDATE_ENV_REFERENCE=1`\n\
         to rewrite this file. Every switch is read once per process.\n\n\
         | Variable | Policy field | Default | How it is read |\n\
         |---|---|---|---|\n",
    );
    let mut flags: Vec<_> = ENV_FLAGS.to_vec();
    flags.sort_by_key(|f| f.0);
    for (env, field, def, rule) in flags {
        out.push_str(&format!(
            "| `{env}` | `{field}` | {} | {} |\n",
            if def { "on" } else { "off" },
            parse_rule(rule)
        ));
    }
    out.push_str("\n| Variable | Policy field | Meaning |\n|---|---|---|\n");
    for (env, field, meaning) in ENV_VALUES {
        out.push_str(&format!("| `{env}` | `{field}` | {meaning} |\n"));
    }
    out
}

static POLICY: OnceLock<RuntimePolicy> = OnceLock::new();

#[inline]
pub fn cpu_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |t| t.get()))
}

#[inline]
pub fn get() -> &'static RuntimePolicy {
    POLICY.get_or_init(RuntimePolicy::from_env)
}

// Mirrors OnceLock::set: a refused policy is handed back to the caller.
#[allow(clippy::result_large_err)]
pub fn install(policy: RuntimePolicy) -> Result<(), RuntimePolicy> {
    POLICY.set(policy)
}

#[cfg(test)]
mod tests {
    use super::RuntimePolicy;

    /// docs/env-reference.md is generated from the policy; adding a switch
    /// without regenerating it fails here.
    #[test]
    fn env_reference_is_current() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/env-reference.md");
        let want = super::env_reference_markdown();
        if std::env::var_os("UPDATE_ENV_REFERENCE").is_some() {
            std::fs::write(path, &want).unwrap();
        }
        let have = std::fs::read_to_string(path)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        assert_eq!(
            have, want,
            "docs/env-reference.md is stale; rerun with UPDATE_ENV_REFERENCE=1"
        );
    }

    /// `yes` and `not_no` keep the spellings the CUDA paths accepted when
    /// they read the environment themselves.
    #[test]
    fn yes_and_not_no_accept_the_old_spellings() {
        let name = "ALLPAKA_TEST_POLICY_PARSE";
        for (value, yes, not_no) in [
            (None, false, true),
            (Some("1"), true, true),
            (Some("TRUE"), true, true),
            (Some("0"), false, false),
            (Some("False"), false, false),
            (Some("2"), false, true),
        ] {
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
            assert_eq!(super::yes(false, name), yes, "{value:?}");
            assert_eq!(super::not_no(true, name), not_no, "{value:?}");
        }
        std::env::remove_var(name);
    }

    #[test]
    fn production_defaults_are_explicit() {
        let p = RuntimePolicy::default();
        assert!(p.normflag && p.prefill_defer && p.prefill_one_buffer);
        assert!(p.gpu_route && p.mm_pipeline);
        assert!(p.q4_mv && p.q8_mv && p.attend_mm && p.mm_ll && p.mv_id_grid);
        assert!(!p.decode_serial && !p.q2_mv && !p.serial && !p.mm_k64);
        assert_eq!(p.attention_split, None);
        assert_eq!(p.mv_tg, 128);
        assert_eq!(p.q8_nr0, 2);
    }
}
