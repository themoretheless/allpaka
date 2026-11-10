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

macro_rules! flags {
    ($($field:ident : bool = $def:expr, env $env:literal, $parse:expr;)*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct RuntimePolicy {
            $(pub $field: bool,)*
            /// `ALLPAKA_ATTN_SPLIT=<n>`: split attention at n rows.
            pub attention_split: Option<usize>,
            /// `ALLPAKA_ATTN_T4=<n>`: prefill non-MM attention row tiling (default 8).
            pub attn_t4: usize,
            /// `ALLPAKA_ATTN_S8=<n>`: decode non-MV attention simdgroups (default 32).
            pub attn_s8: usize,
            /// `ALLPAKA_MV_TG` ∈ {64,128,256}, default 128.
            pub mv_tg: u64,
            /// `ALLPAKA_Q2_NR0` ∈ {1,2,4}, default 4.
            pub q2_nr0: usize,
            /// `ALLPAKA_Q4_NR0` ∈ {1,2,4}, default 2.
            pub q4_nr0: usize,
            /// `ALLPAKA_Q5_NR0` ∈ {1,2,4}, default 2.
            pub q5_nr0: usize,
            /// `ALLPAKA_Q8_NR0` ∈ {1,2,4}, default 2.
            pub q8_nr0: usize,
            /// `ALLPAKA_LPR_DIV` power of two, default 1.
            pub lpr_div: usize,
            /// `ALLPAKA_MMID_BN16` explicit override (`=1` -> `Some(true)`, `=0` -> `Some(false)`).
            pub mmid_bn16: Option<bool>,
            /// `ALLPAKA_MMID_RTILES=<n>`: override grouped MoE row-tile grid count.
            pub mmid_rtiles: Option<u64>,
        }

        impl Default for RuntimePolicy {
            #[inline]
            fn default() -> Self {
                Self {
                    $($field: $def,)*
                    attention_split: None,
                    attn_t4: 8,
                    attn_s8: 32,
                    mv_tg: 128,
                    q2_nr0: 4,
                    q4_nr0: 2,
                    q5_nr0: 2,
                    q8_nr0: 2,
                    lpr_div: 1,
                    mmid_bn16: None,
                    mmid_rtiles: None,
                }
            }
        }

        impl RuntimePolicy {
            pub fn from_env() -> Self {
                Self {
                    $($field: ($parse)($def, $env),)*
                    attention_split: std::env::var("ALLPAKA_ATTN_SPLIT")
                        .ok()
                        .and_then(|v| v.parse().ok()),
                    attn_t4: std::env::var("ALLPAKA_ATTN_T4")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(8),
                    attn_s8: std::env::var("ALLPAKA_ATTN_S8")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(32),
                    mv_tg: std::env::var("ALLPAKA_MV_TG")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|n: &u64| matches!(n, 64 | 128 | 256))
                        .unwrap_or(128),
                    q2_nr0: std::env::var("ALLPAKA_Q2_NR0")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|n: &usize| [1, 2, 4].contains(n))
                        .unwrap_or(4),
                    q4_nr0: std::env::var("ALLPAKA_Q4_NR0")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|n: &usize| [1, 2, 4].contains(n))
                        .unwrap_or(2),
                    q5_nr0: std::env::var("ALLPAKA_Q5_NR0")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|n: &usize| [1, 2, 4].contains(n))
                        .unwrap_or(2),
                    q8_nr0: std::env::var("ALLPAKA_Q8_NR0")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|n: &usize| [1, 2, 4].contains(n))
                        .unwrap_or(2),
                    lpr_div: std::env::var("ALLPAKA_LPR_DIV")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|d: &usize| d.is_power_of_two())
                        .unwrap_or(1),
                    mmid_bn16: std::env::var("ALLPAKA_MMID_BN16")
                        .ok()
                        .map(|v| v != "0"),
                    mmid_rtiles: std::env::var("ALLPAKA_MMID_RTILES")
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
    q2_ilv: bool = false, env "ALLPAKA_Q2_ILV", present;
    qkv_mv: bool = false, env "ALLPAKA_QKV_MV", off;
    attend_mm: bool = true, env "ALLPAKA_ATTN_MM", on;
    attn_mv: bool = true, env "ALLPAKA_ATTN_MV", on;
    fa_db: bool = false, env "ALLPAKA_FA_DB", off;
    mm_ll: bool = true, env "ALLPAKA_MM_LL", on;
    mm_ll_f32: bool = false, env "ALLPAKA_MM_LL_F32", off;
    mm64: bool = false, env "ALLPAKA_MM64", present;
    mm_bn64: bool = false, env "ALLPAKA_MM_BN64", present;
    mm_bn32: bool = false, env "ALLPAKA_MM_BN32", present;
    no_fused_qkv: bool = false, env "ALLPAKA_NO_FUSED_QKV", present;
    debug_qkv: bool = false, env "ALLPAKA_DEBUG_QKV", present;
    q2_mv: bool = true, env "ALLPAKA_Q2_MV", on;
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
    dual_moe: bool = false, env "ALLPAKA_DUAL", present;
    no_route_fused: bool = false, env "ALLPAKA_NO_ROUTE_FUSED", present;
    pf_combine_resnorm: bool = true, env "ALLPAKA_PF_COMBINE_RESNORM", on;
    pf_fused_qk: bool = true, env "ALLPAKA_PF_FQK", on;
    pf_split: bool = false, env "ALLPAKA_PF_SPLIT", present;
    pf_chain: bool = false, env "ALLPAKA_PF_CHAIN", present;
    ffn_split: bool = false, env "ALLPAKA_FFN_SPLIT", present;
    ffn_time: bool = false, env "ALLPAKA_FFN_TIME", present;
    rows_debug: bool = false, env "ALLPAKA_ROWS_DEBUG", present;
    gpu_embed: bool = true, env "ALLPAKA_GPU_EMBED", |_d, e| !std::env::var(e).is_ok_and(|v: String| v == "0" || v.eq_ignore_ascii_case("false"));
    no_stage_barrier: bool = false, env "ALLPAKA_NO_STAGE_BARRIER", present;
    no_barrier_probe: bool = false, env "ALLPAKA_NO_BARRIER", present;
    serial: bool = false, env "ALLPAKA_SERIAL", present;
    prefill_zero_copy: bool = true, env "ALLPAKA_PREFILL_ZERO_COPY", on;
    tokenbuf_debug: bool = false, env "ALLPAKA_TOKENBUF_DEBUG", present;
    decode_split: bool = false, env "ALLPAKA_DECODE_SPLIT", present;
    cuda_graph: bool = false, env "ALLPAKA_CUDA_GRAPH", off;
    cuda_no_fuse: bool = false, env "ALLPAKA_CUDA_NO_FUSE", present;
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

pub fn install(policy: RuntimePolicy) -> Result<(), Box<RuntimePolicy>> {
    POLICY.set(policy).map_err(Box::new)
}

#[cfg(test)]
mod tests {
    use super::RuntimePolicy;

    #[test]
    fn production_defaults_are_explicit() {
        let p = RuntimePolicy::default();
        assert!(p.normflag && p.prefill_defer && p.prefill_one_buffer);
        assert!(p.gpu_route && p.mm_pipeline);
        assert!(
            p.q2_mv && p.q4_mv && p.q8_mv && p.attend_mm && p.attn_mv && p.mm_ll && p.mv_id_grid
        );
        assert!(p.gpu_embed && !p.no_fused_qkv && !p.no_route_fused);
        assert!(!p.decode_serial && !p.serial && !p.mm_k64 && !p.q2_ilv && !p.dual_moe);
        assert!(!p.pf_split && !p.pf_chain && !p.ffn_split && !p.ffn_time && !p.rows_debug);
        assert_eq!(p.attention_split, None);
        assert_eq!(p.attn_t4, 8);
        assert_eq!(p.attn_s8, 32);
        assert_eq!(p.mv_tg, 128);
        assert_eq!(p.q2_nr0, 4);
        assert_eq!(p.q4_nr0, 2);
        assert_eq!(p.q5_nr0, 2);
        assert_eq!(p.q8_nr0, 2);
        assert_eq!(p.lpr_div, 1);
        assert_eq!(p.mmid_bn16, None);
        assert_eq!(p.mmid_rtiles, None);
    }
}
