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
            /// `ALLPAKA_MV_TG` ∈ {64,128,256}, default 128.
            pub mv_tg: u64,
            /// `ALLPAKA_Q8_NR0` ∈ {1,2,4}, default 2.
            pub q8_nr0: usize,
        }

        impl Default for RuntimePolicy {
            #[inline]
            fn default() -> Self {
                Self {
                    $($field: $def,)*
                    attention_split: None,
                    mv_tg: 128,
                    q8_nr0: 2,
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
}

static POLICY: OnceLock<RuntimePolicy> = OnceLock::new();

#[inline]
pub fn get() -> &'static RuntimePolicy {
    POLICY.get_or_init(RuntimePolicy::from_env)
}

pub fn install(policy: RuntimePolicy) -> Result<(), RuntimePolicy> {
    POLICY.set(policy)
}

#[cfg(test)]
mod tests {
    use super::RuntimePolicy;

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
