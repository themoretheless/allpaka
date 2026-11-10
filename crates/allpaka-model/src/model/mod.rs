//! The forward pass: tokens in, logits out.
//!
//! Decomposed into focused submodules:
//! - [`session`]: `Session` KV/SSM/RoPE cache management.
//! - [`loader`]: GGUF weight loading, stage slicing, and BF16 router weight ownership.
//! - [`moe`]: Dense & Mixture-of-Experts FFNs and top-k routing.
//! - [`gdn`]: Gated Delta Net linear-attention recurrence (`qwen35moe`).
//! - [`mtp`]: Multi-Token Prediction (`nextn`) speculative draft block.
//! - [`attention`]: Causal grouped-query attention (`GQA`), QK-norm, RoPE, and KV store.
//! - [`prefill`]: Batched prompt prefill and GPU-fused prefill passes.
//! - [`decode`]: Single-token decode, multi-token verification, and greedy chain execution.

pub(crate) mod attention;
pub(crate) mod decode;
pub(crate) mod gdn;
pub(crate) mod loader;
pub(crate) mod moe;
pub(crate) mod mtp;
pub(crate) mod prefill;
pub(crate) mod session;

pub use moe::route;
pub use session::Session;

use crate::config::Config;
use allpaka_backend::QuantMat;
use gdn::GdnLayer;
use moe::Ffn;
use mtp::MtpLayer;

/// q/k/v projection biases (GLM's attention_bias): the parsed vectors for the
/// CPU path, the raw F32 mmap bytes for GPU binds.
pub(crate) struct AttnBias<'a> {
    pub(crate) q: (Vec<f32>, &'a [u8]),
    pub(crate) k: (Vec<f32>, &'a [u8]),
    pub(crate) v: (Vec<f32>, &'a [u8]),
}

pub(crate) struct Layer<'a> {
    pub(crate) attn_norm: Vec<f32>,
    /// The same weights as raw F32 mmap bytes, for GPU binds (empty when the
    /// tensor is not plain F32 and the whole-token path then declines).
    pub(crate) attn_norm_raw: &'a [u8],
    pub(crate) ffn_norm_raw: &'a [u8],
    /// Full-attention projections; None on gated-delta-net layers (qwen35moe).
    pub(crate) attn: Option<AttnWeights<'a>>,
    /// Gated delta net branch; None on full-attention layers.
    pub(crate) gdn: Option<GdnLayer<'a>>,
    pub(crate) bias: Option<AttnBias<'a>>,
    /// Qwen3 per-head norms, absent on Llama/Mistral.
    pub(crate) q_norm: Option<Vec<f32>>,
    pub(crate) k_norm: Option<Vec<f32>>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) ffn: Ffn<'a>,
}

pub(crate) struct AttnWeights<'a> {
    pub(crate) wq: QuantMat<'a>,
    pub(crate) wk: QuantMat<'a>,
    pub(crate) wv: QuantMat<'a>,
    pub(crate) wo: QuantMat<'a>,
    /// The q projection's second half is a per-head output gate (qwen35):
    /// head h reads q from rows `[h*2*hd, h*2*hd+hd)` and the gate from
    /// `[h*2*hd+hd, h*2*hd+2hd)`.
    pub(crate) gate_in_q: bool,
}

pub struct Model<'a> {
    pub config: Config,
    pub(crate) embd: Option<QuantMat<'a>>,
    pub(crate) layer_start: usize,
    pub(crate) layers: Vec<Layer<'a>>,
    /// The MTP (nextn) speculative block past the trunk, when the GGUF
    /// carries one (qwen35moe with nextn_predict_layers).
    pub(crate) mtp: Option<MtpLayer<'a>>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) output_norm_raw: &'a [u8],
    pub(crate) output: Option<QuantMat<'a>>,
    /// `base^(-2i/d)` per rotary pair, hoisted out of the per-head loops.
    pub(crate) rope_inv_freq: Vec<f32>,
    /// Owned F32 buffers converted from BF16 router/gate tensors at load time.
    /// Placed last so it is dropped after all `QuantMat<'a>` fields borrowing from it.
    pub(crate) _owned_weights: Vec<Box<[f32]>>,
}

static TRACE_HOOK: std::sync::OnceLock<fn(&str)> = std::sync::OnceLock::new();

pub fn set_trace_hook(hook: fn(&str)) {
    let _ = TRACE_HOOK.set(hook);
}

macro_rules! cached_env_flag {
    ($fn_name:ident, $var:literal) => {
        #[inline]
        pub(crate) fn $fn_name() -> bool {
            // Integration tests (`tests/*.rs`) compile `allpaka-model` without
            // `cfg(test)` while toggling `ALLPAKA_*` inside the same process;
            // keep `debug_assertions` uncached so parity tests see live env vars,
            // and cache in `--release` so hot-path decode pays one `Relaxed` load.
            if cfg!(debug_assertions) {
                std::env::var_os($var).is_some()
            } else {
                static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *V.get_or_init(|| std::env::var_os($var).is_some())
            }
        }
    };
    ($fn_name:ident, $var:literal, default_true) => {
        #[inline]
        pub(crate) fn $fn_name() -> bool {
            let parse = || {
                std::env::var($var)
                    .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
                    .unwrap_or(true)
            };
            if cfg!(debug_assertions) {
                parse()
            } else {
                static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *V.get_or_init(parse)
            }
        }
    };
}

cached_env_flag!(trace_enabled, "ALLPAKA_TRACE");
cached_env_flag!(gdn_zero_enabled, "ALLPAKA_GDN_ZERO");
cached_env_flag!(cpu_attn_enabled, "ALLPAKA_CPU_ATTN");
cached_env_flag!(no_tokenbuf_enabled, "ALLPAKA_NO_TOKENBUF");
cached_env_flag!(no_argmax_enabled, "ALLPAKA_NO_ARGMAX");
cached_env_flag!(prefuse_enabled, "ALLPAKA_PREFUSE", default_true);
cached_env_flag!(attn_zero_enabled, "ALLPAKA_ATTN_ZERO");
cached_env_flag!(moe_zero_enabled, "ALLPAKA_MOE_ZERO");
cached_env_flag!(tokenbuf_debug_enabled, "ALLPAKA_TOKENBUF_DEBUG");
cached_env_flag!(greedy_chain_enabled, "ALLPAKA_GREEDY_CHAIN", default_true);

#[inline]
pub(crate) fn skip_host_rope_enabled() -> bool {
    let parse = || {
        std::env::var("ALLPAKA_SKIP_HOST_ROPE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    };
    if cfg!(debug_assertions) {
        parse()
    } else {
        static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *S.get_or_init(parse)
    }
}

#[inline]
pub(crate) fn cpu_threads() -> usize {
    allpaka_backend::runtime::cpu_threads()
}

pub(crate) fn trace(name: &str, li: usize, v: &[f32]) {
    if trace_enabled() {
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        let head: Vec<String> = v.iter().take(3).map(|x| format!("{x:.6}")).collect();
        let tail: Vec<String> = v
            .iter()
            .rev()
            .take(3)
            .rev()
            .map(|x| format!("{x:.6}"))
            .collect();
        let line = format!(
            "trace {name}-{li} sum={sum:.6} head=[{}] tail=[{}]",
            head.join(" "),
            tail.join(" ")
        );
        if let Some(hook) = TRACE_HOOK.get() {
            hook(&line);
        } else {
            eprintln!("{line}");
        }
    }
}
