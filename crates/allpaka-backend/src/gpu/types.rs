//! Shared GPU request, response, and decode telemetry types across backends
//! (`metal`, `cuda`, and `stub`).

use allpaka_dtype::GgmlType;

use super::SharedRegion;

/// One matmul request inside a batch: `x` holds `m` activation rows.
pub struct MatvecReq<'a> {
    pub ty: GgmlType,
    pub w: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
    pub x: &'a [f32],
    pub m: usize,
}

/// One FFN inside a fused batch: `y = swiglu(x·Gᵀ, x·Uᵀ) · Dᵀ` for `m`
/// activation rows, without the CPU touching the intermediates.
pub struct FfnReq<'a> {
    pub gate_ty: GgmlType,
    pub gate_w: &'a [u8],
    pub up_ty: GgmlType,
    pub up_w: &'a [u8],
    pub down_ty: GgmlType,
    pub down_w: &'a [u8],
    pub hidden: usize,
    pub ffn: usize,
    pub x: &'a [f32],
    pub m: usize,
}

/// Single-position causal GQA attention over a shared f16 KV cache.
pub struct AttnReq<'a> {
    /// The shared f16 cache region (`KvCache::gpu_region`).
    pub cache: &'a SharedRegion,
    /// Element offsets of this layer's K and V slabs inside `cache`.
    pub k_off: usize,
    pub v_off: usize,
    /// Query heads, `[n_q_heads * head_dim]` contiguous.
    pub q: &'a [f32],
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub group: usize,
    /// Positions in the cache, `pos + 1`.
    pub n_pos: usize,
    pub scale: f32,
}

/// One decode layer's whole attention half as ONE command buffer: the three
/// QKV projections, per-head norm + RoPE, the f16 store of K/V into the
/// cache, attention over it, and the output projection.
pub struct AttnBlockReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    /// The normed layer input, `hidden` floats.
    pub x: &'a [f32],
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// `(sin, cos)` per rotary pair, `head_dim / 2` entries.
    pub rope: &'a [[f32; 2]],
    /// `true` for adjacent-pair NORM RoPE (Llama 3.x), `false` for split-half NeoX.
    pub rope_norm: bool,
    pub eps: f32,
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub pos: usize,
    pub scale: f32,
}

/// One gated-delta-net (linear attention) layer inside [`TokenLayer`].
pub struct TokenGdn<'a> {
    /// `hidden -> key_dim*2 + value_dim`, Q8_0.
    pub wqkv: (GgmlType, &'a [u8], usize),
    /// `hidden -> value_dim` (the z gate), Q8_0.
    pub zgate: (GgmlType, &'a [u8], usize),
    /// F32 `[hidden -> heads_v]` projections, raw bytes in the mmap.
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    /// F32 conv weights `[channels * d_conv]`, raw bytes.
    pub conv1d: &'a [u8],
    /// Small per-head F32 vectors handed to the kernel by value.
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    /// `value_dim -> hidden`, Q8_0.
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    /// Element offsets into [`TokenReq::ssm`] for this layer's conv window
    /// and `[heads_v, d, d]` state.
    pub conv_off: usize,
    pub state_off: usize,
}

pub struct TokenLayer<'a> {
    /// Raw GGUF bytes of the F32 norm weights (`hidden` floats each).
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    /// Present on gated-delta-net layers (3 of every 4 in qwen35moe); when
    /// `Some`, the attention fields below are unused placeholders.
    pub gdn: Option<TokenGdn<'a>>,
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    /// Per-head Q/K/V additive biases as raw F32 GGUF bytes.
    pub q_bias: Option<&'a [u8]>,
    pub k_bias: Option<&'a [u8]>,
    pub v_bias: Option<&'a [u8]>,
    /// Qwen3.5-MoE: `wq` has `2 * q_dim` output rows (`[q | gate]` per head);
    /// the gate half applies as `attn *= sigmoid(gate)` before `wo`.
    pub gate_in_q: bool,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub k_off: usize,
    pub v_off: usize,
    pub ffn: TokenFfn<'a>,
}

pub enum TokenFfn<'a> {
    Dense {
        gate: (GgmlType, &'a [u8], usize),
        up: (GgmlType, &'a [u8], usize),
        down: (GgmlType, &'a [u8], usize),
    },
    Moe {
        /// `(F32, bytes, n_expert)`.
        router: (GgmlType, &'a [u8], usize),
        /// GLM's `exp_probs_b`, raw F32 bytes (`n_expert` floats): shifts
        /// top-k selection while the gathered weights stay unshifted.
        router_bias: Option<&'a [u8]>,
        /// Packed 3D expert tensors; `slice.len() / n_expert` is the
        /// per-expert byte stride.
        gate: (GgmlType, &'a [u8]),
        up: (GgmlType, &'a [u8]),
        down: (GgmlType, &'a [u8]),
        expert_ffn: usize,
        n_used: usize,
        /// Sigmoid gating (GLM) vs softmax (Qwen3).
        sigmoid: bool,
        /// Shared expert `[gate, up, down]`, running into slot `n_used`
        /// with combine weight 1.0 (or `sigmoid(shared_gate · h)` when
        /// `shared_gate` is present).
        shared: Option<[(GgmlType, &'a [u8], usize); 3]>,
        /// Raw F32 `[hidden]` gate vector for the shared expert (qwen35moe):
        /// its combine weight becomes `sigmoid(dot(shared_gate, h))`.
        shared_gate: Option<&'a [u8]>,
    },
}

pub struct TokenReq<'a> {
    pub layers: &'a [TokenLayer<'a>],
    pub output_norm: &'a [u8],
    pub output: (GgmlType, &'a [u8], usize),
    /// Token embedding table `(ty, bytes, hidden)` plus `token_id`: when both
    /// are `Some` and `m == 1`, `decode_token` dequantizes the embedding row
    /// on the GPU at the head of the command buffer instead of copying `x`
    /// from the CPU.
    pub embd: Option<(GgmlType, &'a [u8], usize)>,
    pub token_id: Option<u32>,
    /// The input row(s), `m * hidden` floats (`m == 1` for plain decode,
    /// `m > 1` for MTP verification). Ignored when `embd` + `token_id` are used.
    pub x: &'a [f32],
    /// Number of consecutive tokens starting at `pos`. `1` is the single-token
    /// decode path; `> 1` runs the speculative verify path.
    pub m: usize,
    /// `(sin, cos)` pairs: `rot_dim / 2` entries per row, `m` rows back to back.
    pub rope: &'a [[f32; 2]],
    /// Rotary dimensions per head (`<= head_dim`; partial RoPE leaves the
    /// tail `[rot_dim..head_dim]` unrotated).
    pub rot_dim: usize,
    /// `true` for adjacent-pair NORM RoPE (Llama 3.x), `false` for split-half NeoX.
    pub rope_norm: bool,
    pub eps: f32,
    pub cache: &'a SharedRegion,
    /// Total token slots in `cache` (`KvCache::capacity`).
    pub cache_capacity: usize,
    /// Shared SSM region (`SsmCache::gpu_region`) when any layer has `gdn`.
    pub ssm: Option<&'a SharedRegion>,
    /// MTP verification (`m > 1`): `(slots_region, slot_stride_elems)` from
    /// `SsmCache::arm_slots`, so the batch GDN kernels snapshot each row's
    /// state for rollback.
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub pos: usize,
    pub scale: f32,
    /// When true, reduce to the argmax token id on the GPU and read back
    /// 4 bytes instead of `vocab * 4`.
    pub argmax: bool,
}

#[derive(Debug)]
pub enum TokenOut {
    Logits(Vec<f32>),
    Argmax(u32),
    /// Multi-row verify (`TokenReq::m > 1`): greedy winner per row plus the
    /// output-normed hidden rows (`m * hidden`), which MTP needs for the
    /// next draft step.
    Rows {
        argmax: Vec<u32>,
        hidden: Vec<f32>,
    },
}

/// A whole-token GPU request was rejected before submission. The checked API
/// keeps this distinct from a successful GPU result so callers cannot silently
/// mistake a CPU fallback for GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeDecline {
    pub stage: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for DecodeDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stage={} reason={}", self.stage, self.reason)
    }
}

impl std::error::Error for DecodeDecline {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodePathStats {
    pub attempts: u64,
    pub successes: u64,
    pub declines: u64,
}

pub(crate) static DECODE_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DECODE_SUCCESSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DECODE_DECLINES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub fn decode_path_stats() -> DecodePathStats {
    use std::sync::atomic::Ordering::Relaxed;
    DecodePathStats {
        attempts: DECODE_ATTEMPTS.load(Relaxed),
        successes: DECODE_SUCCESSES.load(Relaxed),
        declines: DECODE_DECLINES.load(Relaxed),
    }
}

pub fn reset_decode_path_stats() {
    use std::sync::atomic::Ordering::Relaxed;
    DECODE_ATTEMPTS.store(0, Relaxed);
    DECODE_SUCCESSES.store(0, Relaxed);
    DECODE_DECLINES.store(0, Relaxed);
}

/// Backend-neutral checked decode contract. New callers should use this
/// rather than interpreting `None` from the compatibility API.
pub fn decode_token_outcome(req: &TokenReq) -> crate::accel::AccelOutcome<TokenOut> {
    match super::decode_token_checked(req) {
        Ok(out) => crate::accel::AccelOutcome::Executed(out),
        Err(reason) => crate::accel::AccelOutcome::Declined(crate::accel::DeclineReason::Backend {
            operation: "decode-token",
            detail: reason.to_string(),
        }),
    }
}

/// Extra weights for the fused prefill: with this present (and
/// `prefill_begin` called for the chunk), the block reads the residual
/// stream from GPU memory, norms on the GPU, folds the attention output
/// back in, re-norms for the FFN and finishes with the router matmul -
/// returning router logits instead of the projected rows.
pub struct PrefillFusion<'a> {
    /// F32 raw norm weights, `hidden` floats each.
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    /// F32 raw router weights, `n_expert * hidden` floats.
    pub router: &'a [u8],
    pub n_expert: usize,
}

pub struct PrefillAttnReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    /// Normed layer input, `m * hidden` floats.
    pub hs: &'a [f32],
    pub m: usize,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// `(sin, cos)` pairs for every row: `m * rot_dim / 2` entries.
    pub ropes: &'a [[f32; 2]],
    /// Rotary width; `== head_dim` for full RoPE, smaller for GLM's partial,
    /// `64` for qwen35moe's 256-wide heads.
    pub rot_dim: usize,
    /// `true` for adjacent-pair NORM RoPE (Llama 3.x), `false` for split-half NeoX.
    pub rope_norm: bool,
    /// Qwen3.5-MoE: the sigmoid output gate rides inside the Q projection
    /// (`[q | gate]` per head, `wq.n_out == 2 * q_dim`); split and applied
    /// post-attention.
    pub gate_in_q: bool,
    /// GLM's per-head Q/K/V additive biases as raw F32 GGUF bytes.
    pub attn_bias: Option<(&'a [u8], &'a [u8], &'a [u8])>,
    pub eps: f32,
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// First row's position; row `i` attends over `base + i + 1` positions.
    pub base: usize,
    pub scale: f32,
    pub fusion: Option<PrefillFusion<'a>>,
}

/// A prefill chunk of one gated-delta-net layer (qwen35moe linear
/// attention), mirroring [` super::prefill_attn_block`]'s fusion contract:
/// reads the residual stream from `pf_x`, leaves it updated there (branch
/// output folded into the FFN norm), and leaves router logits in `y_arena`
/// for GPU routing.
pub struct PrefillGdnReq<'a> {
    /// `hidden -> conv channels (key_dim*2 + value_dim)`, Q8_0.
    pub wqkv: (GgmlType, &'a [u8], usize),
    /// `hidden -> value_dim` (the z gate), Q8_0.
    pub zgate: (GgmlType, &'a [u8], usize),
    /// F32 `[hidden -> heads_v]` projections, raw bytes in the mmap.
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    /// F32 conv weights `[channels * d_conv]`, raw bytes.
    pub conv1d: &'a [u8],
    /// Small F32 vectors handed to the kernel by value.
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    /// `value_dim -> hidden`, Q8_0.
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    pub hidden: usize,
    pub m: usize,
    pub eps: f32,
    /// The shared SSM region: conv window and deltanet state, updated in
    /// place (element offsets per layer).
    pub ssm: &'a SharedRegion,
    /// MTP verification: per-row rollback slots (`SsmCache::arm_slots`),
    /// `(region, slot stride in elements)`.
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub conv_off: usize,
    pub state_off: usize,
    pub fusion: Option<PrefillFusion<'a>>,
}

pub struct GroupedRoute<'a> {
    pub n_used: usize,
    /// Renormalize the winners to sum to 1 (`norm_topk_prob`).
    pub norm: bool,
    pub scale: f32,
    /// Selection bias (`+bias` shifts only the top-k choice).
    pub bias: Option<&'a [f32]>,
    /// Sigmoid gating (GLM) when true, softmax over all experts (Qwen3)
    /// when false.
    pub sigmoid: bool,
}

pub struct GroupedShared<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    /// Shared-expert FFN width (`n_ff_exp * n_expert_shared`).
    pub ffn: usize,
    /// Qwen3.5-MoE: the shared expert's combine weight is `sigmoid(gate · h)`
    /// per token, an F32 `[hidden]` projection in the mmap.
    pub gate_out: Option<(GgmlType, &'a [u8])>,
}

pub struct GroupedCombine<'a> {
    /// CSR offsets, `m + 1` entries.
    pub tok_off: &'a [u32],
    /// Flat-row index and routing weight per hit.
    pub hit_row: &'a [u32],
    pub hit_w: &'a [f32],
    pub m: usize,
    /// The NEXT layer's attention-norm weight (raw F32) plus RMS eps.
    pub next_norm: Option<(&'a [u8], f32)>,
}

/// A prefill chunk's whole routed FFN — every expert group's gate, up,
/// SwiGLU, and down — as ONE command buffer.
pub struct GroupedFfnReq<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    pub n_expert: usize,
    pub hidden: usize,
    pub ffn: usize,
    /// `[expert, row0, rows]` per hit group; `row0` indexes into the flat row space.
    pub groups: &'a [[u32; 3]],
    /// The UNGATHERED activations (`m` rows); `tok[row]` names each flat
    /// row's source token.
    pub x: &'a [f32],
    pub tok: &'a [u32],
    pub total_rows: usize,
    /// Fused prefill combine parameters.
    pub fused: Option<GroupedCombine<'a>>,
    /// Optional shared expert (fused path).
    pub shared: Option<GroupedShared<'a>>,
    /// Optional GPU-side routing (fused path).
    pub route: Option<GroupedRoute<'a>>,
}
