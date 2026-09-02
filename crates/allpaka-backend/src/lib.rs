//! Tensor operations for the engine.
//!
//! This is the **CPU reference**: written for correctness you can read, not
//! for speed. Every accelerated implementation (Metal, CUDA, SIMD) is checked
//! against these functions on random inputs before it is trusted; when a fast
//! kernel and this file disagree, this file wins until proven wrong against
//! llama.cpp logits.
//!
//! Conventions, chosen to match how GGUF stores weights:
//!
//! * Activations are `f32` slices, row-major, shape written `[rows, cols]`.
//! * A weight matrix is `[n_out, n_in]` with `n_in` contiguous (GGUF
//!   `dims[0]`), so `matmul` computes `y = x · Wᵀ` - one dot product of a
//!   weight row with an activation row per output element, exactly ggml's
//!   `mul_mat` semantics.

pub mod accel;
pub mod command;
pub mod execution;
pub mod profile;
#[cfg(target_os = "macos")]
pub mod gpu;
pub mod ops;
pub mod quantmat;
pub mod runtime;
pub mod telemetry;

/// On platforms without Metal the GPU module is a stub that always declines,
/// and every matmul stays on the CPU reference path.
#[cfg(not(target_os = "macos"))]
pub mod gpu {
    pub fn attach(_mapping: &[u8]) -> bool {
        false
    }
    pub fn is_attached() -> bool {
        false
    }
    pub fn residency_status() -> (usize, bool) {
        (0, false)
    }
    pub fn matvec(
        _ty: allpaka_gguf::GgmlType,
        _w: &[u8],
        _n_in: usize,
        _n_out: usize,
        _x: &[f32],
    ) -> Option<Vec<f32>> {
        None
    }
    pub struct MatvecReq<'a> {
        pub ty: allpaka_gguf::GgmlType,
        pub w: &'a [u8],
        pub n_in: usize,
        pub n_out: usize,
        pub x: &'a [f32],
        pub m: usize,
    }
    pub fn matvec_batch(_reqs: &[MatvecReq]) -> Option<Vec<Vec<f32>>> {
        None
    }
    pub struct FfnReq<'a> {
        pub gate_ty: allpaka_gguf::GgmlType,
        pub gate_w: &'a [u8],
        pub up_ty: allpaka_gguf::GgmlType,
        pub up_w: &'a [u8],
        pub down_ty: allpaka_gguf::GgmlType,
        pub down_w: &'a [u8],
        pub hidden: usize,
        pub ffn: usize,
        pub x: &'a [f32],
        pub m: usize,
    }
    pub fn ffn_batch(_reqs: &[FfnReq]) -> Option<Vec<Vec<f32>>> {
        None
    }
    pub struct SharedRegion;
    pub fn wrap_region(_region: &[u8]) -> Option<SharedRegion> {
        None
    }
    pub struct AttnReq<'a> {
        pub cache: &'a SharedRegion,
        pub k_off: usize,
        pub v_off: usize,
        pub q: &'a [f32],
        pub kv_dim: usize,
        pub head_dim: usize,
        pub n_q_heads: usize,
        pub group: usize,
        pub n_pos: usize,
        pub scale: f32,
    }
    pub fn attend(_req: &AttnReq) -> Option<Vec<f32>> {
        None
    }
    pub struct AttnBlockReq<'a> {
        pub wq: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wk: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wv: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wo: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub x: &'a [f32],
        pub q_norm: Option<&'a [f32]>,
        pub k_norm: Option<&'a [f32]>,
        pub rope: &'a [[f32; 2]],
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
    pub fn attn_block(_req: &AttnBlockReq) -> Option<Vec<f32>> {
        None
    }
    pub struct TokenLayer<'a> {
        pub attn_norm: &'a [u8],
        pub ffn_norm: &'a [u8],
        pub wq: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wk: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wv: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wo: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub q_norm: Option<&'a [f32]>,
        pub k_norm: Option<&'a [f32]>,
        pub k_off: usize,
        pub v_off: usize,
        pub ffn: TokenFfn<'a>,
    }
    pub enum TokenFfn<'a> {
        Dense {
            gate: (allpaka_gguf::GgmlType, &'a [u8], usize),
            up: (allpaka_gguf::GgmlType, &'a [u8], usize),
            down: (allpaka_gguf::GgmlType, &'a [u8], usize),
        },
        Moe {
            router: (allpaka_gguf::GgmlType, &'a [u8], usize),
            gate: (allpaka_gguf::GgmlType, &'a [u8]),
            up: (allpaka_gguf::GgmlType, &'a [u8]),
            down: (allpaka_gguf::GgmlType, &'a [u8]),
            expert_ffn: usize,
            n_used: usize,
        },
    }
    pub struct TokenReq<'a> {
        pub x: &'a [f32],
        pub layers: &'a [TokenLayer<'a>],
        pub cache: &'a SharedRegion,
        pub kv_dim: usize,
        pub head_dim: usize,
        pub n_heads: usize,
        pub n_kv_heads: usize,
        pub pos: usize,
        pub scale: f32,
        pub rope: &'a [[f32; 2]],
        pub eps: f32,
        pub output_norm: &'a [u8],
        pub output: (allpaka_gguf::GgmlType, &'a [u8], usize),
    }
    pub fn decode_token(_req: &TokenReq) -> Option<Vec<f32>> {
        None
    }
    pub struct PrefillAttnReq<'a> {
        pub wq: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wk: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wv: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub wo: (allpaka_gguf::GgmlType, &'a [u8], usize),
        pub hs: &'a [f32],
        pub m: usize,
        pub q_norm: Option<&'a [f32]>,
        pub k_norm: Option<&'a [f32]>,
        pub ropes: &'a [[f32; 2]],
        pub eps: f32,
        pub cache: &'a SharedRegion,
        pub k_off: usize,
        pub v_off: usize,
        pub kv_dim: usize,
        pub head_dim: usize,
        pub n_heads: usize,
        pub n_kv_heads: usize,
        pub base: usize,
        pub scale: f32,
    }
    pub fn prefill_attn_block(_req: &PrefillAttnReq) -> Option<Vec<f32>> {
        None
    }
    pub struct GroupedFfnReq<'a> {
        pub gate: (allpaka_gguf::GgmlType, &'a [u8]),
        pub up: (allpaka_gguf::GgmlType, &'a [u8]),
        pub down: (allpaka_gguf::GgmlType, &'a [u8]),
        pub n_expert: usize,
        pub hidden: usize,
        pub ffn: usize,
        pub groups: &'a [[u32; 3]],
        pub x: &'a [f32],
        pub total_rows: usize,
    }
    pub fn ffn_batch_grouped(_req: &GroupedFfnReq) -> Option<Vec<f32>> {
        None
    }
    /// `(calls, dispatches, encode_ns, wait_ns)`: all zero without a GPU.
    pub fn stats() -> (u64, u64, u64, u64) {
        (0, 0, 0, 0)
    }
    pub fn gpu_time_stats() -> (u64, u64) {
        (0, 0)
    }
}

pub use ops::{matmul_f32, rmsnorm, rope_neox, rope_norm, silu, softmax, swiglu};
pub use quantmat::QuantMat;
