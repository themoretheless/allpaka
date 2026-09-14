//! Decline-everything GPU backend for hosts without Metal or CUDA.

use allpaka_gguf::GgmlType;

pub fn attach(_mapping: &[u8]) -> bool {
    false
}

pub fn is_attached() -> bool {
    false
}

pub fn residency_status() -> (usize, bool) {
    (0, false)
}

pub fn stats() -> (u64, u64, u64, u64) {
    (0, 0, 0, 0)
}

pub fn gpu_time_stats() -> (u64, u64) {
    (0, 0)
}

pub struct MatvecReq<'a> {
    pub ty: GgmlType,
    pub w: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
    pub x: &'a [f32],
    pub m: usize,
}

pub fn matvec(
    _ty: GgmlType,
    _w: &[u8],
    _n_in: usize,
    _n_out: usize,
    _x: &[f32],
) -> Option<Vec<f32>> {
    None
}

pub fn matvec_batch(_reqs: &[MatvecReq]) -> Option<Vec<Vec<f32>>> {
    None
}

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

pub fn attend_batch(_reqs: &[AttnReq]) -> Option<Vec<Vec<f32>>> {
    None
}

pub fn attend_project(
    _req: &AttnReq,
    _wo_ty: GgmlType,
    _wo: &[u8],
    _n_out: usize,
) -> Option<Vec<f32>> {
    None
}

pub struct AttnBlockReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
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

pub struct TokenGdn<'a> {
    pub wqkv: (GgmlType, &'a [u8], usize),
    pub zgate: (GgmlType, &'a [u8], usize),
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    pub conv1d: &'a [u8],
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    pub conv_off: usize,
    pub state_off: usize,
}

pub struct TokenLayer<'a> {
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub gate_in_q: bool,
    pub q_bias: Option<&'a [u8]>,
    pub k_bias: Option<&'a [u8]>,
    pub v_bias: Option<&'a [u8]>,
    pub k_off: usize,
    pub v_off: usize,
    pub gdn: Option<TokenGdn<'a>>,
    pub ffn: TokenFfn<'a>,
}

pub enum TokenFfn<'a> {
    Dense {
        gate: (GgmlType, &'a [u8], usize),
        up: (GgmlType, &'a [u8], usize),
        down: (GgmlType, &'a [u8], usize),
    },
    Moe {
        router: (GgmlType, &'a [u8], usize),
        router_bias: Option<&'a [u8]>,
        gate: (GgmlType, &'a [u8]),
        up: (GgmlType, &'a [u8]),
        down: (GgmlType, &'a [u8]),
        expert_ffn: usize,
        n_used: usize,
        sigmoid: bool,
        shared: Option<[(GgmlType, &'a [u8], usize); 3]>,
        shared_gate: Option<&'a [u8]>,
    },
}

pub struct TokenReq<'a> {
    pub x: &'a [f32],
    pub m: usize,
    pub layers: &'a [TokenLayer<'a>],
    pub cache: &'a SharedRegion,
    pub ssm: Option<&'a SharedRegion>,
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub pos: usize,
    pub scale: f32,
    pub rope: &'a [[f32; 2]],
    pub rot_dim: usize,
    pub eps: f32,
    pub output_norm: &'a [u8],
    pub output: (GgmlType, &'a [u8], usize),
    pub argmax: bool,
}

pub enum TokenOut {
    Logits(Vec<f32>),
    Argmax(u32),
    Rows {
        argmax: Vec<u32>,
        hidden: Vec<f32>,
    },
}

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

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodePathStats {
    pub attempts: u64,
    pub successes: u64,
    pub declines: u64,
}

pub fn decode_path_stats() -> DecodePathStats {
    DecodePathStats::default()
}

pub fn decode_token_checked(_req: &TokenReq) -> Result<TokenOut, DecodeDecline> {
    Err(DecodeDecline {
        stage: "backend-decode",
        reason: "no GPU accelerator attached on this platform",
    })
}

pub fn decode_token_outcome(req: &TokenReq) -> crate::accel::AccelOutcome<TokenOut> {
    match decode_token_checked(req) {
        Ok(out) => crate::accel::AccelOutcome::Executed(out),
        Err(reason) => crate::accel::AccelOutcome::Declined(crate::accel::DeclineReason::Backend {
            operation: "decode-token",
            detail: reason.to_string(),
        }),
    }
}

pub fn decode_token(_req: &TokenReq) -> Option<TokenOut> {
    None
}

pub struct PrefillFusion<'a> {
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    pub router: &'a [u8],
    pub n_expert: usize,
}

pub fn prefill_begin(_xs: &[f32]) -> Option<()> {
    None
}

pub fn prefill_end(_xs: &mut [f32]) -> Option<()> {
    None
}

pub fn prefill_abort() {}

pub struct PrefillAttnReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    pub hs: &'a [f32],
    pub m: usize,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub ropes: &'a [[f32; 2]],
    pub rot_dim: usize,
    pub gate_in_q: bool,
    pub attn_bias: Option<(&'a [u8], &'a [u8], &'a [u8])>,
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
    pub fusion: Option<PrefillFusion<'a>>,
}

pub fn prefill_attn_block(_req: &PrefillAttnReq) -> Option<Vec<f32>> {
    None
}

pub struct PrefillGdnReq<'a> {
    pub wqkv: (GgmlType, &'a [u8], usize),
    pub zgate: (GgmlType, &'a [u8], usize),
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    pub conv1d: &'a [u8],
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    pub hidden: usize,
    pub m: usize,
    pub eps: f32,
    pub ssm: &'a SharedRegion,
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub conv_off: usize,
    pub state_off: usize,
    pub fusion: Option<PrefillFusion<'a>>,
}

pub fn prefill_gdn_block(_req: &PrefillGdnReq) -> Option<Vec<f32>> {
    None
}

pub struct GroupedRoute<'a> {
    pub n_used: usize,
    pub norm: bool,
    pub scale: f32,
    pub bias: Option<&'a [f32]>,
    pub sigmoid: bool,
}

pub struct GroupedShared<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    pub ffn: usize,
    pub gate_out: Option<(GgmlType, &'a [u8])>,
}

pub struct GroupedCombine<'a> {
    pub tok_off: &'a [u32],
    pub hit_row: &'a [u32],
    pub hit_w: &'a [f32],
    pub m: usize,
}

pub struct GroupedFfnReq<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    pub n_expert: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub groups: &'a [[u32; 3]],
    pub x: &'a [f32],
    pub tok: &'a [u32],
    pub total_rows: usize,
    pub fused: Option<GroupedCombine<'a>>,
    pub shared: Option<GroupedShared<'a>>,
    pub route: Option<GroupedRoute<'a>>,
}

pub fn ffn_batch_grouped(_req: &GroupedFfnReq) -> Option<Vec<f32>> {
    None
}
