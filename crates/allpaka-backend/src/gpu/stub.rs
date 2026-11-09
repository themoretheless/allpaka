//! Decline-everything GPU backend for hosts without Metal or CUDA.
//!
//! Mirrors the public surface of the real backends: every entry point
//! declines (`None` / `false`) so callers take their CPU paths unchanged.

use allpaka_dtype::GgmlType;
use std::sync::atomic::Ordering;

pub use super::types::*;

/// Opaque handle; never constructed without a GPU backend.
pub struct SharedRegion;

impl SharedRegion {
    #[inline]
    pub const fn architecture(&self) -> crate::memory::MemoryArchitecture {
        crate::memory::MemoryArchitecture::HostOnly
    }

    #[inline]
    pub const fn is_unified(&self) -> bool {
        false
    }

    #[inline]
    pub const fn is_discrete(&self) -> bool {
        false
    }
}

pub fn attach(_mapping: &[u8]) -> bool {
    false
}

pub fn is_attached() -> bool {
    false
}

pub fn memory_architecture() -> crate::memory::MemoryArchitecture {
    crate::memory::MemoryArchitecture::HostOnly
}

pub fn init_error() -> Option<String> {
    None
}

pub fn residency_status() -> (usize, bool) {
    (0, false)
}

pub fn arena_bytes() -> usize {
    0
}

pub fn stats() -> (u64, u64, u64, u64) {
    (0, 0, 0, 0)
}

pub fn gpu_time_stats() -> (u64, u64) {
    (0, 0)
}

pub fn set_rope_inv_freq(_inv_freq: &[f32]) -> bool {
    false
}

pub fn has_device_rope_freq() -> bool {
    false
}

pub fn prefer_gpu_embed() -> bool {
    false
}

pub fn can_embed_on_gpu(_ty: GgmlType, _bytes: &[u8], _n_in: usize) -> bool {
    false
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

pub fn ffn_batch(_reqs: &[FfnReq]) -> Option<Vec<Vec<f32>>> {
    None
}

pub fn ffn_batch_grouped(_req: &GroupedFfnReq) -> Option<Vec<f32>> {
    None
}

pub fn wrap_region(_region: &[u8]) -> Option<SharedRegion> {
    None
}

pub fn upload_region_range(_region: &mut SharedRegion, _offset: usize, _data: &[u8]) -> bool {
    false
}

pub fn download_region_range(_region: &SharedRegion, _offset: usize, _dst: &mut [u8]) -> bool {
    false
}

pub fn decode_attention_capacity_safe(_capacity: usize) -> bool {
    true
}

pub fn minimum_kv_capacity() -> usize {
    1
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

pub fn attn_block(_req: &AttnBlockReq) -> Option<Vec<f32>> {
    None
}

pub fn decode_token_checked(_req: &TokenReq) -> Result<TokenOut, DecodeDecline> {
    DECODE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    DECODE_DECLINES.fetch_add(1, Ordering::Relaxed);
    Err(DecodeDecline {
        stage: "backend-decode",
        reason: "no GPU accelerator attached on this platform",
    })
}

pub fn decode_token(req: &TokenReq) -> Option<TokenOut> {
    decode_token_checked(req).ok()
}

pub fn decode_greedy_continue(_req: &TokenReq, _n_more: usize) -> Option<Vec<u32>> {
    None
}

pub fn prefill_begin(_xs: &[f32]) -> Option<()> {
    None
}

pub fn prefill_replay(_xs: &[f32]) -> Option<Vec<f32>> {
    None
}

pub fn prefill_end(_xs: &mut [f32]) -> Option<()> {
    None
}

pub fn prefill_end_output(_len: usize) -> Option<crate::activation::Activation> {
    None
}

pub fn prefill_abort() {}

pub fn prefill_attn_block(_req: &PrefillAttnReq) -> Option<Vec<f32>> {
    None
}

pub fn prefill_gdn_block(_req: &PrefillGdnReq) -> Option<Vec<f32>> {
    None
}

pub fn shared_activation_stats() -> crate::activation::SharedStats {
    crate::activation::SharedStats::default()
}
