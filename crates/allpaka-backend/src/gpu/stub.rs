//! Stub GPU module for non-macOS targets (or when CUDA is disabled). Every
//! entry point declines (`None` / `false`) so the callers take their existing
//! CPU paths unchanged.

use allpaka_gguf::GgmlType;
use std::sync::atomic::{AtomicU64, Ordering};

pub use super::types::*;

/// Opaque handle; never constructed on non-macOS.
pub struct SharedRegion {
    _priv: (),
}

/// Unreachable on stub builds (`prefill_end_output` always returns `None`);
/// exists so `Activation::Shared` compiles uniformly across targets.
pub struct SharedVector(Vec<f32>);

impl std::ops::Deref for SharedVector {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.0
    }
}
impl std::ops::DerefMut for SharedVector {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.0
    }
}

pub fn shared_region(_ptr: *mut u8, _len: usize) -> Option<SharedRegion> {
    None
}

pub fn attach(_mapping: &[u8]) -> bool {
    false
}

pub fn is_attached() -> bool {
    false
}

/// Active memory architecture for the stub backend (always `HostOnly`).
#[inline]
pub fn memory_architecture() -> crate::memory::MemoryArchitecture {
    crate::memory::MemoryArchitecture::HostOnly
}

pub fn residency_status() -> (usize, bool) {
    (0, false)
}

pub fn arena_bytes() -> usize {
    0
}

pub fn init_error() -> Option<String> {
    Some("stub GPU module (not macOS)".into())
}

pub fn reset_profile() {}

pub fn take_profile() -> (u64, u64, u64, u64, u64) {
    (0, 0, 0, 0, 0)
}

pub fn matmul(_ty: GgmlType, _w: &[u8], _n_in: usize, _n_out: usize, _x: &[f32]) -> Option<Vec<f32>> {
    None
}

pub fn matvec(_ty: GgmlType, _w: &[u8], _n_in: usize, _n_out: usize, _x: &[f32]) -> Option<Vec<f32>> {
    None
}

pub fn matvec_batch(_reqs: &[MatvecReq]) -> Option<Vec<Vec<f32>>> {
    None
}

pub fn ffn_batch(_reqs: &[FfnReq]) -> Option<Vec<Vec<f32>>> {
    None
}

pub fn attend(_req: &AttnReq) -> Option<Vec<f32>> {
    None
}

pub fn attend_batch(_reqs: &[AttnReq]) -> Option<Vec<Vec<f32>>> {
    None
}

pub fn attn_block(_req: &AttnBlockReq) -> Option<Vec<f32>> {
    None
}

pub fn decode_token_checked(_req: &TokenReq) -> Result<TokenOut, DecodeDecline> {
    DECODE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    DECODE_DECLINES.fetch_add(1, Ordering::Relaxed);
    Err(DecodeDecline {
        stage: "device",
        reason: "GPU backend is compiled out on non-macOS targets",
    })
}

pub fn decode_token(req: &TokenReq) -> Option<TokenOut> {
    decode_token_checked(req).ok()
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

pub fn shared_activation_stats() -> crate::activation::SharedStats {
    crate::activation::SharedStats::default()
}

pub fn prefill_abort() {}

pub fn prefill_attn_block(_req: &PrefillAttnReq) -> Option<Vec<f32>> {
    None
}

pub fn prefill_gdn_block(_req: &PrefillGdnReq) -> Option<Vec<f32>> {
    None
}

pub fn prefill_router_at() -> usize {
    0
}

pub fn ffn_grouped(_req: &GroupedFfnReq) -> Option<Vec<f32>> {
    None
}
