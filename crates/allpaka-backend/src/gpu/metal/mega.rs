//! Per-layer megakernel scaffolding. Disabled: `mega_enabled()` is hard false
//! because the all-threadgroup spin rendezvous can wedge the Metal scheduler.

use super::*;

#[repr(C)]
pub(super) struct GpuMegaArgs {
    pub(super) hidden: u32,
    pub(super) ffn: u32,
    pub(super) n_expert: u32,
    pub(super) n_used: u32,
    pub(super) x_at: u32,
    pub(super) delta_at: u32,
    pub(super) h_at: u32,
    pub(super) logits_at: u32,
    pub(super) ids_at: u32,
    pub(super) wts_at: u32,
    pub(super) gate_at: u32,
    pub(super) up_at: u32,
    pub(super) downo_at: u32,
    pub(super) ctr_at: u32,
    pub(super) n_tg: u32,
    pub(super) ctr_base: u32,
    pub(super) eps: f32,
    pub(super) _pad: u32,
    pub(super) gate_off: u64,
    pub(super) up_off: u64,
    pub(super) down_off: u64,
    pub(super) router_off: u64,
    pub(super) gate_stride: u64,
    pub(super) up_stride: u64,
    pub(super) down_stride: u64,
    pub(super) sigmoid: u32,
    /// 0 = none, 1 = always-on weight 1, 2 = gated via `sh_gout_off`.
    pub(super) has_shared: u32,
    pub(super) sh_gate_off: u64,
    pub(super) sh_up_off: u64,
    pub(super) sh_down_off: u64,
    pub(super) sh_gout_off: u64,
    /// Metal `set_bytes` length should be a multiple of 16; keep trailing pad.
    pub(super) _tail: u64,
}

/// How many threadgroups the megakernel launches.
///
/// Multi-TG uses a device atomic busy-wait that soft-locks the Metal
/// scheduler and can freeze the whole Mac. Until a non-spinning sync exists,
/// this is hard-capped at 1 (local barrier only). `ALLPAKA_MEGA_TG` is ignored.
pub(super) fn mega_tg() -> u32 {
    1
}

pub(super) fn mega_debug() -> bool {
    crate::runtime::get().mega_debug
}

/// MEGA stays off. A live multi-TG device-atomic `mega_sync` soft-locked the
/// Metal scheduler and froze a Mac; the spin was deleted and `mega_tg` is 1,
/// but the fused path is still unfinished. Do not re-enable via env until a
/// non-spinning redesign + a test that cannot wedge the GPU lands.
pub(super) fn mega_enabled() -> bool {
    false
}

/// Format code for the megakernel's expert cores; None = unsupported.
pub(super) fn mega_fmt(ty: GgmlType) -> Option<u32> {
    match ty {
        GgmlType::Q2K => Some(0),
        GgmlType::Q3K => Some(1),
        GgmlType::Q4K => Some(2),
        GgmlType::Q5_0 => Some(3),
        GgmlType::Q5K => Some(4),
        GgmlType::Q8_0 => Some(5),
        _ => None,
    }
}
