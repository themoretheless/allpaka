//! Process-wide GPU counters: call and dispatch totals, CPU-side encode and
//! wait time, and the command buffers' own execution and scheduling clocks.

use std::sync::atomic::Ordering;

pub(super) use crate::gpu::common::{CALLS, DISPATCHES, ENCODE_NS, GPU_BUSY_NS, SCHED_NS, WAIT_NS};

/// Read the completed buffer's own timestamps. CFTimeIntervals in seconds on
/// a shared clock, so differences are valid; zeros (no GPU work) are skipped.
pub(super) fn note_gpu_times(cmd: &metal::CommandBufferRef) {
    use objc::{msg_send, sel, sel_impl};
    let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
    let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
    let ks: f64 = unsafe { msg_send![cmd, kernelStartTime] };
    if ge > gs && gs > 0.0 {
        GPU_BUSY_NS.fetch_add(((ge - gs) * 1e9) as u64, Ordering::Relaxed);
        if gs > ks && ks > 0.0 {
            SCHED_NS.fetch_add(((gs - ks) * 1e9) as u64, Ordering::Relaxed);
        }
    }
}
