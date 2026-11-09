//! Process-wide GPU counters: call and dispatch totals, CPU-side encode and
//! wait time, and the command buffers' own execution and scheduling clocks.

use std::sync::atomic::{AtomicU64, Ordering};

/// Where GPU wall time goes, accumulated across every call. The split that
/// matters: `wait_ns` is dead time the CPU spends blocked on the GPU, and its
/// ratio to `calls` is the per-round-trip cost every extra command buffer pays.
pub(super) static CALLS: AtomicU64 = AtomicU64::new(0);
pub(super) static DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub(super) static ENCODE_NS: AtomicU64 = AtomicU64::new(0);
pub(super) static WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// GPU-side execution time (GPUEndTime - GPUStartTime summed over buffers)
/// and the scheduling delay before it (GPUStartTime - kernelStartTime).
/// The difference between WAIT_NS and their sum is time the CPU spent
/// blocked while the GPU had nothing of ours to run - the number that says
/// whether decode is bound by kernels or by getting work to them.
pub(super) static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);
pub(super) static SCHED_NS: AtomicU64 = AtomicU64::new(0);

/// `(gpu_busy_ns, sched_ns)` since process start.
pub fn gpu_time_stats() -> (u64, u64) {
    (
        GPU_BUSY_NS.load(Ordering::Relaxed),
        SCHED_NS.load(Ordering::Relaxed),
    )
}

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

/// `(calls, dispatches, encode_ns, wait_ns)` since process start.
pub fn stats() -> (u64, u64, u64, u64) {
    (
        CALLS.load(Ordering::Relaxed),
        DISPATCHES.load(Ordering::Relaxed),
        ENCODE_NS.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed),
    )
}
