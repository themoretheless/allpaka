//! Pieces every accelerator backend shares: process-wide call counters, the
//! byte length of a quantized weight row, and the KV-cache bounds check in
//! front of attention. Backends add their own kernel limits on top.

use super::types::AttnReq;
use allpaka_dtype::GgmlType;
use std::sync::atomic::{AtomicU64, Ordering};

/// Where GPU wall time goes, accumulated across every call. The split that
/// matters: `WAIT_NS` is dead time the CPU spends blocked on the GPU, and its
/// ratio to `CALLS` is the per-round-trip cost every extra submission pays.
pub(crate) static CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub(crate) static ENCODE_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// GPU-side execution time and the scheduling delay before it, where the
/// backend can read them. The difference between `WAIT_NS` and their sum is
/// time the CPU spent blocked while the GPU had nothing of ours to run.
pub(crate) static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SCHED_NS: AtomicU64 = AtomicU64::new(0);

/// `(calls, dispatches, encode_ns, wait_ns)` since process start.
pub fn stats() -> (u64, u64, u64, u64) {
    (
        CALLS.load(Ordering::Relaxed),
        DISPATCHES.load(Ordering::Relaxed),
        ENCODE_NS.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed),
    )
}

/// `(gpu_busy_ns, sched_ns)` since process start.
pub fn gpu_time_stats() -> (u64, u64) {
    (
        GPU_BUSY_NS.load(Ordering::Relaxed),
        SCHED_NS.load(Ordering::Relaxed),
    )
}

/// Bytes in one weight row of `n_in` elements, or `None` when the type has
/// no fixed block layout or `n_in` is not a whole number of blocks.
pub(crate) fn row_bytes(ty: GgmlType, n_in: usize) -> Option<usize> {
    let be = ty.block_elements()? as usize;
    let bb = ty.block_bytes()? as usize;
    if n_in % be != 0 {
        return None;
    }
    Some(n_in / be * bb)
}

/// The request is self-consistent and its K and V rows (f16) fit a cache of
/// `cache_len` bytes. Kernel limits such as supported head dims are the
/// backend's; each backend's region type keeps its own length.
pub(crate) fn attn_bounds_ok(req: &AttnReq, cache_len: usize) -> bool {
    req.n_pos > 0
        && req.group > 0
        && req.q.len() == req.n_q_heads * req.head_dim
        && (req.k_off + req.n_pos * req.kv_dim) * 2 <= cache_len
        && (req.v_off + req.n_pos * req.kv_dim) * 2 <= cache_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_bytes_follows_the_block_layout() {
        assert_eq!(row_bytes(GgmlType::Q4K, 512), Some(2 * 144));
        assert_eq!(row_bytes(GgmlType::Q6K, 256), Some(210));
        assert_eq!(row_bytes(GgmlType::Q8_0, 64), Some(2 * 34));
        assert_eq!(row_bytes(GgmlType::F16, 10), Some(20));
        assert_eq!(row_bytes(GgmlType::F32, 3), Some(12));
        assert_eq!(row_bytes(GgmlType::Q4K, 100), None, "partial super-block");
        assert_eq!(row_bytes(GgmlType::Other(77), 256), None);
    }
}
