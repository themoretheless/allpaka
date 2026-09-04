//! Shared admission accounting. Reservations precede allocations and live with owners.
pub(super) use allpaka_backend::memory::{Budget, Exhausted, Lease};

pub(super) fn snapshot(budget: &Budget) -> serde_json::Value {
    let state = budget.snapshot();
    serde_json::json!({"limit_bytes":state.limit_bytes, "reserved_bytes":state.reserved_bytes,
        "peak_reserved_bytes":state.peak_reserved_bytes,
        "scope":"weights,prefix capacity,session KV/SSM/RoPE; excludes scratch and process overhead"})
}

/// Matches page-rounded f16 KV and f32 recurrent storage; reserves twice the
/// logical RoPE size for Vec growth. Does not include MTP rollback slots.
pub(super) fn session_bytes(model: &allpaka_model::Model<'_>, capacity: usize) -> anyhow::Result<u64> {
    let c = &model.config;
    let layers = u128::from(c.n_layers);
    let page = |bytes: u128| bytes.div_ceil(16384) * 16384;
    let kv = page(layers * c.kv_dim() as u128 * capacity as u128 * 4);
    let ssm = c.ssm.as_ref().map_or(0, |s| {
        let channels = u128::from(s.d_inner) + 2 * u128::from(s.n_group) * u128::from(s.d_state);
        let conv = u128::from(s.d_conv.saturating_sub(1)) * channels;
        let state = u128::from(s.dt_rank) * u128::from(s.d_state).pow(2);
        page(layers * (conv + state) * 4)
    });
    let rope = capacity as u128 * u128::from(c.rope_dim / 2) * 8 * 2;
    Ok(u64::try_from(kv + ssm + rope)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    #[test]
    fn failed_reservation_is_atomic_and_drop_releases() {
        let budget = Budget::new(100);
        let lease = budget.reserve(70).unwrap();
        assert!(budget.reserve(31).is_err());
        assert_eq!(snapshot(&budget)["reserved_bytes"], 70);
        drop(lease);
        assert!(budget.reserve(100).is_ok());
    }
    #[test]
    fn replacement_peak_and_overflow_are_charged() {
        let budget = Budget::new(u64::MAX);
        let lease = budget.reserve(u64::MAX).unwrap();
        assert!(budget.reserve(1).is_err());
        drop(lease);
        assert!(budget.reserve(1).is_ok());
    }
    #[test]
    fn concurrent_reservations_cannot_overcommit() {
        let budget = Budget::new(100);
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8).map(|_| {
            let budget = budget.clone(); let barrier = barrier.clone();
            std::thread::spawn(move || {
                let lease = budget.reserve(60);
                barrier.wait();
                lease.is_ok()
            })
        }).collect();
        assert_eq!(handles.into_iter().filter(|h| h.thread().id() != std::thread::current().id())
            .map(|h| usize::from(h.join().unwrap())).sum::<usize>(), 1);
        assert_eq!(snapshot(&budget)["reserved_bytes"], 0);
    }
}
