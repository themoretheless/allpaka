//! Shared admission accounting. Reservations precede allocations and live with owners.
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(super) struct Budget(Arc<Mutex<State>>);
struct State { limit: u64, used: u64, peak: u64 }

#[derive(Debug)]
pub(super) struct Exhausted { requested: u64, available: u64 }
impl std::fmt::Display for Exhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "memory admission rejected: requested {} bytes, available {} bytes", self.requested, self.available)
    }
}
impl std::error::Error for Exhausted {}

pub(super) struct Lease { budget: Budget, bytes: u64 }
impl Budget {
    pub fn new(limit: u64) -> Self {
        Self(Arc::new(Mutex::new(State { limit, used: 0, peak: 0 })))
    }
    pub fn reserve(&self, bytes: u64) -> Result<Lease, Exhausted> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let available = state.limit - state.used;
        if bytes > available { return Err(Exhausted { requested: bytes, available }); }
        state.used += bytes;
        state.peak = state.peak.max(state.used);
        Ok(Lease { budget: self.clone(), bytes })
    }
    pub fn snapshot(&self) -> serde_json::Value {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        serde_json::json!({"limit_bytes":state.limit, "reserved_bytes":state.used,
            "peak_reserved_bytes":state.peak, "scope":"weights,prefix capacity,session KV/SSM/RoPE; excludes scratch and process overhead"})
    }
}
impl Lease { pub fn budget(&self) -> Budget { self.budget.clone() } }
impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.budget.0.lock().unwrap_or_else(|e| e.into_inner());
        state.used -= self.bytes;
    }
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
    #[test]
    fn failed_reservation_is_atomic_and_drop_releases() {
        let budget = Budget::new(100);
        let lease = budget.reserve(70).unwrap();
        assert!(budget.reserve(31).is_err());
        assert_eq!(budget.snapshot()["reserved_bytes"], 70);
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
        assert_eq!(budget.snapshot()["reserved_bytes"], 0);
    }
}
