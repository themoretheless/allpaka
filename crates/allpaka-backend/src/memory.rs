//! Backend-neutral byte reservations tied to resource ownership.
//!
//! Reserve before allocating. Keep the lease alongside the allocation and,
//! for asynchronous devices, until the last command using it has completed.
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Budget(Arc<Mutex<State>>);
struct State { limit: u64, used: u64, peak: u64 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    pub limit_bytes: u64,
    pub reserved_bytes: u64,
    pub peak_reserved_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exhausted { pub requested: u64, pub available: u64 }
impl std::fmt::Display for Exhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "memory admission rejected: requested {} bytes, available {} bytes", self.requested, self.available)
    }
}
impl std::error::Error for Exhausted {}

/// Not cloneable: one charge must be released exactly once. Share an owning
/// resource through Arc instead when multiple commands retain the allocation.
pub struct Lease { budget: Budget, bytes: u64 }
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
    pub fn snapshot(&self) -> Snapshot {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        Snapshot { limit_bytes: state.limit, reserved_bytes: state.used, peak_reserved_bytes: state.peak }
    }
}
impl Lease { pub fn budget(&self) -> Budget { self.budget.clone() } }
impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.budget.0.lock().unwrap_or_else(|e| e.into_inner());
        state.used -= self.bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejection_is_atomic_and_drop_releases() {
        let budget = Budget::new(100);
        let lease = budget.reserve(70).unwrap();
        assert!(budget.reserve(31).is_err());
        assert_eq!(budget.snapshot().reserved_bytes, 70);
        drop(lease);
        assert!(budget.reserve(100).is_ok());
    }
    #[test]
    fn overflow_cannot_overcommit() {
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
        assert_eq!(handles.into_iter().map(|h| usize::from(h.join().unwrap())).sum::<usize>(), 1);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }
    #[test]
    fn asynchronous_owner_keeps_charge_after_replacement() {
        let budget = Budget::new(100);
        let current = Arc::new(budget.reserve(40).unwrap());
        let command = current.clone();
        let replacement = budget.reserve(60).unwrap();
        drop(current);
        assert_eq!(budget.snapshot().reserved_bytes, 100);
        assert!(budget.reserve(1).is_err());
        drop(command);
        assert_eq!(budget.snapshot().reserved_bytes, 60);
        drop(replacement);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
        assert_eq!(budget.snapshot().peak_reserved_bytes, 100);
    }
}
