//! Backend-neutral byte reservations tied to resource ownership, plus explicit
//! UMA vs dUMA memory architecture classification.
//!
//! Reserve before allocating. Keep the lease alongside the allocation and,
//! for asynchronous devices, until the last command using it has completed.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Physical memory topology between host CPU and accelerator GPU.
///
/// * [`MemoryArchitecture::Uma`] — Unified Memory Architecture (e.g. Apple
///   Silicon Metal): CPU and GPU address the same physical RAM pool. Page-aligned
///   host allocations wrapped via `newBufferWithBytesNoCopy` (`StorageModeShared`)
///   are read and mutated in place by both processors without H2D/D2H copies.
/// * [`MemoryArchitecture::Duma`] — Discrete / non-unified memory architecture
///   (e.g. discrete NVIDIA CUDA GPUs): GPU kernels read and mutate dedicated
///   VRAM (`CudaSlice`), requiring explicit Host-to-Device (H2D) and
///   Device-to-Host (D2H) transfers across PCIe/NVLink when state crosses
///   between CPU and GPU.
/// * [`MemoryArchitecture::HostOnly`] — No accelerator attached; all buffers
///   live solely in host RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryArchitecture {
    /// Unified Memory Architecture (shared physical RAM pool between CPU and GPU).
    Uma,
    /// Discrete Memory Architecture (separate host RAM and device VRAM).
    Duma,
    /// Host CPU memory only (no GPU backend active).
    HostOnly,
}

impl MemoryArchitecture {
    /// True when CPU and GPU share the same physical memory pool (`UMA`),
    /// allowing zero-copy buffer aliasing without H2D/D2H transfers.
    #[inline]
    pub const fn is_unified(self) -> bool {
        matches!(self, Self::Uma)
    }

    /// True when GPU execution uses discrete device VRAM (`dUMA`) that must be
    /// explicitly synchronized with host RAM via H2D/D2H copies.
    #[inline]
    pub const fn is_discrete(self) -> bool {
        matches!(self, Self::Duma)
    }

    /// True when host-side mutations to a wrapped region require an explicit
    /// upload (`H2D`) and GPU-side mutations require an explicit download (`D2H`)
    /// before the other processor can observe them.
    #[inline]
    pub const fn requires_explicit_sync(self) -> bool {
        matches!(self, Self::Duma)
    }

    /// True when page-aligned host memory can be wrapped in place for GPU
    /// kernels without allocating a separate device buffer.
    #[inline]
    pub const fn supports_zero_copy_aliases(self) -> bool {
        matches!(self, Self::Uma)
    }

    /// Short human-readable label for diagnostics and telemetry.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Uma => "UMA (unified shared memory)",
            Self::Duma => "dUMA (discrete device memory)",
            Self::HostOnly => "host-only RAM",
        }
    }
}

impl std::fmt::Display for MemoryArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone)]
pub struct Budget(Arc<State>);
struct State {
    limit: u64,
    used: AtomicU64,
    peak: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    pub limit_bytes: u64,
    pub reserved_bytes: u64,
    pub peak_reserved_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exhausted {
    pub requested: u64,
    pub available: u64,
}
impl std::fmt::Display for Exhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "memory admission rejected: requested {} bytes, available {} bytes",
            self.requested, self.available
        )
    }
}
impl std::error::Error for Exhausted {}

/// Not cloneable: one charge must be released exactly once. Share an owning
/// resource through Arc instead when multiple commands retain the allocation.
pub struct Lease {
    budget: Budget,
    bytes: u64,
}
impl Budget {
    pub fn new(limit: u64) -> Self {
        Self(Arc::new(State {
            limit,
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        }))
    }
    #[inline]
    pub fn reserve(&self, bytes: u64) -> Result<Lease, Exhausted> {
        let mut current = self.0.used.load(Ordering::Relaxed);
        loop {
            let available = self.0.limit - current;
            if bytes > available {
                return Err(Exhausted {
                    requested: bytes,
                    available,
                });
            }
            let next = current + bytes;
            match self.0.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.0.peak.fetch_max(next, Ordering::Relaxed);
                    return Ok(Lease {
                        budget: self.clone(),
                        bytes,
                    });
                }
                Err(actual) => current = actual,
            }
        }
    }
    #[inline]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            limit_bytes: self.0.limit,
            reserved_bytes: self.0.used.load(Ordering::Acquire),
            peak_reserved_bytes: self.0.peak.load(Ordering::Relaxed),
        }
    }
}
impl Lease {
    #[inline]
    pub fn budget(&self) -> Budget {
        self.budget.clone()
    }
}
impl Drop for Lease {
    #[inline]
    fn drop(&mut self) {
        self.budget.0.used.fetch_sub(self.bytes, Ordering::Release);
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
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let budget = budget.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let lease = budget.reserve(60);
                    barrier.wait();
                    lease.is_ok()
                })
            })
            .collect();
        assert_eq!(
            handles
                .into_iter()
                .map(|h| usize::from(h.join().unwrap()))
                .sum::<usize>(),
            1
        );
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
    #[test]
    fn memory_architecture_distinguishes_uma_and_duma_invariants() {
        assert!(MemoryArchitecture::Uma.is_unified());
        assert!(!MemoryArchitecture::Uma.is_discrete());
        assert!(!MemoryArchitecture::Uma.requires_explicit_sync());
        assert!(MemoryArchitecture::Uma.supports_zero_copy_aliases());

        assert!(!MemoryArchitecture::Duma.is_unified());
        assert!(MemoryArchitecture::Duma.is_discrete());
        assert!(MemoryArchitecture::Duma.requires_explicit_sync());
        assert!(!MemoryArchitecture::Duma.supports_zero_copy_aliases());

        assert!(!MemoryArchitecture::HostOnly.is_unified());
        assert!(!MemoryArchitecture::HostOnly.is_discrete());
        assert!(!MemoryArchitecture::HostOnly.requires_explicit_sync());
        assert!(!MemoryArchitecture::HostOnly.supports_zero_copy_aliases());
    }
}
