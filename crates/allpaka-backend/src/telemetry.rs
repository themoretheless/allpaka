//! Backend-neutral inference phase accounting.
//!
//! The hot path owns a [`PhaseTrace`] for one measured operation. Callers add
//! durations at existing synchronization boundaries, so tracing does not add
//! GPU waits or force command-buffer completion.

use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Phase {
    Embed,
    Norm,
    Qkv,
    Rope,
    Attention,
    Router,
    Experts,
    KvUpdate,
    CommandEncode,
    GpuWait,
    Other,
}

impl Phase {
    pub const ALL: [Self; 11] = [
        Self::Embed,
        Self::Norm,
        Self::Qkv,
        Self::Rope,
        Self::Attention,
        Self::Router,
        Self::Experts,
        Self::KvUpdate,
        Self::CommandEncode,
        Self::GpuWait,
        Self::Other,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Norm => "norm",
            Self::Qkv => "qkv",
            Self::Rope => "rope",
            Self::Attention => "attention",
            Self::Router => "router",
            Self::Experts => "experts",
            Self::KvUpdate => "kv_update",
            Self::CommandEncode => "command_encode",
            Self::GpuWait => "gpu_wait",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSample {
    pub phase: Phase,
    pub duration: Duration,
    pub calls: u64,
}

#[derive(Debug, Clone)]
pub struct PhaseTrace {
    durations: [Duration; Phase::ALL.len()],
    calls: [u64; Phase::ALL.len()],
}

impl Default for PhaseTrace {
    fn default() -> Self {
        Self {
            durations: [Duration::ZERO; Phase::ALL.len()],
            calls: [0; Phase::ALL.len()],
        }
    }
}

impl PhaseTrace {
    pub fn add(&mut self, phase: Phase, duration: Duration) {
        let index = phase as usize;
        self.durations[index] += duration;
        self.calls[index] += 1;
    }

    pub fn measure<T>(&mut self, phase: Phase, operation: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let value = operation();
        self.add(phase, started.elapsed());
        value
    }

    pub fn merge(&mut self, other: &Self) {
        for index in 0..Phase::ALL.len() {
            self.durations[index] += other.durations[index];
            self.calls[index] += other.calls[index];
        }
    }

    pub fn samples(&self) -> impl ExactSizeIterator<Item = PhaseSample> + '_ {
        Phase::ALL.into_iter().map(|phase| {
            let index = phase as usize;
            PhaseSample {
                phase,
                duration: self.durations[index],
                calls: self.calls[index],
            }
        })
    }

    pub fn accounted(&self) -> Duration {
        self.durations.iter().copied().sum()
    }
}

static GLOBAL_DURATIONS_NS: [AtomicU64; Phase::ALL.len()] =
    [const { AtomicU64::new(0) }; Phase::ALL.len()];
static GLOBAL_CALLS: [AtomicU64; Phase::ALL.len()] =
    [const { AtomicU64::new(0) }; Phase::ALL.len()];

/// Record a completed backend phase without introducing a lock or a new
/// synchronization point. GPU callers use this only after an existing wait.
pub fn record_global(phase: Phase, duration: Duration) {
    let index = phase as usize;
    let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
    GLOBAL_DURATIONS_NS[index].fetch_add(nanos, Ordering::Relaxed);
    GLOBAL_CALLS[index].fetch_add(1, Ordering::Relaxed);
}

pub fn reset_global() {
    for index in 0..Phase::ALL.len() {
        GLOBAL_DURATIONS_NS[index].store(0, Ordering::Relaxed);
        GLOBAL_CALLS[index].store(0, Ordering::Relaxed);
    }
}

/// Atomically take the process-wide backend snapshot and reset it for the
/// next benchmark interval.
pub fn take_global() -> PhaseTrace {
    let mut trace = PhaseTrace::default();
    for phase in Phase::ALL {
        let index = phase as usize;
        trace.durations[index] = Duration::from_nanos(
            GLOBAL_DURATIONS_NS[index].swap(0, Ordering::Relaxed),
        );
        trace.calls[index] = GLOBAL_CALLS[index].swap(0, Ordering::Relaxed);
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::{record_global, reset_global, take_global, Phase, PhaseTrace};
    use std::time::Duration;

    #[test]
    fn phases_merge_without_losing_call_counts() {
        let mut left = PhaseTrace::default();
        left.add(Phase::Attention, Duration::from_millis(3));
        let mut right = PhaseTrace::default();
        right.add(Phase::Attention, Duration::from_millis(4));
        right.add(Phase::Router, Duration::from_millis(2));

        left.merge(&right);
        let attention = left
            .samples()
            .find(|sample| sample.phase == Phase::Attention)
            .unwrap();
        assert_eq!(attention.duration, Duration::from_millis(7));
        assert_eq!(attention.calls, 2);
        assert_eq!(left.accounted(), Duration::from_millis(9));
    }

    #[test]
    fn global_snapshot_is_destructive() {
        reset_global();
        record_global(Phase::Qkv, Duration::from_millis(2));
        record_global(Phase::Qkv, Duration::from_millis(3));

        let first = take_global()
            .samples()
            .find(|sample| sample.phase == Phase::Qkv)
            .unwrap();
        assert_eq!(first.duration, Duration::from_millis(5));
        assert_eq!(first.calls, 2);
        assert_eq!(take_global().accounted(), Duration::ZERO);
    }
}
