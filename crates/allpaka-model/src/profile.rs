//! Where a decode token's CPU time goes.
//!
//! The GPU side already reports itself (`allpaka_backend::gpu::stats`), and on
//! the 235B it accounts for a bit over half a token: the rest is this file's
//! subject. Guessing at it has a bad record here, so every phase of `forward`
//! is timed and the bench prints the split.
//!
//! Timing is a monotonic clock read per span, tens of nanoseconds against
//! phases measured in microseconds, and the counters are relaxed atomics that
//! are never contended on the decode path. The parallel attention section is
//! timed as wall time around the whole scope, so it counts elapsed time and
//! not the sum over threads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The phases, in the order `forward` runs them.
#[derive(Clone, Copy)]
pub enum Phase {
    Embed,
    AttnNorm,
    Qkv,
    QkNormRope,
    KvStore,
    Attend,
    AttnOut,
    FfnNorm,
    Router,
    ExpertSlice,
    Ffn,
    FfnCombine,
    Output,
}

pub const NAMES: [&str; 13] = [
    "embed",
    "attn norm",
    "qkv proj",
    "q/k norm + rope",
    "kv store",
    "attention",
    "attn out proj",
    "ffn norm",
    "router",
    "expert slice",
    "ffn (experts)",
    "ffn combine",
    "output proj",
];

const ZERO: AtomicU64 = AtomicU64::new(0);
static NS: [AtomicU64; NAMES.len()] = [ZERO; NAMES.len()];
static CALLS: [AtomicU64; NAMES.len()] = [ZERO; NAMES.len()];

/// A running phase; adds its elapsed time when dropped.
pub struct Span(usize, Instant);

impl Drop for Span {
    fn drop(&mut self) {
        NS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS[self.0].fetch_add(1, Ordering::Relaxed);
    }
}

/// Start timing `phase` until the returned value goes out of scope.
pub fn span(phase: Phase) -> Span {
    Span(phase as usize, Instant::now())
}

/// Nanoseconds per phase since the last `reset`.
pub fn take() -> [u64; NAMES.len()] {
    take_with_counts().0
}

/// Nanoseconds and span counts per phase since the last `reset`.
pub fn take_with_counts() -> ([u64; NAMES.len()], [u64; NAMES.len()]) {
    let mut out = [0u64; NAMES.len()];
    let mut calls = [0u64; NAMES.len()];
    for (o, n) in out.iter_mut().zip(NS.iter()) {
        *o = n.load(Ordering::Relaxed);
    }
    for (o, n) in calls.iter_mut().zip(CALLS.iter()) {
        *o = n.load(Ordering::Relaxed);
    }
    (out, calls)
}

pub fn reset() {
    for n in NS.iter() {
        n.store(0, Ordering::Relaxed);
    }
    for n in CALLS.iter() {
        n.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::{reset, span, take_with_counts, Phase};

    #[test]
    fn spans_record_duration_and_call_count() {
        reset();
        drop(span(Phase::Router));
        drop(span(Phase::Router));
        let (ns, calls) = take_with_counts();
        assert_eq!(calls[Phase::Router as usize], 2);
        assert!(ns[Phase::Router as usize] > 0);
    }
}
