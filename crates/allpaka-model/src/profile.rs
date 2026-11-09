//! Where a decode token's CPU time goes.
//!
//! Delegates atomic slot storage and span timing to [`allpaka_backend::telemetry`]
//! while preserving the model-level [`Phase`] enum and reporting helpers.

pub use allpaka_backend::telemetry::ProfileSpan as Span;

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

/// Start timing `phase` until the returned value goes out of scope.
#[inline]
pub fn span(phase: Phase) -> Span {
    allpaka_backend::telemetry::span_slot(phase as usize)
}

/// Nanoseconds per phase since the last `reset`.
#[inline]
pub fn take() -> [u64; NAMES.len()] {
    take_with_counts().0
}

/// Nanoseconds and span counts per phase since the last `reset`.
#[inline]
pub fn take_with_counts() -> ([u64; NAMES.len()], [u64; NAMES.len()]) {
    allpaka_backend::telemetry::take_slots_with_counts::<{ NAMES.len() }>()
}

#[inline]
pub fn reset() {
    allpaka_backend::telemetry::reset_slots();
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
