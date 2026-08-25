//! Speculative decoding: fewer forward passes per token of output.
//!
//! The network cost of a pipeline is paid *per forward pass*, not per token.
//! Ordinary decode makes those the same thing: one pass, one token, one hop.
//! Speculative decoding breaks that identity.
//!
//! A small draft model runs entirely on the head machine - no network at all -
//! and guesses `K` tokens ahead. The full model then verifies all `K + 1`
//! candidate positions in a **single** forward pass, because verification is
//! just a batched forward, and a batch of 16 costs a memory-bound model almost
//! exactly what a batch of 1 costs: the weights are read once either way. So
//! one round trip now buys several accepted tokens instead of one.
//!
//! # What it does not do
//!
//! It does not rescue a split that the link would otherwise lose. A cycle
//! divides *both* the compute and the network by the same expected-accepted
//! factor, so the comparison that decides the placement -
//!
//! ```text
//! network_per_pass  <  compute_saved_per_pass
//! ```
//!
//! - comes out identical with and without speculation. That is asserted in
//! `plan::tests::speculation_does_not_change_the_split_verdict`. Speculation is
//! a throughput multiplier applied to whichever placement already won, not a
//! cure for latency.
//!
//! # What else was considered
//!
//! * Compressing the activation: near useless. A 10 KB payload is microseconds
//!   of serialisation against a millisecond of latency.
//! * Fewer cuts: already minimal at one cut for two machines.
//! * Tensor parallelism instead of pipeline: far worse - an all-reduce inside
//!   every layer rather than one hand-off per model.
//! * Batching concurrent requests: raises aggregate throughput, never lowers
//!   the latency of any single token.
//!
//! The one genuine way to cut network cost per token is to emit more tokens
//! per forward pass, which is what this module models.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Speculation {
    /// Weight bytes of the draft model. It runs on the head node only.
    pub draft_weight_bytes: u64,
    /// How many tokens the draft proposes per cycle, `K`.
    pub draft_tokens: u32,
    /// Probability that any given drafted token is accepted, `alpha`.
    ///
    /// This is a property of the draft/target pair and the text, and it must be
    /// measured. A draft from the same family at roughly a tenth the size
    /// typically lands somewhere in 0.6-0.8, but that is a range to check, not
    /// a number to trust.
    pub acceptance_rate: f64,
}

impl Speculation {
    /// Expected tokens accepted per verification pass.
    ///
    /// Drafted tokens are accepted in order and the run stops at the first
    /// rejection, so the count is the sum of the probabilities that the run
    /// reaches each position: `1 + a + a^2 + ... + a^K`. The leading 1 is the
    /// token the verification pass itself produces, which is always correct -
    /// this is why speculative decoding can never be slower than one token per
    /// pass, and why it is lossless rather than an approximation.
    pub fn expected_accepted(&self) -> f64 {
        let a = self.acceptance_rate.clamp(0.0, 1.0);
        let k = self.draft_tokens as f64;
        if a >= 1.0 {
            return k + 1.0;
        }
        (1.0 - a.powf(k + 1.0)) / (1.0 - a)
    }

    /// Positions carried through the verification pass, `K + 1`.
    pub fn verify_batch(&self) -> u32 {
        self.draft_tokens + 1
    }
}

/// What speculation costs and buys for one particular placement.
#[derive(Debug, Clone)]
pub struct SpeculativeCost {
    /// Time the draft model spends per cycle, on the head node.
    pub draft_secs_per_cycle: f64,
    /// Full-model compute for the verification pass.
    pub verify_compute_secs: f64,
    /// Network for the verification pass. One cycle, not one token.
    pub verify_network_secs: f64,
    pub expected_accepted: f64,
    pub secs_per_token: f64,
}

impl SpeculativeCost {
    pub fn cycle_secs(&self) -> f64 {
        self.draft_secs_per_cycle + self.verify_compute_secs + self.verify_network_secs
    }

    /// Network wait per accepted token. The number speculation exists to lower.
    pub fn network_secs_per_token(&self) -> f64 {
        if self.expected_accepted <= 0.0 {
            return f64::INFINITY;
        }
        self.verify_network_secs / self.expected_accepted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(k: u32, a: f64) -> Speculation {
        Speculation { draft_weight_bytes: 1 << 30, draft_tokens: k, acceptance_rate: a }
    }

    #[test]
    fn drafting_nothing_still_yields_the_one_free_token() {
        assert_eq!(spec(0, 0.7).expected_accepted(), 1.0);
    }

    #[test]
    fn a_rejected_draft_costs_nothing_in_correctness() {
        // Acceptance 0 means every draft is thrown away, but the verification
        // pass still emits its own token, so throughput never drops below 1.
        assert_eq!(spec(8, 0.0).expected_accepted(), 1.0);
    }

    #[test]
    fn perfect_acceptance_yields_the_whole_batch() {
        assert_eq!(spec(4, 1.0).expected_accepted(), 5.0);
    }

    #[test]
    fn expected_acceptance_matches_the_series_by_hand() {
        // 1 + 0.5 + 0.25 + 0.125 = 1.875
        let e = spec(3, 0.5).expected_accepted();
        assert!((e - 1.875).abs() < 1e-12, "got {e}");
    }

    #[test]
    fn expected_acceptance_never_exceeds_the_batch() {
        for k in 0..16 {
            for a in [0.0, 0.3, 0.7, 0.95, 1.0] {
                let s = spec(k, a);
                assert!(s.expected_accepted() <= s.verify_batch() as f64 + 1e-9);
            }
        }
    }

    #[test]
    fn drafting_further_helps_less_each_time() {
        // Diminishing returns: each extra drafted token is reached only if all
        // the earlier ones were accepted.
        let gain = |k| spec(k + 1, 0.7).expected_accepted() - spec(k, 0.7).expected_accepted();
        assert!(gain(4) < gain(1));
        assert!(gain(1) < gain(0));
    }

    #[test]
    fn network_per_token_falls_with_acceptance() {
        let mk = |accepted: f64| SpeculativeCost {
            draft_secs_per_cycle: 0.0,
            verify_compute_secs: 0.0,
            verify_network_secs: 0.001,
            expected_accepted: accepted,
            secs_per_token: 0.0,
        };
        assert!(mk(4.0).network_secs_per_token() < mk(1.0).network_secs_per_token());
        assert_eq!(mk(4.0).network_secs_per_token(), 0.00025);
    }
}
