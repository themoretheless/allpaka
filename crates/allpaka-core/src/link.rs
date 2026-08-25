//! Measured properties of the network path between two nodes.
//!
//! These are measurements, not specifications. On Wi-Fi the gap between the two
//! is large enough that planning from the advertised link rate produces answers
//! that are wrong by an order of magnitude.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    /// Sustained one-way throughput in bytes/sec, as measured.
    pub throughput_bytes_per_sec: f64,
    /// Median round-trip time in seconds for a small message.
    pub rtt_p50_secs: f64,
    /// 99th percentile round-trip time in seconds.
    ///
    /// This is the number that matters on Wi-Fi. Decode issues one round trip
    /// per token per cut, so a long tail is paid over and over rather than
    /// averaged away.
    pub rtt_p99_secs: f64,
}

impl Link {
    /// Seconds to push `bytes` one way, using median latency.
    ///
    /// A pipeline stage sends its activations and immediately moves on; it does
    /// not wait for an acknowledgement. So a hop costs one-way latency, which
    /// is estimated as half the round trip. That halving is an approximation -
    /// the true one-way p99 is not exactly half the round-trip p99, and
    /// asymmetric paths break it entirely - but it is far closer than charging
    /// a full round trip for a fire-and-forget send.
    pub fn one_way_p50(&self, bytes: u64) -> f64 {
        self.rtt_p50_secs / 2.0 + self.serialisation_time(bytes)
    }

    /// Same, using the tail latency. This is the number to plan against for
    /// interactive decode, because a per-token stall is visible to the user.
    pub fn one_way_p99(&self, bytes: u64) -> f64 {
        self.rtt_p99_secs / 2.0 + self.serialisation_time(bytes)
    }

    /// Seconds to move `bytes` and get a reply, for exchanges that genuinely
    /// block on an answer.
    pub fn round_trip_p99(&self, bytes: u64) -> f64 {
        self.rtt_p99_secs + self.serialisation_time(bytes)
    }

    fn serialisation_time(&self, bytes: u64) -> f64 {
        if self.throughput_bytes_per_sec <= 0.0 {
            return f64::INFINITY;
        }
        bytes as f64 / self.throughput_bytes_per_sec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link() -> Link {
        Link {
            throughput_bytes_per_sec: 1e9,
            rtt_p50_secs: 0.0002,
            rtt_p99_secs: 0.0004,
        }
    }

    #[test]
    fn a_one_way_hop_costs_half_a_round_trip() {
        let l = link();
        assert_eq!(l.one_way_p99(0), l.rtt_p99_secs / 2.0);
        assert_eq!(l.round_trip_p99(0), l.rtt_p99_secs);
    }

    /// A 10 KB activation over a fast link is dominated by latency, not by
    /// serialisation. This is why compressing the payload buys almost nothing.
    #[test]
    fn latency_dominates_a_decode_sized_payload() {
        let l = link();
        let latency = l.rtt_p99_secs / 2.0;
        let serialisation = l.one_way_p99(10_240) - latency;
        assert!(
            serialisation < latency / 10.0,
            "serialisation {serialisation:.9}s should be negligible against {latency:.9}s"
        );
    }

    #[test]
    fn a_dead_link_costs_infinity_rather_than_dividing_by_zero() {
        let l = Link { throughput_bytes_per_sec: 0.0, rtt_p50_secs: 1.0, rtt_p99_secs: 1.0 };
        assert!(l.one_way_p99(1).is_infinite());
    }
}
