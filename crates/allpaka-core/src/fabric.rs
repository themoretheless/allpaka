//! The network between the nodes.
//!
//! A cluster assembled from whatever is on hand rarely has one uniform network.
//! A 10 GbE run between two desktops and Wi-Fi to a laptop are different worlds:
//! roughly a hundred times the throughput and a hundredth of the tail latency.
//! Planning with a single averaged link would pick cuts that are wrong on both
//! hops, so each pair of nodes carries its own measurement.

use crate::Link;

/// Measured links between node pairs, plus an optional fallback for pairs that
/// have not been measured.
#[derive(Debug, Clone, Default)]
pub struct Fabric {
    /// `(node_a, node_b, link)`. Links are symmetric, so order does not matter.
    edges: Vec<(usize, usize, Link)>,
    /// Applied to any pair with no specific measurement. `None` means an
    /// unmeasured pair is treated as unusable rather than optimistically
    /// guessed - a plan built on an invented link is worse than no plan.
    fallback: Option<Link>,
}

impl Fabric {
    pub fn new() -> Self {
        Self::default()
    }

    /// A cluster where every pair shares one measured link.
    pub fn uniform(link: Link) -> Self {
        Self { edges: Vec::new(), fallback: Some(link) }
    }

    pub fn with_fallback(mut self, link: Link) -> Self {
        self.fallback = Some(link);
        self
    }

    pub fn connect(mut self, a: usize, b: usize, link: Link) -> Self {
        self.edges.retain(|(x, y, _)| !same_pair(*x, *y, a, b));
        self.edges.push((a, b, link));
        self
    }

    /// The link between two nodes, or `None` if that hop was never measured
    /// and no fallback was set.
    pub fn between(&self, a: usize, b: usize) -> Option<&Link> {
        self.edges
            .iter()
            .find(|(x, y, _)| same_pair(*x, *y, a, b))
            .map(|(_, _, l)| l)
            .or(self.fallback.as_ref())
    }

    /// The link applied to pairs with no specific measurement, if any.
    pub fn fallback(&self) -> Option<&Link> {
        self.fallback.as_ref()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() && self.fallback.is_none()
    }

    /// Every measured hop, for reporting.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, &Link)> {
        self.edges.iter().map(|(a, b, l)| (*a, *b, l))
    }
}

fn same_pair(x: usize, y: usize, a: usize, b: usize) -> bool {
    (x == a && y == b) || (x == b && y == a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(mbps: f64) -> Link {
        Link { throughput_bytes_per_sec: mbps * 1e6, rtt_p50_secs: 0.001, rtt_p99_secs: 0.002 }
    }

    #[test]
    fn a_link_is_found_from_either_direction() {
        let f = Fabric::new().connect(0, 1, link(1200.0));
        assert!(f.between(0, 1).is_some());
        assert_eq!(
            f.between(1, 0).unwrap().throughput_bytes_per_sec,
            f.between(0, 1).unwrap().throughput_bytes_per_sec
        );
    }

    #[test]
    fn an_unmeasured_pair_is_unusable_without_a_fallback() {
        let f = Fabric::new().connect(0, 1, link(1200.0));
        assert!(f.between(0, 2).is_none());
    }

    #[test]
    fn a_fallback_covers_unmeasured_pairs_only() {
        let f = Fabric::new().connect(0, 1, link(1200.0)).with_fallback(link(40.0));
        assert_eq!(f.between(0, 1).unwrap().throughput_bytes_per_sec, 1200e6);
        assert_eq!(f.between(0, 2).unwrap().throughput_bytes_per_sec, 40e6);
    }

    #[test]
    fn reconnecting_a_pair_replaces_the_old_measurement() {
        let f = Fabric::new().connect(0, 1, link(40.0)).connect(1, 0, link(1200.0));
        assert_eq!(f.edges().count(), 1);
        assert_eq!(f.between(0, 1).unwrap().throughput_bytes_per_sec, 1200e6);
    }
}
