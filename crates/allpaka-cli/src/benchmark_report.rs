//! Stable, machine-readable benchmark artifacts and regression policy.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub metadata: BenchmarkMetadata,
    pub measurements: Vec<Measurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkMetadata {
    pub generated_at_unix_ms: u128,
    pub git_commit: String,
    pub model: String,
    pub model_fingerprint: String,
    pub device: String,
    pub runtime_profile: String,
    pub resolved_overrides: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Measurement {
    pub name: String,
    pub tokens: usize,
    pub samples_tok_s: Vec<f64>,
    pub summary: Distribution,
    pub fast_path: FastPathStats,
    pub phases: Vec<PhaseMetric>,
    pub peak_memory_bytes: Option<u64>,
    pub kv_bytes_per_token: Option<u64>,
}

impl Measurement {
    pub fn new(name: impl Into<String>, tokens: usize, samples_tok_s: Vec<f64>) -> Self {
        let summary = Distribution::from_samples(&samples_tok_s);
        Self {
            name: name.into(),
            tokens,
            samples_tok_s,
            summary,
            fast_path: FastPathStats::default(),
            phases: Vec::new(),
            peak_memory_bytes: None,
            kv_bytes_per_token: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Distribution {
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
    pub min: f64,
    pub max: f64,
}

impl Distribution {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                median: 0.0,
                p10: 0.0,
                p90: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        Self {
            median: percentile(&sorted, 0.50),
            p10: percentile(&sorted, 0.10),
            p90: percentile(&sorted, 0.90),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        }
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FastPathStats {
    pub attempts: u64,
    pub successes: u64,
    pub declines: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhaseMetric {
    pub name: String,
    pub milliseconds: f64,
    pub calls: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RegressionPolicy {
    pub throughput_tolerance_percent: f64,
    pub fail_on_decline: bool,
}

impl Default for RegressionPolicy {
    fn default() -> Self {
        Self {
            throughput_tolerance_percent: 3.0,
            fail_on_decline: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionVerdict {
    pub passed: bool,
    pub throughput_change_percent: f64,
    pub reasons: Vec<String>,
}

impl RegressionPolicy {
    pub fn evaluate(self, baseline: &Measurement, candidate: &Measurement) -> RegressionVerdict {
        let change = if baseline.summary.median > 0.0 {
            (candidate.summary.median / baseline.summary.median - 1.0) * 100.0
        } else {
            0.0
        };
        let mut reasons = Vec::new();
        if change < -self.throughput_tolerance_percent {
            reasons.push(format!(
                "median throughput regressed by {:.2}% (limit {:.2}%)",
                -change, self.throughput_tolerance_percent
            ));
        }
        if self.fail_on_decline && candidate.fast_path.declines > 0 {
            reasons.push(format!(
                "fast path declined {} time(s)",
                candidate.fast_path.declines
            ));
        }
        RegressionVerdict {
            passed: reasons.is_empty(),
            throughput_change_percent: change,
            reasons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Measurement, RegressionPolicy};

    #[test]
    fn report_round_trips_as_json() {
        let measurement = Measurement::new("decode", 32, vec![100.0, 110.0, 120.0]);
        let encoded = serde_json::to_string(&measurement).unwrap();
        let decoded: Measurement = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, measurement);
        assert_eq!(measurement.summary.median, 110.0);
    }

    #[test]
    fn regression_policy_rejects_slow_or_declined_candidates() {
        let baseline = Measurement::new("decode", 32, vec![100.0]);
        let mut candidate = Measurement::new("decode", 32, vec![95.0]);
        candidate.fast_path.declines = 1;
        let verdict = RegressionPolicy::default().evaluate(&baseline, &candidate);
        assert!(!verdict.passed);
        assert_eq!(verdict.reasons.len(), 2);
    }
}
