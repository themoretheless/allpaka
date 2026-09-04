//! Stable, machine-readable benchmark artifacts and regression policy.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

pub fn model_fingerprint(model_path: &Path, file: &allpaka_gguf::GgufFile) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    file.architecture().hash(&mut hasher);
    std::fs::metadata(model_path)
        .map(|meta| meta.len())
        .unwrap_or(0)
        .hash(&mut hasher);
    for tensor in file.tensors() {
        tensor.name.hash(&mut hasher);
        format!("{:?}", tensor.ggml_type).hash(&mut hasher);
        tensor.dims.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

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
    /// Tokens present in KV before the measured phase. Missing in legacy reports.
    #[serde(default)]
    pub context_tokens: Option<usize>,
    /// Exact input stream consumed by this phase, for workload compatibility.
    #[serde(default)]
    pub input_tokens: Vec<u32>,
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
            context_tokens: None,
            input_tokens: Vec::new(),
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
        if baseline.tokens != candidate.tokens
            || baseline.context_tokens != candidate.context_tokens
            || baseline.input_tokens != candidate.input_tokens
        {
            reasons.push("incompatible token workload or starting context".into());
        }
        if !self.throughput_tolerance_percent.is_finite()
            || self.throughput_tolerance_percent < 0.0
            || self.throughput_tolerance_percent >= 100.0
        {
            reasons.push("invalid throughput tolerance".into());
        }
        for measurement in [baseline, candidate] {
            if measurement.samples_tok_s.is_empty()
                || measurement.samples_tok_s.iter().any(|rate| !rate.is_finite() || *rate <= 0.0)
                || !measurement.summary.median.is_finite()
                || measurement.summary.median <= 0.0
            {
                reasons.push("invalid throughput samples".into());
            }
        }
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
    fn incompatible_context_and_nonfinite_rates_fail_closed() {
        let mut baseline = Measurement::new("decode", 32, vec![100.0]);
        baseline.context_tokens = Some(480);
        let mut candidate = baseline.clone();
        candidate.context_tokens = Some(0);
        assert!(!RegressionPolicy::default().evaluate(&baseline, &candidate).passed);
        candidate = baseline.clone();
        candidate.samples_tok_s = vec![f64::NAN];
        assert!(!RegressionPolicy::default().evaluate(&baseline, &candidate).passed);
    }

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
