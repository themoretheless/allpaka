//! Versioned, model- and device-specific autotune cache.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::benchmark_report::{model_fingerprint, BenchmarkReport};

pub const AUTOTUNE_SCHEMA_VERSION: u32 = 1;
pub const KERNEL_ABI_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct AutotuneKey {
    pub device_registry_id: String,
    pub metal_os_version: String,
    pub model_fingerprint: String,
    pub model_architecture: String,
    pub model_dimensions: String,
    pub tensor_census_hash: String,
    pub kernel_abi_version: u32,
    /// What the profile was chosen for. Entries written before workloads
    /// existed were chosen for decode rate, which is `latency`.
    #[serde(default = "Workload::default_name")]
    pub workload: String,
}

/// What a deployment cares about most; decides which measured profile wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Workload {
    /// Interactive chat: the decode rate is the token latency a user feels.
    #[default]
    Latency,
    /// Batch and retrieval work: long prompts, short answers; what counts is
    /// tokens through the engine per second, prompt included.
    Throughput,
    /// Shared or small machines: the least working memory among the profiles
    /// that keep decode within [`MEMORY_SLACK`] of the fastest.
    Memory,
}

/// The request shape `Throughput` is scored on: a long prompt and a short answer.
const THROUGHPUT_PROMPT_TOKENS: f64 = 2048.0;
const THROUGHPUT_NEW_TOKENS: f64 = 256.0;
/// How much decode rate `Memory` gives up for a smaller footprint.
const MEMORY_SLACK: f64 = 0.10;

impl Workload {
    pub fn name(self) -> &'static str {
        match self {
            Self::Latency => "latency",
            Self::Throughput => "throughput",
            Self::Memory => "memory",
        }
    }

    fn default_name() -> String {
        Self::Latency.name().to_owned()
    }

    /// `ALLPAKA_WORKLOAD`, or `latency` when it is unset.
    pub fn from_env() -> Result<Self> {
        match std::env::var("ALLPAKA_WORKLOAD") {
            Ok(value) => value.parse(),
            Err(_) => Ok(Self::Latency),
        }
    }
}

impl std::str::FromStr for Workload {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "latency" => Ok(Self::Latency),
            "throughput" => Ok(Self::Throughput),
            "memory" => Ok(Self::Memory),
            other => anyhow::bail!("unknown workload {other:?}: use latency, throughput or memory"),
        }
    }
}

/// One measured profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub profile: String,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    /// Working memory the run held beyond the weights, when it was measured.
    pub memory_bytes: Option<u64>,
}

impl Candidate {
    /// Tokens per second over the throughput request: prompt and answer
    /// together, each at its own rate.
    pub fn request_tok_s(&self) -> f64 {
        let seconds = THROUGHPUT_PROMPT_TOKENS / self.prefill_tok_s
            + THROUGHPUT_NEW_TOKENS / self.decode_tok_s;
        (THROUGHPUT_PROMPT_TOKENS + THROUGHPUT_NEW_TOKENS) / seconds
    }
}

/// The candidate that serves `workload` best and the rate it is recorded
/// with: decode for latency and memory, whole-request rate for throughput.
/// Earlier candidates win ties, so the order they are measured in is the
/// order of preference.
pub fn choose(candidates: &[Candidate], workload: Workload) -> Option<(usize, f64)> {
    let best_by = |score: &dyn Fn(&Candidate) -> f64| {
        candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, score(c)))
            .filter(|(_, score)| score.is_finite())
            .reduce(|best, next| if next.1 > best.1 { next } else { best })
    };
    match workload {
        Workload::Latency => best_by(&|c| c.decode_tok_s),
        Workload::Throughput => best_by(&Candidate::request_tok_s),
        Workload::Memory => {
            let (_, fastest) = best_by(&|c| c.decode_tok_s)?;
            candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| c.decode_tok_s >= fastest * (1.0 - MEMORY_SLACK))
                // A profile whose memory was not measured cannot win on memory.
                .min_by_key(|(_, c)| c.memory_bytes.unwrap_or(u64::MAX))
                .map(|(i, c)| (i, c.decode_tok_s))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TunedParameters {
    pub attention_kernel: String,
    pub matmul_tile: String,
    pub rows_per_threadgroup: usize,
    pub prefill_chunk: usize,
    pub command_strategy: String,
    pub expert_grouping: String,
    pub gpu_window_bytes: u64,
    pub kv_block_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutotuneEntry {
    pub parameters: TunedParameters,
    pub objective_tok_s: f64,
    pub measured_at_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AutotuneCache {
    pub schema_version: u32,
    pub entries: Vec<AutotuneRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutotuneRecord {
    pub key: AutotuneKey,
    pub entry: AutotuneEntry,
}

impl AutotuneCache {
    pub fn empty() -> Self {
        Self {
            schema_version: AUTOTUNE_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::empty());
        }
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading autotune cache {}", path.display()))?;
        let cache: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing autotune cache {}", path.display()))?;
        if cache.schema_version != AUTOTUNE_SCHEMA_VERSION {
            return Ok(Self::empty());
        }
        Ok(cache)
    }

    pub fn get(&self, key: &AutotuneKey) -> Option<&AutotuneEntry> {
        if key.kernel_abi_version != KERNEL_ABI_VERSION {
            return None;
        }
        self.entries
            .iter()
            .find(|record| &record.key == key)
            .map(|record| &record.entry)
    }

    pub fn insert(&mut self, key: AutotuneKey, entry: AutotuneEntry) {
        if let Some(record) = self.entries.iter_mut().find(|record| record.key == key) {
            record.entry = entry;
        } else {
            self.entries.push(AutotuneRecord { key, entry });
        }
    }

    pub fn save_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating autotune directory {}", parent.display()))?;
        }
        let temporary = path.with_extension("tmp");
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(&temporary, bytes)
            .with_context(|| format!("writing autotune cache {}", temporary.display()))?;
        std::fs::rename(&temporary, path)
            .with_context(|| format!("installing autotune cache {}", path.display()))?;
        Ok(())
    }
}

pub fn run(
    model_path: &Path,
    cache_path: Option<&Path>,
    force: bool,
    workload: Workload,
) -> Result<()> {
    let file = allpaka_gguf::GgufFile::open(model_path)?;
    let key = key_for(model_path, &file, workload);
    let cache_path = cache_path
        .map(ToOwned::to_owned)
        .unwrap_or_else(default_cache_path);
    let mut cache = AutotuneCache::load(&cache_path)?;
    if !force {
        if let Some(hit) = cache.get(&key) {
            println!(
                "autotune cache hit: workload={} profile={} objective={:.1} tok/s ({})",
                workload.name(),
                hit.parameters.command_strategy,
                hit.objective_tok_s,
                cache_path.display()
            );
            return Ok(());
        }
    }

    let executable = std::env::current_exe()?;
    let mut candidates: Vec<Candidate> = Vec::new();
    // Throughput is about long prompts, so its prefill is measured on one.
    let prompt_tokens = if workload == Workload::Throughput {
        "512"
    } else {
        "128"
    };
    for profile in ["balanced", "max-performance", "safe"] {
        let report_path = std::env::temp_dir().join(format!(
            "allpaka-autotune-{}-{profile}.json",
            std::process::id()
        ));
        println!("autotune: measuring {profile} ...");
        let status = Command::new(&executable)
            .arg("bench")
            .arg("--engine")
            .arg(model_path)
            .env("ALLPAKA_PROFILE", profile)
            .env("ALLPAKA_BENCH_PP", prompt_tokens)
            .env("ALLPAKA_BENCH_REPORT", &report_path)
            .stdout(Stdio::null())
            .status()
            .with_context(|| format!("running autotune candidate {profile}"))?;
        anyhow::ensure!(status.success(), "autotune candidate {profile} failed");
        let report: BenchmarkReport = serde_json::from_slice(&std::fs::read(&report_path)?)?;
        std::fs::remove_file(&report_path).ok();
        let decode = measurement(&report, "decode")?;
        let prefill = measurement(&report, "prefill")?;
        let candidate = Candidate {
            profile: profile.to_string(),
            decode_tok_s: decode.summary.median,
            prefill_tok_s: prefill.summary.median,
            memory_bytes: decode.peak_memory_bytes.max(prefill.peak_memory_bytes),
        };
        println!(
            "  {profile}: decode={:.1} prefill={:.1} tok/s, request={:.1} tok/s, working memory={}",
            candidate.decode_tok_s,
            candidate.prefill_tok_s,
            candidate.request_tok_s(),
            match candidate.memory_bytes {
                Some(bytes) => format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64),
                None => "not measured".to_owned(),
            }
        );
        candidates.push(candidate);
    }
    let (chosen, objective) =
        choose(&candidates, workload).context("autotune produced no successful candidates")?;
    let profile = candidates[chosen].profile.clone();
    let resolved: allpaka_backend::profile::RuntimeProfile = profile.parse()?;
    let policy = resolved.resolve().policy;
    let entry = AutotuneEntry {
        parameters: TunedParameters {
            attention_kernel: "auto".into(),
            matmul_tile: "auto".into(),
            rows_per_threadgroup: 0,
            prefill_chunk: 512,
            command_strategy: profile.clone(),
            expert_grouping: if policy.gpu_route { "gpu" } else { "cpu" }.into(),
            gpu_window_bytes: 0,
            kv_block_tokens: 256,
        },
        objective_tok_s: objective,
        measured_at_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    };
    cache.insert(key, entry);
    cache.save_atomic(&cache_path)?;
    println!(
        "autotune selected {profile} for {}: {objective:.1} tok/s; cached at {}",
        workload.name(),
        cache_path.display()
    );
    Ok(())
}

/// Apply a compatible cached profile before model loading. Explicit process
/// configuration always wins; callers also skip this when a config file was
/// supplied.
pub fn apply_cached(model_path: &Path, cache_path: Option<&Path>) -> Result<bool> {
    if std::env::var_os("ALLPAKA_PROFILE").is_some() {
        return Ok(false);
    }
    let cache_path = cache_path
        .map(ToOwned::to_owned)
        .unwrap_or_else(default_cache_path);
    if !cache_path.exists() {
        return Ok(false);
    }
    let cache = AutotuneCache::load(&cache_path)?;
    if cache.entries.is_empty() {
        return Ok(false);
    }
    let file = allpaka_gguf::GgufFile::open(model_path)?;
    // The profile tuned for this deployment's workload, not just for the
    // model and device: `ALLPAKA_WORKLOAD`, `latency` when unset.
    let workload = Workload::from_env()?;
    let key = key_for(model_path, &file, workload);
    let Some(hit) = cache.get(&key) else {
        return Ok(false);
    };
    // Startup is still single-threaded here. Publish the resolved profile so
    // legacy capability reporters and remaining env-backed probes cannot
    // disagree with the installed typed policy.
    std::env::set_var("ALLPAKA_PROFILE", &hit.parameters.command_strategy);
    let profile: allpaka_backend::profile::RuntimeProfile =
        hit.parameters.command_strategy.parse()?;
    let resolved = profile.resolve_with_env();
    let _ = allpaka_backend::runtime::install(resolved.policy);
    println!(
        "autotune cache applied: workload={} profile={} objective={:.1} tok/s ({})",
        workload.name(),
        hit.parameters.command_strategy,
        hit.objective_tok_s,
        cache_path.display()
    );
    Ok(true)
}

fn key_for(model_path: &Path, file: &allpaka_gguf::GgufFile, workload: Workload) -> AutotuneKey {
    AutotuneKey {
        workload: workload.name().to_owned(),
        device_registry_id: host_value("sysctl", &["-n", "machdep.cpu.brand_string"]),
        metal_os_version: host_value("sw_vers", &["-productVersion"]),
        model_fingerprint: model_fingerprint(model_path, file),
        model_architecture: file.architecture().to_string(),
        model_dimensions: format!("tensors={}", file.tensors().len()),
        tensor_census_hash: tensor_census(file),
        kernel_abi_version: KERNEL_ABI_VERSION,
    }
}

fn measurement<'a>(
    report: &'a BenchmarkReport,
    name: &str,
) -> Result<&'a crate::benchmark_report::Measurement> {
    report
        .measurements
        .iter()
        .find(|measurement| measurement.name == name)
        .with_context(|| format!("autotune report has no {name} measurement"))
}

fn tensor_census(file: &allpaka_gguf::GgufFile) -> String {
    let mut rows = file
        .tensors()
        .iter()
        .map(|tensor| format!("{:?}:{}", tensor.ggml_type, tensor.byte_size().unwrap_or(0)))
        .collect::<Vec<_>>();
    rows.sort();
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    rows.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn host_value(command: &str, args: &[&str]) -> String {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn default_cache_path() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".cache/allpaka/autotune.json")
}

#[cfg(test)]
mod tests {
    use super::{
        choose, AutotuneCache, AutotuneEntry, AutotuneKey, Candidate, TunedParameters, Workload,
        KERNEL_ABI_VERSION,
    };

    fn key(model: &str) -> AutotuneKey {
        AutotuneKey {
            device_registry_id: "m4-max".into(),
            metal_os_version: "test".into(),
            model_fingerprint: model.into(),
            model_architecture: "qwen3moe".into(),
            model_dimensions: "48x2048".into(),
            tensor_census_hash: "q4k-q6k".into(),
            kernel_abi_version: KERNEL_ABI_VERSION,
            workload: "latency".into(),
        }
    }

    fn entry() -> AutotuneEntry {
        AutotuneEntry {
            parameters: TunedParameters {
                attention_kernel: "gqa-v2".into(),
                matmul_tile: "32x32".into(),
                rows_per_threadgroup: 8,
                prefill_chunk: 512,
                command_strategy: "one-buffer".into(),
                expert_grouping: "gpu-topk".into(),
                gpu_window_bytes: 18 << 30,
                kv_block_tokens: 256,
            },
            objective_tok_s: 123.4,
            measured_at_unix_ms: 1,
        }
    }

    #[test]
    fn cache_round_trip_keeps_models_isolated() {
        let path = std::env::temp_dir().join(format!(
            "allpaka-autotune-{}-{}.json",
            std::process::id(),
            // Thread names hold `::`, which Windows rejects in file names.
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ));
        let mut cache = AutotuneCache::empty();
        cache.insert(key("model-a"), entry());
        cache.save_atomic(&path).unwrap();
        let loaded = AutotuneCache::load(&path).unwrap();
        assert!(loaded.get(&key("model-a")).is_some());
        assert!(loaded.get(&key("model-b")).is_none());
        std::fs::remove_file(path).unwrap();
    }

    fn candidate(profile: &str, decode: f64, prefill: f64, gib: Option<u64>) -> Candidate {
        Candidate {
            profile: profile.into(),
            decode_tok_s: decode,
            prefill_tok_s: prefill,
            memory_bytes: gib.map(|g| g << 30),
        }
    }

    #[test]
    fn the_workload_decides_which_profile_wins() {
        // Fast to decode but slow on prompts; the reverse; and a lean one.
        let measured = [
            candidate("balanced", 80.0, 3000.0, Some(6)),
            candidate("max-performance", 79.0, 3900.0, Some(9)),
            candidate("safe", 73.0, 2500.0, Some(2)),
        ];
        let pick = |workload| {
            let (i, score) = choose(&measured, workload).unwrap();
            (measured[i].profile.as_str(), score)
        };
        assert_eq!(pick(Workload::Latency), ("balanced", 80.0));
        let (profile, score) = pick(Workload::Throughput);
        assert_eq!(profile, "max-performance");
        // 2304 tokens in 2048/3900 + 256/79 seconds, against 2048/3000 + 256/80.
        assert!((score - 2304.0 / (2048.0 / 3900.0 + 256.0 / 79.0)).abs() < 1e-9);
        // 73 is within 10% of 80, and it holds the least memory.
        assert_eq!(pick(Workload::Memory), ("safe", 73.0));
    }

    #[test]
    fn memory_does_not_trade_away_more_than_the_slack() {
        let measured = [
            candidate("balanced", 80.0, 3000.0, Some(6)),
            candidate("safe", 60.0, 2500.0, Some(1)),
            candidate("max-performance", 79.0, 3900.0, None),
        ];
        // 60 is 25% slower: out. The unmeasured one cannot win on memory.
        let (i, _) = choose(&measured, Workload::Memory).unwrap();
        assert_eq!(measured[i].profile, "balanced");
        // Ties go to the profile measured first.
        let tie = [
            candidate("a", 50.0, 100.0, Some(1)),
            candidate("b", 50.0, 100.0, Some(1)),
        ];
        for workload in [Workload::Latency, Workload::Throughput, Workload::Memory] {
            assert_eq!(choose(&tie, workload).unwrap().0, 0);
        }
        assert!(choose(&[], Workload::Latency).is_none());
    }

    #[test]
    fn an_entry_from_before_workloads_is_a_latency_entry() {
        let old = r#"{"device_registry_id":"m4-max","metal_os_version":"test","model_fingerprint":"model-a",
            "model_architecture":"qwen3moe","model_dimensions":"48x2048","tensor_census_hash":"q4k-q6k",
            "kernel_abi_version":2}"#;
        let parsed: AutotuneKey = serde_json::from_str(old).unwrap();
        assert_eq!(parsed, key("model-a"));
        let mut other = key("model-a");
        other.workload = "throughput".into();
        assert_ne!(parsed, other);
        assert_eq!("memory".parse::<Workload>().unwrap(), Workload::Memory);
        assert!("fastest".parse::<Workload>().is_err());
    }
}
