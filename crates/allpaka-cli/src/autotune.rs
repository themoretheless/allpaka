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

pub fn run(model_path: &Path, cache_path: Option<&Path>, force: bool) -> Result<()> {
    let file = allpaka_gguf::GgufFile::open(model_path)?;
    let key = key_for(model_path, &file);
    let cache_path = cache_path
        .map(ToOwned::to_owned)
        .unwrap_or_else(default_cache_path);
    let mut cache = AutotuneCache::load(&cache_path)?;
    if !force {
        if let Some(hit) = cache.get(&key) {
            println!(
                "autotune cache hit: profile={} objective={:.1} tok/s ({})",
                hit.parameters.command_strategy,
                hit.objective_tok_s,
                cache_path.display()
            );
            return Ok(());
        }
    }

    let executable = std::env::current_exe()?;
    let mut best: Option<(String, BenchmarkReport)> = None;
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
            .env("ALLPAKA_BENCH_PP", "128")
            .env("ALLPAKA_BENCH_REPORT", &report_path)
            .stdout(Stdio::null())
            .status()
            .with_context(|| format!("running autotune candidate {profile}"))?;
        anyhow::ensure!(status.success(), "autotune candidate {profile} failed");
        let report: BenchmarkReport = serde_json::from_slice(&std::fs::read(&report_path)?)?;
        std::fs::remove_file(&report_path).ok();
        let decode = measurement(&report, "decode")?.summary.median;
        let prefill = measurement(&report, "prefill")?.summary.median;
        println!("  {profile}: decode={decode:.1} prefill={prefill:.1} tok/s");
        let replace = match best.as_ref() {
            None => true,
            Some((_, current)) => measurement(current, "decode")
                .map(|value| decode > value.summary.median)
                .unwrap_or(true),
        };
        if replace {
            best = Some((profile.to_string(), report));
        }
    }
    let (profile, report) = best.context("autotune produced no successful candidates")?;
    let decode = measurement(&report, "decode")?.summary.median;
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
        objective_tok_s: decode,
        measured_at_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    };
    cache.insert(key, entry);
    cache.save_atomic(&cache_path)?;
    println!(
        "autotune selected {profile}: {decode:.1} decode tok/s; cached at {}",
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
    let file = allpaka_gguf::GgufFile::open(model_path)?;
    let key = key_for(model_path, &file);
    let cache_path = cache_path
        .map(ToOwned::to_owned)
        .unwrap_or_else(default_cache_path);
    let cache = AutotuneCache::load(&cache_path)?;
    let Some(hit) = cache.get(&key) else {
        return Ok(false);
    };
    // Startup is still single-threaded here. Publish the resolved profile so
    // legacy capability reporters and remaining env-backed probes cannot
    // disagree with the installed typed policy.
    std::env::set_var(
        "ALLPAKA_PROFILE",
        &hit.parameters.command_strategy,
    );
    let profile: allpaka_backend::profile::RuntimeProfile =
        hit.parameters.command_strategy.parse()?;
    let resolved = profile.resolve_with_env();
    let _ = allpaka_backend::runtime::install(resolved.policy);
    println!(
        "autotune cache applied: profile={} objective={:.1} tok/s ({})",
        hit.parameters.command_strategy,
        hit.objective_tok_s,
        cache_path.display()
    );
    Ok(true)
}

fn key_for(model_path: &Path, file: &allpaka_gguf::GgufFile) -> AutotuneKey {
    AutotuneKey {
        device_registry_id: host_value("sysctl", &["-n", "machdep.cpu.brand_string"]),
        metal_os_version: host_value("sw_vers", &["-productVersion"]),
        model_fingerprint: model_fingerprint(model_path, file),
        model_architecture: file.architecture().to_string(),
        model_dimensions: format!("tensors={}", file.tensors().len()),
        tensor_census_hash: tensor_census(file),
        kernel_abi_version: KERNEL_ABI_VERSION,
    }
}

fn measurement<'a>(report: &'a BenchmarkReport, name: &str) -> Result<&'a crate::benchmark_report::Measurement> {
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
        AutotuneCache, AutotuneEntry, AutotuneKey, TunedParameters, KERNEL_ABI_VERSION,
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
            std::thread::current().name().unwrap_or("test")
        ));
        let mut cache = AutotuneCache::empty();
        cache.insert(key("model-a"), entry());
        cache.save_atomic(&path).unwrap();
        let loaded = AutotuneCache::load(&path).unwrap();
        assert!(loaded.get(&key("model-a")).is_some());
        assert!(loaded.get(&key("model-b")).is_none());
        std::fs::remove_file(path).unwrap();
    }
}
