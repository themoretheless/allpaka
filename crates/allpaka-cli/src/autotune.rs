//! Versioned, model- and device-specific autotune cache.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const AUTOTUNE_SCHEMA_VERSION: u32 = 1;
pub const KERNEL_ABI_VERSION: u32 = 1;

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
