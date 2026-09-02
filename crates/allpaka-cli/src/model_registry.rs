//! Lease-safe, byte-budgeted registry for multi-model serving.

use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug)]
struct ModelEntry<T> {
    model: Arc<T>,
    bytes: u64,
    last_used: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryError {
    ModelExceedsBudget,
    CapacityPinnedByActiveRequests,
}

#[derive(Debug)]
pub struct ModelRegistry<T> {
    entries: BTreeMap<String, ModelEntry<T>>,
    budget_bytes: u64,
    resident_bytes: u64,
    clock: u64,
}

impl<T> ModelRegistry<T> {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            entries: BTreeMap::new(),
            budget_bytes,
            resident_bytes: 0,
            clock: 0,
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn model_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn get(&mut self, name: &str) -> Option<Arc<T>> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(name)?;
        entry.last_used = self.clock;
        Some(Arc::clone(&entry.model))
    }

    pub fn install(&mut self, name: String, model: T, bytes: u64) -> Result<(), RegistryError> {
        if bytes > self.budget_bytes {
            return Err(RegistryError::ModelExceedsBudget);
        }
        let replaced_bytes = self.entries.get(&name).map_or(0, |entry| entry.bytes);
        let projected = self
            .resident_bytes
            .saturating_sub(replaced_bytes)
            .saturating_add(bytes);
        self.evict_until(projected, Some(&name))?;
        if let Some(replaced) = self.entries.remove(&name) {
            self.resident_bytes -= replaced.bytes;
        }
        self.clock = self.clock.wrapping_add(1);
        self.entries.insert(
            name,
            ModelEntry {
                model: Arc::new(model),
                bytes,
                last_used: self.clock,
            },
        );
        self.resident_bytes += bytes;
        Ok(())
    }

    pub fn unload(&mut self, name: &str) -> bool {
        let can_unload = self
            .entries
            .get(name)
            .is_some_and(|entry| Arc::strong_count(&entry.model) == 1);
        if !can_unload {
            return false;
        }
        let entry = self.entries.remove(name).expect("entry existed");
        self.resident_bytes -= entry.bytes;
        true
    }

    fn evict_until(
        &mut self,
        mut projected_bytes: u64,
        protected_name: Option<&str>,
    ) -> Result<(), RegistryError> {
        while projected_bytes > self.budget_bytes {
            let candidate = self
                .entries
                .iter()
                .filter(|(name, entry)| {
                    Some(name.as_str()) != protected_name && Arc::strong_count(&entry.model) == 1
                })
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(name, _)| name.clone());
            let Some(candidate) = candidate else {
                return Err(RegistryError::CapacityPinnedByActiveRequests);
            };
            let removed = self.entries.remove(&candidate).expect("candidate existed");
            self.resident_bytes -= removed.bytes;
            projected_bytes = projected_bytes.saturating_sub(removed.bytes);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelRegistry, RegistryError};

    #[test]
    fn active_leases_prevent_eviction_and_release_allows_it() {
        let mut registry = ModelRegistry::new(100);
        registry.install("a".into(), "model-a", 60).unwrap();
        let lease = registry.get("a").unwrap();
        assert_eq!(
            registry.install("b".into(), "model-b", 60),
            Err(RegistryError::CapacityPinnedByActiveRequests)
        );
        drop(lease);
        registry.install("b".into(), "model-b", 60).unwrap();
        assert!(registry.get("a").is_none());
        assert_eq!(*registry.get("b").unwrap(), "model-b");
    }

    #[test]
    fn hot_replacement_accounts_only_the_new_model() {
        let mut registry = ModelRegistry::new(100);
        registry.install("chat".into(), 1, 80).unwrap();
        registry.install("chat".into(), 2, 90).unwrap();
        assert_eq!(registry.resident_bytes(), 90);
        assert_eq!(*registry.get("chat").unwrap(), 2);
    }
}
