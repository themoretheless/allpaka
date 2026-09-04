//! Byte-budgeted longest-prefix cache with ref-counted leases.

use std::sync::{Arc, Weak};

#[derive(Debug)]
struct Entry<V> {
    tokens: Arc<[u32]>,
    value: Arc<V>,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug, Clone)]
pub struct PrefixHit<V> {
    pub matched_tokens: usize,
    pub value: Arc<V>,
}

#[derive(Debug)]
pub struct PrefixCache<V> {
    entries: Vec<Entry<V>>,
    budget_bytes: usize,
    resident_bytes: usize,
    clock: u64,
    retired: Vec<(Weak<V>, usize)>,
}

impl<V> PrefixCache<V> {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget_bytes,
            resident_bytes: 0,
            clock: 0,
            retired: Vec::new(),
        }
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Declared payload bytes still alive, including evicted external leases.
    pub fn allocated_bytes(&self) -> usize {
        self.resident_bytes.saturating_add(self.retired.iter()
            .filter(|(value, _)| value.strong_count() > 0)
            .map(|(_, bytes)| *bytes).sum::<usize>())
    }

    pub fn pinned_bytes(&self) -> usize {
        let cached = self.entries.iter().filter(|entry| Arc::strong_count(&entry.value) > 1)
            .map(|entry| entry.bytes).sum::<usize>();
        cached.saturating_add(self.allocated_bytes().saturating_sub(self.resident_bytes))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn longest_prefix(&mut self, tokens: &[u32]) -> Option<PrefixHit<V>> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| tokens.starts_with(&entry.tokens))
            .max_by_key(|(_, entry)| entry.tokens.len())
            .map(|(index, _)| index)?;
        self.clock = self.clock.wrapping_add(1);
        let entry = &mut self.entries[index];
        entry.last_used = self.clock;
        Some(PrefixHit {
            matched_tokens: entry.tokens.len(),
            value: Arc::clone(&entry.value),
        })
    }

    pub fn insert(&mut self, tokens: Vec<u32>, value: V, value_bytes: usize) {
        let bytes = value_bytes.saturating_add(tokens.len().saturating_mul(size_of::<u32>()));
        if self.budget_bytes == 0 || bytes > self.budget_bytes {
            return;
        }
        self.retired.retain(|(value, _)| value.strong_count() > 0);
        self.clock = self.clock.wrapping_add(1);
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.tokens.as_ref() == tokens.as_slice())
        {
            self.remove_entry(index);
        }
        while self.allocated_bytes().saturating_add(bytes) > self.budget_bytes {
            let Some(index) = self.entries.iter().enumerate()
                .min_by_key(|(_, entry)| entry.last_used).map(|(index, _)| index)
            else { return };
            self.remove_entry(index);
        }
        self.entries.push(Entry {
            tokens: tokens.into(), value: Arc::new(value), bytes, last_used: self.clock,
        });
        self.resident_bytes += bytes;
    }

    fn remove_entry(&mut self, index: usize) {
        let removed = self.entries.swap_remove(index);
        self.resident_bytes -= removed.bytes;
        if Arc::strong_count(&removed.value) > 1 {
            // Conservatively retain the key charge until the value lease dies.
            self.retired.push((Arc::downgrade(&removed.value), removed.bytes));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixCache;

    #[test]
    fn evicted_leases_still_consume_budget_until_the_last_drop() {
        let mut cache = PrefixCache::new(80);
        cache.insert(vec![1], vec![0u8; 60], 60);
        let lease = cache.longest_prefix(&[1]).unwrap();
        cache.insert(vec![2], vec![0u8; 60], 60);
        assert!(cache.longest_prefix(&[2]).is_none());
        assert_eq!(cache.allocated_bytes(), 64);
        assert_eq!(cache.pinned_bytes(), 64);
        drop(lease);
        assert_eq!(cache.allocated_bytes(), 0);
        cache.insert(vec![2], vec![0u8; 60], 60);
        assert!(cache.longest_prefix(&[2]).is_some());
    }

    #[test]
    fn longest_prefix_wins_and_survives_cache_eviction_as_a_lease() {
        let mut cache = PrefixCache::new(80);
        cache.insert(vec![1, 2], "short", 8);
        cache.insert(vec![1, 2, 3], "long", 8);
        let hit = cache.longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(hit.matched_tokens, 3);
        assert_eq!(*hit.value, "long");

        cache.insert(vec![9; 12], "replacement", 16);
        assert_eq!(*hit.value, "long");
        assert!(cache.resident_bytes() <= 80);
    }

    #[test]
    fn entries_larger_than_the_budget_are_not_cached() {
        let mut cache = PrefixCache::new(16);
        cache.insert(vec![1, 2], 7, 32);
        assert!(cache.is_empty());
    }
}
