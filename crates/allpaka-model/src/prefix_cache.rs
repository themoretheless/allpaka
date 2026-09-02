//! Byte-budgeted longest-prefix cache with ref-counted leases.

use std::sync::Arc;

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
}

impl<V> PrefixCache<V> {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget_bytes,
            resident_bytes: 0,
            clock: 0,
        }
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
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
        let bytes = value_bytes.saturating_add(tokens.len() * size_of::<u32>());
        if bytes > self.budget_bytes {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.tokens.as_ref() == tokens.as_slice())
        {
            self.resident_bytes -= self.entries[index].bytes;
            self.entries[index] = Entry {
                tokens: tokens.into(),
                value: Arc::new(value),
                bytes,
                last_used: self.clock,
            };
            self.resident_bytes += bytes;
        } else {
            self.entries.push(Entry {
                tokens: tokens.into(),
                value: Arc::new(value),
                bytes,
                last_used: self.clock,
            });
            self.resident_bytes += bytes;
        }
        self.evict_to_budget();
    }

    fn evict_to_budget(&mut self) {
        while self.resident_bytes > self.budget_bytes {
            let Some((index, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
            else {
                break;
            };
            let removed = self.entries.swap_remove(index);
            self.resident_bytes -= removed.bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixCache;

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
