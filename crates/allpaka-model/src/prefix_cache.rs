//! Byte-budgeted longest-prefix cache with ref-counted leases and a prefix-trie index.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

#[derive(Debug)]
struct Entry<V> {
    tokens: Arc<[u32]>,
    value: Arc<V>,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<u32, usize>,
    entry_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct PrefixHit<V> {
    pub matched_tokens: usize,
    pub value: Arc<V>,
}

#[derive(Debug)]
pub struct PrefixCache<V> {
    entries: Vec<Entry<V>>,
    trie: Vec<TrieNode>,
    budget_bytes: usize,
    resident_bytes: usize,
    clock: u64,
    retired: Vec<(Weak<V>, usize)>,
    hits: u64,
    misses: u64,
    reused_tokens: u64,
}

impl<V> PrefixCache<V> {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            trie: vec![TrieNode::default()],
            budget_bytes,
            resident_bytes: 0,
            clock: 0,
            retired: Vec::new(),
            hits: 0,
            misses: 0,
            reused_tokens: 0,
        }
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn reused_tokens(&self) -> u64 {
        self.reused_tokens
    }

    /// Declared payload bytes still alive, including evicted external leases.
    pub fn allocated_bytes(&self) -> usize {
        self.resident_bytes.saturating_add(
            self.retired
                .iter()
                .filter(|(value, _)| value.strong_count() > 0)
                .map(|(_, bytes)| *bytes)
                .sum::<usize>(),
        )
    }

    pub fn pinned_bytes(&self) -> usize {
        let cached = self
            .entries
            .iter()
            .filter(|entry| Arc::strong_count(&entry.value) > 1)
            .map(|entry| entry.bytes)
            .sum::<usize>();
        let retired_bytes = self
            .retired
            .iter()
            .filter(|(value, _)| value.strong_count() > 0)
            .map(|(_, bytes)| *bytes)
            .sum::<usize>();
        cached.saturating_add(retired_bytes)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn trie_insert(&mut self, tokens: &[u32], entry_index: usize) {
        let mut node_idx = 0usize;
        for &tok in tokens {
            if let Some(&next) = self.trie[node_idx].children.get(&tok) {
                node_idx = next;
            } else {
                let next = self.trie.len();
                self.trie.push(TrieNode::default());
                self.trie[node_idx].children.insert(tok, next);
                node_idx = next;
            }
        }
        self.trie[node_idx].entry_index = Some(entry_index);
    }

    fn trie_set(&mut self, tokens: &[u32], entry_index: Option<usize>) {
        let mut node_idx = 0usize;
        for &tok in tokens {
            let Some(&next) = self.trie[node_idx].children.get(&tok) else {
                return;
            };
            node_idx = next;
        }
        self.trie[node_idx].entry_index = entry_index;
    }

    fn trie_exact(&self, tokens: &[u32]) -> Option<usize> {
        let mut node_idx = 0usize;
        for &tok in tokens {
            node_idx = *self.trie[node_idx].children.get(&tok)?;
        }
        self.trie[node_idx].entry_index
    }

    fn remove_entry(&mut self, index: usize) -> Entry<V> {
        let removed = self.entries.swap_remove(index);
        self.trie_set(&removed.tokens, None);
        if index < self.entries.len() {
            let moved_tokens = Arc::clone(&self.entries[index].tokens);
            self.trie_set(&moved_tokens, Some(index));
        }
        removed
    }

    pub fn longest_prefix(&mut self, tokens: &[u32]) -> Option<PrefixHit<V>> {
        let mut node_idx = 0usize;
        let mut best_index = self.trie[0].entry_index;
        for &tok in tokens {
            let Some(&next) = self.trie[node_idx].children.get(&tok) else {
                break;
            };
            node_idx = next;
            if let Some(idx) = self.trie[node_idx].entry_index {
                best_index = Some(idx);
            }
        }
        let Some(index) = best_index else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.clock = self.clock.wrapping_add(1);
        self.hits = self.hits.saturating_add(1);
        let entry = &mut self.entries[index];
        entry.last_used = self.clock;
        let matched = entry.tokens.len();
        self.reused_tokens = self.reused_tokens.saturating_add(matched as u64);
        Some(PrefixHit {
            matched_tokens: matched,
            value: Arc::clone(&entry.value),
        })
    }

    pub fn insert(&mut self, tokens: Vec<u32>, value: V, value_bytes: usize) {
        let bytes = value_bytes.saturating_add(tokens.len().saturating_mul(size_of::<u32>()));
        if self.budget_bytes == 0 || bytes > self.budget_bytes {
            return;
        }
        let mut retired_bytes = 0usize;
        self.retired.retain(|(value, b)| {
            let alive = value.strong_count() > 0;
            if alive {
                retired_bytes = retired_bytes.saturating_add(*b);
            }
            alive
        });
        self.clock = self.clock.wrapping_add(1);
        if let Some(index) = self.trie_exact(&tokens) {
            let removed = self.remove_entry(index);
            self.resident_bytes -= removed.bytes;
            if Arc::strong_count(&removed.value) > 1 {
                self.retired
                    .push((Arc::downgrade(&removed.value), removed.bytes));
                retired_bytes = retired_bytes.saturating_add(removed.bytes);
            }
        }
        while self
            .resident_bytes
            .saturating_add(retired_bytes)
            .saturating_add(bytes)
            > self.budget_bytes
        {
            // Prefer evicting an unpinned entry first (which actually frees budget bytes).
            let unpinned_lru = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| Arc::strong_count(&entry.value) == 1)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index);
            if let Some(index) = unpinned_lru {
                let removed = self.remove_entry(index);
                self.resident_bytes -= removed.bytes;
                continue;
            }
            // Only pinned entries remain in `self.entries`. Retiring them will not reduce
            // `allocated_bytes()`, so retire the oldest pinned entry and abort insertion.
            if let Some(index) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
            {
                let removed = self.remove_entry(index);
                self.resident_bytes -= removed.bytes;
                if Arc::strong_count(&removed.value) > 1 {
                    self.retired
                        .push((Arc::downgrade(&removed.value), removed.bytes));
                }
            }
            return;
        }
        let new_index = self.entries.len();
        let tokens_arc: Arc<[u32]> = tokens.into();
        self.trie_insert(&tokens_arc, new_index);
        self.entries.push(Entry {
            tokens: tokens_arc,
            value: Arc::new(value),
            bytes,
            last_used: self.clock,
        });
        self.resident_bytes += bytes;
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

    #[test]
    fn telemetry_tracks_hits_misses_and_reused_tokens() {
        let mut cache = PrefixCache::new(80);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.reused_tokens(), 0);

        // Miss
        assert!(cache.longest_prefix(&[1, 2, 3]).is_none());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.reused_tokens(), 0);

        // Insert and Hit
        cache.insert(vec![1, 2], "v", 8);
        let hit = cache.longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(hit.matched_tokens, 2);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.reused_tokens(), 2);

        // Second Hit with longer match
        cache.insert(vec![1, 2, 3], "v2", 8);
        let hit2 = cache.longest_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(hit2.matched_tokens, 3);
        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.reused_tokens(), 5);
    }
}
