//! Mutable decode session state (`KV`, `SSM`, and precomputed RoPE tables).

use super::Model;
use crate::kv::KvCache;

/// Mutable decode state: the cache plus the implicit position.
pub struct Session {
    pub(crate) kv: KvCache,
    pub(crate) rope_cache: Vec<[f32; 2]>,
    pub(crate) rope_cache_pairs: usize,
    /// Present only for models with gated-delta-net layers (qwen35moe).
    pub ssm: Option<crate::kv::SsmCache>,
}

impl Session {
    /// Tokens consumed so far; the next token lands at this position.
    pub fn pos(&self) -> usize {
        self.kv.len()
    }

    /// Most tokens this session can ever hold.
    pub fn capacity(&self) -> usize {
        self.kv.capacity()
    }

    /// Roll the session back to its first `keep` tokens.
    pub fn truncate(&mut self, keep: usize) {
        self.kv.truncate(keep);
        // RoPE sin/cos depends only on position and rope_inv_freq, not on token content,
        // so keeping already-computed positions in `rope_cache` avoids recomputation.
    }

    /// Ensure the session's RoPE sin/cos table is populated up to `needed_positions`.
    #[inline]
    pub(crate) fn ensure_rope_cache(&mut self, rope_inv_freq: &[f32], needed_positions: usize) {
        debug_assert_eq!(rope_inv_freq.len(), self.rope_cache_pairs);
        if self.rope_cache_pairs == 0 {
            return;
        }
        let have = self.rope_cache.len() / self.rope_cache_pairs;
        if have < needed_positions {
            self.rope_cache
                .reserve((needed_positions - have) * self.rope_cache_pairs);
            for pos in have..needed_positions {
                for &freq in rope_inv_freq {
                    self.rope_cache.push((pos as f32 * freq).sin_cos().into());
                }
            }
        }
    }
}

impl<'a> Model<'a> {
    /// A session whose KV cache has one extra layer for the MTP block
    /// (index n_layers); plain sessions do not pay for it.
    pub fn new_session_mtp(&self, context_tokens: usize) -> Session {
        let context_tokens = context_tokens.max(allpaka_backend::gpu::minimum_kv_capacity());
        let mut kv_active: Vec<bool> = self.layers.iter().map(|l| l.attn.is_some()).collect();
        kv_active.extend(std::iter::repeat_n(true, self.config.nextn as usize));
        let ssm_active: Vec<bool> = self.layers.iter().map(|l| l.gdn.is_some()).collect();
        Session {
            kv: KvCache::with_active_layers(&kv_active, self.config.kv_dim(), context_tokens),
            rope_cache: Vec::new(),
            rope_cache_pairs: self.config.rope_dim as usize / 2,
            ssm: self.config.ssm.as_ref().map(|s| {
                crate::kv::SsmCache::with_active_layers(
                    &ssm_active,
                    s.d_conv as usize,
                    (s.d_inner + 2 * (s.n_group * s.d_state)) as usize,
                    s.dt_rank as usize,
                    s.d_state as usize,
                )
            }),
        }
    }

    pub fn new_session(&self, context_tokens: usize) -> Session {
        let context_tokens = context_tokens.max(allpaka_backend::gpu::minimum_kv_capacity());
        let kv_active: Vec<bool> = self.layers.iter().map(|l| l.attn.is_some()).collect();
        let ssm_active: Vec<bool> = self.layers.iter().map(|l| l.gdn.is_some()).collect();
        Session {
            kv: KvCache::with_active_layers(&kv_active, self.config.kv_dim(), context_tokens),
            rope_cache: Vec::new(),
            rope_cache_pairs: self.config.rope_dim as usize / 2,
            ssm: self.config.ssm.as_ref().map(|s| {
                crate::kv::SsmCache::with_active_layers(
                    &ssm_active,
                    s.d_conv as usize,
                    (s.d_inner + 2 * (s.n_group * s.d_state)) as usize,
                    s.dt_rank as usize,
                    s.d_state as usize,
                )
            }),
        }
    }
}
