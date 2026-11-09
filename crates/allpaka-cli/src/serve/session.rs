//! Per-model KV/SSM session state and namespaced prefix snapshot caching.

use super::memory;
use allpaka_model::prefix_cache::PrefixCache;
use allpaka_model::{Model, Session, Tokenizer};
use anyhow::{bail, Result};
use std::sync::Arc;

/// The KV cache carried across requests.
///
/// A chat regenerates the same prompt prefix on every turn: the whole prior
/// conversation, byte for byte, plus one new user message. Keeping the session
/// and the token list lets each request pay only for what is actually new -
/// without this, turn N re-runs the forward pass over every earlier turn and
/// the chat slows down linearly with its own length.
pub(crate) struct ChatState {
    pub(crate) session: Session,
    pub(crate) tokens: Vec<u32>,
    pub(crate) memory: memory::Lease,
}

impl ChatState {
    pub(crate) fn new(
        model: &Model<'static>,
        budget: &memory::Budget,
        capacity: usize,
    ) -> Result<Self> {
        let memory = budget.reserve(memory::session_bytes(model, capacity)?)?;
        Ok(Self {
            session: model.new_session(capacity),
            tokens: Vec::new(),
            memory,
        })
    }

    pub(crate) fn reset(&mut self, model: &Model<'static>, capacity: usize) -> Result<()> {
        // Reserve the new storage while the old storage is still alive.
        // A too-large KV allocation asserts inside the cache; keep the
        // running session and report it instead of killing the process.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Self::new(model, &self.memory.budget(), capacity)
        }));
        match built {
            Ok(session) => {
                *self = session?;
                Ok(())
            }
            Err(_) => bail!("out of memory for a {capacity}-token session"),
        }
    }

    /// Reconcile `session.pos()` and `tokens.len()` after an interrupted or failed generation.
    pub(crate) fn sanitize_after_failure(&mut self, model: &Model<'static>) -> Result<()> {
        if self.session.pos() != self.tokens.len() {
            if self.session.ssm.is_some() {
                let cap = self.session.capacity();
                self.reset(model, cap)?;
                self.tokens.clear();
            } else {
                let valid = self.tokens.len().min(self.session.pos());
                self.session.truncate(valid);
                self.tokens.truncate(valid);
            }
        }
        Ok(())
    }
}

/// Byte-budgeted prompt-boundary snapshot shared by requests for one model.
pub(crate) struct PrefixSnap {
    /// Token count of the prompt this state was captured after.
    pub(crate) tokens: usize,
    /// `SsmCache::snapshot()` at the prompt's end for hybrid models.
    pub(crate) ssm_state: Option<Vec<f32>>,
    /// Last-position logits let an exact repeated prompt skip all prefill.
    pub(crate) logits: Arc<[f32]>,
}

pub(crate) struct ModelService {
    pub(crate) model: Model<'static>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) force_think: bool,
    pub(crate) chat: ChatState,
    pub(crate) sessions: std::collections::HashMap<String, (ChatState, std::time::Instant)>,
}

pub(crate) fn model_prefix_namespace(model: &str) -> u32 {
    model.bytes().fold(2_166_136_261u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(16_777_619)
    })
}

pub(crate) fn prefix_key(namespace: u32, tokens: &[u32]) -> Vec<u32> {
    let mut key = Vec::with_capacity(tokens.len() + 1);
    key.push(namespace);
    key.extend_from_slice(tokens);
    key
}

pub(crate) fn exact_prefix_snapshot(
    cache: &mut PrefixCache<PrefixSnap>,
    namespace: u32,
    tokens: &[u32],
) -> Option<Arc<PrefixSnap>> {
    let key = prefix_key(namespace, tokens);
    let hit = cache.longest_prefix(&key)?;
    (hit.matched_tokens == key.len() && hit.value.tokens == tokens.len()).then_some(hit.value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_hits_are_model_namespaced() {
        let mut cache = PrefixCache::new(1024);
        let first = model_prefix_namespace("first");
        let second = model_prefix_namespace("second");
        cache.insert(
            prefix_key(first, &[1, 2, 3]),
            PrefixSnap {
                tokens: 3,
                ssm_state: None,
                logits: Arc::from([4.0f32]),
            },
            4,
        );

        assert_eq!(
            &*exact_prefix_snapshot(&mut cache, first, &[1, 2, 3])
                .unwrap()
                .logits,
            &[4.0]
        );
        assert!(exact_prefix_snapshot(&mut cache, first, &[1, 2]).is_none());
        assert!(exact_prefix_snapshot(&mut cache, second, &[1, 2, 3]).is_none());
    }
}
