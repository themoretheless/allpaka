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
    /// Shared so a worker generation (see `interleave`) can run on it while
    /// the service is locked by another request.
    pub(crate) model: Arc<Model<'static>>,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) force_think: bool,
    pub(crate) chat: ChatState,
    /// Named sessions not in use, outside the service lock so a generation
    /// running beside the current request (`interleave`) can check one out.
    pub(crate) sessions: NamedSessions,
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

/// Parked named sessions (`session_id`) and when each was last used. A
/// request checks its session out (removes it) for its whole run and puts it
/// back after, so a session is never in two generations at once.
pub(crate) type NamedSessions =
    Arc<std::sync::Mutex<std::collections::HashMap<String, (ChatState, std::time::Instant)>>>;

/// Most named sessions parked at once.
pub(crate) const NAMED_SESSIONS_MAX: usize = 4;

/// Parked sessions unused this long are dropped.
pub(crate) const NAMED_SESSION_IDLE_SECS: u64 = 600;

/// Take `id` out of the parked sessions, or a fresh session for a new id;
/// `Err` with the status and message to answer when neither is possible. The
/// flag says the session existed.
pub(crate) fn check_out_named(
    sessions: &NamedSessions,
    id: &str,
    model: &Model<'static>,
    budget: &memory::Budget,
) -> std::result::Result<(ChatState, bool), (u16, String)> {
    let mut parked = sessions.lock().unwrap();
    if named_in_use(id) {
        return Err((409, format!("session {id} is in use by another request")));
    }
    let now = std::time::Instant::now();
    parked.retain(|_, (_, used)| now.duration_since(*used).as_secs() < NAMED_SESSION_IDLE_SECS);
    let existing = parked.remove(id);
    if existing.is_none() && parked.len() >= NAMED_SESSIONS_MAX {
        return Err((429, "resident session capacity exhausted".into()));
    }
    IN_USE.lock().unwrap().push(id.to_owned());
    drop(parked);
    if let Some((chat, _)) = existing {
        return Ok((chat, true));
    }
    ChatState::new(model, budget, 1)
        .map(|chat| (chat, false))
        .map_err(|error| {
            release_named(id);
            (429, error.to_string())
        })
}

/// Named sessions checked out right now: a session runs in one generation at
/// a time; a second request for it is refused while it runs.
static IN_USE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn named_in_use(id: &str) -> bool {
    IN_USE.lock().unwrap().iter().any(|used| used == id)
}

fn release_named(id: &str) {
    IN_USE.lock().unwrap().retain(|used| used != id);
}

/// Put a checked-out session back (`succeeded` or it existed before).
pub(crate) fn check_in_named(sessions: &NamedSessions, id: String, chat: ChatState, keep: bool) {
    let mut parked = sessions.lock().unwrap();
    release_named(&id);
    if keep {
        parked.insert(id, (chat, std::time::Instant::now()));
    }
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
