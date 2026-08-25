//! Greedy speculative decoding: a small draft proposes, the target verifies.
//!
//! The economics on this engine, measured before this module existed: the
//! target's decode is compute-bound on dequantising expert weights, and that
//! cost barely grows with batch size - verifying K tokens in one batched
//! forward costs about as much as decoding one. So every accepted draft
//! token is nearly free target throughput, and the draft (a 0.6B next to a
//! 235B) drafts an order of magnitude faster than the target decodes.
//!
//! Both models decode greedily, and verification accepts a draft token only
//! when the target's own argmax agrees. The emitted stream is therefore
//! IDENTICAL to what the target alone would emit - speculation changes the
//! speed, never the tokens. `bench --draft` asserts exactly that.
//!
//! Bookkeeping invariant, chosen to keep rollback trivial: between rounds,
//! both caches hold exactly the emitted tokens, and `next` - the token the
//! target has already chosen but not yet consumed - sits in neither. Each
//! round feeds it to both (the draft first, the target as the head of the
//! verification batch), so a round always emits at least `next` itself.

use crate::model::{Model, Session};
use anyhow::{bail, Result};

/// One draft-verify round. `next` is the pending token (in neither cache
/// yet). Returns the tokens emitted this round - `next` plus every accepted
/// draft - and the pending token for the following round.
pub struct Round {
    pub emitted: Vec<u32>,
    pub next: u32,
    pub drafted: usize,
    pub accepted: usize,
}

pub struct Speculator<'a> {
    pub target: &'a Model<'a>,
    pub target_session: &'a mut Session,
    pub draft: &'a Model<'a>,
    pub draft_session: &'a mut Session,
    /// Draft tokens per round.
    pub k: usize,
}

impl Speculator<'_> {
    /// Sanity that makes the whole scheme valid: same token ids must mean
    /// the same strings in both models.
    pub fn compatible(target: &Model, draft: &Model) -> bool {
        target.config.vocab == draft.config.vocab
    }

    pub fn round(&mut self, next: u32) -> Result<Round> {
        let k = self.k.max(1);
        // Room for `next` plus k drafts in both caches, or fall back to a
        // plain (non-speculative) step.
        let need = self.target_session.pos() + k + 1;
        if need > self.target_session.capacity() || need > self.draft_session.capacity() {
            bail!("kv capacity exhausted for a speculative round");
        }

        // The draft consumes `next`, then proposes k tokens greedily.
        let mut drafts = Vec::with_capacity(k);
        let mut feed = next;
        for _ in 0..k {
            let logits = self.draft.forward(feed, self.draft_session)?;
            let t = argmax(&logits);
            drafts.push(t);
            feed = t;
        }

        // The target consumes [next, drafts..] as one batch. Row i answers
        // "what follows the batch's first i+1 tokens": row 0 arbitrates
        // drafts[0], and the last row supplies the continuation when every
        // draft is accepted.
        let mut batch = Vec::with_capacity(k + 1);
        batch.push(next);
        batch.extend_from_slice(&drafts);
        let all = self.target.forward_batch_full(&batch, self.target_session)?;
        let vocab = self.target.config.vocab as usize;

        let mut emitted = vec![next];
        let mut accepted = 0;
        let mut new_next = 0u32;
        for (i, &d) in drafts.iter().enumerate() {
            let want = argmax(&all[i * vocab..(i + 1) * vocab]);
            if want == d {
                emitted.push(d);
                accepted += 1;
            } else {
                new_next = want;
                break;
            }
        }
        if accepted == drafts.len() {
            new_next = argmax(&all[k * vocab..(k + 1) * vocab]);
        }

        // Roll both caches back to the committed prefix. The target keeps
        // its committed tokens plus the new `next`'s predecessors: it
        // consumed k+1 positions, of which 1 + accepted are valid. The draft
        // consumed k positions (next, then all but the last draft... no -
        // next plus k-1 drafts were INPUTS; the k-th draft never entered its
        // cache), of which 1 + accepted are the same valid prefix.
        let committed = self.target_session.pos() - (k + 1) + 1 + accepted;
        self.target_session.truncate(committed);
        let draft_keep = committed.min(self.draft_session.pos());
        self.draft_session.truncate(draft_keep);
        // If the draft's cache is now short of the committed prefix (every
        // draft accepted: it never consumed its own last proposal), feed the
        // missing committed tokens so the next round starts aligned.
        if draft_keep < committed {
            let missing = &emitted[emitted.len() - (committed - draft_keep)..];
            self.draft.forward_batch(missing, self.draft_session)?;
        }

        Ok(Round { emitted, next: new_next, drafted: k, accepted })
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
