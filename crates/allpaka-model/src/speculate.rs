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
            && target.config.architecture == draft.config.architecture
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
            let t = self.draft.forward_greedy(feed, self.draft_session)?;
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

        let mut emitted = Vec::with_capacity(k + 1);
        emitted.push(next);
        let mut accepted = 0;
        let mut new_next = 0u32;

        if let Some((picks, _h_all)) = self.target.verify_tokens(&batch, self.target_session)? {
            for (i, &d) in drafts.iter().enumerate() {
                let want = picks[i];
                if want == d {
                    emitted.push(d);
                    accepted += 1;
                } else {
                    new_next = want;
                    break;
                }
            }
            if accepted == drafts.len() {
                new_next = picks[k];
            }
        } else {
            let all = self
                .target
                .forward_batch_full(&batch, self.target_session)?;
            let vocab = self.target.config.vocab as usize;
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
            let _ = self
                .draft
                .forward_batch_hidden(missing, self.draft_session)?;
        }

        Ok(Round {
            emitted,
            next: new_next,
            drafted: k,
            accepted,
        })
    }
}

#[inline]
fn argmax(logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let mut best_idx = 0usize;
    let mut best_value = logits[0];
    for (idx, &value) in logits.iter().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_idx = idx;
        }
    }
    best_idx as u32
}

#[inline]
fn mtp_debug_enabled() -> bool {
    static DBG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DBG.get_or_init(|| std::env::var_os("ALLPAKA_MTP_DEBUG").is_some())
}

#[inline]
fn verify_parity_enabled() -> bool {
    static PARITY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PARITY.get_or_init(|| std::env::var_os("ALLPAKA_VERIFY_PARITY").is_some())
}

/// MTP (nextn) speculation: the draft is the target model's own MTP block,
/// not a second model. Both draft and verify share one session (the draft
/// writes only the MTP layer's KV slot at explicit positions, the trunk's
/// position counter advances only through the verify batch).
///
/// Rollback for a hybrid (qwen35moe) target, the same class as the serve
/// prompt cache: the KV truncates, but the gated-delta-net state is
/// irreversible. Each round snapshots it once (66 MB memcpy); a partially
/// accepted round restores the snapshot and replays the accepted tokens
/// through the trunk - llama.cpp's approach for recurrent models, and the
/// reason a fully-accepted round is the cheap one.
pub struct MtpSpeculator<'a> {
    pub model: &'a Model<'a>,
    pub session: &'a mut Session,
    /// Draft tokens per round.
    pub k: usize,
    /// The draft h chain seed: the h_out of the step that consumed the
    /// committed prefix's last token (the trunk's output-normed hidden to
    /// start the chain).
    pub h: Vec<f32>,
}

impl MtpSpeculator<'_> {
    pub fn round(&mut self, next: u32) -> Result<Round> {
        let k = self.k.max(1);
        let model = self.model;
        let s = &mut *self.session;
        let dbg = mtp_debug_enabled();
        let t0 = dbg.then(std::time::Instant::now);
        let start = s.pos();
        if start + k + 1 > s.capacity() {
            bail!("kv capacity exhausted for a speculative round");
        }
        // Rollback without a replay: the verify writes every row's SSM
        // state into the rollback slots (GPU batch kernels and the CPU
        // fallback alike); a partial round restores the accepted row's slot
        // with one contiguous copy. The KV truncates as usual.
        if let Some(ssm) = s.ssm.as_mut() {
            ssm.arm_slots(k + 1);
        }
        if let Some(t0) = t0 {
            eprintln!("mtp round @{start}: arm {:?}", t0.elapsed());
        }

        // The draft consumes `next`, then proposes k tokens greedily; the h
        // chains through the round from the seed (steps after the first use
        // the draft's own h_out, as llama's draft graph does).
        let mut drafts = Vec::with_capacity(k);
        let mut feed = next;
        for i in 0..k {
            let ts = dbg.then(std::time::Instant::now);
            let (logits, h) = model.mtp_step(&self.h, feed, start + i, s)?;
            if let Some(ts) = ts {
                eprintln!("  draft step {i}: {:?}", ts.elapsed());
            }
            let t = argmax(&logits);
            drafts.push(t);
            self.h = h;
            feed = t;
        }
        if let Some(t0) = t0 {
            eprintln!("  draft total: {:?}\n  drafts: {drafts:?}", t0.elapsed());
        }

        // The target consumes [next, drafts..] as one batch: row i
        // arbitrates drafts[i], the last row supplies the continuation when
        // every draft is accepted. The one-buffer GPU verify (matvec
        // numerics, bit-compatible with plain decode) is preferred; the
        // batch path is the fallback. The output-normed hidden rows seed
        // the next round's draft from the trunk's t_h_nextn at the
        // committed row.
        let mut batch = Vec::with_capacity(k + 1);
        batch.push(next);
        batch.extend_from_slice(&drafts);
        let tv = dbg.then(std::time::Instant::now);
        let hidden = model.config.hidden as usize;
        let parity = verify_parity_enabled();
        let gpu0 = dbg.then(allpaka_backend::gpu::stats);
        let (row_argmax, hidden_rows) = if parity {
            let (all, h) = model.forward_batch_full_hn(&batch, s)?;
            let vocab = model.config.vocab as usize;
            let am: Vec<u32> = (0..batch.len())
                .map(|i| argmax(&all[i * vocab..(i + 1) * vocab]))
                .collect();
            // GPU verify on a rewound session would re-run the whole batch;
            // instead just report the batch path here and let the GPU path
            // run on the next rounds (env is a bisect probe).
            eprintln!("  parity batch-path seeds:");
            for r in 0..batch.len() {
                let row = &h[r * hidden..(r + 1) * hidden];
                let sum: f64 = row.iter().map(|&x| x as f64).sum();
                eprintln!("    row {r}: argmax {} sum={sum:.6}", am[r]);
            }
            (am, h)
        } else {
            match model.verify_tokens(&batch, s)? {
                Some(pair) => pair,
                None => {
                    let (all, h) = model.forward_batch_full_hn(&batch, s)?;
                    let vocab = model.config.vocab as usize;
                    let am = (0..batch.len())
                        .map(|i| argmax(&all[i * vocab..(i + 1) * vocab]))
                        .collect();
                    (am, h)
                }
            }
        };
        if let (Some(tv), Some(gpu0)) = (tv, gpu0) {
            let gpu1 = allpaka_backend::gpu::stats();
            eprintln!(
                "  verify {} tok: {:?} ({} dispatches, wait {:.1} ms)",
                batch.len(),
                tv.elapsed(),
                gpu1.1 - gpu0.1,
                (gpu1.3 - gpu0.3) as f64 / 1e6,
            );
        }

        let mut emitted = Vec::with_capacity(k + 1);
        emitted.push(next);
        let mut accepted = 0;
        let mut new_next = 0u32;
        for (i, &d) in drafts.iter().enumerate() {
            let want = row_argmax[i];
            if dbg {
                eprintln!("  row {i}: draft {d} vs target {want}");
            }
            if want == d {
                emitted.push(d);
                accepted += 1;
            } else {
                new_next = want;
                break;
            }
        }
        if accepted == drafts.len() {
            new_next = row_argmax[k];
        }
        if dbg {
            let seed = &hidden_rows[accepted * hidden..(accepted + 1) * hidden];
            let sum: f64 = seed.iter().map(|&x| x as f64).sum();
            eprintln!(
                "  accepted {accepted}, seed sum={sum:.6} head=[{:.6} {:.6} {:.6}]",
                seed[0], seed[1], seed[2]
            );
        }

        if accepted == k {
            // The last committed token never passed through the MTP block;
            // append only its KV slot at `start + k` (skipping wq, attention,
            // wo, MoE FFN, and output_head since its proposal is unused).
            model.mtp_prefill_kv_only(
                &hidden_rows[(k - 1) * hidden..k * hidden],
                drafts[k - 1],
                start + k,
                s,
            )?;
            if let Some(ssm) = s.ssm.as_mut() {
                ssm.disarm_slots();
            }
            self.h.resize(hidden, 0.0);
            self.h
                .copy_from_slice(&hidden_rows[k * hidden..(k + 1) * hidden]);
        } else {
            // Roll back to the accepted row: KV truncates to the committed
            // prefix, the SSM state restores from that row's slot - one
            // contiguous copy, no replay.
            let tr = dbg.then(std::time::Instant::now);
            s.truncate(start + 1 + accepted);
            if let Some(ssm) = s.ssm.as_mut() {
                ssm.restore_slot(accepted);
            }
            if let Some(tr) = tr {
                eprintln!("  restore slot {accepted}: {:?}", tr.elapsed());
            }
            self.h.resize(hidden, 0.0);
            self.h
                .copy_from_slice(&hidden_rows[accepted * hidden..(accepted + 1) * hidden]);
        }
        if let Some(t0) = t0 {
            eprintln!("  round total: {:?}\n", t0.elapsed());
        }
        debug_assert_eq!(s.pos(), start + 1 + accepted);
        Ok(Round {
            emitted,
            next: new_next,
            drafted: k,
            accepted,
        })
    }
}

/// Find up to `max_k` draft continuation tokens by searching backwards through
/// `prediction` (if provided) and `history` for the longest matching N-gram
/// suffix ending with `next` (from `ngram_max` down to `ngram_min`).
///
/// The suffix matched against `source` is `[..history_tail, next]` of length `n`:
/// - For `n == 1`, the suffix is `[next]`.
/// - For `n > 1`, the suffix is `&history[history.len() - (n - 1)..]` followed by `next`.
pub fn find_prompt_drafts(
    history: &[u32],
    prediction: &[u32],
    next: u32,
    max_k: usize,
    ngram_min: usize,
    ngram_max: usize,
) -> Vec<u32> {
    if max_k == 0 {
        return Vec::new();
    }
    let min_n = ngram_min.max(1);
    let max_n = ngram_max.max(min_n).min(history.len() + 1);

    // Single-pass reverse scan over `haystack`: for each position `pos` where
    // `haystack[pos] == next`, measure how many preceding tokens match the tail
    // of `history` (up to `max_n - 1`). If a match reaches `max_n`, return
    // immediately; otherwise record the newest occurrence of each match length
    // `n in min_n..=max_n`.
    let scan_slice = |haystack: &[u32]| -> (usize, Vec<u32>) {
        if haystack.len() <= min_n {
            return (0, Vec::new());
        }
        let mut best_n = 0usize;
        let mut best_cont_start = 0usize;
        // `pos` is the index of `next` in `haystack`; at least 1 continuation token must follow (`pos + 1 < haystack.len()`).
        for pos in (min_n - 1..haystack.len() - 1).rev() {
            if haystack[pos] != next {
                continue;
            }
            let max_prefix = pos.min(max_n - 1);
            if max_prefix < best_n {
                continue;
            }
            let mut matched = 1usize;
            while matched <= max_prefix
                && haystack[pos - matched] == history[history.len() - matched]
            {
                matched += 1;
            }
            if matched >= min_n && matched > best_n {
                best_n = matched;
                best_cont_start = pos + 1;
                if best_n == max_n {
                    break;
                }
            }
        }
        if best_n >= min_n {
            let cont_end = (best_cont_start + max_k).min(haystack.len());
            (best_n, haystack[best_cont_start..cont_end].to_vec())
        } else {
            (0, Vec::new())
        }
    };

    let (pred_n, pred_drafts) = if !prediction.is_empty() {
        scan_slice(prediction)
    } else {
        (0, Vec::new())
    };
    if pred_n == max_n {
        return pred_drafts;
    }
    let (hist_n, hist_drafts) = scan_slice(history);
    if pred_n >= hist_n && pred_n >= min_n {
        pred_drafts
    } else if hist_n >= min_n {
        hist_drafts
    } else {
        Vec::new()
    }
}

/// Prompt Lookup Decoding (PLD) + Predicted Outputs speculator.
///
/// Drafts up to `k` tokens directly from N-gram matches in the conversation
/// history or an explicit `prediction` token sequence (OpenAI Predicted Outputs)
/// with zero draft-model overhead, then verifies `[next, drafts..]` in a single
/// batched target pass (`verify_tokens` / `forward_batch_full`).
pub struct PromptLookupSpeculator<'a> {
    pub model: &'a Model<'a>,
    pub session: &'a mut Session,
    pub k: usize,
    pub ngram_min: usize,
    pub ngram_max: usize,
}

impl PromptLookupSpeculator<'_> {
    /// Verify `drafts` (proposed after `next`) in a single target pass.
    ///
    /// When `drafts` is empty, falls back to a single greedy step.
    pub fn round_with_drafts(&mut self, next: u32, drafts: &[u32]) -> Result<Round> {
        let model = self.model;
        let s = &mut *self.session;
        let start = s.pos();
        let k = drafts.len();
        if k == 0 {
            if start + 1 > s.capacity() {
                bail!("kv capacity exhausted for a decode step");
            }
            let new_next = model.forward_greedy(next, s)?;
            return Ok(Round {
                emitted: vec![next],
                next: new_next,
                drafted: 0,
                accepted: 0,
            });
        }
        if start + k + 1 > s.capacity() {
            bail!("kv capacity exhausted for a speculative round");
        }
        if let Some(ssm) = s.ssm.as_mut() {
            ssm.arm_slots(k + 1);
        }

        let mut batch = Vec::with_capacity(k + 1);
        batch.push(next);
        batch.extend_from_slice(drafts);

        let row_argmax = match model.verify_tokens(&batch, s)? {
            Some((am, _)) => am,
            None => {
                let all = model.forward_batch_full(&batch, s)?;
                let vocab = model.config.vocab as usize;
                (0..batch.len())
                    .map(|i| argmax(&all[i * vocab..(i + 1) * vocab]))
                    .collect()
            }
        };

        let mut emitted = Vec::with_capacity(k + 1);
        emitted.push(next);
        let mut accepted = 0usize;
        let mut new_next = 0u32;
        for (i, &d) in drafts.iter().enumerate() {
            let want = row_argmax[i];
            if want == d {
                emitted.push(d);
                accepted += 1;
            } else {
                new_next = want;
                break;
            }
        }
        if accepted == k {
            new_next = row_argmax[k];
            if let Some(ssm) = s.ssm.as_mut() {
                ssm.disarm_slots();
            }
        } else {
            s.truncate(start + 1 + accepted);
            if let Some(ssm) = s.ssm.as_mut() {
                ssm.restore_slot(accepted);
            }
        }
        debug_assert_eq!(s.pos(), start + 1 + accepted);
        Ok(Round {
            emitted,
            next: new_next,
            drafted: k,
            accepted,
        })
    }

    /// Draft up to `self.k` tokens from `prediction` / `history` and verify them.
    pub fn round(&mut self, history: &[u32], prediction: &[u32], next: u32) -> Result<Round> {
        let avail = self
            .session
            .capacity()
            .saturating_sub(self.session.pos() + 1);
        let max_k = self.k.min(avail);
        let drafts = find_prompt_drafts(
            history,
            prediction,
            next,
            max_k,
            self.ngram_min,
            self.ngram_max,
        );
        self.round_with_drafts(next, &drafts)
    }

    /// Draft up to `self.k` tokens from `prediction` / `history` and verify them
    /// using a caller-supplied sampling/arbitration closure `pick(row_logits, recent_tokens) -> u32`.
    ///
    /// When `drafts` is empty, returns `emitted: vec![next]` and advances the
    /// session by 1 step using `model.forward`.
    pub fn round_with_sampler<F>(
        &mut self,
        history: &[u32],
        prediction: &[u32],
        next: u32,
        mut pick: F,
    ) -> Result<(Round, Vec<f32>)>
    where
        F: FnMut(&[f32], &[u32]) -> u32,
    {
        let avail = self
            .session
            .capacity()
            .saturating_sub(self.session.pos() + 1);
        let max_k = self.k.min(avail);
        let drafts = find_prompt_drafts(
            history,
            prediction,
            next,
            max_k,
            self.ngram_min,
            self.ngram_max,
        );
        let model = self.model;
        let s = &mut *self.session;
        let start = s.pos();
        let k = drafts.len();
        if k == 0 {
            if start + 1 > s.capacity() {
                bail!("kv capacity exhausted for a decode step");
            }
            let logits = model.forward(next, s)?;
            let mut recent = Vec::with_capacity(history.len() + 1);
            recent.extend_from_slice(history);
            recent.push(next);
            let new_next = pick(&logits, &recent);
            return Ok((
                Round {
                    emitted: vec![next],
                    next: new_next,
                    drafted: 0,
                    accepted: 0,
                },
                logits,
            ));
        }
        if start + k + 1 > s.capacity() {
            bail!("kv capacity exhausted for a speculative round");
        }
        if let Some(ssm) = s.ssm.as_mut() {
            ssm.arm_slots(k + 1);
        }

        let mut batch = Vec::with_capacity(k + 1);
        batch.push(next);
        batch.extend_from_slice(&drafts);
        let all = model.forward_batch_full(&batch, s)?;
        let vocab = model.config.vocab as usize;

        let mut emitted = Vec::with_capacity(k + 1);
        emitted.push(next);
        let mut recent = Vec::with_capacity(history.len() + k + 1);
        recent.extend_from_slice(history);
        recent.push(next);

        let mut accepted = 0usize;
        let mut new_next = 0u32;
        for (i, &d) in drafts.iter().enumerate() {
            let row_logits = &all[i * vocab..(i + 1) * vocab];
            let want = pick(row_logits, &recent);
            if want == d {
                emitted.push(d);
                recent.push(d);
                accepted += 1;
            } else {
                new_next = want;
                break;
            }
        }
        if accepted == k {
            let last_logits = &all[k * vocab..(k + 1) * vocab];
            new_next = pick(last_logits, &recent);
            if let Some(ssm) = s.ssm.as_mut() {
                ssm.disarm_slots();
            }
        } else {
            s.truncate(start + 1 + accepted);
            if let Some(ssm) = s.ssm.as_mut() {
                ssm.restore_slot(accepted);
            }
        }
        debug_assert_eq!(s.pos(), start + 1 + accepted);
        let committed_logits = all[accepted * vocab..(accepted + 1) * vocab].to_vec();
        Ok((
            Round {
                emitted,
                next: new_next,
                drafted: k,
                accepted,
            },
            committed_logits,
        ))
    }
}

/// Draft-length controller for a speculation loop.
///
/// A rejected draft is not free: the verify batch carries k+1 positions
/// whatever the outcome, and on this engine a batch costs a little more than
/// one token (measured: draft-model k=3 at 33% acceptance ran 8.2 tok/s
/// against 14.7 plain). So the draft length follows recent acceptance:
/// a fully accepted round widens k (up to `max_k`), a round that keeps less
/// than a quarter of its drafts halves it, and at k=0 the loop decodes
/// plainly for `cooldown` steps before probing again with k=1. The emitted
/// tokens never depend on k - only how many passes produce them.
#[derive(Debug, Clone)]
pub struct AdaptiveK {
    max_k: usize,
    k: usize,
    cooldown: usize,
    rest: usize,
}

impl AdaptiveK {
    pub fn new(max_k: usize) -> Self {
        Self {
            max_k,
            k: max_k,
            cooldown: 16,
            rest: 0,
        }
    }

    /// Draft tokens to propose for the next round (0 = decode plainly).
    pub fn k(&self) -> usize {
        if self.rest > 0 {
            0
        } else {
            self.k
        }
    }

    /// Record one round. `drafted == 0` is a plain step (no drafts were
    /// found or the controller is resting) and only advances the cooldown.
    pub fn record(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            if self.rest > 0 {
                self.rest -= 1;
                if self.rest == 0 {
                    self.k = 1;
                }
            }
            return;
        }
        if accepted >= drafted {
            self.k = (self.k + 1).min(self.max_k);
        } else if accepted * 4 < drafted {
            self.k /= 2;
            if self.k == 0 {
                self.rest = self.cooldown;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{find_prompt_drafts, AdaptiveK};

    #[test]
    fn adaptive_k_backs_off_on_rejections_and_recovers_on_hits() {
        let mut c = AdaptiveK::new(8);
        assert_eq!(c.k(), 8);
        c.record(8, 0);
        assert_eq!(c.k(), 4);
        c.record(4, 0);
        c.record(2, 0);
        assert_eq!(c.k(), 1);
        c.record(1, 0);
        // Resting: plain decode for the cooldown, then probe with k=1.
        assert_eq!(c.k(), 0);
        for _ in 0..15 {
            c.record(0, 0);
            assert_eq!(c.k(), 0);
        }
        c.record(0, 0);
        assert_eq!(c.k(), 1);
        // Full acceptance widens one step at a time, capped at max_k.
        for want in 2..=8 {
            c.record(c.k(), c.k());
            assert_eq!(c.k(), want);
        }
        c.record(8, 8);
        assert_eq!(c.k(), 8);
        // Partial acceptance at or above a quarter holds k.
        c.record(8, 2);
        assert_eq!(c.k(), 8);
    }

    #[test]
    fn adaptive_k_ignores_plain_steps_when_not_resting() {
        let mut c = AdaptiveK::new(4);
        c.record(0, 0);
        c.record(0, 0);
        assert_eq!(c.k(), 4);
    }

    #[test]
    fn adaptive_k_zero_max_never_drafts() {
        let mut c = AdaptiveK::new(0);
        assert_eq!(c.k(), 0);
        c.record(0, 0);
        assert_eq!(c.k(), 0);
    }

    #[test]
    fn prompt_lookup_prefers_longest_and_most_recent_ngram() {
        // History has two occurrences of [10, 20]:
        // first at 0..2 followed by [30, 40, 50],
        // second at 5..7 followed by [31, 41, 51],
        // plus a 3-gram [9, 10, 20] at 0..3 if history starts with 9.
        let history = vec![9, 10, 20, 30, 40, 50, 10, 20, 31, 41, 51, 9, 10];
        // Current suffix is [.., 9, 10] + next=20 -> 3-gram [9, 10, 20] matches index 0..3 -> [30, 40, 50]
        let drafts = find_prompt_drafts(&history, &[], 20, 3, 2, 4);
        assert_eq!(drafts, vec![30, 40, 50]);

        // If suffix is [.., 99, 10] + next=20 -> 2-gram [10, 20] matches most recent at 6..8 -> [31, 41, 51]
        let history2 = vec![9, 10, 20, 30, 40, 50, 10, 20, 31, 41, 51, 99, 10];
        let drafts2 = find_prompt_drafts(&history2, &[], 20, 3, 2, 4);
        assert_eq!(drafts2, vec![31, 41, 51]);
    }

    #[test]
    fn prompt_lookup_prioritizes_explicit_prediction_tokens() {
        let history = vec![1, 2, 3, 4, 5, 1, 2];
        let prediction = vec![1, 2, 3, 90, 91, 92];
        // Both have 3-gram [1, 2, 3], prediction wins and yields [90, 91, 92].
        let drafts = find_prompt_drafts(&history, &prediction, 3, 4, 1, 4);
        assert_eq!(drafts, vec![90, 91, 92]);
    }
}
