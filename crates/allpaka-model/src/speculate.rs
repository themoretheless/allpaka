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
        let dbg = std::env::var_os("ALLPAKA_MTP_DEBUG").is_some();
        let t0 = std::time::Instant::now();
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
        if dbg {
            eprintln!("mtp round @{start}: arm {:?}", t0.elapsed());
        }

        // The draft consumes `next`, then proposes k tokens greedily; the h
        // chains through the round from the seed (steps after the first use
        // the draft's own h_out, as llama's draft graph does).
        let mut drafts = Vec::with_capacity(k);
        let mut feed = next;
        for i in 0..k {
            let ts = std::time::Instant::now();
            let (logits, h) = model.mtp_step(&self.h, feed, start + i, s)?;
            if dbg {
                eprintln!("  draft step {i}: {:?}", ts.elapsed());
            }
            let t = argmax(&logits);
            drafts.push(t);
            self.h = h;
            feed = t;
        }
        if dbg {
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
        let tv = std::time::Instant::now();
        let hidden = model.config.hidden as usize;
        let parity = std::env::var_os("ALLPAKA_VERIFY_PARITY").is_some();
        let gpu0 = allpaka_backend::gpu::stats();
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
        if dbg {
            let gpu1 = allpaka_backend::gpu::stats();
            eprintln!(
                "  verify {} tok: {:?} ({} dispatches, wait {:.1} ms)",
                batch.len(),
                tv.elapsed(),
                gpu1.1 - gpu0.1,
                (gpu1.3 - gpu0.3) as f64 / 1e6,
            );
        }

        let mut emitted = vec![next];
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
            // one extra step appends its position to the MTP layer's KV
            // (its proposal is unused). The trunk's state is already
            // exactly at the committed prefix; the slots go stale.
            let (_, _) = model.mtp_step(
                &hidden_rows[(k - 1) * hidden..k * hidden],
                drafts[k - 1],
                start + k,
                s,
            )?;
            if let Some(ssm) = s.ssm.as_mut() {
                ssm.disarm_slots();
            }
            self.h = hidden_rows[k * hidden..(k + 1) * hidden].to_vec();
        } else {
            // Roll back to the accepted row: KV truncates to the committed
            // prefix, the SSM state restores from that row's slot - one
            // contiguous copy, no replay.
            let tr = std::time::Instant::now();
            s.truncate(start + 1 + accepted);
            if let Some(ssm) = s.ssm.as_mut() {
                ssm.restore_slot(accepted);
            }
            if dbg {
                eprintln!("  restore slot {accepted}: {:?}", tr.elapsed());
            }
            self.h = hidden_rows[accepted * hidden..(accepted + 1) * hidden].to_vec();
        }
        if dbg {
            eprintln!("  round total: {:?}\n", t0.elapsed());
        }
        debug_assert_eq!(s.pos(), start + 1 + accepted);
        Ok(Round { emitted, next: new_next, drafted: k, accepted })
    }
}
