//! One decode step for several sessions in one pass over the weights.
//!
//! Decoding is bandwidth-bound: one token reads every weight once. With N
//! sessions decoding together, the projections and the feed-forward run as
//! one N-row matmul (the prefill kernels), so the weights are read once for N
//! tokens. Attention stays per session, each against its own KV cache and
//! position. The result matches N separate `forward` calls
//! (`tests/cuda_decode_multi_parity.rs`: 64 of 64 greedy tokens agree in each
//! of four sessions on Qwen2.5-7B).
//!
//! This is the reference, not yet the fast path: it goes layer by layer with
//! a host round trip per step, 65 ms per step of four on an RTX 5090
//! (attention 35-50 ms, feed-forward 20-25, projections 6-10) against
//! 3.6 ms per token for the fused, graph-replayed single-session decode.
//! `serve` does not use it; a fused multi-session token buffer must beat
//! N x 3.6 ms first.

use super::attention::{AttnInputs, AttnLayer, AttnMode};
use super::{Model, Session};
use allpaka_backend::{ops, quantmat::QuantMat};
use anyhow::{bail, Result};

impl Model<'_> {
    /// Whether `forward_multi` can batch this model: dense attention layers
    /// only (the gated-delta-net recurrence keeps one state per session and
    /// is not batched here).
    pub fn supports_multi(&self) -> bool {
        self.layer_start == 0 && self.layers.iter().all(|l| l.mixer.gdn().is_none())
    }

    /// Consume `tokens[i]` in `sessions[i]` for every i, as one batched step;
    /// returns each session's logits.
    pub fn forward_multi(
        &self,
        tokens: &[u32],
        sessions: &mut [&mut Session],
    ) -> Result<Vec<Vec<f32>>> {
        let c = &self.config;
        let n = tokens.len();
        if n == 0 || n != sessions.len() {
            bail!("forward_multi: {n} tokens for {} sessions", sessions.len());
        }
        if n == 1 || !self.supports_multi() {
            return tokens
                .iter()
                .zip(sessions.iter_mut())
                .map(|(&t, s)| self.forward(t, s))
                .collect();
        }
        let hidden = c.hidden as usize;

        let embd = self.embedding()?;
        let mut xs = vec![0.0f32; n * hidden];
        for (i, &t) in tokens.iter().enumerate() {
            if t >= c.vocab {
                bail!("token {t} out of vocabulary {}", c.vocab);
            }
            embd.row_into(t as usize, &mut xs[i * hidden..(i + 1) * hidden])?;
        }
        let positions: Vec<usize> = sessions.iter().map(|s| s.pos()).collect();
        for (s, &pos) in sessions.iter_mut().zip(&positions) {
            if pos >= s.capacity() {
                bail!("session context exhausted");
            }
            s.ensure_rope_cache(&self.rope_inv_freq, pos + 1);
        }
        let mut hs = vec![0.0f32; n * hidden];
        for (li, layer) in self.layers.iter().enumerate() {
            ops::rmsnorm_rows_into(&mut hs, &xs, &layer.attn_norm, c.rms_eps);
            let aw = layer.mixer.attn().expect("attention layer without weights");
            let mut qkv =
                QuantMat::matmul_many(&[(&aw.wq, &hs[..]), (&aw.wk, &hs[..]), (&aw.wv, &hs[..])])?;
            let v = qkv.pop().expect("v");
            let k = qkv.pop().expect("k");
            let q = qkv.pop().expect("q");
            let (qw, kw) = (q.len() / n, k.len() / n);
            for (i, s) in sessions.iter_mut().enumerate() {
                let pos = positions[i];
                let pairs = s.rope_cache_pairs;
                let rope = s.rope_cache[pos * pairs..(pos + 1) * pairs].to_vec();
                let projected = self.run_cpu_attention_from_qkv(
                    AttnLayer {
                        aw,
                        q_norm: layer.q_norm.as_deref(),
                        k_norm: layer.k_norm.as_deref(),
                        bias: layer.bias.as_ref(),
                    },
                    AttnInputs {
                        q: q[i * qw..(i + 1) * qw].to_vec(),
                        k: k[i * kw..(i + 1) * kw].to_vec(),
                        v: v[i * kw..(i + 1) * kw].to_vec(),
                        m: 1,
                        base_pos: pos,
                        rope_flat: &rope,
                    },
                    &mut s.kv,
                    li,
                    AttnMode::Trunk { trace_layer: None },
                )?;
                ops::add_assign_slice(&mut xs[i * hidden..(i + 1) * hidden], &projected);
            }
            ops::rmsnorm_rows_into(&mut hs, &xs, &layer.ffn_norm, c.rms_eps);
            let down = layer.ffn.forward_batch(&hs, n, hidden)?;
            ops::add_assign_par(&mut xs, &down);
        }
        for s in sessions.iter_mut() {
            s.kv.advance();
        }
        ops::rmsnorm_rows_into(&mut hs, &xs, &self.output_norm, c.rms_eps);
        let logits = self.output_head()?.matmul(&hs, n)?;
        let vocab = logits.len() / n;
        Ok(logits.chunks(vocab).map(<[f32]>::to_vec).collect())
    }
}
