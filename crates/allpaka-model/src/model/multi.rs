//! One decode step for several sessions in one pass over the weights.
//!
//! Decoding is bandwidth-bound: one token reads every weight once. With N
//! sessions decoding together, the projections and the feed-forward run as
//! one N-row matmul (the prefill kernels), so the weights are read once for N
//! tokens. Attention stays per session, each against its own KV cache and
//! position. Two paths:
//!
//! - fused (`forward_multi_fused`): the residual stream stays on the GPU for
//!   the whole step, the prefill blocks run over one row per session
//!   (`MultiRow`: own cache, own position, split-KV flash attention), and the
//!   step is captured as a CUDA graph and replayed while the same sessions
//!   decode together. 7.2 ms for four sessions on Qwen2.5-7B, RTX 5090:
//!   1.79 ms per token against 3.67 ms one at a time.
//! - per layer, the reference the fused path falls back to (no GPU, MoE,
//!   non-NEOX RoPE, or `ALLPAKA_NO_MULTI_FUSED=1`).
//!
//! Both match one-at-a-time decode (`tests/cuda_decode_multi_parity.rs`).
//! `serve` batches the decode steps of concurrent generations through it
//! (`serve/interleave.rs`).

use super::attention::{AttnInputs, AttnLayer, AttnMode};
use super::moe::Ffn;
use super::{cpu_attn_enabled, no_tokenbuf_enabled, Model, Session};
use crate::config::RopeStyle;
use allpaka_backend::{ops, quantmat::QuantMat};
use anyhow::{bail, Result};

impl Model<'_> {
    /// Whether `forward_multi` can batch this model: dense attention layers
    /// only (the gated-delta-net recurrence keeps one state per session and
    /// is not batched here).
    pub fn supports_multi(&self) -> bool {
        self.layer_start == 0
            && self.config.gemma.is_none()
            && self.layers.iter().all(|l| l.mixer.gdn().is_none())
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
        if let Some(logits) = self.forward_multi_fused(tokens, sessions)? {
            return Ok(logits);
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
                        window: None,
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

    /// The batched step with the residual stream resident on the GPU for the
    /// whole token: the fused prefill blocks over one row per session, with
    /// per-session K/V stores and attention (`MultiRow`). `None` when it
    /// declines (no GPU, MoE, non-NEOX RoPE); any K/V it wrote before
    /// declining is rewritten identically by the reference path.
    fn forward_multi_fused(
        &self,
        tokens: &[u32],
        sessions: &mut [&mut Session],
    ) -> Result<Option<Vec<Vec<f32>>>> {
        let c = &self.config;
        let n = tokens.len();
        let hidden = c.hidden as usize;
        let head_dim = c.head_dim as usize;
        let dense = self.layers.iter().all(|l| {
            matches!(l.ffn, Ffn::Dense { .. })
                && !l.attn_norm_raw.is_empty()
                && !l.ffn_norm_raw.is_empty()
                && l.mixer.attn().is_some_and(|a| !a.gate_in_q)
        });
        if !allpaka_backend::gpu::is_attached()
            || !dense
            || !matches!(c.rope_style, RopeStyle::Neox)
            || cpu_attn_enabled()
            || no_tokenbuf_enabled()
            || std::env::var_os("ALLPAKA_NO_MULTI_FUSED").is_some()
        {
            return Ok(None);
        }
        let embd = self.embedding()?;
        let mut xs = vec![0.0f32; n * hidden];
        for (i, &t) in tokens.iter().enumerate() {
            if t >= c.vocab {
                bail!("token {t} out of vocabulary {}", c.vocab);
            }
            embd.row_into(t as usize, &mut xs[i * hidden..(i + 1) * hidden])?;
        }
        let positions: Vec<usize> = sessions.iter().map(|s| s.pos()).collect();
        let mut ropes = Vec::new();
        for (s, &pos) in sessions.iter_mut().zip(&positions) {
            if pos >= s.kv.capacity() {
                return Ok(None);
            }
            s.ensure_rope_cache(&self.rope_inv_freq, pos + 1);
            let pairs = s.rope_cache_pairs;
            ropes.extend_from_slice(&s.rope_cache[pos * pairs..(pos + 1) * pairs]);
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Replay the captured step while the same sessions decode together in
        // the same attention bucket: the graph bakes their caches and the FA
        // slot; positions and RoPE rows are staged on the device each step.
        let longest = positions.iter().map(|p| p + 1).max().unwrap_or(1);
        let bucket = longest.next_power_of_two().max(256);
        let mut key = 0x6d75_6c74_6900_0000u64
            ^ (self as *const Self as u64)
            ^ (n as u64) << 40
            ^ bucket as u64;
        for s in sessions.iter_mut() {
            let id = s.kv.gpu_view(0).map_or(0, |(cache, _, _)| cache.graph_id());
            if id == 0 {
                key = 0;
                break;
            }
            key = key.rotate_left(13) ^ id.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
        let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let rope_flat: Vec<f32> = ropes.iter().flat_map(|r| [r[0], r[1]]).collect();
        if allpaka_backend::gpu::multi_stage(key, &pos_u32, &rope_flat).is_none() {
            return Ok(None);
        }
        allpaka_backend::gpu::prefill_graph_key(key);
        if let Some(out) = allpaka_backend::gpu::prefill_replay(&xs) {
            return self.finish_multi(out, sessions).map(Some);
        }
        allpaka_backend::gpu::prefill_graph_key(key);
        if allpaka_backend::gpu::prefill_begin(&xs).is_none() {
            return Ok(None);
        }
        let ok = (|| -> Option<()> {
            for (li, layer) in self.layers.iter().enumerate() {
                let aw = layer.mixer.attn()?;
                let mut rows = Vec::with_capacity(n);
                for (s, &pos) in sessions.iter_mut().zip(&positions) {
                    let (cache, k_off, v_off) = s.kv.gpu_view(li)?;
                    rows.push(allpaka_backend::gpu::MultiRow {
                        cache,
                        k_off,
                        v_off,
                        pos,
                    });
                }
                QuantMat::prefill_attn_block(
                    &aw.wq,
                    &aw.wk,
                    &aw.wv,
                    &aw.wo,
                    &xs,
                    n,
                    layer.q_norm.as_deref(),
                    layer.k_norm.as_deref(),
                    &ropes,
                    c.rope_dim as usize,
                    false,
                    layer.bias.as_ref().map(|b| (b.q.1, b.k.1, b.v.1)),
                    c.rms_eps,
                    rows[0].cache,
                    (rows[0].k_off, rows[0].v_off),
                    (
                        c.kv_dim(),
                        head_dim,
                        c.n_heads as usize,
                        c.n_kv_heads as usize,
                    ),
                    positions[0],
                    scale,
                    Some(allpaka_backend::gpu::PrefillFusion {
                        attn_norm: layer.attn_norm_raw,
                        ffn_norm: layer.ffn_norm_raw,
                        router: &[],
                        n_expert: 0,
                    }),
                    Some(&rows),
                )?;
                let Ffn::Dense {
                    w_gate,
                    w_up,
                    w_down,
                    ..
                } = &layer.ffn
                else {
                    return None;
                };
                let tok: Vec<u32> = (0..n as u32).collect();
                let tok_off: Vec<u32> = (0..=n as u32).collect();
                let hit_w = vec![1.0f32; n];
                let next_norm = self
                    .layers
                    .get(li + 1)
                    .map(|l| (l.attn_norm_raw, c.rms_eps))
                    .filter(|(raw, _)| !raw.is_empty());
                QuantMat::ffn_grouped(
                    w_gate,
                    w_up,
                    w_down,
                    1,
                    hidden,
                    w_gate.n_out,
                    &[[0u32, 0u32, n as u32]],
                    &[],
                    &tok,
                    n,
                    Some(allpaka_backend::gpu::GroupedCombine {
                        tok_off: &tok_off,
                        hit_row: &tok,
                        hit_w: &hit_w,
                        m: n,
                        next_norm,
                    }),
                    None,
                    None,
                )?;
            }
            Some(())
        })();
        if ok.is_none() {
            allpaka_backend::gpu::prefill_abort();
            return Ok(None);
        }
        let Some(out) = allpaka_backend::gpu::prefill_end_output(n * hidden) else {
            return Ok(None);
        };
        self.finish_multi(out.into_vec(), sessions).map(Some)
    }

    /// Advance every session past its token and apply the output head.
    fn finish_multi(
        &self,
        mut hs: Vec<f32>,
        sessions: &mut [&mut Session],
    ) -> Result<Vec<Vec<f32>>> {
        let c = &self.config;
        let hidden = c.hidden as usize;
        let n = sessions.len();
        for s in sessions.iter_mut() {
            s.kv.advance();
        }
        for row in hs.chunks_mut(hidden) {
            ops::rmsnorm(row, &self.output_norm, c.rms_eps);
        }
        let logits = self.output_head()?.matmul(&hs, n)?;
        let vocab = logits.len() / n;
        Ok(logits.chunks(vocab).map(<[f32]>::to_vec).collect())
    }
}
