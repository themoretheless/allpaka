//! Multi-Token Prediction (`MTP` / `nextn`) speculative draft block (`qwen35moe`).

use super::{moe::Ffn, AttnWeights, Model, Session};
use allpaka_backend::{ops, quantmat::QuantMat};
use anyhow::{bail, Result};

/// The MTP block (llama.cpp graph_mtp): one full-attention transformer layer
/// whose input is [enorm(emb(next)) | hnorm(trunk hidden)] projected by
/// eh_proj, and whose output feeds the trunk's LM head via shared_head_norm.
/// A nextn.shared_head_head / nextn.embed_tokens pair is absent in the
/// Qwen3.6 GGUF, so the trunk head and embedding are reused.
pub(crate) struct MtpLayer<'a> {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) attn: AttnWeights<'a>,
    pub(crate) q_norm: Option<Vec<f32>>,
    pub(crate) k_norm: Option<Vec<f32>>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) ffn: Ffn<'a>,
    pub(crate) eh_proj: QuantMat<'a>,
    pub(crate) enorm: Vec<f32>,
    pub(crate) hnorm: Vec<f32>,
    pub(crate) shared_head_norm: Vec<f32>,
}

impl<'a> Model<'a> {
    /// rmsnorm(row, output_norm): the trunk's final hidden in the form the
    /// MTP block's hnorm expects (llama's t_h_nextn).
    pub fn output_normed(&self, row: &[f32]) -> Vec<f32> {
        let mut v = row.to_vec();
        ops::rmsnorm(&mut v, &self.output_norm, self.config.rms_eps);
        v
    }

    /// The LM head over an already output-normed hidden row.
    pub fn lm_head(&self, normed: &[f32]) -> Result<Vec<f32>> {
        self.output_head()?.matmul(normed, 1)
    }

    /// One MTP draft step (qwen35moe nextn): run `token` through the MTP
    /// block at the explicit position `pos` and return (logits, h_out).
    /// `h_prev` seeds the chain (the trunk's output-normed hidden; later
    /// steps take the previous step's h_out). The step writes only the MTP
    /// layer's KV slot (index n_layers) at `pos` - the shared position
    /// counter stays with the trunk's verify batch.
    pub fn mtp_step(
        &self,
        h_prev: &[f32],
        token: u32,
        pos: usize,
        s: &mut Session,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.config;
        let m = self.mtp.as_ref().expect("mtp_step without an mtp block");
        if token >= c.vocab {
            bail!("token {token} out of vocabulary {}", c.vocab);
        }

        // eh_proj([enorm(emb(token)) | hnorm(h_prev)]).
        let hidden = c.hidden as usize;
        let mut cat = vec![0.0f32; 2 * hidden];
        self.embedding()?
            .row_into(token as usize, &mut cat[..hidden])?;
        ops::rmsnorm(&mut cat[..hidden], &m.enorm, c.rms_eps);
        ops::rmsnorm_into(&mut cat[hidden..], h_prev, &m.hnorm, c.rms_eps);
        let mut x = m.eh_proj.matmul(&cat, 1)?;

        // Attention, identical to a trunk full-attention decode step.
        let aw = &m.attn;
        let mut h = vec![0.0f32; hidden];
        ops::rmsnorm_into(&mut h, &x, &m.attn_norm, c.rms_eps);
        let mut qkv = QuantMat::matmul_many(&[
            (&aw.wq, h.as_slice()),
            (&aw.wk, h.as_slice()),
            (&aw.wv, h.as_slice()),
        ])?;
        let v = qkv.pop().expect("v");
        let k = qkv.pop().expect("k");
        let q = qkv.pop().expect("q");

        s.ensure_rope_cache(&self.rope_inv_freq, pos + 1);
        let rope_pairs = &s.rope_cache[pos * s.rope_cache_pairs..(pos + 1) * s.rope_cache_pairs];
        let li = c.n_layers as usize;
        let projected = self.run_cpu_attention_from_qkv(
            aw,
            m.q_norm.as_deref(),
            m.k_norm.as_deref(),
            None,
            q,
            k,
            v,
            1,
            pos,
            rope_pairs,
            &mut s.kv,
            li,
            true,
            false,
            None,
        )?;
        ops::add_assign_slice(&mut x, &projected);

        // MoE FFN, the shared head norm, and the trunk LM head (the GGUF
        // carries no nextn.shared_head_head).
        ops::rmsnorm_into(&mut h, &x, &m.ffn_norm, c.rms_eps);
        let down = m.ffn.forward(&h)?;
        ops::add_assign_slice(&mut x, &down);
        ops::rmsnorm(&mut x, &m.shared_head_norm, c.rms_eps);
        let logits = self.output_head()?.matmul(&x, 1)?;
        Ok((logits, x))
    }

    /// Fast path for MTP KV catch-up when all `k` draft tokens in a speculative
    /// round are accepted: computes only `eh_proj` + `wk`/`wv` + `k_norm`/RoPE
    /// and writes `(k, v)` into the MTP KV slot at `pos`, skipping `wq`,
    /// `attend_group`, `wo`, MoE `ffn`, and the 150k-vocab `output_head`.
    pub fn mtp_prefill_kv_only(
        &self,
        h_prev: &[f32],
        token: u32,
        pos: usize,
        s: &mut Session,
    ) -> Result<()> {
        let c = &self.config;
        let m = self
            .mtp
            .as_ref()
            .expect("mtp_prefill_kv_only without an mtp block");
        let head_dim = c.head_dim as usize;
        if token >= c.vocab {
            bail!("token {token} out of vocabulary {}", c.vocab);
        }

        let hidden = c.hidden as usize;
        let mut cat = vec![0.0f32; 2 * hidden];
        self.embedding()?
            .row_into(token as usize, &mut cat[..hidden])?;
        ops::rmsnorm(&mut cat[..hidden], &m.enorm, c.rms_eps);
        ops::rmsnorm_into(&mut cat[hidden..], h_prev, &m.hnorm, c.rms_eps);
        let mut h = m.eh_proj.matmul(&cat, 1)?;
        ops::rmsnorm(&mut h, &m.attn_norm, c.rms_eps);

        let aw = &m.attn;
        let mut kv_proj = QuantMat::matmul_many(&[(&aw.wk, h.as_slice()), (&aw.wv, h.as_slice())])?;
        let v = kv_proj.pop().expect("v");
        let mut k = kv_proj.pop().expect("k");
        s.ensure_rope_cache(&self.rope_inv_freq, pos + 1);
        let rope_pairs = &s.rope_cache[pos * s.rope_cache_pairs..(pos + 1) * s.rope_cache_pairs];
        if let Some(w) = &m.k_norm {
            for head in k.chunks_mut(head_dim) {
                ops::rmsnorm(head, w, c.rms_eps);
                self.rope_from_arrays(head, rope_pairs);
            }
        } else {
            for head in k.chunks_mut(head_dim) {
                self.rope_from_arrays(head, rope_pairs);
            }
        }
        let li = c.n_layers as usize;
        s.kv.store_at_rewrite(li, pos, &k, &v);
        Ok(())
    }
}
