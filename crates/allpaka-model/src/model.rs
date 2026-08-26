//! The forward pass: tokens in, logits out.
//!
//! One code path serves every Llama-family model; the two honest differences
//! (RoPE pairing, QK norms) come in through [`Config`]. Everything is written
//! against the CPU reference ops - this is the correctness baseline that the
//! accelerated path will be compared to, and it is deliberately simple enough
//! to audit against the paper.

use crate::config::{Config, Gating, MoeConfig, RopeStyle};
use crate::kv::KvCache;
use crate::profile;
use allpaka_backend::{ops, QuantMat};
use allpaka_gguf::GgufFile;
use anyhow::{bail, Context, Result};

/// q/k/v projection biases (GLM's attention_bias): the parsed vectors for the
/// CPU path, the raw F32 mmap bytes for GPU binds.
struct AttnBias<'a> {
    q: (Vec<f32>, &'a [u8]),
    k: (Vec<f32>, &'a [u8]),
    v: (Vec<f32>, &'a [u8]),
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    /// The same weights as raw F32 mmap bytes, for GPU binds (empty when the
    /// tensor is not plain F32 and the whole-token path then declines).
    attn_norm_raw: &'a [u8],
    ffn_norm_raw: &'a [u8],
    wq: QuantMat<'a>,
    wk: QuantMat<'a>,
    wv: QuantMat<'a>,
    wo: QuantMat<'a>,
    bias: Option<AttnBias<'a>>,
    /// Qwen3 per-head norms, absent on Llama/Mistral.
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    ffn: Ffn<'a>,
}

/// The always-on shared expert of GLM-style MoE: a dense FFN of
/// `expert_ffn * n_shared` width, added to the routed output unweighted.
struct SharedFfn<'a> {
    gate: QuantMat<'a>,
    up: QuantMat<'a>,
    down: QuantMat<'a>,
    ffn: usize,
}

impl SharedFfn<'_> {
    /// The shared FFN over `m` normed rows; borrow rules keep it out of
    /// [`Ffn`]'s match arms.
    fn forward_batch(&self, hs: &[f32], m: usize) -> Result<Vec<f32>> {
        if let Some(mut outs) = QuantMat::ffn_many(&[(&self.gate, &self.up, &self.down, hs)]) {
            return Ok(outs.pop().expect("one fused item"));
        }
        let mut gate = self.gate.matmul(hs, m)?;
        let up = self.up.matmul(hs, m)?;
        ops::swiglu(&mut gate, &up);
        self.down.matmul(&gate, m)
    }
}

enum Ffn<'a> {
    Dense {
        w_gate: QuantMat<'a>,
        w_up: QuantMat<'a>,
        w_down: QuantMat<'a>,
    },
    /// The stacked-expert tensors of a MoE layer. Expert `e` is a
    /// `slice_rows` band of each stack; nothing is copied per token.
    Moe {
        router: QuantMat<'a>,
        /// GLM's sigmoid router bias (`exp_probs_b`), added to the logits
        /// before gating; raw bytes kept for GPU binds.
        router_bias: Option<(Vec<f32>, &'a [u8])>,
        gate_exps: QuantMat<'a>,
        up_exps: QuantMat<'a>,
        down_exps: QuantMat<'a>,
        shared: Option<SharedFfn<'a>>,
        gating: Gating,
        weights_norm: bool,
        weights_scale: f32,
        expert_ffn: usize,
        hidden: usize,
        n_used: usize,
    },
}

impl Ffn<'_> {
    /// The FFN half of a block over `m` normed activation rows: deltas for
    /// the residual stream out.
    fn forward_batch(&self, hs: &[f32], m: usize, hidden: usize) -> Result<Vec<f32>> {
        match self {
            Ffn::Dense { w_gate, w_up, w_down } => {
                // One command buffer for the whole FFN when the GPU takes it.
                if let Some(mut outs) = QuantMat::ffn_many(&[(w_gate, w_up, w_down, hs)]) {
                    return Ok(outs.pop().expect("one fused item"));
                }
                let mut gate = w_gate.matmul(hs, m)?;
                let up = w_up.matmul(hs, m)?;
                ops::swiglu(&mut gate, &up);
                w_down.matmul(&gate, m)
            }
            Ffn::Moe { .. } => {
                let mut out = vec![0f32; m * hidden];
                // Each token routes to its own experts; the matmuls of every
                // (token, expert) pair in the chunk still land in single
                // batched submissions.
                self.moe_batch(hs, m, hidden, &mut out)?;
                Ok(out)
            }
        }
    }

    /// The routed path for a whole chunk. Written separately to keep the
    /// borrow bookkeeping of the per-pair weight slices readable.
    fn moe_batch(&self, hs: &[f32], m: usize, hidden: usize, out: &mut [f32]) -> Result<()> {
        let Ffn::Moe {
            router,
            router_bias,
            gate_exps,
            up_exps,
            down_exps,
            shared,
            gating,
            weights_norm,
            weights_scale,
            expert_ffn,
            n_used,
            ..
        } = self
        else {
            unreachable!("moe_batch is only called on the Moe variant");
        };
        // The shared expert runs for every token, unweighted; it is
        // independent of the routing below, so it lands first.
        if let Some(sh) = shared {
            let d = sh.forward_batch(hs, m)?;
            for (o, d) in out.iter_mut().zip(&d) {
                *o += d;
            }
        }
        let router_span = profile::span(profile::Phase::Router);
        // Raw router logits; the selection bias is applied inside
        // route_gated AFTER gating (selection-only, llama.cpp semantics).
        let probs_all = router.matmul(hs, m)?;
        let n_expert = router.n_out;
        let rbias = router_bias.as_ref().map(|rb| rb.0.as_slice());

        // Group the chunk's tokens by the expert they routed to: every token
        // that picked expert `e` shares one matmul, so `e`'s weights are
        // streamed once per chunk. This grouping is the whole reason MoE
        // prefill can be fast - per-(token, expert) matmuls would re-read the
        // weights for every token.
        // Per-token routing is independent: softmax + top-k spread over
        // cores, the group build stays serial (it is just pushes).
        let routed: Vec<Vec<(usize, f32)>> = {
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let per = m.div_ceil(threads).max(1);
            let mut routed: Vec<Vec<(usize, f32)>> = vec![Vec::new(); m];
            std::thread::scope(|scope| {
                for (o, chunk) in routed.chunks_mut(per).zip(probs_all.chunks(per * n_expert)) {
                    scope.spawn(move || {
                        for (dst, row) in o.iter_mut().zip(chunk.chunks(n_expert)) {
                            *dst = route_gated(row, *n_used, *gating, *weights_norm, *weights_scale, rbias);
                        }
                    });
                }
            });
            routed
        };
        let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_expert];
        for (i, r) in routed.iter().enumerate() {
            for &(e, weight) in r {
                groups[e].push((i, weight));
            }
        }
        let used: Vec<usize> = (0..n_expert).filter(|&e| !groups[e].is_empty()).collect();
        drop(router_span);

        let slice_span = profile::span(profile::Phase::ExpertSlice);
        drop(slice_span);
        let ffn_span = profile::span(profile::Phase::Ffn);

        // The grouped path: every hit expert's whole FFN as ONE dispatch per
        // stage, with a (group -> expert, rows) table on the GPU. The
        // gathered activations go straight into the flat buffer - the old
        // per-group Vec then re-concatenate copied every activation twice
        // (~12 GB per 235B prefill) - and the disjoint destination ranges
        // let the copy spread over cores.
        let grouped = {
            let mut table = Vec::with_capacity(used.len());
            let mut tok = Vec::new();
            let mut row0 = 0u32;
            for &e in &used {
                let rows = groups[e].len() as u32;
                table.push([e as u32, row0, rows]);
                for &(i, _) in &groups[e] {
                    tok.push(i as u32);
                }
                row0 += rows;
            }
            let total_rows = row0 as usize;
            // The gather itself happens inside the gate/up kernels via the
            // token table - no CPU-side activation copy at all.
            QuantMat::ffn_grouped(
                gate_exps, up_exps, down_exps,
                n_expert, hidden, *expert_ffn,
                &table, hs, &tok, total_rows, None, None, None,
            )
            .map(|flat| (flat, table))
        };
        if let Some((flat, table)) = grouped {
            drop(ffn_span);
            let _s = profile::span(profile::Phase::FfnCombine);
            // Invert group->token into token->rows so the weighted adds
            // parallelise over tokens: each out row then has exactly one
            // writer.
            let mut hits: Vec<Vec<(usize, f32)>> = vec![Vec::with_capacity(*n_used); m];
            for (gi, &e) in used.iter().enumerate() {
                let row0 = table[gi][1] as usize;
                for (row_idx, &(i, weight)) in groups[e].iter().enumerate() {
                    hits[i].push((row0 + row_idx, weight));
                }
            }
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let per = m.div_ceil(threads).max(1);
            let flat_ref = &flat;
            std::thread::scope(|scope| {
                for (c, hc) in out.chunks_mut(per * hidden).zip(hits.chunks(per)) {
                    scope.spawn(move || {
                        for (out_row, h) in c.chunks_mut(hidden).zip(hc) {
                            for &(row, weight) in h {
                                let down = &flat_ref[row * hidden..(row + 1) * hidden];
                                for (o, d) in out_row.iter_mut().zip(down) {
                                    *o += weight * d;
                                }
                            }
                        }
                    });
                }
            });
            return Ok(());
        }

        // Fallback paths only from here on (the grouped dispatch declined):
        // rebuild the per-group weight slices and gathered activations the
        // grouped path no longer materialises.
        let mut gate_mats = Vec::with_capacity(used.len());
        let mut up_mats = Vec::with_capacity(used.len());
        let mut down_mats = Vec::with_capacity(used.len());
        let mut gathered = Vec::with_capacity(used.len());
        for &e in &used {
            gate_mats.push(gate_exps.slice_rows(e * expert_ffn, *expert_ffn)?);
            up_mats.push(up_exps.slice_rows(e * expert_ffn, *expert_ffn)?);
            down_mats.push(down_exps.slice_rows(e * hidden, hidden)?);
            let mut xe = Vec::with_capacity(groups[e].len() * hidden);
            for &(i, _) in &groups[e] {
                xe.extend_from_slice(&hs[i * hidden..(i + 1) * hidden]);
            }
            gathered.push(xe);
        }

        // Every expert's whole FFN in one command buffer when the GPU takes
        // it; otherwise the two-batch matmul_many path with CPU swiglu.
        let fused_items: Vec<(&QuantMat, &QuantMat, &QuantMat, &[f32])> = (0..used.len())
            .map(|gi| (&gate_mats[gi], &up_mats[gi], &down_mats[gi], gathered[gi].as_slice()))
            .collect();
        let downs = match QuantMat::ffn_many(&fused_items) {
            Some(d) => d,
            None => {
                let mut items = Vec::with_capacity(used.len() * 2);
                for gi in 0..used.len() {
                    items.push((&gate_mats[gi], gathered[gi].as_slice()));
                    items.push((&up_mats[gi], gathered[gi].as_slice()));
                }
                let mut projected = QuantMat::matmul_many(&items)?;

                let mut gates = Vec::with_capacity(used.len());
                for _ in 0..used.len() {
                    let up = projected.pop().expect("one up per expert");
                    let mut gate = projected.pop().expect("one gate per expert");
                    ops::swiglu(&mut gate, &up);
                    gates.push(gate);
                }
                gates.reverse();

                let down_items: Vec<(&QuantMat, &[f32])> = gates
                    .iter()
                    .enumerate()
                    .map(|(gi, gate)| (&down_mats[gi], gate.as_slice()))
                    .collect();
                QuantMat::matmul_many(&down_items)?
            }
        };

        drop(ffn_span);
        let _s = profile::span(profile::Phase::FfnCombine);
        for (gi, &e) in used.iter().enumerate() {
            for (row_idx, &(i, weight)) in groups[e].iter().enumerate() {
                let down = &downs[gi][row_idx * hidden..(row_idx + 1) * hidden];
                let out_row = &mut out[i * hidden..(i + 1) * hidden];
                for (o, d) in out_row.iter_mut().zip(down) {
                    *o += weight * d;
                }
            }
        }
        Ok(())
    }

    /// The FFN half of a block: normed activations in, delta for the residual
    /// stream out.
    fn forward(&self, h: &[f32]) -> Result<Vec<f32>> {
        match self {
            Ffn::Dense { w_gate, w_up, w_down } => {
                if let Some(mut outs) = QuantMat::ffn_many(&[(w_gate, w_up, w_down, h)]) {
                    return Ok(outs.pop().expect("one fused item"));
                }
                let mut gate = w_gate.matmul(h, 1)?;
                let up = w_up.matmul(h, 1)?;
                ops::swiglu(&mut gate, &up);
                w_down.matmul(&gate, 1)
            }
            Ffn::Moe {
                router,
                router_bias,
                gate_exps,
                up_exps,
                down_exps,
                shared,
                gating,
                weights_norm,
                weights_scale,
                expert_ffn,
                hidden,
                n_used,
            } => {
                let router_span = profile::span(profile::Phase::Router);
                let logits = router.matmul(h, 1)?;

                // Every selected expert's gate and up projections are
                // independent, so they run as one batch (one GPU wait, or one
                // CPU thread each); the down projections batch the same way
                // after the CPU applies SwiGLU.
                let picked = route_gated(
                    &logits,
                    *n_used,
                    *gating,
                    *weights_norm,
                    *weights_scale,
                    router_bias.as_ref().map(|rb| rb.0.as_slice()),
                );
                drop(router_span);

                let slice_span = profile::span(profile::Phase::ExpertSlice);
                let mut gate_mats = Vec::with_capacity(picked.len());
                let mut up_mats = Vec::with_capacity(picked.len());
                let mut down_mats = Vec::with_capacity(picked.len());
                for &(e, _) in &picked {
                    gate_mats.push(gate_exps.slice_rows(e * expert_ffn, *expert_ffn)?);
                    up_mats.push(up_exps.slice_rows(e * expert_ffn, *expert_ffn)?);
                    down_mats.push(down_exps.slice_rows(e * hidden, *hidden)?);
                }
                drop(slice_span);

                // All routed experts' FFNs in one command buffer when the
                // GPU takes it; the two-batch path otherwise.
                let fused_items: Vec<(&QuantMat, &QuantMat, &QuantMat, &[f32])> = (0..picked
                    .len())
                    .map(|i| (&gate_mats[i], &up_mats[i], &down_mats[i], h))
                    .collect();
                let ffn_span = profile::span(profile::Phase::Ffn);
                let downs = match QuantMat::ffn_many(&fused_items) {
                    Some(d) => d,
                    None => {
                        let mut items = Vec::with_capacity(picked.len() * 2);
                        for i in 0..picked.len() {
                            items.push((&gate_mats[i], h));
                            items.push((&up_mats[i], h));
                        }
                        let mut projected = QuantMat::matmul_many(&items)?;

                        let mut gates = Vec::with_capacity(picked.len());
                        for i in (0..picked.len()).rev() {
                            let up = projected.pop().expect("one up per expert");
                            let mut gate = projected.pop().expect("one gate per expert");
                            ops::swiglu(&mut gate, &up);
                            gates.push((i, gate));
                        }
                        gates.reverse();

                        let down_items: Vec<(&QuantMat, &[f32])> = gates
                            .iter()
                            .map(|(i, gate)| (&down_mats[*i], gate.as_slice()))
                            .collect();
                        QuantMat::matmul_many(&down_items)?
                    }
                };

                drop(ffn_span);

                let _s = profile::span(profile::Phase::FfnCombine);
                let mut out = vec![0f32; *hidden];
                for ((_, weight), down) in picked.iter().zip(downs) {
                    for (o, d) in out.iter_mut().zip(&down) {
                        *o += weight * d;
                    }
                }
                // The shared expert runs for every token, unweighted.
                if let Some(sh) = shared {
                    let d = sh.forward_batch(h, 1)?;
                    for (o, d) in out.iter_mut().zip(&d) {
                        *o += d;
                    }
                }
                Ok(out)
            }
        }
    }
}

/// Turn raw router logits into the top `k` (expert, weight) picks under the
/// model's gating rule: softmax across experts (Qwen3) or independent
/// sigmoid scores (GLM). `weights_norm` renormalises the winners to sum to 1
/// (`norm_topk_prob`); `scale` multiplies the final weights.
/// Returned best-first.
pub fn route_gated(
    logits: &[f32],
    k: usize,
    gating: Gating,
    norm: bool,
    scale: f32,
    bias: Option<&[f32]>,
) -> Vec<(usize, f32)> {
    let mut scores = logits.to_vec();
    match gating {
        Gating::Softmax => ops::softmax(&mut scores),
        Gating::Sigmoid => ops::sigmoid(&mut scores),
    }
    // GLM/DeepSeek-style selection bias: it shifts ONLY the top-k choice;
    // the weights stay the unbiased gated scores (llama.cpp does the same:
    // `selection_probs = probs + exp_probs_b`, weights from `probs`).
    let selection: Vec<f32> = match bias {
        Some(b) => scores.iter().zip(b).map(|(s, b)| s + b).collect(),
        None => scores.clone(),
    };
    // Partial selection, not a full sort: this runs once per token per layer
    // (480 x 94 in a prefill), and sorting 128 floats to take 8 was a
    // measurable slice of the router phase. select_nth is O(n), the tail
    // sort is over k elements. Returned best-first, like the sort produced.
    let k = k.min(scores.len());
    let mut order: Vec<usize> = (0..scores.len()).collect();
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, |&a, &b| selection[b].total_cmp(&selection[a]));
        order.truncate(k);
    }
    order.sort_by(|&a, &b| selection[b].total_cmp(&selection[a]));
    let total: f32 = order.iter().map(|&e| scores[e]).sum();
    order
        .into_iter()
        .map(|e| {
            let w = if norm && total > 0.0 {
                scores[e] / total
            } else if norm {
                1.0 / k as f32
            } else {
                scores[e]
            };
            (e, w * scale)
        })
        .collect()
}

/// Pick the top `k` experts and renormalise their probabilities to sum to 1,
/// which is what Qwen3-MoE does (`norm_topk_prob`). Returned best-first.
///
/// `probs` must already be softmaxed; this is the pre-GLM contract kept for
/// the tests that pin it. New code takes raw logits and [`route_gated`].
pub fn route(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(probs.len());
    let mut order: Vec<usize> = (0..probs.len()).collect();
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, |&a, &b| probs[b].total_cmp(&probs[a]));
        order.truncate(k);
    }
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let total: f32 = order.iter().map(|&e| probs[e]).sum();
    order
        .into_iter()
        .map(|e| (e, if total > 0.0 { probs[e] / total } else { 1.0 / k as f32 }))
        .collect()
}

pub struct Model<'a> {
    pub config: Config,
    embd: QuantMat<'a>,
    layers: Vec<Layer<'a>>,
    output_norm: Vec<f32>,
    output_norm_raw: &'a [u8],
    output: QuantMat<'a>,
    /// `base^(-2i/d)` per rotary pair, hoisted out of the per-head loops.
    rope_inv_freq: Vec<f32>,
}

/// Mutable decode state: the cache plus the implicit position.
pub struct Session {
    kv: KvCache,
    rope_cache: Vec<[f32; 2]>,
    rope_cache_pairs: usize,
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
        if self.rope_cache_pairs > 0 {
            let keep_pairs = keep.min(self.rope_cache.len() / self.rope_cache_pairs);
            self.rope_cache.truncate(keep_pairs * self.rope_cache_pairs);
        }
    }

    fn rope_cache(&mut self, rope_inv_freq: &[f32], start: usize, count: usize) -> &[[f32; 2]] {
        debug_assert_eq!(rope_inv_freq.len(), self.rope_cache_pairs);
        if count == 0 {
            return &[];
        }
        let needed_positions = start + count;
        let have = self.rope_cache.len() / self.rope_cache_pairs;
        if have < needed_positions {
            for pos in have..needed_positions {
                for &freq in rope_inv_freq {
                    self.rope_cache.push((pos as f32 * freq).sin_cos().into());
                }
            }
        }
        let start = start * self.rope_cache_pairs;
        let end = needed_positions * self.rope_cache_pairs;
        &self.rope_cache[start..end]
    }
}

impl<'a> Model<'a> {
    pub fn load(f: &'a GgufFile) -> Result<Self> {
        let config = Config::from_gguf(f)?;
        // Offer the weights to the GPU; on a machine without Metal this is a
        // no-op and every matmul stays on the CPU reference. Split files get
        // one GPU window set per part.
        for m in f.mappings() {
            allpaka_backend::gpu::attach(m);
        }
        let hidden = config.hidden as usize;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let ffn = config.ffn_hidden as usize;

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for i in 0..config.n_layers {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            // GLM calls the pre-FFN norm "post_attention_norm".
            let ffn_norm = if f.tensor(&name("ffn_norm")).is_some() {
                name("ffn_norm")
            } else {
                name("post_attention_norm")
            };
            let bias = if config.has_attn_bias {
                Some(AttnBias {
                    q: bias_vec_raw(f, &format!("blk.{i}.attn_q.bias"), q_dim)?,
                    k: bias_vec_raw(f, &format!("blk.{i}.attn_k.bias"), kv_dim)?,
                    v: bias_vec_raw(f, &format!("blk.{i}.attn_v.bias"), kv_dim)?,
                })
            } else {
                None
            };
            layers.push(Layer {
                attn_norm: norm_vec(f, &name("attn_norm"), hidden)?,
                attn_norm_raw: norm_raw(f, &name("attn_norm")),
                ffn_norm_raw: norm_raw(f, &ffn_norm),
                wq: qmat(f, &name("attn_q"), q_dim, hidden)?,
                wk: qmat(f, &name("attn_k"), kv_dim, hidden)?,
                wv: qmat(f, &name("attn_v"), kv_dim, hidden)?,
                wo: qmat(f, &name("attn_output"), hidden, q_dim)?,
                bias,
                q_norm: if config.has_qk_norm {
                    Some(norm_vec(f, &name("attn_q_norm"), config.head_dim as usize)?)
                } else {
                    None
                },
                k_norm: if config.has_qk_norm {
                    Some(norm_vec(f, &name("attn_k_norm"), config.head_dim as usize)?)
                } else {
                    None
                },
                ffn_norm: norm_vec(f, &ffn_norm, hidden)?,
                ffn: match &config.moe {
                    // GLM's leading blocks are plain dense FFNs before the
                    // routing starts.
                    Some(moe) if i < moe.leading_dense => Ffn::Dense {
                        w_gate: qmat(f, &name("ffn_gate"), ffn, hidden)?,
                        w_up: qmat(f, &name("ffn_up"), ffn, hidden)?,
                        w_down: qmat(f, &name("ffn_down"), hidden, ffn)?,
                    },
                    None => Ffn::Dense {
                        w_gate: qmat(f, &name("ffn_gate"), ffn, hidden)?,
                        w_up: qmat(f, &name("ffn_up"), ffn, hidden)?,
                        w_down: qmat(f, &name("ffn_down"), hidden, ffn)?,
                    },
                    Some(moe) => moe_ffn(f, &name, moe, hidden)?,
                },
            });
        }

        // A model without a separate output head ties it to the embedding.
        let output = if f.tensor("output.weight").is_some() {
            qmat(f, "output.weight", config.vocab as usize, hidden)?
        } else {
            qmat(f, "token_embd.weight", config.vocab as usize, hidden)?
        };

        Ok(Model {
            embd: qmat(f, "token_embd.weight", config.vocab as usize, hidden)?,
            output_norm: norm_vec(f, "output_norm.weight", hidden)?,
            output_norm_raw: norm_raw(f, "output_norm.weight"),
            layers,
            output,
            rope_inv_freq: ops::rope_inv_freq(config.rope_dim as usize, config.rope_freq_base),
            config,
        })
    }

    pub fn new_session(&self, context_tokens: usize) -> Session {
        Session {
            kv: KvCache::new(
                self.config.n_layers as usize,
                self.config.kv_dim(),
                context_tokens,
            ),
            rope_cache: Vec::new(),
            rope_cache_pairs: self.config.rope_dim as usize / 2,
        }
    }

    /// Consume a chunk of tokens at consecutive positions, return logits for
    /// the last one.
    ///
    /// This is the prefill path: the whole chunk moves through each weight
    /// matrix as one batched matmul, so the weights are streamed once per
    /// chunk instead of once per token. Attention stays per-token (it is
    /// inherently causal and cheap next to the FFN at prompt lengths this
    /// serves).
    /// How many prompt tokens to push through `forward_batch` at once.
    ///
    /// One chunk is one pass over the weights, so bigger amortises directly
    /// now that the mm kernels, grouped expert dispatch and residency set
    /// exist: 480-in-one-chunk measured 112 tok/s on the 235B against 99 at
    /// 256. 512 matches llama.cpp's default ubatch. (An earlier sweep on the
    /// pre-mm engine concluded chunk size barely mattered - that verdict
    /// died with the kernels it measured.)
    ///
    /// The costs are activation memory (linear in the chunk) and a coarser
    /// unit of work; `ALLPAKA_PREFILL_CHUNK` overrides it for a machine where
    /// that memory is tight.
    pub fn prefill_chunk() -> usize {
        std::env::var("ALLPAKA_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(512)
    }

    pub fn forward_batch(&self, tokens: &[u32], s: &mut Session) -> Result<Vec<f32>> {
        let m = tokens.len();
        let hidden = self.config.hidden as usize;
        let xs = self.forward_batch_hidden(tokens, s)?;
        let mut last = xs[(m - 1) * hidden..].to_vec();
        ops::rmsnorm(&mut last, &self.output_norm, self.config.rms_eps);
        self.output.matmul(&last, 1)
    }

    /// The shared body of the batch paths: every layer over every position,
    /// returning the residual stream `[m, hidden]` before the output norm.
    fn forward_batch_hidden(&self, tokens: &[u32], s: &mut Session) -> Result<Vec<f32>> {
        let c = &self.config;
        let m = tokens.len();
        if m == 0 {
            bail!("empty token batch");
        }
        for &t in tokens {
            if t >= c.vocab {
                bail!("token {t} out of vocabulary {}", c.vocab);
            }
        }
        let hidden = c.hidden as usize;
        let head_dim = c.head_dim as usize;
        let base = s.pos();

        let embed_span = profile::span(profile::Phase::Embed);
        let mut xs = Vec::with_capacity(m * hidden);
        for &t in tokens {
            xs.extend_from_slice(&self.embd.row(t as usize)?);
        }
        let rope_flat = s.rope_cache(&self.rope_inv_freq, base, m).to_vec();
        let half = c.rope_dim as usize / 2;
        let prefill_fused_ok = matches!(c.rope_style, RopeStyle::Neox)
            && std::env::var_os("ALLPAKA_CPU_ATTN").is_none()
            && std::env::var_os("ALLPAKA_NO_TOKENBUF").is_none();
        drop(embed_span);

        // The fully fused prefill: xs lives on the GPU for the whole chunk,
        // norms / residual adds / router / combine all run there, and the
        // CPU only routes tokens to experts between the two command buffers
        // of each layer. Any decline falls back to the per-layer path below
        // with the ORIGINAL xs - every cache write is recomputed
        // identically, so a mid-chunk decline costs time, not correctness.
        if prefill_fused_ok && std::env::var("ALLPAKA_PREFUSE").map_or(true, |v| v != "0") {
            if let Some(done) = self.forward_batch_fused(&xs, &rope_flat, m, base, s) {
                s.kv.commit(base + m);
                return Ok(done);
            }
        }

        // One scratch for the normalised activations, reused by every layer:
        // the old per-layer `xs.clone()` cost an allocation and an extra
        // write pass over ~8 MB, twice per layer.
        let mut hs_scratch = vec![0f32; m * hidden];
        for (li, layer) in self.layers.iter().enumerate() {
            // Attention, batched projections.
            let hs = {
                let _s = profile::span(profile::Phase::AttnNorm);
                ops::rmsnorm_rows_into(&mut hs_scratch, &xs, &layer.attn_norm, c.rms_eps);
                &hs_scratch
            };
            // The whole attention half - qkv, per-row norm+rope, cache
            // store, causal attention, output projection - as one command
            // buffer, mirroring the decode token buffer. Declines fall to
            // the step-by-step path below.
            if prefill_fused_ok {
                let fused = {
                    let _s = profile::span(profile::Phase::Attend);
                    let scale = 1.0 / (head_dim as f32).sqrt();
                    s.kv.gpu_view(li).and_then(|(cache, k_off, v_off)| {
                        QuantMat::prefill_attn_block(
                            &layer.wq, &layer.wk, &layer.wv, &layer.wo,
                            hs, m,
                            layer.q_norm.as_deref(),
                            layer.k_norm.as_deref(),
                            &rope_flat,
                            c.rope_dim as usize,
                            layer.bias.as_ref().map(|b| (b.q.1, b.k.1, b.v.1)),
                            c.rms_eps,
                            cache,
                            (k_off, v_off),
                            (c.kv_dim(), head_dim, c.n_heads as usize, c.n_kv_heads as usize),
                            base,
                            scale,
                            None,
                        )
                    })
                };
                if let Some(projected) = fused {
                    {
                        let _s = profile::span(profile::Phase::AttnOut);
                        ops::add_assign_par(&mut xs, &projected);
                    }
                    let hs = {
                        let _s = profile::span(profile::Phase::FfnNorm);
                        ops::rmsnorm_rows_into(&mut hs_scratch, &xs, &layer.ffn_norm, c.rms_eps);
                        &hs_scratch
                    };
                    let down = layer.ffn.forward_batch(hs, m, hidden)?;
                    {
                        let _s = profile::span(profile::Phase::FfnCombine);
                        ops::add_assign_par(&mut xs, &down);
                    }
                    continue;
                }
            }

            // Batch the three projection matrices together: on the GPU this is
            // one command buffer and one wait instead of three independent
            // decode passes, which matters a lot for prefill throughput.
            let qkv_span = profile::span(profile::Phase::Qkv);
            let mut qkv = QuantMat::matmul_many(&[
                (&layer.wq, &hs[..]),
                (&layer.wk, &hs[..]),
                (&layer.wv, &hs[..]),
            ])?;
            drop(qkv_span);
            let mut v = qkv.pop().expect("v");
            let mut k = qkv.pop().expect("k");
            let mut q = qkv.pop().expect("q");
            if let Some(b) = &layer.bias {
                add_bias_rows(&mut q, &b.q.0);
                add_bias_rows(&mut k, &b.k.0);
                add_bias_rows(&mut v, &b.v.0);
            }

            let rope_span = profile::span(profile::Phase::QkNormRope);
            for (i, row) in q.chunks_mut(c.q_dim()).enumerate() {
                let rope = &rope_flat[i * half..(i + 1) * half];
                for head in row.chunks_mut(head_dim) {
                    if let Some(w) = &layer.q_norm {
                        ops::rmsnorm(head, w, c.rms_eps);
                    }
                    self.rope_from_arrays(head, rope);
                }
            }
            for (i, row) in k.chunks_mut(c.kv_dim()).enumerate() {
                let rope = &rope_flat[i * half..(i + 1) * half];
                for head in row.chunks_mut(head_dim) {
                    if let Some(w) = &layer.k_norm {
                        ops::rmsnorm(head, w, c.rms_eps);
                    }
                    self.rope_from_arrays(head, rope);
                }
            }
            drop(rope_span);
            let store_span = profile::span(profile::Phase::KvStore);
            for i in 0..m {
                s.kv.store_at(
                    li,
                    base + i,
                    &k[i * c.kv_dim()..(i + 1) * c.kv_dim()],
                    &v[i * c.kv_dim()..(i + 1) * c.kv_dim()],
                );
            }

            // Per-token causal attention over the cache; causality is each
            // row's own `n_pos`, not a mask. The whole chunk goes to the GPU
            // as one command buffer; without a device the rows run on CPU
            // threads, chunked to the core count.
            drop(store_span);
            let attend_span = profile::span(profile::Phase::Attend);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_out = vec![0f32; m * c.q_dim()];
            let q_ref = &q;
            let cpu_only = std::env::var_os("ALLPAKA_CPU_ATTN").is_some();
            let on_gpu = {
                let kv = &mut s.kv;
                kv.gpu_view(li).filter(|_| !cpu_only).and_then(|(cache, k_off, v_off)| {
                    let reqs: Vec<allpaka_backend::gpu::AttnReq> = (0..m)
                        .map(|i| allpaka_backend::gpu::AttnReq {
                            cache,
                            k_off,
                            v_off,
                            q: &q_ref[i * c.q_dim()..(i + 1) * c.q_dim()],
                            kv_dim: c.kv_dim(),
                            head_dim,
                            n_q_heads: c.n_heads as usize,
                            group: c.group_size(),
                            n_pos: base + i + 1,
                            scale,
                        })
                        .collect();
                    allpaka_backend::gpu::attend_batch(&reqs)
                })
            };
            match on_gpu {
                Some(rows) => {
                    for (out, row) in attn_out.chunks_mut(c.q_dim()).zip(rows) {
                        out.copy_from_slice(&row);
                    }
                }
                None => {
                    let threads =
                        std::thread::available_parallelism().map_or(1, |n| n.get());
                    let rows_per = m.div_ceil(threads).max(1);
                    let kv = &s.kv;
                    std::thread::scope(|scope| {
                        for (ci, out_chunk) in
                            attn_out.chunks_mut(rows_per * c.q_dim()).enumerate()
                        {
                            let first = ci * rows_per;
                            scope.spawn(move || {
                                for (i, out_row) in
                                    out_chunk.chunks_mut(c.q_dim()).enumerate()
                                {
                                    let row = first + i;
                                    attend_one(
                                        c, kv, li, base + row,
                                        &q_ref[row * c.q_dim()..(row + 1) * c.q_dim()],
                                        out_row, scale,
                                    );
                                }
                            });
                        }
                    });
                }
            }
            drop(attend_span);
            {
                let _s = profile::span(profile::Phase::AttnOut);
                let projected = layer.wo.matmul(&attn_out, m)?;
                for (a, b) in xs.iter_mut().zip(&projected) {
                    *a += b;
                }
            }

            // Feed-forward, batched.
            let hs = {
                let _s = profile::span(profile::Phase::FfnNorm);
                ops::rmsnorm_rows_into(&mut hs_scratch, &xs, &layer.ffn_norm, c.rms_eps);
                &hs_scratch
            };
            let down = layer.ffn.forward_batch(hs, m, hidden)?;
            {
                let _s = profile::span(profile::Phase::FfnCombine);
                ops::add_assign_par(&mut xs, &down);
            }
        }
        s.kv.commit(base + m);
        Ok(xs)
    }

    /// The GPU-resident prefill chunk; `None` means "fall back to the
    /// per-layer path" and guarantees no partial state the fallback could
    /// trip over (the KV cache is rewritten identically from layer 0).
    fn forward_batch_fused(
        &self,
        xs: &[f32],
        rope_flat: &[[f32; 2]],
        m: usize,
        base: usize,
        s: &mut Session,
    ) -> Option<Vec<f32>> {
        let c = &self.config;
        let hidden = c.hidden as usize;
        let head_dim = c.head_dim as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        allpaka_backend::gpu::prefill_begin(xs)?;
        for (li, layer) in self.layers.iter().enumerate() {
            if layer.attn_norm_raw.is_empty() || layer.ffn_norm_raw.is_empty() {
                return None;
            }
            // Router bytes for the fused attention block: MoE layers have a
            // real F32 router; a leading Dense layer (GLM's layer 0) has
            // none and skips the router matmul entirely.
            let (router_raw, n_expert) = match &layer.ffn {
                Ffn::Moe { router, .. } => {
                    let (router_ty, router_raw) = router.raw();
                    if router_ty != allpaka_gguf::GgmlType::F32 {
                        return None;
                    }
                    (router_raw, router.n_out)
                }
                Ffn::Dense { .. } => (&[][..], 0),
                _ => return None,
            };
            let (cache, k_off, v_off) = s.kv.gpu_view(li)?;
            let logits = {
                let _s = profile::span(profile::Phase::Attend);
                QuantMat::prefill_attn_block(
                    &layer.wq, &layer.wk, &layer.wv, &layer.wo,
                    xs, m,
                    layer.q_norm.as_deref(),
                    layer.k_norm.as_deref(),
                    rope_flat,
                    c.rope_dim as usize,
                    layer.bias.as_ref().map(|b| (b.q.1, b.k.1, b.v.1)),
                    c.rms_eps,
                    cache,
                    (k_off, v_off),
                    (c.kv_dim(), head_dim, c.n_heads as usize, c.n_kv_heads as usize),
                    base,
                    scale,
                    Some(allpaka_backend::gpu::PrefillFusion {
                        attn_norm: layer.attn_norm_raw,
                        ffn_norm: layer.ffn_norm_raw,
                        router: router_raw,
                        n_expert,
                    }),
                )?
            };

            // CPU routing between the two command buffers: gating + top-k
            // per token over the logits the attention buffer produced. A
            // Dense layer is one "group" covering every token with weight 1.
            let router_span = profile::span(profile::Phase::Router);
            let (gate_m, up_m, down_m, ffn_w, sh_m);
            let mut gpu_route = None;
            let (table, tok, total_rows, hits_per_token) = match &layer.ffn {
                Ffn::Moe {
                    gate_exps, up_exps, down_exps, shared, expert_ffn, n_used,
                    gating, weights_norm, weights_scale, router_bias, ..
                } => {
                    gate_m = gate_exps;
                    up_m = up_exps;
                    down_m = down_exps;
                    ffn_w = *expert_ffn;
                    sh_m = shared.as_ref().map(|sh| (&sh.gate, &sh.up, &sh.down));
                    if logits.is_empty() {
                        // GPU routing: the attention block left the logits in
                        // y_arena; route_pick/scan/scatter build everything
                        // below on-device at the head of the FFN buffer.
                        if *gating != Gating::Sigmoid || *n_used > 8 {
                            return None;
                        }
                        gpu_route = Some(allpaka_backend::gpu::GroupedRoute {
                            n_used: *n_used,
                            norm: *weights_norm,
                            scale: *weights_scale,
                            bias: router_bias.as_ref().map(|b| b.0.as_slice()),
                        });
                        (Vec::new(), Vec::new(), m * *n_used, Vec::new())
                    } else {
                    let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_expert];
                    for i in 0..m {
                        let row = &logits[i * n_expert..(i + 1) * n_expert];
                        for (e, weight) in route_gated(
                            row, *n_used, *gating, *weights_norm, *weights_scale,
                            router_bias.as_ref().map(|b| b.0.as_slice()),
                        ) {
                            groups[e].push((i, weight));
                        }
                    }
                    let used: Vec<usize> =
                        (0..n_expert).filter(|&e| !groups[e].is_empty()).collect();
                    let mut table = Vec::with_capacity(used.len());
                    let mut tok = Vec::new();
                    let mut row0 = 0u32;
                    for &e in &used {
                        let rows = groups[e].len() as u32;
                        table.push([e as u32, row0, rows]);
                        for &(i, _) in &groups[e] {
                            tok.push(i as u32);
                        }
                        row0 += rows;
                    }
                    let total_rows = row0 as usize;
                    // CSR hits per token for the GPU-side combine. GLM's
                    // shared expert adds one weight-1 hit per token pointing
                    // at the rows right after the expert rows.
                    let mut hits: Vec<Vec<(u32, f32)>> = vec![Vec::new(); m];
                    for (gi, &e) in used.iter().enumerate() {
                        let r0 = table[gi][1];
                        for (ri, &(i, weight)) in groups[e].iter().enumerate() {
                            hits[i].push((r0 + ri as u32, weight));
                        }
                    }
                    if shared.is_some() {
                        for (i, h) in hits.iter_mut().enumerate() {
                            h.push(((total_rows + i) as u32, 1.0));
                        }
                    }
                    (table, tok, total_rows, hits)
                    }
                }
                Ffn::Dense { w_gate, w_up, w_down, .. } => {
                    gate_m = w_gate;
                    up_m = w_up;
                    down_m = w_down;
                    ffn_w = w_gate.n_out;
                    sh_m = None;
                    (
                        vec![[0u32, 0u32, m as u32]],
                        (0..m as u32).collect::<Vec<_>>(),
                        m,
                        (0..m).map(|i| vec![(i as u32, 1.0f32)]).collect::<Vec<_>>(),
                    )
                }
                _ => return None,
            };
            let mut tok_off = Vec::with_capacity(m + 1);
            let mut hit_row = Vec::with_capacity(total_rows + m);
            let mut hit_w = Vec::with_capacity(total_rows + m);
            tok_off.push(0u32);
            for h in &hits_per_token {
                for &(r, w) in h {
                    hit_row.push(r);
                    hit_w.push(w);
                }
                tok_off.push(hit_row.len() as u32);
            }
            drop(router_span);

            let _f = profile::span(profile::Phase::Ffn);
            QuantMat::ffn_grouped(
                gate_m, up_m, down_m,
                n_expert.max(1), hidden, ffn_w,
                &table, &[], &tok, total_rows,
                Some(allpaka_backend::gpu::GroupedCombine {
                    tok_off: &tok_off,
                    hit_row: &hit_row,
                    hit_w: &hit_w,
                    m,
                }),
                sh_m,
                gpu_route,
            )?;
        }
        let mut out = vec![0f32; m * hidden];
        allpaka_backend::gpu::prefill_end(&mut out)?;
        Some(out)
    }

    /// Like [`forward_batch`], but returns the logits of EVERY position,
    /// `[m, vocab]` row-major.
    ///
    /// This is what speculative verification consumes: the target model runs
    /// the draft's tokens as one batch and needs its own next-token opinion
    /// at each of them, not just the last. The cost over `forward_batch` is
    /// m output projections instead of one - noise next to the layers.
    pub fn forward_batch_full(&self, tokens: &[u32], s: &mut Session) -> Result<Vec<f32>> {
        let m = tokens.len();
        let hidden = self.config.hidden as usize;
        let mut xs = self.forward_batch_hidden(tokens, s)?;
        for row in xs.chunks_mut(hidden) {
            ops::rmsnorm(row, &self.output_norm, self.config.rms_eps);
        }
        self.output.matmul(&xs, m)
    }

    /// Build the whole-token GPU request and run it; Ok(None) means the GPU
    /// declined and the caller should take the per-layer path.
    fn forward_token_gpu(
        &self,
        token: u32,
        s: &mut Session,
        pos: usize,
    ) -> Result<Option<Vec<f32>>> {
        use allpaka_backend::gpu::{TokenFfn, TokenLayer, TokenReq};
        let c = &self.config;
        let dbg = std::env::var_os("ALLPAKA_TOKENBUF_DEBUG").is_some();
        if self.output_norm_raw.is_empty() {
            if dbg { eprintln!("tokenbuf declined: output norm not raw F32"); }
            return Ok(None);
        }
        let rope_table = s.rope_cache(&self.rope_inv_freq, pos, 1).to_vec();
        let Some((cache, _, _)) = s.kv.gpu_view_ref(0) else {
            if dbg { eprintln!("tokenbuf declined: no gpu cache view"); }
            return Ok(None);
        };
        let x = self.embd.row(token as usize)?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for (li, layer) in self.layers.iter().enumerate() {
            if layer.attn_norm_raw.is_empty() || layer.ffn_norm_raw.is_empty() {
                if dbg { eprintln!("tokenbuf declined: layer {li} norm not raw F32"); }
                return Ok(None);
            }
            let (_, k_off, v_off) =
                s.kv.gpu_view_ref(li).expect("region wrapped, checked above");
            let ffn = match &layer.ffn {
                Ffn::Dense { w_gate, w_up, w_down } => {
                    let (gt, gb) = w_gate.raw();
                    let (ut, ub) = w_up.raw();
                    let (dt, db) = w_down.raw();
                    TokenFfn::Dense {
                        gate: (gt, gb, w_gate.n_out),
                        up: (ut, ub, w_up.n_out),
                        down: (dt, db, w_down.n_out),
                    }
                }
                Ffn::Moe { router, router_bias, gate_exps, up_exps, down_exps, shared, gating, weights_norm, weights_scale, expert_ffn, n_used, .. } => {
                    let (rt, rb) = router.raw();
                    let (gt, gb) = gate_exps.raw();
                    let (ut, ub) = up_exps.raw();
                    let (dt, db) = down_exps.raw();
                    let sigmoid = matches!(gating, Gating::Sigmoid);
                    // The GPU top-k kernels only implement renormalized
                    // weights with scale 1; anything else stays on the CPU.
                    if sigmoid && (!*weights_norm || *weights_scale != 1.0) {
                        if dbg { eprintln!("tokenbuf declined: sigmoid gating with norm/scale"); }
                        return Ok(None);
                    }
                    let shared = shared.as_ref().map(|sh| {
                        let (gt, gb) = sh.gate.raw();
                        let (ut, ub) = sh.up.raw();
                        let (dt, db) = sh.down.raw();
                        [
                            (gt, gb, sh.gate.n_out),
                            (ut, ub, sh.up.n_out),
                            (dt, db, sh.down.n_out),
                        ]
                    });
                    TokenFfn::Moe {
                        router: (rt, rb, router.n_out),
                        router_bias: router_bias.as_ref().map(|b| b.1),
                        gate: (gt, gb),
                        up: (ut, ub),
                        down: (dt, db),
                        expert_ffn: *expert_ffn,
                        n_used: *n_used,
                        sigmoid,
                        shared,
                    }
                }
            };
            let (qt, qb) = layer.wq.raw();
            let (kt, kb) = layer.wk.raw();
            let (vt, vb) = layer.wv.raw();
            let (ot, ob) = layer.wo.raw();
            layers.push(TokenLayer {
                attn_norm: layer.attn_norm_raw,
                ffn_norm: layer.ffn_norm_raw,
                wq: (qt, qb, layer.wq.n_out),
                wk: (kt, kb, layer.wk.n_out),
                wv: (vt, vb, layer.wv.n_out),
                wo: (ot, ob, layer.wo.n_out),
                q_norm: layer.q_norm.as_deref(),
                k_norm: layer.k_norm.as_deref(),
                q_bias: layer.bias.as_ref().map(|b| b.q.1),
                k_bias: layer.bias.as_ref().map(|b| b.k.1),
                v_bias: layer.bias.as_ref().map(|b| b.v.1),
                k_off,
                v_off,
                ffn,
            });
        }
        let (out_ty, out_bytes) = self.output.raw();
        Ok(allpaka_backend::gpu::decode_token(&TokenReq {
            x: &x,
            layers: &layers,
            cache,
            kv_dim: c.kv_dim(),
            head_dim: c.head_dim as usize,
            n_heads: c.n_heads as usize,
            n_kv_heads: c.n_kv_heads as usize,
            pos,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
            rope: &rope_table,
            rot_dim: c.rope_dim as usize,
            eps: c.rms_eps,
            output_norm: self.output_norm_raw,
            output: (out_ty, out_bytes, self.output.n_out),
        }))
    }

    /// Consume one token, return logits over the vocabulary.
    pub fn forward(&self, token: u32, s: &mut Session) -> Result<Vec<f32>> {
        let c = &self.config;
        if token >= c.vocab {
            bail!("token {token} out of vocabulary {}", c.vocab);
        }
        let head_dim = c.head_dim as usize;
        let pos = s.pos();

        // The whole token as ONE GPU command buffer: every layer's attention
        // and FFN (routing included, on the GPU), through to the vocabulary
        // logits. Declines - non-NEOX rope, a norm that is not F32, no
        // device, kill switches - fall through to the per-layer path below,
        // which remains the reference this path is tested against.
        if matches!(c.rope_style, RopeStyle::Neox)
            && pos < s.kv.capacity()
            && std::env::var_os("ALLPAKA_CPU_ATTN").is_none()
            && std::env::var_os("ALLPAKA_NO_TOKENBUF").is_none()
        {
            if let Some(logits) = self.forward_token_gpu(token, s, pos)? {
                let _ = profile::span(profile::Phase::Attend);
                s.kv.advance();
                return Ok(logits);
            }
        }

        let mut x = {
            let _s = profile::span(profile::Phase::Embed);
            self.embd.row(token as usize)?
        };

        let rope_pairs = s.rope_cache(&self.rope_inv_freq, pos, 1).to_vec();
        let cpu_attn = std::env::var_os("ALLPAKA_CPU_ATTN").is_some();

        for (li, layer) in self.layers.iter().enumerate() {
            // Attention.
            let h = {
                let _s = profile::span(profile::Phase::AttnNorm);
                let mut h = x.clone();
                ops::rmsnorm(&mut h, &layer.attn_norm, c.rms_eps);
                h
            };

            // The whole attention half - qkv, norms, rope, cache store,
            // attention, output projection - as one GPU command buffer. By
            // the GPU's own clock, five short buffers cost more in driver
            // scheduling than in execution, so the win here is the merge
            // itself. Declines (non-NEOX rope, no GPU) fall through to the
            // step-by-step path below.
            if matches!(c.rope_style, RopeStyle::Neox) && !cpu_attn && pos < s.kv.capacity() {
                let scale = 1.0 / (head_dim as f32).sqrt();
                let blocked = {
                    let _s = profile::span(profile::Phase::Attend);
                    s.kv.gpu_view(li).and_then(|(cache, k_off, v_off)| {
                        QuantMat::attn_block(
                            &layer.wq, &layer.wk, &layer.wv, &layer.wo,
                            &h,
                            layer.q_norm.as_deref(),
                            layer.k_norm.as_deref(),
                            &rope_pairs,
                            c.rms_eps,
                            cache,
                            (k_off, v_off),
                            (c.kv_dim(), head_dim, c.n_heads as usize, c.n_kv_heads as usize),
                            pos,
                            scale,
                        )
                    })
                };
                if let Some(projected) = blocked {
                    for (a, b) in x.iter_mut().zip(&projected) {
                        *a += b;
                    }
                    let h = {
                        let _s = profile::span(profile::Phase::FfnNorm);
                        let mut h = x.clone();
                        ops::rmsnorm(&mut h, &layer.ffn_norm, c.rms_eps);
                        h
                    };
                    let down = layer.ffn.forward(&h)?;
                    let _s = profile::span(profile::Phase::FfnCombine);
                    for (a, b) in x.iter_mut().zip(&down) {
                        *a += b;
                    }
                    continue;
                }
            }
            // q, k and v are independent: one batch, one GPU wait.
            let mut qkv = {
                let _s = profile::span(profile::Phase::Qkv);
                QuantMat::matmul_many(&[
                    (&layer.wq, h.as_slice()),
                    (&layer.wk, h.as_slice()),
                    (&layer.wv, h.as_slice()),
                ])?
            };
            let mut v = qkv.pop().expect("v");
            let mut k = qkv.pop().expect("k");
            let mut q = qkv.pop().expect("q");
            if let Some(b) = &layer.bias {
                add_bias_rows(&mut q, &b.q.0);
                add_bias_rows(&mut k, &b.k.0);
                add_bias_rows(&mut v, &b.v.0);
            }

            {
                let _s = profile::span(profile::Phase::QkNormRope);
                for head in q.chunks_mut(head_dim) {
                    if let Some(w) = &layer.q_norm {
                        ops::rmsnorm(head, w, c.rms_eps);
                    }
                    self.rope_from_arrays(head, &rope_pairs);
                }
                for head in k.chunks_mut(head_dim) {
                    if let Some(w) = &layer.k_norm {
                        ops::rmsnorm(head, w, c.rms_eps);
                    }
                    self.rope_from_arrays(head, &rope_pairs);
                }
            }

            {
                let _s = profile::span(profile::Phase::KvStore);
                s.kv.store(li, &k, &v);
            }

            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_out = vec![0f32; c.q_dim()];
            // One thread per kv head: each streams its K/V exactly once for
            // the whole GQA group, so parallelism never multiplies traffic.
            let kv = &s.kv;
            let q_ref = &q;
            let group_span = c.group_size() * head_dim;
            // Attention and the output projection go to the GPU together:
            // the kernel reads the cache where the CPU just wrote it (the
            // storage is page-aligned and wrapped once per session, so
            // nothing is copied), and its result feeds `wo` inside the same
            // command buffer. None means no device, or a head width the
            // kernel is not written for, and then both run on the CPU.
            let fused = {
                let _s = profile::span(profile::Phase::Attend);
                let cpu_only = std::env::var_os("ALLPAKA_CPU_ATTN").is_some();
                kv.gpu_view_ref(li).filter(|_| !cpu_only).and_then(|(cache, k_off, v_off)| {
                    layer.wo.attend_project(&allpaka_backend::gpu::AttnReq {
                        cache,
                        k_off,
                        v_off,
                        q: q_ref,
                        kv_dim: c.kv_dim(),
                        head_dim,
                        n_q_heads: c.n_heads as usize,
                        group: c.group_size(),
                        n_pos: pos + 1,
                        scale,
                    })
                })
            };
            let projected = match fused {
                Some(projected) => projected,
                None => {
                    {
                        let _s = profile::span(profile::Phase::Attend);
                        std::thread::scope(|scope| {
                            for (kv_head, out_group) in
                                attn_out.chunks_mut(group_span).enumerate()
                            {
                                scope.spawn(move || {
                                    attend_group(
                                        c, kv, li, pos,
                                        &q_ref[kv_head * group_span..(kv_head + 1) * group_span],
                                        kv_head, out_group, scale,
                                    );
                                });
                            }
                        });
                    }
                    let _s = profile::span(profile::Phase::AttnOut);
                    layer.wo.matmul(&attn_out, 1)?
                }
            };
            for (a, b) in x.iter_mut().zip(&projected) {
                *a += b;
            }

            // Feed-forward, dense or routed.
            let h = {
                let _s = profile::span(profile::Phase::FfnNorm);
                let mut h = x.clone();
                ops::rmsnorm(&mut h, &layer.ffn_norm, c.rms_eps);
                h
            };
            let down = layer.ffn.forward(&h)?;
            {
                let _s = profile::span(profile::Phase::FfnCombine);
                for (a, b) in x.iter_mut().zip(&down) {
                    *a += b;
                }
            }
        }
        s.kv.advance();

        let _s = profile::span(profile::Phase::Output);
        ops::rmsnorm(&mut x, &self.output_norm, c.rms_eps);
        self.output.matmul(&x, 1)
    }

    /// Rotate one head by a per-position table from [`ops::rope_sin_cos`].
    /// The table is shared by every head of every layer at that position.
    fn rope_from_arrays(&self, head: &mut [f32], table: &[[f32; 2]]) {
        match self.config.rope_style {
            RopeStyle::Norm => ops::rope_norm_cached_from_array(head, table),
            RopeStyle::Neox => ops::rope_neox_cached_from_array(head, table),
        }
    }
}

/// One kv head attending for its whole GQA group in a single K/V pass.
///
/// The group's query heads are contiguous in the q row (head `g` serves q
/// indices `g*group..(g+1)*group`), so one streaming pass over the cached K
/// and V feeds all of them - `group_size` times less cache traffic than the
/// naive pass per q head, which is what decode bandwidth drowns in at long
/// context. The softmax is the online form (running max, rescaled
/// accumulators), so no per-position score buffer is allocated either.
///
/// `q_group` and `out_group` are the group's `group_size * head_dim` slices
/// of the q and output rows; `out_group` must arrive zeroed.
#[inline(always)]
fn attend_group(
    c: &Config,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    q_group: &[f32],
    kv_head: usize,
    out_group: &mut [f32],
    scale: f32,
) {
    let head_dim = c.head_dim as usize;
    let group = c.group_size();
    debug_assert_eq!(q_group.len(), group * head_dim);
    debug_assert_eq!(out_group.len(), group * head_dim);

    let mut maxes = vec![f32::NEG_INFINITY; group];
    let mut denoms = vec![0f32; group];

    for t in 0..=pos {
        let kt = kv.k_at(layer, t, kv_head, head_dim);
        let vt = kv.v_at(layer, t, kv_head, head_dim);

        for h in 0..group {
            let qh = &q_group[h * head_dim..(h + 1) * head_dim];
            // Attention's inner product is the same shape as every other one
            // in the engine and had the same problem: a single f32
            // accumulator the compiler is not allowed to reassociate.
            let s = ops::f16::dot(qh, kt) * scale;

            let acc_start = h * head_dim;
            let acc = &mut out_group[acc_start..acc_start + head_dim];
            if s > maxes[h] {
                if denoms[h] > 0.0 {
                    let rescale = (maxes[h] - s).exp();
                    for a in acc.iter_mut() {
                        *a *= rescale;
                    }
                    denoms[h] *= rescale;
                }
                maxes[h] = s;
            }

            let p = (s - maxes[h]).exp();
            denoms[h] += p;
            ops::f16::axpy(acc, p, vt);
        }
    }

    for h in 0..group {
        let d = denoms[h].max(f32::MIN_POSITIVE);
        let acc = &mut out_group[h * head_dim..(h + 1) * head_dim];
        for a in acc.iter_mut() {
            *a /= d;
        }
    }
}

/// All heads of one token row attending over the cache: one group pass per
/// kv head.
fn attend_one(
    c: &Config,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    q_row: &[f32],
    out_row: &mut [f32],
    scale: f32,
) {
    let head_dim = c.head_dim as usize;
    let group = c.group_size();
    for kv_head in 0..c.n_kv_heads as usize {
        let at = kv_head * group * head_dim;
        attend_group(
            c, kv, layer, pos,
            &q_row[at..at + group * head_dim],
            kv_head,
            &mut out_row[at..at + group * head_dim],
            scale,
        );
    }
}

/// The naive reference: one q head, one full pass, buffered softmax. Kept
/// only to pin `attend_group` against an independently simple formulation.
#[cfg(test)]
fn attend_head(
    c: &Config,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    qh: &[f32],
    qi: usize,
    out: &mut [f32],
    scale: f32,
) {
    let head_dim = c.head_dim as usize;
    let kv_head = qi / c.group_size();
    // Deliberately the naive form - buffer every score, then softmax - so it
    // arbitrates the online version. It reads the cache one half at a time
    // for the same reason: no shared helper to be wrong in both places.
    let mut scores: Vec<f32> = (0..=pos)
        .map(|t| {
            let kt = kv.k_at(layer, t, kv_head, head_dim);
            qh.iter().zip(kt).map(|(a, &b)| a * ops::f16::to_f32(b)).sum::<f32>() * scale
        })
        .collect();
    ops::softmax(&mut scores);
    for (t, &p) in scores.iter().enumerate() {
        let vt = kv.v_at(layer, t, kv_head, head_dim);
        for (o, &vv) in out.iter_mut().zip(vt) {
            *o += p * ops::f16::to_f32(vv);
        }
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use crate::config::RopeStyle;

    /// Attach the GPU to a throwaway page-aligned region, once per process.
    /// The weights it points at are never read - the attention kernel takes
    /// its cache from a separate wrapping - but `attach` is what creates the
    /// device and the pipelines. False when the host has no Metal device
    /// (GitHub-hosted runners); the test then skips instead of failing.
    fn attach_a_device_for_tests() -> bool {
        use std::sync::OnceLock;
        static REGION: OnceLock<bool> = OnceLock::new();
        *REGION.get_or_init(|| {
            const PAGE: usize = 16384;
            let layout = std::alloc::Layout::from_size_align(PAGE, PAGE).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            let region: &'static [u8] = unsafe { std::slice::from_raw_parts(ptr, PAGE) };
            allpaka_backend::gpu::attach(region)
        })
    }

    /// The Metal attention kernel against the CPU path it replaces, on a
    /// head width the kernel actually supports (128) and a position count
    /// that is not a multiple of its four SIMD groups, so the flash merge
    /// has to combine groups that saw different numbers of positions.
    #[test]
    fn gpu_attention_matches_the_cpu_path() {
        if !attach_a_device_for_tests() {
            eprintln!("SKIP: no Metal device");
            return;
        }
        // A small odd case, then the 30B's real shape at a context long
        // enough that every SIMD group rescales its accumulator many times.
        check_gpu_attention(8, 2, 71);
        check_gpu_attention(16, 4, 544);
    }

    fn check_gpu_attention(n_heads: u32, n_kv_heads: u32, positions: usize) {
        let head_dim = 128usize;
        let c = Config {
            architecture: "test".into(),
            n_layers: 1,
            hidden: n_heads * head_dim as u32,
            n_heads,
            n_kv_heads,
            head_dim: head_dim as u32,
            ffn_hidden: 8,
            vocab: 8,
            rms_eps: 1e-6,
            rope_freq_base: 10000.0,
            rope_style: RopeStyle::Neox,
            has_qk_norm: false,
            moe: None,
            has_attn_bias: false,
            rope_dim: head_dim as u32,
        };
        // The GPU module wakes up on the first `attach`, which normally comes
        // from loading a model; without one the cache is never wrapped and
        // this test would quietly compare nothing.
        attach_a_device_for_tests();

        let kv_dim = c.kv_dim();
        let mut kv = KvCache::new(1, kv_dim, positions);
        let mut state = 0x0fed_cba9_8765_4321u64;
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..positions {
            let k: Vec<f32> = (0..kv_dim).map(|_| rnd()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| rnd()).collect();
            kv.store(0, &k, &v);
            kv.advance();
        }
        let q: Vec<f32> = (0..c.q_dim()).map(|_| rnd()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let pos = positions - 1;

        let Some((cache, k_off, v_off)) = kv.gpu_view(0) else {
            panic!("no Metal device, or the cache was not wrapped for it");
        };
        let Some(got) = allpaka_backend::gpu::attend(&allpaka_backend::gpu::AttnReq {
            cache,
            k_off,
            v_off,
            q: &q,
            kv_dim,
            head_dim,
            n_q_heads: n_heads as usize,
            group: c.group_size(),
            n_pos: positions,
            scale,
        }) else {
            panic!("a GPU is attached but the kernel declined a 128-wide head");
        };

        let mut want = vec![0f32; c.q_dim()];
        attend_one(&c, &kv, 0, pos, &q, &mut want, scale);

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4 * (1.0 + w.abs()),
                "element {i}: gpu {g}, cpu {w}"
            );
        }
    }

    /// The online-softmax group pass must agree with the naive buffered
    /// per-head reference to float precision.
    #[test]
    fn attend_group_matches_the_naive_reference() {
        let (n_heads, n_kv_heads, head_dim, positions) = (8u32, 2u32, 16usize, 33usize);
        let c = Config {
            architecture: "test".into(),
            n_layers: 1,
            hidden: n_heads * head_dim as u32,
            n_heads,
            n_kv_heads,
            head_dim: head_dim as u32,
            ffn_hidden: 8,
            vocab: 8,
            rms_eps: 1e-6,
            rope_freq_base: 10000.0,
            rope_style: RopeStyle::Neox,
            has_qk_norm: false,
            moe: None,
            has_attn_bias: false,
            rope_dim: head_dim as u32,
        };
        let kv_dim = c.kv_dim();
        let mut kv = KvCache::new(1, kv_dim, positions);
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..positions {
            let k: Vec<f32> = (0..kv_dim).map(|_| rnd()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| rnd()).collect();
            kv.store(0, &k, &v);
            kv.advance();
        }
        let q: Vec<f32> = (0..c.q_dim()).map(|_| rnd()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let pos = positions - 1;

        let mut got = vec![0f32; c.q_dim()];
        attend_one(&c, &kv, 0, pos, &q, &mut got, scale);

        let mut want = vec![0f32; c.q_dim()];
        for qi in 0..n_heads as usize {
            attend_head(
                &c, &kv, 0, pos,
                &q[qi * head_dim..(qi + 1) * head_dim],
                qi,
                &mut want[qi * head_dim..(qi + 1) * head_dim],
                scale,
            );
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5 * (1.0 + w.abs()), "element {i}: {g} vs {w}");
        }
    }
}

/// The stacked-expert weights of one MoE layer.
fn moe_ffn<'a>(
    f: &'a GgufFile,
    name: &dyn Fn(&str) -> String,
    moe: &MoeConfig,
    hidden: usize,
) -> Result<Ffn<'a>> {
    let n_expert = moe.n_expert as usize;
    let expert_ffn = moe.expert_ffn as usize;
    // GLM's always-on shared expert: `n_shared` expert-width FFNs stacked
    // into one dense triple (`ffn_gate_shexp` and friends).
    let shared = if moe.n_shared > 0 {
        let shared_ffn = expert_ffn * moe.n_shared as usize;
        Some(SharedFfn {
            gate: qmat(f, &name("ffn_gate_shexp"), shared_ffn, hidden)?,
            up: qmat(f, &name("ffn_up_shexp"), shared_ffn, hidden)?,
            down: qmat(f, &name("ffn_down_shexp"), hidden, shared_ffn)?,
            ffn: shared_ffn,
        })
    } else {
        None
    };
    // GLM's sigmoid router adds a learned bias to the logits.
    let router_bias_name = name("exp_probs_b").replace(".weight", ".bias");
    let router_bias = if f.tensor(&router_bias_name).is_some() {
        Some(bias_vec_raw(f, &router_bias_name, n_expert)?)
    } else {
        None
    };
    Ok(Ffn::Moe {
        router: qmat(f, &name("ffn_gate_inp"), n_expert, hidden)?,
        router_bias,
        gate_exps: qmat3(f, &name("ffn_gate_exps"), expert_ffn, hidden, n_expert)?,
        up_exps: qmat3(f, &name("ffn_up_exps"), expert_ffn, hidden, n_expert)?,
        down_exps: qmat3(f, &name("ffn_down_exps"), hidden, expert_ffn, n_expert)?,
        shared,
        gating: moe.gating,
        weights_norm: moe.weights_norm,
        weights_scale: moe.weights_scale,
        expert_ffn,
        hidden,
        n_used: moe.n_used as usize,
    })
}

/// A projection bias vector: dequantised for the CPU path, raw F32 bytes for
/// GPU binds (empty raw when the tensor is not F32, which then declines the
/// whole-token GPU path).
fn bias_vec_raw<'a>(f: &'a GgufFile, tensor: &str, len: usize) -> Result<(Vec<f32>, &'a [u8])> {
    Ok((norm_vec(f, tensor, len)?, norm_raw(f, tensor)))
}

/// Add a per-row bias to a row-major buffer whose row length is `bias.len()`.
fn add_bias_rows(rows: &mut [f32], bias: &[f32]) {
    for row in rows.chunks_exact_mut(bias.len()) {
        for (x, b) in row.iter_mut().zip(bias) {
            *x += b;
        }
    }
}

/// A stacked-expert tensor `[n_in, n_out, n_expert]`, opened as one tall
/// matrix of `n_out * n_expert` rows; experts are addressed by row bands.
fn qmat3<'a>(
    f: &'a GgufFile,
    name: &str,
    n_out: usize,
    n_in: usize,
    n_expert: usize,
) -> Result<QuantMat<'a>> {
    let t = f.tensor(name).with_context(|| format!("GGUF has no tensor {name:?}"))?;
    if t.dims.len() != 3
        || t.dims[0] != n_in as u64
        || t.dims[1] != n_out as u64
        || t.dims[2] != n_expert as u64
    {
        bail!(
            "tensor {name:?} has shape {:?}, expected [{n_in}, {n_out}, {n_expert}]",
            t.dims
        );
    }
    QuantMat::new(f.data(t)?, t.ggml_type, n_out * n_expert, n_in)
}

/// A weight matrix by name, with its GGUF shape checked against what the
/// graph expects. A silently transposed tensor produces plausible garbage;
/// a named shape error produces a fix.
fn qmat<'a>(f: &'a GgufFile, name: &str, n_out: usize, n_in: usize) -> Result<QuantMat<'a>> {
    let t = f.tensor(name).with_context(|| format!("GGUF has no tensor {name:?}"))?;
    if t.dims.len() != 2 || t.dims[0] != n_in as u64 || t.dims[1] != n_out as u64 {
        bail!("tensor {name:?} has shape {:?}, expected [{n_in}, {n_out}]", t.dims);
    }
    QuantMat::new(f.data(t)?, t.ggml_type, n_out, n_in)
}

/// A norm weight vector, dequantised once at load.
/// The raw bytes of an F32 norm tensor inside the mmap, or empty. The GPU
/// binds these directly; anything but F32 falls back to per-layer paths.
fn norm_raw<'a>(f: &'a GgufFile, name: &str) -> &'a [u8] {
    f.tensor(name)
        .filter(|t| t.ggml_type == allpaka_gguf::GgmlType::F32)
        .and_then(|t| f.data(t).ok())
        .unwrap_or(&[])
}

fn norm_vec(f: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = f.tensor(name).with_context(|| format!("GGUF has no tensor {name:?}"))?;
    if t.elements() != len as u64 {
        bail!("tensor {name:?} has {} elements, expected {len}", t.elements());
    }
    f.dequant(t)
}
