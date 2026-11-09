//! Dense and Mixture-of-Experts (`MoE`) feed-forward networks and top-k gating.

use super::{cpu_threads, trace, trace_enabled};
use crate::config::Gating;
use crate::profile;
use allpaka_backend::{ops, quantmat::QuantMat};
use anyhow::Result;
use std::borrow::Cow;

/// The always-on shared expert of GLM-style MoE: a dense FFN of
/// `expert_ffn * n_shared` width, added to the routed output unweighted.
pub(crate) struct SharedFfn<'a> {
    pub(crate) gate: QuantMat<'a>,
    pub(crate) up: QuantMat<'a>,
    pub(crate) down: QuantMat<'a>,
    /// qwen35moe: a per-token sigmoid gate on the shared expert's output
    /// (`ffn_gate_inp_shexp`, hidden -> 1); None for GLM's unweighted share.
    pub(crate) gate_out: Option<QuantMat<'a>>,
}

impl SharedFfn<'_> {
    /// The shared FFN over `m` normed rows; borrow rules keep it out of
    /// [`Ffn`]'s match arms.
    pub(crate) fn forward_batch(&self, hs: &[f32], m: usize) -> Result<Vec<f32>> {
        let out =
            if let Some(mut outs) = QuantMat::ffn_many(&[(&self.gate, &self.up, &self.down, hs)]) {
                return self.gate_out(outs.pop().expect("one fused item"), hs, m);
            } else {
                let mut gate = self.gate.matmul(hs, m)?;
                let up = self.up.matmul(hs, m)?;
                ops::swiglu(&mut gate, &up);
                self.down.matmul(&gate, m)?
            };
        self.gate_out(out, hs, m)
    }

    fn gate_out(&self, out: Vec<f32>, hs: &[f32], m: usize) -> Result<Vec<f32>> {
        match &self.gate_out {
            None => Ok(out),
            Some(w) => {
                // The gate is one sigmoid scalar PER TOKEN (hidden -> 1); it
                // scales that token's whole shared-expert row.
                let g = w.matmul(hs, m)?;
                if trace_enabled() {
                    static L: std::sync::atomic::AtomicUsize =
                        std::sync::atomic::AtomicUsize::new(0);
                    let li = L.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    trace("shexp_gate", li, &g[..1]);
                    trace("shexp_out", li, &out[..out.len() / m]);
                }
                let row = out.len() / m.max(1);
                let mut out = out;
                for (tok, chunk) in out.chunks_mut(row).enumerate() {
                    let s = 1.0 / (1.0 + (-g[tok]).exp());
                    for o in chunk.iter_mut() {
                        *o *= s;
                    }
                }
                Ok(out)
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum Ffn<'a> {
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
    pub(crate) fn forward_batch(&self, hs: &[f32], m: usize, hidden: usize) -> Result<Vec<f32>> {
        match self {
            Ffn::Dense {
                w_gate,
                w_up,
                w_down,
            } => {
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
    pub(crate) fn moe_batch(
        &self,
        hs: &[f32],
        m: usize,
        hidden: usize,
        out: &mut [f32],
    ) -> Result<()> {
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
        let trace_on = trace_enabled();
        let li = if trace_on {
            static L: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            L.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        } else {
            0
        };
        if let Some(sh) = shared {
            let d = sh.forward_batch(hs, m)?;
            ops::add_assign_slice(out, &d);
            if trace_on {
                trace("moe_after_shared", li, &out[..hidden]);
            }
        }
        let router_span = profile::span(profile::Phase::Router);
        // Raw router logits; the selection bias is applied inside
        // route_gated AFTER gating (selection-only, llama.cpp semantics).
        let probs_all = router.matmul(hs, m)?;
        let n_expert = router.n_out;
        let k = (*n_used).min(n_expert);
        let rbias = router_bias.as_ref().map(|rb| rb.0.as_slice());
        if trace_on {
            trace("moe_logits", li, &probs_all[..n_expert]);
        }

        // Group the chunk's tokens by the expert they routed to: every token
        // that picked expert `e` shares one matmul, so `e`'s weights are
        // streamed once per chunk. This grouping is the whole reason MoE
        // prefill can be fast - per-(token, expert) matmuls would re-read the
        // weights for every token.
        // Per-token routing is independent: softmax + top-k spread over
        // persistent worker threads into a single flat buffer, zero per-token heap allocs.
        let mut routed_flat: Vec<(usize, f32)> = vec![(0, 0.0); m * k];
        if k > 0 && m > 0 {
            let threads = cpu_threads().min(m);
            let routed_addr = routed_flat.as_mut_ptr() as usize;
            let gating_val = *gating;
            let norm_val = *weights_norm;
            let scale_val = *weights_scale;
            allpaka_backend::thread_pool::par_for_chunks(m, threads, |_worker, start_t, end_t| {
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        (routed_addr as *mut (usize, f32)).add(start_t * k),
                        (end_t - start_t) * k,
                    )
                };
                let src = &probs_all[start_t * n_expert..end_t * n_expert];
                for (dst_row, row) in dst.chunks_mut(k).zip(src.chunks(n_expert)) {
                    route_gated_into_slice(row, k, gating_val, norm_val, scale_val, rbias, dst_row);
                }
            });
        }

        let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_expert];
        if k > 0 {
            for (i, r) in routed_flat.chunks(k).enumerate() {
                for &(e, weight) in r {
                    groups[e].push((i, weight));
                }
            }
        }
        if trace_on && m > 0 {
            let r0 = &routed_flat[..k];
            let ids: Vec<String> = r0.iter().map(|&(e, w)| format!("{e}:{w:.6}")).collect();
            let id_sum: usize = r0.iter().map(|&(e, _)| e).sum();
            let w_sum: f32 = r0.iter().map(|&(_, w)| w).sum();
            eprintln!(
                "trace moe_topk id_sum={id_sum} w_sum={w_sum:.6} [{}]",
                ids.join(" ")
            );
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
            let mut tok = Vec::with_capacity(m * k);
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
                gate_exps,
                up_exps,
                down_exps,
                n_expert,
                hidden,
                *expert_ffn,
                &table,
                hs,
                &tok,
                total_rows,
                None,
                None,
                None,
            )
            .map(|flat| (flat, table))
        };
        if let Some((flat, table)) = grouped {
            drop(ffn_span);
            let _s = profile::span(profile::Phase::FfnCombine);
            // Invert group->token into flat token->rows so the weighted adds
            // parallelise over tokens: each out row then has exactly one
            // writer, with zero per-token Vec allocations.
            let mut hits_flat = vec![(0usize, 0.0f32); m * k];
            let mut hits_counts = vec![0usize; m];
            for (gi, &e) in used.iter().enumerate() {
                let row0 = table[gi][1] as usize;
                for (row_idx, &(i, weight)) in groups[e].iter().enumerate() {
                    let c = hits_counts[i];
                    if c < k {
                        hits_flat[i * k + c] = (row0 + row_idx, weight);
                        hits_counts[i] = c + 1;
                    }
                }
            }
            let threads = cpu_threads().min(m.max(1));
            let out_addr = out.as_mut_ptr() as usize;
            let flat_ref = &flat;
            let hits_ref = &hits_flat;
            let counts_ref = &hits_counts;
            allpaka_backend::thread_pool::par_for_chunks(m, threads, |_worker, start_t, end_t| {
                let out_chunk = unsafe {
                    std::slice::from_raw_parts_mut(
                        (out_addr as *mut f32).add(start_t * hidden),
                        (end_t - start_t) * hidden,
                    )
                };
                for (local_t, out_row) in out_chunk.chunks_mut(hidden).enumerate() {
                    let t = start_t + local_t;
                    let count = counts_ref[t];
                    for &(row, weight) in &hits_ref[t * k..t * k + count] {
                        let down = &flat_ref[row * hidden..(row + 1) * hidden];
                        ops::add_scaled_slice(out_row, down, weight);
                    }
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
            .map(|gi| {
                (
                    &gate_mats[gi],
                    &up_mats[gi],
                    &down_mats[gi],
                    gathered[gi].as_slice(),
                )
            })
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
                ops::add_scaled_slice(out_row, down, weight);
            }
        }
        if trace_on {
            trace("moe_total", li, &out[..hidden]);
            for (gi, &e) in used.iter().enumerate() {
                if groups[e].iter().any(|&(i, _)| i == 0) {
                    trace(&format!("expert{e}_down"), li, &downs[gi][..hidden]);
                }
            }
        }
        Ok(())
    }

    /// The FFN half of a block: normed activations in, delta for the residual
    /// stream out.
    pub(crate) fn forward(&self, h: &[f32]) -> Result<Vec<f32>> {
        match self {
            Ffn::Dense {
                w_gate,
                w_up,
                w_down,
            } => {
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
                let fused_items: Vec<(&QuantMat, &QuantMat, &QuantMat, &[f32])> = (0..picked.len())
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
                    ops::add_scaled_slice(&mut out, &down, *weight);
                }
                // The shared expert runs for every token, unweighted.
                if let Some(sh) = shared {
                    let d = sh.forward_batch(h, 1)?;
                    ops::add_scaled_slice(&mut out, &d, 1.0);
                }
                Ok(out)
            }
        }
    }
}

/// Zero-heap-allocation top-k router writing directly into `dst` (length `k`).
/// Uses stack buffers for `n_expert <= 256` (all Qwen3-MoE, GLM, DeepSeek models).
#[inline]
pub(crate) fn route_gated_into_slice(
    logits: &[f32],
    k: usize,
    gating: Gating,
    norm: bool,
    scale: f32,
    bias: Option<&[f32]>,
    dst: &mut [(usize, f32)],
) {
    let n = logits.len();
    let k = k.min(n).min(dst.len());
    if k == 0 {
        return;
    }

    let mut scores_stack = [0.0f32; 256];
    let mut scores_vec;
    let scores: &mut [f32] = if n <= scores_stack.len() {
        scores_stack[..n].copy_from_slice(logits);
        &mut scores_stack[..n]
    } else {
        scores_vec = logits.to_vec();
        &mut scores_vec
    };

    match gating {
        Gating::Softmax => ops::softmax(scores),
        Gating::Sigmoid => ops::sigmoid(scores),
    }

    let mut sel_stack = [0.0f32; 256];
    let selection: Cow<'_, [f32]> = match bias {
        Some(b) if n <= sel_stack.len() => {
            for i in 0..n {
                sel_stack[i] = scores[i] + b[i];
            }
            Cow::Borrowed(&sel_stack[..n])
        }
        Some(b) => Cow::Owned(scores.iter().zip(b).map(|(s, b)| s + b).collect()),
        None => Cow::Borrowed(scores),
    };

    let mut order_buf = [0usize; 256];
    let mut order_vec;
    let order: &mut [usize] = if n <= order_buf.len() {
        for (i, slot) in order_buf[..n].iter_mut().enumerate() {
            *slot = i;
        }
        &mut order_buf[..n]
    } else {
        order_vec = (0..n).collect::<Vec<usize>>();
        &mut order_vec
    };

    if k < order.len() {
        order.select_nth_unstable_by(k - 1, |&a, &b| selection[b].total_cmp(&selection[a]));
    }
    let top_k = &mut order[..k];
    top_k.sort_by(|&a, &b| selection[b].total_cmp(&selection[a]));
    let total: f32 = top_k.iter().map(|&e| scores[e]).sum();
    for (slot, &e) in dst[..k].iter_mut().zip(top_k.iter()) {
        let w = if norm && total > 0.0 {
            scores[e] / total
        } else if norm {
            1.0 / k as f32
        } else {
            scores[e]
        };
        *slot = (e, w * scale);
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
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let mut out = vec![(0usize, 0.0f32); k];
    route_gated_into_slice(logits, k, gating, norm, scale, bias, &mut out);
    out
}

/// Pick the top `k` experts and renormalise their probabilities to sum to 1,
/// which is what Qwen3-MoE does (`norm_topk_prob`). Returned best-first.
///
/// `probs` must already be softmaxed; this is the pre-GLM contract kept for
/// the tests that pin it. New code takes raw logits and [`route_gated`].
pub fn route(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(probs.len());
    if k == 0 {
        return Vec::new();
    }
    let mut order_buf = [0usize; 256];
    let mut order_vec;
    let order: &mut [usize] = if probs.len() <= order_buf.len() {
        for (i, slot) in order_buf[..probs.len()].iter_mut().enumerate() {
            *slot = i;
        }
        &mut order_buf[..probs.len()]
    } else {
        order_vec = (0..probs.len()).collect::<Vec<usize>>();
        &mut order_vec
    };
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, |&a, &b| probs[b].total_cmp(&probs[a]));
    }
    let top_k = &mut order[..k];
    top_k.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let total: f32 = top_k.iter().map(|&e| probs[e]).sum();
    top_k
        .iter()
        .map(|&e| {
            (
                e,
                if total > 0.0 {
                    probs[e] / total
                } else {
                    1.0 / k as f32
                },
            )
        })
        .collect()
}
