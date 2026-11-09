//! Batched prompt prefill and speculative verification passes.

use super::{
    attn_zero_enabled, cpu_attn_enabled, moe::{route_gated_into_slice, Ffn},
    moe_zero_enabled, no_tokenbuf_enabled, prefuse_enabled, trace, Model, Session,
};
use crate::config::{Gating, RopeStyle};
use crate::profile;
use allpaka_backend::{activation::Activation, ops, quantmat::QuantMat};
use anyhow::{bail, Result};

impl<'a> Model<'a> {
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
        static CHUNK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CHUNK.get_or_init(|| {
            std::env::var("ALLPAKA_PREFILL_CHUNK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(512)
        })
    }

    pub fn forward_batch(&self, tokens: &[u32], s: &mut Session) -> Result<Vec<f32>> {
        let m = tokens.len();
        let hidden = self.config.hidden as usize;
        let mut xs = self.forward_batch_activations(tokens, s)?;
        let last_offset = (m - 1) * hidden;
        let last = &mut xs[last_offset..];
        ops::rmsnorm(last, &self.output_norm, self.config.rms_eps);
        // Debug hook: dump the post-norm hidden state (the head's input) to
        // compare against llama.cpp's /embedding for the same context.
        static DUMP_HIDDEN: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        if let Some(path) = DUMP_HIDDEN.get_or_init(|| std::env::var("ALLPAKA_DUMP_HIDDEN").ok()) {
            use std::io::Write as _;
            let body: Vec<String> = last.iter().map(|v| format!("{v:.8e}")).collect();
            std::fs::File::create(path)
                .and_then(|mut f| f.write_all(body.join(" ").as_bytes()))
                .unwrap_or_else(|e| eprintln!("ALLPAKA_DUMP_HIDDEN write failed: {e}"));
        }
        self.output_head()?.matmul_activation(&xs, last_offset, 1)
    }

    /// The shared body of the batch paths: every layer over every position,
    /// returning the residual stream `[m, hidden]` before the output norm.
    pub fn forward_batch_hidden(&self, tokens: &[u32], s: &mut Session) -> Result<Vec<f32>> {
        self.forward_batch_activations(tokens, s)
            .map(Activation::into_vec)
    }

    pub(crate) fn forward_batch_activations(
        &self,
        tokens: &[u32],
        s: &mut Session,
    ) -> Result<Activation> {
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
        let embd = self.embedding()?;
        let mut xs = vec![0.0f32; m * hidden];
        for (i, &t) in tokens.iter().enumerate() {
            embd.row_into(t as usize, &mut xs[i * hidden..(i + 1) * hidden])?;
        }
        s.ensure_rope_cache(&self.rope_inv_freq, base + m);
        let prefill_fused_ok = matches!(c.rope_style, RopeStyle::Neox)
            && !cpu_attn_enabled()
            && !no_tokenbuf_enabled();
        drop(embed_span);

        // The fully fused prefill: xs lives on the GPU for the whole chunk,
        // norms / residual adds / router / combine all run there, and the
        // CPU only routes tokens to experts between the two command buffers
        // of each layer. Any decline falls back to the per-layer path below
        // with the ORIGINAL xs - every cache write is recomputed
        // identically, so a mid-chunk decline costs time, not correctness.
        if prefill_fused_ok && prefuse_enabled() {
            if let Some(done) = self.forward_batch_fused(&xs, m, base, s) {
                s.kv.commit(base + m);
                return Ok(done);
            }
        }

        // One scratch for the normalised activations, reused by every layer:
        // the old per-layer `xs.clone()` cost an allocation and an extra
        // write pass over ~8 MB, twice per layer.
        let mut hs_scratch = vec![0f32; m * hidden];
        trace("input_embed", 0, &xs[..hidden]);
        for (li, layer) in self.layers.iter().enumerate() {
            // Attention, batched projections.
            let hs = {
                let _s = profile::span(profile::Phase::AttnNorm);
                ops::rmsnorm_rows_into(&mut hs_scratch, &xs, &layer.attn_norm, c.rms_eps);
                trace("attn_norm", li, &hs_scratch[..hidden]);
                &hs_scratch
            };

            // The gated-delta-net branch (linear-attention layers): the
            // recurrence is sequential by construction, the projections stay
            // batched over the whole chunk.
            if let Some(g) = &layer.gdn {
                let projected = self.gdn_forward(li, g, hs, m, s)?;
                {
                    let _s = profile::span(profile::Phase::AttnOut);
                    ops::add_assign_par(&mut xs, &projected);
                }
                trace("attn_residual", li, &xs[..hidden]);
                let hs = {
                    let _s = profile::span(profile::Phase::FfnNorm);
                    ops::rmsnorm_rows_into(&mut hs_scratch, &xs, &layer.ffn_norm, c.rms_eps);
                    trace("attn_post_norm", li, &hs_scratch[..hidden]);
                    &hs_scratch
                };
                let down = layer.ffn.forward_batch(hs, m, hidden)?;
                trace("ffn_out", li, &down[..hidden]);
                {
                    let _s = profile::span(profile::Phase::FfnCombine);
                    ops::add_assign_par(&mut xs, &down);
                }
                trace("l_out", li, &xs[..hidden]);
                continue;
            }
            let aw = layer
                .attn
                .as_ref()
                .expect("attention layer without weights");
            let rope_flat =
                &s.rope_cache[base * s.rope_cache_pairs..(base + m) * s.rope_cache_pairs];
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
                            &aw.wq,
                            &aw.wk,
                            &aw.wv,
                            &aw.wo,
                            hs,
                            m,
                            layer.q_norm.as_deref(),
                            layer.k_norm.as_deref(),
                            rope_flat,
                            c.rope_dim as usize,
                            aw.gate_in_q,
                            layer.bias.as_ref().map(|b| (b.q.1, b.k.1, b.v.1)),
                            c.rms_eps,
                            cache,
                            (k_off, v_off),
                            (
                                c.kv_dim(),
                                head_dim,
                                c.n_heads as usize,
                                c.n_kv_heads as usize,
                            ),
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
            let mut qkv =
                QuantMat::matmul_many(&[(&aw.wq, &hs[..]), (&aw.wk, &hs[..]), (&aw.wv, &hs[..])])?;
            drop(qkv_span);
            let v = qkv.pop().expect("v");
            let k = qkv.pop().expect("k");
            let q = qkv.pop().expect("q");

            let mut projected = self.run_cpu_attention_from_qkv(
                aw,
                layer.q_norm.as_deref(),
                layer.k_norm.as_deref(),
                layer.bias.as_ref(),
                q,
                k,
                v,
                m,
                base,
                rope_flat,
                &mut s.kv,
                li,
                false,
                true,
                Some(li),
            )?;
            if attn_zero_enabled() {
                projected.iter_mut().for_each(|v| *v = 0.0);
            }
            trace("attn_output", li, &projected[..hidden]);
            ops::add_assign_slice(&mut xs, &projected);
            trace("attn_residual", li, &xs[..hidden]);

            // Feed-forward, batched.
            let hs = {
                let _s = profile::span(profile::Phase::FfnNorm);
                ops::rmsnorm_rows_into(&mut hs_scratch, &xs, &layer.ffn_norm, c.rms_eps);
                trace("attn_post_norm", li, &hs_scratch[..hidden]);
                &hs_scratch
            };
            let down = if moe_zero_enabled() {
                vec![0f32; m * hidden]
            } else {
                layer.ffn.forward_batch(hs, m, hidden)?
            };
            trace("ffn_out", li, &down[..hidden]);
            {
                let _s = profile::span(profile::Phase::FfnCombine);
                ops::add_assign_par(&mut xs, &down);
            }
            trace("l_out", li, &xs[..hidden]);
        }
        s.kv.commit(base + m);
        Ok(xs.into())
    }

    /// The GPU-resident prefill chunk; `None` means "fall back to the
    /// per-layer path" and guarantees no partial state the fallback could
    /// trip over (the KV cache is rewritten identically from layer 0).
    fn forward_batch_fused(
        &self,
        xs: &[f32],
        m: usize,
        base: usize,
        s: &mut Session,
    ) -> Option<Activation> {
        let hidden = self.config.hidden as usize;
        if let Some(done) = allpaka_backend::gpu::prefill_replay(xs) {
            return Some(done.into());
        }
        allpaka_backend::gpu::prefill_begin(xs)?;
        // Deferred buffers of earlier layers may still be in flight when a
        // layer declines; wait them out before the CPU fallback runs.
        if self.forward_batch_fused_layers(xs, m, base, s).is_none() {
            allpaka_backend::gpu::prefill_abort();
            return None;
        }
        allpaka_backend::gpu::prefill_end_output(m * hidden)
    }

    /// The per-layer loop of the fused prefill chunk, split out so a
    /// mid-chunk decline can drain the deferred buffers first.
    fn forward_batch_fused_layers(
        &self,
        xs: &[f32],
        m: usize,
        base: usize,
        s: &mut Session,
    ) -> Option<()> {
        let c = &self.config;
        let hidden = c.hidden as usize;
        let head_dim = c.head_dim as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope_flat =
            &s.rope_cache[base * s.rope_cache_pairs..(base + m) * s.rope_cache_pairs];
        let mut picked_buf = [(0usize, 0.0f32); 256];
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
            };
            let (cache, k_off, v_off) = s.kv.gpu_view(li)?;
            let fusion = Some(allpaka_backend::gpu::PrefillFusion {
                attn_norm: layer.attn_norm_raw,
                ffn_norm: layer.ffn_norm_raw,
                router: router_raw,
                n_expert,
            });
            // The gated-delta-net branch (qwen35moe linear-attention layers):
            // its own fused block, same contract as the attention one.
            let logits = if let Some(g) = &layer.gdn {
                let ssm = s.ssm.as_mut().expect("gdn layer without ssm cache");
                let (conv_off, state_off) = (ssm.conv_off(li), ssm.state_off(li));
                let (region, slots) = ssm.regions_for_gpu();
                let region = region?;
                let sc = c.ssm.as_ref().expect("gdn arch");
                let _s = profile::span(profile::Phase::Attend);
                QuantMat::prefill_gdn_block(
                    &g.wqkv,
                    &g.zgate,
                    &g.alpha,
                    &g.beta,
                    g.conv1d_raw,
                    &g.a,
                    &g.dt,
                    &g.ssm_norm,
                    &g.ssm_out,
                    (
                        sc.n_group as usize,
                        sc.dt_rank as usize,
                        sc.d_state as usize,
                        sc.d_conv as usize,
                    ),
                    hidden,
                    m,
                    c.rms_eps,
                    region,
                    slots,
                    (conv_off, state_off),
                    fusion,
                )?
            } else {
                let aw = layer.attn.as_ref().expect("attn weights");
                let _s = profile::span(profile::Phase::Attend);
                QuantMat::prefill_attn_block(
                    &aw.wq,
                    &aw.wk,
                    &aw.wv,
                    &aw.wo,
                    xs,
                    m,
                    layer.q_norm.as_deref(),
                    layer.k_norm.as_deref(),
                    rope_flat,
                    c.rope_dim as usize,
                    aw.gate_in_q,
                    layer.bias.as_ref().map(|b| (b.q.1, b.k.1, b.v.1)),
                    c.rms_eps,
                    cache,
                    (k_off, v_off),
                    (
                        c.kv_dim(),
                        head_dim,
                        c.n_heads as usize,
                        c.n_kv_heads as usize,
                    ),
                    base,
                    scale,
                    fusion,
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
                    gate_exps,
                    up_exps,
                    down_exps,
                    shared,
                    expert_ffn,
                    n_used,
                    gating,
                    weights_norm,
                    weights_scale,
                    router_bias,
                    ..
                } => {
                    gate_m = gate_exps;
                    up_m = up_exps;
                    down_m = down_exps;
                    ffn_w = *expert_ffn;
                    sh_m = shared
                        .as_ref()
                        .map(|sh| (&sh.gate, &sh.up, &sh.down, sh.gate_out.as_ref()));
                    if logits.is_empty() {
                        // GPU routing: the attention block left the logits in
                        // y_arena; route_pick/scan/scatter build everything
                        // below on-device at the head of the FFN buffer,
                        // including Softmax with or without selection bias or
                        // winner renormalisation.
                        let gpu_ok = *n_used <= 8;
                        if !gpu_ok {
                            return None;
                        }
                        gpu_route = Some(allpaka_backend::gpu::GroupedRoute {
                            n_used: *n_used,
                            norm: *weights_norm,
                            scale: *weights_scale,
                            bias: router_bias.as_ref().map(|b| b.0.as_slice()),
                            sigmoid: *gating == Gating::Sigmoid,
                        });
                        (Vec::new(), Vec::new(), m * *n_used, Vec::new())
                    } else {
                        // CPU routing can't evaluate qwen35moe's shared-expert
                        // gate (the normed activations live on the GPU); the
                        // per-layer CPU path takes the layer instead.
                        if shared.as_ref().is_some_and(|sh| sh.gate_out.is_some()) {
                            return None;
                        }
                        let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_expert];
                        let k = (*n_used).min(n_expert).min(picked_buf.len());
                        let rbias = router_bias.as_ref().map(|b| b.0.as_slice());
                        for i in 0..m {
                            let row = &logits[i * n_expert..(i + 1) * n_expert];
                            route_gated_into_slice(
                                row,
                                k,
                                *gating,
                                *weights_norm,
                                *weights_scale,
                                rbias,
                                &mut picked_buf[..k],
                            );
                            for &(e, weight) in &picked_buf[..k] {
                                groups[e].push((i, weight));
                            }
                        }
                        let used: Vec<usize> =
                            (0..n_expert).filter(|&e| !groups[e].is_empty()).collect();
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
                Ffn::Dense {
                    w_gate,
                    w_up,
                    w_down,
                    ..
                } => {
                    gate_m = w_gate;
                    up_m = w_up;
                    down_m = w_down;
                    ffn_w = w_gate.n_out;
                    sh_m = None;
                    (
                        vec![[0u32, 0u32, m as u32]],
                        (0..m as u32).collect::<Vec<_>>(),
                        m,
                        Vec::new(),
                    )
                }
            };
            let mut tok_off = Vec::with_capacity(m + 1);
            let mut hit_row = Vec::with_capacity(total_rows + m);
            let mut hit_w = Vec::with_capacity(total_rows + m);
            tok_off.push(0u32);
            if hits_per_token.is_empty() {
                // Dense identity: one hit per token, no per-row Vec.
                hit_row.extend(0..total_rows as u32);
                hit_w.resize(total_rows, 1.0);
                tok_off.extend(1..=total_rows as u32);
            } else {
                for h in &hits_per_token {
                    for &(r, w) in h {
                        hit_row.push(r);
                        hit_w.push(w);
                    }
                    tok_off.push(hit_row.len() as u32);
                }
            }
            drop(router_span);

            let _f = profile::span(profile::Phase::Ffn);
            // The FFN combine folds in the NEXT layer's attention norm
            // (combine_rows_resnorm); the last layer leaves the output norm
            // to the CPU.
            let next_norm = self
                .layers
                .get(li + 1)
                .map(|l| (l.attn_norm_raw, c.rms_eps))
                .filter(|(raw, _)| !raw.is_empty());
            QuantMat::ffn_grouped(
                gate_m,
                up_m,
                down_m,
                n_expert.max(1),
                hidden,
                ffn_w,
                &table,
                &[],
                &tok,
                total_rows,
                Some(allpaka_backend::gpu::GroupedCombine {
                    tok_off: &tok_off,
                    hit_row: &hit_row,
                    hit_w: &hit_w,
                    m,
                    next_norm,
                }),
                sh_m,
                gpu_route,
            )?;
        }
        Some(())
    }

    /// [`forward_batch_full`] plus every row's output-normed hidden (the
    /// LM head's input): MTP speculation needs the trunk hidden at the
    /// accepted row to seed the next draft round (llama's t_h_nextn).
    pub fn forward_batch_full_hn(
        &self,
        tokens: &[u32],
        s: &mut Session,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let m = tokens.len();
        let hidden = self.config.hidden as usize;
        let mut xs = self.forward_batch_activations(tokens, s)?;
        for row in xs.chunks_mut(hidden) {
            ops::rmsnorm(row, &self.output_norm, self.config.rms_eps);
        }
        let logits = self.output_head()?.matmul_activation(&xs, 0, m)?;
        Ok((logits, xs.into_vec()))
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
        let mut xs = self.forward_batch_activations(tokens, s)?;
        for row in xs.chunks_mut(hidden) {
            ops::rmsnorm(row, &self.output_norm, self.config.rms_eps);
        }
        self.output_head()?.matmul_activation(&xs, 0, m)
    }
}
