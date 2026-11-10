//! Single-token decode, multi-token GPU verification, and greedy chain execution.

use super::{
    attn_zero_enabled, cpu_attn_enabled, greedy_chain_enabled, moe::Ffn, moe_zero_enabled,
    no_argmax_enabled, no_tokenbuf_enabled, skip_host_rope_enabled, tokenbuf_debug_enabled, Layer,
    Model, Session,
};
use crate::config::{Config, Gating, RopeStyle};
use crate::profile;
use allpaka_backend::{ops, quantmat::QuantMat};
use anyhow::{bail, Result};

impl<'a> Model<'a> {
    /// Build the whole-token GPU request and run it; Ok(None) means the GPU
    /// declined and the caller should take the per-layer path. `argmax` asks
    /// for the greedy winner only (TokenOut::Argmax) instead of the full
    /// vocabulary logits.
    pub(crate) fn forward_token_gpu(
        &self,
        token: u32,
        s: &mut Session,
        pos: usize,
        argmax: bool,
    ) -> Result<Option<allpaka_backend::gpu::TokenOut>> {
        use allpaka_backend::gpu::TokenReq;
        let c = &self.config;
        let dbg = tokenbuf_debug_enabled();
        if self.output_norm_raw.is_empty() {
            if dbg {
                eprintln!("tokenbuf declined: output norm not raw F32");
            }
            return Ok(None);
        }
        // Keep host rope table by default; skip only when explicitly opted in.
        // Device freq rope still used inside CUDA when d_rope_freq is set.
        let skip_host_rope =
            allpaka_backend::gpu::has_device_rope_freq() && skip_host_rope_enabled();
        if !skip_host_rope {
            s.ensure_rope_cache(&self.rope_inv_freq, pos + 1);
        }
        let Some(layers) = self.token_layers(s)? else {
            if dbg {
                eprintln!("tokenbuf declined: token_layers");
            }
            return Ok(None);
        };
        let cache_capacity = s.kv.capacity();
        let (cache, _, _) = match s.kv.gpu_view_checked(0) {
            Ok(view) => view,
            Err(reason) => {
                if dbg {
                    eprintln!("tokenbuf declined: stage=kv-cache reason={reason}");
                }
                return Ok(None);
            }
        };
        // The GDN arch decodes on the GPU when the SSM region wraps; its
        // attention layers take the head_dim-256 kernels.
        let ssm_region = match s.ssm.as_mut() {
            Some(ssm) => match ssm.gpu_view() {
                Some(r) => Some(r),
                None => {
                    if dbg {
                        eprintln!("tokenbuf declined: no gpu ssm region");
                    }
                    return Ok(None);
                }
            },
            None => None,
        };
        let rope_table: &[[f32; 2]] = if skip_host_rope {
            &[]
        } else {
            &s.rope_cache[pos * s.rope_cache_pairs..(pos + 1) * s.rope_cache_pairs]
        };
        let embd = self.embedding()?;
        let (embd_ty, embd_bytes, embd_n_in) = embd.embd_desc();
        let use_gpu_embed = allpaka_backend::gpu::can_embed_on_gpu(embd_ty, embd_bytes, embd_n_in);
        let x_host = if use_gpu_embed {
            None
        } else {
            Some(embd.row(token as usize)?)
        };
        let x: &[f32] = x_host.as_deref().unwrap_or(&[]);
        let request = TokenReq {
            x,
            m: 1,
            layers: &layers,
            cache,
            cache_capacity,
            ssm: ssm_region,
            ssm_slots: None,
            kv_dim: c.kv_dim(),
            head_dim: c.head_dim as usize,
            n_heads: c.n_heads as usize,
            n_kv_heads: c.n_kv_heads as usize,
            pos,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
            rope: rope_table,
            rot_dim: c.rope_dim as usize,
            rope_norm: matches!(c.rope_style, RopeStyle::Norm),
            eps: c.rms_eps,
            output_norm: self.output_norm_raw,
            output: self.output_head()?.desc(),
            argmax,
            embd: Some((embd_ty, embd_bytes, embd_n_in)),
            token_id: Some(token),
        };
        match allpaka_backend::gpu::decode_token_checked(&request) {
            Ok(out) => Ok(Some(out)),
            Err(reason) => {
                if dbg {
                    eprintln!("tokenbuf declined: {reason}");
                }
                Ok(None)
            }
        }
    }

    /// The whole-token GPU request for a batch of `m` consecutive tokens
    /// (the speculative verify): one command buffer, per-row argmax and the
    /// output-normed hidden rows. Ok(None) declines to the batch path.
    pub fn verify_tokens(
        &self,
        tokens: &[u32],
        s: &mut Session,
    ) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        use allpaka_backend::gpu::{TokenOut, TokenReq};
        let dbg = tokenbuf_debug_enabled();
        let c = &self.config;
        let m = tokens.len();
        if m == 0 || m > 8 || no_tokenbuf_enabled() {
            return Ok(None);
        }
        if self.output_norm_raw.is_empty() {
            return Ok(None);
        }
        let pos = s.pos();
        s.ensure_rope_cache(&self.rope_inv_freq, pos + m);
        let Some(layers) = self.token_layers(s)? else {
            if dbg {
                eprintln!("tokenbuf declined: verify token_layers");
            }
            return Ok(None);
        };
        let cache_capacity = s.kv.capacity();
        let (cache, _, _) = match s.kv.gpu_view_checked(0) {
            Ok(view) => view,
            Err(reason) => {
                if dbg {
                    eprintln!("tokenbuf declined: stage=verify-kv-cache reason={reason}");
                }
                return Ok(None);
            }
        };
        let (ssm_region, slots) = match s.ssm.as_mut() {
            Some(ssm) => ssm.regions_for_gpu(),
            None => (None, None),
        };
        let rope_table = &s.rope_cache[pos * s.rope_cache_pairs..(pos + m) * s.rope_cache_pairs];
        let hidden_dim = c.hidden as usize;
        let mut x = vec![0.0f32; m * hidden_dim];
        let embd = self.embedding()?;
        for (ti, &t) in tokens.iter().enumerate() {
            if t >= c.vocab {
                bail!("token {t} out of vocabulary {}", c.vocab);
            }
            embd.row_into(t as usize, &mut x[ti * hidden_dim..(ti + 1) * hidden_dim])?;
        }
        let request = TokenReq {
            x: &x,
            m,
            layers: &layers,
            cache,
            cache_capacity,
            ssm: ssm_region,
            ssm_slots: slots,
            kv_dim: c.kv_dim(),
            head_dim: c.head_dim as usize,
            n_heads: c.n_heads as usize,
            n_kv_heads: c.n_kv_heads as usize,
            pos,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
            rope: rope_table,
            rot_dim: c.rope_dim as usize,
            rope_norm: matches!(c.rope_style, RopeStyle::Norm),
            eps: c.rms_eps,
            output_norm: self.output_norm_raw,
            output: self.output_head()?.desc(),
            argmax: true,
            embd: None,
            token_id: None,
        };
        match allpaka_backend::gpu::decode_token_checked(&request) {
            Ok(TokenOut::Rows { argmax, hidden }) => {
                s.kv.commit(pos + m);
                Ok(Some((argmax, hidden)))
            }
            Ok(_) => Ok(None),
            Err(reason) => {
                if dbg {
                    eprintln!("tokenbuf declined: verify {reason}");
                }
                Ok(None)
            }
        }
    }

    /// Build the static `TokenLayer` weight template once at load time.
    /// Session-independent SSM slot offsets are baked in directly; only `k_off`/`v_off`
    /// depend on the runtime `Session` capacity and are stamped in [`Self::token_layers`].
    pub(crate) fn build_token_layer_template(
        c: &Config,
        model_layers: &[Layer<'a>],
    ) -> Result<Vec<allpaka_backend::gpu::TokenLayer<'a>>, &'static str> {
        use allpaka_backend::gpu::{TokenFfn, TokenGdn, TokenLayer};
        let n_gdn = model_layers.iter().filter(|l| l.gdn.is_some()).count();
        let (conv_stride, state_stride, conv_elems) = match &c.ssm {
            Some(sc) => {
                let channels = (sc.d_inner + 2 * sc.n_group * sc.d_state) as usize;
                let conv_s = (sc.d_conv as usize - 1) * channels;
                let state_s = (sc.dt_rank * sc.d_state * sc.d_state) as usize;
                (conv_s, state_s, n_gdn * conv_s)
            }
            None => (0, 0, 0),
        };
        let mut gdn_slot = 0usize;
        let mut out = Vec::with_capacity(model_layers.len());
        for layer in model_layers {
            if layer.attn_norm_raw.is_empty() || layer.ffn_norm_raw.is_empty() {
                return Err("layer norm not raw F32");
            }
            let ffn = match &layer.ffn {
                Ffn::Dense {
                    w_gate,
                    w_up,
                    w_down,
                } => TokenFfn::Dense {
                    gate: w_gate.desc(),
                    up: w_up.desc(),
                    down: w_down.desc(),
                },
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
                    n_used,
                    ..
                } => {
                    let sigmoid = matches!(gating, Gating::Sigmoid);
                    // The GPU top-k kernels only implement renormalized
                    // weights with scale 1; anything else stays on the CPU.
                    if sigmoid && (!*weights_norm || *weights_scale != 1.0) {
                        return Err("sigmoid gating with norm/scale");
                    }
                    // qwen35moe: the shared expert's sigmoid gate
                    // (ffn_gate_inp_shexp), F32 [hidden] in the mmap.
                    let shared_gate =
                        shared
                            .as_ref()
                            .and_then(|sh| sh.gate_out.as_ref())
                            .map(|g| {
                                let (t, b) = g.raw();
                                debug_assert_eq!(t, allpaka_gguf::GgmlType::F32);
                                b
                            });
                    let shared = shared
                        .as_ref()
                        .map(|sh| [sh.gate.desc(), sh.up.desc(), sh.down.desc()]);
                    TokenFfn::Moe {
                        router: router.desc(),
                        router_bias: router_bias.as_ref().map(|b| b.1),
                        gate: gate_exps.raw(),
                        up: up_exps.raw(),
                        down: down_exps.raw(),
                        expert_ffn: *expert_ffn,
                        n_used: *n_used,
                        sigmoid,
                        shared,
                        shared_gate,
                    }
                }
            };
            let gdn = layer.gdn.as_ref().map(|g| {
                let sc = c.ssm.as_ref().expect("gdn arch");
                let (at, ab) = g.alpha.raw();
                let (bt, bb) = g.beta.raw();
                debug_assert_eq!(at, allpaka_gguf::GgmlType::F32);
                debug_assert_eq!(bt, allpaka_gguf::GgmlType::F32);
                let slot = gdn_slot;
                gdn_slot += 1;
                TokenGdn {
                    wqkv: g.wqkv.desc(),
                    zgate: g.zgate.desc(),
                    alpha: ab,
                    beta: bb,
                    conv1d: g.conv1d_raw,
                    a: g.a,
                    dt: g.dt,
                    ssm_norm: g.ssm_norm,
                    ssm_out: g.ssm_out.desc(),
                    heads_k: sc.n_group as usize,
                    heads_v: sc.dt_rank as usize,
                    d: sc.d_state as usize,
                    d_conv: sc.d_conv as usize,
                    conv_off: slot * conv_stride,
                    state_off: conv_elems + slot * state_stride,
                }
            });
            let (wq, wk, wv, wo) = match &layer.attn {
                Some(aw) => (aw.wq.desc(), aw.wk.desc(), aw.wv.desc(), aw.wo.desc()),
                None => (
                    (allpaka_gguf::GgmlType::F32, &[][..], 0),
                    (allpaka_gguf::GgmlType::F32, &[][..], 0),
                    (allpaka_gguf::GgmlType::F32, &[][..], 0),
                    (allpaka_gguf::GgmlType::F32, &[][..], 0),
                ),
            };
            out.push(TokenLayer {
                attn_norm: layer.attn_norm_raw,
                ffn_norm: layer.ffn_norm_raw,
                wq,
                wk,
                wv,
                wo,
                q_norm: layer.q_norm,
                k_norm: layer.k_norm,
                gate_in_q: layer.attn.as_ref().is_some_and(|aw| aw.gate_in_q),
                q_bias: layer.bias.as_ref().map(|b| b.q.1),
                k_bias: layer.bias.as_ref().map(|b| b.k.1),
                v_bias: layer.bias.as_ref().map(|b| b.v.1),
                k_off: 0,
                v_off: 0,
                gdn,
                ffn,
            });
        }
        Ok(out)
    }

    /// The decode layers of the whole-token GPU paths, shared by
    /// `forward_token_gpu`, `verify_tokens`, and `try_forward_greedy_chain`.
    /// Clones the pre-validated static template and stamps per-session KV offsets.
    pub(crate) fn token_layers(
        &self,
        s: &mut Session,
    ) -> Result<Option<Vec<allpaka_backend::gpu::TokenLayer<'a>>>> {
        let dbg = tokenbuf_debug_enabled();
        let template = match &self.token_layer_template {
            Ok(t) => t,
            Err(reason) => {
                if dbg {
                    eprintln!("tokenbuf declined: {reason}");
                }
                return Ok(None);
            }
        };
        if !template.is_empty() {
            if let Err(reason) = s.kv.gpu_view_checked(0) {
                if dbg {
                    eprintln!("tokenbuf declined: stage=token-layers reason={reason}");
                }
                return Ok(None);
            }
        }
        let mut layers = template.clone();
        for (li, tl) in layers.iter_mut().enumerate() {
            let (k_off, v_off) = s.kv.layer_offsets(li);
            tl.k_off = k_off;
            tl.v_off = v_off;
            if let (Some(gdn), Some(ssm)) = (&tl.gdn, s.ssm.as_ref()) {
                debug_assert_eq!(gdn.conv_off, ssm.conv_off(li));
                debug_assert_eq!(gdn.state_off, ssm.state_off(li));
            }
        }
        Ok(Some(layers))
    }

    /// Consume one token, return logits over the vocabulary.
    pub fn forward(&self, token: u32, s: &mut Session) -> Result<Vec<f32>> {
        let c = &self.config;
        if token >= c.vocab {
            bail!("token {token} out of vocabulary {}", c.vocab);
        }
        let pos = s.pos();

        // The whole token as ONE GPU command buffer: every layer's attention
        // and FFN (routing included, on the GPU), through to the vocabulary
        // logits. Declines - a norm that is not F32, no device, kill switches -
        // fall through to the per-layer path below, which remains the reference
        // this path is tested against.
        if pos < s.kv.capacity() && !cpu_attn_enabled() && !no_tokenbuf_enabled() {
            let dbg = tokenbuf_debug_enabled();
            if let Some(allpaka_backend::gpu::TokenOut::Logits(logits)) =
                self.forward_token_gpu(token, s, pos, false)?
            {
                let _ = profile::span(profile::Phase::Attend);
                s.kv.advance();
                return Ok(logits);
            } else if dbg {
                eprintln!("tokenbuf declined: forward_token_gpu for token {token} pos {pos}");
            }
        }

        let hidden = self.embed_token(token)?;
        let hidden = self.forward_layers(&hidden, s, 0, self.layers.len())?;
        self.finish_hidden(&hidden)
    }

    /// Embed a token at the head of a distributed pipeline.
    pub fn embed_token(&self, token: u32) -> Result<Vec<f32>> {
        if token >= self.config.vocab {
            bail!("token {token} out of vocabulary {}", self.config.vocab);
        }
        self.embedding()?.row(token as usize)
    }

    /// Execute a contiguous stage. Each stage owns a separate session and
    /// advances it once per token. Layer indices refer to the original model.
    pub fn forward_layers(
        &self,
        hidden: &[f32],
        s: &mut Session,
        first_layer: usize,
        end_layer: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        if first_layer >= end_layer
            || first_layer < self.layer_start
            || end_layer > self.layer_start + self.layers.len()
        {
            bail!("invalid layer range {first_layer}..{end_layer}");
        }
        if hidden.len() != c.hidden as usize || hidden.iter().any(|v| !v.is_finite()) {
            bail!("invalid pipeline activation");
        }
        if s.pos() >= s.capacity() {
            bail!("pipeline session context exhausted");
        }
        let head_dim = c.head_dim as usize;
        let pos = s.pos();
        let mut x = hidden.to_vec();
        let mut h = vec![0.0f32; c.hidden as usize];
        s.ensure_rope_cache(&self.rope_inv_freq, pos + 1);
        let cpu_attn = cpu_attn_enabled();

        for (li, layer) in self
            .layers
            .iter()
            .enumerate()
            .take(end_layer - self.layer_start)
            .skip(first_layer - self.layer_start)
        {
            // Attention.
            {
                let _s = profile::span(profile::Phase::AttnNorm);
                ops::rmsnorm_into(&mut h, &x, &layer.attn_norm, c.rms_eps);
            }

            // The gated-delta-net branch (linear-attention layers).
            if let Some(g) = &layer.gdn {
                let projected = self.gdn_forward(li, g, &h, 1, s)?;
                ops::add_assign_slice(&mut x, &projected);
                run_layer_ffn(layer, &mut x, &mut h, c.rms_eps)?;
                continue;
            }
            let aw = layer
                .attn
                .as_ref()
                .expect("attention layer without weights");
            let rope_pairs =
                &s.rope_cache[pos * s.rope_cache_pairs..(pos + 1) * s.rope_cache_pairs];

            // The whole attention half - qkv, norms, rope, cache store,
            // attention, output projection - as one GPU command buffer. By
            // the GPU's own clock, five short buffers cost more in driver
            // scheduling than in execution, so the win here is the merge
            // itself. Declines (no GPU) fall through to the step-by-step path below.
            if !cpu_attn
                && pos < s.kv.capacity()
                && allpaka_backend::gpu::decode_attention_capacity_safe(s.kv.capacity())
            {
                let scale = 1.0 / (head_dim as f32).sqrt();
                let blocked = {
                    let _s = profile::span(profile::Phase::Attend);
                    s.kv.gpu_view(li).and_then(|(cache, k_off, v_off)| {
                        QuantMat::attn_block(
                            &aw.wq,
                            &aw.wk,
                            &aw.wv,
                            &aw.wo,
                            &h,
                            layer.q_norm,
                            layer.k_norm,
                            rope_pairs,
                            matches!(c.rope_style, RopeStyle::Norm),
                            c.rms_eps,
                            cache,
                            (k_off, v_off),
                            (
                                c.kv_dim(),
                                head_dim,
                                c.n_heads as usize,
                                c.n_kv_heads as usize,
                            ),
                            pos,
                            scale,
                        )
                    })
                };
                if let Some(projected) = blocked {
                    ops::add_assign_slice(&mut x, &projected);
                    run_layer_ffn(layer, &mut x, &mut h, c.rms_eps)?;
                    continue;
                }
            }
            // q, k and v are independent: one batch, one GPU wait.
            let mut qkv = {
                let _s = profile::span(profile::Phase::Qkv);
                QuantMat::matmul_many(&[
                    (&aw.wq, h.as_slice()),
                    (&aw.wk, h.as_slice()),
                    (&aw.wv, h.as_slice()),
                ])?
            };
            let v = qkv.pop().expect("v");
            let k = qkv.pop().expect("k");
            let q = qkv.pop().expect("q");

            let mut projected = self.run_cpu_attention_from_qkv(
                aw,
                layer.q_norm,
                layer.k_norm,
                layer.bias.as_ref(),
                q,
                k,
                v,
                1,
                pos,
                rope_pairs,
                &mut s.kv,
                li,
                false,
                true,
                None,
            )?;
            if attn_zero_enabled() {
                projected.iter_mut().for_each(|v| *v = 0.0);
            }
            ops::add_assign_slice(&mut x, &projected);

            // Feed-forward, dense or routed.
            run_layer_ffn(layer, &mut x, &mut h, c.rms_eps)?;
        }
        s.kv.advance();
        Ok(x)
    }

    /// Apply the output head only at the last pipeline stage.
    pub fn finish_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.config.hidden as usize || hidden.iter().any(|v| !v.is_finite()) {
            bail!("invalid pipeline output activation");
        }
        let output = self.output_head()?;
        let mut x = hidden.to_vec();
        ops::rmsnorm(&mut x, &self.output_norm, self.config.rms_eps);
        output.matmul(&x, 1)
    }

    /// Consume one token, return the greedy (argmax) next token. The
    /// tokenbuf path runs the argmax on the GPU and reads back one word
    /// instead of the whole vocabulary; ties resolve to the higher index,
    /// matching the CPU `max_by` below. `ALLPAKA_NO_ARGMAX` forces the
    /// logits path.
    pub fn forward_greedy(&self, token: u32, s: &mut Session) -> Result<u32> {
        let c = &self.config;
        let pos = s.pos();
        if token < c.vocab
            && pos < s.kv.capacity()
            && !cpu_attn_enabled()
            && !no_tokenbuf_enabled()
            && !no_argmax_enabled()
        {
            if let Some(allpaka_backend::gpu::TokenOut::Argmax(next)) =
                self.forward_token_gpu(token, s, pos, true)?
            {
                let _ = profile::span(profile::Phase::Attend);
                s.kv.advance();
                return Ok(next);
            }
        }
        let logits = self.forward(token, s)?;
        Ok((0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap_or(0) as u32)
    }

    /// Greedy decode of `n` tokens starting from `token`. Prefers an on-device
    /// CUDA chain (one sync) when graphs + embed kernels are available.
    pub fn forward_greedy_n(&self, token: u32, s: &mut Session, n: usize) -> Result<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        if greedy_chain_enabled() && n > 1 {
            if let Some(out) = self.try_forward_greedy_chain(token, s, n)? {
                return Ok(out);
            }
        }
        let mut out = Vec::with_capacity(n);
        let mut t = token;
        for _ in 0..n {
            t = self.forward_greedy(t, s)?;
            out.push(t);
        }
        Ok(out)
    }

    fn try_forward_greedy_chain(
        &self,
        token: u32,
        s: &mut Session,
        n: usize,
    ) -> Result<Option<Vec<u32>>> {
        let c = &self.config;
        let pos = s.pos();
        if token >= c.vocab
            || pos + n > s.kv.capacity()
            || cpu_attn_enabled()
            || no_tokenbuf_enabled()
            || no_argmax_enabled()
        {
            return Ok(None);
        }
        // First token: normal GPU greedy (captures/replays graph, syncs).
        let Some(allpaka_backend::gpu::TokenOut::Argmax(first)) =
            self.forward_token_gpu(token, s, pos, true)?
        else {
            return Ok(None);
        };
        s.kv.advance();
        if n == 1 {
            return Ok(Some(vec![first]));
        }
        // Remaining tokens stay on GPU: embed(d_argmax) + d_pos++ + graph.
        let dbg = tokenbuf_debug_enabled();
        if self.output_norm_raw.is_empty() {
            return Ok(None);
        }
        s.ensure_rope_cache(&self.rope_inv_freq, pos + n);
        let Some(layers) = self.token_layers(s)? else {
            return Ok(None);
        };
        let cache_capacity = s.kv.capacity();
        let (cache, _, _) = match s.kv.gpu_view_checked(0) {
            Ok(view) => view,
            Err(_) => return Ok(None),
        };
        let rope_table =
            &s.rope_cache[(pos + 1) * s.rope_cache_pairs..(pos + n) * s.rope_cache_pairs];
        let empty_x: &[f32] = &[];
        let request = allpaka_backend::gpu::TokenReq {
            x: empty_x,
            m: 1,
            layers: &layers,
            cache,
            cache_capacity,
            ssm: None,
            ssm_slots: None,
            kv_dim: c.kv_dim(),
            head_dim: c.head_dim as usize,
            n_heads: c.n_heads as usize,
            n_kv_heads: c.n_kv_heads as usize,
            pos: pos + 1,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
            rope: rope_table,
            rot_dim: c.rope_dim as usize,
            rope_norm: matches!(c.rope_style, RopeStyle::Norm),
            eps: c.rms_eps,
            output_norm: self.output_norm_raw,
            output: self.output_head()?.desc(),
            argmax: true,
            embd: Some(self.embedding()?.embd_desc()),
            token_id: Some(first),
        };
        match allpaka_backend::gpu::decode_greedy_continue(&request, n - 1) {
            Some(rest) => {
                for _ in 0..(n - 1) {
                    s.kv.advance();
                }
                let mut out = Vec::with_capacity(n);
                out.push(first);
                out.extend(rest);
                Ok(Some(out))
            }
            None => {
                if dbg {
                    eprintln!("tokenbuf: greedy chain continue declined; falling back");
                }
                // First token already committed; finish the rest one-by-one.
                let mut out = Vec::with_capacity(n);
                out.push(first);
                let mut t = first;
                for _ in 1..n {
                    t = self.forward_greedy(t, s)?;
                    out.push(t);
                }
                Ok(Some(out))
            }
        }
    }
}

#[inline]
fn run_layer_ffn(layer: &Layer<'_>, x: &mut [f32], h: &mut [f32], rms_eps: f32) -> Result<()> {
    {
        let _s = profile::span(profile::Phase::FfnNorm);
        ops::rmsnorm_into(h, x, &layer.ffn_norm, rms_eps);
    }
    if moe_zero_enabled() {
        return Ok(());
    }
    let down = layer.ffn.forward(h)?;
    let _s = profile::span(profile::Phase::FfnCombine);
    ops::add_assign_slice(x, &down);
    Ok(())
}
