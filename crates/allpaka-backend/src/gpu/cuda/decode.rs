//! Whole-token decode entry points, the greedy on-device chain, the GDN
//! layer, and the unfused per-stage fallback path.

use super::*;

pub fn decode_token_checked(req: &TokenReq) -> Result<TokenOut, DecodeDecline> {
    DECODE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    match decode_token(req) {
        Some(out) => {
            DECODE_SUCCESSES.fetch_add(1, Ordering::Relaxed);
            Ok(out)
        }
        None => {
            DECODE_DECLINES.fetch_add(1, Ordering::Relaxed);
            Err(DecodeDecline {
                stage: "backend-decode",
                reason: "request shape, tensor format, or CUDA path unsupported",
            })
        }
    }
}

pub fn decode_token(req: &TokenReq) -> Option<TokenOut> {
    if !is_attached() {
        return None;
    }
    let m = req.m;
    if m == 0 || m > 8 {
        return None;
    }
    if m == 1 && !decode_attention_capacity_safe(req.cache_capacity) {
        return None;
    }
    let gpu_embed = m == 1 && req.x.is_empty() && req.token_id.is_some() && req.embd.is_some();
    let hidden = if gpu_embed {
        req.embd?.2
    } else {
        if req.x.len() % m != 0 {
            return None;
        }
        req.x.len() / m
    };
    if hidden == 0 {
        return None;
    }
    if !matches!(req.head_dim, 64 | 128 | 256) {
        return None;
    }
    let has_gdn = req.layers.iter().any(|l| l.gdn.is_some());
    if has_gdn {
        if req.ssm.is_none() {
            return None;
        }
        if req
            .layers
            .iter()
            .filter_map(|l| l.gdn.as_ref())
            .any(|g| g.d != 128 || g.d_conv < 2)
        {
            return None;
        }
    }
    let rope_ok = req.rope.len() == m * req.rot_dim / 2
        || (m == 1 && req.rope.is_empty() && has_device_rope_freq());
    if !rope_ok {
        return None;
    }
    if m > 1 {
        let mut argmax = Vec::with_capacity(m);
        let mut hidden_out = Vec::with_capacity(m * hidden);
        for row in 0..m {
            let xrow = &req.x[row * hidden..(row + 1) * hidden];
            let rope = &req.rope[row * req.rot_dim / 2..(row + 1) * req.rot_dim / 2];
            let TokenOut::Rows {
                argmax: a,
                hidden: h,
            } = decode_token_one(req, xrow, rope, req.pos + row)?
            else {
                return None;
            };
            argmax.push(a[0]);
            hidden_out.extend_from_slice(&h);
        }
        return Some(TokenOut::Rows {
            argmax,
            hidden: hidden_out,
        });
    }
    decode_token_one(req, req.x, req.rope, req.pos)
}

/// After a successful greedy `decode_token` (argmax), run `n_more` further tokens
/// on-device: embed(d_argmax) -> d_pos++ -> graph.launch, one sync at the end.
/// Requires a live whole-token CUDA graph + embed chain kernels + `req.embd`.
pub fn decode_greedy_continue(req: &TokenReq, n_more: usize) -> Option<Vec<u32>> {
    if n_more == 0 {
        return Some(Vec::new());
    }
    if !is_attached() || !req.argmax {
        return None;
    }
    let (ty, w, hidden) = req.embd?;
    if hidden == 0 || !fuse_decode_eligible(req) {
        return None;
    }
    if !gpu_has_chain_kernels() {
        return None;
    }
    let use_graph = crate::runtime::get().cuda_graph;
    if !use_graph {
        return None;
    }

    DECODE_ATTEMPTS.fetch_add(n_more as u64, Ordering::Relaxed);
    let out = with_gpu(|gpu| {
        if gpu.decode_graph.is_none() || gpu.decode_graph_key == 0 {
            return None;
        }
        if !gpu.fns.contains_key("embed_row_f32_dtoken") || !gpu.fns.contains_key("inc_u32") {
            return None;
        }
        let stream = Arc::clone(&gpu.stream);
        crate::gpu::cuda::ggml::prepare_for_cudarc();

        // Resolve arena bases once — DevicePtr per token records stream events.
        let ybase = {
            let (p, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(g);
            p
        };
        let (chunk, w_base) = resolve_w(gpu, w)?;
        let rb = row_bytes(ty, hidden)?;
        let fmt = dq_fmt(ty)?;
        let hist = stream.alloc_zeros::<u32>(n_more).ok()?;
        let (argmax_p, pos_p, hist_p) = {
            let (a, g) = DevicePtr::device_ptr(&gpu.d_argmax, &stream);
            drop(g);
            let (p, g) = DevicePtr::device_ptr(&gpu.d_pos, &stream);
            drop(g);
            let (h, g) = DevicePtr::device_ptr(&hist, &stream);
            drop(g);
            (a, p, h)
        };
        let wbuf = &gpu.chunks[chunk].buf;
        let out_p = ybase; // x_at = 0
        let f_embed = gpu.func_owned("embed_row_f32_dtoken")?;
        let f_inc = gpu.func_owned("inc_u32")?;
        let f_store = gpu.func_owned("store_u32_at")?;
        let n_in_u = hidden as u32;
        let fmt_u = fmt;
        let rb_u = rb as u32;
        let embed_cfg = if matches!(fmt, 0 | 1 | 2) && hidden % 256 == 0 {
            LaunchConfig {
                grid_dim: (1, n_in_u / 256, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        let one_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let t0 = Instant::now();
        for i in 0..n_more {
            unsafe {
                stream
                    .launch_builder_pdl(&f_embed)
                    .arg(wbuf)
                    .arg(&out_p)
                    .arg(&n_in_u)
                    .arg(&w_base)
                    .arg(&rb_u)
                    .arg(&fmt_u)
                    .arg(&argmax_p)
                    .launch_pdl(embed_cfg)
            }
            .ok()?;
            unsafe {
                stream
                    .launch_builder_pdl(&f_inc)
                    .arg(&pos_p)
                    .launch_pdl(one_cfg)
            }
            .ok()?;
            gpu.decode_graph
                .as_ref()?
                .launch()
                .map_err(|e| {
                    eprintln!("cuda: greedy chain graph launch failed: {e}");
                    e
                })
                .ok()?;
            let idx = i as u32;
            unsafe {
                stream
                    .launch_builder_pdl(&f_store)
                    .arg(&hist_p)
                    .arg(&argmax_p)
                    .arg(&idx)
                    .launch_pdl(one_cfg)
            }
            .ok()?;
        }
        let enc = t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();
        gpu.sync()?;
        note_call(n_more as u64, enc, t1.elapsed().as_nanos() as u64);
        crate::gpu::cuda::ggml::mark_cudarc_dirty();
        stream.clone_dtoh(&hist).ok()
    });
    match out {
        Some(v) if v.len() == n_more => {
            DECODE_SUCCESSES.fetch_add(n_more as u64, Ordering::Relaxed);
            Some(v)
        }
        _ => {
            // Do not count declines: caller may finish one-by-one without
            // poisoning fail-closed attempt/success accounting.
            DECODE_ATTEMPTS.fetch_sub(n_more as u64, Ordering::Relaxed);
            None
        }
    }
}

pub(super) fn gpu_has_chain_kernels() -> bool {
    with_gpu(|gpu| {
        Some(
            gpu.fns.contains_key("embed_row_f32_dtoken")
                && gpu.fns.contains_key("inc_u32")
                && gpu.fns.contains_key("store_u32_at"),
        )
    })
    .unwrap_or(false)
}

/// One GDN decode step: wqkv/z/α/β matvecs, depthwise conv + deltanet +
/// gated out-norm on the SSM region, then ssm_out projection.
pub(super) fn gdn_decode_layer(
    g: &TokenGdn<'_>,
    h: &[f32],
    hidden: usize,
    eps: f32,
    ssm: &SharedRegion,
) -> Option<Vec<f32>> {
    if g.d != 128 || g.d_conv < 2 {
        return None;
    }
    let key_dim = g.heads_k * g.d;
    let value_dim = g.heads_v * g.d;
    let channels = key_dim * 2 + value_dim;
    if g.wqkv.2 != channels || g.zgate.2 != value_dim || g.ssm_out.2 != hidden {
        return None;
    }
    if g.alpha.len() != hidden * g.heads_v * 4
        || g.beta.len() != hidden * g.heads_v * 4
        || g.conv1d.len() != channels * g.d_conv * 4
        || g.a.len() != g.heads_v
        || g.dt.len() != g.heads_v
        || g.ssm_norm.len() != g.d
    {
        return None;
    }
    let conv_span = g.conv_off + (g.d_conv - 1) * channels;
    let state_span = g.state_off + g.heads_v * g.d * g.d;
    if conv_span * 4 > ssm.len || state_span * 4 > ssm.len {
        return None;
    }
    // Prove conv1d lives in the attached weight window (same residency gate as Metal).
    let _ = with_gpu(|gpu| resolve_w(gpu, g.conv1d))?;

    let qkv = matvec(g.wqkv.0, g.wqkv.1, hidden, channels, h)?;
    let z = matvec(g.zgate.0, g.zgate.1, hidden, value_dim, h)?;
    let alpha = matvec(GgmlType::F32, g.alpha, hidden, g.heads_v, h)?;
    let beta = matvec(GgmlType::F32, g.beta, hidden, g.heads_v, h)?;
    let conv_w = norm_f32_bytes(g.conv1d, channels * g.d_conv)?;

    let gy = with_gpu(|gpu| {
        // x_arena: ab | z | a | dt | ssm_norm | conv1d
        // y_arena: qkv (in/out of conv) then deltanet out
        let ab_at = 0usize;
        let z_at = ab_at + 2 * g.heads_v;
        let a_at = z_at + value_dim;
        let dt_at = a_at + g.heads_v;
        let sn_at = dt_at + g.heads_v;
        let cw_at = sn_at + g.d;
        let x_need = cw_at + conv_w.len();
        gpu.ensure_arenas(x_need * 4, channels.max(value_dim) * 4)?;

        let mut xhost = vec![0f32; x_need];
        xhost[ab_at..ab_at + g.heads_v].copy_from_slice(&alpha);
        xhost[ab_at + g.heads_v..ab_at + 2 * g.heads_v].copy_from_slice(&beta);
        xhost[z_at..z_at + value_dim].copy_from_slice(&z);
        xhost[a_at..a_at + g.heads_v].copy_from_slice(g.a);
        xhost[dt_at..dt_at + g.heads_v].copy_from_slice(g.dt);
        xhost[sn_at..sn_at + g.d].copy_from_slice(g.ssm_norm);
        xhost[cw_at..cw_at + conv_w.len()].copy_from_slice(&conv_w);
        gpu.stream
            .memcpy_htod(&xhost, &mut gpu.x_arena.slice_mut(0..x_need))
            .ok()?;
        gpu.stream
            .memcpy_htod(&qkv, &mut gpu.y_arena.slice_mut(0..channels))
            .ok()?;

        let channels_u = channels as u32;
        let d_conv_u = g.d_conv as u32;
        let conv_off = g.conv_off as u64;
        let state_off = g.state_off as u64;
        let heads_k = g.heads_k as u32;
        let heads_v = g.heads_v as u32;
        let d_u = g.d as u32;
        let key_dim_u = key_dim as u32;

        {
            let f = gpu.func_owned("gdn_conv")?;
            let cfg = LaunchConfig {
                grid_dim: (channels_u.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut qkv_d = gpu.y_arena.slice_mut(0..channels);
            let w = gpu.x_arena.slice(cw_at..cw_at + conv_w.len());
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&mut qkv_d)
                    .arg(&ssm.buf)
                    .arg(&w)
                    .arg(&channels_u)
                    .arg(&d_conv_u)
                    .arg(&conv_off)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        {
            let f = gpu.func_owned("gdn_step")?;
            let cfg = LaunchConfig {
                grid_dim: (d_u / 4, heads_v, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            };
            // qkv lives in y[0..channels]; write out into a side alloc then
            // copy back so we do not mutably alias y.
            let mut out_buf = unsafe { gpu.stream.alloc::<f32>(value_dim) }.ok()?;
            let qkv_d = gpu.y_arena.slice(0..channels);
            let ab = gpu.x_arena.slice(ab_at..ab_at + 2 * g.heads_v);
            let a_log = gpu.x_arena.slice(a_at..a_at + g.heads_v);
            let dt_bias = gpu.x_arena.slice(dt_at..dt_at + g.heads_v);
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&ssm.buf)
                    .arg(&qkv_d)
                    .arg(&ab)
                    .arg(&a_log)
                    .arg(&dt_bias)
                    .arg(&mut out_buf)
                    .arg(&heads_k)
                    .arg(&heads_v)
                    .arg(&d_u)
                    .arg(&key_dim_u)
                    .arg(&eps)
                    .arg(&state_off)
                    .launch_pdl(cfg)
            }
            .ok()?;
            {
                let f_n = gpu.func_owned("gdn_out_norm")?;
                let cfg = LaunchConfig {
                    grid_dim: (heads_v, 1, 1),
                    block_dim: (d_u, 1, 1),
                    shared_mem_bytes: 0,
                };
                let z_d = gpu.x_arena.slice(z_at..z_at + value_dim);
                let wn = gpu.x_arena.slice(sn_at..sn_at + g.d);
                unsafe {
                    gpu.stream
                        .launch_builder_pdl(&f_n)
                        .arg(&mut out_buf)
                        .arg(&z_d)
                        .arg(&wn)
                        .arg(&heads_v)
                        .arg(&d_u)
                        .arg(&eps)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            gpu.sync()?;
            return gpu.stream.clone_dtoh(&out_buf).ok();
        }
    })?;

    matvec(g.ssm_out.0, g.ssm_out.1, value_dim, hidden, &gy)
}

pub(super) fn fuse_decode_eligible(req: &TokenReq) -> bool {
    if crate::runtime::get().cuda_no_fuse {
        return false;
    }
    for layer in req.layers {
        // Attn bias (qwen2) is handled in the fused stream via add_bias_f32.
        if layer.gdn.is_some() || layer.gate_in_q {
            return false;
        }
        if !matches!(layer.ffn, TokenFfn::Dense { .. }) {
            return false;
        }
    }
    true
}

pub(super) fn align64(n: usize) -> usize {
    (n + 63) & !63
}

pub(super) fn rope_replicated(rope: &[[f32; 2]], n_heads: usize) -> Vec<f32> {
    let pairs = rope.len();
    let mut out = vec![0f32; n_heads * pairs * 2];
    for h in 0..n_heads {
        for (i, p) in rope.iter().enumerate() {
            let o = (h * pairs + i) * 2;
            out[o] = p[0];
            out[o + 1] = p[1];
        }
    }
    out
}

pub(super) fn decode_token_one(
    req: &TokenReq,
    x_in: &[f32],
    rope_in: &[[f32; 2]],
    pos: usize,
) -> Option<TokenOut> {
    if fuse_decode_eligible(req) {
        match decode_token_one_fused(req, x_in, rope_in, pos) {
            Some(out) => return Some(out),
            None => {
                eprintln!(
                    "cuda: fused decode returned None (graph={})",
                    crate::runtime::get().cuda_graph
                );
            }
        }
    }

    let hidden = x_in.len();
    let hd = req.head_dim;
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let group = req.n_heads / req.n_kv_heads.max(1);
    let mut x = x_in.to_vec();

    for layer in req.layers {
        let attn_w = norm_f32_bytes(layer.attn_norm, hidden)?;
        let mut h = x.clone();
        crate::ops::rmsnorm(&mut h, &attn_w, req.eps);

        if let Some(g) = &layer.gdn {
            if std::env::var_os("ALLPAKA_GDN_ZERO").is_none() {
                let ssm = req.ssm?;
                let delta = gdn_decode_layer(g, &h, hidden, req.eps, ssm)?;
                for (a, b) in x.iter_mut().zip(&delta) {
                    *a += *b;
                }
            }
        } else {
            let wq_out = if layer.gate_in_q { 2 * q_dim } else { q_dim };
            if layer.wq.2 != wq_out || layer.wk.2 != kv || layer.wv.2 != kv || layer.wo.2 != hidden
            {
                return None;
            }
            let mut q_raw = matvec(layer.wq.0, layer.wq.1, hidden, wq_out, &h)?;
            let mut k = matvec(layer.wk.0, layer.wk.1, hidden, kv, &h)?;
            let mut v = matvec(layer.wv.0, layer.wv.1, hidden, kv, &h)?;

            if let Some(b) = layer.q_bias {
                let bias = norm_f32_bytes(b, wq_out)?;
                for (a, b) in q_raw.iter_mut().zip(&bias) {
                    *a += *b;
                }
            }
            if let Some(b) = layer.k_bias {
                let bias = norm_f32_bytes(b, kv)?;
                for (a, b) in k.iter_mut().zip(&bias) {
                    *a += *b;
                }
            }
            if let Some(b) = layer.v_bias {
                let bias = norm_f32_bytes(b, kv)?;
                for (a, b) in v.iter_mut().zip(&bias) {
                    *a += *b;
                }
            }

            let (mut q, gate_q) = if layer.gate_in_q {
                let mut q = vec![0f32; q_dim];
                let mut gate = vec![0f32; q_dim];
                for hi in 0..req.n_heads {
                    let src = &q_raw[hi * 2 * hd..(hi + 1) * 2 * hd];
                    q[hi * hd..(hi + 1) * hd].copy_from_slice(&src[..hd]);
                    gate[hi * hd..(hi + 1) * hd].copy_from_slice(&src[hd..]);
                }
                (q, Some(gate))
            } else {
                (q_raw, None)
            };

            let rot_dim = req.rot_dim.min(hd);
            let rope = &rope_in[..rot_dim / 2];
            for hi in 0..req.n_heads {
                let qh = &mut q[hi * hd..(hi + 1) * hd];
                if let Some(wn) = layer.q_norm {
                    crate::ops::rmsnorm(qh, wn, req.eps);
                }
                if req.rope_norm {
                    crate::ops::rope_norm_cached_from_array(&mut qh[..rot_dim], rope);
                } else {
                    crate::ops::rope_neox_cached_from_array(&mut qh[..rot_dim], rope);
                }
            }
            for hi in 0..req.n_kv_heads {
                let kh = &mut k[hi * hd..(hi + 1) * hd];
                if let Some(wn) = layer.k_norm {
                    crate::ops::rmsnorm(kh, wn, req.eps);
                }
                if req.rope_norm {
                    crate::ops::rope_norm_cached_from_array(&mut kh[..rot_dim], rope);
                } else {
                    crate::ops::rope_neox_cached_from_array(&mut kh[..rot_dim], rope);
                }
            }

            store_kv(req.cache, &k, layer.k_off, pos, req.kv_dim)?;
            store_kv(req.cache, &v, layer.v_off, pos, req.kv_dim)?;

            let mut attn = attend(&AttnReq {
                cache: req.cache,
                k_off: layer.k_off,
                v_off: layer.v_off,
                q: &q,
                kv_dim: req.kv_dim,
                head_dim: hd,
                n_q_heads: req.n_heads,
                group,
                n_pos: pos + 1,
                scale: req.scale,
            })?;
            if let Some(gate) = gate_q {
                for (a, g) in attn.iter_mut().zip(&gate) {
                    *a *= 1.0 / (1.0 + (-*g).exp());
                }
            }
            let delta = matvec(layer.wo.0, layer.wo.1, q_dim, hidden, &attn)?;
            for (a, b) in x.iter_mut().zip(&delta) {
                *a += *b;
            }
        } // end else attention

        let ffn_w = norm_f32_bytes(layer.ffn_norm, hidden)?;
        let mut h = x.clone();
        crate::ops::rmsnorm(&mut h, &ffn_w, req.eps);

        let ffn_out = match &layer.ffn {
            TokenFfn::Dense { gate, up, down } => {
                let g = matvec(gate.0, gate.1, hidden, gate.2, &h)?;
                let u = matvec(up.0, up.1, hidden, up.2, &h)?;
                let mut act = g;
                crate::ops::swiglu(&mut act, &u);
                matvec(down.0, down.1, gate.2, hidden, &act)?
            }
            TokenFfn::Moe {
                router,
                router_bias,
                gate,
                up,
                down,
                expert_ffn,
                n_used,
                sigmoid,
                shared,
                shared_gate,
            } => {
                if router.0 != GgmlType::F32 || *n_used == 0 {
                    return None;
                }
                let mut logits = matvec(GgmlType::F32, router.1, hidden, router.2, &h)?;
                if let Some(b) = router_bias {
                    let bias = norm_f32_bytes(b, router.2)?;
                    for (l, b) in logits.iter_mut().zip(&bias) {
                        *l += *b;
                    }
                }
                let n_expert = router.2;
                let mut scores: Vec<(f32, usize)> = logits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let s = if *sigmoid {
                            1.0 / (1.0 + (-v).exp())
                        } else {
                            v
                        };
                        (s, i)
                    })
                    .collect();
                if !*sigmoid {
                    let mx = scores.iter().map(|s| s.0).fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for s in &mut scores {
                        s.0 = (s.0 - mx).exp();
                        sum += s.0;
                    }
                    for s in &mut scores {
                        s.0 /= sum;
                    }
                }
                scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let mut picks: Vec<(usize, f32)> = scores
                    .into_iter()
                    .take(*n_used)
                    .map(|(w, i)| (i, w))
                    .collect();
                let wsum: f32 = picks.iter().map(|p| p.1).sum();
                if wsum > 0.0 {
                    for p in &mut picks {
                        p.1 /= wsum;
                    }
                }
                let gate_rb = gate.1.len() / n_expert;
                let up_rb = up.1.len() / n_expert;
                let down_rb = down.1.len() / n_expert;
                let mut acc = vec![0f32; hidden];
                for &(eid, wt) in &picks {
                    let g = matvec(
                        gate.0,
                        &gate.1[eid * gate_rb..(eid + 1) * gate_rb],
                        hidden,
                        *expert_ffn,
                        &h,
                    )?;
                    let u = matvec(
                        up.0,
                        &up.1[eid * up_rb..(eid + 1) * up_rb],
                        hidden,
                        *expert_ffn,
                        &h,
                    )?;
                    let mut act = g;
                    crate::ops::swiglu(&mut act, &u);
                    let d = matvec(
                        down.0,
                        &down.1[eid * down_rb..(eid + 1) * down_rb],
                        *expert_ffn,
                        hidden,
                        &act,
                    )?;
                    for (a, b) in acc.iter_mut().zip(&d) {
                        *a += wt * *b;
                    }
                }
                if let Some(sh) = shared {
                    let g = matvec(sh[0].0, sh[0].1, hidden, sh[0].2, &h)?;
                    let u = matvec(sh[1].0, sh[1].1, hidden, sh[1].2, &h)?;
                    let mut act = g;
                    crate::ops::swiglu(&mut act, &u);
                    let d = matvec(sh[2].0, sh[2].1, sh[0].2, hidden, &act)?;
                    let sh_scale = match shared_gate {
                        Some(sg_bytes) => {
                            let sg_logit = matvec(GgmlType::F32, sg_bytes, hidden, 1, &h)?;
                            1.0 / (1.0 + (-sg_logit[0]).exp())
                        }
                        None => 1.0,
                    };
                    for (a, b) in acc.iter_mut().zip(&d) {
                        *a += sh_scale * *b;
                    }
                }
                acc
            }
        };
        for (a, b) in x.iter_mut().zip(&ffn_out) {
            *a += *b;
        }
    }

    let out_w = norm_f32_bytes(req.output_norm, hidden)?;
    crate::ops::rmsnorm(&mut x, &out_w, req.eps);
    let logits = matvec(req.output.0, req.output.1, hidden, req.output.2, &x)?;
    if req.m > 1 {
        let idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        Some(TokenOut::Rows {
            argmax: vec![idx],
            hidden: x,
        })
    } else if req.argmax {
        let idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        Some(TokenOut::Argmax(idx))
    } else {
        Some(TokenOut::Logits(logits))
    }
}
