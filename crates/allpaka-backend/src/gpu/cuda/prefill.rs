//! Device-resident prefill: residual stream upload and readback, fused
//! attention and GDN blocks, and the dense feed-forward fast path.

use super::*;

pub fn prefill_begin(xs: &[f32]) -> Option<()> {
    with_gpu(|gpu| {
        gpu.ensure_prefill(xs.len())?;
        // Dual-queue: ggml MMQ/FA on private stream, cudarc on peer.
        // Opt out with ALLPAKA_PREFILL_DUAL=0 (shared stream, no overlap).
        let dual = std::env::var("ALLPAKA_PREFILL_DUAL")
            .map_or(true, |v| !(v == "0" || v.eq_ignore_ascii_case("false")));
        if dual {
            crate::gpu::cuda::ggml::clear_shared_for_prefill();
        }
        if pf_graph_on() && gpu.pf_graph_len != xs.len() {
            gpu.pf_graph = None;
            gpu.pf_graph_seen = 0;
            gpu.pf_graph_len = xs.len();
        }
        gpu.stream
            .memcpy_htod(xs, &mut gpu.pf_x.slice_mut(0..xs.len()))
            .ok()?;
        gpu.pf_active = true;
        gpu.pf_len = xs.len();
        gpu.pf_ffn_done = false;
        gpu.pf_rope_n = 0;
        if pf_graph_on() && !dual && gpu.pf_graph.is_none() && gpu.pf_graph_seen == 1 {
            if gpu
                .stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .is_ok()
            {
                gpu.pf_capturing = true;
            }
        }
        Some(())
    })
}

/// Replay a captured prefill. `None` means the caller should run the normal path.
pub fn prefill_replay(xs: &[f32]) -> Option<Vec<f32>> {
    if !pf_graph_on() {
        return None;
    }
    with_gpu(|gpu| {
        if gpu.pf_graph_len != xs.len() {
            return Some(None);
        }
        if gpu.pf_graph.is_none() {
            return Some(None);
        }
        gpu.ensure_prefill(xs.len())?;
        gpu.stream
            .memcpy_htod(xs, &mut gpu.pf_x.slice_mut(0..xs.len()))
            .ok()?;
        gpu.pf_graph.as_ref()?.launch().ok()?;
        crate::gpu::cuda::ggml::sync();
        gpu.sync()?;
        let mut out = vec![0f32; xs.len()];
        gpu.stream
            .memcpy_dtoh(&gpu.pf_x.slice(0..xs.len()), &mut out)
            .ok()?;
        Some(Some(out))
    })
    .flatten()
}

pub(super) fn pf_graph_on() -> bool {
    std::env::var("ALLPAKA_PF_GRAPH")
        .map_or(true, |v| !(v == "0" || v.eq_ignore_ascii_case("false")))
}

pub fn prefill_end_output(len: usize) -> Option<crate::activation::Activation> {
    let mut out = vec![0.0; len];
    prefill_end(&mut out)?;
    Some(out.into())
}

pub fn prefill_end(xs: &mut [f32]) -> Option<()> {
    with_gpu(|gpu| {
        if !gpu.pf_active || xs.len() != gpu.pf_len {
            return None;
        }
        if gpu.pf_capturing {
            let graph_flags = unsafe { std::mem::transmute::<u32, CUgraphInstantiate_flags>(0) };
            match gpu.stream.end_capture(graph_flags) {
                Ok(Some(g)) => {
                    eprintln!("cuda: prefill graph captured ({} floats)", gpu.pf_len);
                    gpu.pf_graph = Some(g);
                    gpu.pf_graph_len = gpu.pf_len;
                }
                Ok(None) => eprintln!("cuda: prefill graph empty"),
                Err(e) => eprintln!("cuda: prefill end_capture failed: {e}"),
            }
            gpu.pf_capturing = false;
        }
        gpu.pf_graph_seen = gpu.pf_graph_seen.saturating_add(1);
        // Drain ggml private queue before sharing cudarc again (graph capture).
        crate::gpu::cuda::ggml::sync();
        gpu.sync()?;
        gpu.stream
            .memcpy_dtoh(&gpu.pf_x.slice(0..xs.len()), xs)
            .ok()?;
        gpu.pf_active = false;
        // Re-install shared stream for decode CUDA graphs + FA-inline.
        crate::gpu::cuda::ggml::bind_peer_stream(&gpu.stream);
        Some(())
    })
}

pub fn prefill_abort() {
    let _ = with_gpu(|gpu| {
        gpu.pf_active = false;
        gpu.pf_ffn_done = false;
        Some(())
    });
}

/// Dense fused prefill attention: keep activations on device (Metal-style).
/// No mid-layer D2H; Q/K RMSNorm + RoPE + store_kv + attend + WO + residual
/// + FFN-norm all stream-ordered. Returns empty Vec for dense (`n_expert==0`)
/// or router logits for MoE.
pub(super) fn try_prefill_attn_fused_dev(
    req: &PrefillAttnReq,
    fusion: &PrefillFusion<'_>,
) -> Option<Vec<f32>> {
    let hidden = req.hs.len() / req.m.max(1);
    let hd = req.head_dim;
    let m = req.m;
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let rot = req.rot_dim.min(hd);
    if rot == 0 || rot % 2 != 0 {
        return None;
    }
    let pairs = rot / 2;
    if req.ropes.len() < m * pairs {
        return None;
    }
    // Flatten [[sin,cos]; ...] without copy when layout matches.
    let rope_host: &[f32] =
        unsafe { std::slice::from_raw_parts(req.ropes.as_ptr() as *const f32, m * pairs * 2) };

    with_gpu(|gpu| {
        if !gpu.pf_active || gpu.pf_len < m * hidden {
            return None;
        }
        let q_at = 0usize;
        let k_at = q_at + m * q_dim;
        let v_at = k_at + m * kv;
        let attn_at = v_at + m * kv;
        let proj_at = attn_at + m * q_dim;
        let y_need = (proj_at + m * hidden) * 4;
        let rope_at = m * hidden;
        let qn_at = rope_at + m * pairs * 2;
        let kn_at = qn_at + hd;
        let qb_at = kn_at + hd;
        let kb_at = qb_at + q_dim;
        let vb_at = kb_at + kv;
        let x_need = if req.attn_bias.is_some() {
            (vb_at + kv) * 4
        } else {
            (kn_at + hd) * 4
        };
        gpu.ensure_arenas(x_need, y_need)?;

        // attn RMSNorm: pf_x -> pf_hs
        {
            let w = norm_f32_bytes(fusion.attn_norm, hidden)?;
            rmsnorm_pf_x_into_hs(gpu, &w, hidden, m, req.eps)?;
        }
        if gpu.pf_rope_n != rope_host.len() {
            gpu.stream
                .memcpy_htod(
                    rope_host,
                    &mut gpu.x_arena.slice_mut(rope_at..rope_at + rope_host.len()),
                )
                .ok()?;
            gpu.pf_rope_n = rope_host.len();
        }

        let (wq_c, wq_o) = resolve_w(gpu, req.wq.1)?;
        let (wk_c, wk_o) = resolve_w(gpu, req.wk.1)?;
        let (wv_c, wv_o) = resolve_w(gpu, req.wv.1)?;
        let (wo_c, wo_o) = resolve_w(gpu, req.wo.1)?;
        launch_gemm_dequant(
            gpu,
            req.wq.0,
            wq_c,
            wq_o,
            hidden,
            q_dim,
            m,
            true,
            &[],
            q_at,
            false,
            true,
            None,
        )?;
        launch_gemm_dequant(
            gpu,
            req.wk.0,
            wk_c,
            wk_o,
            hidden,
            kv,
            m,
            true,
            &[],
            k_at,
            true,
            true,
            None,
        )?;
        launch_gemm_dequant(
            gpu,
            req.wv.0,
            wv_c,
            wv_o,
            hidden,
            kv,
            m,
            true,
            &[],
            v_at,
            true,
            true,
            None,
        )?;

        crate::gpu::cuda::ggml::prepare_for_cudarc();
        let stream = Arc::clone(&gpu.stream);
        // Stage head-norm weights before taking device ptrs (borrowck).
        if let Some(wn) = req.q_norm {
            if wn.len() < hd {
                return None;
            }
            stream
                .memcpy_htod(&wn[..hd], &mut gpu.x_arena.slice_mut(qn_at..qn_at + hd))
                .ok()?;
        }
        if let Some(wn) = req.k_norm {
            if wn.len() < hd {
                return None;
            }
            stream
                .memcpy_htod(&wn[..hd], &mut gpu.x_arena.slice_mut(kn_at..kn_at + hd))
                .ok()?;
        }
        if let Some((qb, kb, vb)) = req.attn_bias {
            let qb = norm_f32_bytes(qb, q_dim)?;
            let kb = norm_f32_bytes(kb, kv)?;
            let vb = norm_f32_bytes(vb, kv)?;
            stream
                .memcpy_htod(&qb, &mut gpu.x_arena.slice_mut(qb_at..qb_at + q_dim))
                .ok()?;
            stream
                .memcpy_htod(&kb, &mut gpu.x_arena.slice_mut(kb_at..kb_at + kv))
                .ok()?;
            stream
                .memcpy_htod(&vb, &mut gpu.x_arena.slice_mut(vb_at..vb_at + kv))
                .ok()?;
            let f = gpu.func_owned("add_bias_rows_f32")?;
            let (ybase, _yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            let (xbase, _xg) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
            let m_u = m as u32;
            for (off, bias_off, n) in [(q_at, qb_at, q_dim), (k_at, kb_at, kv), (v_at, vb_at, kv)] {
                let n_u = n as u32;
                let cfg = LaunchConfig {
                    grid_dim: (n_u.div_ceil(256), m_u, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let xp = ybase + (off * 4) as u64;
                let bp = xbase + (bias_off * 4) as u64;
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(&xp)
                        .arg(&bp)
                        .arg(&m_u)
                        .arg(&n_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
        }

        {
            let (ybase, _yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            let (xbase, _xg) = DevicePtr::device_ptr(&gpu.x_arena, &stream);

            if req.q_norm.is_some() {
                let f = gpu.func_owned("rmsnorm_f32")?;
                let n_u = hd as u32;
                let rows_u = (m * req.n_heads) as u32;
                let eps = req.eps;
                let cfg = LaunchConfig {
                    grid_dim: (rows_u, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let xp = ybase + (q_at * 4) as u64;
                let wp = xbase + (qn_at * 4) as u64;
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(&xp)
                        .arg(&wp)
                        .arg(&n_u)
                        .arg(&eps)
                        .arg(&rows_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            {
                let f = gpu.func_owned("rope_neox_batch")?;
                let m_u = m as u32;
                let heads_u = req.n_heads as u32;
                let hd_u = hd as u32;
                let rot_u = rot as u32;
                let cfg = LaunchConfig {
                    grid_dim: (heads_u, m_u, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let xp = ybase + (q_at * 4) as u64;
                let rp = xbase + (rope_at * 4) as u64;
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(&xp)
                        .arg(&rp)
                        .arg(&m_u)
                        .arg(&heads_u)
                        .arg(&hd_u)
                        .arg(&rot_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            if req.k_norm.is_some() {
                let f = gpu.func_owned("rmsnorm_f32")?;
                let n_u = hd as u32;
                let rows_u = (m * req.n_kv_heads) as u32;
                let eps = req.eps;
                let cfg = LaunchConfig {
                    grid_dim: (rows_u, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let xp = ybase + (k_at * 4) as u64;
                let wp = xbase + (kn_at * 4) as u64;
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(&xp)
                        .arg(&wp)
                        .arg(&n_u)
                        .arg(&eps)
                        .arg(&rows_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            {
                let f = gpu.func_owned("rope_neox_batch")?;
                let m_u = m as u32;
                let heads_u = req.n_kv_heads as u32;
                let hd_u = hd as u32;
                let rot_u = rot as u32;
                let cfg = LaunchConfig {
                    grid_dim: (heads_u, m_u, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let xp = ybase + (k_at * 4) as u64;
                let rp = xbase + (rope_at * 4) as u64;
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(&xp)
                        .arg(&rp)
                        .arg(&m_u)
                        .arg(&heads_u)
                        .arg(&hd_u)
                        .arg(&rot_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            {
                let f = gpu.func_owned("store_kv_batch_f16")?;
                let kv_u = kv as u32;
                let m_u = m as u32;
                let pos_u = req.base as u32;
                let cfg = LaunchConfig {
                    grid_dim: (kv_u.div_ceil(256), m_u, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let k_off = req.k_off as u64;
                let v_off = req.v_off as u64;
                let kp = ybase + (k_at * 4) as u64;
                let vp = ybase + (v_at * 4) as u64;
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(&req.cache.buf)
                        .arg(&kp)
                        .arg(&k_off)
                        .arg(&kv_u)
                        .arg(&m_u)
                        .arg(&pos_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(&req.cache.buf)
                        .arg(&vp)
                        .arg(&v_off)
                        .arg(&kv_u)
                        .arg(&m_u)
                        .arg(&pos_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            {
                let group_usz = req.n_heads / req.n_kv_heads.max(1);
                let k_off = req.k_off as u64;
                let v_off = req.v_off as u64;
                let kv_dim = req.kv_dim as u32;
                let head_dim = hd as u32;
                let n_q = req.n_heads as u32;
                let group = group_usz as u32;
                let m_u = m as u32;
                let base = req.base as u32;
                let scale = req.scale;
                let qp = ybase + (q_at * 4) as u64;
                let op = ybase + (attn_at * 4) as u64;

                let mut used_fa = false;
                if crate::gpu::cuda::ggml::enabled() && m >= 32 && hd <= 256 && group_usz > 0 {
                    crate::gpu::cuda::ggml::prepare_from_cudarc(&stream);
                    let (cache_p, cg) = DevicePtr::device_ptr(&req.cache.buf, &stream);
                    drop(cg);
                    let k_dev = cache_p + k_off * 2;
                    let v_dev = cache_p + v_off * 2;
                    used_fa = crate::gpu::cuda::ggml::flash_attn(
                        qp,
                        k_dev,
                        v_dev,
                        op,
                        hd,
                        req.n_heads,
                        req.n_kv_heads.max(1),
                        m,
                        req.base,
                        req.kv_dim,
                        scale,
                        None,
                    );
                    if used_fa {
                        crate::gpu::cuda::ggml::mark_cudarc_dirty();
                    }
                }
                if !used_fa {
                    let use_kv = false && attend_use_kv_fused(hd, group_usz);
                    let f = if use_kv {
                        gpu.func_owned("attend_gqa_batch_kv")?
                    } else {
                        gpu.func_owned("attend_gqa_batch")?
                    };
                    let grid_x = if use_kv {
                        req.n_kv_heads.max(1) as u32
                    } else {
                        n_q
                    };
                    let cfg = if use_kv {
                        attend_gqa_kv_launch_cfg(grid_x, m_u, group, head_dim)
                    } else {
                        attend_gqa_launch_cfg(grid_x, m_u)
                    };
                    unsafe {
                        stream
                            .launch_builder_pdl(&f)
                            .arg(&qp)
                            .arg(&req.cache.buf)
                            .arg(&op)
                            .arg(&k_off)
                            .arg(&v_off)
                            .arg(&kv_dim)
                            .arg(&head_dim)
                            .arg(&n_q)
                            .arg(&group)
                            .arg(&m_u)
                            .arg(&base)
                            .arg(&scale)
                            .launch_pdl(cfg)
                    }
                    .ok()?;
                    crate::gpu::cuda::ggml::mark_cudarc_dirty();
                }
            }
        } // drop DevicePtr guards

        let attn_dev = {
            let (p, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(g);
            p + (attn_at * 4) as u64
        };
        launch_gemm_dequant(
            gpu,
            req.wo.0,
            wo_c,
            wo_o,
            q_dim,
            hidden,
            m,
            true,
            &[],
            proj_at,
            false,
            false,
            Some(attn_dev),
        )?;
        crate::gpu::cuda::ggml::prepare_for_cudarc();
        // residual into pf_x
        {
            let f_add = gpu.func_owned("residual_add")?;
            let n = (m * hidden) as u32;
            let cfg = CudaGpu::cfg_1d(n, 256);
            let mut a = gpu.pf_x.slice_mut(0..m * hidden);
            let b = gpu.y_arena.slice(proj_at..proj_at + m * hidden);
            unsafe {
                stream
                    .launch_builder_pdl(&f_add)
                    .arg(&mut a)
                    .arg(&b)
                    .arg(&n)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        // FFN RMSNorm into pf_hs
        {
            let w = norm_f32_bytes(fusion.ffn_norm, hidden)?;
            rmsnorm_pf_x_into_hs(gpu, &w, hidden, m, req.eps)?;
        }
        if fusion.n_expert == 0 {
            // No sync: dense FFN continues from pf_hs on-device.
            return Some(Vec::new());
        }
        // MoE: need router logits on host path — sync once then D2H hs for router.
        gpu.sync()?;
        let hs = gpu
            .stream
            .clone_dtoh(&gpu.pf_hs.slice(0..m * hidden))
            .ok()?;
        run_matvec(
            gpu,
            GgmlType::F32,
            fusion.router,
            hidden,
            fusion.n_expert,
            &hs,
            m,
        )
    })
}

/// Dense FFN from `pf_hs` without D2H: gate/up/swiglu/down + residual into `pf_x`.
pub(super) fn try_dense_ffn_pf(req: &GroupedFfnReq) -> Option<Vec<f32>> {
    let comb = req.fused.as_ref()?;
    let m = comb.m;
    if req.n_expert > 1 || req.shared.is_some() || req.route.is_some() {
        return None;
    }
    if req.groups.len() != 1 || req.groups[0] != [0, 0, m as u32] {
        return None;
    }
    if req.tok.len() != m {
        return None;
    }
    for (i, &t) in req.tok.iter().enumerate() {
        if t as usize != i {
            return None;
        }
    }
    let hidden = req.hidden;
    let ffn = req.ffn;
    with_gpu(|gpu| {
        if !gpu.pf_active || gpu.pf_len < m * hidden {
            return None;
        }
        let gate_at = 0usize;
        let up_at = m * ffn;
        let down_at = up_at + m * ffn;
        gpu.ensure_arenas(m * ffn.max(hidden) * 4, (down_at + m * hidden) * 4)?;
        let (g_c, g_o) = resolve_w(gpu, req.gate.1)?;
        let (u_c, u_o) = resolve_w(gpu, req.up.1)?;
        let (d_c, d_o) = resolve_w(gpu, req.down.1)?;
        launch_gemm_dequant(
            gpu,
            req.gate.0,
            g_c,
            g_o,
            hidden,
            ffn,
            m,
            true,
            &[],
            gate_at,
            false,
            true,
            None,
        )?;
        launch_gemm_dequant(
            gpu,
            req.up.0,
            u_c,
            u_o,
            hidden,
            ffn,
            m,
            true,
            &[],
            up_at,
            true,
            true,
            None,
        )?;
        crate::gpu::cuda::ggml::prepare_for_cudarc();
        let stream = Arc::clone(&gpu.stream);
        {
            let (ybase, _yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            {
                let f = gpu.func_owned("swiglu")?;
                let n = (m * ffn) as u32;
                let cfg = CudaGpu::cfg_1d(n, 256);
                let gp = ybase + (gate_at * 4) as u64;
                let up_p = ybase + (up_at * 4) as u64;
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(&gp)
                        .arg(&up_p)
                        .arg(&n)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
        }
        crate::gpu::cuda::ggml::mark_cudarc_dirty();
        let gate_dev = {
            let (p, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(g);
            p + (gate_at * 4) as u64
        };
        launch_gemm_dequant(
            gpu,
            req.down.0,
            d_c,
            d_o,
            ffn,
            hidden,
            m,
            true,
            &[],
            down_at,
            false,
            false,
            Some(gate_dev),
        )?;
        crate::gpu::cuda::ggml::prepare_for_cudarc();
        {
            let f_add = gpu.func_owned("residual_add")?;
            let n = (m * hidden) as u32;
            let cfg = CudaGpu::cfg_1d(n, 256);
            let mut a = gpu.pf_x.slice_mut(0..m * hidden);
            let b = gpu.y_arena.slice(down_at..down_at + m * hidden);
            unsafe {
                stream
                    .launch_builder_pdl(&f_add)
                    .arg(&mut a)
                    .arg(&b)
                    .arg(&n)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        // Stream-ordered; prefill_end syncs once.
        Some(Vec::new())
    })
}

pub fn prefill_attn_block(req: &PrefillAttnReq) -> Option<Vec<f32>> {
    let hidden = req.hs.len() / req.m.max(1);
    let hd = req.head_dim;
    if !matches!(hd, 64 | 128 | 256) || req.m < MM_MIN_M {
        return None;
    }
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let wq_out = if req.gate_in_q { 2 * q_dim } else { q_dim };
    if req.wq.2 != wq_out || req.wk.2 != kv || req.wv.2 != kv || req.wo.2 != hidden {
        return None;
    }

    // Device-resident fused path: no mid-layer D2H for dense Qwen-style layers.
    // Attn bias (qwen2) is applied on-device after QKV GEMMs.
    if let Some(f) = &req.fusion {
        if !req.gate_in_q {
            if let Some(out) = try_prefill_attn_fused_dev(req, f) {
                return Some(out);
            }
        }
    }

    let hs_owned = if let Some(f) = &req.fusion {
        with_gpu(|gpu| {
            if !gpu.pf_active || gpu.pf_len < req.m * hidden {
                return None;
            }
            let w = norm_f32_bytes(f.attn_norm, hidden)?;
            rmsnorm_pf_x_into_hs(gpu, &w, hidden, req.m, req.eps)?;
            gpu.sync()?;
            gpu.stream
                .clone_dtoh(&gpu.pf_hs.slice(0..req.m * hidden))
                .ok()
        })
    } else {
        None
    };
    let hs = hs_owned.as_deref().unwrap_or(req.hs);

    let mut qkv = matvec_batch(&[
        MatvecReq {
            ty: req.wq.0,
            w: req.wq.1,
            n_in: hidden,
            n_out: wq_out,
            x: hs,
            m: req.m,
        },
        MatvecReq {
            ty: req.wk.0,
            w: req.wk.1,
            n_in: hidden,
            n_out: kv,
            x: hs,
            m: req.m,
        },
        MatvecReq {
            ty: req.wv.0,
            w: req.wv.1,
            n_in: hidden,
            n_out: kv,
            x: hs,
            m: req.m,
        },
    ])?;
    let mut v = qkv.pop().unwrap();
    let mut k = qkv.pop().unwrap();
    let mut q = qkv.pop().unwrap();

    if let Some((qb, kb, vb)) = req.attn_bias {
        let qb = norm_f32_bytes(qb, wq_out)?;
        let kb = norm_f32_bytes(kb, kv)?;
        let vb = norm_f32_bytes(vb, kv)?;
        for row in 0..req.m {
            for i in 0..wq_out {
                q[row * wq_out + i] += qb[i];
            }
            for i in 0..kv {
                k[row * kv + i] += kb[i];
                v[row * kv + i] += vb[i];
            }
        }
    }

    let mut q_use = if req.gate_in_q {
        let mut qq = vec![0f32; req.m * q_dim];
        for row in 0..req.m {
            for h in 0..req.n_heads {
                let src = &q[row * wq_out + h * 2 * hd..];
                qq[row * q_dim + h * hd..row * q_dim + (h + 1) * hd].copy_from_slice(&src[..hd]);
            }
        }
        qq
    } else {
        q
    };

    let pairs = req.rot_dim / 2;
    for row in 0..req.m {
        let rope = &req.ropes[row * pairs..(row + 1) * pairs];
        for h in 0..req.n_heads {
            let qh = &mut q_use[row * q_dim + h * hd..row * q_dim + (h + 1) * hd];
            if let Some(wn) = req.q_norm {
                crate::ops::rmsnorm(qh, wn, req.eps);
            }
            crate::ops::rope_neox_cached_from_array(&mut qh[..req.rot_dim.min(hd)], rope);
        }
        for h in 0..req.n_kv_heads {
            let kh = &mut k[row * kv + h * hd..row * kv + (h + 1) * hd];
            if let Some(wn) = req.k_norm {
                crate::ops::rmsnorm(kh, wn, req.eps);
            }
            crate::ops::rope_neox_cached_from_array(&mut kh[..req.rot_dim.min(hd)], rope);
        }
    }
    store_kv_batch(req.cache, &k, req.k_off, req.base, req.kv_dim, req.m)?;
    store_kv_batch(req.cache, &v, req.v_off, req.base, req.kv_dim, req.m)?;

    let attn = with_gpu(|gpu| {
        let out_len = req.m * q_dim;
        gpu.ensure_arenas(q_use.len() * 4, out_len * 4)?;
        gpu.stream
            .memcpy_htod(&q_use, &mut gpu.x_arena.slice_mut(0..q_use.len()))
            .ok()?;
        let group_usz = req.n_heads / req.n_kv_heads.max(1);
        let use_kv = false && attend_use_kv_fused(hd, group_usz);
        let f = if use_kv {
            gpu.func_owned("attend_gqa_batch_kv")?
        } else {
            gpu.func_owned("attend_gqa_batch")?
        };
        let k_off = req.k_off as u64;
        let v_off = req.v_off as u64;
        let kv_dim = req.kv_dim as u32;
        let head_dim = hd as u32;
        let n_q = req.n_heads as u32;
        let group = group_usz as u32;
        let m_u = req.m as u32;
        let base = req.base as u32;
        let scale = req.scale;
        let grid_x = if use_kv {
            req.n_kv_heads.max(1) as u32
        } else {
            n_q
        };
        let cfg = if use_kv {
            attend_gqa_kv_launch_cfg(grid_x, m_u, group, head_dim)
        } else {
            attend_gqa_launch_cfg(grid_x, m_u)
        };
        let qv = gpu.x_arena.slice(0..q_use.len());
        let mut ov = gpu.y_arena.slice_mut(0..out_len);
        unsafe {
            gpu.stream
                .launch_builder_pdl(&f)
                .arg(&qv)
                .arg(&req.cache.buf)
                .arg(&mut ov)
                .arg(&k_off)
                .arg(&v_off)
                .arg(&kv_dim)
                .arg(&head_dim)
                .arg(&n_q)
                .arg(&group)
                .arg(&m_u)
                .arg(&base)
                .arg(&scale)
                .launch_pdl(cfg)
        }
        .ok()?;
        gpu.sync()?;
        gpu.stream.clone_dtoh(&gpu.y_arena.slice(0..out_len)).ok()
    })?;

    let proj = matvec_batch(&[MatvecReq {
        ty: req.wo.0,
        w: req.wo.1,
        n_in: q_dim,
        n_out: hidden,
        x: &attn,
        m: req.m,
    }])?
    .remove(0);

    if let Some(f) = &req.fusion {
        return with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            gpu.ensure_arenas(proj.len() * 4, 4)?;
            gpu.stream
                .memcpy_htod(&proj, &mut gpu.y_arena.slice_mut(0..proj.len()))
                .ok()?;
            {
                let f_add = gpu.func_owned("residual_add")?;
                let n = (req.m * hidden) as u32;
                let cfg = CudaGpu::cfg_1d(n, 256);
                let mut a = gpu.pf_x.slice_mut(0..req.m * hidden);
                let b = gpu.y_arena.slice(0..proj.len());
                unsafe {
                    gpu.stream
                        .launch_builder_pdl(&f_add)
                        .arg(&mut a)
                        .arg(&b)
                        .arg(&n)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            let w = norm_f32_bytes(f.ffn_norm, hidden)?;
            rmsnorm_pf_x_into_hs(gpu, &w, hidden, req.m, req.eps)?;
            gpu.sync()?;
            if f.n_expert == 0 {
                return Some(Vec::new());
            }
            let hs = gpu
                .stream
                .clone_dtoh(&gpu.pf_hs.slice(0..req.m * hidden))
                .ok()?;
            run_matvec(gpu, GgmlType::F32, f.router, hidden, f.n_expert, &hs, req.m)
        });
    }
    Some(proj)
}

pub fn prefill_gdn_block(req: &PrefillGdnReq) -> Option<Vec<f32>> {
    if !is_attached() {
        return None;
    }
    if std::env::var_os("ALLPAKA_GDN_ZERO").is_some() {
        // Mirror CPU/Metal kill switch: zero branch, still honor fusion residual path.
        if let Some(f) = &req.fusion {
            return with_gpu(|gpu| {
                if !gpu.pf_active || gpu.pf_len < req.m * req.hidden {
                    return None;
                }
                let w = norm_f32_bytes(f.ffn_norm, req.hidden)?;
                rmsnorm_pf_x_into_hs(gpu, &w, req.hidden, req.m, req.eps)?;
                gpu.sync()?;
                if f.n_expert == 0 {
                    return Some(Vec::new());
                }
                let hs = gpu
                    .stream
                    .clone_dtoh(&gpu.pf_hs.slice(0..req.m * req.hidden))
                    .ok()?;
                run_matvec(
                    gpu,
                    GgmlType::F32,
                    f.router,
                    req.hidden,
                    f.n_expert,
                    &hs,
                    req.m,
                )
            });
        }
        return Some(vec![0f32; req.m * req.hidden]);
    }

    let hidden = req.hidden;
    let key_dim = req.heads_k * req.d;
    let value_dim = req.heads_v * req.d;
    let channels = key_dim * 2 + value_dim;
    if req.d != 128
        || req.d_conv < 2
        || req.wqkv.2 != channels
        || req.zgate.2 != value_dim
        || req.ssm_out.2 != hidden
        || req.m < MM_MIN_M
        || req.alpha.len() != hidden * req.heads_v * 4
        || req.beta.len() != hidden * req.heads_v * 4
        || req.conv1d.len() != channels * req.d_conv * 4
        || req.a.len() != req.heads_v
        || req.dt.len() != req.heads_v
        || req.ssm_norm.len() != req.d
    {
        return None;
    }
    let conv_span = req.conv_off + (req.d_conv - 1) * channels;
    let state_span = req.state_off + req.heads_v * req.d * req.d;
    if conv_span * 4 > req.ssm.len || state_span * 4 > req.ssm.len {
        return None;
    }
    let _ = with_gpu(|gpu| resolve_w(gpu, req.conv1d))?;

    let hs_owned = if let Some(f) = &req.fusion {
        with_gpu(|gpu| {
            if !gpu.pf_active || gpu.pf_len < req.m * hidden {
                return None;
            }
            let w = norm_f32_bytes(f.attn_norm, hidden)?;
            rmsnorm_pf_x_into_hs(gpu, &w, hidden, req.m, req.eps)?;
            gpu.sync()?;
            gpu.stream
                .clone_dtoh(&gpu.pf_hs.slice(0..req.m * hidden))
                .ok()
        })
    } else {
        None
    };
    // PrefillGdnReq has no host hs; fusion (attn-norm from pf_x) is required,
    // matching Metal's prefill_gdn_block contract.
    let hs = hs_owned.as_deref()?;

    let mut parts = matvec_batch(&[
        MatvecReq {
            ty: req.wqkv.0,
            w: req.wqkv.1,
            n_in: hidden,
            n_out: channels,
            x: hs,
            m: req.m,
        },
        MatvecReq {
            ty: req.zgate.0,
            w: req.zgate.1,
            n_in: hidden,
            n_out: value_dim,
            x: hs,
            m: req.m,
        },
        MatvecReq {
            ty: GgmlType::F32,
            w: req.alpha,
            n_in: hidden,
            n_out: req.heads_v,
            x: hs,
            m: req.m,
        },
        MatvecReq {
            ty: GgmlType::F32,
            w: req.beta,
            n_in: hidden,
            n_out: req.heads_v,
            x: hs,
            m: req.m,
        },
    ])?;
    let beta = parts.pop()?;
    let alpha = parts.pop()?;
    let z = parts.pop()?;
    let qkv = parts.pop()?;
    let conv_w = norm_f32_bytes(req.conv1d, channels * req.d_conv)?;

    let gy = with_gpu(|gpu| {
        let align = |n: usize| (n + 63) & !63;
        // x_arena: qkv | z | alpha | beta | a | dt | sn | cw | qkc | win
        // y_arena: temporary qkc during conv, then deltanet out
        let qkv_at = 0usize;
        let z_at = qkv_at + align(req.m * channels);
        let ab_a = z_at + align(req.m * value_dim);
        let ab_b = ab_a + align(req.m * req.heads_v);
        let a_at = ab_b + align(req.m * req.heads_v);
        let dt_at = a_at + align(req.heads_v);
        let sn_at = dt_at + align(req.heads_v);
        let cw_at = sn_at + align(req.d);
        let qkc_at = cw_at + align(conv_w.len());
        let win_at = qkc_at + align(req.m * channels);
        let window_rows = req.d_conv - 1;
        let x_need = win_at + align(window_rows * channels);
        let y_need = (req.m * channels).max(req.m * value_dim);
        gpu.ensure_arenas(x_need * 4, y_need * 4)?;

        let mut host = vec![0f32; x_need];
        host[qkv_at..qkv_at + qkv.len()].copy_from_slice(&qkv);
        host[z_at..z_at + z.len()].copy_from_slice(&z);
        host[ab_a..ab_a + alpha.len()].copy_from_slice(&alpha);
        host[ab_b..ab_b + beta.len()].copy_from_slice(&beta);
        host[a_at..a_at + req.a.len()].copy_from_slice(req.a);
        host[dt_at..dt_at + req.dt.len()].copy_from_slice(req.dt);
        host[sn_at..sn_at + req.ssm_norm.len()].copy_from_slice(req.ssm_norm);
        host[cw_at..cw_at + conv_w.len()].copy_from_slice(&conv_w);
        gpu.stream
            .memcpy_htod(&host, &mut gpu.x_arena.slice_mut(0..x_need))
            .ok()?;

        let channels_u = channels as u32;
        let d_conv_u = req.d_conv as u32;
        let m_u = req.m as u32;
        let conv_off = req.conv_off as u64;
        let state_off = req.state_off as u64;
        let heads_k = req.heads_k as u32;
        let heads_v = req.heads_v as u32;
        let d_u = req.d as u32;
        let key_dim_u = key_dim as u32;
        let slot_total = match req.ssm_slots {
            Some((_, total)) => total as u32,
            None => 0u32,
        };

        {
            let f = gpu.func_owned("gdn_conv_batch")?;
            let cfg = LaunchConfig {
                grid_dim: (channels_u.div_ceil(256), m_u, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let qkv_d = gpu.x_arena.slice(qkv_at..qkv_at + req.m * channels);
            let mut qkc = gpu.y_arena.slice_mut(0..req.m * channels);
            let w = gpu.x_arena.slice(cw_at..cw_at + conv_w.len());
            let slots_buf = match req.ssm_slots {
                Some((r, _)) => &r.buf,
                None => &req.ssm.buf,
            };
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&qkv_d)
                    .arg(&mut qkc)
                    .arg(&req.ssm.buf)
                    .arg(&w)
                    .arg(slots_buf)
                    .arg(&channels_u)
                    .arg(&d_conv_u)
                    .arg(&m_u)
                    .arg(&conv_off)
                    .arg(&slot_total)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        // Park conv output in x_arena so y can hold the recurrence out.
        {
            let f_copy = gpu.func_owned("copy_f32")?;
            let n = (req.m * channels) as u32;
            let cfg = CudaGpu::cfg_1d(n, 256);
            let src = gpu.y_arena.slice(0..req.m * channels);
            let mut dst = gpu.x_arena.slice_mut(qkc_at..qkc_at + req.m * channels);
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f_copy)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&n)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }

        // Session window := last d_conv-1 RAW qkv rows (or merged with prior window).
        {
            let f_store = gpu.func_owned("ssm_store_f32")?;
            let nwin = (window_rows * channels) as u32;
            let cfg = CudaGpu::cfg_1d(nwin, 256);
            if req.m >= window_rows {
                let src_at = qkv_at + (req.m - window_rows) * channels;
                let src = gpu.x_arena.slice(src_at..src_at + window_rows * channels);
                unsafe {
                    gpu.stream
                        .launch_builder_pdl(&f_store)
                        .arg(&req.ssm.buf)
                        .arg(&src)
                        .arg(&conv_off)
                        .arg(&nwin)
                        .launch_pdl(cfg)
                }
                .ok()?;
            } else {
                let f_load = gpu.func_owned("ssm_load_f32")?;
                let old = (window_rows - req.m) * channels;
                let old_u = old as u32;
                let load_off = (req.conv_off + req.m * channels) as u64;
                let cfg_old = CudaGpu::cfg_1d(old_u, 256);
                {
                    let mut dst = gpu.x_arena.slice_mut(win_at..win_at + old);
                    unsafe {
                        gpu.stream
                            .launch_builder_pdl(&f_load)
                            .arg(&mut dst)
                            .arg(&req.ssm.buf)
                            .arg(&load_off)
                            .arg(&old_u)
                            .launch_pdl(cfg_old)
                    }
                    .ok()?;
                }
                {
                    let f_copy = gpu.func_owned("copy_f32")?;
                    let new_n = (req.m * channels) as u32;
                    let cfg_new = CudaGpu::cfg_1d(new_n, 256);
                    let src = gpu.x_arena.slice(qkv_at..qkv_at + req.m * channels);
                    // Copy into a side buffer then merge — avoid aliasing x_arena.
                    let mut tmp = unsafe { gpu.stream.alloc::<f32>(req.m * channels) }.ok()?;
                    unsafe {
                        gpu.stream
                            .launch_builder_pdl(&f_copy)
                            .arg(&mut tmp)
                            .arg(&src)
                            .arg(&new_n)
                            .launch_pdl(cfg_new)
                    }
                    .ok()?;
                    let mut dst = gpu
                        .x_arena
                        .slice_mut(win_at + old..win_at + old + req.m * channels);
                    unsafe {
                        gpu.stream
                            .launch_builder_pdl(&f_copy)
                            .arg(&mut dst)
                            .arg(&tmp)
                            .arg(&new_n)
                            .launch_pdl(cfg_new)
                    }
                    .ok()?;
                }
                let src = gpu.x_arena.slice(win_at..win_at + window_rows * channels);
                unsafe {
                    gpu.stream
                        .launch_builder_pdl(&f_store)
                        .arg(&req.ssm.buf)
                        .arg(&src)
                        .arg(&conv_off)
                        .arg(&nwin)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
        }

        {
            let f = gpu.func_owned("gdn_step_batch")?;
            let cfg = LaunchConfig {
                grid_dim: (d_u / 4, heads_v, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            };
            let qkc = gpu.x_arena.slice(qkc_at..qkc_at + req.m * channels);
            let alpha_d = gpu.x_arena.slice(ab_a..ab_a + req.m * req.heads_v);
            let beta_d = gpu.x_arena.slice(ab_b..ab_b + req.m * req.heads_v);
            let a_log = gpu.x_arena.slice(a_at..a_at + req.heads_v);
            let dt_bias = gpu.x_arena.slice(dt_at..dt_at + req.heads_v);
            let mut out = gpu.y_arena.slice_mut(0..req.m * value_dim);
            let slots_buf = match req.ssm_slots {
                Some((r, _)) => &r.buf,
                None => &req.ssm.buf,
            };
            let eps = req.eps;
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&req.ssm.buf)
                    .arg(&qkc)
                    .arg(&alpha_d)
                    .arg(&beta_d)
                    .arg(&a_log)
                    .arg(&dt_bias)
                    .arg(&mut out)
                    .arg(slots_buf)
                    .arg(&heads_k)
                    .arg(&heads_v)
                    .arg(&d_u)
                    .arg(&key_dim_u)
                    .arg(&m_u)
                    .arg(&eps)
                    .arg(&state_off)
                    .arg(&slot_total)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        {
            let f = gpu.func_owned("gdn_out_norm_batch")?;
            let cfg = LaunchConfig {
                grid_dim: (heads_v, m_u, 1),
                block_dim: (d_u, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut y = gpu.y_arena.slice_mut(0..req.m * value_dim);
            let z_d = gpu.x_arena.slice(z_at..z_at + req.m * value_dim);
            let wn = gpu.x_arena.slice(sn_at..sn_at + req.d);
            let eps = req.eps;
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&mut y)
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
        gpu.stream
            .clone_dtoh(&gpu.y_arena.slice(0..req.m * value_dim))
            .ok()
    })?;

    let proj = matvec_batch(&[MatvecReq {
        ty: req.ssm_out.0,
        w: req.ssm_out.1,
        n_in: value_dim,
        n_out: hidden,
        x: &gy,
        m: req.m,
    }])?
    .remove(0);

    if let Some(f) = &req.fusion {
        return with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            gpu.ensure_arenas(proj.len() * 4, req.m * f.n_expert.max(1) * 4)?;
            gpu.stream
                .memcpy_htod(&proj, &mut gpu.y_arena.slice_mut(0..proj.len()))
                .ok()?;
            {
                let f_add = gpu.func_owned("residual_add")?;
                let n = (req.m * hidden) as u32;
                let cfg = CudaGpu::cfg_1d(n, 256);
                let mut a = gpu.pf_x.slice_mut(0..req.m * hidden);
                let b = gpu.y_arena.slice(0..proj.len());
                unsafe {
                    gpu.stream
                        .launch_builder_pdl(&f_add)
                        .arg(&mut a)
                        .arg(&b)
                        .arg(&n)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            let w = norm_f32_bytes(f.ffn_norm, hidden)?;
            rmsnorm_pf_x_into_hs(gpu, &w, hidden, req.m, req.eps)?;
            gpu.sync()?;
            if f.n_expert == 0 {
                return Some(Vec::new());
            }
            let hs = gpu
                .stream
                .clone_dtoh(&gpu.pf_hs.slice(0..req.m * hidden))
                .ok()?;
            run_matvec(gpu, GgmlType::F32, f.router, hidden, f.n_expert, &hs, req.m)
        });
    }
    Some(proj)
}
