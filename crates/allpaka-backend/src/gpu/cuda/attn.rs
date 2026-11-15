//! Attention on CUDA: single and batched attend, the fused decode attention
//! block, KV stores, and the small norm and scratch helpers they share.

use super::*;

pub(super) fn attn_shape_ok(req: &AttnReq) -> bool {
    matches!(req.head_dim, 64 | 128 | 256) && crate::gpu::common::attn_bounds_ok(req, req.cache.len)
}

pub(super) fn attend_gqa_launch_cfg(n_q_or_kv: u32, m: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n_q_or_kv, m, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn attend_gqa_kv_launch_cfg(
    n_kv: u32,
    m: u32,
    group: u32,
    head_dim: u32,
) -> LaunchConfig {
    // TILE=64 × head_dim × 2 (K+V) × sizeof(uint16)
    let smem = 64u32 * head_dim * 2 * 2;
    LaunchConfig {
        grid_dim: (n_kv, m, 1),
        block_dim: (32, group, 1),
        shared_mem_bytes: smem,
    }
}

pub(super) fn attend_use_kv_fused(head_dim: usize, group: usize) -> bool {
    group > 1 && group <= 8 && head_dim > 0 && head_dim <= 256
}

pub fn attend(req: &AttnReq) -> Option<Vec<f32>> {
    if !attn_shape_ok(req) {
        return None;
    }
    with_gpu(|gpu| {
        let out_len = req.n_q_heads * req.head_dim;
        gpu.ensure_arenas(req.q.len() * 4, out_len * 4)?;
        gpu.stream
            .memcpy_htod(req.q, &mut gpu.x_arena.slice_mut(0..req.q.len()))
            .ok()?;
        let f = if attend_use_kv_fused(req.head_dim, req.group) {
            gpu.func_owned("attend_gqa_kv")?
        } else {
            gpu.func_owned("attend_gqa")?
        };
        let k_off = req.k_off as u64;
        let v_off = req.v_off as u64;
        let kv_dim = req.kv_dim as u32;
        let head_dim = req.head_dim as u32;
        let n_q = req.n_q_heads as u32;
        let group = req.group as u32;
        let n_pos = req.n_pos as u32;
        let scale = req.scale;
        let grid_x = if attend_use_kv_fused(req.head_dim, req.group) {
            n_q / group
        } else {
            n_q
        };
        let cfg = if attend_use_kv_fused(req.head_dim, req.group) {
            attend_gqa_kv_launch_cfg(grid_x, 1, group, head_dim)
        } else {
            attend_gqa_launch_cfg(grid_x, 1)
        };
        let t0 = Instant::now();
        {
            let q = gpu.x_arena.slice(0..req.q.len());
            let mut out = gpu.y_arena.slice_mut(0..out_len);
            let cache = &req.cache.buf;
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&q)
                    .arg(cache)
                    .arg(&mut out)
                    .arg(&k_off)
                    .arg(&v_off)
                    .arg(&kv_dim)
                    .arg(&head_dim)
                    .arg(&n_q)
                    .arg(&group)
                    .arg(&n_pos)
                    .arg(&scale)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        let enc = t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();
        gpu.sync()?;
        note_call(1, enc, t1.elapsed().as_nanos() as u64);
        gpu.stream.clone_dtoh(&gpu.y_arena.slice(0..out_len)).ok()
    })
}

pub fn attend_batch(reqs: &[AttnReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::with_capacity(reqs.len());
    for r in reqs {
        out.push(attend(r)?);
    }
    Some(out)
}

pub fn attend_project(req: &AttnReq, wo_ty: GgmlType, wo: &[u8], n_out: usize) -> Option<Vec<f32>> {
    let attn = attend(req)?;
    matvec(wo_ty, wo, attn.len(), n_out, &attn)
}

pub fn attn_block(req: &AttnBlockReq) -> Option<Vec<f32>> {
    let hidden = req.x.len();
    let hd = req.head_dim;
    if !matches!(hd, 64 | 128 | 256) {
        return None;
    }
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    if req.wq.2 != q_dim || req.wk.2 != kv || req.wv.2 != kv || req.wo.2 != hidden {
        return None;
    }
    let mut q = matvec(req.wq.0, req.wq.1, hidden, q_dim, req.x)?;
    let mut k = matvec(req.wk.0, req.wk.1, hidden, kv, req.x)?;
    let v = matvec(req.wv.0, req.wv.1, hidden, kv, req.x)?;
    let rot_pairs = req.rope.len();
    let rot_dim = rot_pairs * 2;
    if rot_dim > hd {
        return None;
    }
    for h in 0..req.n_heads {
        let qh = &mut q[h * hd..(h + 1) * hd];
        if let Some(wn) = req.q_norm {
            crate::ops::rmsnorm(qh, wn, req.eps);
        }
        crate::ops::rope_neox_cached_from_array(&mut qh[..rot_dim], req.rope);
    }
    for h in 0..req.n_kv_heads {
        let kh = &mut k[h * hd..(h + 1) * hd];
        if let Some(wn) = req.k_norm {
            crate::ops::rmsnorm(kh, wn, req.eps);
        }
        crate::ops::rope_neox_cached_from_array(&mut kh[..rot_dim], req.rope);
    }
    store_kv(req.cache, &k, req.k_off, req.pos, req.kv_dim)?;
    store_kv(req.cache, &v, req.v_off, req.pos, req.kv_dim)?;
    let group = req.n_heads / req.n_kv_heads.max(1);
    let attn = attend(&AttnReq {
        cache: req.cache,
        k_off: req.k_off,
        v_off: req.v_off,
        q: &q,
        kv_dim: req.kv_dim,
        head_dim: hd,
        n_q_heads: req.n_heads,
        group,
        n_pos: req.pos + 1,
        scale: req.scale,
    })?;
    matvec(req.wo.0, req.wo.1, q_dim, hidden, &attn)
}

pub(super) fn store_kv(
    cache: &SharedRegion,
    src: &[f32],
    off: usize,
    pos: usize,
    kv_dim: usize,
) -> Option<()> {
    with_gpu(|gpu| {
        let f = gpu.func_owned("store_kv_f16")?;
        gpu.ensure_arenas(src.len() * 4, 4)?;
        gpu.stream
            .memcpy_htod(src, &mut gpu.x_arena.slice_mut(0..src.len()))
            .ok()?;
        let n = src.len() as u32;
        let elem_off = (off + pos * kv_dim) as u64;
        let cfg = CudaGpu::cfg_1d(n, 256);
        let xv = gpu.x_arena.slice(0..src.len());
        unsafe {
            gpu.stream
                .launch_builder_pdl(&f)
                .arg(&cache.buf)
                .arg(&xv)
                .arg(&elem_off)
                .arg(&n)
                .launch_pdl(cfg)
        }
        .ok()?;
        // Stream-ordered; callers that need host visibility sync themselves.
        Some(())
    })
}

pub(super) fn store_kv_batch(
    cache: &SharedRegion,
    src: &[f32],
    off: usize,
    pos0: usize,
    kv_dim: usize,
    m: usize,
) -> Option<()> {
    if src.len() != m * kv_dim {
        return None;
    }
    with_gpu(|gpu| {
        let f = gpu.func_owned("store_kv_batch_f16")?;
        gpu.ensure_arenas(src.len() * 4, 4)?;
        gpu.stream
            .memcpy_htod(src, &mut gpu.x_arena.slice_mut(0..src.len()))
            .ok()?;
        let kv_u = kv_dim as u32;
        let m_u = m as u32;
        let pos_u = pos0 as u32;
        let base_off = off as u64;
        let cfg = LaunchConfig {
            grid_dim: (kv_u.div_ceil(256), m_u, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let xv = gpu.x_arena.slice(0..src.len());
        unsafe {
            gpu.stream
                .launch_builder_pdl(&f)
                .arg(&cache.buf)
                .arg(&xv)
                .arg(&base_off)
                .arg(&kv_u)
                .arg(&m_u)
                .arg(&pos_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        Some(())
    })
}

pub(super) fn norm_f32_bytes(raw: &[u8], n: usize) -> Option<Vec<f32>> {
    if raw.len() < n * 4 {
        return None;
    }
    let mut v = vec![0f32; n];
    for (i, slot) in v.iter_mut().enumerate() {
        let o = i * 4;
        *slot = f32::from_le_bytes(raw[o..o + 4].try_into().ok()?);
    }
    Some(v)
}

/// Device address of a small persistent host f32 tensor (Q/K norm weights),
/// uploaded once. Keyed by host address with a content check so a freed and
/// reused address never serves stale values. No per-call H2D, so prefill
/// layers stop paying pageable copies and the address is graph-stable.
pub(super) fn dev_f32(gpu: &mut CudaGpu, key: usize, host: &[f32]) -> Option<u64> {
    let hit = matches!(gpu.f32_cache.get(&key), Some((h, _)) if h.as_slice() == host);
    if !hit {
        // An upload from pageable `host` must not end up in a prefill graph:
        // a replay would read that address again, long after it was freed.
        if gpu.pf_capturing {
            gpu.pf_capture_bad = true;
        }
        let mut d = gpu.stream.alloc_zeros::<f32>(host.len().max(1)).ok()?;
        gpu.stream
            .memcpy_htod(host, &mut d.slice_mut(0..host.len()))
            .ok()?;
        gpu.f32_cache.insert(key, (host.to_vec(), d));
    }
    let stream = Arc::clone(&gpu.stream);
    let (p, g) = DevicePtr::device_ptr(&gpu.f32_cache.get(&key)?.1, &stream);
    drop(g);
    Some(p)
}

/// Device address of an f32 norm weight given as raw little-endian bytes:
/// the resident weight chunk when the bytes live there, else [`dev_f32`].
/// Rows of `x` (device pointer, `m` rows of `n`) normalised in place:
/// x = rmsnorm(x) * w, `w` raw F32 GGUF bytes. Gemma's sandwich norms.
pub(super) fn rmsnorm_rows_dev(
    gpu: &mut CudaGpu,
    x: u64,
    w_raw: &[u8],
    n: usize,
    m: usize,
    eps: f32,
) -> Option<()> {
    let wp = dev_norm_bytes(gpu, w_raw, n)?;
    let f = gpu.func_owned("rmsnorm_f32")?;
    let n_u = n as u32;
    let rows = m as u32;
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&x)
            .arg(&wp)
            .arg(&n_u)
            .arg(&eps)
            .arg(&rows)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

pub(super) fn dev_norm_bytes(gpu: &mut CudaGpu, raw: &[u8], n: usize) -> Option<u64> {
    if raw.len() < n * 4 {
        return None;
    }
    let raw = &raw[..n * 4];
    if raw.as_ptr() as usize % 4 == 0 {
        if let Some((c, off)) = resolve_w(gpu, raw) {
            let stream = Arc::clone(&gpu.stream);
            let (p, g) = DevicePtr::device_ptr(&gpu.chunks[c].buf, &stream);
            drop(g);
            return Some(p + off);
        }
    }
    let w = norm_f32_bytes(raw, n)?;
    dev_f32(gpu, raw.as_ptr() as usize, &w)
}

pub(super) fn rmsnorm_pf_x_into_hs(
    gpu: &mut CudaGpu,
    w_raw: &[u8],
    hidden: usize,
    m: usize,
    eps: f32,
) -> Option<()> {
    match gpu.pf_pending_add {
        Some((add, len)) if len == m * hidden => {
            gpu.pf_pending_add = None;
            return add_rmsnorm_pf(gpu, add, w_raw, hidden, m, eps);
        }
        Some(_) => gpu.pf_flush_add()?,
        None => {}
    }
    let wp = dev_norm_bytes(gpu, w_raw, hidden)?;
    let f_rms = gpu.func_owned("rmsnorm_into_f32")?;
    let n = hidden as u32;
    let rows = m as u32;
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let src = gpu.pf_x.slice(0..m * hidden);
    let mut dst = gpu.pf_hs.slice_mut(0..m * hidden);
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f_rms)
            .arg(&mut dst)
            .arg(&src)
            .arg(&wp)
            .arg(&n)
            .arg(&eps)
            .arg(&rows)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

/// `pf_x += add` then `pf_hs = rmsnorm(pf_x) * w` in one launch. `add` is a
/// device address of `m * hidden` f32.
pub(super) fn add_rmsnorm_pf(
    gpu: &mut CudaGpu,
    add: u64,
    w_raw: &[u8],
    hidden: usize,
    m: usize,
    eps: f32,
) -> Option<()> {
    gpu.pf_flush_add()?;
    let wp = dev_norm_bytes(gpu, w_raw, hidden)?;
    let f = gpu.func_owned("add_rmsnorm_into_f32")?;
    let n = hidden as u32;
    let rows = m as u32;
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut x = gpu.pf_x.slice_mut(0..m * hidden);
    let mut dst = gpu.pf_hs.slice_mut(0..m * hidden);
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&mut dst)
            .arg(&mut x)
            .arg(&add)
            .arg(&wp)
            .arg(&n)
            .arg(&eps)
            .arg(&rows)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}
