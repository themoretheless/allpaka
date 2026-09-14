//! CUDA GPU backend for Windows/Linux (feature `cuda`).
//!
//! Public surface matches `stub.rs` / Metal exactly. Weights are copied into
//! VRAM at attach time; KV and other shared regions are H2D-wrapped buffers.

mod kernels;
mod runtime;

use allpaka_gguf::GgmlType;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use runtime::{
    launch_gemm_dequant, launch_matvec_y_ptr, note_call, resolve_w, run_matvec, try_init, with_gpu,
    CudaGpu, DECODE_ATTEMPTS, DECODE_DECLINES, DECODE_SUCCESSES, CALLS, DISPATCHES, ENCODE_NS, GPU,
    GPU_BUSY_NS, MM_MIN_M, SCHED_NS, WAIT_NS,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

pub fn attach(mapping: &[u8]) -> bool {
    if std::env::var_os("ALLPAKA_NO_GPU").is_some() {
        return false;
    }
    if try_init().is_none() {
        return false;
    }
    with_gpu(|gpu| {
        if gpu.covers(mapping.as_ptr() as usize, mapping.len()) {
            return Some(true);
        }
        Some(gpu.add_mapping(mapping))
    })
    .unwrap_or(false)
}

pub fn is_attached() -> bool {
    GPU.get().map_or(false, Option::is_some)
}

pub fn residency_status() -> (usize, bool) {
    let Some(Some(cell)) = GPU.get() else {
        return (0, false);
    };
    cell.lock()
        .map(|gpu| (gpu.chunks.len(), !gpu.chunks.is_empty()))
        .unwrap_or((0, false))
}

pub fn stats() -> (u64, u64, u64, u64) {
    (
        CALLS.load(Ordering::Relaxed),
        DISPATCHES.load(Ordering::Relaxed),
        ENCODE_NS.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed),
    )
}

pub fn gpu_time_stats() -> (u64, u64) {
    (
        GPU_BUSY_NS.load(Ordering::Relaxed),
        SCHED_NS.load(Ordering::Relaxed),
    )
}

pub struct MatvecReq<'a> {
    pub ty: GgmlType,
    pub w: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
    pub x: &'a [f32],
    pub m: usize,
}

pub fn matvec(ty: GgmlType, w: &[u8], n_in: usize, n_out: usize, x: &[f32]) -> Option<Vec<f32>> {
    matvec_batch(&[MatvecReq {
        ty,
        w,
        n_in,
        n_out,
        x,
        m: 1,
    }])
    .map(|mut v| v.pop().unwrap())
}

pub fn matvec_batch(reqs: &[MatvecReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    // Fast path: one H2D of x, packed GEMMs, single sync, then D2H.
    let same_x = reqs.iter().all(|r| {
        r.m == reqs[0].m
            && r.n_in == reqs[0].n_in
            && r.x.len() == reqs[0].x.len()
            && std::ptr::eq(r.x.as_ptr(), reqs[0].x.as_ptr())
    });
    let all_gemm = reqs
        .iter()
        .all(|r| r.m >= MM_MIN_M && runtime::dq_fmt(r.ty).is_some());
    if same_x && all_gemm && reqs.len() > 1 {
        return with_gpu(|gpu| {
            let x = reqs[0].x;
            let mut y_off = 0usize;
            let mut offs = Vec::with_capacity(reqs.len());
            let t0 = Instant::now();
            for (i, r) in reqs.iter().enumerate() {
                let (chunk, w_off) = resolve_w(gpu, r.w)?;
                launch_gemm_dequant(
                    gpu,
                    r.ty,
                    chunk,
                    w_off,
                    r.n_in,
                    r.n_out,
                    r.m,
                    i > 0,
                    x,
                    y_off,
                    i > 0,
                    false,
                )?;
                offs.push((y_off, r.m * r.n_out));
                y_off += r.m * r.n_out;
            }
            let enc = t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            gpu.sync()?;
            note_call(reqs.len() as u64 * 2, enc, t1.elapsed().as_nanos() as u64);
            let mut out = Vec::with_capacity(reqs.len());
            for &(off, len) in &offs {
                out.push(
                    gpu.stream
                        .clone_dtoh(&gpu.y_arena.slice(off..off + len))
                        .ok()?,
                );
            }
            Some(out)
        });
    }
    with_gpu(|gpu| {
        let mut out = Vec::with_capacity(reqs.len());
        for r in reqs {
            out.push(run_matvec(gpu, r.ty, r.w, r.n_in, r.n_out, r.x, r.m)?);
        }
        Some(out)
    })
}

pub struct FfnReq<'a> {
    pub gate_ty: GgmlType,
    pub gate_w: &'a [u8],
    pub up_ty: GgmlType,
    pub up_w: &'a [u8],
    pub down_ty: GgmlType,
    pub down_w: &'a [u8],
    pub hidden: usize,
    pub ffn: usize,
    pub x: &'a [f32],
    pub m: usize,
}

pub fn ffn_batch(reqs: &[FfnReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    with_gpu(|gpu| {
        let mut outs = Vec::with_capacity(reqs.len());
        for r in reqs {
            if r.x.len() != r.m * r.hidden {
                return None;
            }
            let gate = run_matvec(gpu, r.gate_ty, r.gate_w, r.hidden, r.ffn, r.x, r.m)?;
            let up = run_matvec(gpu, r.up_ty, r.up_w, r.hidden, r.ffn, r.x, r.m)?;
            gpu.ensure_arenas(r.m * r.ffn * 4, r.m * r.ffn * 4)?;
            gpu.stream
                .memcpy_htod(&gate, &mut gpu.x_arena.slice_mut(0..gate.len()))
                .ok()?;
            gpu.stream
                .memcpy_htod(&up, &mut gpu.y_arena.slice_mut(0..up.len()))
                .ok()?;
            {
                let f = gpu.func_owned("swiglu")?;
                let n = gate.len() as u32;
                let cfg = CudaGpu::cfg_1d(n, 256);
                let mut g = gpu.x_arena.slice_mut(0..gate.len());
                let u = gpu.y_arena.slice(0..up.len());
                unsafe {
                    gpu.stream
                        .launch_builder(&f)
                        .arg(&mut g)
                        .arg(&u)
                        .arg(&n)
                        .launch(cfg)
                }
                .ok()?;
            }
            gpu.sync()?;
            let act = gpu
                .stream
                .clone_dtoh(&gpu.x_arena.slice(0..gate.len()))
                .ok()?;
            let down = run_matvec(gpu, r.down_ty, r.down_w, r.ffn, r.hidden, &act, r.m)?;
            outs.push(down);
        }
        Some(outs)
    })
}

pub struct SharedRegion {
    buf: CudaSlice<u8>,
    pub(crate) len: usize,
    #[allow(dead_code)]
    host: usize,
}

unsafe impl Send for SharedRegion {}
unsafe impl Sync for SharedRegion {}

pub fn wrap_region(region: &[u8]) -> Option<SharedRegion> {
    const PAGE: usize = 16384;
    if region.is_empty() || region.as_ptr() as usize % PAGE != 0 || region.len() % PAGE != 0 {
        return None;
    }
    with_gpu(|gpu| {
        let mut buf = unsafe { gpu.stream.alloc::<u8>(region.len()) }.ok()?;
        gpu.stream.memcpy_htod(region, &mut buf).ok()?;
        Some(SharedRegion {
            buf,
            len: region.len(),
            host: region.as_ptr() as usize,
        })
    })
}

pub struct AttnReq<'a> {
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub q: &'a [f32],
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub group: usize,
    pub n_pos: usize,
    pub scale: f32,
}

fn attn_shape_ok(req: &AttnReq) -> bool {
    matches!(req.head_dim, 64 | 128 | 256)
        && req.group > 0
        && req.q.len() == req.n_q_heads * req.head_dim
        && req.n_pos > 0
        && (req.k_off + req.n_pos * req.kv_dim) * 2 <= req.cache.len
        && (req.v_off + req.n_pos * req.kv_dim) * 2 <= req.cache.len
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
        let f = gpu.func_owned("attend_gqa")?;
        let k_off = req.k_off as u64;
        let v_off = req.v_off as u64;
        let kv_dim = req.kv_dim as u32;
        let head_dim = req.head_dim as u32;
        let n_q = req.n_q_heads as u32;
        let group = req.group as u32;
        let n_pos = req.n_pos as u32;
        let scale = req.scale;
        let cfg = LaunchConfig {
            grid_dim: (n_q, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let t0 = Instant::now();
        {
            let q = gpu.x_arena.slice(0..req.q.len());
            let mut out = gpu.y_arena.slice_mut(0..out_len);
            let cache = &req.cache.buf;
            unsafe {
                gpu.stream
                    .launch_builder(&f)
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
                    .launch(cfg)
            }
            .ok()?;
        }
        let enc = t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();
        gpu.sync()?;
        note_call(1, enc, t1.elapsed().as_nanos() as u64);
        gpu.stream
            .clone_dtoh(&gpu.y_arena.slice(0..out_len))
            .ok()
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

pub fn attend_project(
    req: &AttnReq,
    wo_ty: GgmlType,
    wo: &[u8],
    n_out: usize,
) -> Option<Vec<f32>> {
    let attn = attend(req)?;
    matvec(wo_ty, wo, attn.len(), n_out, &attn)
}

pub struct AttnBlockReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    pub x: &'a [f32],
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub rope: &'a [[f32; 2]],
    pub eps: f32,
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub pos: usize,
    pub scale: f32,
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

fn store_kv(cache: &SharedRegion, src: &[f32], off: usize, pos: usize, kv_dim: usize) -> Option<()> {
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
                .launch_builder(&f)
                .arg(&cache.buf)
                .arg(&xv)
                .arg(&elem_off)
                .arg(&n)
                .launch(cfg)
        }
        .ok()?;
        // Stream-ordered; callers that need host visibility sync themselves.
        Some(())
    })
}

fn store_kv_batch(
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
                .launch_builder(&f)
                .arg(&cache.buf)
                .arg(&xv)
                .arg(&base_off)
                .arg(&kv_u)
                .arg(&m_u)
                .arg(&pos_u)
                .launch(cfg)
        }
        .ok()?;
        Some(())
    })
}

fn norm_f32_bytes(raw: &[u8], n: usize) -> Option<Vec<f32>> {
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

/// Upload RMSNorm weights into persistent VRAM scratch (no per-layer alloc).
fn upload_w_scratch(gpu: &mut CudaGpu, w: &[f32]) -> Option<u64> {
    gpu.ensure_w_scratch(w.len())?;
    gpu.stream
        .memcpy_htod(w, &mut gpu.w_scratch.slice_mut(0..w.len()))
        .ok()?;
    let stream = Arc::clone(&gpu.stream);
    let (p, g) = DevicePtr::device_ptr(&gpu.w_scratch, &stream);
    drop(g);
    Some(p)
}

fn rmsnorm_pf_x_into_hs(gpu: &mut CudaGpu, w: &[f32], hidden: usize, m: usize, eps: f32) -> Option<()> {
    let wp = upload_w_scratch(gpu, w)?;
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
            .launch_builder(&f_rms)
            .arg(&mut dst)
            .arg(&src)
            .arg(&wp)
            .arg(&n)
            .arg(&eps)
            .arg(&rows)
            .launch(cfg)
    }
    .ok()?;
    Some(())
}

pub struct TokenGdn<'a> {
    pub wqkv: (GgmlType, &'a [u8], usize),
    pub zgate: (GgmlType, &'a [u8], usize),
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    pub conv1d: &'a [u8],
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    pub conv_off: usize,
    pub state_off: usize,
}

pub struct TokenLayer<'a> {
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub gate_in_q: bool,
    pub q_bias: Option<&'a [u8]>,
    pub k_bias: Option<&'a [u8]>,
    pub v_bias: Option<&'a [u8]>,
    pub k_off: usize,
    pub v_off: usize,
    pub gdn: Option<TokenGdn<'a>>,
    pub ffn: TokenFfn<'a>,
}

pub enum TokenFfn<'a> {
    Dense {
        gate: (GgmlType, &'a [u8], usize),
        up: (GgmlType, &'a [u8], usize),
        down: (GgmlType, &'a [u8], usize),
    },
    Moe {
        router: (GgmlType, &'a [u8], usize),
        router_bias: Option<&'a [u8]>,
        gate: (GgmlType, &'a [u8]),
        up: (GgmlType, &'a [u8]),
        down: (GgmlType, &'a [u8]),
        expert_ffn: usize,
        n_used: usize,
        sigmoid: bool,
        shared: Option<[(GgmlType, &'a [u8], usize); 3]>,
        shared_gate: Option<&'a [u8]>,
    },
}

pub struct TokenReq<'a> {
    pub x: &'a [f32],
    pub m: usize,
    pub layers: &'a [TokenLayer<'a>],
    pub cache: &'a SharedRegion,
    pub ssm: Option<&'a SharedRegion>,
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub pos: usize,
    pub scale: f32,
    pub rope: &'a [[f32; 2]],
    pub rot_dim: usize,
    pub eps: f32,
    pub output_norm: &'a [u8],
    pub output: (GgmlType, &'a [u8], usize),
    pub argmax: bool,
}

pub enum TokenOut {
    Logits(Vec<f32>),
    Argmax(u32),
    Rows { argmax: Vec<u32>, hidden: Vec<f32> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeDecline {
    pub stage: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for DecodeDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stage={} reason={}", self.stage, self.reason)
    }
}

impl std::error::Error for DecodeDecline {}

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodePathStats {
    pub attempts: u64,
    pub successes: u64,
    pub declines: u64,
}

pub fn decode_path_stats() -> DecodePathStats {
    DecodePathStats {
        attempts: DECODE_ATTEMPTS.load(Ordering::Relaxed),
        successes: DECODE_SUCCESSES.load(Ordering::Relaxed),
        declines: DECODE_DECLINES.load(Ordering::Relaxed),
    }
}

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

pub fn decode_token_outcome(req: &TokenReq) -> crate::accel::AccelOutcome<TokenOut> {
    match decode_token_checked(req) {
        Ok(out) => crate::accel::AccelOutcome::Executed(out),
        Err(reason) => crate::accel::AccelOutcome::Declined(crate::accel::DeclineReason::Backend {
            operation: "decode-token",
            detail: reason.to_string(),
        }),
    }
}

pub fn decode_token(req: &TokenReq) -> Option<TokenOut> {
    if !is_attached() {
        return None;
    }
    let m = req.m;
    if m == 0 || m > 8 || req.x.len() % m != 0 {
        return None;
    }
    let hidden = req.x.len() / m;
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
    if req.rope.len() != m * req.rot_dim / 2 {
        return None;
    }
    if m > 1 {
        let mut argmax = Vec::with_capacity(m);
        let mut hidden_out = Vec::with_capacity(m * hidden);
        for row in 0..m {
            let xrow = &req.x[row * hidden..(row + 1) * hidden];
            let rope = &req.rope[row * req.rot_dim / 2..(row + 1) * req.rot_dim / 2];
            let out = decode_token_one(req, xrow, rope, req.pos + row)?;
            match out {
                TokenOut::Argmax(a) => {
                    argmax.push(a);
                    // Recompute final hidden by running without argmax is expensive;
                    // store the residual estimate from a dedicated path.
                    hidden_out.extend_from_slice(xrow);
                }
                TokenOut::Logits(l) => {
                    let a = l
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0);
                    argmax.push(a);
                    hidden_out.extend_from_slice(xrow);
                }
                TokenOut::Rows { argmax: a, hidden: h } => {
                    argmax.push(a[0]);
                    hidden_out.extend_from_slice(&h);
                }
            }
        }
        return Some(TokenOut::Rows {
            argmax,
            hidden: hidden_out,
        });
    }
    decode_token_one(req, req.x, req.rope, req.pos)
}
/// One GDN decode step: wqkv/z/α/β matvecs, depthwise conv + deltanet +
/// gated out-norm on the SSM region, then ssm_out projection.
fn gdn_decode_layer(
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
                    .launch_builder(&f)
                    .arg(&mut qkv_d)
                    .arg(&ssm.buf)
                    .arg(&w)
                    .arg(&channels_u)
                    .arg(&d_conv_u)
                    .arg(&conv_off)
                    .launch(cfg)
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
                    .launch_builder(&f)
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
                    .launch(cfg)
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
                        .launch_builder(&f_n)
                        .arg(&mut out_buf)
                        .arg(&z_d)
                        .arg(&wn)
                        .arg(&heads_v)
                        .arg(&d_u)
                        .arg(&eps)
                        .launch(cfg)
                }
                .ok()?;
            }
            gpu.sync()?;
            return gpu.stream.clone_dtoh(&out_buf).ok();
        }
    })?;

    matvec(g.ssm_out.0, g.ssm_out.1, value_dim, hidden, &gy)
}

fn fuse_decode_eligible(req: &TokenReq) -> bool {
    if std::env::var_os("ALLPAKA_CUDA_NO_FUSE").is_some() {
        return false;
    }
    for layer in req.layers {
        if layer.gdn.is_some()
            || layer.gate_in_q
            || layer.q_bias.is_some()
            || layer.k_bias.is_some()
            || layer.v_bias.is_some()
        {
            return false;
        }
        if !matches!(layer.ffn, TokenFfn::Dense { .. }) {
            return false;
        }
    }
    true
}

fn align64(n: usize) -> usize {
    (n + 63) & !63
}

fn rope_replicated(rope: &[[f32; 2]], n_heads: usize) -> Vec<f32> {
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

/// Whole-token dense decode on one stream: activations stay in `y_arena`, one sync.
fn decode_token_one_fused(
    req: &TokenReq,
    x_in: &[f32],
    rope_in: &[[f32; 2]],
    pos: usize,
) -> Option<TokenOut> {
    let hidden = x_in.len();
    let hd = req.head_dim;
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let group = req.n_heads / req.n_kv_heads.max(1);
    let rot_dim = req.rot_dim.min(hd);
    let rope_pairs = rot_dim / 2;
    if rope_in.len() < rope_pairs {
        return None;
    }
    let rope = &rope_in[..rope_pairs];

    let mut ffn_dim = 0usize;
    let mut wq_max = q_dim;
    for layer in req.layers {
        if layer.wq.2 != q_dim || layer.wk.2 != kv || layer.wv.2 != kv || layer.wo.2 != hidden {
            return None;
        }
        wq_max = wq_max.max(layer.wq.2);
        if let TokenFfn::Dense { gate, up, down } = &layer.ffn {
            if gate.2 != up.2 || down.2 != hidden {
                return None;
            }
            ffn_dim = ffn_dim.max(gate.2);
        } else {
            return None;
        }
    }
    let vocab = req.output.2;

    // Arena layout (f32 elems), Metal-style dense subset.
    let x_at = 0usize;
    let delta_at = x_at + align64(hidden);
    let h_at = delta_at + align64(hidden);
    let q_at = h_at + align64(hidden);
    let k_at = q_at + align64(wq_max);
    let v_at = k_at + align64(kv);
    let attn_at = v_at + align64(kv);
    let gate_at = attn_at + align64(q_dim);
    let up_at = gate_at + align64(ffn_dim);
    let out_logits_at = up_at + align64(ffn_dim);
    let amax_at = out_logits_at + align64(vocab);
    let y_elems = amax_at + align64(2);

    // x_arena: rope + output norm + per-layer norms (stable for CUDA graph).
    // Per layer: attn_norm[H], ffn_norm[H], q_norm[hd], k_norm[hd].
    let rope_q_at = 0usize;
    let rope_k_at = rope_q_at + req.n_heads * rope_pairs * 2;
    let out_norm_at = rope_k_at + req.n_kv_heads * rope_pairs * 2;
    let norms_base = out_norm_at + hidden;
    let per_layer_norms = 2 * hidden + 2 * hd;
    let x_elems = norms_base + req.layers.len() * per_layer_norms;

    let rope_q = rope_replicated(rope, req.n_heads);
    let rope_k = rope_replicated(rope, req.n_kv_heads);
    let out_norm = norm_f32_bytes(req.output_norm, hidden)?;

    let graph_key = {
        let mut h = 0xcbf29ce484222325u64;
        for &v in &[
            req.layers.len() as u64,
            hidden as u64,
            q_dim as u64,
            kv as u64,
            vocab as u64,
            ffn_dim as u64,
            req.kv_dim as u64,
            hd as u64,
        ] {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    };
    let use_graph = std::env::var("ALLPAKA_CUDA_GRAPH").map_or(false, |v| v == "1");

    with_gpu(|gpu| {
        gpu.ensure_arenas(x_elems * 4, y_elems * 4)?;
        let stream = Arc::clone(&gpu.stream);

        stream
            .memcpy_htod(x_in, &mut gpu.y_arena.slice_mut(x_at..x_at + hidden))
            .ok()?;
        stream
            .memcpy_htod(&rope_q, &mut gpu.x_arena.slice_mut(rope_q_at..rope_q_at + rope_q.len()))
            .ok()?;
        stream
            .memcpy_htod(&rope_k, &mut gpu.x_arena.slice_mut(rope_k_at..rope_k_at + rope_k.len()))
            .ok()?;
        stream
            .memcpy_htod(&[pos as u32], &mut gpu.d_pos)
            .ok()?;

        if gpu.decode_norms_key != graph_key {
            stream
                .memcpy_htod(
                    &out_norm,
                    &mut gpu.x_arena.slice_mut(out_norm_at..out_norm_at + hidden),
                )
                .ok()?;
            for (li, layer) in req.layers.iter().enumerate() {
                let base = norms_base + li * per_layer_norms;
                let attn_w = norm_f32_bytes(layer.attn_norm, hidden)?;
                let ffn_w = norm_f32_bytes(layer.ffn_norm, hidden)?;
                stream
                    .memcpy_htod(&attn_w, &mut gpu.x_arena.slice_mut(base..base + hidden))
                    .ok()?;
                stream
                    .memcpy_htod(
                        &ffn_w,
                        &mut gpu.x_arena.slice_mut(base + hidden..base + 2 * hidden),
                    )
                    .ok()?;
                if let Some(wn) = layer.q_norm {
                    if wn.len() < hd {
                        return None;
                    }
                    stream
                        .memcpy_htod(
                            &wn[..hd],
                            &mut gpu.x_arena
                                .slice_mut(base + 2 * hidden..base + 2 * hidden + hd),
                        )
                        .ok()?;
                }
                if let Some(wn) = layer.k_norm {
                    if wn.len() < hd {
                        return None;
                    }
                    stream
                        .memcpy_htod(
                            &wn[..hd],
                            &mut gpu.x_arena.slice_mut(
                                base + 2 * hidden + hd..base + 2 * hidden + 2 * hd,
                            ),
                        )
                        .ok()?;
                }
            }
            gpu.decode_norms_key = graph_key;
            gpu.decode_graph = None;
            gpu.decode_graph_key = 0;
        }

        let t0 = Instant::now();
        let mut launches = 0u64;

        // Arena device addresses are stable until ensure_arenas reallocates
        // (already done above). Drop SyncOnDrop immediately so we can mutably
        // borrow `gpu` for launches; single-stream mode does not event-track.
        let ybase = {
            let (p, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(g);
            p
        };
        let xbase = {
            let (p, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
            drop(g);
            p
        };

        let capturing = use_graph
            && (gpu.decode_graph.is_none() || gpu.decode_graph_key != graph_key);
        if capturing {
            gpu.decode_graph = None;
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .ok()?;
        } else if use_graph {
            if let Some(g) = gpu.decode_graph.as_ref() {
                g.launch().ok()?;
                launches = 1;
                let enc = t0.elapsed().as_nanos() as u64;
                let t1 = Instant::now();
                gpu.sync()?;
                note_call(launches, enc, t1.elapsed().as_nanos() as u64);
                let out = if req.argmax {
                    let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
                    TokenOut::Argmax(idx[0])
                } else {
                    let logits = stream
                        .clone_dtoh(&gpu.y_arena.slice(out_logits_at..out_logits_at + vocab))
                        .ok()?;
                    TokenOut::Logits(logits)
                };
                return Some(out);
            }
        }

        // Always end_capture on exit if we began — leave the stream clean even
        // when a mid-token launch fails and we fall back.
        struct CaptureGuard<'a> {
            stream: &'a Arc<cudarc::driver::CudaStream>,
            active: bool,
        }
        impl Drop for CaptureGuard<'_> {
            fn drop(&mut self) {
                if self.active {
                    let _ = self.stream.end_capture(
                        CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD,
                    );
                }
            }
        }
        let mut capture_guard = CaptureGuard {
            stream: &stream,
            active: capturing,
        };

        let launch_rms_into = |gpu: &mut CudaGpu,
                               dst: usize,
                               src: usize,
                               w_off: usize,
                               n: usize,
                               rows: usize|
         -> Option<()> {
            let f = gpu.func_owned("rmsnorm_into_f32")?;
            let n_u = n as u32;
            let rows_u = rows as u32;
            let eps = req.eps;
            let cfg = LaunchConfig {
                grid_dim: (rows_u, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let dst_p = ybase + (dst * 4) as u64;
            let src_p = ybase + (src * 4) as u64;
            let w_p = xbase + (w_off * 4) as u64;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&dst_p)
                    .arg(&src_p)
                    .arg(&w_p)
                    .arg(&n_u)
                    .arg(&eps)
                    .arg(&rows_u)
                    .launch(cfg)
            }
            .ok()?;
            Some(())
        };

        let launch_rms_inplace = |gpu: &mut CudaGpu,
                                  x_off: usize,
                                  w_off: usize,
                                  n: usize,
                                  rows: usize|
         -> Option<()> {
            let f = gpu.func_owned("rmsnorm_f32")?;
            let n_u = n as u32;
            let rows_u = rows as u32;
            let eps = req.eps;
            let cfg = LaunchConfig {
                grid_dim: (rows_u, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let xp = ybase + (x_off * 4) as u64;
            let wp = xbase + (w_off * 4) as u64;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&xp)
                    .arg(&wp)
                    .arg(&n_u)
                    .arg(&eps)
                    .arg(&rows_u)
                    .launch(cfg)
            }
            .ok()?;
            Some(())
        };

        let _launch_add = |gpu: &mut CudaGpu, a: usize, b: usize, n: usize| -> Option<()> {
            let f = gpu.func_owned("residual_add")?;
            let n_u = n as u32;
            let cfg = CudaGpu::cfg_1d(n_u, 256);
            let ap = ybase + (a * 4) as u64;
            let bp = ybase + (b * 4) as u64;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&ap)
                    .arg(&bp)
                    .arg(&n_u)
                    .launch(cfg)
            }
            .ok()?;
            Some(())
        };

        let launch_swiglu = |gpu: &mut CudaGpu, gate: usize, up: usize, n: usize| -> Option<()> {
            let f = gpu.func_owned("swiglu")?;
            let n_u = n as u32;
            let cfg = CudaGpu::cfg_1d(n_u, 256);
            let gp = ybase + (gate * 4) as u64;
            let up_p = ybase + (up * 4) as u64;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&gp)
                    .arg(&up_p)
                    .arg(&n_u)
                    .launch(cfg)
            }
            .ok()?;
            Some(())
        };

        let launch_rope = |gpu: &mut CudaGpu,
                           x_off: usize,
                           rope_off: usize,
                           heads: usize|
         -> Option<()> {
            let f = gpu.func_owned("rope_neox")?;
            let heads_u = heads as u32;
            let hd_u = hd as u32;
            let rot_u = rot_dim as u32;
            let cfg = LaunchConfig {
                grid_dim: (heads_u, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let xp = ybase + (x_off * 4) as u64;
            let rp = xbase + (rope_off * 4) as u64;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&xp)
                    .arg(&rp)
                    .arg(&heads_u)
                    .arg(&hd_u)
                    .arg(&rot_u)
                    .launch(cfg)
            }
            .ok()?;
            Some(())
        };

        let launch_store = |gpu: &mut CudaGpu, src: usize, off: usize, n: usize| -> Option<()> {
            let f = gpu.func_owned("store_kv_f16_dpos")?;
            let n_u = n as u32;
            let base_off = off as u64;
            let kv_dim_u = req.kv_dim as u32;
            let cfg = CudaGpu::cfg_1d(n_u, 256);
            let sp = ybase + (src * 4) as u64;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&req.cache.buf)
                    .arg(&sp)
                    .arg(&base_off)
                    .arg(&gpu.d_pos)
                    .arg(&kv_dim_u)
                    .arg(&n_u)
                    .launch(cfg)
            }
            .ok()?;
            Some(())
        };

        let launch_attend =
            |gpu: &mut CudaGpu, q_off: usize, out_off: usize, k_off: usize, v_off: usize| -> Option<()> {
                let f = gpu.func_owned("attend_gqa_dpos")?;
                let k_off_u = k_off as u64;
                let v_off_u = v_off as u64;
                let kv_dim_u = req.kv_dim as u32;
                let head_dim_u = hd as u32;
                let n_q = req.n_heads as u32;
                let group_u = group as u32;
                let scale = req.scale;
                let cfg = LaunchConfig {
                    grid_dim: (n_q, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let qp = ybase + (q_off * 4) as u64;
                let op = ybase + (out_off * 4) as u64;
                unsafe {
                    stream
                        .launch_builder(&f)
                        .arg(&qp)
                        .arg(&req.cache.buf)
                        .arg(&op)
                        .arg(&k_off_u)
                        .arg(&v_off_u)
                        .arg(&kv_dim_u)
                        .arg(&head_dim_u)
                        .arg(&n_q)
                        .arg(&group_u)
                        .arg(&gpu.d_pos)
                        .arg(&scale)
                        .launch(cfg)
                }
                .ok()?;
                Some(())
            };

        for (li, layer) in req.layers.iter().enumerate() {
            let layer_base = norms_base + li * per_layer_norms;
            let attn_norm_at = layer_base;
            let ffn_norm_at = layer_base + hidden;
            let q_norm_at = layer_base + 2 * hidden;
            let k_norm_at = q_norm_at + hd;

            launch_rms_into(gpu, h_at, x_at, attn_norm_at, hidden, 1)?;
            gpu.invalidate_q8();
            launches += 1;

            let (wq_c, wq_o) = resolve_w(gpu, layer.wq.1)?;
            let (wk_c, wk_o) = resolve_w(gpu, layer.wk.1)?;
            let (wv_c, wv_o) = resolve_w(gpu, layer.wv.1)?;
            let (wo_c, wo_o) = resolve_w(gpu, layer.wo.1)?;
            launch_matvec_y_ptr(gpu, layer.wq.0, wq_c, wq_o, hidden, q_dim, h_at, q_at, 1, ybase, false)?;
            launch_matvec_y_ptr(gpu, layer.wk.0, wk_c, wk_o, hidden, kv, h_at, k_at, 1, ybase, false)?;
            launch_matvec_y_ptr(gpu, layer.wv.0, wv_c, wv_o, hidden, kv, h_at, v_at, 1, ybase, false)?;
            launches += 3;

            if layer.q_norm.is_some() {
                launch_rms_inplace(gpu, q_at, q_norm_at, hd, req.n_heads)?;
                launches += 1;
            }
            launch_rope(gpu, q_at, rope_q_at, req.n_heads)?;
            launches += 1;

            if layer.k_norm.is_some() {
                launch_rms_inplace(gpu, k_at, k_norm_at, hd, req.n_kv_heads)?;
                launches += 1;
            }
            launch_rope(gpu, k_at, rope_k_at, req.n_kv_heads)?;
            launches += 1;

            launch_store(gpu, k_at, layer.k_off, kv)?;
            launch_store(gpu, v_at, layer.v_off, kv)?;
            launch_attend(gpu, q_at, attn_at, layer.k_off, layer.v_off)?;
            launches += 3;

            launch_matvec_y_ptr(gpu, layer.wo.0, wo_c, wo_o, q_dim, hidden, attn_at, x_at, 1, ybase, true)?;
            launches += 1;

            launch_rms_into(gpu, h_at, x_at, ffn_norm_at, hidden, 1)?;
            gpu.invalidate_q8();
            launches += 1;

            let TokenFfn::Dense { gate, up, down } = &layer.ffn else {
                return None;
            };
            let (g_c, g_o) = resolve_w(gpu, gate.1)?;
            let (u_c, u_o) = resolve_w(gpu, up.1)?;
            let (d_c, d_o) = resolve_w(gpu, down.1)?;
            launch_matvec_y_ptr(gpu, gate.0, g_c, g_o, hidden, gate.2, h_at, gate_at, 1, ybase, false)?;
            launch_matvec_y_ptr(gpu, up.0, u_c, u_o, hidden, up.2, h_at, up_at, 1, ybase, false)?;
            launch_swiglu(gpu, gate_at, up_at, gate.2)?;
            gpu.invalidate_q8();
            launch_matvec_y_ptr(gpu, down.0, d_c, d_o, gate.2, hidden, gate_at, x_at, 1, ybase, true)?;
            launches += 4;
        }

        // Final norm + LM head.
        launch_rms_into(gpu, h_at, x_at, out_norm_at, hidden, 1)?;
        gpu.invalidate_q8();
        let (o_c, o_o) = resolve_w(gpu, req.output.1)?;
        launch_matvec_y_ptr(
            gpu,
            req.output.0,
            o_c,
            o_o,
            hidden,
            vocab,
            h_at,
            out_logits_at,
            1,
            ybase,
            false,
        )?;
        launches += 2;

        let out = if req.argmax {
            let f = gpu.func_owned("argmax_f32")?;
            let n_u = vocab as u32;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let xp = ybase + (out_logits_at * 4) as u64;
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&xp)
                    .arg(&n_u)
                    .arg(&mut gpu.d_argmax)
                    .launch(cfg)
            }
            .ok()?;
            launches += 1;
            if capturing {
                capture_guard.active = false;
                match stream
                    .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD)
                {
                    Ok(Some(g)) => {
                        let _ = g.upload();
                        gpu.decode_graph = Some(g);
                        gpu.decode_graph_key = graph_key;
                    }
                    Ok(None) | Err(_) => {
                        gpu.decode_graph = None;
                    }
                }
            }
            drop(capture_guard);
            let enc = t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            gpu.sync()?;
            note_call(launches, enc, t1.elapsed().as_nanos() as u64);
            let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
            TokenOut::Argmax(idx[0])
        } else {
            if capturing {
                capture_guard.active = false;
                match stream
                    .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD)
                {
                    Ok(Some(g)) => {
                        let _ = g.upload();
                        gpu.decode_graph = Some(g);
                        gpu.decode_graph_key = graph_key;
                    }
                    Ok(None) | Err(_) => {
                        gpu.decode_graph = None;
                    }
                }
            }
            drop(capture_guard);
            let enc = t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            gpu.sync()?;
            note_call(launches, enc, t1.elapsed().as_nanos() as u64);
            let logits = stream
                .clone_dtoh(&gpu.y_arena.slice(out_logits_at..out_logits_at + vocab))
                .ok()?;
            TokenOut::Logits(logits)
        };
        Some(out)
    })
}

fn decode_token_one(
    req: &TokenReq,
    x_in: &[f32],
    rope_in: &[[f32; 2]],
    pos: usize,
) -> Option<TokenOut> {
    if fuse_decode_eligible(req) {
        if let Some(out) = decode_token_one_fused(req, x_in, rope_in, pos) {
            return Some(out);
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
        if layer.wq.2 != wq_out || layer.wk.2 != kv || layer.wv.2 != kv || layer.wo.2 != hidden {
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
            crate::ops::rope_neox_cached_from_array(&mut qh[..rot_dim], rope);
        }
        for hi in 0..req.n_kv_heads {
            let kh = &mut k[hi * hd..(hi + 1) * hd];
            if let Some(wn) = layer.k_norm {
                crate::ops::rmsnorm(kh, wn, req.eps);
            }
            crate::ops::rope_neox_cached_from_array(&mut kh[..rot_dim], rope);
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
                shared_gate: _,
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
                let mut picks: Vec<(usize, f32)> =
                    scores.into_iter().take(*n_used).map(|(w, i)| (i, w)).collect();
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
                    for (a, b) in acc.iter_mut().zip(&d) {
                        *a += *b;
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
    if req.argmax {
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
pub struct PrefillFusion<'a> {
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    pub router: &'a [u8],
    pub n_expert: usize,
}

pub fn prefill_begin(xs: &[f32]) -> Option<()> {
    with_gpu(|gpu| {
        gpu.ensure_prefill(xs.len())?;
        gpu.stream
            .memcpy_htod(xs, &mut gpu.pf_x.slice_mut(0..xs.len()))
            .ok()?;
        gpu.pf_active = true;
        gpu.pf_len = xs.len();
        gpu.pf_ffn_done = false;
        Some(())
    })
}

pub fn prefill_end(xs: &mut [f32]) -> Option<()> {
    with_gpu(|gpu| {
        if !gpu.pf_active || xs.len() != gpu.pf_len {
            return None;
        }
        gpu.sync()?;
        gpu.stream
            .memcpy_dtoh(&gpu.pf_x.slice(0..xs.len()), xs)
            .ok()?;
        gpu.pf_active = false;
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
fn try_prefill_attn_fused_dev(
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
        let x_need = (kn_at + hd) * 4;
        gpu.ensure_arenas(x_need, y_need)?;

        // attn RMSNorm: pf_x -> pf_hs
        {
            let w = norm_f32_bytes(fusion.attn_norm, hidden)?;
            rmsnorm_pf_x_into_hs(gpu, &w, hidden, m, req.eps)?;
        }
        gpu.stream
            .memcpy_htod(rope_host, &mut gpu.x_arena.slice_mut(rope_at..rope_at + rope_host.len()))
            .ok()?;

        let (wq_c, wq_o) = resolve_w(gpu, req.wq.1)?;
        let (wk_c, wk_o) = resolve_w(gpu, req.wk.1)?;
        let (wv_c, wv_o) = resolve_w(gpu, req.wv.1)?;
        let (wo_c, wo_o) = resolve_w(gpu, req.wo.1)?;
        launch_gemm_dequant(
            gpu, req.wq.0, wq_c, wq_o, hidden, q_dim, m, true, &[], q_at, false, true,
        )?;
        launch_gemm_dequant(
            gpu, req.wk.0, wk_c, wk_o, hidden, kv, m, true, &[], k_at, true, true,
        )?;
        launch_gemm_dequant(
            gpu, req.wv.0, wv_c, wv_o, hidden, kv, m, true, &[], v_at, true, true,
        )?;

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
                        .launch_builder(&f)
                        .arg(&xp)
                        .arg(&wp)
                        .arg(&n_u)
                        .arg(&eps)
                        .arg(&rows_u)
                        .launch(cfg)
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
                        .launch_builder(&f)
                        .arg(&xp)
                        .arg(&rp)
                        .arg(&m_u)
                        .arg(&heads_u)
                        .arg(&hd_u)
                        .arg(&rot_u)
                        .launch(cfg)
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
                        .launch_builder(&f)
                        .arg(&xp)
                        .arg(&wp)
                        .arg(&n_u)
                        .arg(&eps)
                        .arg(&rows_u)
                        .launch(cfg)
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
                        .launch_builder(&f)
                        .arg(&xp)
                        .arg(&rp)
                        .arg(&m_u)
                        .arg(&heads_u)
                        .arg(&hd_u)
                        .arg(&rot_u)
                        .launch(cfg)
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
                        .launch_builder(&f)
                        .arg(&req.cache.buf)
                        .arg(&kp)
                        .arg(&k_off)
                        .arg(&kv_u)
                        .arg(&m_u)
                        .arg(&pos_u)
                        .launch(cfg)
                }
                .ok()?;
                unsafe {
                    stream
                        .launch_builder(&f)
                        .arg(&req.cache.buf)
                        .arg(&vp)
                        .arg(&v_off)
                        .arg(&kv_u)
                        .arg(&m_u)
                        .arg(&pos_u)
                        .launch(cfg)
                }
                .ok()?;
            }
            {
                let f = gpu.func_owned("attend_gqa_batch")?;
                let k_off = req.k_off as u64;
                let v_off = req.v_off as u64;
                let kv_dim = req.kv_dim as u32;
                let head_dim = hd as u32;
                let n_q = req.n_heads as u32;
                let group = (req.n_heads / req.n_kv_heads.max(1)) as u32;
                let m_u = m as u32;
                let base = req.base as u32;
                let scale = req.scale;
                let cfg = LaunchConfig {
                    grid_dim: (n_q, m_u, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let qp = ybase + (q_at * 4) as u64;
                let op = ybase + (attn_at * 4) as u64;
                unsafe {
                    stream
                        .launch_builder(&f)
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
                        .launch(cfg)
                }
                .ok()?;
            }
            // attn -> x_arena via raw ptrs (avoid mut borrow vs DevicePtr)
            {
                let f_copy = gpu.func_owned("copy_f32")?;
                let n = (m * q_dim) as u32;
                let cfg = CudaGpu::cfg_1d(n, 256);
                let sp = ybase + (attn_at * 4) as u64;
                let dp = xbase;
                unsafe {
                    stream
                        .launch_builder(&f_copy)
                        .arg(&dp)
                        .arg(&sp)
                        .arg(&n)
                        .launch(cfg)
                }
                .ok()?;
            }
        } // drop DevicePtr guards

        launch_gemm_dequant(
            gpu, req.wo.0, wo_c, wo_o, q_dim, hidden, m, true, &[], proj_at, false, false,
        )?;
        // residual into pf_x
        {
            let f_add = gpu.func_owned("residual_add")?;
            let n = (m * hidden) as u32;
            let cfg = CudaGpu::cfg_1d(n, 256);
            let mut a = gpu.pf_x.slice_mut(0..m * hidden);
            let b = gpu.y_arena.slice(proj_at..proj_at + m * hidden);
            unsafe {
                stream
                    .launch_builder(&f_add)
                    .arg(&mut a)
                    .arg(&b)
                    .arg(&n)
                    .launch(cfg)
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
        run_matvec(gpu, GgmlType::F32, fusion.router, hidden, fusion.n_expert, &hs, m)
    })
}

/// Dense FFN from `pf_hs` without D2H: gate/up/swiglu/down + residual into `pf_x`.
fn try_dense_ffn_pf(req: &GroupedFfnReq) -> Option<Vec<f32>> {
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
            gpu, req.gate.0, g_c, g_o, hidden, ffn, m, true, &[], gate_at, false, true,
        )?;
        launch_gemm_dequant(
            gpu, req.up.0, u_c, u_o, hidden, ffn, m, true, &[], up_at, true, true,
        )?;
        let stream = Arc::clone(&gpu.stream);
        {
            let (ybase, _yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            let (xbase, _xg) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
            {
                let f = gpu.func_owned("swiglu")?;
                let n = (m * ffn) as u32;
                let cfg = CudaGpu::cfg_1d(n, 256);
                let gp = ybase + (gate_at * 4) as u64;
                let up_p = ybase + (up_at * 4) as u64;
                unsafe {
                    stream
                        .launch_builder(&f)
                        .arg(&gp)
                        .arg(&up_p)
                        .arg(&n)
                        .launch(cfg)
                }
                .ok()?;
            }
            {
                let f_copy = gpu.func_owned("copy_f32")?;
                let n = (m * ffn) as u32;
                let cfg = CudaGpu::cfg_1d(n, 256);
                let sp = ybase + (gate_at * 4) as u64;
                let dp = xbase;
                unsafe {
                    stream
                        .launch_builder(&f_copy)
                        .arg(&dp)
                        .arg(&sp)
                        .arg(&n)
                        .launch(cfg)
                }
                .ok()?;
            }
        }
        launch_gemm_dequant(
            gpu, req.down.0, d_c, d_o, ffn, hidden, m, true, &[], down_at, false, false,
        )?;
        {
            let f_add = gpu.func_owned("residual_add")?;
            let n = (m * hidden) as u32;
            let cfg = CudaGpu::cfg_1d(n, 256);
            let mut a = gpu.pf_x.slice_mut(0..m * hidden);
            let b = gpu.y_arena.slice(down_at..down_at + m * hidden);
            unsafe {
                stream
                    .launch_builder(&f_add)
                    .arg(&mut a)
                    .arg(&b)
                    .arg(&n)
                    .launch(cfg)
            }
            .ok()?;
        }
        // Stream-ordered; prefill_end syncs once.
        Some(Vec::new())
    })
}

pub struct PrefillAttnReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    pub hs: &'a [f32],
    pub m: usize,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub ropes: &'a [[f32; 2]],
    pub rot_dim: usize,
    pub gate_in_q: bool,
    pub attn_bias: Option<(&'a [u8], &'a [u8], &'a [u8])>,
    pub eps: f32,
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub base: usize,
    pub scale: f32,
    pub fusion: Option<PrefillFusion<'a>>,
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
    if let Some(f) = &req.fusion {
        if !req.gate_in_q && req.attn_bias.is_none() {
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
            let mut dw = unsafe { gpu.stream.alloc::<f32>(hidden) }.ok()?;
            gpu.stream.memcpy_htod(&w, &mut dw).ok()?;
            let f_rms = gpu.func_owned("rmsnorm_into_f32")?;
            let n = hidden as u32;
            let rows = req.m as u32;
            let eps = req.eps;
            let cfg = LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let src = gpu.pf_x.slice(0..req.m * hidden);
            let mut dst = gpu.pf_hs.slice_mut(0..req.m * hidden);
            unsafe {
                gpu.stream
                    .launch_builder(&f_rms)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&dw)
                    .arg(&n)
                    .arg(&eps)
                    .arg(&rows)
                    .launch(cfg)
            }
            .ok()?;
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
                qq[row * q_dim + h * hd..row * q_dim + (h + 1) * hd]
                    .copy_from_slice(&src[..hd]);
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
        let f = gpu.func_owned("attend_gqa_batch")?;
        let k_off = req.k_off as u64;
        let v_off = req.v_off as u64;
        let kv_dim = req.kv_dim as u32;
        let head_dim = hd as u32;
        let n_q = req.n_heads as u32;
        let group = (req.n_heads / req.n_kv_heads.max(1)) as u32;
        let m_u = req.m as u32;
        let base = req.base as u32;
        let scale = req.scale;
        let cfg = LaunchConfig {
            grid_dim: (n_q, m_u, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let qv = gpu.x_arena.slice(0..q_use.len());
        let mut ov = gpu.y_arena.slice_mut(0..out_len);
        unsafe {
            gpu.stream
                .launch_builder(&f)
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
                .launch(cfg)
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
                        .launch_builder(&f_add)
                        .arg(&mut a)
                        .arg(&b)
                        .arg(&n)
                        .launch(cfg)
                }
                .ok()?;
            }
            let w = norm_f32_bytes(f.ffn_norm, hidden)?;
            let mut dw = unsafe { gpu.stream.alloc::<f32>(hidden) }.ok()?;
            gpu.stream.memcpy_htod(&w, &mut dw).ok()?;
            {
                let f_rms = gpu.func_owned("rmsnorm_into_f32")?;
                let n = hidden as u32;
                let rows = req.m as u32;
                let eps = req.eps;
                let cfg = LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let src = gpu.pf_x.slice(0..req.m * hidden);
                let mut dst = gpu.pf_hs.slice_mut(0..req.m * hidden);
                unsafe {
                    gpu.stream
                        .launch_builder(&f_rms)
                        .arg(&mut dst)
                        .arg(&src)
                        .arg(&dw)
                        .arg(&n)
                        .arg(&eps)
                        .arg(&rows)
                        .launch(cfg)
                }
                .ok()?;
            }
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

pub struct PrefillGdnReq<'a> {
    pub wqkv: (GgmlType, &'a [u8], usize),
    pub zgate: (GgmlType, &'a [u8], usize),
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    pub conv1d: &'a [u8],
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    pub hidden: usize,
    pub m: usize,
    pub eps: f32,
    pub ssm: &'a SharedRegion,
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub conv_off: usize,
    pub state_off: usize,
    pub fusion: Option<PrefillFusion<'a>>,
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
                let mut dw = unsafe { gpu.stream.alloc::<f32>(req.hidden) }.ok()?;
                gpu.stream.memcpy_htod(&w, &mut dw).ok()?;
                let f_rms = gpu.func_owned("rmsnorm_into_f32")?;
                let n = req.hidden as u32;
                let rows = req.m as u32;
                let eps = req.eps;
                let cfg = LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let src = gpu.pf_x.slice(0..req.m * req.hidden);
                let mut dst = gpu.pf_hs.slice_mut(0..req.m * req.hidden);
                unsafe {
                    gpu.stream
                        .launch_builder(&f_rms)
                        .arg(&mut dst)
                        .arg(&src)
                        .arg(&dw)
                        .arg(&n)
                        .arg(&eps)
                        .arg(&rows)
                        .launch(cfg)
                }
                .ok()?;
                gpu.sync()?;
                if f.n_expert == 0 {
                    return Some(Vec::new());
                }
                let hs = gpu
                    .stream
                    .clone_dtoh(&gpu.pf_hs.slice(0..req.m * req.hidden))
                    .ok()?;
                run_matvec(gpu, GgmlType::F32, f.router, req.hidden, f.n_expert, &hs, req.m)
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
            let mut dw = unsafe { gpu.stream.alloc::<f32>(hidden) }.ok()?;
            gpu.stream.memcpy_htod(&w, &mut dw).ok()?;
            let f_rms = gpu.func_owned("rmsnorm_into_f32")?;
            let n = hidden as u32;
            let rows = req.m as u32;
            let eps = req.eps;
            let cfg = LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let src = gpu.pf_x.slice(0..req.m * hidden);
            let mut dst = gpu.pf_hs.slice_mut(0..req.m * hidden);
            unsafe {
                gpu.stream
                    .launch_builder(&f_rms)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&dw)
                    .arg(&n)
                    .arg(&eps)
                    .arg(&rows)
                    .launch(cfg)
            }
            .ok()?;
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
                    .launch_builder(&f)
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
                    .launch(cfg)
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
                    .launch_builder(&f_copy)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&n)
                    .launch(cfg)
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
                        .launch_builder(&f_store)
                        .arg(&req.ssm.buf)
                        .arg(&src)
                        .arg(&conv_off)
                        .arg(&nwin)
                        .launch(cfg)
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
                            .launch_builder(&f_load)
                            .arg(&mut dst)
                            .arg(&req.ssm.buf)
                            .arg(&load_off)
                            .arg(&old_u)
                            .launch(cfg_old)
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
                            .launch_builder(&f_copy)
                            .arg(&mut tmp)
                            .arg(&src)
                            .arg(&new_n)
                            .launch(cfg_new)
                    }
                    .ok()?;
                    let mut dst = gpu
                        .x_arena
                        .slice_mut(win_at + old..win_at + old + req.m * channels);
                    unsafe {
                        gpu.stream
                            .launch_builder(&f_copy)
                            .arg(&mut dst)
                            .arg(&tmp)
                            .arg(&new_n)
                            .launch(cfg_new)
                    }
                    .ok()?;
                }
                let src = gpu.x_arena.slice(win_at..win_at + window_rows * channels);
                unsafe {
                    gpu.stream
                        .launch_builder(&f_store)
                        .arg(&req.ssm.buf)
                        .arg(&src)
                        .arg(&conv_off)
                        .arg(&nwin)
                        .launch(cfg)
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
                    .launch_builder(&f)
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
                    .launch(cfg)
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
                    .launch_builder(&f)
                    .arg(&mut y)
                    .arg(&z_d)
                    .arg(&wn)
                    .arg(&heads_v)
                    .arg(&d_u)
                    .arg(&eps)
                    .launch(cfg)
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
                        .launch_builder(&f_add)
                        .arg(&mut a)
                        .arg(&b)
                        .arg(&n)
                        .launch(cfg)
                }
                .ok()?;
            }
            let w = norm_f32_bytes(f.ffn_norm, hidden)?;
            let mut dw = unsafe { gpu.stream.alloc::<f32>(hidden) }.ok()?;
            gpu.stream.memcpy_htod(&w, &mut dw).ok()?;
            {
                let f_rms = gpu.func_owned("rmsnorm_into_f32")?;
                let n = hidden as u32;
                let rows = req.m as u32;
                let eps = req.eps;
                let cfg = LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let src = gpu.pf_x.slice(0..req.m * hidden);
                let mut dst = gpu.pf_hs.slice_mut(0..req.m * hidden);
                unsafe {
                    gpu.stream
                        .launch_builder(&f_rms)
                        .arg(&mut dst)
                        .arg(&src)
                        .arg(&dw)
                        .arg(&n)
                        .arg(&eps)
                        .arg(&rows)
                        .launch(cfg)
                }
                .ok()?;
            }
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
pub struct GroupedRoute<'a> {
    pub n_used: usize,
    pub norm: bool,
    pub scale: f32,
    pub bias: Option<&'a [f32]>,
    pub sigmoid: bool,
}

pub struct GroupedShared<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    pub ffn: usize,
    pub gate_out: Option<(GgmlType, &'a [u8])>,
}

pub struct GroupedCombine<'a> {
    pub tok_off: &'a [u32],
    pub hit_row: &'a [u32],
    pub hit_w: &'a [f32],
    pub m: usize,
}

pub struct GroupedFfnReq<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    pub n_expert: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub groups: &'a [[u32; 3]],
    pub x: &'a [f32],
    pub tok: &'a [u32],
    pub total_rows: usize,
    pub fused: Option<GroupedCombine<'a>>,
    pub shared: Option<GroupedShared<'a>>,
    pub route: Option<GroupedRoute<'a>>,
}

pub fn ffn_batch_grouped(req: &GroupedFfnReq) -> Option<Vec<f32>> {
    // Dense fused prefill: FFN stays on-device from pf_hs.
    if req.fused.is_some() && req.route.is_none() && req.shared.is_none() && req.n_expert <= 1 {
        if let Some(out) = try_dense_ffn_pf(req) {
            return Some(out);
        }
    }

    // GPU routing: logits live in y_arena from the fused attn block; rebuild
    // groups/tok/CSR on the host from pf_hs + those logits, then fall through.
    let owned = if let Some(route) = &req.route {
        let (hs, logits) = with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            let m = req.fused.as_ref()?.m;
            let n_expert = req.n_expert;
            let hs = gpu
                .stream
                .clone_dtoh(&gpu.pf_hs.slice(0..m * req.hidden))
                .ok()?;
            // Router was run at the end of prefill_attn_block into y_arena.
            let logits = gpu
                .stream
                .clone_dtoh(&gpu.y_arena.slice(0..m * n_expert))
                .ok()?;
            Some((hs, logits))
        })?;
        let m = req.fused.as_ref()?.m;
        let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); req.n_expert];
        for i in 0..m {
            let row = &logits[i * req.n_expert..(i + 1) * req.n_expert];
            let mut scored: Vec<(f32, usize)> = row
                .iter()
                .enumerate()
                .map(|(e, &v)| {
                    let mut s = if route.sigmoid {
                        1.0 / (1.0 + (-v).exp())
                    } else {
                        v
                    };
                    if let Some(b) = route.bias {
                        s += b.get(e).copied().unwrap_or(0.0);
                    }
                    (s, e)
                })
                .collect();
            if !route.sigmoid {
                let mx = scored.iter().map(|s| s.0).fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scored {
                    s.0 = ((s.0 - mx) * route.scale).exp();
                    sum += s.0;
                }
                for s in &mut scored {
                    s.0 /= sum.max(1e-30);
                }
            } else if route.scale != 1.0 {
                for s in &mut scored {
                    s.0 *= route.scale;
                }
            }
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut picks: Vec<(usize, f32)> = scored
                .into_iter()
                .take(route.n_used)
                .map(|(w, e)| (e, w))
                .collect();
            if route.norm {
                let wsum: f32 = picks.iter().map(|p| p.1).sum();
                if wsum > 0.0 {
                    for p in &mut picks {
                        p.1 /= wsum;
                    }
                }
            }
            for (e, w) in picks {
                groups[e].push((i, w));
            }
        }
        let used: Vec<usize> = (0..req.n_expert).filter(|&e| !groups[e].is_empty()).collect();
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
        let mut hits: Vec<Vec<(u32, f32)>> = vec![Vec::new(); m];
        for (gi, &e) in used.iter().enumerate() {
            let r0 = table[gi][1];
            for (ri, &(i, weight)) in groups[e].iter().enumerate() {
                hits[i].push((r0 + ri as u32, weight));
            }
        }
        if req.shared.is_some() {
            for (i, h) in hits.iter_mut().enumerate() {
                h.push(((total_rows + i) as u32, 1.0));
            }
        }
        let mut tok_off = Vec::with_capacity(m + 1);
        let mut hit_row = Vec::new();
        let mut hit_w = Vec::new();
        tok_off.push(0u32);
        for h in &hits {
            for &(r, w) in h {
                hit_row.push(r);
                hit_w.push(w);
            }
            tok_off.push(hit_row.len() as u32);
        }
        Some((hs, table, tok, total_rows, tok_off, hit_row, hit_w, m))
    } else {
        None
    };

    let (groups, tok, total_rows, fused_comb, act_src) = if let Some((
        hs,
        ref table,
        ref tok,
        total_rows,
        ref tok_off,
        ref hit_row,
        ref hit_w,
        m,
    )) = owned
    {
        (
            table.as_slice(),
            tok.as_slice(),
            total_rows,
            Some(GroupedCombine {
                tok_off,
                hit_row,
                hit_w,
                m,
            }),
            Some(hs),
        )
    } else {
        if req.groups.is_empty() || req.tok.len() != req.total_rows {
            return None;
        }
        (req.groups, req.tok, req.total_rows, None, None)
    };
    let fused = fused_comb.as_ref().or(req.fused.as_ref());

    if req.shared.is_some() && fused.is_none() {
        return None;
    }

    // Fused prefill leaves normed activations in pf_hs; `x` is often empty.
    let mut gathered = vec![0f32; total_rows * req.hidden];
    if let Some(hs) = act_src.as_ref() {
        for (r, &t) in tok.iter().enumerate() {
            let s = (t as usize) * req.hidden;
            if s + req.hidden > hs.len() {
                return None;
            }
            gathered[r * req.hidden..(r + 1) * req.hidden]
                .copy_from_slice(&hs[s..s + req.hidden]);
        }
    } else if fused.is_some() && req.x.is_empty() {
        let m = fused.as_ref()?.m;
        let hs = with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            gpu.stream
                .clone_dtoh(&gpu.pf_hs.slice(0..m * req.hidden))
                .ok()
        })?;
        for (r, &t) in tok.iter().enumerate() {
            let s = (t as usize) * req.hidden;
            if s + req.hidden > hs.len() {
                return None;
            }
            gathered[r * req.hidden..(r + 1) * req.hidden]
                .copy_from_slice(&hs[s..s + req.hidden]);
        }
    } else {
        let m_src = req.x.len() / req.hidden.max(1);
        for (r, &t) in tok.iter().enumerate() {
            if (t as usize) >= m_src {
                return None;
            }
            let s = (t as usize) * req.hidden;
            gathered[r * req.hidden..(r + 1) * req.hidden]
                .copy_from_slice(&req.x[s..s + req.hidden]);
        }
    }

    let gate_rb = req.gate.1.len() / req.n_expert.max(1);
    let up_rb = req.up.1.len() / req.n_expert.max(1);
    let down_rb = req.down.1.len() / req.n_expert.max(1);

    let mut gate_out = vec![0f32; total_rows * req.ffn];
    let mut up_out = vec![0f32; total_rows * req.ffn];
    for g in groups {
        let (eid, row0, rows) = (g[0] as usize, g[1] as usize, g[2] as usize);
        if eid >= req.n_expert.max(1) || row0 + rows > total_rows {
            return None;
        }
        let xg = &gathered[row0 * req.hidden..(row0 + rows) * req.hidden];
        let go = with_gpu(|gpu| {
            run_matvec(
                gpu,
                req.gate.0,
                &req.gate.1[eid * gate_rb..(eid + 1) * gate_rb],
                req.hidden,
                req.ffn,
                xg,
                rows,
            )
        })?;
        let uo = with_gpu(|gpu| {
            run_matvec(
                gpu,
                req.up.0,
                &req.up.1[eid * up_rb..(eid + 1) * up_rb],
                req.hidden,
                req.ffn,
                xg,
                rows,
            )
        })?;
        gate_out[row0 * req.ffn..(row0 + rows) * req.ffn].copy_from_slice(&go);
        up_out[row0 * req.ffn..(row0 + rows) * req.ffn].copy_from_slice(&uo);
    }
    crate::ops::swiglu(&mut gate_out, &up_out);

    let mut down_out = vec![0f32; total_rows * req.hidden];
    for g in groups {
        let (eid, row0, rows) = (g[0] as usize, g[1] as usize, g[2] as usize);
        let ag = &gate_out[row0 * req.ffn..(row0 + rows) * req.ffn];
        let d = with_gpu(|gpu| {
            run_matvec(
                gpu,
                req.down.0,
                &req.down.1[eid * down_rb..(eid + 1) * down_rb],
                req.ffn,
                req.hidden,
                ag,
                rows,
            )
        })?;
        down_out[row0 * req.hidden..(row0 + rows) * req.hidden].copy_from_slice(&d);
    }

    if let (Some(sh), Some(comb)) = (&req.shared, fused) {
        let hs = with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            gpu.stream
                .clone_dtoh(&gpu.pf_hs.slice(0..comb.m * req.hidden))
                .ok()
        })?;
        let g = with_gpu(|gpu| {
            run_matvec(gpu, sh.gate.0, sh.gate.1, req.hidden, sh.ffn, &hs, comb.m)
        })?;
        let u = with_gpu(|gpu| {
            run_matvec(gpu, sh.up.0, sh.up.1, req.hidden, sh.ffn, &hs, comb.m)
        })?;
        let mut act = g;
        crate::ops::swiglu(&mut act, &u);
        let d = with_gpu(|gpu| {
            run_matvec(gpu, sh.down.0, sh.down.1, sh.ffn, req.hidden, &act, comb.m)
        })?;
        down_out.extend_from_slice(&d);
    }

    if let Some(comb) = fused {
        with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            gpu.ensure_arenas(down_out.len() * 4, 4)?;
            gpu.stream
                .memcpy_htod(&down_out, &mut gpu.y_arena.slice_mut(0..down_out.len()))
                .ok()?;
            let mut tok_off = unsafe { gpu.stream.alloc::<u32>(comb.tok_off.len()) }.ok()?;
            let mut hit_row = unsafe { gpu.stream.alloc::<u32>(comb.hit_row.len()) }.ok()?;
            let mut hit_w = unsafe { gpu.stream.alloc::<f32>(comb.hit_w.len()) }.ok()?;
            gpu.stream.memcpy_htod(comb.tok_off, &mut tok_off).ok()?;
            gpu.stream.memcpy_htod(comb.hit_row, &mut hit_row).ok()?;
            gpu.stream.memcpy_htod(comb.hit_w, &mut hit_w).ok()?;
            let f = gpu.func_owned("moe_combine_csr")?;
            let m_u = comb.m as u32;
            let hidden_u = req.hidden as u32;
            let cfg = LaunchConfig {
                grid_dim: (hidden_u.div_ceil(256), m_u, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let downs = gpu.y_arena.slice(0..down_out.len());
            let mut xs = gpu.pf_x.slice_mut(0..comb.m * req.hidden);
            unsafe {
                gpu.stream
                    .launch_builder(&f)
                    .arg(&mut xs)
                    .arg(&downs)
                    .arg(&tok_off)
                    .arg(&hit_row)
                    .arg(&hit_w)
                    .arg(&m_u)
                    .arg(&hidden_u)
                    .launch(cfg)
            }
            .ok()?;
            gpu.sync()?;
            Some(Vec::new())
        })
    } else {
        Some(down_out)
    }
}
