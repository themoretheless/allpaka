//! Dequant matvec launchers: the generic per-format path and the fused
//! Q4_K two-matrix and QKV variants used by decode.

use super::*;

/// Matvec activations addressed by raw device pointers: `y[m, n_out] (+)= W · x[m, n_in]`.
#[derive(Clone, Copy, Debug)]
pub struct MatvecPtrs {
    pub x: u64,
    pub y: u64,
    pub m: usize,
    /// Accumulate into `y` instead of overwriting it.
    pub add: bool,
}

/// Matvec activations addressed as f32 offsets into the `y_arena`, whose
/// device base is resolved once by the caller (CUDA graph capture must not
/// record per-launch `DevicePtr` events).
#[derive(Clone, Copy, Debug)]
pub struct ArenaMatvec {
    pub y_base: u64,
    pub x_off: usize,
    pub y_off: usize,
    pub m: usize,
    /// Accumulate into `y` instead of overwriting it.
    pub add: bool,
}

/// Fused gate+up Q4_K matvec: both weights share `[n_out, n_in]` and one input.
#[derive(Clone, Copy, Debug)]
pub struct GateUpReq {
    pub gate: WeightLoc,
    pub up: WeightLoc,
    pub n_in: usize,
    pub n_out: usize,
    pub y_base: u64,
    pub x_off: usize,
    pub gate_off: usize,
    pub up_off: usize,
    pub m: usize,
}

/// Destination for V written straight into the f16 KV cache by the fused QKV matvec.
#[derive(Clone, Copy, Debug)]
pub struct VCacheDst<'a> {
    pub buf: &'a CudaSlice<u8>,
    /// Element offset of this layer's V rows in the cache.
    pub off: u64,
    pub kv_dim: u32,
}

/// Fused Q/K/V matvec over one input; Q and K are Q4_K, V is `v_ty`.
#[derive(Clone, Copy, Debug)]
pub struct QkvReq<'a> {
    pub q: WeightLoc,
    pub k: WeightLoc,
    pub v: WeightLoc,
    pub v_ty: GgmlType,
    pub n_in: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub y_base: u64,
    pub x_off: usize,
    pub q_off: usize,
    pub k_off: usize,
    pub v_off: usize,
    pub m: usize,
    pub v_cache: Option<VCacheDst<'a>>,
}

pub(super) fn launch_qk_q8_matvec_on(
    gpu: &mut CudaGpu,
    stream: &CudaStream,
    w: DevWeight,
    io: MatvecPtrs,
) -> Option<()> {
    let DevWeight {
        ty,
        loc: WeightLoc { chunk, off: w_off },
        n_in,
        n_out,
    } = w;
    let MatvecPtrs {
        x: x_ptr,
        y: y_ptr,
        m,
        add,
    } = io;
    if n_in % 256 != 0 {
        return None;
    }
    // Prefer nwarps=8 for long-K Q6 (down-proj). Q4 stays at 4.
    // Huge n_out (LM head ~152k) prefers nwarps=4: more CTAs on short K.
    let long_q6 = matches!(ty, GgmlType::Q6K) && n_in >= 4096;
    let huge_out = n_out >= 65536;
    let policy = crate::runtime::get();
    let force2 = policy.cuda_nwarps == Some(2);
    let force8 = policy.cuda_nwarps == Some(8);
    let force4 = policy.cuda_nwarps == Some(4);
    let rows2 = policy.cuda_mmvq_rows2;
    let out4 = huge_out && policy.cuda_out_n4; // nwarps=4 on LM head (~152k rows) beats n8 on 5090
    let auto8 = long_q6 && !force2 && !force4 && !rows2 && !(out4 && !force8);
    // Q6 + f32 X: pack Q8 in smem inside the matvec (skip separate quantize_q8_1).
    // Opt-in only: measured slower than quantize+q8 on 5090 (~43 vs ~64 tg).
    let q6_f32 = matches!(ty, GgmlType::Q6K)
        && auto8
        && gpu.fns.contains_key("matvec_q6_k_f32_n8")
        && policy.cuda_q6_f32;
    // FA→WO: consume pending FA f32 ptr with fused pack-in-mmvq (skips permute_q8).
    let wo_fa = matches!(ty, GgmlType::Q4K)
        && m == 1
        && n_in == 8192
        && gpu.wo_fa_x != 0
        && gpu.fns.contains_key("matvec_q4_k_f32_k8192");
    if wo_fa {
        let x_fa = gpu.wo_fa_x;
        gpu.wo_fa_x = 0;
        let f = gpu.func_owned("matvec_q4_k_f32_k8192")?;
        let n_u = n_in as u32;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let w_off_u = w_off;
        let add_u = add as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_out_u, m_u, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let wb = &gpu.chunks[chunk].buf;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(&x_fa)
                .arg(&y_ptr)
                .arg(&n_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .arg(&add_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        gpu.invalidate_q8();
        return Some(());
    }
    if q6_f32 {
        let f = gpu.func_owned("matvec_q6_k_f32_n8")?;
        let n_u = n_in as u32;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let w_off_u = w_off;
        let add_u = add as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_out_u, m_u, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let wb = &gpu.chunks[chunk].buf;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(&x_ptr)
                .arg(&y_ptr)
                .arg(&n_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .arg(&add_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        // X was consumed as f32; invalidate any cached Q8 of this pointer.
        if gpu.q8_src == x_ptr {
            gpu.invalidate_q8();
        }
        return Some(());
    }
    let (kern, nwarps, grid_x) = match (ty, force8 || auto8, force2, rows2) {
        (GgmlType::Q4K, false, false, true) => {
            ("matvec_q4_k_q8_r2", 4u32, (n_out as u32).div_ceil(2))
        }
        (GgmlType::Q4K, true, _, _) => ("matvec_q4_k_q8_n8", 8u32, n_out as u32),
        (GgmlType::Q4K, false, true, _) => ("matvec_q4_k_q8_n2", 2u32, n_out as u32),
        (GgmlType::Q4K, false, false, false)
            if n_in == 8192 && crate::runtime::get().cuda_k8192 =>
        {
            ("matvec_q4_k_q8_k8192", 4u32, n_out as u32)
        }
        // Opt-in: measured ~61 tg vs ~70.6 generic on Qwen3-32B / 5090.
        (GgmlType::Q4K, false, false, false)
            if n_in == 5120
                && gpu.fns.contains_key("matvec_q4_k_q8_k5120")
                && crate::runtime::get().cuda_k5120 =>
        {
            ("matvec_q4_k_q8_k5120", 4u32, n_out as u32)
        }
        (GgmlType::Q4K, false, false, false) => ("matvec_q4_k_q8", 4u32, n_out as u32),
        // Opt-in: soft-pipe unroll regressed ~58 tg vs ~70.6 n8 on 5090.
        (GgmlType::Q6K, true, _, _)
            if n_in == 25600
                && gpu.fns.contains_key("matvec_q6_k_q8_n8_k25600")
                && crate::runtime::get().cuda_q6_k25600 =>
        {
            ("matvec_q6_k_q8_n8_k25600", 8u32, n_out as u32)
        }
        (GgmlType::Q6K, true, _, _) => ("matvec_q6_k_q8_n8", 8u32, n_out as u32),
        (GgmlType::Q6K, _, _, _) => ("matvec_q6_k_q8", 4u32, n_out as u32),
        _ => return None,
    };
    let q_elems = q8_packed_elems(n_in, m);
    gpu.ensure_q8_bytes(gpu.q8_off + q_elems)?;
    apply_q8_l2_persist(gpu, stream);
    apply_w_l2_stream(gpu, stream, chunk);
    let n_u = n_in as u32;
    let rows_u = m as u32;
    let need_q = gpu.q8_src != x_ptr || gpu.q8_src_n != n_in || gpu.q8_src_m != m;
    if need_q {
        let f = gpu.func_owned("quantize_q8_1")?;
        let cfg = quantize_q8_cfg(n_u, rows_u);
        let mut q = gpu.q8_q.slice_mut(0..q_elems);
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&x_ptr)
                .arg(&mut q)
                .arg(&n_u)
                .arg(&rows_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        gpu.q8_src = x_ptr;
        gpu.q8_src_n = n_in;
        gpu.q8_src_m = m;
        gpu.q8_off = 0;
    }
    let f = gpu.func_owned(kern)?;
    let n_out_u = n_out as u32;
    let m_u = m as u32;
    let w_off_u = w_off;
    let add_u = add as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, m_u, 1),
        block_dim: (32, nwarps, 1),
        shared_mem_bytes: 0,
    };
    let wb = &gpu.chunks[chunk].buf;
    let q_off = gpu.q8_off;
    let q = gpu.q8_q.slice(q_off..q_off + q_elems);
    unsafe {
        stream
            .launch_builder_pdl(&f)
            .arg(wb)
            .arg(&q)
            .arg(&y_ptr)
            .arg(&n_u)
            .arg(&n_out_u)
            .arg(&w_off_u)
            .arg(&m_u)
            .arg(&add_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

pub fn run_matvec(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    w: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    m: usize,
) -> Option<Vec<f32>> {
    if x.len() != m * n_in {
        return None;
    }
    let (chunk, w_off) = resolve_w(gpu, w)?;
    let _ = matvec_kernel(ty)?;

    if m >= MM_MIN_M && dq_fmt(ty).is_some() {
        let w = DevWeight {
            ty,
            loc: WeightLoc { chunk, off: w_off },
            n_in,
            n_out,
        };
        return gemm_dequant(gpu, w, m, x);
    }

    gpu.ensure_arenas(m * n_in * 4, m * n_out * 4)?;
    gpu.stream
        .memcpy_htod(x, &mut gpu.x_arena.slice_mut(0..m * n_in))
        .ok()?;

    let t0 = Instant::now();
    if matches!(ty, GgmlType::Q4K | GgmlType::Q6K) && n_in % 256 == 0 && q8_decode_enabled() {
        let stream = Arc::clone(&gpu.stream);
        let (xbase, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
        drop(g);
        let (ybase, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
        drop(g);
        let w = DevWeight {
            ty,
            loc: WeightLoc { chunk, off: w_off },
            n_in,
            n_out,
        };
        let io = MatvecPtrs {
            x: xbase,
            y: ybase,
            m,
            add: false,
        };
        launch_qk_q8_matvec(gpu, w, io)?;
    } else {
        let name = matvec_kernel(ty)?;
        let f = gpu.func_owned(name)?;
        let n_in_u = n_in as u32;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let w_off_u = w_off;
        let add_u = 0u32;
        let cfg = matvec_launch_cfg(ty, n_out_u, m_u);
        let wb = &gpu.chunks[chunk].buf;
        let xv = gpu.x_arena.slice(0..m * n_in);
        let mut yv = gpu.y_arena.slice_mut(0..m * n_out);
        let add_k = matches!(ty, GgmlType::Q4K | GgmlType::Q6K);
        unsafe {
            if add_k {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(wb)
                    .arg(&xv)
                    .arg(&mut yv)
                    .arg(&n_in_u)
                    .arg(&n_out_u)
                    .arg(&w_off_u)
                    .arg(&m_u)
                    .arg(&add_u)
                    .launch_pdl(cfg)
            } else {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(wb)
                    .arg(&xv)
                    .arg(&mut yv)
                    .arg(&n_in_u)
                    .arg(&n_out_u)
                    .arg(&w_off_u)
                    .arg(&m_u)
                    .launch_pdl(cfg)
            }
        }
        .ok()?;
    }
    let encode_ns = t0.elapsed().as_nanos() as u64;
    let t1 = Instant::now();
    gpu.sync()?;
    note_call(1, encode_ns, t1.elapsed().as_nanos() as u64);
    gpu.stream.clone_dtoh(&gpu.y_arena.slice(0..m * n_out)).ok()
}

/// Like [`launch_matvec_y`] but uses a pre-resolved `y_arena` device base so
/// CUDA graph capture can avoid per-launch `DevicePtr` event recording.
pub fn launch_matvec_y_ptr(gpu: &mut CudaGpu, w: DevWeight, io: ArenaMatvec) -> Option<()> {
    let stream = Arc::clone(&gpu.stream);
    launch_matvec_y_ptr_on(gpu, &stream, w, io)
}

/// Device-resident matvec on an explicit stream (for Q/K/V overlap).
pub fn launch_matvec_y_ptr_on(
    gpu: &mut CudaGpu,
    stream: &CudaStream,
    w: DevWeight,
    io: ArenaMatvec,
) -> Option<()> {
    let DevWeight {
        ty,
        loc: WeightLoc { chunk, off: w_off },
        n_in,
        n_out,
    } = w;
    let ArenaMatvec {
        y_base,
        x_off,
        y_off,
        m,
        add,
    } = io;
    let xv = y_base + (x_off * std::mem::size_of::<f32>()) as u64;
    let yv = y_base + (y_off * std::mem::size_of::<f32>()) as u64;

    // Prefill-sized batches use ggml MMQ via launch_gemm_dequant.
    // ALLPAKA_GGML_DECODE=1: llama MMVQ for non-add (legacy; disables fusion in decode).
    // ALLPAKA_GGML_MMVQ=1: llama MMVQ for all m=1 including add, keep our QKV/gate fusion.
    // Opt-in: native MMVQ measured ~70 tg vs ~66 ggml on 32B/5090 (hot window).
    let ggml_mmvq = crate::gpu::cuda::ggml::enabled()
        && (crate::runtime::get().cuda_ggml_mmvq
            || (crate::runtime::get().cuda_ggml_decode && !add));
    if m == 1 && ggml_mmvq && matches!(ty, GgmlType::Q4K | GgmlType::Q6K) && n_in % 256 == 0 {
        if let Some(wty) = crate::gpu::cuda::ggml::ggml_type_code(ty) {
            // Prefill clears shared stream; force rebind so MMVQ and residual share one queue.
            if !crate::gpu::cuda::ggml::shared_stream() {
                crate::gpu::cuda::ggml::bind_peer_stream(stream);
            }
            crate::gpu::cuda::ggml::prepare_from_cudarc(stream);
            let (wp, wg) = DevicePtr::device_ptr(&gpu.chunks[chunk].buf, stream);
            drop(wg);
            let w_dev = wp + w_off;
            if add {
                gpu.ensure_w_scratch(n_out)?;
                let (sp, sg) = DevicePtr::device_ptr(&gpu.w_scratch, stream);
                drop(sg);
                if crate::gpu::cuda::ggml::mul_mat_vec(w_dev, xv, sp, n_in, n_out, wty) {
                    // Events path: drain ggml before residual_add reads scratch.
                    crate::gpu::cuda::ggml::prepare_for_cudarc();
                    let f = gpu.func_owned("residual_add")?;
                    let n_u = n_out as u32;
                    let cfg = CudaGpu::cfg_1d(n_u, 256);
                    unsafe {
                        stream
                            .launch_builder_pdl(&f)
                            .arg(&yv)
                            .arg(&sp)
                            .arg(&n_u)
                            .launch_pdl(cfg)
                    }
                    .ok()?;
                    gpu.invalidate_q8();
                    return Some(());
                }
            } else if crate::gpu::cuda::ggml::mul_mat_vec(w_dev, xv, yv, n_in, n_out, wty) {
                crate::gpu::cuda::ggml::prepare_for_cudarc();
                gpu.invalidate_q8();
                return Some(());
            }
        }
    }

    if matches!(ty, GgmlType::Q4K | GgmlType::Q6K) && n_in % 256 == 0 && q8_decode_enabled() {
        let io = MatvecPtrs {
            x: xv,
            y: yv,
            m,
            add,
        };
        let r = launch_qk_q8_matvec_on(gpu, stream, w, io);
        return r;
    }
    let name = matvec_kernel(ty)?;
    let f = gpu.func_owned(name)?;
    let n_in_u = n_in as u32;
    let n_out_u = n_out as u32;
    let m_u = m as u32;
    let w_off_u = w_off;
    let add_u = add as u32;
    let cfg = matvec_launch_cfg(ty, n_out_u, m_u);
    let wb = &gpu.chunks[chunk].buf;
    let add_k = matches!(ty, GgmlType::Q4K | GgmlType::Q6K);
    unsafe {
        if add_k {
            stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(&xv)
                .arg(&yv)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .arg(&add_u)
                .launch_pdl(cfg)
        } else {
            stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(&xv)
                .arg(&yv)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .launch_pdl(cfg)
        }
    }
    .ok()?;
    Some(())
}

/// Fused gate+up Q4_K matvecs (identical shapes). Returns false if types differ.
pub fn launch_matvec_q4k_2(gpu: &mut CudaGpu, req: &GateUpReq) -> Option<()> {
    let GateUpReq {
        gate: WeightLoc {
            chunk: chunk0,
            off: w_off0,
        },
        up: WeightLoc {
            chunk: chunk1,
            off: w_off1,
        },
        n_in,
        n_out,
        y_base,
        x_off,
        gate_off: y0_off,
        up_off: y1_off,
        m,
    } = *req;
    if chunk0 != chunk1 || n_in % 256 != 0 {
        return None;
    }
    // Q8 path: one input quantize (via cache) + fused dual mmvq.
    if q8_decode_enabled() {
        let xv = y_base + (x_off * 4) as u64;
        gpu.ensure_q8(n_in, m)?;
        let n_u = n_in as u32;
        let rows_u = m as u32;
        let need_q =
            gpu.q8_src != xv || gpu.q8_src_n != n_in || gpu.q8_src_m != m || gpu.q8_off != 0;
        if need_q {
            let f = gpu.func_owned("quantize_q8_1")?;
            let cfg = quantize_q8_cfg(n_u, rows_u);
            let mut q = gpu.q8_q.slice_mut(0..q8_packed_elems(n_in, m));
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&xv)
                    .arg(&mut q)
                    .arg(&n_u)
                    .arg(&rows_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            gpu.q8_src = xv;
            gpu.q8_src_n = n_in;
            gpu.q8_src_m = m;
            gpu.q8_off = 0;
        }
        // Opt-in SwiGLU→Q8 epilogue (ALLPAKA_GU_Q8=1). Default keeps parallel
        // matvec_q4_k_q8_2 + separate quantize (faster on 5090 than serial-32 pack).
        if n_out % 32 == 0
            && gpu.fns.contains_key("matvec_q4_k_q8_2_q8")
            && crate::runtime::get().cuda_gu_q8
        {
            let pack_in = q8_packed_elems(n_in, m);
            let pack_out = q8_packed_elems(n_out, m);
            gpu.ensure_q8_bytes(pack_in + pack_out)?;
            // ensure may have invalidated; re-quantize into slot 0 if needed.
            if gpu.q8_src != xv || gpu.q8_src_n != n_in || gpu.q8_off != 0 {
                let f = gpu.func_owned("quantize_q8_1")?;
                let cfg = quantize_q8_cfg(n_u, rows_u);
                let mut q = gpu.q8_q.slice_mut(0..pack_in);
                unsafe {
                    gpu.stream
                        .launch_builder_pdl(&f)
                        .arg(&xv)
                        .arg(&mut q)
                        .arg(&n_u)
                        .arg(&rows_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
                gpu.q8_src = xv;
                gpu.q8_src_n = n_in;
                gpu.q8_src_m = m;
                gpu.q8_off = 0;
            }
            let f = gpu.func_owned("matvec_q4_k_q8_2_q8")?;
            let n_out_u = n_out as u32;
            let m_u = m as u32;
            let cfg = LaunchConfig {
                grid_dim: (n_out_u / 32, m_u, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            };
            let wb = &gpu.chunks[chunk0].buf;
            let y0 = y_base + (y0_off * 4) as u64;
            let (q8_base, _) = DevicePtr::device_ptr(&gpu.q8_q, &gpu.stream);
            let q_ptr = q8_base;
            let yq_ptr = q8_base + pack_in as u64;
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(wb)
                    .arg(wb)
                    .arg(&q_ptr)
                    .arg(&y0)
                    .arg(&yq_ptr)
                    .arg(&n_u)
                    .arg(&n_out_u)
                    .arg(&w_off0)
                    .arg(&w_off1)
                    .arg(&m_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            gpu.q8_src = y0;
            gpu.q8_src_n = n_out;
            gpu.q8_src_m = m;
            gpu.q8_off = pack_in;
            return Some(());
        }
        // nwarps=4 wins for gate+up on 5090; n8 dual kernel kept for A/B (ALLPAKA_GU_N8=1).
        let use_n8 = n_in >= 4096
            && gpu.fns.contains_key("matvec_q4_k_q8_2_n8")
            && crate::runtime::get().cuda_gu_n8;
        // Balanced nb=20 GU dual (5 warps, 2 blocks each). Opt-in; do not enable single-row k5120.
        let use_k5120_gu = !use_n8
            && n_in == 5120
            && gpu.fns.contains_key("matvec_q4_k_q8_2_k5120_bal")
            && crate::runtime::get().cuda_k5120_gu;
        // Opt-in only; imbalanced 4-warp k5120 regressed on 5090.
        let use_k5120 = !use_n8
            && !use_k5120_gu
            && n_in == 5120
            && gpu.fns.contains_key("matvec_q4_k_q8_2_k5120")
            && crate::runtime::get().cuda_k5120;
        let f = gpu.func_owned(if use_n8 {
            "matvec_q4_k_q8_2_n8"
        } else if use_k5120_gu {
            "matvec_q4_k_q8_2_k5120_bal"
        } else if use_k5120 {
            "matvec_q4_k_q8_2_k5120"
        } else {
            "matvec_q4_k_q8_2"
        })?;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let nwarps = if use_n8 {
            8u32
        } else if use_k5120_gu {
            5u32
        } else {
            4u32
        };
        let cfg = LaunchConfig {
            grid_dim: (n_out_u, m_u, 1),
            block_dim: (32, nwarps, 1),
            shared_mem_bytes: 0,
        };
        let wb = &gpu.chunks[chunk0].buf;
        let q = gpu.q8_q.slice(0..q8_packed_elems(n_in, m));
        let y0 = y_base + (y0_off * 4) as u64;
        let y1 = y_base + (y1_off * 4) as u64;
        unsafe {
            gpu.stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(wb)
                .arg(&q)
                .arg(&y0)
                .arg(&y1)
                .arg(&n_u)
                .arg(&n_out_u)
                .arg(&w_off0)
                .arg(&w_off1)
                .arg(&m_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        return Some(());
    }
    let f = gpu.func_owned("matvec_q4_k_2")?;
    let n_in_u = n_in as u32;
    let n_out_u = n_out as u32;
    let m_u = m as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_out_u.div_ceil(16), m_u, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };
    let wb = &gpu.chunks[chunk0].buf;
    let xv = y_base + (x_off * 4) as u64;
    let y0 = y_base + (y0_off * 4) as u64;
    let y1 = y_base + (y1_off * 4) as u64;
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(wb)
            .arg(wb)
            .arg(&xv)
            .arg(&y0)
            .arg(&y1)
            .arg(&n_in_u)
            .arg(&n_out_u)
            .arg(&w_off0)
            .arg(&w_off1)
            .arg(&m_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

/// Fused Q/K/V matvecs. Supports Q4/Q4/Q4 or Q4_K_M's Q4/Q4/Q6.
/// When `v_cache` is set (decode m=1), V is also written as f16 into the KV cache.
pub fn launch_matvec_q4k_qkv(gpu: &mut CudaGpu, req: &QkvReq<'_>) -> Option<()> {
    let QkvReq {
        q: WeightLoc {
            chunk: chunk_q,
            off: oq,
        },
        k: WeightLoc {
            chunk: chunk_k,
            off: ok,
        },
        v: WeightLoc {
            chunk: chunk_v,
            off: ov,
        },
        v_ty,
        n_in,
        n_q,
        n_kv,
        y_base,
        x_off,
        q_off,
        k_off,
        v_off,
        m,
        v_cache,
    } = *req;
    if chunk_q != chunk_k || chunk_q != chunk_v || n_in % 256 != 0 {
        if crate::runtime::get().cuda_fuse_log {
            eprintln!("cuda: qkv fuse skip chunks=({chunk_q},{chunk_k},{chunk_v}) n_in={n_in}");
        }
        return None;
    }
    let kern = match v_ty {
        GgmlType::Q4K => "matvec_q4_k_q8_qkv",
        GgmlType::Q6K => "matvec_q4_q4_q6_q8_qkv",
        _ => return None,
    };
    let xv = y_base + (x_off * 4) as u64;
    let q = y_base + (q_off * 4) as u64;
    let k = y_base + (k_off * 4) as u64;
    let v = y_base + (v_off * 4) as u64;
    let n_in_u = n_in as u32;
    let n_q_u = n_q as u32;
    let n_kv_u = n_kv as u32;
    let m_u = m as u32;

    if !q8_decode_enabled() {
        return None;
    }
    gpu.ensure_q8(n_in, m)?;
    let need_q = gpu.q8_src != xv || gpu.q8_src_n != n_in || gpu.q8_src_m != m || gpu.q8_off != 0;
    if need_q {
        let f = gpu.func_owned("quantize_q8_1")?;
        let rows_u = m as u32;
        let cfg = quantize_q8_cfg(n_in_u, rows_u);
        let mut qq = gpu.q8_q.slice_mut(0..q8_packed_elems(n_in, m));
        unsafe {
            gpu.stream
                .launch_builder_pdl(&f)
                .arg(&xv)
                .arg(&mut qq)
                .arg(&n_in_u)
                .arg(&rows_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        gpu.q8_src = xv;
        gpu.q8_src_n = n_in;
        gpu.q8_src_m = m;
        gpu.q8_off = 0;
    }
    let f = gpu.func_owned(kern)?;
    // GQA split (opt-in): fat Q4/Q4/Q6 kernel only for n_kv rows; thin Q4 for Q tail.
    // Decode m=1 only — thin matvec stride must match contiguous Q[n_kv..).
    let split = matches!(v_ty, GgmlType::Q6K)
        && m == 1
        && n_q > n_kv
        && crate::runtime::get().cuda_qkv_split;
    let grid_x = if split { n_kv_u } else { n_q_u.max(n_kv_u) };
    let cfg = LaunchConfig {
        grid_dim: (grid_x, m_u, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };
    let wb = &gpu.chunks[chunk_q].buf;
    let q_elems = q8_packed_elems(n_in, m);
    let q8_off = gpu.q8_off;
    let qq = gpu.q8_q.slice(q8_off..q8_off + q_elems);
    let null_u64 = 0u64;
    let kv_dim_u = v_cache.map(|c| c.kv_dim).unwrap_or(0);
    let v_cache_off = v_cache.map(|c| c.off).unwrap_or(0);
    unsafe {
        match v_cache {
            Some(VCacheDst { buf: cache_buf, .. }) => gpu
                .stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(wb)
                .arg(wb)
                .arg(&qq)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&n_in_u)
                .arg(&n_q_u)
                .arg(&n_kv_u)
                .arg(&oq)
                .arg(&ok)
                .arg(&ov)
                .arg(&m_u)
                .arg(cache_buf)
                .arg(&v_cache_off)
                .arg(&gpu.d_pos)
                .arg(&kv_dim_u)
                .launch_pdl(cfg),
            None => gpu
                .stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(wb)
                .arg(wb)
                .arg(&qq)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&n_in_u)
                .arg(&n_q_u)
                .arg(&n_kv_u)
                .arg(&oq)
                .arg(&ok)
                .arg(&ov)
                .arg(&m_u)
                .arg(&null_u64)
                .arg(&v_cache_off)
                .arg(&null_u64)
                .arg(&kv_dim_u)
                .launch_pdl(cfg),
        }
    }
    .ok()?;
    if split {
        let rb4 = (n_in / 256) * 144;
        let q_tail = n_q - n_kv;
        let oq_tail = oq + (n_kv as u64) * (rb4 as u64);
        let q_tail_ptr = q + (n_kv * 4) as u64;
        // Reuse already-quantized Q8 at gpu.q8_off (same xv).
        let w = DevWeight {
            ty: GgmlType::Q4K,
            loc: WeightLoc {
                chunk: chunk_q,
                off: oq_tail,
            },
            n_in,
            n_out: q_tail,
        };
        let io = MatvecPtrs {
            x: xv,
            y: q_tail_ptr,
            m,
            add: false,
        };
        launch_qk_q8_matvec(gpu, w, io)?;
        if crate::runtime::get().cuda_fuse_log {
            eprintln!("cuda: qkv_split n_kv={n_kv} q_tail={q_tail}");
        }
    }
    Some(())
}
