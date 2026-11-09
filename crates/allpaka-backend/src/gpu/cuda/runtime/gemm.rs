//! Dequantising GEMM for prefill batches: stage the weight tile as f16 and
//! multiply through cuBLASLt.

use super::*;

/// Launch dequant+GEMM into `y_arena[y_off..]`. No synchronize / D2H.
///
/// f16 scratch layout is `[X | W | C]` so Q/K/V (same X, different n_out) can
/// reuse the X cast. `reuse_x_f16` skips that cast when the front of the arena
/// already holds `m * n_in` f16 activations. `x_from_pf_hs` reads X from the
/// prefill residual-norm buffer instead of `x_arena`.
pub fn launch_gemm_dequant(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    m: usize,
    x_already_in_arena: bool,
    x_elems: &[f32],
    y_off: usize,
    reuse_x_f16: bool,
    x_from_pf_hs: bool,
    x_dev_override: Option<u64>,
) -> Option<()> {
    gpu.ensure_arenas(m * n_in * 4, (y_off + m * n_out) * 4)?;
    if !x_already_in_arena {
        gpu.stream
            .memcpy_htod(x_elems, &mut gpu.x_arena.slice_mut(0..m * n_in))
            .ok()?;
    }

    // llama.cpp ggml-cuda MMQ (fast on 5090 vs dequant+cuBLASLt).
    // Opt out with ALLPAKA_GGML_GEMM=0.
    let ggml_gemm = std::env::var("ALLPAKA_GGML_GEMM")
        .map_or(true, |v| !(v == "0" || v.eq_ignore_ascii_case("false")));
    if ggml_gemm && crate::gpu::cuda::ggml::enabled() {
        if let Some(wty) = crate::gpu::cuda::ggml::ggml_type_code(ty) {
            if matches!(ty, GgmlType::Q4K | GgmlType::Q6K) && n_in % 256 == 0 {
                let stream = Arc::clone(&gpu.stream);
                crate::gpu::cuda::ggml::prepare_from_cudarc(&stream);
                let (wp, wg) = DevicePtr::device_ptr(&gpu.chunks[chunk].buf, &stream);
                drop(wg);
                let w_dev = wp + w_off;
                let x_dev = if let Some(p) = x_dev_override {
                    p
                } else if x_from_pf_hs {
                    let (p, g) = DevicePtr::device_ptr(&gpu.pf_hs, &stream);
                    drop(g);
                    p
                } else {
                    let (p, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
                    drop(g);
                    p
                };
                let (ybase, yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
                drop(yg);
                let y_dev = ybase + (y_off * 4) as u64;
                if crate::gpu::cuda::ggml::mul_mat(w_dev, x_dev, y_dev, n_in, n_out, m, wty) {
                    return Some(());
                }
            }
        }
    }
    if x_dev_override.is_some() {
        return None;
    }

    crate::gpu::cuda::ggml::prepare_for_cudarc();

    // Int8 TC MMQ: ALLPAKA_MMQ=1 (NVRTC), =tile (float smem), =nvcc (standalone PTX).
    // Neither beats dequant+cuBLAS yet; all stay opt-in.
    if matches!(ty, GgmlType::Q4K) && n_in % 256 == 0 {
        let mmq = std::env::var("ALLPAKA_MMQ").unwrap_or_default();
        if mmq == "1" || mmq == "tile" || mmq == "nvcc" {
            let stream = Arc::clone(&gpu.stream);
            let use_tile = mmq == "tile";
            let use_nvcc = mmq == "nvcc";
            if !use_tile {
                let x_ptr = if x_from_pf_hs {
                    let (p, g) = DevicePtr::device_ptr(&gpu.pf_hs, &stream);
                    drop(g);
                    p
                } else {
                    let (p, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
                    drop(g);
                    p
                };
                if gpu.q8_src != x_ptr || gpu.q8_src_n != n_in || gpu.q8_src_m != m {
                    launch_quantize_q8(gpu, x_ptr, n_in, m)?;
                }
            }
            let fname = if use_nvcc {
                "mm_q4_k_mmq"
            } else if use_tile {
                "mm_q4_k_tile"
            } else {
                "mm_q4_k_mma"
            };
            let f = gpu.func_owned(fname)?;
            let n_in_u = n_in as u32;
            let n_out_u = n_out as u32;
            let m_u = m as u32;
            let w_off_u = w_off;
            // nvcc: tile 128×64, K-step 256; dynamic smem As..Bw ≈ 52 KiB
            const NVCC_MMQ_SMEM: u32 = 52_224;
            let cfg = if use_nvcc {
                LaunchConfig {
                    grid_dim: (n_out_u.div_ceil(128), m_u.div_ceil(64), 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: NVCC_MMQ_SMEM,
                }
            } else if use_tile {
                LaunchConfig {
                    grid_dim: (n_out_u.div_ceil(64), m_u.div_ceil(32), 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                LaunchConfig {
                    grid_dim: (n_out_u.div_ceil(64), m_u.div_ceil(16), 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 0,
                }
            };
            let (ybase, yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(yg);
            let dst_p = ybase + (y_off * 4) as u64;
            let w = &gpu.chunks[chunk].buf;
            if use_tile {
                let xv = if x_from_pf_hs {
                    gpu.pf_hs.slice(0..m * n_in)
                } else {
                    gpu.x_arena.slice(0..m * n_in)
                };
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(w)
                        .arg(&xv)
                        .arg(&dst_p)
                        .arg(&n_in_u)
                        .arg(&n_out_u)
                        .arg(&w_off_u)
                        .arg(&m_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            } else {
                // Opt-in MMQ still expects flat q+d; packed ABI leaves this path stale.
                let q = gpu.q8_q.slice(0..q8_packed_elems(n_in, m));
                let d = gpu.q8_d.slice(0..1);
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(w)
                        .arg(&q)
                        .arg(&d)
                        .arg(&dst_p)
                        .arg(&n_in_u)
                        .arg(&n_out_u)
                        .arg(&w_off_u)
                        .arg(&m_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            return Some(());
        }
    }

    let rb = row_bytes(ty, n_in)?;
    let fmt = dq_fmt(ty)?;
    let x_elems_n = m * n_in;
    let w_elems = n_out * n_in;
    let c_elems = m * n_out;
    // Ping-pong W on a second stream. Grow the reserved X region / W pitch only
    // after draining both streams so dequant never clobbers an in-flight GEMM.
    if w_elems > gpu.w_pitch || x_elems_n > gpu.f16_x_region {
        let _ = gpu.stream.synchronize();
        let _ = gpu.dq_stream.synchronize();
        gpu.w_pitch = gpu.w_pitch.max(w_elems);
        gpu.f16_x_region = gpu.f16_x_region.max(x_elems_n);
        gpu.w_ping = 0;
        gpu.x_f16_n = 0;
    }
    let x16 = 0usize;
    let ping = gpu.w_ping;
    let w16 = gpu.f16_x_region + ping * gpu.w_pitch;
    let c16 = gpu.f16_x_region + 2 * gpu.w_pitch;
    gpu.ensure_f16(c16 + c_elems)?;
    {
        let f = gpu.func_owned("dequant_rows_f16")?;
        let n_in_u = n_in as u32;
        let n_out_u = n_out as u32;
        let fmt_u = fmt;
        let rb_u = rb as u32;
        let w_off_u = w_off;
        const ROWS: u32 = 16;
        let k_blocks = matches!(fmt, 0 | 1 | 2) && n_in % 256 == 0;
        let cfg = if k_blocks {
            LaunchConfig {
                grid_dim: (n_out_u.div_ceil(ROWS), n_in_u / 256, 1),
                block_dim: (32, ROWS, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (n_out_u.div_ceil(ROWS), 1, 1),
                block_dim: (32, ROWS, 1),
                shared_mem_bytes: 0,
            }
        };
        let w = &gpu.chunks[chunk].buf;
        let mut out = gpu.f16_arena.slice_mut(w16..w16 + w_elems);
        unsafe {
            gpu.dq_stream
                .launch_builder_pdl(&f)
                .arg(w)
                .arg(&mut out)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&rb_u)
                .arg(&fmt_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        let ev = if ping == 0 { &gpu.ev_w0 } else { &gpu.ev_w1 };
        ev.record(&gpu.dq_stream).ok()?;
        gpu.w_ping = 1 - ping;
    }
    if !(reuse_x_f16 && gpu.x_f16_n == x_elems_n) {
        let f = gpu.func_owned("cast_f32_to_f16")?;
        let n = x_elems_n as u32;
        let cfg = CudaGpu::cfg_1d(n, 256);
        let mut dst = gpu.f16_arena.slice_mut(x16..x16 + x_elems_n);
        if x_from_pf_hs {
            let src = gpu.pf_hs.slice(0..x_elems_n);
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&n)
                    .launch_pdl(cfg)
            }
            .ok()?;
        } else {
            let src = gpu.x_arena.slice(0..x_elems_n);
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&n)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        gpu.x_f16_n = x_elems_n;
    }
    {
        let ev = if ping == 0 { &gpu.ev_w0 } else { &gpu.ev_w1 };
        gpu.stream.wait(ev).ok()?;
    }
    let mut used_blas = false;
    if gpu.blaslt.is_some() {
        let (head, mut c16s) = gpu.f16_arena.split_at_mut(c16);
        let a = head.slice(x16..x16 + m * n_in);
        let b = head.slice(w16..w16 + w_elems);
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: n_out as u64,
            n: m as u64,
            k: n_in as u64,
            alpha: 1.0,
            lda: n_in as i64,
            ldb: n_in as i64,
            beta: 0.0,
            ldc: n_out as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        let blas_ok = {
            let blas = gpu.blaslt.as_ref().unwrap();
            unsafe { Matmul::matmul(blas, cfg, &b, &a, &mut c16s, None, None) }.is_ok()
        };
        if blas_ok {
            let f = gpu.func_owned("cast_f16_to_f32")?;
            let n = (m * n_out) as u32;
            let lcfg = CudaGpu::cfg_1d(n, 256);
            let stream = Arc::clone(&gpu.stream);
            let (ybase, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(g);
            let dst_p = ybase + (y_off * 4) as u64;
            let src = gpu.f16_arena.slice(c16..c16 + m * n_out);
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&dst_p)
                    .arg(&src)
                    .arg(&n)
                    .launch_pdl(lcfg)
            }
            .ok()?;
            used_blas = true;
        }
    }
    if !used_blas {
        let f = gpu.func_owned("gemm_f16_f32")?;
        let m_u = m as u32;
        let n_u = n_out as u32;
        let k_u = n_in as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(16), m_u.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let stream = Arc::clone(&gpu.stream);
        let (ybase, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
        drop(g);
        let dst_p = ybase + (y_off * 4) as u64;
        let a = gpu.f16_arena.slice(x16..x16 + m * n_in);
        let b = gpu.f16_arena.slice(w16..w16 + w_elems);
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&a)
                .arg(&b)
                .arg(&dst_p)
                .arg(&m_u)
                .arg(&n_u)
                .arg(&k_u)
                .launch_pdl(cfg)
        }
        .ok()?;
    }
    crate::gpu::cuda::ggml::mark_cudarc_dirty();
    Some(())
}

/// Dequant weight matrix to f16 then GEMM: y[m,n_out] = x[m,n_in] @ W^T.
pub fn gemm_dequant(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    m: usize,
    x_elems: &[f32],
) -> Option<Vec<f32>> {
    let t0 = Instant::now();
    launch_gemm_dequant(
        gpu, ty, chunk, w_off, n_in, n_out, m, false, x_elems, 0, false, false, None,
    )?;
    let encode_ns = t0.elapsed().as_nanos() as u64;
    let t1 = Instant::now();
    gpu.sync()?;
    note_call(2, encode_ns, t1.elapsed().as_nanos() as u64);
    gpu.stream.clone_dtoh(&gpu.y_arena.slice(0..m * n_out)).ok()
}
