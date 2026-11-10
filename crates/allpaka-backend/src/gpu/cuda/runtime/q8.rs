//! Q8 activation quantisation and the permuted Q8 layouts the integer
//! matvec kernels consume.

use super::*;

pub(crate) fn q8_decode_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("ALLPAKA_Q8") {
        // Default on: Q4_K+Q6_K dp4a mmvq. Set ALLPAKA_Q8=0 to force float Metal.
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    })
}

pub(super) fn quantize_q8_cfg(n: u32, rows: u32) -> LaunchConfig {
    let nblk = n / 32;
    let (grid_x, block_y) = if n >= 256 {
        ((nblk + 7) / 8, 8u32)
    } else {
        (nblk, 1u32)
    };
    LaunchConfig {
        grid_dim: (grid_x, rows, 1),
        block_dim: (32, block_y, 1),
        shared_mem_bytes: 0,
    }
}

/// Packed Q8 byte count as i8 elems: (n/32)*44*rows.
#[inline]
pub(super) fn q8_packed_elems(n: usize, rows: usize) -> usize {
    (n / 32) * 44 * rows
}

pub(super) fn launch_quantize_q8_kernel(
    gpu: &mut CudaGpu,
    kernel: &str,
    x_ptr: u64,
    n: usize,
    rows: usize,
) -> Option<()> {
    if n % 32 != 0 {
        return None;
    }
    gpu.ensure_q8(n, rows)?;
    let n_u = n as u32;
    let rows_u = rows as u32;
    let f = gpu.func_owned(kernel)?;
    let cfg = quantize_q8_cfg(n_u, rows_u);
    let mut q = gpu.q8_q.slice_mut(0..q8_packed_elems(n, rows));
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&x_ptr)
            .arg(&mut q)
            .arg(&n_u)
            .arg(&rows_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    gpu.q8_src = x_ptr;
    gpu.q8_src_n = n;
    gpu.q8_src_m = rows;
    gpu.q8_off = 0;
    Some(())
}

pub fn launch_quantize_q8(gpu: &mut CudaGpu, x_ptr: u64, n: usize, rows: usize) -> Option<()> {
    launch_quantize_q8_kernel(gpu, "quantize_q8_1", x_ptr, n, rows)
}

pub fn launch_quantize_q8_q6(gpu: &mut CudaGpu, x_ptr: u64, n: usize, rows: usize) -> Option<()> {
    launch_quantize_q8_kernel(gpu, "quantize_q8_1_q6", x_ptr, n, rows)
}

/// Decode: FA [hd,n_q] -> out [n_q,hd] + pack Q8 for o_proj (one kernel).
pub fn launch_permute_q8_m1(
    gpu: &mut CudaGpu,
    fa_src: u64,
    out_ptr: u64,
    hd: usize,
    n_q: usize,
) -> Option<()> {
    let n = hd * n_q;
    if n % 32 != 0 {
        return None;
    }
    gpu.ensure_q8(n, 1)?;
    crate::gpu::cuda::ggml::prepare_for_cudarc();
    let hd_u = hd as u32;
    let nq_u = n_q as u32;
    let f = gpu.func_owned("permute_q8_m1")?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32) / 32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut q = gpu.q8_q.slice_mut(0..q8_packed_elems(n, 1));
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&fa_src)
            .arg(&out_ptr)
            .arg(&mut q)
            .arg(&hd_u)
            .arg(&nq_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    gpu.q8_src = out_ptr;
    gpu.q8_src_n = n;
    gpu.q8_src_m = 1;
    gpu.q8_off = 0;
    Some(())
}

pub fn launch_permute_q8_m1_qonly(
    gpu: &mut CudaGpu,
    fa_src: u64,
    out_identity: u64,
    hd: usize,
    n_q: usize,
) -> Option<()> {
    let n = hd * n_q;
    if n % 32 != 0 {
        return None;
    }
    gpu.ensure_q8(n, 1)?;
    crate::gpu::cuda::ggml::prepare_for_cudarc();
    let hd_u = hd as u32;
    let nq_u = n_q as u32;
    let f = gpu.func_owned("permute_q8_m1_qonly")?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32) / 32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut q = gpu.q8_q.slice_mut(0..q8_packed_elems(n, 1));
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&fa_src)
            .arg(&mut q)
            .arg(&hd_u)
            .arg(&nq_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    gpu.q8_src = out_identity;
    gpu.q8_src_n = n;
    gpu.q8_src_m = 1;
    gpu.q8_off = 0;
    Some(())
}

pub(super) fn launch_qk_q8_matvec(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    x_ptr: u64,
    y_ptr: u64,
    m: usize,
    add: bool,
) -> Option<()> {
    let stream = Arc::clone(&gpu.stream);
    launch_qk_q8_matvec_on(
        gpu, &stream, ty, chunk, w_off, n_in, n_out, x_ptr, y_ptr, m, add,
    )
}
