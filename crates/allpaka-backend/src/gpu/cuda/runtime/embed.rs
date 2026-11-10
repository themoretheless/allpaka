//! Small device-side helpers for the fused decode chain: embedding row
//! dequant and the u32 counter and store kernels.

use super::*;

/// Dequant one `token_embd` row into `y_arena[y_off..y_off+n_in]` (no sync).
pub fn launch_embed_row(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    w: &[u8],
    token: u32,
    n_in: usize,
    y_off: usize,
) -> Option<()> {
    let (chunk, base) = resolve_w(gpu, w)?;
    let rb = row_bytes(ty, n_in)?;
    let fmt = dq_fmt(ty)?;
    let w_off = base.checked_add(token as u64 * rb as u64)?;
    // Bounds: token row must lie inside the mapped chunk window.
    let end = w_off.checked_add(rb as u64)?;
    if end > gpu.chunks[chunk].len as u64 {
        return None;
    }
    gpu.ensure_arenas((y_off + n_in) * 4, (y_off + n_in) * 4)?;
    crate::gpu::cuda::ggml::prepare_for_cudarc();
    let f = gpu.func_owned("embed_row_f32")?;
    let n_in_u = n_in as u32;
    let fmt_u = fmt;
    let cfg = LaunchConfig {
        grid_dim: (1, embed_row_groups(fmt, n_in), 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let stream = Arc::clone(&gpu.stream);
    let wbuf = &gpu.chunks[chunk].buf;
    let (ybase, yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
    drop(yg);
    let out_p = ybase + (y_off * 4) as u64;
    unsafe {
        stream
            .launch_builder_pdl(&f)
            .arg(wbuf)
            .arg(&out_p)
            .arg(&n_in_u)
            .arg(&w_off)
            .arg(&fmt_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

/// Embed from device token id pointer into `y_arena[y_off..]`.
#[allow(dead_code)]
pub fn launch_embed_row_dtoken_ptr(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    w: &[u8],
    n_in: usize,
    y_off: usize,
    d_token_p: u64,
) -> Option<()> {
    let (chunk, base) = resolve_w(gpu, w)?;
    let rb = row_bytes(ty, n_in)?;
    let fmt = dq_fmt(ty)?;
    gpu.ensure_arenas((y_off + n_in) * 4, (y_off + n_in) * 4)?;
    crate::gpu::cuda::ggml::prepare_for_cudarc();
    let f = gpu.func_owned("embed_row_f32_dtoken")?;
    let n_in_u = n_in as u32;
    let fmt_u = fmt;
    let rb_u = rb as u32;
    let w_base = base;
    let cfg = LaunchConfig {
        grid_dim: (1, embed_row_groups(fmt, n_in), 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let stream = Arc::clone(&gpu.stream);
    let wbuf = &gpu.chunks[chunk].buf;
    let (ybase, yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
    drop(yg);
    let out_p = ybase + (y_off * 4) as u64;
    unsafe {
        stream
            .launch_builder_pdl(&f)
            .arg(wbuf)
            .arg(&out_p)
            .arg(&n_in_u)
            .arg(&w_base)
            .arg(&rb_u)
            .arg(&fmt_u)
            .arg(&d_token_p)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

#[allow(dead_code)]
pub fn launch_inc_u32_ptr(gpu: &mut CudaGpu, p: u64) -> Option<()> {
    let f = gpu.func_owned("inc_u32")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { gpu.stream.launch_builder_pdl(&f).arg(&p).launch_pdl(cfg) }.ok()?;
    Some(())
}

#[allow(dead_code)]
pub fn launch_store_u32_at_ptr(gpu: &mut CudaGpu, dst_p: u64, src_p: u64, idx: u32) -> Option<()> {
    let f = gpu.func_owned("store_u32_at")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&dst_p)
            .arg(&src_p)
            .arg(&idx)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}
