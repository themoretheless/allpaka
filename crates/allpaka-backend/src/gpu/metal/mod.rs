//! The Metal path: fused dequant-matvec kernels on Apple GPUs.
//!
//! Only matrix-vector products go to the GPU - they are where every byte and
//! every FLOP of decode lives. Norms, RoPE and softmax stay on the CPU: they
//! touch kilobytes, and a dispatch costs more than they do.
//!
//! Weights are never copied. The GGUF mmap is wrapped in `bytesNoCopy` shared
//! buffers (Apple Silicon memory is unified; the GPU reads the same pages the
//! page cache holds), and kernels address tensors by byte offset. A mapping
//! larger than the device's `maxBufferLength` is covered by several
//! overlapping chunk buffers - one giant buffer would silently come back nil
//! and every kernel would read garbage, which is exactly how an 88 GiB model
//! produced NaN logits while a 17 GiB one verified clean.
//!
//! The kernels mirror `quantmat.rs` line for line. The CPU implementation is
//! the reference; `verify` against llama.cpp arbitrates both.

#![cfg(target_os = "macos")]

use metal::foreign_types::ForeignType;
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use allpaka_gguf::GgmlType;

pub use super::types::*;

/// Stamp a GPUTimestamp after a dispatch. No-op unless ALLPAKA_GPU_COUNTERS.
macro_rules! cstamp {
    ($gpu:expr, $enc:expr, $label:expr) => {
        if let Some(buf) = &$gpu.cstamps {
            let i = $gpu.cstamp_idx.get();
            if i < CSTAMP_CAP {
                $enc.sample_counters_in_buffer(buf, i as u64, false);
                $gpu.cstamp_labels.borrow_mut().push($label);
                $gpu.cstamp_idx.set(i + 1);
            }
        }
    };
}

mod telemetry;

pub use telemetry::{gpu_time_stats, stats};
use telemetry::*;

mod select;
use select::*;

mod verify;
use verify::*;

mod decode;
pub use decode::*;

mod prefill;
pub use prefill::*;

mod ffn_grouped;
pub use ffn_grouped::*;

mod matvec;
pub use matvec::*;

mod mega;
use mega::*;

mod args;
use args::*;

mod attn;
pub use attn::*;

mod device;
pub use device::*;

mod region;
pub use region::*;

/// y_arena f32 offset of the router logits the last fused attention block
/// produced, when GPU-side routing skipped the CPU readback.
static PF_ROUTER_AT: AtomicU64 = AtomicU64::new(u64::MAX);

/// Effective GB/s of one INDEXED matvec (slots experts × n_out rows).
/// Used by `gpu_glm_mvbench` to A/B `ALLPAKA_MV_ID` geometry vs plain.
/// `w` must hold at least `slots` contiguous expert matrices of shape
/// `[n_out, n_in]` (same layout as decode expert packs).
pub fn indexed_matvec_bandwidth(
    w: &[u8],
    ty: GgmlType,
    n_out: usize,
    n_in: usize,
    slots: usize,
    rounds: u32,
) -> Option<f64> {
    let gpu_m = GPU.get()?.as_ref()?;
    let mut gpu = gpu_m.lock().ok()?;
    let mat = resolve(&gpu, ty, w)?;
    let lpr = lanes_per_row(ty, n_in);
    let state = gpu
        .pipeline_full(mat.kernel, 1, lpr, true, false)?
        .to_owned();
    let expert_bytes = match ty {
        GgmlType::Q4K => n_out * (n_in / 256) * 144,
        GgmlType::Q5K => n_out * (n_in / 256) * 176,
        GgmlType::Q8_0 => n_out * (n_in / 32) * 34,
        _ => return None,
    };
    if w.len() < expert_bytes.saturating_mul(slots.max(1)) {
        return None;
    }
    let stride = expert_bytes as u64;
    // Stage x / ids / y in the shared arenas.
    let x_elems = n_in.max(slots);
    let y_elems = n_out * slots + slots;
    gpu.ensure_arenas(x_elems * 4, y_elems * 4);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut f32;
        for i in 0..n_in {
            *xp.add(i) = 0.01;
        }
        let yp = gpu.y_arena.contents() as *mut u32;
        for s in 0..slots {
            *yp.add(s) = (s % slots.max(1)) as u32;
        }
    }
    let before = gpu_time_stats();
    objc::rc::autoreleasepool(|| {
        for _ in 0..rounds {
            let cmd = gpu.queue.new_command_buffer();
            let enc =
                cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
            enc.set_compute_pipeline_state(&state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(&gpu.x_arena), 0);
            enc.set_buffer(2, Some(&gpu.y_arena), (slots * 4) as u64);
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(&gpu.y_arena), 0);
            let idx = GpuIdxArgs {
                stride,
                slots: slots as u32,
                x_stride: 0,
                ids_stride: 0,
                x_row_stride: 0,
                y_row_stride: 0,
                n_rows: 0,
            };
            enc.set_bytes(
                7,
                std::mem::size_of::<GpuIdxArgs>() as u64,
                &idx as *const GpuIdxArgs as *const _,
            );
            dispatch_mv_indexed(enc, ty, n_in, n_out, slots);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
        }
    });
    let after = gpu_time_stats();
    let busy = (after.0 - before.0) as f64 / 1e9;
    if busy <= 0.0 {
        return None;
    }
    let total = (expert_bytes * slots) as f64 * rounds as f64;
    Some(total / busy / 1e9)
}

pub use super::metal_kernels::KERNELS;

/// A resolved norm bind: which chunk window holds the F32 weights.
#[derive(Clone, Copy)]
struct NormRef {
    chunk: usize,
    off: u64,
}

/// F32 weights for the router: resolve() deliberately excludes F32 so the
/// general matmul paths never route tiny test tensors to a kernel that
/// assumes float4-divisible rows; the whole-token path opts in here, with
/// that assumption checked.
fn resolve_f32(gpu: &Gpu, w: &[u8], n_in: usize) -> Option<MatRef> {
    if n_in % 4 != 0 || w.as_ptr() as usize % 4 != 0 {
        return None;
    }
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some(MatRef {
        kernel: "matvec_f32",
        ty: GgmlType::F32,
        chunk,
        w_off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

fn resolve_norm(gpu: &Gpu, raw: &[u8], hidden: usize) -> Option<NormRef> {
    if raw.len() != hidden * 4 {
        return None;
    }
    let addr = raw.as_ptr() as usize;
    if addr % 4 != 0 {
        return None;
    }
    let chunk = gpu.chunk_for(addr, raw.len())?;
    Some(NormRef {
        chunk,
        off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

/// A validated weight reference: kernel name plus window-relative offset.
#[derive(Clone, Copy)]
struct MatRef {
    kernel: &'static str,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
}

fn resolve(gpu: &Gpu, ty: GgmlType, w: &[u8]) -> Option<MatRef> {
    let kernel = match ty {
        GgmlType::Q5_0 => "matvec_q5_0",
        GgmlType::Q8_0 => q8_kernel(),
        GgmlType::Q2K => q2_kernel(),
        GgmlType::Q3K => q3_kernel(),
        GgmlType::Q4K => q4_kernel(),
        GgmlType::Q5K => q5_kernel(),
        GgmlType::Q6K => q6_kernel(),
        _ => return None,
    };
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some(MatRef {
        kernel,
        ty,
        chunk,
        w_off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

/// Encode one logical matmul as tiled sub-dispatches. Offsets into the
/// x/y buffers are per sub-batch; row strides keep them 16-byte aligned for
/// the kernels' float4 loads (n_in and n_out are whole quant blocks).
///
/// `states` holds one pipeline per tile, in `tiles_for(m)` order; they are
/// specialised ahead of encoding because that needs `&mut Gpu`.
#[allow(clippy::too_many_arguments)]
fn encode_matvec(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    mat: &MatRef,
    states: &[ComputePipelineState],
    n_in: usize,
    n_out: usize,
    x_buf: &Buffer,
    x_off: usize,
    y_buf: &Buffer,
    y_off: usize,
    m: usize,
) -> u64 {
    let mut dispatched = 0u64;
    if m >= MM_MIN_M {
        // One tile-matmul dispatch for the whole batch; `states` holds the
        // single mm pipeline the caller resolved under the same condition.
        enc.set_compute_pipeline_state(&states[0]);
        enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
        enc.set_buffer(1, Some(x_buf), x_off as u64);
        enc.set_buffer(2, Some(y_buf), y_off as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        let m32 = m as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
        enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
        let (bm, bn) = mm_kernel_for(mat.ty, m).map_or((32, 32), |(_, bm, bn)| (bm, bn));
        enc.dispatch_thread_groups(
            MTLSize::new(
                (n_out as u64).div_ceil(bm as u64),
                (m as u64).div_ceil(bn as u64),
                1,
            ),
            MTLSize::new(128, 1, 1),
        );
        return 1;
    }
    for ((start, _), state) in tiles_for(m).into_iter().zip(states) {
        enc.set_compute_pipeline_state(state);
        enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
        enc.set_buffer(1, Some(x_buf), (x_off + start * n_in * 4) as u64);
        enc.set_buffer(2, Some(y_buf), (y_off + start * n_out * 4) as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
        let per_group = 128u64;
        let total_threads = n_out as u64 * lanes_per_row(mat.ty, n_in) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(total_threads.div_ceil(per_group), 1, 1),
            MTLSize::new(per_group, 1, 1),
        );
        dispatched += 1;
    }
    dispatched
}

/// Run whole FFNs (gate, up, swiglu, down) as ONE command buffer with ONE
/// wait. The compute encoder is serial, so each dispatch sees the previous
/// one's writes; the intermediates never leave the GPU arena.
///
/// Per decoded token on a MoE this replaces two waits (gate+up batch, down
/// batch) with one, and removes the CPU swiglu pass between them.
///
/// All-or-nothing, like matvec_batch: any request without kernels for all
/// three weights declines the whole batch to the CPU path.
pub fn ffn_batch(reqs: &[FfnReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    let mut mats = Vec::with_capacity(reqs.len());
    for r in reqs {
        if r.x.len() != r.m * r.hidden || r.hidden % 4 != 0 || r.ffn % 4 != 0 {
            return None;
        }
        mats.push((
            resolve(&gpu, r.gate_ty, r.gate_w)?,
            resolve(&gpu, r.up_ty, r.up_w)?,
            resolve(&gpu, r.down_ty, r.down_w)?,
        ));
    }

    // Arena plan per request: x in the x arena; gate, up and the final out
    // rows in the y arena, each 256-aligned at the request level.
    struct Plan {
        x_off: usize,
        gate_off: usize,
        up_off: usize,
        out_off: usize,
    }
    let align = |n: usize| (n + 255) & !255;
    let mut x_need = 0usize;
    let mut y_need = 0usize;
    let mut plans = Vec::with_capacity(reqs.len());
    for r in reqs {
        let p = Plan {
            x_off: x_need,
            gate_off: y_need,
            up_off: y_need + align(r.m * r.ffn * 4),
            out_off: y_need + 2 * align(r.m * r.ffn * 4),
        };
        x_need += align(r.m * r.hidden * 4);
        y_need = p.out_off + align(r.m * r.hidden * 4);
        plans.push(p);
    }
    gpu.ensure_arenas(x_need, y_need);

    // One pipeline per (matrix, tile), specialised before encoding starts;
    // see encode_matvec. Gate and up share a shape, down has its own.
    let mut states = Vec::with_capacity(reqs.len());
    for (r, (gate, up, down)) in reqs.iter().zip(&mats) {
        let mut per_mat = Vec::with_capacity(3);
        for (mat, n_in) in [(gate, r.hidden), (up, r.hidden), (down, r.ffn)] {
            // Same routing as matvec_batch: a big batch is one tile-matmul
            // dispatch, a small one is exact matvec tiles.
            if r.m >= MM_MIN_M {
                per_mat.push(vec![gpu
                    .pipeline(mm_kernel_for(mat.ty, r.m)?.0, 1, 1)?
                    .to_owned()]);
            } else {
                let lpr = lanes_per_row(mat.ty, n_in);
                let mut tiles = Vec::new();
                for (_, rows) in tiles_for(r.m) {
                    tiles.push(gpu.pipeline(mat.kernel, rows, lpr)?.to_owned());
                }
                per_mat.push(tiles);
            }
        }
        states.push(per_mat);
    }

    unsafe {
        let xp = gpu.x_arena.contents() as *mut u8;
        for (r, p) in reqs.iter().zip(&plans) {
            std::ptr::copy_nonoverlapping(
                r.x.as_ptr() as *const u8,
                xp.add(p.x_off),
                r.x.len() * 4,
            );
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Concurrent encoder in three barriered waves: every expert's gate
        // AND up together, every swiglu, every down. One expert's matvec is
        // far too small to fill the GPU; eight experts' worth at once is
        // what the serial encoder was quietly forbidding.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let mut dispatched = 0u64;
        for (((r, p), (gate, up, _)), st) in reqs.iter().zip(&plans).zip(&mats).zip(&states) {
            dispatched += encode_matvec(
                enc,
                &gpu,
                gate,
                &st[0],
                r.hidden,
                r.ffn,
                &gpu.x_arena,
                p.x_off,
                &gpu.y_arena,
                p.gate_off,
                r.m,
            );
            dispatched += encode_matvec(
                enc,
                &gpu,
                up,
                &st[1],
                r.hidden,
                r.ffn,
                &gpu.x_arena,
                p.x_off,
                &gpu.y_arena,
                p.up_off,
                r.m,
            );
        }
        enc.memory_barrier_with_resources(&[&gpu.y_arena]);
        for (r, p) in reqs.iter().zip(&plans) {
            // swiglu(gate, up) in place over m*ffn elements.
            enc.set_compute_pipeline_state(&gpu.pipelines[&("swiglu", 1, 1)]);
            enc.set_buffer(0, Some(&gpu.y_arena), p.gate_off as u64);
            enc.set_buffer(1, Some(&gpu.y_arena), p.up_off as u64);
            let n32 = (r.m * r.ffn) as u32;
            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
            let per_group = 256u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n32 as u64).div_ceil(per_group), 1, 1),
                MTLSize::new(per_group, 1, 1),
            );
            dispatched += 1;
        }
        enc.memory_barrier_with_resources(&[&gpu.y_arena]);
        for (((r, p), (_, _, down)), st) in reqs.iter().zip(&plans).zip(&mats).zip(&states) {
            dispatched += encode_matvec(
                enc,
                &gpu,
                down,
                &st[2],
                r.ffn,
                r.hidden,
                &gpu.y_arena,
                p.gate_off,
                &gpu.y_arena,
                p.out_off,
                r.m,
            );
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(dispatched, Ordering::Relaxed);

        let yp = gpu.y_arena.contents() as *const u8;
        reqs.iter()
            .zip(&plans)
            .map(|(r, p)| {
                let mut v = vec![0f32; r.m * r.hidden];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        yp.add(p.out_off),
                        v.as_mut_ptr() as *mut u8,
                        v.len() * 4,
                    );
                }
                v
            })
            .collect()
    });
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{tiles_for, TILES};

    #[test]
    fn tiles_cover_every_row_exactly_once() {
        for m in 0..40 {
            let tiles = tiles_for(m);
            let mut at = 0;
            for &(start, rows) in &tiles {
                assert_eq!(start, at, "tiles for {m} leave a gap or overlap");
                assert!(TILES.contains(&rows), "{rows} has no compiled kernel");
                at += rows;
            }
            assert_eq!(at, m, "tiles for {m} do not reach the last row");
        }
    }

    #[test]
    fn tiles_prefer_the_widest_kernel() {
        // A dispatch per 8 rows plus an exact tail - never 8 one-row passes
        // over the same weights.
        let widest = *TILES.last().unwrap();
        assert_eq!(tiles_for(widest), vec![(0, widest)]);
        assert_eq!(tiles_for(11), vec![(0, 8), (8, 2), (10, 1)]);
        assert_eq!(tiles_for(16), vec![(0, 8), (8, 8)]);
    }
}
