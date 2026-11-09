//! CUDA GPU backend for Windows/Linux (feature `cuda`).
//!
//! Public surface matches `stub.rs` / Metal exactly. Weights are copied into
//! VRAM at attach time; KV and other shared regions are H2D-wrapped buffers.

mod attn;
mod decode;
mod decode_fused;
mod ffn_grouped;
mod ggml;
mod kernels;
mod pdl;
mod prefill;
mod region;
mod runtime;

use allpaka_gguf::GgmlType;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use pdl::{LaunchArgsPdl, StreamPdlExt};
use runtime::{
    dq_fmt, launch_embed_row, launch_embed_row_dtoken_ptr, launch_gemm_dequant, launch_inc_u32_ptr,
    launch_matvec_q4k_2, launch_matvec_q4k_qkv, launch_matvec_y_ptr, launch_store_u32_at_ptr,
    note_call, q8_decode_enabled, resolve_w, row_bytes, run_matvec, try_init, with_gpu, CudaGpu,
    CALLS, DISPATCHES, ENCODE_NS, GPU, GPU_BUSY_NS, MM_MIN_M, SCHED_NS, WAIT_NS,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

pub use super::types::*;

pub use attn::*;
pub use decode::*;
use decode_fused::*;
pub use ffn_grouped::*;
pub use prefill::*;
pub use region::*;

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

/// Upload NeoX RoPE inv_freq once so decode can skip per-token rope H2D.
pub fn set_rope_inv_freq(inv_freq: &[f32]) -> bool {
    if inv_freq.is_empty() {
        return false;
    }
    with_gpu(|gpu| {
        if gpu.d_rope_freq_n != inv_freq.len() || gpu.d_rope_freq.is_none() {
            let buf = gpu.stream.alloc_zeros::<f32>(inv_freq.len()).ok()?;
            gpu.d_rope_freq = Some(buf);
            gpu.d_rope_freq_n = inv_freq.len();
        }
        if let Some(buf) = gpu.d_rope_freq.as_mut() {
            gpu.stream.memcpy_htod(inv_freq, buf).ok()?;
        }
        // Rope inputs changed — drop graphs.
        gpu.decode_graph = None;
        gpu.decode_fa_graphs = None;
        gpu.decode_fa_attn = None;
        gpu.decode_fa_kv_offs = None;
        gpu.decode_fa_execs = None;
        gpu.decode_graph_key = 0;
        Some(true)
    })
    .unwrap_or(false)
}

/// True when decode can build NeoX rope from device inv_freq + d_pos.
pub fn has_device_rope_freq() -> bool {
    with_gpu(|gpu| Some(gpu.d_rope_freq.is_some() && gpu.d_rope_freq_n > 0)).unwrap_or(false)
}

/// Prefer on-GPU token_embd dequant (skip host embd.row + x H2D). Opt-in: ALLPAKA_GPU_EMBED=1.
pub fn prefer_gpu_embed() -> bool {
    is_attached()
        && std::env::var("ALLPAKA_GPU_EMBED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

pub fn can_embed_on_gpu(ty: GgmlType, bytes: &[u8], n_in: usize) -> bool {
    prefer_gpu_embed()
        && dq_fmt(ty).is_some()
        && row_bytes(ty, n_in).is_some()
        && with_gpu(|gpu| Some(resolve_w(gpu, bytes).is_some())).unwrap_or(false)
}

pub fn is_attached() -> bool {
    GPU.get().map_or(false, Option::is_some)
}

/// Active physical memory architecture for the CUDA backend.
///
/// Discrete NVIDIA GPUs operate in `dUMA` mode: device tensors and caches live
/// in dedicated VRAM (`CudaSlice`), separate from host RAM, requiring explicit
/// H2D and D2H transfers when state crosses between CPU and GPU.
#[inline]
pub fn memory_architecture() -> crate::memory::MemoryArchitecture {
    if is_attached() {
        crate::memory::MemoryArchitecture::Duma
    } else {
        crate::memory::MemoryArchitecture::HostOnly
    }
}

pub fn init_error() -> Option<String> {
    None
}

pub fn residency_status() -> (usize, bool) {
    let Some(Some(cell)) = GPU.get() else {
        return (0, false);
    };
    cell.lock()
        .map(|gpu| (gpu.chunks.len(), !gpu.chunks.is_empty()))
        .unwrap_or((0, false))
}

pub fn arena_bytes() -> usize {
    0
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
                    None,
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
            let mut gate = run_matvec(gpu, r.gate_ty, r.gate_w, r.hidden, r.ffn, r.x, r.m)?;
            let up = run_matvec(gpu, r.up_ty, r.up_w, r.hidden, r.ffn, r.x, r.m)?;
            // `run_matvec` already materialized `gate` and `up` on the host; on
            // dUMA, running elementwise SwiGLU in place on the CPU avoids two
            // redundant H2D uploads and one D2H download across PCIe before `down`.
            crate::ops::swiglu(&mut gate, &up);
            let down = run_matvec(gpu, r.down_ty, r.down_w, r.ffn, r.hidden, &gate, r.m)?;
            outs.push(down);
        }
        Some(outs)
    })
}

pub fn shared_activation_stats() -> crate::activation::SharedStats {
    crate::activation::SharedStats::default()
}
