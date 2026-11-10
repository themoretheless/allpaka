//! Per-format kernel names, launch geometry and shared dequant format codes.

use super::*;

pub use crate::gpu::common::{dq_fmt, embed_row_groups, row_bytes};

pub fn matvec_kernel(ty: GgmlType) -> Option<&'static str> {
    Some(match ty {
        GgmlType::F32 => "matvec_f32",
        GgmlType::F16 => "matvec_f16",
        GgmlType::BF16 => "matvec_bf16",
        GgmlType::Q5_0 => "matvec_q5_0",
        GgmlType::Q8_0 => "matvec_q8_0",
        GgmlType::Q2K => "matvec_q2_k",
        GgmlType::Q3K => "matvec_q3_k",
        GgmlType::Q4K => "matvec_q4_k",
        GgmlType::Q5K => "matvec_q5_k",
        GgmlType::Q6K => "matvec_q6_k",
        GgmlType::MlxQ4 | GgmlType::MlxQ8 | GgmlType::Other(_) => return None,
    })
}

/// Q4/Q6_K: Metal-style — 4 warps × 4 output rows per block.
/// Q5_K: one warp per output row, several rows per block.
#[allow(dead_code)]
pub fn matvec_warp_rows(ty: GgmlType) -> Option<u32> {
    match ty {
        GgmlType::Q4K | GgmlType::Q6K => Some(4),
        GgmlType::Q5K => Some(8),
        _ => None,
    }
}

pub fn matvec_launch_cfg(ty: GgmlType, n_out: u32, m: u32) -> LaunchConfig {
    match ty {
        // Q4/Q6_K: 4 warps × 4 rows = 16 outs/block (best Metal-style decode).
        GgmlType::Q4K | GgmlType::Q6K => LaunchConfig {
            grid_dim: (n_out.div_ceil(16), m, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        },
        GgmlType::Q5K => LaunchConfig {
            grid_dim: (n_out.div_ceil(8), m, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        },
        _ => {
            let threads = 256u32;
            LaunchConfig {
                grid_dim: (n_out.div_ceil(threads), m, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            }
        }
    }
}

pub fn resolve_w(gpu: &CudaGpu, w: &[u8]) -> Option<(usize, u64)> {
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some((chunk, (addr - gpu.chunks[chunk].start) as u64))
}
