//! Per-format kernel names, launch geometry and dequant format codes.
//! The codes are the CUDA kernel ABI and differ from the Metal ones.

use super::*;

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

pub fn row_bytes(ty: GgmlType, n_in: usize) -> Option<usize> {
    let be = ty.block_elements()? as usize;
    let bb = ty.block_bytes()? as usize;
    if n_in % be != 0 {
        return None;
    }
    Some(n_in / be * bb)
}

pub fn dq_fmt(ty: GgmlType) -> Option<u32> {
    Some(match ty {
        GgmlType::Q4K => 0,
        GgmlType::Q5K => 1,
        GgmlType::Q6K => 2,
        GgmlType::Q8_0 => 3,
        GgmlType::Q2K => 4,
        GgmlType::Q3K => 5,
        GgmlType::Q5_0 => 6,
        GgmlType::F16 => 7,
        GgmlType::F32 => 8,
        GgmlType::BF16 => 9,
        GgmlType::MlxQ4 | GgmlType::MlxQ8 | GgmlType::Other(_) => return None,
    })
}

pub fn resolve_w(gpu: &CudaGpu, w: &[u8]) -> Option<(usize, u64)> {
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some((chunk, (addr - gpu.chunks[chunk].start) as u64))
}
