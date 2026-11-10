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

pub(crate) use crate::gpu::common::row_bytes;

/// `fmt` codes the CUDA dequant kernels switch on. They must match the
/// `// fmt:` line above `dequant_rows_f16` in `kernels.rs`; a test parses it.
/// Metal numbers its formats differently (`metal::decode::fmt`).
pub mod fmt {
    pub const Q4K: u32 = 0;
    pub const Q5K: u32 = 1;
    pub const Q6K: u32 = 2;
    pub const Q8_0: u32 = 3;
    pub const Q2K: u32 = 4;
    pub const Q3K: u32 = 5;
    pub const Q5_0: u32 = 6;
    pub const F16: u32 = 7;
    pub const F32: u32 = 8;
    pub const BF16: u32 = 9;
}

pub fn dq_fmt(ty: GgmlType) -> Option<u32> {
    Some(match ty {
        GgmlType::Q4K => fmt::Q4K,
        GgmlType::Q5K => fmt::Q5K,
        GgmlType::Q6K => fmt::Q6K,
        GgmlType::Q8_0 => fmt::Q8_0,
        GgmlType::Q2K => fmt::Q2K,
        GgmlType::Q3K => fmt::Q3K,
        GgmlType::Q5_0 => fmt::Q5_0,
        GgmlType::F16 => fmt::F16,
        GgmlType::F32 => fmt::F32,
        GgmlType::BF16 => fmt::BF16,
        GgmlType::MlxQ4 | GgmlType::MlxQ8 | GgmlType::Other(_) => return None,
    })
}

#[cfg(test)]
mod dq_fmt_tests {
    use super::*;

    /// The kernel source documents its codes as `// fmt: 0=q4_k 1=q5_k ...`;
    /// every type the host maps must sit at the same code there.
    #[test]
    fn codes_match_the_kernel_comment() {
        let line = crate::gpu::cuda::kernels::KERNELS
            .lines()
            .find(|l| l.trim_start().starts_with("// fmt: 0="))
            .expect("kernels.rs documents the dequant fmt codes");
        let documented: Vec<(u32, &str)> = line
            .trim_start()
            .trim_start_matches("// fmt:")
            .split_whitespace()
            .map(|pair| {
                let (code, name) = pair.split_once('=').unwrap();
                (code.parse().unwrap(), name)
            })
            .collect();
        let host = [
            (GgmlType::Q4K, "q4_k"),
            (GgmlType::Q5K, "q5_k"),
            (GgmlType::Q6K, "q6_k"),
            (GgmlType::Q8_0, "q8_0"),
            (GgmlType::Q2K, "q2_k"),
            (GgmlType::Q3K, "q3_k"),
            (GgmlType::Q5_0, "q5_0"),
            (GgmlType::F16, "f16"),
            (GgmlType::F32, "f32"),
            (GgmlType::BF16, "bf16"),
        ];
        assert_eq!(documented.len(), host.len());
        for (ty, name) in host {
            let code = dq_fmt(ty).unwrap();
            assert!(
                documented.contains(&(code, name)),
                "{ty:?} maps to {code}, kernel comment says {line}"
            );
        }
        for ty in [GgmlType::MlxQ4, GgmlType::MlxQ8, GgmlType::Other(99)] {
            assert_eq!(dq_fmt(ty), None);
        }
    }
}

pub fn resolve_w(gpu: &CudaGpu, w: &[u8]) -> Option<(usize, u64)> {
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some((chunk, (addr - gpu.chunks[chunk].start) as u64))
}
