//! CUDA device runtime: context, streams, weight residency, scratch arenas.

use crate::gpu::cuda::kernels::{EMBED_KERNELS, KERNELS};
use crate::gpu::cuda::pdl::{KernelFn, RawModule, StreamPdlExt};
use allpaka_dtype::GgmlType;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use half::f16;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub(crate) use crate::gpu::common::{CALLS, DISPATCHES, ENCODE_NS, WAIT_NS};

/// Batch size from which dequant + GEMM beats a loop of matvecs.
pub const MM_MIN_M: usize = 32;

mod device;
pub use device::*;
mod embed;
pub use embed::*;
mod gemm;
pub use gemm::*;
mod l2;
use l2::*;
mod matvec;
pub use matvec::*;
mod q8;
pub use q8::*;
mod select;
pub use select::*;
