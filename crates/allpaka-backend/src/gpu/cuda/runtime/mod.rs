//! CUDA device runtime: context, streams, weight residency, scratch arenas.

use crate::gpu::cuda::kernels::{EMBED_KERNELS, KERNELS};
use crate::gpu::cuda::pdl::{LaunchArgsPdl, StreamPdlExt};
use allpaka_gguf::GgmlType;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use half::f16;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub static CALLS: AtomicU64 = AtomicU64::new(0);
pub static DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub static ENCODE_NS: AtomicU64 = AtomicU64::new(0);
pub static WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);
pub static SCHED_NS: AtomicU64 = AtomicU64::new(0);

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
