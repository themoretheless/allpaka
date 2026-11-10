//! GPU acceleration dispatch.
//!
//! Exports the active GPU backend implementation (`metal` on macOS, `cuda` on
//! Linux/Windows with the `cuda` feature, or `stub` as the CPU-only fallback)
//! behind a unified module API (`allpaka_backend::gpu::*`).

pub mod types;

#[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
mod common;

#[cfg(target_os = "macos")]
mod metal_kernels;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use metal::*;

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
mod cuda;
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
pub use cuda::*;

#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
mod stub;
#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
pub use stub::*;
