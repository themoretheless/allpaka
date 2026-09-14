//! GPU acceleration: Metal on macOS, CUDA elsewhere when the `cuda` feature
//! is enabled, otherwise a decline-only stub.

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
mod cuda;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(not(any(target_os = "macos", feature = "cuda")))]
mod stub;

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
pub use cuda::*;
#[cfg(target_os = "macos")]
pub use metal::*;
#[cfg(not(any(target_os = "macos", feature = "cuda")))]
pub use stub::*;
