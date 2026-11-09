//! Metal Shading Language (MSL) compute kernel sources.
//!
//! Externalized into `shaders/kernels.metal` and embedded at compile time.

pub const KERNELS: &str = include_str!("shaders/kernels.metal");
