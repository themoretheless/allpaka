//! Metal Shading Language kernel library source.
//!
//! `shaders/kernels.metal` is the single source of truth for all Metal compute
//! kernels; keeping it in a standalone `.metal` file eliminates duplicate
//! copies and enables offline syntax/compiler inspection with `xcrun metal`.

pub const KERNELS: &str = include_str!("shaders/kernels.metal");
