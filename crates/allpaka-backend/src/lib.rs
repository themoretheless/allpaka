//! Tensor operations for the engine.
//!
//! This is the **CPU reference**: written for correctness you can read, not
//! for speed. Every accelerated implementation (Metal, CUDA, SIMD) is checked
//! against these functions on random inputs before it is trusted; when a fast
//! kernel and this file disagree, this file wins until proven wrong against
//! llama.cpp logits.
//!
//! Conventions, chosen to match how GGUF stores weights:
//!
//! * Activations are `f32` slices, row-major, shape written `[rows, cols]`.
//! * A weight matrix is `[n_out, n_in]` with `n_in` contiguous (GGUF
//!   `dims[0]`), so `matmul` computes `y = x · Wᵀ` - one dot product of a
//!   weight row with an activation row per output element, exactly ggml's
//!   `mul_mat` semantics.

pub mod accel;
pub mod capability;
pub mod command;
pub mod execution;
pub mod memory;
pub mod profile;
pub mod gpu;
pub mod ops;
pub mod quantmat;
pub mod runtime;
pub mod telemetry;

pub use ops::{matmul_f32, rmsnorm, rope_neox, rope_norm, silu, softmax, swiglu};
pub use quantmat::QuantMat;
