//! FFI bindings to libmlx for runtime operations.
//!
//! This module provides raw FFI declarations for calling into the MLX C API
//! (libmlxc.dylib). These are the only functions needed for allpaka:
//! - Quantization helpers that repack MLX group-affine weights into allpaka blocks
//! - Gather/mm ops that dispatch to quantized kernels on the GPU
//!
//! Note: bindgen is not used here — the interface is minimal and stable.
//! See /opt/homebrew/include/mlx/mlx.h for the upstream declarations.

#[repr(C)]
pub struct MlxBlob {
    pub data: *mut std::ffi::c_void,
    pub shape: *const u64,
    pub ndim: usize,
    pub dtype: mlx_dtype_t,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum mlx_dtype_t {
    FLOAT32 = 0,
    BFLOAT16 = 1,
    FLOAT16 = 2,
    Q8_0 = 8,   // reusing GGML type id for compatibility
    Q4_0 = 4,
    MLX_Q4_AFFINE = 100,
    MLX_Q8_AFFINE = 101,
}

// Repack MLX weight bytes into allpaka-compatible quantized blocks.
// src_data points to dequantised MLX block(s); dst writes packed GgmlType blocks.
#[cfg(feature = "libmlx")]
#[link(name = "mlx")]
extern "C" {
    /// Repack a buffer of f32 values from MLX layout into allpaka block format.
    /// The caller owns both buffers; we just change representation.
    pub fn mlx_repack_to_allpaka(
        src: *const f32,
        dst: *mut std::ffi::c_void,      // points to an allocated BlockBuffer
        n_elements: usize,
        ty: GgmlType,
    ) -> *mut std::ffi::c_void;       // returns same pointer on success, null on error

    /// Load a safetensors tensor directly into memory without full dequantize.
    /// Returns a Blob ready for GPU-only quantized matmul paths.
    pub fn mlx_load_tensor_sparse(
        tensor_name: *const std::ffi::c_char,
        layout_ptr: *const std::ffi::c_void, // MlxSafetensors* opaque
    ) -> *mut MlxBlob;                   // blob owned by caller, must free_mlx_blob

    /// Free a blob returned by mlx_load_tensor_sparse.
    pub fn free_mlx_blob(blob: *mut MlxBlob);
}

// Stubs when libmlx is not available
#[cfg(not(feature = "libmlx"))]
mod ffi_stubs {
    use super::*;
    
    #[inline]
    pub fn mlx_repack_to_allpaka(_src: *const f32, _dst: *mut std::ffi::c_void, _n_elements: usize, _ty: GgmlType) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }
    
    #[inline]
    pub fn mlx_load_tensor_sparse(_tensor_name: *const std::ffi::c_char, _layout_ptr: *const std::ffi::c_void) -> *mut MlxBlob {
        std::ptr::null_mut()
    }
    
    #[inline]
    pub unsafe fn free_mlx_blob(_blob: *mut MlxBlob) {}
}

#[cfg(not(feature = "libmlx"))]
pub use ffi_stubs::*;

/// Allpaka-side representation of GgmlType for FFI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q6_K = 14,
    MlxQ4 = 99,  // MLX affine 4-bit
    MlxQ8 = 100, // MLX affine 8-bit
}

impl From<GgmlType> for mlx_dtype_t {
    fn from(t: GgmlType) -> Self {
        match t {
            GgmlType::MlxQ4 => mlx_dtype_t::MLX_Q4_AFFINE,
            GgmlType::MlxQ8 => mlx_dtype_t::MLX_Q8_AFFINE,
            GgmlType::Q4_0 => mlx_dtype_t::Q4_0,
            GgmlType::Q8_0 => mlx_dtype_t::Q8_0,
            _ => mlx_dtype_t::FLOAT32,
        }
    }
}

#[cfg(all(test, feature = "libmlx"))]
mod tests {
    use super::*;

    #[test]
    fn test_ggmltype_conversion() {
        assert_eq!(mlx_dtype_t::from(GgmlType::MlxQ4), mlx_dtype_t::MLX_Q4_AFFINE);
        assert_eq!(mlx_dtype_t::from(GgmlType::MlxQ8), mlx_dtype_t::MLX_Q8_AFFINE);
    }
}
