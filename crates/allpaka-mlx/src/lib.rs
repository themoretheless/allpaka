//! MLX weight container format (safetensors) and affine quantization.
//!
//! Two layers:
//! * Container — read a list of `.safetensors` files, parse the metadata header,
//!   produce a flat tensor table with data offsets. This is analogous to
//!   `GgufFile`, but supports multi-part safetensors index.json layouts.
//! * Quant formats — affine group quantization (MLX "affine" mode): 4-bit / 8-bit
//!   weights stored as little-endian u32 words plus one bfloat16 scale and one
//!   bfloat16 bias per group. Dequantize row-by-row to f32 for the CPU reference;
//!   GPU kernels can operate directly on the packed layout.
//! * FFI bindings — raw declarations to libmlxc.dylib for runtime ops like
//!   repacking weights into allpaka blocks and loading tensors sparsely.

mod container;
mod dequant;
mod ffi;
mod shapes;

pub use container::{MlxSafetensors, TensorInfo};
pub use dequant::{bf16_to_f32, dequant_mlx_q4_block, dequant_mlx_q8_block};
pub use ffi::{free_mlx_blob, GgmlType, MlxBlob, mlx_dtype_t};
pub use shapes::{flatten_for_matmul, MlxShape};

/// A quantized tensor dtype in MLX's affine style. Matches GgmlType variants
/// MlxQ4/MlxQ8 added in allpaka-gguf so we can repack into a single block and
/// share with every other quantised matrix in the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxQuantType {
    /// eight 8-bit elements in a u32 word, 64 elements per group, bf16 scale/bias.
    Q4Affine,
    /// sixty-four 8-bit elements in a u32 word, 64 elements per group, bf16 scale/bias.
    Q8Affine,
}

impl MlxQuantType {
    pub fn bits(self) -> u32 {
        match self {
            MlxQuantType::Q4Affine => 4,
            MlxQuantType::Q8Affine => 8,
        }
    }

    pub fn bytes_per_block(self, group_size: usize) -> usize {
        let el = group_size;
        let words_per_group = (el + self.elements_per_word() - 1) / self.elements_per_word();
        words_per_group * 4 + 4 // u16 scale + u16 bias (stored as bf16)
    }

    pub fn elements_per_word(&self) -> usize {
        match self {
            MlxQuantType::Q4Affine => 8,
            MlxQuantType::Q8Affine => 4,
        }
    }
}

/// Parse an MLX checkpoint directory containing `*.safetensors`.
pub fn load_safetensors<P>(path: P) -> anyhow::Result<MlxSafetensors>
where
    P: AsRef<std::path::Path>,
{
    container::load(path.as_ref())
}
