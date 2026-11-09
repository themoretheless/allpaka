//! Dequantisation of MLX affine quantization to f32, backed by `allpaka-dtype`.

pub use allpaka_dtype::bf16_to_f32;

/// Dequantise one Q4Affine block (36 bytes: eight little-endian u32 words + bf16 scale + bf16 bias).
#[inline]
pub fn dequant_mlx_q4_block_slice(block: &[u8], out: &mut [f32; 64]) {
    allpaka_dtype::dequant_mlx_q4_block_slice(block, out);
}

#[inline]
pub fn dequant_mlx_q4_block(block: &[u8], out: &mut Vec<f32>) {
    let base = out.len();
    out.resize(base + 64, 0.0);
    allpaka_dtype::dequant_mlx_q4_block_slice(block, &mut out[base..base + 64]);
}

/// Dequantise one Q8Affine block (68 bytes: 64 bytes payload + bf16 scale + bf16 bias).
#[inline]
pub fn dequant_mlx_q8_block_slice(block: &[u8], out: &mut [f32; 64]) {
    allpaka_dtype::dequant_mlx_q8_block_slice(block, out);
}

#[inline]
pub fn dequant_mlx_q8_block(block: &[u8], out: &mut Vec<f32>) {
    let base = out.len();
    out.resize(base + 64, 0.0);
    allpaka_dtype::dequant_mlx_q8_block_slice(block, &mut out[base..base + 64]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_one() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x3f00), 0.5);
    }

    #[test]
    fn test_q4_block_minimal() {
        let mut block = [0u8; 36];
        block[32..34].copy_from_slice(&0x3f80u16.to_le_bytes());
        block[34..36].copy_from_slice(&0u16.to_le_bytes());
        let mut out = Vec::with_capacity(64);
        dequant_mlx_q4_block(&block, &mut out);
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_q4_block_element_order() {
        let mut block = [0u8; 36];
        block[32..34].copy_from_slice(&0x3f80u16.to_le_bytes());
        block[34..36].copy_from_slice(&0u16.to_le_bytes());
        block[0..4].copy_from_slice(&0x8765_4321u32.to_le_bytes());
        block[4..8].copy_from_slice(&0x0000_0009u32.to_le_bytes());

        let mut out = Vec::with_capacity(64);
        dequant_mlx_q4_block(&block, &mut out);
        assert_eq!(&out[0..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(out[8], 9.0);
    }

    #[test]
    fn test_q8_block_minimal() {
        let mut block = [0u8; 68];
        block[64..66].copy_from_slice(&0x3f80u16.to_le_bytes());
        block[66..68].copy_from_slice(&0u16.to_le_bytes());
        block[..64].fill(128u8);

        let mut out = Vec::with_capacity(64);
        dequant_mlx_q8_block(&block, &mut out);
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&v| v == 128.0));
    }
}
