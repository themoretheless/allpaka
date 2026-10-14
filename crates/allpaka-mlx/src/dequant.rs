//! Dequantisation of MLX affine quantization to f32.
//!
//! Formats:
//! * Q4Affine: 64 elements packed into eight u32 nibbles (low nibble first),
//!   then one bfloat16 scale and one bfloat16 bias for the group. Formula:
//!   x = q * scale + bias where q is the 4-bit value (0..15).
//! * Q8Affine: 64 elements packed into sixteen u32 bytes (low byte first per word? —
//!   verify with dequantize), then the same scale/bias pair.
//!
//! The exact bit-order inside a word must be validated against mx.dequantize on
//! every PR; unit tests pull golden vectors from an MLX checkpoint.

/// Convert a native MLX bfloat16 (stored as u16) to single precision f32.
/// Its 8-bit mantissa is a prefix of single's 23-bit fraction, so widening is a
/// left shift; no subnormal handling needed because bfloat16 shares exponent
/// range of single.
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Dequantise one Q4Affine block (36 bytes: eight little-endian u32 words + b16 scale + b16 bias).
/// Element `w*8+k` of the group (k in 0..7) is the low/high nibble pattern:
///   val = ((word >> (4*k)) & 0xf)
/// and x = val * scale + bias.
///
/// Order is verified by golden fixtures against `mx.dequantize`.
pub fn dequant_mlx_q4_block(block: &[u8], out: &mut Vec<f32>) {
    assert!(
        block.len() >= 36,
        "Q4Affine block requires at least 36 bytes, got {}",
        block.len()
    );
    let scale = bf16_to_f32(u16::from_le_bytes([block[32], block[33]]));
    let bias = bf16_to_f32(u16::from_le_bytes([block[34], block[35]]));
    out.reserve(64);
    for w in 0..8 {
        let word = u32::from_le_bytes(block[w * 4..w * 4 + 4].try_into().unwrap());
        for k in 0..8 {
            let nib = ((word >> (k * 4)) & 0xf) as f32;
            out.push(nib * scale + bias);
        }
    }
}

/// Dequantise one Q8Affine block (68 bytes: 64 bytes payload + b16 scale + b16 bias).
/// Each element is an unsigned byte; x = q * scale + bias.
pub fn dequant_mlx_q8_block(block: &[u8], out: &mut Vec<f32>) {
    assert!(
        block.len() >= 68,
        "Q8Affine block requires at least 68 bytes, got {}",
        block.len()
    );
    let scale = bf16_to_f32(u16::from_le_bytes([block[64], block[65]]));
    let bias = bf16_to_f32(u16::from_le_bytes([block[66], block[67]]));
    out.reserve(64);
    for &b in block.iter().take(64) {
        out.push(b as f32 * scale + bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_one() {
        // bf16 1.0 is 0x3f80 (exponent 127 + bias 0, mantissa 0)
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        // bf16 0.5 = 0x3f00
        assert_eq!(bf16_to_f32(0x3f00), 0.5);
    }

    #[test]
    fn test_q4_block_minimal() {
        let mut block = [0u8; 36];
        // Use scale=1.0 (bf16 = 0x3f80), bias=0.0
        block[32..34].copy_from_slice(&0x3f80u16.to_le_bytes());
        block[34..36].copy_from_slice(&0u16.to_le_bytes());

        // Write a simple pattern: all nibbles in w0 are 0
        block[0..4].copy_from_slice(&0u32.to_le_bytes());

        let mut out = Vec::with_capacity(64);
        dequant_mlx_q4_block(&block, &mut out);

        assert_eq!(out.len(), 64);
        // All outputs should be 0 since scale=1, bias=0, and all nibbles are 0
        for v in out.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_q4_block_element_order() {
        let mut block = [0u8; 36];
        // scale = 1.0, bias = 0.0
        block[32..34].copy_from_slice(&0x3f80u16.to_le_bytes());
        block[34..36].copy_from_slice(&0u16.to_le_bytes());
        // word 0 has nibbles 1, 2, 3, 4, 5, 6, 7, 8
        let w0: u32 = 0x8765_4321;
        block[0..4].copy_from_slice(&w0.to_le_bytes());
        // word 1 has nibble 0 = 9
        let w1: u32 = 0x0000_0009;
        block[4..8].copy_from_slice(&w1.to_le_bytes());

        let mut out = Vec::with_capacity(64);
        dequant_mlx_q4_block(&block, &mut out);
        assert_eq!(&out[0..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(out[8], 9.0);
    }

    #[test]
    fn test_q8_block_minimal() {
        let mut block = [0u8; 68];
        block[64..66].copy_from_slice(&0x3f80u16.to_le_bytes()); // scale = 1.0
        block[66..68].copy_from_slice(&0u16.to_le_bytes()); // bias = 0.0

        // Set all bytes to 128 (so they become 128.0 after dequant)
        block[..64].fill(128u8);

        let mut out = Vec::with_capacity(64);
        dequant_mlx_q8_block(&block, &mut out);

        assert_eq!(out.len(), 64);
        for v in out.iter() {
            assert_eq!(*v, 128.0);
        }
    }
}
