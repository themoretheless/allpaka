//! Dequantisation of MLX affine quantization to f32.

pub use allpaka_dtype::bf16_to_f32;

/// Dequantise one Q4Affine block (36 bytes: eight little-endian u32 words + bf16 scale + bf16 bias).
#[inline]
pub fn dequant_mlx_q4_block_slice(block: &[u8], out: &mut [f32; 64]) {
    assert!(
        block.len() >= 36,
        "Q4Affine block requires at least 36 input bytes"
    );
    let scale = bf16_to_f32(u16::from_le_bytes([block[32], block[33]]));
    let bias = bf16_to_f32(u16::from_le_bytes([block[34], block[35]]));
    let mut table = [0.0f32; 16];
    for (q, slot) in table.iter_mut().enumerate() {
        *slot = (q as f32).mul_add(scale, bias);
    }
    for w in 0..8 {
        let word = u32::from_le_bytes([
            block[w * 4],
            block[w * 4 + 1],
            block[w * 4 + 2],
            block[w * 4 + 3],
        ]);
        let dst = &mut out[w * 8..w * 8 + 8];
        dst[0] = table[(word & 0xf) as usize];
        dst[1] = table[((word >> 4) & 0xf) as usize];
        dst[2] = table[((word >> 8) & 0xf) as usize];
        dst[3] = table[((word >> 12) & 0xf) as usize];
        dst[4] = table[((word >> 16) & 0xf) as usize];
        dst[5] = table[((word >> 20) & 0xf) as usize];
        dst[6] = table[((word >> 24) & 0xf) as usize];
        dst[7] = table[((word >> 28) & 0xf) as usize];
    }
}

#[inline]
pub fn dequant_mlx_q4_block(block: &[u8], out: &mut Vec<f32>) {
    let base = out.len();
    out.resize(base + 64, 0.0);
    let dst: &mut [f32; 64] = (&mut out[base..base + 64]).try_into().unwrap();
    dequant_mlx_q4_block_slice(block, dst);
}

/// Dequantise one Q8Affine block (68 bytes: 64 bytes payload + bf16 scale + bf16 bias).
#[inline]
pub fn dequant_mlx_q8_block_slice(block: &[u8], out: &mut [f32; 64]) {
    assert!(
        block.len() >= 68,
        "Q8Affine block requires at least 68 input bytes"
    );
    let scale = bf16_to_f32(u16::from_le_bytes([block[64], block[65]]));
    let bias = bf16_to_f32(u16::from_le_bytes([block[66], block[67]]));
    for (&q, slot) in block[..64].iter().zip(out.iter_mut()) {
        *slot = (q as f32).mul_add(scale, bias);
    }
}

#[inline]
pub fn dequant_mlx_q8_block(block: &[u8], out: &mut Vec<f32>) {
    let base = out.len();
    out.resize(base + 64, 0.0);
    let dst: &mut [f32; 64] = (&mut out[base..base + 64]).try_into().unwrap();
    dequant_mlx_q8_block_slice(block, dst);
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
