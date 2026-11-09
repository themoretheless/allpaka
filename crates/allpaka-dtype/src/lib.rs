//! Core tensor data types, quantization block geometry, and SIMD float conversions.
//!
//! Centralizes primitive format conversions (`f16`, `bf16`, `f32`), `GgmlType`
//! geometry, K-quant scale unpacking, and MLX affine block dequantization so
//! `allpaka-gguf`, `allpaka-mlx`, and `allpaka-backend` share a single
//! zero-duplication foundation.

use anyhow::{bail, Result};

/// A tensor quantization or floating-point dtype, with block geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    /// bfloat16 (ggml id 30): e.g. qwen35moe MTP router weights.
    BF16,
    Q5_0,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    /// MLX affine group quant, 4 bits per element in a 64-element group: eight
    /// little-endian `u32` nibble words, then one bfloat16 scale and one
    /// bfloat16 bias, value `q * scale + bias`.
    MlxQ4,
    /// MLX affine group quant at 8 bits: 64 payload bytes, bf16 scale, bf16 bias.
    MlxQ8,
    Other(u32),
}

impl GgmlType {
    #[inline]
    pub const fn from_id(id: u32) -> Self {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            30 => GgmlType::BF16,
            6 => GgmlType::Q5_0,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            other => GgmlType::Other(other),
        }
    }

    /// Elements per quantisation block.
    #[inline]
    pub const fn block_elements(self) -> Option<u64> {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => Some(1),
            GgmlType::Q5_0 | GgmlType::Q8_0 => Some(32),
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => {
                Some(256)
            }
            GgmlType::MlxQ4 | GgmlType::MlxQ8 => Some(64),
            GgmlType::Other(_) => None,
        }
    }

    /// Bytes per quantisation block.
    #[inline]
    pub const fn block_bytes(self) -> Option<u64> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 | GgmlType::BF16 => Some(2),
            GgmlType::Q8_0 => Some(34),
            GgmlType::Q5_0 => Some(22),
            GgmlType::Q2K => Some(84),
            GgmlType::Q3K => Some(110),
            GgmlType::Q4K => Some(144),
            GgmlType::Q5K => Some(176),
            GgmlType::Q6K => Some(210),
            GgmlType::MlxQ4 => Some(36),
            GgmlType::MlxQ8 => Some(68),
            GgmlType::Other(_) => None,
        }
    }

    /// Compute exact byte length required for `elements` of this dtype.
    #[inline]
    pub fn expected_bytes(self, elements: usize) -> Result<usize> {
        let (be, bb) = match (self.block_elements(), self.block_bytes()) {
            (Some(be), Some(bb)) => (be as usize, bb as usize),
            _ => bail!("ggml type {self:?} is not supported"),
        };
        if elements % be != 0 {
            bail!("{elements} elements is not a whole number of {be}-element blocks");
        }
        Ok(elements / be * bb)
    }
}

/// IEEE 754 half (`f16`) to single (`f32`). Handles subnormals, infinities and NaN.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, f) => {
            let p = 31 - f.leading_zeros();
            let exp = 127 - 24 + p;
            let frac = (f << (23 - p)) & 0x7f_ffff;
            (sign << 31) | (exp << 23) | frac
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, f) => (sign << 31) | 0x7f80_0000 | (f << 13),
        (e, f) => (sign << 31) | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// Single (`f32`) to IEEE 754 half (`f16`), round-to-nearest-even, with overflow
/// to infinity and subnormals preserved.
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mag = bits & 0x7fff_ffff;
    if mag >= 0x7f80_0000 {
        let man = if mag > 0x7f80_0000 { 0x200 } else { 0 };
        return sign | 0x7c00 | man;
    }
    let exp = (mag >> 23) as i32 - 127;
    if exp >= 16 {
        return sign | 0x7c00;
    }
    if exp >= -14 {
        let man = mag & 0x7f_ffff;
        let mut h = (((exp + 15) as u16) << 10) | (man >> 13) as u16;
        let rest = man & 0x1fff;
        if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    if exp < -25 {
        return sign;
    }
    let man = (mag & 0x7f_ffff) | 0x80_0000;
    let shift = (-exp - 14) as u32;
    let mut h = (man >> (13 + shift)) as u16;
    let rest = man & ((1 << (13 + shift)) - 1);
    let halfway = 1u32 << (12 + shift);
    if rest > halfway || (rest == halfway && (h & 1) == 1) {
        h += 1;
    }
    sign | h
}

/// bfloat16 (`u16`) to single (`f32`).
#[inline]
pub const fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Single (`f32`) to bfloat16 (`u16`), round-to-nearest-even.
#[inline]
pub const fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Copy little-endian `f32` bytes into `dst`. Uses bulk `copy_nonoverlapping` on little-endian targets.
#[inline]
pub fn f32_bytes_to_f32(bytes: &[u8], dst: &mut [f32]) {
    assert_eq!(bytes.len(), dst.len() * 4);
    #[cfg(target_endian = "little")]
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.as_mut_ptr() as *mut u8, bytes.len());
    }
    #[cfg(not(target_endian = "little"))]
    for (c, out) in bytes.chunks_exact(4).zip(dst.iter_mut()) {
        *out = f32::from_le_bytes(c.try_into().unwrap());
    }
}

/// Convert little-endian `f16` bytes into `dst: &mut [f32]` using NEON / F16C SIMD when available.
#[allow(clippy::incompatible_msrv)]
pub fn f16_bytes_to_f32(bytes: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    assert_eq!(bytes.len(), n * 2);
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let ptr = bytes.as_ptr() as *const u16;
        while i + 8 <= n {
            let u = vld1q_u16(ptr.add(i));
            let h0: float16x4_t = std::mem::transmute(vget_low_u16(u));
            let h1: float16x4_t = std::mem::transmute(vget_high_u16(u));
            vst1q_f32(dst.as_mut_ptr().add(i), vcvt_f32_f16(h0));
            vst1q_f32(dst.as_mut_ptr().add(i + 4), vcvt_f32_f16(h1));
            i += 8;
        }
        while i + 4 <= n {
            let h: float16x4_t = std::mem::transmute(vld1_u16(ptr.add(i)));
            vst1q_f32(dst.as_mut_ptr().add(i), vcvt_f32_f16(h));
            i += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") {
            #[target_feature(enable = "f16c")]
            unsafe fn convert_f16c(bytes: &[u8], dst: &mut [f32], i: &mut usize) {
                use std::arch::x86_64::*;
                let n = dst.len();
                let ptr = bytes.as_ptr();
                while *i + 8 <= n {
                    let h = _mm_loadu_si128(ptr.add(*i * 2) as *const __m128i);
                    let f = _mm256_cvtph_ps(h);
                    _mm256_storeu_ps(dst.as_mut_ptr().add(*i), f);
                    *i += 8;
                }
            }
            unsafe { convert_f16c(bytes, dst, &mut i) };
        }
    }

    for (c, out) in bytes[i * 2..].chunks_exact(2).zip(dst[i..].iter_mut()) {
        *out = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
    }
}

/// Convert little-endian `bf16` bytes into `dst: &mut [f32]` using NEON / AVX2 SIMD when available.
pub fn bf16_bytes_to_f32(bytes: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    assert_eq!(bytes.len(), n * 2);
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let ptr = bytes.as_ptr() as *const u16;
        while i + 8 <= n {
            let u = vld1q_u16(ptr.add(i));
            let lo = vshll_n_u16(vget_low_u16(u), 16);
            let hi = vshll_n_u16(vget_high_u16(u), 16);
            vst1q_f32(dst.as_mut_ptr().add(i), vreinterpretq_f32_u32(lo));
            vst1q_f32(dst.as_mut_ptr().add(i + 4), vreinterpretq_f32_u32(hi));
            i += 8;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            #[target_feature(enable = "avx2")]
            unsafe fn convert_avx2(bytes: &[u8], dst: &mut [f32], i: &mut usize) {
                use std::arch::x86_64::*;
                let n = dst.len();
                let ptr = bytes.as_ptr();
                while *i + 8 <= n {
                    let h = _mm_loadu_si128(ptr.add(*i * 2) as *const __m128i);
                    let wide = _mm256_cvtepu16_epi32(h);
                    let shifted = _mm256_slli_epi32(wide, 16);
                    _mm256_storeu_ps(dst.as_mut_ptr().add(*i), _mm256_castsi256_ps(shifted));
                    *i += 8;
                }
            }
            unsafe { convert_avx2(bytes, dst, &mut i) };
        }
    }

    for (c, out) in bytes[i * 2..].chunks_exact(2).zip(dst[i..].iter_mut()) {
        *out = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
    }
}

/// Convert `src: &[f32]` to `dst: &mut [u16]` (IEEE 754 half) using NEON / F16C when available.
#[allow(clippy::incompatible_msrv)]
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 8 <= src.len() {
            let v0 = vld1q_f32(src.as_ptr().add(i));
            let v1 = vld1q_f32(src.as_ptr().add(i + 4));
            let h0: uint16x4_t = std::mem::transmute(vcvt_f16_f32(v0));
            let h1: uint16x4_t = std::mem::transmute(vcvt_f16_f32(v1));
            vst1q_u16(dst.as_mut_ptr().add(i), vcombine_u16(h0, h1));
            i += 8;
        }
        while i + 4 <= src.len() {
            let v = vld1q_f32(src.as_ptr().add(i));
            let h: uint16x4_t = std::mem::transmute(vcvt_f16_f32(v));
            vst1_u16(dst.as_mut_ptr().add(i), h);
            i += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") {
            #[target_feature(enable = "f16c")]
            unsafe fn convert_f16c(src: &[f32], dst: &mut [u16], i: &mut usize) {
                use std::arch::x86_64::*;
                let n = src.len();
                while *i + 8 <= n {
                    let v = _mm256_loadu_ps(src.as_ptr().add(*i));
                    let h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm_storeu_si128(dst.as_mut_ptr().add(*i) as *mut __m128i, h);
                    *i += 8;
                }
            }
            unsafe { convert_f16c(src, dst, &mut i) };
        }
    }

    for (d, s) in dst[i..].iter_mut().zip(&src[i..]) {
        *d = f32_to_f16(*s);
    }
}

/// Unpack the 6-bit `(scale, min)` pair of sub-block `j` from the 12 packed
/// bytes shared by `Q4_K` and `Q5_K`.
#[inline]
pub fn scale_min_k4(j: usize, packed: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((packed[j] & 63) as f32, (packed[j + 4] & 63) as f32)
    } else {
        let sc = (packed[j + 4] & 0xf) | ((packed[j - 4] >> 6) << 4);
        let mn = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
        (sc as f32, mn as f32)
    }
}

/// Unpack 16 signed 6-bit scales (unbiased, caller subtracts 32 or interprets as `i8`)
/// from the 12 packed scale bytes of a `Q3_K` block using fast 32-bit SWAR operations.
#[inline]
pub fn unpack_q3_k_scales_raw(packed: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let a0 = u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
    let a1 = u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
    let tmp = u32::from_le_bytes([packed[8], packed[9], packed[10], packed[11]]);
    let u0 = (a0 & KMASK2) | (((tmp) & KMASK1) << 4);
    let u1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let u2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    let u3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    unsafe { std::mem::transmute::<[u32; 4], [i8; 16]>([u0, u1, u2, u3]) }
}

/// Dequantise one MLX affine 4-bit block (36 bytes: eight little-endian `u32` nibble words + bf16 scale + bf16 bias).
#[inline]
pub fn dequant_mlx_q4_block_slice(block: &[u8], out: &mut [f32]) {
    assert!(
        block.len() >= 36 && out.len() >= 64,
        "Q4Affine block requires 36 input bytes and 64 output elements"
    );
    let scale = bf16_to_f32(u16::from_le_bytes([block[32], block[33]]));
    let bias = bf16_to_f32(u16::from_le_bytes([block[34], block[35]]));
    let mut table = [0.0f32; 16];
    for (q, slot) in table.iter_mut().enumerate() {
        *slot = (q as f32).mul_add(scale, bias);
    }
    for w in 0..8 {
        let word = u32::from_le_bytes(block[w * 4..w * 4 + 4].try_into().unwrap());
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

/// Dequantise one MLX affine 8-bit block (68 bytes: 64 unsigned payload bytes + bf16 scale + bf16 bias).
#[inline]
pub fn dequant_mlx_q8_block_slice(block: &[u8], out: &mut [f32]) {
    assert!(
        block.len() >= 68 && out.len() >= 64,
        "Q8Affine block requires 68 input bytes and 64 output elements"
    );
    let scale = bf16_to_f32(u16::from_le_bytes([block[64], block[65]]));
    let bias = bf16_to_f32(u16::from_le_bytes([block[66], block[67]]));
    for (&q, slot) in block[..64].iter().zip(out[..64].iter_mut()) {
        *slot = (q as f32).mul_add(scale, bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_and_bf16_simd_slices_match_scalar() {
        let n = 77;
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 - 38.0) * 0.125).collect();
        let mut h = vec![0u16; n];
        f32_to_f16_slice(&vals, &mut h);
        for i in 0..n {
            assert_eq!(h[i], f32_to_f16(vals[i]));
        }

        let mut h_bytes = Vec::with_capacity(n * 2);
        for &v in &h {
            h_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut out = vec![0.0f32; n];
        f16_bytes_to_f32(&h_bytes, &mut out);
        for i in 0..n {
            assert_eq!(out[i], f16_to_f32(h[i]));
        }

        let mut bf_bytes = Vec::with_capacity(n * 2);
        for &v in &vals {
            let b = f32_to_bf16(v);
            bf_bytes.extend_from_slice(&b.to_le_bytes());
        }
        let mut bf_out = vec![0.0f32; n];
        bf16_bytes_to_f32(&bf_bytes, &mut bf_out);
        for i in 0..n {
            assert_eq!(bf_out[i], bf16_to_f32(f32_to_bf16(vals[i])));
        }
    }

    #[test]
    fn q3_k_swar_unpack_matches_reference() {
        let packed: [u8; 12] = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x1b, 0xe4, 0x27, 0xb1,
        ];
        let swar = unpack_q3_k_scales_raw(&packed);
        for (i, &got) in swar.iter().enumerate() {
            let lo = if i < 8 {
                packed[i] & 0xf
            } else {
                packed[i - 8] >> 4
            };
            let hi = (packed[8 + i % 4] >> (2 * (i / 4))) & 3;
            let expected = (lo | (hi << 4)) as i8;
            assert_eq!(got, expected, "scale {i}");
        }
    }
}
