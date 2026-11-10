//! Dequantisation of GGML block formats to f32.
//!
//! Layouts follow ggml's `ggml-common.h` / `dequantize_row_*` exactly; the
//! ordering of sub-blocks inside a K-quant superblock is deliberate on ggml's
//! side (it matches their SIMD kernels) and nothing here may "simplify" it.
//! Conformance is ultimately proven not by these unit tests but by comparing
//! whole-model logits against llama.cpp - that is the engine milestone's
//! acceptance test. The unit tests here pin the bit layout so a refactor
//! cannot silently move a shift.

use crate::tensors::GgmlType;
use allpaka_dtype::{
    bf16_bytes_to_f32, f16_bytes_to_f32, f32_bytes_to_f32, unpack_q3_k_scales_raw,
};
pub use allpaka_dtype::{bf16_to_f32, f16_to_f32, scale_min_k4};
use anyhow::{bail, Result};

/// Dequantise `data` holding `elements` values of type `ty`.
pub fn dequant(ty: GgmlType, data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; elements];
    dequant_into(ty, data, &mut out)?;
    Ok(out)
}

/// Dequantise `data` holding `dst.len()` values of type `ty` into an existing slice without heap allocations.
pub fn dequant_into(ty: GgmlType, data: &[u8], dst: &mut [f32]) -> Result<()> {
    let elements = dst.len();
    let expected = ty.expected_bytes(elements)?;
    if data.len() != expected {
        bail!(
            "expected {expected} bytes for {elements} elements of {ty:?}, got {}",
            data.len()
        );
    }
    match ty {
        GgmlType::F32 => f32_bytes_to_f32(data, dst),
        GgmlType::F16 => f16_bytes_to_f32(data, dst),
        GgmlType::BF16 => bf16_bytes_to_f32(data, dst),
        GgmlType::Q5_0 => {
            for (block, out) in data.chunks_exact(22).zip(dst.chunks_exact_mut(32)) {
                dequant_q5_0_block_slice(block, out);
            }
        }
        GgmlType::Q8_0 => {
            for (block, out) in data.chunks_exact(34).zip(dst.chunks_exact_mut(32)) {
                dequant_q8_0_block_slice(block, out);
            }
        }
        GgmlType::Q2K => {
            for (block, out) in data.chunks_exact(84).zip(dst.chunks_exact_mut(256)) {
                dequant_q2_k_block_slice(block, out);
            }
        }
        GgmlType::Q3K => {
            for (block, out) in data.chunks_exact(110).zip(dst.chunks_exact_mut(256)) {
                dequant_q3_k_block_slice(block, out);
            }
        }
        GgmlType::Q4K => {
            for (block, out) in data.chunks_exact(144).zip(dst.chunks_exact_mut(256)) {
                dequant_q4_k_block_slice(block, out);
            }
        }
        GgmlType::Q5K => {
            for (block, out) in data.chunks_exact(176).zip(dst.chunks_exact_mut(256)) {
                dequant_q5_k_block_slice(block, out);
            }
        }
        GgmlType::Q6K => {
            for (block, out) in data.chunks_exact(210).zip(dst.chunks_exact_mut(256)) {
                dequant_q6_k_block_slice(block, out);
            }
        }
        GgmlType::MlxQ4 | GgmlType::MlxQ8 => {
            bail!("mlx affine type {ty:?} is dequantised by allpaka-mlx")
        }
        GgmlType::Other(id) => bail!("ggml type id {id} is not supported"),
    }
    Ok(())
}

/// Q8_0: an f16 scale followed by 32 signed bytes. `x = d * q`.
fn dequant_q8_0_block_slice(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let vd = vdupq_n_f32(d);
        let q_ptr = block.as_ptr().add(2) as *const i8;
        for g in 0..2 {
            let q16 = vld1q_s8(q_ptr.add(g * 16));
            let lo8 = vmovl_s8(vget_low_s8(q16));
            let hi8 = vmovl_s8(vget_high_s8(q16));
            let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo8)));
            let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo8)));
            let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi8)));
            let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi8)));
            let dst = out.as_mut_ptr().add(g * 16);
            vst1q_f32(dst, vmulq_f32(vd, f0));
            vst1q_f32(dst.add(4), vmulq_f32(vd, f1));
            vst1q_f32(dst.add(8), vmulq_f32(vd, f2));
            vst1q_f32(dst.add(12), vmulq_f32(vd, f3));
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (&q, slot) in block[2..34].iter().zip(out.iter_mut()) {
        *slot = d * (q as i8) as f32;
    }
}

/// Q5_0: an f16 scale, a 32-bit high-bit mask, then 16 bytes of nibbles.
/// `q = nibble | (high_bit << 4)`, `x = d * (q - 16)`; the second 16 values
/// take the high nibbles, mirroring ggml's dequantize_row_q5_0.
fn dequant_q5_0_block_slice(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
    let qs = &block[6..22];
    for j in 0..16 {
        let x0 = ((qs[j] & 0x0f) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
        let x1 = ((qs[j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
        out[j] = d * x0 as f32;
        out[j + 16] = d * x1 as f32;
    }
}

/// Q2_K: a 256-element superblock of 16 sub-blocks with 4-bit scales and mins.
///
/// Layout: scales[16], qs[64], d (f16), dmin (f16). Each scale byte packs a
/// 4-bit scale (low nibble) and a 4-bit min (high nibble); the value is
/// `d*scale*q - dmin*min` with q a 2-bit quant. The qs walk in two 128-element
/// halves; within a half, each shift level (0,2,4,6) yields two consecutive
/// 16-element sub-blocks read from the same 32 bytes.
fn dequant_q2_k_block_slice(block: &[u8], out: &mut [f32]) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

    let mut is = 0; // sub-block index, 0..16
    let mut pos = 0;
    for half in 0..2 {
        let q = &qs[half * 32..half * 32 + 32];
        for shift in [0u8, 2, 4, 6] {
            for group in 0..2 {
                let sc = scales[is];
                let dl = d * (sc & 0xf) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    let quant = (q[group * 16 + l] >> shift) & 3;
                    out[pos + l] = dl * quant as f32 - ml;
                }
                pos += 16;
                is += 1;
            }
        }
    }
}

fn dequant_q3_k_block_slice(block: &[u8], out: &mut [f32]) {
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let packed = &block[96..108];
    let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

    let scales_raw = unpack_q3_k_scales_raw(packed);

    let mut is = 0; // sub-block index, 0..16
    let mut m: u8 = 1; // walking high-bit mask
    let mut pos = 0;
    for half in 0..2 {
        let q = &qs[half * 32..half * 32 + 32];
        for shift in [0u8, 2, 4, 6] {
            for group in 0..2 {
                let dl = d_all * (scales_raw[is] as i32 - 32) as f32;
                for l in 0..16 {
                    let idx = group * 16 + l;
                    let quant =
                        ((q[idx] >> shift) & 3) as i32 - if hmask[idx] & m != 0 { 0 } else { 4 };
                    out[pos + l] = dl * quant as f32;
                }
                pos += 16;
                is += 1;
            }
            m <<= 1;
        }
    }
}

/// Q4_K: a 256-element superblock of 4-bit quants in 8 sub-blocks of 32, each
/// with a 6-bit scale and 6-bit min: `d*scale*q - dmin*min`.
///
/// Layout: d (f16), dmin (f16), scales[12], qs[128]. Each 32-byte row of qs
/// serves two sub-blocks: low nibbles first, then high nibbles.
fn dequant_q4_k_block_slice(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let packed = &block[4..16];
    let qs = &block[16..144];

    for pair in 0..4 {
        let q = &qs[pair * 32..pair * 32 + 32];
        let (sc1, mn1) = scale_min_k4(pair * 2, packed);
        let (sc2, mn2) = scale_min_k4(pair * 2 + 1, packed);
        let d_sc1 = d * sc1;
        let dm1 = dmin * mn1;
        let d_sc2 = d * sc2;
        let dm2 = dmin * mn2;
        let base = pair * 64;
        for (l, &b) in q.iter().enumerate() {
            out[base + l] = d_sc1 * (b & 0xf) as f32 - dm1;
            out[base + 32 + l] = d_sc2 * (b >> 4) as f32 - dm2;
        }
    }
}

/// Q5_K: Q4_K plus one extra bit per value from a separate 32-byte mask,
/// giving 5-bit quants: `d*scale*(q4 + 16*bit) - dmin*min`.
///
/// Layout: d (f16), dmin (f16), scales[12], qh[32], qs[128]. The mask byte
/// `qh[l]` is shared by all eight sub-blocks; sub-block `j` owns bit `j`.
fn dequant_q5_k_block_slice(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let packed = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    for pair in 0..4 {
        let q = &qs[pair * 32..pair * 32 + 32];
        let (sc1, mn1) = scale_min_k4(pair * 2, packed);
        let (sc2, mn2) = scale_min_k4(pair * 2 + 1, packed);
        let d_sc1 = d * sc1;
        let dm1 = dmin * mn1;
        let d_sc2 = d * sc2;
        let dm2 = dmin * mn2;
        let bit1 = 1u8 << (pair * 2);
        let bit2 = 1u8 << (pair * 2 + 1);
        let base = pair * 64;
        for (l, &b) in q.iter().enumerate() {
            let hi1 = if qh[l] & bit1 != 0 { 16 } else { 0 };
            let hi2 = if qh[l] & bit2 != 0 { 16 } else { 0 };
            out[base + l] = d_sc1 * ((b & 0xf) + hi1) as f32 - dm1;
            out[base + 32 + l] = d_sc2 * ((b >> 4) + hi2) as f32 - dm2;
        }
    }
}

/// Q6_K: a 256-element superblock of 6-bit quants with 16 signed 8-bit scales.
///
/// Layout: ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] (i8),
/// d (f16). The value is `d * scale[sub] * (q - 32)`. Bits are interleaved in
/// two 128-element halves; in each half the four 32-element quarters take
/// their high bits from successive 2-bit fields of the same qh byte.
fn dequant_q6_k_block_slice(block: &[u8], out: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    // Two halves of 128 elements each.
    for half in 0..2 {
        let ql = &ql[half * 64..half * 64 + 64];
        let qh = &qh[half * 32..half * 32 + 32];
        let sc = &scales[half * 8..half * 8 + 8];
        let out_half = &mut out[half * 128..half * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8 as i32 - 32;
            let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
            out_half[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            out_half[32 + l] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            out_half[64 + l] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            out_half[96 + l] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaka_dtype::f32_to_f16;

    #[test]
    fn f16_round_trips_the_values_quantisation_uses() {
        for x in [1.0f32, -1.0, 0.5, 2.0, 65504.0, 0.099975586] {
            let back = f16_to_f32(f32_to_f16(x));
            assert!((back - x).abs() <= x.abs() * 1e-3, "{x} -> {back}");
        }
        // Special values.
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7c01).is_nan());
        // A subnormal half: 1 ulp = 2^-24.
        assert!((f16_to_f32(0x0001) - 2f32.powi(-24)).abs() < 1e-12);
    }

    #[test]
    fn q8_0_is_scale_times_value() {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        block[2] = 10i8 as u8;
        block[3] = -20i8 as u8;
        block[33] = 127i8 as u8;

        let out = dequant(GgmlType::Q8_0, &block, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], 5.0);
        assert_eq!(out[1], -10.0);
        assert_eq!(out[31], 63.5);
        assert_eq!(out[2], 0.0);
    }

    /// Pin the Q2_K layout: each 16-element sub-block must use its own scale
    /// byte, and the quants must come from the right shift of the right byte.
    #[test]
    fn q2_k_applies_the_right_scale_to_the_right_sub_block() {
        let mut block = vec![0u8; 84];
        // d = 1.0, dmin = 0 so values are scale * quant.
        block[80..82].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        // scales: sub-block j gets scale j (4-bit, so 0..=15; min stays 0).
        for (j, b) in block[..16].iter_mut().enumerate() {
            *b = j as u8;
        }
        // qs all 0b11_10_01_00: shift 0 -> 0, shift 2 -> 1, shift 4 -> 2, shift 6 -> 3.
        for b in &mut block[16..80] {
            *b = 0b1110_0100;
        }

        let out = dequant(GgmlType::Q2K, &block, 256).unwrap();
        // Sub-block 0 (first 16 elems): shift 0 -> quant 0 -> 0.
        assert!(out[..16].iter().all(|&v| v == 0.0));
        // Sub-block 2 (elems 32..48): shift 2 -> quant 1, scale 2 -> 2.0.
        assert!(out[32..48].iter().all(|&v| v == 2.0), "{:?}", &out[32..48]);
        // Sub-block 7 (elems 112..128): shift 6 -> quant 3, scale 7 -> 21.0.
        assert!(out[112..128].iter().all(|&v| v == 21.0));
        // Second half repeats the shift cycle with scales 8..15:
        // sub-block 8 (elems 128..144): shift 0 -> quant 0 -> 0.
        assert!(out[128..144].iter().all(|&v| v == 0.0));
        // Sub-block 15 (elems 240..256): shift 6 -> quant 3, scale 15 -> 45.0.
        assert!(out[240..256].iter().all(|&v| v == 45.0));
    }

    #[test]
    fn q2_k_min_is_subtracted() {
        let mut block = vec![0u8; 84];
        block[80..82].copy_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d
        block[82..84].copy_from_slice(&f32_to_f16(2.0).to_le_bytes()); // dmin
        block[0] = 0x51; // scale 1, min 5 -> value = q - 10
                         // qs zero: quant 0 everywhere -> first sub-block = -10.
        let out = dequant(GgmlType::Q2K, &block, 256).unwrap();
        assert!(out[..16].iter().all(|&v| v == -10.0));
    }

    /// Pin the Q6_K bit interleave: one chosen element per quarter, everything
    /// else zero, so a moved shift shows up as a wrong index or value.
    #[test]
    fn q6_k_reassembles_six_bit_values_from_the_right_bits() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d = 1
        for s in &mut block[192..208] {
            *s = 1; // all scales 1
        }
        // Element 0 (quarter 0, l=0): ql[0] low nibble = 5, qh[0] bits 0-1 = 1
        // -> q = 5 | (1<<4) = 21; value = 21 - 32 = -11.
        block[0] = 0x05;
        block[128] = 0b0000_0001;
        // Element 32+3 (quarter 1, l=3): ql[3+32] low nibble = 2, qh[3] bits 2-3 = 2
        // -> q = 2 | (2<<4) = 34; value = 2.
        block[35] = 0x02;
        block[131] = 0b0000_1000;
        // Element 64+7 (quarter 2, l=7): ql[7] high nibble = 9, qh[7] bits 4-5 = 1
        // -> q = 9 | 16 = 25; value = -7.
        block[7] |= 0x90;
        block[135] |= 0b0001_0000;
        // Element 96+9 (quarter 3, l=9): ql[9+32] high nibble = 15, qh[9] bits 6-7 = 3
        // -> q = 15 | 48 = 63; value = 31.
        block[41] |= 0xf0;
        block[137] |= 0b1100_0000;

        let out = dequant(GgmlType::Q6K, &block, 256).unwrap();
        assert_eq!(out[0], -11.0);
        assert_eq!(out[32 + 3], 2.0);
        assert_eq!(out[64 + 7], -7.0);
        assert_eq!(out[96 + 9], 31.0);
        // An untouched element decodes q=0 -> -32, scale 1 -> -32.
        assert_eq!(out[1], -32.0);
        // The second half uses scales[8..]: with identical zero bits it also
        // lands on -32 everywhere.
        assert!(out[128..].iter().all(|&v| v == -32.0));
    }

    #[test]
    fn q6_k_negative_scales_flip_the_sign() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        block[192] = (-2i8) as u8; // scale of sub-block 0
        for s in &mut block[193..208] {
            *s = 0;
        }
        // q = 0 everywhere -> element 0 value = 0.5 * -2 * (0 - 32) = 32.
        let out = dequant(GgmlType::Q6K, &block, 256).unwrap();
        assert_eq!(out[0], 32.0);
        // Sub-block 1 has scale 0 -> exactly 0.
        assert_eq!(out[16], 0.0);
    }

    /// Pin the shared Q4_K/Q5_K scale packing: the split 6-bit fields of
    /// sub-blocks 4..8 must reassemble from the right nibbles and top bits.
    #[test]
    fn k4_scale_packing_reassembles_split_fields() {
        let mut packed = [0u8; 12];
        // Sub-block 0: plain fields. scale 33, min 21.
        packed[0] = 33;
        packed[4] = 21;
        // Sub-block 5: scale 0b10_0110 (38), min 0b01_1001 (25).
        // low 4 of scale -> packed[9] low nibble; high 2 -> packed[1] top bits.
        // low 4 of min   -> packed[9] high nibble; high 2 -> packed[5] top bits.
        packed[9] = (0b1001 << 4) | 0b0110;
        packed[1] |= 0b10 << 6;
        packed[5] |= 0b01 << 6;

        assert_eq!(scale_min_k4(0, &packed), (33.0, 21.0));
        assert_eq!(scale_min_k4(5, &packed), (38.0, 25.0));
    }

    /// Q4_K: each 32-byte row of quants serves two sub-blocks, low nibbles
    /// then high nibbles, each with its own scale and min.
    #[test]
    fn q4_k_low_and_high_nibbles_go_to_adjacent_sub_blocks() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&f32_to_f16(1.0).to_le_bytes()); // dmin
                                                                     // scales: sub-block 0 -> scale 2, min 1; sub-block 1 -> scale 3, min 2.
        block[4] = 2;
        block[5] = 3;
        block[8] = 1;
        block[9] = 2;
        // First qs byte: low nibble 5 (sub-block 0), high nibble 7 (sub-block 1).
        block[16] = 0x75;

        let out = dequant(GgmlType::Q4K, &block, 256).unwrap();
        assert_eq!(out[0], 2.0 * 5.0 - 1.0); // 9
        assert_eq!(out[1], -1.0); // quant 0, min 1
        assert_eq!(out[32], 3.0 * 7.0 - 2.0); // 19: same byte, high nibble
        assert_eq!(out[33], -2.0);
        // Sub-blocks 2.. read later qs rows, all zero here: -min = 0.
        assert!(out[64..].iter().all(|&v| v == 0.0));
    }

    /// Q5_K: the extra bit adds 16, and sub-block j reads bit j of the shared
    /// qh byte.
    #[test]
    fn q5_k_high_bit_adds_sixteen_for_the_owning_sub_block() {
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
        // All eight scales 1 (sub-blocks 0..4 in bytes 4..8; 4..8 split).
        for j in 0..4 {
            block[4 + j] = 1;
        }
        for j in 0..4 {
            block[12 + j] = 0x01; // low nibble: scale of sub-block 4+j
        }
        // qh[0] bit 0 set: element 0 of sub-block 0 gains +16.
        // qh[0] bit 3 set: element 0 of sub-block 3 gains +16.
        block[16] = 0b0000_1001;

        let out = dequant(GgmlType::Q5K, &block, 256).unwrap();
        assert_eq!(out[0], 16.0, "sub-block 0, element 0");
        assert_eq!(out[1], 0.0);
        assert_eq!(out[32], 0.0, "bit 1 not set: sub-block 1 unaffected");
        // Sub-block 3 = pair 1 high nibbles, elements 96..128.
        assert_eq!(out[96], 16.0, "sub-block 3, element 0");
        assert_eq!(out[97], 0.0);
    }

    /// Q3_K: the separate mask bit is worth +4, and the packed 6-bit scales
    /// carry a -32 bias.
    #[test]
    fn q3_k_combines_mask_bits_and_packed_scales() {
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        // All 16 scales = 33 (-> dl = 1.0): low nibble 1 in bytes 96..104,
        // high bits 0b10 in every 2-bit field of bytes 104..108.
        for b in &mut block[96..104] {
            *b = 0x11;
        }
        for b in &mut block[104..108] {
            *b = 0b10_10_10_10;
        }
        // hmask all set: no -4 anywhere; quants all zero -> everything 0.
        for b in &mut block[0..32] {
            *b = 0xff;
        }
        let out = dequant(GgmlType::Q3K, &block, 256).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));

        // Drop the mask bit of element 0 only (bit 0 of hmask[0]): -4.
        let mut block2 = block.clone();
        block2[0] = 0xfe;
        let out2 = dequant(GgmlType::Q3K, &block2, 256).unwrap();
        assert_eq!(out2[0], -4.0);
        assert!(out2[1..].iter().all(|&v| v == 0.0));

        // Element 128 sits in the second half: its mask byte is hmask[0]
        // again, but the walking bit has reached bit 4.
        let mut block3 = block.clone();
        block3[0] = !(1 << 4);
        let out3 = dequant(GgmlType::Q3K, &block3, 256).unwrap();
        assert_eq!(out3[128], -4.0, "{:?}", &out3[124..132]);
        assert!(out3[..128].iter().all(|&v| v == 0.0));
    }

    /// A quant of 2 with scale 34 pins the sign handling: (34-32)=2 scale,
    /// value = 2 * (2 - 4) = -4 when the mask bit is clear.
    #[test]
    fn q3_k_scale_bias_is_minus_32() {
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        // Scale of sub-block 0 = 34: low nibble 2, high bits 0b10.
        block[96] = 0x02;
        block[104] = 0b10;
        // Element 0: quant bits 0b10 at shift 0.
        block[32] = 0b10;
        let out = dequant(GgmlType::Q3K, &block, 256).unwrap();
        assert_eq!(out[0], 2.0 * (2.0 - 4.0));
    }

    #[test]
    fn a_wrong_byte_count_is_an_error_not_a_panic() {
        assert!(dequant(GgmlType::Q8_0, &[0u8; 33], 32).is_err());
        assert!(dequant(GgmlType::Q2K, &[0u8; 84], 255).is_err());
        assert!(dequant(GgmlType::Other(99), &[0u8; 4], 1).is_err());
    }
}
