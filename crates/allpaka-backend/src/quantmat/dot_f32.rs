//! Fused quantised-row × f32-activation dot products.

use allpaka_gguf::dequant;

#[inline(always)]
pub(super) fn dot_f16(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_endian = "little")]
    {
        let (head, mid, tail) = unsafe { row.align_to::<u16>() };
        if head.is_empty() && tail.is_empty() {
            return crate::ops::f16::dot(x, mid);
        }
    }
    row.chunks_exact(2)
        .zip(x)
        .map(|(c, &xv)| dequant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])) * xv)
        .sum()
}

#[inline(always)]
pub(super) fn dot_bf16(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        super::dot_neon::dot_bf16_neon(row, x)
    }
    #[cfg(not(target_arch = "aarch64"))]
    row.chunks_exact(2)
        .zip(x)
        .map(|(c, &xv)| dequant::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) * xv)
        .sum()
}

#[inline(always)]
pub(super) fn dot_q5_0(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(22).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
        let qs = &block[6..22];
        let xs = &x[bi * 32..bi * 32 + 32];
        let mut s = 0f32;
        for j in 0..16 {
            let x0 = ((qs[j] & 0x0f) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
            s += x0 as f32 * xs[j] + x1 as f32 * xs[j + 16];
        }
        total += d * s;
    }
    total
}

#[inline(always)]
pub(super) fn dot_q3_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(110).enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let packed = &block[96..108];
        let d_all = dequant::f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let xb = &x[bi * 256..bi * 256 + 256];

        let raw_scales = allpaka_dtype::unpack_q3_k_scales_raw(packed);

        let mut is = 0;
        let mut m: u8 = 1;
        let mut pos = 0;
        let mut blk_sum = 0f32;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            for shift in [0u8, 2, 4, 6] {
                for group in 0..2 {
                    let mut sub = 0f32;
                    for l in 0..16 {
                        let idx = group * 16 + l;
                        let quant = ((q[idx] >> shift) & 3) as i32
                            - if hmask[idx] & m != 0 { 0 } else { 4 };
                        sub += quant as f32 * xb[pos + l];
                    }
                    blk_sum += (raw_scales[is] as i32 - 32) as f32 * sub;
                    pos += 16;
                    is += 1;
                }
                m <<= 1;
            }
        }
        total += d_all * blk_sum;
    }
    total
}

#[inline(always)]
pub(super) fn dot_q5_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(176).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let xb = &x[bi * 256..bi * 256 + 256];

        let mut acc_d = 0f32;
        let mut acc_m = 0f32;
        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let x_lo = &xb[pair * 64..pair * 64 + 32];
            let x_hi = &xb[pair * 64 + 32..pair * 64 + 64];
            let (sc1, mn1) = dequant::scale_min_k4(pair * 2, packed);
            let (sc2, mn2) = dequant::scale_min_k4(pair * 2 + 1, packed);
            let bit1 = 1u8 << (pair * 2);
            let bit2 = 1u8 << (pair * 2 + 1);
            let mut dot_lo = 0f32;
            let mut sum_lo = 0f32;
            let mut dot_hi = 0f32;
            let mut sum_hi = 0f32;
            for l in 0..32 {
                let hi1 = if qh[l] & bit1 != 0 { 16 } else { 0 };
                let hi2 = if qh[l] & bit2 != 0 { 16 } else { 0 };
                dot_lo += ((q[l] & 0xf) + hi1) as f32 * x_lo[l];
                sum_lo += x_lo[l];
                dot_hi += ((q[l] >> 4) + hi2) as f32 * x_hi[l];
                sum_hi += x_hi[l];
            }
            acc_d += sc1 * dot_lo + sc2 * dot_hi;
            acc_m += mn1 * sum_lo + mn2 * sum_hi;
        }
        total += d * acc_d - dmin * acc_m;
    }
    total
}

/// Fused Q8_0 dot product: `Σ_blocks d * Σ q_i * x_i`, straight off the
/// quantised bytes.
#[inline(always)]
pub(super) fn dot_q8_0(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(34).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let xs = &x[bi * 32..bi * 32 + 32];
        #[cfg(target_arch = "aarch64")]
        let s = unsafe { super::dot_neon::dot_q8_0_block_neon(block, xs) };
        #[cfg(not(target_arch = "aarch64"))]
        let s = {
            let mut s = 0f32;
            for (q, xv) in block[2..34].iter().zip(xs) {
                s += (*q as i8) as f32 * xv;
            }
            s
        };
        total += d * s;
    }
    total
}

/// Fused Q4_K dot product, following the same 8-sub-block layout as the
/// dequantiser: `Σ d*sc*Σ(q·x) - dmin*mn*Σx` per sub-block, so the min term
/// needs only the activation sum, not a per-element subtraction.
#[inline(always)]
pub(super) fn dot_q4_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(144).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let qs = &block[16..144];
        let xb = &x[bi * 256..bi * 256 + 256];

        let mut acc_d = 0f32;
        let mut acc_m = 0f32;
        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let x_lo = &xb[pair * 64..pair * 64 + 32];
            let x_hi = &xb[pair * 64 + 32..pair * 64 + 64];
            let (sc1, mn1) = dequant::scale_min_k4(pair * 2, packed);
            let (sc2, mn2) = dequant::scale_min_k4(pair * 2 + 1, packed);

            // Unpack first, reduce second: single-purpose fixed-bound loops
            // are what the auto-vectoriser actually turns into NEON/AVX.
            let mut lo = [0f32; 32];
            let mut hi = [0f32; 32];
            for l in 0..32 {
                lo[l] = (q[l] & 0xf) as f32;
                hi[l] = (q[l] >> 4) as f32;
            }
            let mut dot_lo = 0f32;
            let mut sum_lo = 0f32;
            for l in 0..32 {
                dot_lo += lo[l] * x_lo[l];
                sum_lo += x_lo[l];
            }
            let mut dot_hi = 0f32;
            let mut sum_hi = 0f32;
            for l in 0..32 {
                dot_hi += hi[l] * x_hi[l];
                sum_hi += x_hi[l];
            }
            acc_d += sc1 * dot_lo + sc2 * dot_hi;
            acc_m += mn1 * sum_lo + mn2 * sum_hi;
        }
        total += d * acc_d - dmin * acc_m;
    }
    total
}

/// Fused Q2_K dot product: `Σ dl*Σ(q·x) - ml*Σx` per 16-element sub-block.
#[inline(always)]
pub(super) fn dot_q2_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(84).enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let xb = &x[bi * 256..bi * 256 + 256];

        let mut is = 0;
        let mut acc_d = 0f32;
        let mut acc_m = 0f32;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            for shift in [0u8, 2, 4, 6] {
                for group in 0..2 {
                    let sc = scales[is];
                    let x0 = half * 128 + (shift as usize / 2) * 32 + group * 16;
                    let xs = &xb[x0..x0 + 16];
                    let mut dotq = 0f32;
                    let mut sum = 0f32;
                    for l in 0..16 {
                        dotq += ((q[group * 16 + l] >> shift) & 3) as f32 * xs[l];
                        sum += xs[l];
                    }
                    acc_d += (sc & 0xf) as f32 * dotq;
                    acc_m += (sc >> 4) as f32 * sum;
                    is += 1;
                }
            }
        }
        total += d * acc_d - dmin * acc_m;
    }
    total
}

/// Fused Q6_K dot product, mirroring the dequantiser's half/quarter walk.
#[inline(always)]
pub(super) fn dot_q6_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(210).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let xb = &x[bi * 256..bi * 256 + 256];

        let mut blk_sum = 0f32;
        for half in 0..2 {
            let ql = &ql[half * 64..half * 64 + 64];
            let qh = &qh[half * 32..half * 32 + 32];
            let sc = &scales[half * 8..half * 8 + 8];
            let xh = &xb[half * 128..half * 128 + 128];
            let mut acc = [0f32; 8];
            for (is, range) in [(0usize, 0usize..16), (1usize, 16usize..32)] {
                for l in range {
                    let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8 as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
                    acc[is] += q1 as f32 * xh[l];
                    acc[2 + is] += q2 as f32 * xh[l + 32];
                    acc[4 + is] += q3 as f32 * xh[l + 64];
                    acc[6 + is] += q4 as f32 * xh[l + 96];
                }
            }
            for (g, a) in acc.iter().enumerate() {
                blk_sum += (sc[g] as i8) as f32 * a;
            }
        }
        total += d * blk_sum;
    }
    total
}

/// A borrowed f32 tensor as a slice, when the mapping is aligned for it.
///
/// GGUF aligns tensor data to 32 bytes and the mapping starts on a page, so
/// this succeeds for every real file; the fallback exists because a slice
/// cast that is wrong about alignment is undefined behaviour, not a wrong
/// answer.
#[inline(always)]
pub(super) fn as_f32(bytes: &[u8]) -> Option<&[f32]> {
    let (head, mid, tail) = unsafe { bytes.align_to::<f32>() };
    (head.is_empty() && tail.is_empty()).then_some(mid)
}
