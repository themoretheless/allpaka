//! Q8 activation quantisation and integer `i8×i8` dot product kernels.

use allpaka_gguf::dequant;

/// Activations quantised to signed 8-bit in blocks of 32:
/// `x[b*32+l] ≈ d[b] * q[b*32+l]`.
///
/// This is llama.cpp's Q8 trick: the weight side is already integers, so
/// quantising the activation side once per matvec turns every inner product
/// into i8×i8 integer math. `sum[b]` keeps the exact f32 block sum so the
/// Q4_K min term loses nothing to the activation quantisation.
pub(super) struct Q8Acts {
    pub(super) d: Vec<f32>,
    pub(super) sum: Vec<f32>,
    pub(super) q: Vec<i8>,
}

#[derive(Clone, Copy)]
pub(super) struct Q8ActsSlice<'a> {
    pub(super) d: &'a [f32],
    pub(super) sum: &'a [f32],
    pub(super) q: &'a [i8],
}

impl Q8Acts {
    pub(super) fn from_f32(x: &[f32]) -> Self {
        debug_assert_eq!(x.len() % 32, 0);
        let blocks = x.len() / 32;
        let mut d = vec![0f32; blocks];
        let mut sum = vec![0f32; blocks];
        let mut q = vec![0i8; x.len()];
        for b in 0..blocks {
            let xs = &x[b * 32..b * 32 + 32];
            #[cfg(target_arch = "aarch64")]
            let (amax, block_sum) = unsafe { super::dot_neon::q8_block_amax_sum(xs) };
            #[cfg(not(target_arch = "aarch64"))]
            let (amax, block_sum) = (
                xs.iter().fold(0f32, |m, v| m.max(v.abs())),
                xs.iter().sum::<f32>(),
            );
            let db = if amax.is_finite() { amax / 127.0 } else { 0.0 };
            let id = if db > 0.0 && db.is_finite() {
                1.0 / db
            } else {
                0.0
            };
            #[cfg(target_arch = "aarch64")]
            unsafe {
                super::dot_neon::quantize_q8_block(xs, id, q.as_mut_ptr().add(b * 32));
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (slot, v) in q[b * 32..b * 32 + 32].iter_mut().zip(xs) {
                *slot = (v * id).round() as i8;
            }
            d[b] = db;
            sum[b] = block_sum;
        }
        Self { d, sum, q }
    }

    #[inline]
    pub(super) fn row(&self, row: usize, n_in: usize) -> Q8ActsSlice<'_> {
        let blocks = n_in / 32;
        let first = row * blocks;
        Q8ActsSlice {
            d: &self.d[first..first + blocks],
            sum: &self.sum[first..first + blocks],
            q: &self.q[first * 32..(first + blocks) * 32],
        }
    }
}

impl Q8ActsSlice<'_> {
    #[inline]
    pub(super) fn block(&self, b: usize) -> &[i8] {
        &self.q[b * 32..b * 32 + 32]
    }
}

/// Q8_0 × Q8 activations: `Σ_blocks d_w * d_x * Σ q_w · q_x`.
#[inline(always)]
pub(super) fn dot_q8_0_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(34).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // SAFETY: i8 and u8 share size and alignment; construct a 32-element i8 slice from pointer.
        let qw: &[i8] =
            unsafe { std::slice::from_raw_parts(block[2..34].as_ptr() as *const i8, 32) };
        total += d * acts.d[bi] * dot_i8(qw, acts.block(bi)) as f32;
    }
    total
}

/// Q5_0 × Q8 activations: 32 5-bit signed values (`q - 16`) per 22-byte block.
#[inline(always)]
pub(super) fn dot_q5_0_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(22).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
        let qs = &block[6..22];
        let mut qw = [0i8; 32];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            super::dot_neon::unpack_q5_0_i8(qs, qh, &mut qw);
        }
        #[cfg(not(target_arch = "aarch64"))]
        for j in 0..16 {
            let x0 = ((qs[j] & 0x0f) | ((((qh >> j) & 1) as u8) << 4)) as i8 - 16;
            let x1 = ((qs[j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)) as i8 - 16;
            qw[j] = x0;
            qw[j + 16] = x1;
        }
        total += d * acts.d[bi] * dot_i8(&qw, acts.block(bi)) as f32;
    }
    total
}

/// Q2_K × Q8 activations: 256 2-bit weights per 84-byte super-block.
#[inline(always)]
pub(super) fn dot_q2_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(84).enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let mut is = 0;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                total += super::dot_neon::dot_q2_k_half_i8(
                    q, scales, d, dmin, bi, half, &mut is, acts,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (s_idx, shift) in [0u8, 2, 4, 6].into_iter().enumerate() {
                let blk = bi * 8 + half * 4 + s_idx;
                let xq = acts.block(blk);
                let dx = acts.d[blk];
                let mut qw = [0i8; 32];
                for l in 0..32 {
                    qw[l] = ((q[l] >> shift) & 3) as i8;
                }
                for group in 0..2 {
                    let sc = scales[is];
                    let dl = d * (sc & 0xf) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    let dotq = dot_i8(
                        &qw[group * 16..group * 16 + 16],
                        &xq[group * 16..group * 16 + 16],
                    ) as f32
                        * dx;
                    let sum_x = xq[group * 16..group * 16 + 16]
                        .iter()
                        .map(|&v| v as i32)
                        .sum::<i32>() as f32
                        * dx;
                    total += dl * dotq - ml * sum_x;
                    is += 1;
                }
            }
        }
    }
    total
}

/// Q3_K × Q8 activations: 256 signed 3-bit weights per 110-byte super-block.
#[inline(always)]
pub(super) fn dot_q3_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(110).enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let packed = &block[96..108];
        let d_all = dequant::f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        let raw_scales = allpaka_dtype::unpack_q3_k_scales_raw(packed);

        let mut is = 0;
        let mut m: u8 = 1;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            for (s_idx, shift) in [0u8, 2, 4, 6].into_iter().enumerate() {
                let blk = bi * 8 + half * 4 + s_idx;
                let xq = acts.block(blk);
                let dx = acts.d[blk];
                let mut qw = [0i8; 32];
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    super::dot_neon::unpack_q3_k_i8(q, hmask, shift, m, &mut qw);
                }
                #[cfg(not(target_arch = "aarch64"))]
                for l in 0..32 {
                    qw[l] = (((q[l] >> shift) & 3) as i8) - if hmask[l] & m != 0 { 0 } else { 4 };
                }
                let dot0 = dot_i8(&qw[0..16], &xq[0..16]) as f32;
                let dot1 = dot_i8(&qw[16..32], &xq[16..32]) as f32;
                let sc0 = (raw_scales[is] as i32 - 32) as f32;
                let sc1 = (raw_scales[is + 1] as i32 - 32) as f32;
                total += d_all * dx * (sc0 * dot0 + sc1 * dot1);
                is += 2;
                m <<= 1;
            }
        }
    }
    total
}

/// Q5_K × Q8 activations: 256 5-bit weights per 176-byte super-block.
#[inline(always)]
pub(super) fn dot_q5_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(176).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let (sc1, mn1) = dequant::scale_min_k4(pair * 2, packed);
            let (sc2, mn2) = dequant::scale_min_k4(pair * 2 + 1, packed);
            let bit_shift1 = (pair * 2) as u8;
            let bit_shift2 = (pair * 2 + 1) as u8;
            let mut lo = [0i8; 32];
            let mut hi = [0i8; 32];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                super::dot_neon::unpack_q5_k_i8(q, qh, bit_shift1, bit_shift2, &mut lo, &mut hi);
            }
            #[cfg(not(target_arch = "aarch64"))]
            for l in 0..32 {
                let h1 = ((qh[l] >> bit_shift1) & 1) << 4;
                let h2 = ((qh[l] >> bit_shift2) & 1) << 4;
                lo[l] = ((q[l] & 0x0f) | h1) as i8;
                hi[l] = ((q[l] >> 4) | h2) as i8;
            }
            let b_lo = bi * 8 + pair * 2;
            let b_hi = b_lo + 1;
            let dot_lo = dot_i8(&lo, acts.block(b_lo)) as f32 * acts.d[b_lo];
            let dot_hi = dot_i8(&hi, acts.block(b_hi)) as f32 * acts.d[b_hi];
            total += d * sc1 * dot_lo - dmin * mn1 * acts.sum[b_lo];
            total += d * sc2 * dot_hi - dmin * mn2 * acts.sum[b_hi];
        }
    }
    total
}

/// Q4_K × Q8 activations, same sub-block walk as `dot_q4_k` with the q·x
/// reduction done in integers and the min term taken off the exact block sum.
#[inline(always)]
pub(super) fn dot_q4_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(144).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let qs = &block[16..144];

        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let (sc1, mn1) = dequant::scale_min_k4(pair * 2, packed);
            let (sc2, mn2) = dequant::scale_min_k4(pair * 2 + 1, packed);

            let mut lo = [0i8; 32];
            let mut hi = [0i8; 32];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                super::dot_neon::unpack_q4_k_i8(q, &mut lo, &mut hi);
            }
            #[cfg(not(target_arch = "aarch64"))]
            for l in 0..32 {
                lo[l] = (q[l] & 0xf) as i8;
                hi[l] = (q[l] >> 4) as i8;
            }
            let b_lo = bi * 8 + pair * 2;
            let b_hi = b_lo + 1;
            let dot_lo = dot_i8(&lo, acts.block(b_lo)) as f32 * acts.d[b_lo];
            let dot_hi = dot_i8(&hi, acts.block(b_hi)) as f32 * acts.d[b_hi];
            total += d * sc1 * dot_lo - dmin * mn1 * acts.sum[b_lo];
            total += d * sc2 * dot_hi - dmin * mn2 * acts.sum[b_hi];
        }
    }
    total
}

/// Q6_K × Q8 activations. Scales cover 16 elements, so each quarter's
/// 32-value dot splits into two 16-lane integer dots against the halves of
/// one activation block.
#[inline(always)]
pub(super) fn dot_q6_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(210).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        for half in 0..2 {
            let ql = &ql[half * 64..half * 64 + 64];
            let qh = &qh[half * 32..half * 32 + 32];
            let sc = &scales[half * 8..half * 8 + 8];

            let mut quarters = [[0i8; 32]; 4];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                super::dot_neon::unpack_q6_k_i8(ql, qh, &mut quarters);
            }
            #[cfg(not(target_arch = "aarch64"))]
            for l in 0..32 {
                quarters[0][l] = (((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8) - 32;
                quarters[1][l] = (((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8) - 32;
                quarters[2][l] = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8) - 32;
                quarters[3][l] = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8) - 32;
            }
            for (qi, qq) in quarters.iter().enumerate() {
                let blk = bi * 8 + half * 4 + qi;
                let xq = acts.block(blk);
                let dot0 = dot_i8(&qq[0..16], &xq[0..16]) as f32;
                let dot1 = dot_i8(&qq[16..32], &xq[16..32]) as f32;
                let sc0 = (sc[2 * qi] as i8) as f32;
                let sc1 = (sc[2 * qi + 1] as i8) as f32;
                total += d * acts.d[blk] * (sc0 * dot0 + sc1 * dot1);
            }
        }
    }
    total
}

/// i8×i8 dot product over equal-length slices, accumulated in i32.
///
/// On aarch64 with `dotprod` (ARMv8.2+, all Apple Silicon M1–M4), uses
/// `vdotq_s32` (`SDOT`) to compute 16 `i8×i8` products per instruction.
#[inline(always)]
pub(super) fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        super::dot_neon::dot_i8_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    a.iter().zip(b).map(|(&p, &q)| p as i32 * q as i32).sum()
}
