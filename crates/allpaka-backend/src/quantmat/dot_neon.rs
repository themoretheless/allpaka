//! AArch64 NEON kernels for quantised dot products and Q8 activation quantisation.

#![cfg(target_arch = "aarch64")]

use super::q8_acts::Q8ActsSlice;
use allpaka_gguf::dequant;
use std::arch::aarch64::*;

#[inline(always)]
pub(super) unsafe fn dot_bf16_neon(row: &[u8], x: &[f32]) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let n = x.len();
    let rp = row.as_ptr();
    let xp = x.as_ptr();
    while i + 8 <= n {
        let bits = vld1q_u16(rp.add(i * 2) as *const u16);
        let f0 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bits)));
        let f1 = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bits));
        acc0 = vfmaq_f32(acc0, f0, vld1q_f32(xp.add(i)));
        acc1 = vfmaq_f32(acc1, f1, vld1q_f32(xp.add(i + 4)));
        i += 8;
    }
    let mut total = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        let c0 = *rp.add(i * 2);
        let c1 = *rp.add(i * 2 + 1);
        total += dequant::bf16_to_f32(u16::from_le_bytes([c0, c1])) * *xp.add(i);
        i += 1;
    }
    total
}

#[inline(always)]
pub(super) unsafe fn dot_q8_0_block_neon(block: &[u8], xs: &[f32]) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let qp = block.as_ptr().add(2) as *const i8;
    let xp = xs.as_ptr();
    for seg in [0usize, 16] {
        let q8 = vld1q_s8(qp.add(seg));
        let q16_lo = vmovl_s8(vget_low_s8(q8));
        let q16_hi = vmovl_high_s8(q8);
        let qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo)));
        let qf1 = vcvtq_f32_s32(vmovl_high_s16(q16_lo));
        let qf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi)));
        let qf3 = vcvtq_f32_s32(vmovl_high_s16(q16_hi));
        acc0 = vfmaq_f32(acc0, qf0, vld1q_f32(xp.add(seg)));
        acc1 = vfmaq_f32(acc1, qf1, vld1q_f32(xp.add(seg + 4)));
        acc0 = vfmaq_f32(acc0, qf2, vld1q_f32(xp.add(seg + 8)));
        acc1 = vfmaq_f32(acc1, qf3, vld1q_f32(xp.add(seg + 12)));
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[inline(always)]
pub(super) unsafe fn q8_block_amax_sum(xs: &[f32]) -> (f32, f32) {
    let mut vmax0 = vdupq_n_f32(0.0);
    let mut vmax1 = vdupq_n_f32(0.0);
    let mut vsum0 = vdupq_n_f32(0.0);
    let mut vsum1 = vdupq_n_f32(0.0);
    let xp = xs.as_ptr();
    for k in (0..32).step_by(8) {
        let v0 = vld1q_f32(xp.add(k));
        let v1 = vld1q_f32(xp.add(k + 4));
        vmax0 = vmaxq_f32(vmax0, vabsq_f32(v0));
        vmax1 = vmaxq_f32(vmax1, vabsq_f32(v1));
        vsum0 = vaddq_f32(vsum0, v0);
        vsum1 = vaddq_f32(vsum1, v1);
    }
    (
        vmaxvq_f32(vmaxq_f32(vmax0, vmax1)),
        vaddvq_f32(vaddq_f32(vsum0, vsum1)),
    )
}

#[inline(always)]
pub(super) unsafe fn quantize_q8_block(xs: &[f32], id: f32, qp: *mut i8) {
    let vid = vdupq_n_f32(id);
    let xp = xs.as_ptr();
    for k in (0..32).step_by(16) {
        let s0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp.add(k)), vid));
        let s1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp.add(k + 4)), vid));
        let s2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp.add(k + 8)), vid));
        let s3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp.add(k + 12)), vid));
        let h01 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
        let h23 = vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3));
        let b0123 = vcombine_s8(vqmovn_s16(h01), vqmovn_s16(h23));
        vst1q_s8(qp.add(k), b0123);
    }
}

#[inline(always)]
pub(super) unsafe fn unpack_q5_0_i8(qs: &[u8], qh: u32, qw: &mut [i8; 32]) {
    let qv = vld1q_u8(qs.as_ptr());
    let mask_0f = vdupq_n_u8(0x0f);
    let bias_16 = vdupq_n_s8(16);
    let bit_mask: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let vbm = vld1q_u8(bit_mask.as_ptr());
    let qh_b0 = vdupq_n_u8((qh & 0xff) as u8);
    let qh_b1 = vdupq_n_u8(((qh >> 8) & 0xff) as u8);
    let qh_b2 = vdupq_n_u8(((qh >> 16) & 0xff) as u8);
    let qh_b3 = vdupq_n_u8(((qh >> 24) & 0xff) as u8);
    let qh_lo_bytes = vcombine_u8(vget_low_u8(qh_b0), vget_low_u8(qh_b1));
    let qh_hi_bytes = vcombine_u8(vget_low_u8(qh_b2), vget_low_u8(qh_b3));
    let h0 = vandq_u8(vtstq_u8(qh_lo_bytes, vbm), vdupq_n_u8(0x10));
    let h1 = vandq_u8(vtstq_u8(qh_hi_bytes, vbm), vdupq_n_u8(0x10));
    let x0 = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qv, mask_0f), h0)),
        bias_16,
    );
    let x1 = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qv, 4), h1)),
        bias_16,
    );
    vst1q_s8(qw.as_mut_ptr(), x0);
    vst1q_s8(qw.as_mut_ptr().add(16), x1);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn dot_q2_k_half_i8(
    q: &[u8],
    scales: &[u8],
    d: f32,
    dmin: f32,
    bi: usize,
    half: usize,
    is: &mut usize,
    acts: &Q8ActsSlice<'_>,
) -> f32 {
    let mut total = 0.0f32;
    let q0 = vld1q_u8(q.as_ptr());
    let q1 = vld1q_u8(q.as_ptr().add(16));
    let m3 = vdupq_n_u8(3);
    let planes = [
        (
            vreinterpretq_s8_u8(vandq_u8(q0, m3)),
            vreinterpretq_s8_u8(vandq_u8(q1, m3)),
        ),
        (
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 2), m3)),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 2), m3)),
        ),
        (
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 4), m3)),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 4), m3)),
        ),
        (
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 6), m3)),
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 6), m3)),
        ),
    ];
    let mut qw = [0i8; 32];
    for (s_idx, &(p0, p1)) in planes.iter().enumerate() {
        let blk = bi * 8 + half * 4 + s_idx;
        let xq = acts.block(blk);
        let dx = acts.d[blk];
        vst1q_s8(qw.as_mut_ptr(), p0);
        vst1q_s8(qw.as_mut_ptr().add(16), p1);
        let xv0 = vld1q_s8(xq.as_ptr());
        let xv1 = vld1q_s8(xq.as_ptr().add(16));
        let sum0 = vaddlvq_s8(xv0) as f32 * dx;
        let sum1 = vaddlvq_s8(xv1) as f32 * dx;
        let dot0 = dot_i8_neon(&qw[0..16], &xq[0..16]) as f32 * dx;
        let dot1 = dot_i8_neon(&qw[16..32], &xq[16..32]) as f32 * dx;
        let sc0 = scales[*is];
        let sc1 = scales[*is + 1];
        total += (d * (sc0 & 0xf) as f32) * dot0 - (dmin * (sc0 >> 4) as f32) * sum0;
        total += (d * (sc1 & 0xf) as f32) * dot1 - (dmin * (sc1 >> 4) as f32) * sum1;
        *is += 2;
    }
    total
}

#[inline(always)]
pub(super) unsafe fn unpack_q3_k_i8(q: &[u8], hmask: &[u8], shift: u8, m: u8, qw: &mut [i8; 32]) {
    let m3 = vdupq_n_u8(3);
    let vm = vdupq_n_u8(m);
    let four = vdupq_n_u8(4);
    let vsh = vdupq_n_s8(-(shift as i8));
    for seg in [0usize, 16] {
        let qv = vld1q_u8(q.as_ptr().add(seg));
        let hv = vld1q_u8(hmask.as_ptr().add(seg));
        let crumbs = vandq_u8(vshlq_u8(qv, vsh), m3);
        let sub4 = vbicq_u8(four, vtstq_u8(hv, vm));
        let val = vsubq_s8(vreinterpretq_s8_u8(crumbs), vreinterpretq_s8_u8(sub4));
        vst1q_s8(qw.as_mut_ptr().add(seg), val);
    }
}

#[inline(always)]
pub(super) unsafe fn unpack_q5_k_i8(
    q: &[u8],
    qh: &[u8],
    bit_shift1: u8,
    bit_shift2: u8,
    lo: &mut [i8; 32],
    hi: &mut [i8; 32],
) {
    let mask_0f = vdupq_n_u8(0x0f);
    let mask_10 = vdupq_n_u8(0x10);
    let bm1 = vdupq_n_u8(1u8 << bit_shift1);
    let bm2 = vdupq_n_u8(1u8 << bit_shift2);
    for seg in [0usize, 16] {
        let qv = vld1q_u8(q.as_ptr().add(seg));
        let qhv = vld1q_u8(qh.as_ptr().add(seg));
        let h1 = vandq_u8(vtstq_u8(qhv, bm1), mask_10);
        let h2 = vandq_u8(vtstq_u8(qhv, bm2), mask_10);
        let vlo = vorrq_u8(vandq_u8(qv, mask_0f), h1);
        let vhi = vorrq_u8(vshrq_n_u8(qv, 4), h2);
        vst1q_s8(lo.as_mut_ptr().add(seg), vreinterpretq_s8_u8(vlo));
        vst1q_s8(hi.as_mut_ptr().add(seg), vreinterpretq_s8_u8(vhi));
    }
}

#[inline(always)]
pub(super) unsafe fn unpack_q4_k_i8(q: &[u8], lo: &mut [i8; 32], hi: &mut [i8; 32]) {
    let mask = vdupq_n_u8(0x0f);
    let q0 = vld1q_u8(q.as_ptr());
    let q1 = vld1q_u8(q.as_ptr().add(16));
    vst1q_s8(lo.as_mut_ptr(), vreinterpretq_s8_u8(vandq_u8(q0, mask)));
    vst1q_s8(
        lo.as_mut_ptr().add(16),
        vreinterpretq_s8_u8(vandq_u8(q1, mask)),
    );
    vst1q_s8(hi.as_mut_ptr(), vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)));
    vst1q_s8(
        hi.as_mut_ptr().add(16),
        vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)),
    );
}

#[inline(always)]
pub(super) unsafe fn unpack_q6_k_i8(ql: &[u8], qh: &[u8], quarters: &mut [[i8; 32]; 4]) {
    let mask_0f = vdupq_n_u8(0x0f);
    let mask_30 = vdupq_n_u8(0x30);
    let bias_32 = vdupq_n_s8(32);
    for seg in [0usize, 16] {
        let ql_lo = vld1q_u8(ql.as_ptr().add(seg));
        let ql_hi = vld1q_u8(ql.as_ptr().add(32 + seg));
        let qh_v = vld1q_u8(qh.as_ptr().add(seg));

        let q0_u = vorrq_u8(
            vandq_u8(ql_lo, mask_0f),
            vandq_u8(vshlq_n_u8(qh_v, 4), mask_30),
        );
        let q1_u = vorrq_u8(
            vandq_u8(ql_hi, mask_0f),
            vandq_u8(vshlq_n_u8(qh_v, 2), mask_30),
        );
        let q2_u = vorrq_u8(vshrq_n_u8(ql_lo, 4), vandq_u8(qh_v, mask_30));
        let q3_u = vorrq_u8(vshrq_n_u8(ql_hi, 4), vandq_u8(vshrq_n_u8(qh_v, 2), mask_30));

        vst1q_s8(
            quarters[0].as_mut_ptr().add(seg),
            vsubq_s8(vreinterpretq_s8_u8(q0_u), bias_32),
        );
        vst1q_s8(
            quarters[1].as_mut_ptr().add(seg),
            vsubq_s8(vreinterpretq_s8_u8(q1_u), bias_32),
        );
        vst1q_s8(
            quarters[2].as_mut_ptr().add(seg),
            vsubq_s8(vreinterpretq_s8_u8(q2_u), bias_32),
        );
        vst1q_s8(
            quarters[3].as_mut_ptr().add(seg),
            vsubq_s8(vreinterpretq_s8_u8(q3_u), bias_32),
        );
    }
}

#[inline(always)]
pub(super) unsafe fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(any(target_os = "macos", target_feature = "dotprod"))]
    {
        dot_i8_sdot(a, b)
    }
    #[cfg(not(any(target_os = "macos", target_feature = "dotprod")))]
    {
        let mut acc = vdupq_n_s32(0);
        let mut i = 0;
        while i + 16 <= a.len() {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
            acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
            i += 16;
        }
        let mut total = vaddvq_s32(acc);
        while i < a.len() {
            total += a[i] as i32 * b[i] as i32;
            i += 1;
        }
        total
    }
}

#[cfg(any(target_os = "macos", target_feature = "dotprod"))]
#[target_feature(enable = "dotprod")]
#[allow(clippy::incompatible_msrv)]
#[inline]
unsafe fn dot_i8_sdot(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;
    while i + 16 <= a.len() {
        let va = vld1q_s8(a.as_ptr().add(i));
        let vb = vld1q_s8(b.as_ptr().add(i));
        acc = vdotq_s32(acc, va, vb);
        i += 16;
    }
    let mut total = vaddvq_s32(acc);
    while i < a.len() {
        total += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    total
}
