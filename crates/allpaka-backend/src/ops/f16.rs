//! Half-precision helpers for the KV cache.
//!
//! The cache is the one tensor the engine writes every step and reads in full
//! every step, so it is stored as f16: half the bytes for a format whose
//! precision is far beyond what attention scores need. The GPU reads `half`
//! natively; these are for the CPU path, which is the fallback for decode and
//! still carries the whole of prefill.
//!
//! Conversion is one instruction per four values on aarch64. A scalar loop
//! would give back more than the halved traffic wins - the same trap the
//! scalar `dot` was.

#![allow(clippy::incompatible_msrv)]

/// Convert `src` to half, in place into `dst`. Lengths must match.
#[inline]
pub fn from_f32(src: &[f32], dst: &mut [u16]) {
    allpaka_dtype::f32_to_f16_slice(src, dst);
}

/// `sum(a[i] * b[i])` with `b` in half.
pub fn dot(a: &[f32], b: &[u16]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        // Four independent accumulators for the same reason ops::dot has
        // eight: one would serialise the whole loop.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            let u01 = vld1q_u16(b.as_ptr().add(i));
            let u23 = vld1q_u16(b.as_ptr().add(i + 8));
            let h0: float16x4_t = std::mem::transmute(vget_low_u16(u01));
            let h1: float16x4_t = std::mem::transmute(vget_high_u16(u01));
            let h2: float16x4_t = std::mem::transmute(vget_low_u16(u23));
            let h3: float16x4_t = std::mem::transmute(vget_high_u16(u23));
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vcvt_f32_f16(h0));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a.as_ptr().add(i + 4)), vcvt_f32_f16(h1));
            acc2 = vfmaq_f32(acc2, vld1q_f32(a.as_ptr().add(i + 8)), vcvt_f32_f16(h2));
            acc3 = vfmaq_f32(acc3, vld1q_f32(a.as_ptr().add(i + 12)), vcvt_f32_f16(h3));
            i += 16;
        }
        while i + 4 <= n {
            let h: float16x4_t = std::mem::transmute(vld1_u16(b.as_ptr().add(i)));
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vcvt_f32_f16(h));
            i += 4;
        }
        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let mut total = vaddvq_f32(vaddq_f32(sum01, sum23));
        for k in i..n {
            total += a[k] * to_f32(b[k]);
        }
        total
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut total = 0.0;
        for k in 0..n {
            total += a[k] * to_f32(b[k]);
        }
        total
    }
}

/// `acc += p * v`, with `v` in half.
pub fn axpy(acc: &mut [f32], p: f32, v: &[u16]) {
    let n = acc.len().min(v.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let pv = vdupq_n_f32(p);
        let mut i = 0;
        while i + 16 <= n {
            let u01 = vld1q_u16(v.as_ptr().add(i));
            let u23 = vld1q_u16(v.as_ptr().add(i + 8));
            let h0: float16x4_t = std::mem::transmute(vget_low_u16(u01));
            let h1: float16x4_t = std::mem::transmute(vget_high_u16(u01));
            let h2: float16x4_t = std::mem::transmute(vget_low_u16(u23));
            let h3: float16x4_t = std::mem::transmute(vget_high_u16(u23));
            let a0 = vld1q_f32(acc.as_ptr().add(i));
            let a1 = vld1q_f32(acc.as_ptr().add(i + 4));
            let a2 = vld1q_f32(acc.as_ptr().add(i + 8));
            let a3 = vld1q_f32(acc.as_ptr().add(i + 12));
            vst1q_f32(acc.as_mut_ptr().add(i), vfmaq_f32(a0, pv, vcvt_f32_f16(h0)));
            vst1q_f32(
                acc.as_mut_ptr().add(i + 4),
                vfmaq_f32(a1, pv, vcvt_f32_f16(h1)),
            );
            vst1q_f32(
                acc.as_mut_ptr().add(i + 8),
                vfmaq_f32(a2, pv, vcvt_f32_f16(h2)),
            );
            vst1q_f32(
                acc.as_mut_ptr().add(i + 12),
                vfmaq_f32(a3, pv, vcvt_f32_f16(h3)),
            );
            i += 16;
        }
        while i + 4 <= n {
            let h: float16x4_t = std::mem::transmute(vld1_u16(v.as_ptr().add(i)));
            let a = vld1q_f32(acc.as_ptr().add(i));
            vst1q_f32(acc.as_mut_ptr().add(i), vfmaq_f32(a, pv, vcvt_f32_f16(h)));
            i += 4;
        }
        for k in i..n {
            acc[k] += p * to_f32(v[k]);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for k in 0..n {
        acc[k] += p * to_f32(v[k]);
    }
}

/// One value back to f32; the arbiter for the vector paths above.
#[inline]
pub fn to_f32(h: u16) -> f32 {
    allpaka_dtype::f16_to_f32(h)
}

/// One value to half, round-to-nearest-even, with overflow to infinity
/// and subnormals preserved.
#[inline]
pub fn scalar_from_f32(x: f32) -> u16 {
    allpaka_dtype::f32_to_f16(x)
}
