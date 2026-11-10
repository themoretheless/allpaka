//! RMSNorm and L2 normalization operations with NEON and AVX2+FMA fast paths.

use super::blas::dot;

#[inline]
pub(crate) fn sum_sq_f64(x: &[f32]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: loads read 8 f32 at `i` only while `i + 8 <= n = x.len()`; the tail
    // below uses safe indexing. `x` is a typed slice, so `xp` is f32-aligned.
    unsafe {
        use std::arch::aarch64::*;
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut acc2 = vdupq_n_f64(0.0);
        let mut acc3 = vdupq_n_f64(0.0);
        let mut i = 0usize;
        let n = x.len();
        let xp = x.as_ptr();
        while i + 8 <= n {
            let v0 = vld1q_f32(xp.add(i));
            let v1 = vld1q_f32(xp.add(i + 4));
            let d0 = vcvt_f64_f32(vget_low_f32(v0));
            let d1 = vcvt_high_f64_f32(v0);
            let d2 = vcvt_f64_f32(vget_low_f32(v1));
            let d3 = vcvt_high_f64_f32(v1);
            acc0 = vfmaq_f64(acc0, d0, d0);
            acc1 = vfmaq_f64(acc1, d1, d1);
            acc2 = vfmaq_f64(acc2, d2, d2);
            acc3 = vfmaq_f64(acc3, d3, d3);
            i += 8;
        }
        let mut total = vaddvq_f64(vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3)));
        while i < n {
            let v = *xp.add(i) as f64;
            total += v * v;
            i += 1;
        }
        total
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn sum_sq_avx2(x: &[f32]) -> f64 {
                use std::arch::x86_64::*;
                let mut acc0 = _mm256_setzero_pd();
                let mut acc1 = _mm256_setzero_pd();
                let mut i = 0usize;
                let n = x.len();
                let xp = x.as_ptr();
                while i + 8 <= n {
                    let v0 = _mm_loadu_ps(xp.add(i));
                    let v1 = _mm_loadu_ps(xp.add(i + 4));
                    let d0 = _mm256_cvtps_pd(v0);
                    let d1 = _mm256_cvtps_pd(v1);
                    acc0 = _mm256_fmadd_pd(d0, d0, acc0);
                    acc1 = _mm256_fmadd_pd(d1, d1, acc1);
                    i += 8;
                }
                let sum = _mm256_add_pd(acc0, acc1);
                let hi = _mm256_extractf128_pd(sum, 1);
                let lo = _mm256_castpd256_pd128(sum);
                let pair = _mm_add_pd(lo, hi);
                let mut buf = [0.0f64; 2];
                _mm_storeu_pd(buf.as_mut_ptr(), pair);
                let mut total = buf[0] + buf[1];
                while i < n {
                    let v = *xp.add(i) as f64;
                    total += v * v;
                    i += 1;
                }
                total
            }
            // SAFETY: avx2 and fma were detected at runtime just above; the function
            // reads only indices below `x.len()`.
            return unsafe { sum_sq_avx2(x) };
        }
        let mut acc = [0.0f64; 4];
        let mut chunks = x.chunks_exact(4);
        for c in &mut chunks {
            acc[0] += (c[0] as f64) * (c[0] as f64);
            acc[1] += (c[1] as f64) * (c[1] as f64);
            acc[2] += (c[2] as f64) * (c[2] as f64);
            acc[3] += (c[3] as f64) * (c[3] as f64);
        }
        let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for &v in chunks.remainder() {
            total += (v as f64) * (v as f64);
        }
        total
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let mut acc = [0.0f64; 4];
        let mut chunks = x.chunks_exact(4);
        for c in &mut chunks {
            acc[0] += (c[0] as f64) * (c[0] as f64);
            acc[1] += (c[1] as f64) * (c[1] as f64);
            acc[2] += (c[2] as f64) * (c[2] as f64);
            acc[3] += (c[3] as f64) * (c[3] as f64);
        }
        let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for &v in chunks.remainder() {
            total += (v as f64) * (v as f64);
        }
        total
    }
}

/// RMSNorm: `x * w / sqrt(mean(x²) + eps)`, in place over one row.
///
/// The mean is accumulated in f64: at hidden sizes in the thousands, f32
/// accumulation loses enough bits to show up in logit comparisons.
pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len());
    let mean_sq = sum_sq_f64(x) / x.len() as f64;
    let scale = 1.0 / (mean_sq + eps as f64).sqrt();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `x.len() == weight.len()` (asserted above) and every access is at
    // `i < n = x.len()`; `x` is a unique borrow, so writes cannot alias `weight`.
    unsafe {
        use std::arch::aarch64::*;
        let vscale = vdupq_n_f64(scale);
        let mut i = 0usize;
        let n = x.len();
        let xp = x.as_mut_ptr();
        let wp = weight.as_ptr();
        while i + 8 <= n {
            let x0 = vld1q_f32(xp.add(i));
            let x1 = vld1q_f32(xp.add(i + 4));
            let w0 = vld1q_f32(wp.add(i));
            let w1 = vld1q_f32(wp.add(i + 4));
            let s0_lo = vcvt_f32_f64(vmulq_f64(vcvt_f64_f32(vget_low_f32(x0)), vscale));
            let s0_hi = vcvt_f32_f64(vmulq_f64(vcvt_high_f64_f32(x0), vscale));
            let s1_lo = vcvt_f32_f64(vmulq_f64(vcvt_f64_f32(vget_low_f32(x1)), vscale));
            let s1_hi = vcvt_f32_f64(vmulq_f64(vcvt_high_f64_f32(x1), vscale));
            vst1q_f32(xp.add(i), vmulq_f32(vcombine_f32(s0_lo, s0_hi), w0));
            vst1q_f32(xp.add(i + 4), vmulq_f32(vcombine_f32(s1_lo, s1_hi), w1));
            i += 8;
        }
        while i < n {
            *xp.add(i) = (*xp.add(i) as f64 * scale) as f32 * *wp.add(i);
            i += 1;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            #[target_feature(enable = "avx2")]
            unsafe fn rmsnorm_avx2(x: &mut [f32], weight: &[f32], scale: f64) {
                use std::arch::x86_64::*;
                let vscale = _mm256_set1_pd(scale);
                let mut i = 0usize;
                let n = x.len();
                let xp = x.as_mut_ptr();
                let wp = weight.as_ptr();
                while i + 8 <= n {
                    let x0 = _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i)));
                    let x1 = _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 4)));
                    let s0 = _mm256_cvtpd_ps(_mm256_mul_pd(x0, vscale));
                    let s1 = _mm256_cvtpd_ps(_mm256_mul_pd(x1, vscale));
                    let w0 = _mm_loadu_ps(wp.add(i));
                    let w1 = _mm_loadu_ps(wp.add(i + 4));
                    _mm_storeu_ps(xp.add(i), _mm_mul_ps(s0, w0));
                    _mm_storeu_ps(xp.add(i + 4), _mm_mul_ps(s1, w1));
                    i += 8;
                }
                while i < n {
                    *xp.add(i) = (*xp.add(i) as f64 * scale) as f32 * *wp.add(i);
                    i += 1;
                }
            }
            // SAFETY: avx2 was detected at runtime just above; the function touches
            // indices below `x.len()`, which equals `weight.len()` per the assert.
            unsafe { rmsnorm_avx2(x, weight, scale) };
            return;
        }
        for (v, w) in x.iter_mut().zip(weight) {
            *v = (*v as f64 * scale) as f32 * w;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for (v, w) in x.iter_mut().zip(weight) {
        *v = (*v as f64 * scale) as f32 * w;
    }
}

/// RMSNorm from `src` into `dst` (same formula as `rmsnorm`): one read and
/// one write per element, where clone-then-normalise-in-place costs a read
/// and TWO writes. The prefill loop runs this over every layer's full
/// activation batch, so the saved pass is real bandwidth.
pub fn rmsnorm_into(dst: &mut [f32], src: &[f32], weight: &[f32], eps: f32) {
    assert_eq!(src.len(), weight.len());
    assert_eq!(dst.len(), src.len());
    let mean_sq = sum_sq_f64(src) / src.len() as f64;
    let scale = 1.0 / (mean_sq + eps as f64).sqrt();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `dst`, `src` and `weight` have the same length (asserted above) and
    // every access is at `i < n = src.len()`; `dst` is a unique borrow, so writes
    // alias neither input.
    unsafe {
        use std::arch::aarch64::*;
        let vscale = vdupq_n_f64(scale);
        let mut i = 0usize;
        let n = src.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let wp = weight.as_ptr();
        while i + 8 <= n {
            let x0 = vld1q_f32(sp.add(i));
            let x1 = vld1q_f32(sp.add(i + 4));
            let w0 = vld1q_f32(wp.add(i));
            let w1 = vld1q_f32(wp.add(i + 4));
            let s0_lo = vcvt_f32_f64(vmulq_f64(vcvt_f64_f32(vget_low_f32(x0)), vscale));
            let s0_hi = vcvt_f32_f64(vmulq_f64(vcvt_high_f64_f32(x0), vscale));
            let s1_lo = vcvt_f32_f64(vmulq_f64(vcvt_f64_f32(vget_low_f32(x1)), vscale));
            let s1_hi = vcvt_f32_f64(vmulq_f64(vcvt_high_f64_f32(x1), vscale));
            vst1q_f32(dp.add(i), vmulq_f32(vcombine_f32(s0_lo, s0_hi), w0));
            vst1q_f32(dp.add(i + 4), vmulq_f32(vcombine_f32(s1_lo, s1_hi), w1));
            i += 8;
        }
        while i < n {
            *dp.add(i) = (*sp.add(i) as f64 * scale) as f32 * *wp.add(i);
            i += 1;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            #[target_feature(enable = "avx2")]
            unsafe fn rmsnorm_into_avx2(dst: &mut [f32], src: &[f32], weight: &[f32], scale: f64) {
                use std::arch::x86_64::*;
                let vscale = _mm256_set1_pd(scale);
                let mut i = 0usize;
                let n = src.len();
                let dp = dst.as_mut_ptr();
                let sp = src.as_ptr();
                let wp = weight.as_ptr();
                while i + 8 <= n {
                    let x0 = _mm256_cvtps_pd(_mm_loadu_ps(sp.add(i)));
                    let x1 = _mm256_cvtps_pd(_mm_loadu_ps(sp.add(i + 4)));
                    let s0 = _mm256_cvtpd_ps(_mm256_mul_pd(x0, vscale));
                    let s1 = _mm256_cvtpd_ps(_mm256_mul_pd(x1, vscale));
                    let w0 = _mm_loadu_ps(wp.add(i));
                    let w1 = _mm_loadu_ps(wp.add(i + 4));
                    _mm_storeu_ps(dp.add(i), _mm_mul_ps(s0, w0));
                    _mm_storeu_ps(dp.add(i + 4), _mm_mul_ps(s1, w1));
                    i += 8;
                }
                while i < n {
                    *dp.add(i) = (*sp.add(i) as f64 * scale) as f32 * *wp.add(i);
                    i += 1;
                }
            }
            // SAFETY: avx2 was detected at runtime just above; all three slices have
            // `src.len()` elements per the asserts, and the function stays below that.
            unsafe { rmsnorm_into_avx2(dst, src, weight, scale) };
            return;
        }
        for ((d, &v), w) in dst.iter_mut().zip(src).zip(weight) {
            *d = (v as f64 * scale) as f32 * w;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for ((d, &v), w) in dst.iter_mut().zip(src).zip(weight) {
        *d = (v as f64 * scale) as f32 * w;
    }
}

/// Rows of `src` normalised into the matching rows of `dst`, spread over the
/// persistent worker pool. Prefill batches are hundreds of rows of thousands of
/// elements - memory-bound, and single-threaded it was a visible slice of
/// the whole prefill.
pub fn rmsnorm_rows_into(dst: &mut [f32], src: &[f32], weight: &[f32], eps: f32) {
    let n = weight.len();
    let rows = src.len() / n;
    assert!(dst.len() >= rows * n, "dst holds fewer rows than src");
    let threads = crate::thread_pool::cpu_threads().min(rows.max(1));
    if threads <= 1 || rows < 4 || src.len() < 16_384 {
        for (d, s) in dst.chunks_mut(n).zip(src.chunks(n)) {
            rmsnorm_into(d, s, weight, eps);
        }
        return;
    }
    let dst_addr = dst.as_mut_ptr() as usize;
    let src_addr = src.as_ptr() as usize;
    crate::thread_pool::par_for_chunks(rows, threads, |_worker, start, end| {
        let len = (end - start) * n;
        // SAFETY: row ranges `[start, end)` from par_for_chunks are disjoint and end at
        // `rows`, so `(start * n, len)` is a unique sub-slice of `dst` as long as `dst`
        // holds `rows * n` floats (checked at the top). `dst` stays mutably borrowed
        // until par_for_chunks returns, after every worker finished.
        let dc =
            unsafe { std::slice::from_raw_parts_mut((dst_addr as *mut f32).add(start * n), len) };
        // SAFETY: the same row range of `src`, which has at least `rows * n` floats by the
        // definition of `rows`; it is only read and cannot overlap the unique `dst`.
        let sc =
            unsafe { std::slice::from_raw_parts((src_addr as *const f32).add(start * n), len) };
        for (d, s) in dc.chunks_mut(n).zip(sc.chunks(n)) {
            rmsnorm_into(d, s, weight, eps);
        }
    });
}

/// L2 normalisation to unit norm, like `ggml_l2_norm`: `x / sqrt(sum(x^2) + eps)`.
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let s = dot(x, x);
    let inv = 1.0 / (s + eps).sqrt();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: every access is at `i < n = x.len()` (8-wide while `i + 8 <= n`, then
    // scalar) through a pointer from the unique borrow `x`.
    unsafe {
        use std::arch::aarch64::*;
        let vinv = vdupq_n_f32(inv);
        let mut i = 0usize;
        let n = x.len();
        let xp = x.as_mut_ptr();
        while i + 8 <= n {
            vst1q_f32(xp.add(i), vmulq_f32(vld1q_f32(xp.add(i)), vinv));
            vst1q_f32(xp.add(i + 4), vmulq_f32(vld1q_f32(xp.add(i + 4)), vinv));
            i += 8;
        }
        while i < n {
            *xp.add(i) *= inv;
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for v in x.iter_mut() {
        *v *= inv;
    }
}
