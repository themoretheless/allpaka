//! Dense f32 BLAS-1 / BLAS-3 helpers (`dot`, `matmul_f32`, `add_assign_slice`, `add_assign_par`, `add_scaled_slice`)
//! with aarch64 NEON and x86_64 AVX2+FMA fast paths.

/// `y = x · Wᵀ` over plain f32, shapes `x: [m, k]`, `w: [n, k]`, `y: [m, n]`.
///
/// For quantised weights use [`crate::QuantMat::matmul`]; this exists for the
/// few f32 tensors (norms are vectors, but e.g. token embeddings may be f32)
/// and for tests.
pub fn matmul_f32(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * k);
    let mut y = vec![0f32; m * n];
    for i in 0..m {
        let xi = &x[i * k..(i + 1) * k];
        let y_row = &mut y[i * n..(i + 1) * n];
        for (j, out) in y_row.iter_mut().enumerate() {
            let wj = &w[j * k..(j + 1) * k];
            *out = dot(xi, wj);
        }
    }
    y
}

#[inline(always)]
pub fn add_scaled_slice(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let vs = vdupq_n_f32(scale);
        let mut i = 0usize;
        let n = dst.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        while i + 16 <= n {
            vst1q_f32(
                dp.add(i),
                vfmaq_f32(vld1q_f32(dp.add(i)), vld1q_f32(sp.add(i)), vs),
            );
            vst1q_f32(
                dp.add(i + 4),
                vfmaq_f32(vld1q_f32(dp.add(i + 4)), vld1q_f32(sp.add(i + 4)), vs),
            );
            vst1q_f32(
                dp.add(i + 8),
                vfmaq_f32(vld1q_f32(dp.add(i + 8)), vld1q_f32(sp.add(i + 8)), vs),
            );
            vst1q_f32(
                dp.add(i + 12),
                vfmaq_f32(vld1q_f32(dp.add(i + 12)), vld1q_f32(sp.add(i + 12)), vs),
            );
            i += 16;
        }
        while i < n {
            *dp.add(i) = (*sp.add(i)).mul_add(scale, *dp.add(i));
            i += 1;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn add_scaled_avx2(dst: &mut [f32], src: &[f32], scale: f32) {
                use std::arch::x86_64::*;
                let vs = _mm256_set1_ps(scale);
                let mut i = 0usize;
                let n = dst.len();
                let dp = dst.as_mut_ptr();
                let sp = src.as_ptr();
                while i + 16 <= n {
                    let d0 = _mm256_loadu_ps(dp.add(i));
                    let s0 = _mm256_loadu_ps(sp.add(i));
                    _mm256_storeu_ps(dp.add(i), _mm256_fmadd_ps(s0, vs, d0));
                    let d1 = _mm256_loadu_ps(dp.add(i + 8));
                    let s1 = _mm256_loadu_ps(sp.add(i + 8));
                    _mm256_storeu_ps(dp.add(i + 8), _mm256_fmadd_ps(s1, vs, d1));
                    i += 16;
                }
                while i < n {
                    *dp.add(i) = (*sp.add(i)).mul_add(scale, *dp.add(i));
                    i += 1;
                }
            }
            unsafe { add_scaled_avx2(dst, src, scale) };
            return;
        }
        for (d, &s) in dst.iter_mut().zip(src) {
            *d += scale * s;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += scale * s;
    }
}

#[inline(always)]
pub fn add_assign_slice(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut i = 0usize;
        let n = a.len();
        let ap = a.as_mut_ptr();
        let bp = b.as_ptr();
        while i + 16 <= n {
            vst1q_f32(
                ap.add(i),
                vaddq_f32(vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i))),
            );
            vst1q_f32(
                ap.add(i + 4),
                vaddq_f32(vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4))),
            );
            vst1q_f32(
                ap.add(i + 8),
                vaddq_f32(vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8))),
            );
            vst1q_f32(
                ap.add(i + 12),
                vaddq_f32(vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12))),
            );
            i += 16;
        }
        while i < n {
            *ap.add(i) += *bp.add(i);
            i += 1;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            #[target_feature(enable = "avx2")]
            unsafe fn add_assign_avx2(a: &mut [f32], b: &[f32]) {
                use std::arch::x86_64::*;
                let mut i = 0usize;
                let n = a.len();
                let ap = a.as_mut_ptr();
                let bp = b.as_ptr();
                while i + 16 <= n {
                    let a0 = _mm256_loadu_ps(ap.add(i));
                    let b0 = _mm256_loadu_ps(bp.add(i));
                    _mm256_storeu_ps(ap.add(i), _mm256_add_ps(a0, b0));
                    let a1 = _mm256_loadu_ps(ap.add(i + 8));
                    let b1 = _mm256_loadu_ps(bp.add(i + 8));
                    _mm256_storeu_ps(ap.add(i + 8), _mm256_add_ps(a1, b1));
                    i += 16;
                }
                while i < n {
                    *ap.add(i) += *bp.add(i);
                    i += 1;
                }
            }
            unsafe { add_assign_avx2(a, b) };
            return;
        }
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// `a += b` element-wise, spread over the persistent worker pool; the prefill
/// residual adds run over the same hundreds-of-rows batches as the norms.
pub fn add_assign_par(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let threads = crate::thread_pool::cpu_threads();
    let per = a.len().div_ceil(threads).max(1 << 16);
    if a.len() <= per || threads <= 1 {
        add_assign_slice(a, b);
        return;
    }
    let n_chunks = a.len().div_ceil(per);
    let a_addr = a.as_mut_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let total = a.len();
    crate::thread_pool::par_for_chunks(n_chunks, threads, |_worker, c_start, c_end| {
        let elem_start = c_start * per;
        let elem_end = (c_end * per).min(total);
        if elem_start < elem_end {
            let len = elem_end - elem_start;
            let ac = unsafe { std::slice::from_raw_parts_mut((a_addr as *mut f32).add(elem_start), len) };
            let bc = unsafe { std::slice::from_raw_parts((b_addr as *const f32).add(elem_start), len) };
            add_assign_slice(ac, bc);
        }
    });
}

/// `sum(a[i] * b[i])` over the shorter of the two.
///
/// Used by every CPU inner product on the hot path - the quantised matvec
/// reference, the F32 router, and attention's q-dot-k.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            i += 4;
        }
        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let mut total = vaddvq_f32(vaddq_f32(sum01, sum23));
        for k in i..n {
            total += a[k] * b[k];
        }
        total
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn dot_avx2(a: &[f32], b: &[f32], n: usize) -> f32 {
                use std::arch::x86_64::*;
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut i = 0usize;
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                while i + 32 <= n {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 8)), _mm256_loadu_ps(bp.add(i + 8)), acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 16)), _mm256_loadu_ps(bp.add(i + 16)), acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 24)), _mm256_loadu_ps(bp.add(i + 24)), acc3);
                    i += 32;
                }
                while i + 8 <= n {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc0);
                    i += 8;
                }
                let sum01 = _mm256_add_ps(acc0, acc1);
                let sum23 = _mm256_add_ps(acc2, acc3);
                let sum = _mm256_add_ps(sum01, sum23);
                let hi = _mm256_extractf128_ps(sum, 1);
                let lo = _mm256_castps256_ps128(sum);
                let sum128 = _mm_add_ps(lo, hi);
                let shuf = _mm_movehdup_ps(sum128);
                let sums = _mm_add_ps(sum128, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let mut total = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
                for k in i..n {
                    total = a[k].mul_add(b[k], total);
                }
                total
            }
            return unsafe { dot_avx2(a, b, n) };
        }
        let a = &a[..n];
        let b = &b[..n];
        const LANES: usize = 8;
        let mut acc = [0f32; LANES];
        let mut ac = a.chunks_exact(LANES);
        let mut bc = b.chunks_exact(LANES);
        for (x, y) in ac.by_ref().zip(bc.by_ref()) {
            for l in 0..LANES {
                acc[l] += x[l] * y[l];
            }
        }
        let mut total: f32 = acc.iter().sum();
        for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
            total += x * y;
        }
        total
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let a = &a[..n];
        let b = &b[..n];
        const LANES: usize = 8;
        let mut acc = [0f32; LANES];
        let mut ac = a.chunks_exact(LANES);
        let mut bc = b.chunks_exact(LANES);
        for (x, y) in ac.by_ref().zip(bc.by_ref()) {
            for l in 0..LANES {
                acc[l] += x[l] * y[l];
            }
        }
        let mut total: f32 = acc.iter().sum();
        for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
            total += x * y;
        }
        total
    }
}
