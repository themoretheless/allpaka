//! Nonlinearities (`softmax`, `silu`, `sigmoid`, `swiglu`, `softplus`) with aarch64 NEON vectorization.

/// Softmax in place over one row, with the usual max subtraction so large
/// logits do not overflow to infinity.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// SiLU: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `ln(1 + e^x)`, the delta-net's dt activation.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid in place over one row: the GLM-family MoE router scores each
/// expert independently instead of softmaxing across them.
pub fn sigmoid(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// SwiGLU: `silu(gate) * up`, elementwise into `gate`.
///
/// This is the FFN nonlinearity of every model this engine targets; the two
/// inputs are the gate and up projections of the same activation row.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = gate.len();
        let gp = gate.as_mut_ptr();
        let up_p = up.as_ptr();
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        let exp_hi = vdupq_n_f32(88.376_26_f32);
        let exp_lo = vdupq_n_f32(-88.376_26_f32);
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        // Cody-Waite split of ln(2) = c1 + c2
        let ln2_hi = vdupq_n_f32(0.693_145_75_f32);
        let ln2_lo = vdupq_n_f32(1.428_606_8e-6_f32);
        let c0 = vdupq_n_f32(1.987_569_1e-4_f32);
        let c1 = vdupq_n_f32(1.398_2e-3_f32);
        let c2 = vdupq_n_f32(8.333_452e-3_f32);
        let c3 = vdupq_n_f32(4.166_579_6e-2_f32);
        let c4 = vdupq_n_f32(1.666_666_6e-1_f32);
        let c5 = vdupq_n_f32(0.5_f32);
        let bias = vdupq_n_s32(0x7f);

        let mut i = 0usize;
        while i + 4 <= n {
            let g = vld1q_f32(gp.add(i));
            let u = vld1q_f32(up_p.add(i));
            let neg_g = vnegq_f32(g);
            let x = vmaxq_f32(vminq_f32(neg_g, exp_hi), exp_lo);
            let fx = vrndnq_f32(vmulq_f32(x, log2e));
            let r = vfmsq_f32(vfmsq_f32(x, fx, ln2_hi), fx, ln2_lo);
            let r2 = vmulq_f32(r, r);
            let mut y = vfmaq_f32(c1, c0, r);
            y = vfmaq_f32(c2, y, r);
            y = vfmaq_f32(c3, y, r);
            y = vfmaq_f32(c4, y, r);
            y = vfmaq_f32(c5, y, r);
            y = vfmaq_f32(r, y, r2);
            y = vaddq_f32(y, one);
            let emm0 = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), bias), 23);
            let exp_neg_g = vmulq_f32(y, vreinterpretq_f32_s32(emm0));
            let silu_g = vbslq_f32(
                vcgtq_f32(neg_g, exp_hi),
                zero,
                vdivq_f32(g, vaddq_f32(one, exp_neg_g)),
            );
            vst1q_f32(gp.add(i), vmulq_f32(silu_g, u));
            i += 4;
        }
        while i < n {
            *gp.add(i) = silu(*gp.add(i)) * *up_p.add(i);
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (g, u) in gate.iter_mut().zip(up) {
        *g = silu(*g) * u;
    }
}
