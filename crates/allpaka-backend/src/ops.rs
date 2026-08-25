//! The activation-side operations of a Llama-family forward pass.
//!
//! Formulas follow ggml's f32 paths. Where two conventions exist in the wild
//! (RoPE most of all) both are provided under explicit names, because the
//! silent-wrong-variant failure mode produces plausible garbage that only a
//! logit comparison catches.

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
        for j in 0..n {
            let wj = &w[j * k..(j + 1) * k];
            y[i * n + j] = xi.iter().zip(wj).map(|(a, b)| a * b).sum();
        }
    }
    y
}

/// RMSNorm: `x * w / sqrt(mean(x²) + eps)`, in place over one row.
///
/// The mean is accumulated in f64: at hidden sizes in the thousands, f32
/// accumulation loses enough bits to show up in logit comparisons.
pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len());
    let mean_sq = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (mean_sq + eps as f64).sqrt();
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
    let mean_sq = src.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / src.len() as f64;
    let scale = 1.0 / (mean_sq + eps as f64).sqrt();
    for ((d, &v), w) in dst.iter_mut().zip(src).zip(weight) {
        *d = (v as f64 * scale) as f32 * w;
    }
}

/// Rows of `src` normalised into the matching rows of `dst`, spread over the
/// available cores. Prefill batches are hundreds of rows of thousands of
/// elements - memory-bound, and single-threaded it was a visible slice of
/// the whole prefill.
pub fn rmsnorm_rows_into(
    dst: &mut [f32],
    src: &[f32],
    weight: &[f32],
    eps: f32,
) {
    let n = weight.len();
    let rows = src.len() / n;
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(rows.max(1));
    if threads <= 1 || rows < 4 {
        for (d, s) in dst.chunks_mut(n).zip(src.chunks(n)) {
            rmsnorm_into(d, s, weight, eps);
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (dc, sc) in dst.chunks_mut(per * n).zip(src.chunks(per * n)) {
            scope.spawn(move || {
                for (d, s) in dc.chunks_mut(n).zip(sc.chunks(n)) {
                    rmsnorm_into(d, s, weight, eps);
                }
            });
        }
    });
}

/// `a += b` element-wise, spread over cores; the prefill residual adds run
/// over the same hundreds-of-rows batches as the norms.
pub fn add_assign_par(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let per = a.len().div_ceil(threads).max(1 << 14);
    if a.len() <= per {
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
        return;
    }
    std::thread::scope(|scope| {
        for (ac, bc) in a.chunks_mut(per).zip(b.chunks(per)) {
            scope.spawn(move || {
                for (x, y) in ac.iter_mut().zip(bc) {
                    *x += y;
                }
            });
        }
    });
}

/// Softmax in place over one row, with the usual max subtraction so large
/// logits do not overflow to infinity.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// SiLU: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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
    for (g, u) in gate.iter_mut().zip(up) {
        *g = silu(*g) * u;
    }
}

/// The position-independent half of RoPE: `inv_freq[i] = base^(-2i/d)`.
///
/// Only `head_dim / 2` values exist per model, yet the naive path recomputes
/// `powf` per element per head per layer per token - over 100k calls per
/// decoded token on a 48-layer model. Compute once at load.
pub fn rope_inv_freq(head_dim: usize, freq_base: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| freq_base.powf(-2.0 * i as f32 / head_dim as f32))
        .collect()
}

/// The per-position half of RoPE: `(sin, cos)` of `pos * inv_freq[i]`.
///
/// A token's position is shared by every head of every layer, so this is
/// computed once per position and reused ~1,700 times per token.
pub fn rope_sin_cos(pos: u32, inv_freq: &[f32], out: &mut Vec<(f32, f32)>) {
    out.clear();
    out.extend(inv_freq.iter().map(|&f| (pos as f32 * f).sin_cos()));
}

/// [`rope_norm`] over a precomputed per-position table. Bit-identical to the
/// naive form: the table holds the same `sin_cos` of the same `f32` products.
pub fn rope_norm_cached(x: &mut [f32], table: &[(f32, f32)]) {
    assert_eq!(x.len(), table.len() * 2);
    for (i, &(sin, cos)) in table.iter().enumerate() {
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * cos - b * sin;
        x[2 * i + 1] = a * sin + b * cos;
    }
}

/// [`rope_neox`] over a precomputed per-position table.
pub fn rope_neox_cached(x: &mut [f32], table: &[(f32, f32)]) {
    assert_eq!(x.len(), table.len() * 2);
    let half = table.len();
    for (i, &(sin, cos)) in table.iter().enumerate() {
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
    }
}

/// [`rope_norm`] over a precomputed per-position table stored as arrays.
pub fn rope_norm_cached_from_array(x: &mut [f32], table: &[[f32; 2]]) {
    assert_eq!(x.len(), table.len() * 2);
    for (i, pair) in table.iter().enumerate() {
        let sin = pair[0];
        let cos = pair[1];
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * cos - b * sin;
        x[2 * i + 1] = a * sin + b * cos;
    }
}

/// [`rope_neox`] over a precomputed per-position table stored as arrays.
///
/// The table may cover only a leading slice of the head (partial rotary,
/// GLM): those channels rotate and the tail passes through unchanged.
pub fn rope_neox_cached_from_array(x: &mut [f32], table: &[[f32; 2]]) {
    assert!(x.len() >= table.len() * 2);
    let half = table.len();
    for (i, pair) in table.iter().enumerate() {
        let sin = pair[0];
        let cos = pair[1];
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
    }
}

/// Rotary embedding, "NORM" style: adjacent pairs `(x[2i], x[2i+1])` rotate
/// together. Llama 3.x uses this.
///
/// `x` is one head of dimension `head_dim` at sequence position `pos`;
/// `freq_base` is the model's `rope.freq_base` (10000 classically, 500000 for
/// Llama 3). This form recomputes the frequencies per call; the forward pass
/// uses the `_cached` variants above, checked against this one in tests.
pub fn rope_norm(x: &mut [f32], pos: u32, freq_base: f32) {
    let d = x.len();
    assert!(d % 2 == 0);
    for i in 0..d / 2 {
        let theta = pos as f32 * freq_base.powf(-2.0 * i as f32 / d as f32);
        let (sin, cos) = theta.sin_cos();
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * cos - b * sin;
        x[2 * i + 1] = a * sin + b * cos;
    }
}

/// Rotary embedding, "NEOX" style: split pairs `(x[i], x[i + d/2])` rotate
/// together. Qwen and most recent non-Llama models use this.
pub fn rope_neox(x: &mut [f32], pos: u32, freq_base: f32) {
    let d = x.len();
    assert!(d % 2 == 0);
    let half = d / 2;
    for i in 0..half {
        let theta = pos as f32 * freq_base.powf(-2.0 * i as f32 / d as f32);
        let (sin, cos) = theta.sin_cos();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
    }
}

/// Half-precision helpers for the KV cache.
///
/// The cache is the one tensor the engine writes every step and reads in full
/// every step, so it is stored as f16: half the bytes for a format whose
/// precision is far beyond what attention scores need. The GPU reads `half`
/// natively; these are for the CPU path, which is the fallback for decode and
/// still carries the whole of prefill.
///
/// Conversion is one instruction per four values on aarch64. A scalar loop
/// would give back more than the halved traffic wins - the same trap the
/// scalar `dot` was.
pub mod f16 {
    /// Convert `src` to half, in place into `dst`. Lengths must match.
    pub fn from_f32(src: &[f32], dst: &mut [u16]) {
        assert_eq!(src.len(), dst.len());
        #[cfg(target_arch = "aarch64")]
        {
            let mut i = 0;
            unsafe {
                use std::arch::aarch64::*;
                while i + 4 <= src.len() {
                    let v = vld1q_f32(src.as_ptr().add(i));
                    let h = vcvt_f16_f32(v);
                    std::ptr::copy_nonoverlapping(
                        &h as *const _ as *const u16,
                        dst.as_mut_ptr().add(i),
                        4,
                    );
                    i += 4;
                }
            }
            for (d, s) in dst[i..].iter_mut().zip(&src[i..]) {
                *d = scalar_from_f32(*s);
            }
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (d, s) in dst.iter_mut().zip(src) {
            *d = scalar_from_f32(*s);
        }
    }

    /// `sum(a[i] * b[i])` with `b` in half.
    pub fn dot(a: &[f32], b: &[u16]) -> f32 {
        let n = a.len().min(b.len());
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // Four independent accumulators for the same reason ops::dot has
            // eight: one would serialise the whole loop.
            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 4 <= n {
                let h: float16x4_t = std::mem::transmute(std::ptr::read_unaligned(
                    b.as_ptr().add(i) as *const [u16; 4],
                ));
                acc = vfmaq_f32(acc, vld1q_f32(a.as_ptr().add(i)), vcvt_f32_f16(h));
                i += 4;
            }
            let mut total = vaddvq_f32(acc);
            for k in i..n {
                total += a[k] * to_f32(b[k]);
            }
            return total;
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
            while i + 4 <= n {
                let h: float16x4_t = std::mem::transmute(std::ptr::read_unaligned(
                    v.as_ptr().add(i) as *const [u16; 4],
                ));
                let a = vld1q_f32(acc.as_ptr().add(i));
                vst1q_f32(acc.as_mut_ptr().add(i), vfmaq_f32(a, pv, vcvt_f32_f16(h)));
                i += 4;
            }
            for k in i..n {
                acc[k] += p * to_f32(v[k]);
            }
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
        for k in 0..n {
            acc[k] += p * to_f32(v[k]);
        }
    }

    /// One value back to f32; the arbiter for the vector paths above.
    pub fn to_f32(h: u16) -> f32 {
        allpaka_gguf::dequant::f16_to_f32(h)
    }

    /// One value to half, round-to-nearest-even, with overflow to infinity
    /// and subnormals preserved.
    pub fn scalar_from_f32(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let mag = bits & 0x7fff_ffff;
        if mag >= 0x7f80_0000 {
            // Inf or NaN: keep NaN non-zero in the mantissa.
            let man = if mag > 0x7f80_0000 { 0x200 } else { 0 };
            return sign | 0x7c00 | man;
        }
        // Round-to-nearest-even in f32 first, by adding half an ulp of the
        // half-precision mantissa, then truncate.
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
        // Subnormal half: shift the implicit one into place.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5 * (1.0 + b.abs())
    }

    /// Round-tripping through half must reproduce exactly the value the
    /// reader gives back, across normals, subnormals, the overflow edge and
    /// zero. A wrong rounding here would be invisible in a benchmark and
    /// would quietly cost accuracy in every cached key.
    #[test]
    fn half_conversion_round_trips_through_the_gguf_reader() {
        let cases: Vec<f32> = vec![
            0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 65504.0, -65504.0,
            1e-5, -1e-5, 6.0e-8, 1.0 / 3.0, 1e8, -1e8, f32::INFINITY,
            -f32::INFINITY, 0.30078125, 1.0009765625,
        ];
        for &x in &cases {
            let h = f16::scalar_from_f32(x);
            let back = f16::to_f32(h);
            let expect = if x.abs() > 65519.0 {
                f32::INFINITY * x.signum()
            } else {
                back
            };
            assert_eq!(back, expect, "{x} -> {h:04x} -> {back}");
            // The value must be the nearest half, so re-rounding is a no-op.
            assert_eq!(f16::scalar_from_f32(back), h, "{x} does not settle");
        }
        assert!(f16::to_f32(f16::scalar_from_f32(f32::NAN)).is_nan());
    }

    /// The vector paths and the scalar reference must agree exactly: they are
    /// the same arithmetic, and any divergence is a bug in the intrinsics.
    #[test]
    fn vector_half_paths_match_the_scalar_reference() {
        let n = 131; // deliberately not a multiple of the vector width
        let src: Vec<f32> = (0..n).map(|i| (i as f32 - 65.0) * 0.017).collect();
        let mut halves = vec![0u16; n];
        f16::from_f32(&src, &mut halves);
        for (i, (&h, &x)) in halves.iter().zip(&src).enumerate() {
            assert_eq!(h, f16::scalar_from_f32(x), "element {i}");
        }

        let a: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let want: f32 = a.iter().zip(&halves).map(|(x, &h)| x * f16::to_f32(h)).sum();
        let got = f16::dot(&a, &halves);
        assert!((got - want).abs() < 1e-3 * (1.0 + want.abs()), "{got} vs {want}");

        let mut acc = vec![1.0f32; n];
        let mut want_acc = acc.clone();
        f16::axpy(&mut acc, 0.25, &halves);
        for (w, &h) in want_acc.iter_mut().zip(&halves) {
            *w += 0.25 * f16::to_f32(h);
        }
        for (i, (g, w)) in acc.iter().zip(&want_acc).enumerate() {
            assert!((g - w).abs() < 1e-6, "element {i}: {g} vs {w}");
        }
    }

    #[test]
    fn matmul_matches_a_hand_example() {
        // x = [[1,2],[3,4]], w rows = [[1,0],[0,1],[1,1]]
        let y = matmul_f32(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2, 2, 3);
        assert_eq!(y, vec![1.0, 2.0, 3.0, 3.0, 4.0, 7.0]);
    }

    #[test]
    fn rmsnorm_normalises_to_unit_rms_before_the_weight() {
        let mut x = vec![3.0f32, -4.0, 12.0, 0.0];
        let w = vec![1.0f32; 4];
        rmsnorm(&mut x, &w, 0.0);
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!(close(rms, 1.0), "{rms}");
        // Direction is preserved.
        assert!(x[0] > 0.0 && x[1] < 0.0 && x[3] == 0.0);
    }

    #[test]
    fn rmsnorm_applies_the_weight_per_channel() {
        let mut x = vec![1.0f32, 1.0];
        rmsnorm(&mut x, &[2.0, 3.0], 0.0);
        assert!(close(x[0], 2.0) && close(x[1], 3.0), "{x:?}");
    }

    /// The eps term must matter for tiny inputs - it is what keeps a
    /// zero-activation row finite.
    #[test]
    fn rmsnorm_eps_keeps_zero_input_finite() {
        let mut x = vec![0.0f32; 8];
        rmsnorm(&mut x, &[1.0; 8], 1e-5);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_sums_to_one_and_orders_like_the_input() {
        let mut x = vec![1.0f32, 3.0, 2.0];
        softmax(&mut x);
        assert!(close(x.iter().sum::<f32>(), 1.0));
        assert!(x[1] > x[2] && x[2] > x[0]);
    }

    #[test]
    fn softmax_survives_huge_logits() {
        let mut x = vec![1000.0f32, 999.0];
        softmax(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(close(x[0] + x[1], 1.0));
        assert!(x[0] > x[1]);
    }

    #[test]
    fn silu_has_the_right_fixed_points() {
        assert_eq!(silu(0.0), 0.0);
        assert!(close(silu(1.0), 1.0 / (1.0 + (-1.0f32).exp())));
        // Large negative saturates to ~0, large positive to ~x.
        assert!(silu(-20.0).abs() < 1e-7);
        assert!(close(silu(20.0), 20.0));
    }

    #[test]
    fn swiglu_is_silu_of_gate_times_up() {
        let mut gate = vec![1.0f32, -2.0];
        swiglu(&mut gate, &[3.0, 5.0]);
        assert!(close(gate[0], silu(1.0) * 3.0));
        assert!(close(gate[1], silu(-2.0) * 5.0));
    }

    /// Rotation preserves the norm of every pair, and position 0 is identity.
    #[test]
    fn rope_is_norm_preserving_and_identity_at_position_zero() {
        for rope in [rope_norm, rope_neox] {
            let orig: Vec<f32> = (0..8).map(|i| (i as f32) - 3.5).collect();
            let mut x = orig.clone();
            rope(&mut x, 0, 10000.0);
            assert_eq!(x, orig, "pos 0 must be identity");

            rope(&mut x, 17, 10000.0);
            let n0: f32 = orig.iter().map(|v| v * v).sum();
            let n1: f32 = x.iter().map(|v| v * v).sum();
            assert!(close(n0, n1), "{n0} vs {n1}");
            assert_ne!(x, orig);
        }
    }

    /// Pin each variant's pairing: rotate a one-hot vector one step and watch
    /// which other channel receives the energy.
    #[test]
    fn rope_variants_pair_different_channels() {
        let mut x = vec![0f32; 8];
        x[0] = 1.0;
        rope_norm(&mut x, 1, 10000.0);
        // NORM pairs (0,1): channel 1 gets sin, channels 4.. stay 0.
        assert!(close(x[1], 1f32.sin()));
        assert_eq!(x[4], 0.0);

        let mut x = vec![0f32; 8];
        x[0] = 1.0;
        rope_neox(&mut x, 1, 10000.0);
        // NEOX pairs (0,4): channel 4 gets sin, channel 1 stays 0.
        assert!(close(x[4], 1f32.sin()));
        assert_eq!(x[1], 0.0);
    }

    /// The cached variants must agree with the naive ones exactly: same f32
    /// products, same sin_cos, just hoisted out of the inner loops.
    #[test]
    fn cached_rope_is_bit_identical_to_the_naive_form() {
        let inv = rope_inv_freq(8, 10000.0);
        let mut table = Vec::new();
        for pos in [0u32, 1, 17, 5000] {
            rope_sin_cos(pos, &inv, &mut table);

            let orig: Vec<f32> = (0..8).map(|i| (i as f32) - 3.5).collect();
            let mut a = orig.clone();
            let mut b = orig.clone();
            rope_norm(&mut a, pos, 10000.0);
            rope_norm_cached(&mut b, &table);
            assert_eq!(a, b, "norm at pos {pos}");

            let mut a = orig.clone();
            let mut b = orig;
            rope_neox(&mut a, pos, 10000.0);
            rope_neox_cached(&mut b, &table);
            assert_eq!(a, b, "neox at pos {pos}");
        }
    }

    /// Frequencies must fall with channel index: the last pair barely moves
    /// at small positions while the first pair rotates by ~1 radian.
    #[test]
    fn rope_frequencies_decay_across_the_head() {
        let mut x = vec![1.0f32; 8];
        rope_neox(&mut x, 1, 10000.0);
        let first_moved = (x[0] - 1.0).abs();
        let last_moved = (x[3] - 1.0).abs();
        assert!(first_moved > 0.3, "{first_moved}");
        assert!(last_moved < 0.01, "{last_moved}");
    }

    /// The rotation must compose: rotating by pos p equals rotating twice by
    /// smaller steps is NOT true (absolute encoding), but rotating a vector at
    /// pos a and another at pos b must give a dot product that depends only on
    /// b - a. That relative property is the entire point of RoPE.
    #[test]
    fn rope_dot_products_depend_only_on_relative_position() {
        let q: Vec<f32> = (0..8).map(|i| ((i * 7 + 3) % 5) as f32 - 2.0).collect();
        let k: Vec<f32> = (0..8).map(|i| ((i * 3 + 1) % 7) as f32 - 3.0).collect();

        let dot_at = |pq: u32, pk: u32| -> f32 {
            let mut a = q.clone();
            let mut b = k.clone();
            rope_neox(&mut a, pq, 10000.0);
            rope_neox(&mut b, pk, 10000.0);
            a.iter().zip(&b).map(|(x, y)| x * y).sum()
        };

        assert!(close(dot_at(5, 3), dot_at(102, 100)));
        assert!(close(dot_at(9, 0), dot_at(59, 50)));
        assert!(!close(dot_at(5, 3), dot_at(5, 0)), "different gaps must differ");
    }
}

/// `sum(a[i] * b[i])` over the shorter of the two.
///
/// Used by every CPU inner product on the hot path - the quantised matvec
/// reference, the F32 router, and attention's q-dot-k.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight running sums, not one. A single accumulator is a serial
    // dependency chain that the compiler may not reassociate - f32 addition
    // is not associative, so it is not allowed to - and the loop stays
    // scalar. Independent lanes give it something it can vectorise, at the
    // cost of a different (not worse) summation order.
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
