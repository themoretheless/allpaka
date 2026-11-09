//! The activation-side operations of a Llama-family forward pass.
//!
//! Formulas follow ggml's f32 paths. Where two conventions exist in the wild
//! (RoPE most of all) both are provided under explicit names, because the
//! silent-wrong-variant failure mode produces plausible garbage that only a
//! logit comparison catches.

pub mod activation;
pub mod blas;
pub mod deltanet;
pub mod f16;
pub mod norm;
pub mod rope;

pub use activation::*;
pub use blas::*;
pub use deltanet::*;
pub use norm::*;
pub use rope::*;

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
    #[allow(clippy::excessive_precision)]
    fn half_conversion_round_trips_through_the_gguf_reader() {
        let cases: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            65504.0,
            -65504.0,
            1e-5,
            -1e-5,
            6.0e-8,
            1.0 / 3.0,
            1e8,
            -1e8,
            f32::INFINITY,
            -f32::INFINITY,
            0.30078125,
            1.0009765625,
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
        let want: f32 = a
            .iter()
            .zip(&halves)
            .map(|(x, &h)| x * f16::to_f32(h))
            .sum();
        let got = f16::dot(&a, &halves);
        assert!(
            (got - want).abs() < 1e-3 * (1.0 + want.abs()),
            "{got} vs {want}"
        );

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
        let y = matmul_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
            2,
            3,
        );
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
        let mut gate: Vec<f32> = (-25..26).map(|i| i as f32 * 0.7).collect();
        let up: Vec<f32> = (0..gate.len()).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let orig_gate = gate.clone();
        swiglu(&mut gate, &up);
        for ((&got, &g), &u) in gate.iter().zip(&orig_gate).zip(&up) {
            assert!(close(got, silu(g) * u), "g={g} u={u}: got={got} vs {}", silu(g) * u);
        }
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
        assert!(
            !close(dot_at(5, 3), dot_at(5, 0)),
            "different gaps must differ"
        );
    }
}

#[cfg(test)]
mod gdn_tests {
    use super::*;

    /// The gated-delta-net step, written out independently from the
    /// llama.cpp autoregressive formula with explicit temporaries:
    /// S1 = S*exp(g); sk = S1.k; d = (v-sk)*beta; S2 = S1 + k.d; y = S2.q/sqrt(d).
    fn reference_step(
        s0: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: f32,
        b: f32,
        d: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut s1 = s0.to_vec();
        for x in s1.iter_mut() {
            *x *= g.exp();
        }
        let mut out = vec![0f32; d];
        let mut s2 = s1.clone();
        for i in 0..d {
            let mut sk = 0.0f32;
            for j in 0..d {
                sk += s1[i * d + j] * k[j];
            }
            let delta = (v[i] - sk) * b;
            for j in 0..d {
                s2[i * d + j] += k[j] * delta;
                out[i] += s2[i * d + j] * q[j];
            }
            out[i] /= (d as f32).sqrt();
        }
        (s2, out)
    }

    #[test]
    fn deltanet_step_matches_the_independent_formula() {
        let (heads, d) = (3usize, 8usize);
        let mut seed = 0x5eedu64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as u32) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let state0: Vec<f32> = (0..heads * d * d).map(|_| rnd()).collect();
        let q: Vec<f32> = (0..heads * d).map(|_| rnd()).collect();
        let k: Vec<f32> = (0..heads * d).map(|_| rnd()).collect();
        let v: Vec<f32> = (0..heads * d).map(|_| rnd()).collect();
        let gate: Vec<f32> = (0..heads).map(|_| rnd()).collect();
        let beta: Vec<f32> = (0..heads).map(|_| rnd().abs()).collect();

        let mut state = state0.clone();
        let mut out = vec![0f32; heads * d];
        gated_deltanet_step(&mut state, &q, &k, &v, &gate, &beta, &mut out);

        for h in 0..heads {
            let (s_ref, o_ref) = reference_step(
                &state0[h * d * d..(h + 1) * d * d],
                &q[h * d..(h + 1) * d],
                &k[h * d..(h + 1) * d],
                &v[h * d..(h + 1) * d],
                gate[h],
                beta[h],
                d,
            );
            for j in 0..d * d {
                let a = state[h * d * d + j];
                let b = s_ref[j];
                assert!((a - b).abs() < 1e-5, "state mismatch h{h} j{j}: {a} vs {b}");
            }
            for i in 0..d {
                let a = out[h * d + i];
                let b = o_ref[i];
                assert!((a - b).abs() < 1e-5, "out mismatch h{h} i{i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn l2_norm_unitizes_and_softplus_is_stable() {
        let mut x = vec![3.0f32, 4.0];
        l2_norm(&mut x, 1e-6);
        let n: f32 = x.iter().map(|v| v * v).sum();
        assert!((n - 1.0).abs() < 1e-5);
        assert_eq!(softplus(50.0), 50.0);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
    }
}
