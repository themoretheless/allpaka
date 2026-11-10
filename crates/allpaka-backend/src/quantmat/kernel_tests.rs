//! Every CPU dot kernel against the dequantizer: the float kernels must match
//! `dequant(row) . x`, and the Q8-activation kernels must match
//! `dequant(row) . q8(x)`, where q8(x) is the activation as the kernel sees
//! it after quantization. Rows are random bytes with tamed f16 scales, so the
//! packed fields (scales, mins, high bits) take arbitrary values.

use super::dot_f32::*;
use super::q8_acts::*;
use allpaka_gguf::{dequant, GgmlType};

const N_IN: usize = 512;

fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        })
        .collect()
}

/// A row of `ty` with every f16 scale at `scale_offsets` set to ~0.01.
fn row(ty: GgmlType, block_bytes: usize, block_elems: usize, scale_offsets: &[usize]) -> Vec<u8> {
    let mut row = random_bytes(
        N_IN / block_elems * block_bytes,
        0x5EED ^ block_bytes as u64,
    );
    for block in row.chunks_exact_mut(block_bytes) {
        for &at in scale_offsets {
            block[at..at + 2].copy_from_slice(&0x211Fu16.to_le_bytes());
        }
    }
    assert_eq!(ty.expected_bytes(N_IN).unwrap(), row.len());
    row
}

fn activations() -> Vec<f32> {
    (0..N_IN)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.03)
        .collect()
}

fn check(name: &str, got: f32, w: &[f32], x: &[f32]) {
    let want: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
    let mag: f32 = w.iter().zip(x).map(|(a, b)| (a * b).abs()).sum();
    assert!(
        (got - want).abs() <= 1e-4 * mag + 1e-5,
        "{name}: kernel {got} vs reference {want} (sum|w*x| {mag})"
    );
}

type FloatDot = fn(&[u8], &[f32]) -> f32;
type Q8Dot = fn(&[u8], &Q8ActsSlice<'_>) -> f32;

#[test]
fn float_dot_kernels_match_the_dequantizer() {
    let x = activations();
    let cases: [(GgmlType, usize, usize, &[usize], FloatDot); 7] = [
        (GgmlType::Q5_0, 22, 32, &[0], dot_q5_0),
        (GgmlType::Q8_0, 34, 32, &[0], dot_q8_0),
        (GgmlType::Q2K, 84, 256, &[80, 82], dot_q2_k),
        (GgmlType::Q3K, 110, 256, &[108], dot_q3_k),
        (GgmlType::Q4K, 144, 256, &[0, 2], dot_q4_k),
        (GgmlType::Q5K, 176, 256, &[0, 2], dot_q5_k),
        (GgmlType::Q6K, 210, 256, &[208], dot_q6_k),
    ];
    for (ty, bb, be, scales, dot) in cases {
        let r = row(ty, bb, be, scales);
        let w = dequant::dequant(ty, &r, N_IN).unwrap();
        check(&format!("{ty:?}"), dot(&r, &x), &w, &x);
    }
    let values: Vec<f32> = (0..N_IN).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let f16: Vec<u8> = values
        .iter()
        .flat_map(|v| allpaka_dtype::f32_to_f16(*v).to_le_bytes())
        .collect();
    let w = dequant::dequant(GgmlType::F16, &f16, N_IN).unwrap();
    check("F16", dot_f16(&f16, &x), &w, &x);
    let bf16: Vec<u8> = values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let w = dequant::dequant(GgmlType::BF16, &bf16, N_IN).unwrap();
    check("BF16", dot_bf16(&bf16, &x), &w, &x);
}

#[test]
fn q8_activation_kernels_match_the_dequantizer_on_quantized_activations() {
    let x = activations();
    let acts = Q8Acts::from_f32(&x);
    let slice = acts.row(0, N_IN);
    let seen: Vec<f32> = (0..N_IN)
        .map(|i| acts.d[i / 32] * acts.q[i] as f32)
        .collect();
    let cases: [(GgmlType, usize, usize, &[usize], Q8Dot); 7] = [
        (GgmlType::Q5_0, 22, 32, &[0], dot_q5_0_i8),
        (GgmlType::Q8_0, 34, 32, &[0], dot_q8_0_i8),
        (GgmlType::Q2K, 84, 256, &[80, 82], dot_q2_k_i8),
        (GgmlType::Q3K, 110, 256, &[108], dot_q3_k_i8),
        (GgmlType::Q4K, 144, 256, &[0, 2], dot_q4_k_i8),
        (GgmlType::Q5K, 176, 256, &[0, 2], dot_q5_k_i8),
        (GgmlType::Q6K, 210, 256, &[208], dot_q6_k_i8),
    ];
    for (ty, bb, be, scales, dot) in cases {
        let r = row(ty, bb, be, scales);
        let w = dequant::dequant(ty, &r, N_IN).unwrap();
        // The K-quant min terms use the exact f32 block sums, not q8(x).
        let got = dot(&r, &slice);
        let want_q8: f32 = w.iter().zip(&seen).map(|(a, b)| a * b).sum();
        let want_f32: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let mag: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
        let tol = 2e-3 * mag + 1e-4;
        assert!(
            (got - want_q8).abs() <= tol || (got - want_f32).abs() <= tol,
            "{ty:?}: kernel {got} vs q8 reference {want_q8} / f32 {want_f32} (sum|w*x| {mag})"
        );
    }
}

#[test]
fn integer_dot_is_exact() {
    let a: Vec<i8> = (0..64).map(|i| (i * 37 % 255) as u8 as i8).collect();
    let b: Vec<i8> = (0..64).map(|i| (i * 11 % 255) as u8 as i8).collect();
    let want: i32 = a.iter().zip(&b).map(|(x, y)| *x as i32 * *y as i32).sum();
    assert_eq!(dot_i8(&a, &b), want);
}
