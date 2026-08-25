//! GPU kernels against the CPU reference on synthetic weights.
//!
//! `gpu::attach` normally wraps a model's mmap; here it wraps a page-aligned
//! allocation filled with pseudo-random quantised blocks, which lets the
//! Metal kernels run on data the CPU path can independently recompute. The
//! same matrix is opened twice: once inside the attached region (GPU path)
//! and once as a heap copy (the GPU declines foreign bytes, so the fused CPU
//! kernels produce the reference).

#![cfg(target_os = "macos")]

use allpaka_backend::{gpu, QuantMat};
use allpaka_gguf::GgmlType;

const PAGE: usize = 16384;

/// Leaked page-aligned region the tests attach to the GPU once.
fn attached_region() -> &'static [u8] {
    use std::sync::OnceLock;
    static REGION: OnceLock<&'static [u8]> = OnceLock::new();
    REGION.get_or_init(|| {
        let len = 64 << 20;
        let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        // Deterministic pseudo-random bytes: valid quant blocks for every
        // format under test (any bit pattern decodes; f16 scales are tamed
        // below per matrix).
        let mut state = 0x243F_6A88_85A3_08D3u64;
        for b in region.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (state >> 56) as u8;
        }
        assert!(gpu::attach(region), "no Metal device; parity tests need one");
        region
    })
}

/// Overwrite every f16 scale field in the region's blocks with small sane
/// values, so dot products stay finite (random f16 bits include inf/NaN).
fn tame_scales(region: &mut [u8], ty: GgmlType, off: usize, rows: usize, n_in: usize) {
    // F32 has no scales; the random bytes themselves would be inf/NaN soup,
    // so the whole tensor is rewritten with small finite values instead.
    if ty == GgmlType::F32 {
        for (i, c) in region[off..off + rows * n_in * 4].chunks_exact_mut(4).enumerate() {
            c.copy_from_slice(&(((i % 61) as f32 - 30.0) * 0.03).to_le_bytes());
        }
        return;
    }
    let (bb, scale_offs): (usize, &[usize]) = match ty {
        GgmlType::Q8_0 => (34, &[0]),
        GgmlType::Q2K => (84, &[80, 82]),
        GgmlType::Q3K => (110, &[108]),
        GgmlType::Q4K => (144, &[0, 2]),
        GgmlType::Q5K => (176, &[0, 2]),
        GgmlType::Q6K => (210, &[208]),
        _ => unreachable!(),
    };
    let be = match ty {
        GgmlType::F32 => 1,
        GgmlType::Q8_0 => 32,
        _ => 256,
    };
    let row_bytes = n_in / be * bb;
    // 0.01 in f16.
    let half = 0x211Fu16.to_le_bytes();
    for r in 0..rows {
        for blk in 0..n_in / be {
            for &so in scale_offs {
                let at = off + r * row_bytes + blk * bb + so;
                region[at] = half[0];
                region[at + 1] = half[1];
            }
        }
    }
}

fn check_parity(ty: GgmlType, n_out: usize, n_in: usize, m: usize, off: usize) {
    let region = attached_region();
    // Redo the taming through a raw pointer: the region is logically ours.
    let region_mut =
        unsafe { std::slice::from_raw_parts_mut(region.as_ptr() as *mut u8, region.len()) };
    tame_scales(region_mut, ty, off, n_out, n_in);

    let be = match ty {
        GgmlType::F32 => 1usize,
        GgmlType::Q8_0 => 32,
        _ => 256,
    };
    let bb = match ty {
        GgmlType::F32 => 4,
        GgmlType::Q8_0 => 34,
        GgmlType::Q2K => 84,
        GgmlType::Q3K => 110,
        GgmlType::Q4K => 144,
        GgmlType::Q5K => 176,
        GgmlType::Q6K => 210,
        _ => unreachable!(),
    };
    let bytes = n_out * n_in / be * bb;
    let inside = &region[off..off + bytes];
    let copy = inside.to_vec();

    let on_gpu = QuantMat::new(inside, ty, n_out, n_in).unwrap();
    let on_cpu = QuantMat::new(&copy, ty, n_out, n_in).unwrap();

    let x: Vec<f32> = (0..m * n_in).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
    let got = on_gpu.matmul(&x, m).unwrap();
    // The reference is dequantise-then-dot in f32: QuantMat::matmul on the
    // CPU now quantises activations to Q8 (an approximation of its own), so
    // it cannot arbitrate the GPU's exact f32 arithmetic.
    //
    // Batches of 16+ run the tile-matmul kernel, which stages both weights
    // and activations as HALF (llama.cpp's mul_mm precision). The reference
    // applies the same roundings, so the tolerance stays tight and still
    // catches indexing bugs - a wrong element is orders of magnitude out,
    // half rounding a fraction of a percent.
    let round = |v: f32| -> f32 {
        use allpaka_backend::ops::f16;
        if m >= 16 { f16::to_f32(f16::scalar_from_f32(v)) } else { v }
    };
    let x_ref: Vec<f32> = x.iter().map(|&v| round(v)).collect();
    let mut want = vec![0f32; m * n_out];
    for j in 0..n_out {
        let row: Vec<f32> = on_cpu.row(j).unwrap().iter().map(|&v| round(v)).collect();
        for i in 0..m {
            let xi = &x_ref[i * n_in..(i + 1) * n_in];
            want[i * n_out + j] = row.iter().zip(xi).map(|(a, b)| a * b).sum();
        }
    }

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-2 + 2e-3 * w.abs(),
            "{ty:?} m={m} element {i}: gpu {g} vs cpu {w}"
        );
    }
}

/// The fused gate/up/swiglu/down path against dequant-and-compute in f32.
#[test]
fn fused_ffn_matches_cpu_reference() {
    let region = attached_region();
    let region_mut =
        unsafe { std::slice::from_raw_parts_mut(region.as_ptr() as *mut u8, region.len()) };
    let (hidden, ffn) = (512usize, 256usize);
    let (gate_off, up_off, down_off) = (12 << 20, 16 << 20, 20 << 20);
    tame_scales(region_mut, GgmlType::Q4K, gate_off, ffn, hidden);
    tame_scales(region_mut, GgmlType::Q4K, up_off, ffn, hidden);
    tame_scales(region_mut, GgmlType::Q6K, down_off, hidden, ffn);

    let gate_bytes = ffn * hidden / 256 * 144;
    let down_bytes = hidden * ffn / 256 * 210;
    let gate = QuantMat::new(&region[gate_off..gate_off + gate_bytes], GgmlType::Q4K, ffn, hidden)
        .unwrap();
    let up = QuantMat::new(&region[up_off..up_off + gate_bytes], GgmlType::Q4K, ffn, hidden)
        .unwrap();
    let down = QuantMat::new(&region[down_off..down_off + down_bytes], GgmlType::Q6K, hidden, ffn)
        .unwrap();

    for m in [1usize, 5] {
        let x: Vec<f32> = (0..m * hidden).map(|i| ((i % 17) as f32 - 8.0) * 0.06).collect();
        let got = QuantMat::ffn_many(&[(&gate, &up, &down, x.as_slice())])
            .expect("GPU must take the fused FFN")
            .pop()
            .unwrap();

        // f32 reference from dequantised rows.
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let mut want = vec![0f32; m * hidden];
        let gate_rows: Vec<Vec<f32>> = (0..ffn).map(|j| gate.row(j).unwrap()).collect();
        let up_rows: Vec<Vec<f32>> = (0..ffn).map(|j| up.row(j).unwrap()).collect();
        let down_rows: Vec<Vec<f32>> = (0..hidden).map(|j| down.row(j).unwrap()).collect();
        for i in 0..m {
            let xi = &x[i * hidden..(i + 1) * hidden];
            let act: Vec<f32> = (0..ffn)
                .map(|j| {
                    let g: f32 = gate_rows[j].iter().zip(xi).map(|(a, b)| a * b).sum();
                    let u: f32 = up_rows[j].iter().zip(xi).map(|(a, b)| a * b).sum();
                    silu(g) * u
                })
                .collect();
            for (j, w) in want[i * hidden..(i + 1) * hidden].iter_mut().enumerate() {
                *w = down_rows[j].iter().zip(&act).map(|(a, b)| a * b).sum();
            }
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 2e-2 + 1e-2 * w.abs(),
                "fused ffn m={m} element {i}: gpu {g} vs cpu {w}"
            );
        }
    }
}

#[test]
fn q8_0_matches_cpu_reference() {
    check_parity(GgmlType::Q8_0, 64, 1024, 1, 0);
    check_parity(GgmlType::Q8_0, 64, 1024, 20, 1 << 20);
}

#[test]
fn q2_k_matches_cpu_reference() {
    check_parity(GgmlType::Q2K, 48, 1024, 1, 2 << 20);
    check_parity(GgmlType::Q2K, 48, 1024, 20, 3 << 20);
    check_parity(GgmlType::Q2K, 64, 1024, 64, 37 << 20);
}

/// The F32 tile-matmul: the MoE router's format. Only the mm path exists
/// for it (decode-sized F32 matvecs stay on the CPU by design), so only
/// batched shapes are checked.
#[test]
fn f32_router_matmul_matches_cpu_reference() {
    check_parity(GgmlType::F32, 128, 2048, 64, 39 << 20);
    check_parity(GgmlType::F32, 128, 2048, 33, 41 << 20);
    // 7 rows split as 4+2+1: the odd tiles, all in one batch.
    check_parity(GgmlType::Q2K, 48, 1024, 7, 31 << 20);
}

#[test]
fn q3_k_matches_cpu_reference() {
    check_parity(GgmlType::Q3K, 48, 1024, 1, 24 << 20);
    check_parity(GgmlType::Q3K, 48, 1024, 20, 26 << 20);
}

#[test]
fn q4_k_matches_cpu_reference() {
    check_parity(GgmlType::Q4K, 48, 1024, 1, 4 << 20);
    check_parity(GgmlType::Q4K, 48, 1024, 20, 5 << 20);
    // 7 rows split as 4+2+1: the odd tiles, all in one batch.
    check_parity(GgmlType::Q4K, 48, 1024, 7, 32 << 20);
    // Two full column tiles of the tile-matmul kernel, and an edge tile in
    // both dimensions at once (48 rows x 33 columns over 32-wide tiles).
    check_parity(GgmlType::Q4K, 64, 1024, 64, 33 << 20);
    check_parity(GgmlType::Q4K, 48, 1024, 33, 35 << 20);
    // 768 wide is 3 blocks per row, the shape of a 30B expert's down
    // projection: the lane group is 4 wide and one of its lanes gets no
    // block at all, so the shuffle reduction has to fold a zero correctly.
    check_parity(GgmlType::Q4K, 48, 768, 1, 33 << 20);
    check_parity(GgmlType::Q4K, 48, 768, 8, 34 << 20);
}

#[test]
fn q5_k_matches_cpu_reference() {
    check_parity(GgmlType::Q5K, 48, 1024, 1, 28 << 20);
    check_parity(GgmlType::Q5K, 48, 1024, 20, 30 << 20);
}

#[test]
fn q6_k_matches_cpu_reference() {
    check_parity(GgmlType::Q6K, 48, 1024, 1, 6 << 20);
    check_parity(GgmlType::Q6K, 48, 1024, 20, 8 << 20);
}
