//! Regression: two small-batch K-quant matvecs of the same shape must not
//! share the cached Q8 activations.
//!
//! `run_matvec` uploads every `x` to the same `x_arena` address, and the Q8
//! cache is keyed by (pointer, n_in, m). Qwen2.5-7B has q_dim == hidden, so the
//! attention output projection reused the quantised FFN input of the previous
//! layer and the model produced garbage on CUDA (CPU was fine).
//!
//! Skips when `ALLPAKA_NO_GPU` is set or no CUDA device is available.

#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use allpaka_backend::gpu;
use allpaka_gguf::GgmlType;

const PAGE: usize = 16384;

#[test]
fn same_shape_matvecs_requantize_x() {
    if std::env::var_os("ALLPAKA_NO_GPU").is_some() {
        eprintln!("SKIP: ALLPAKA_NO_GPU set");
        return;
    }
    let n_in = 512usize;
    let n_out = 16usize;
    // Below MM_MIN_M (32): the Q8 matvec path, not the GEMM path.
    let m = 5usize;
    let row_bytes = n_in / 256 * 144;

    let len = PAGE * 2;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    // SAFETY: `ptr` is from `alloc_zeroed`, so every byte is initialised. `ptr` is non-null
    // (asserted), owns `len` bytes at `PAGE` alignment, and is never freed, so the slice is unique
    // and may live for the whole test.
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut state = 0x5EED_00F0_0C8C_4A11u64;
    for b in region.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    // Tame the f16 super-block scales (d, dmin) so the sums stay finite.
    let small = 0x211Fu16.to_le_bytes(); // ~0.01
    for r in 0..n_out * (n_in / 256) {
        let at = r * 144;
        region[at..at + 2].copy_from_slice(&small);
        region[at + 2..at + 4].copy_from_slice(&small);
    }
    if !gpu::attach(region) {
        eprintln!("SKIP: no CUDA device / attach failed");
        return;
    }

    let w = &region[..n_out * row_bytes];
    let x1: Vec<f32> = (0..m * n_in)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.07)
        .collect();
    // Same shape, different values; negation quantises to exactly -q(x1).
    let x2: Vec<f32> = x1.iter().map(|v| -v).collect();

    let run = |x: &[f32]| {
        gpu::matvec_batch(&[gpu::MatvecReq {
            ty: GgmlType::Q4K,
            w,
            n_in,
            n_out,
            x,
            m,
        }])
        .expect("cuda q4k matvec")
        .pop()
        .unwrap()
    };
    let y1 = run(&x1);
    let y2 = run(&x2);
    assert!(y1.iter().any(|v| v.abs() > 1e-3), "degenerate output {y1:?}");
    for (i, (a, b)) in y1.iter().zip(&y2).enumerate() {
        assert!(
            (a + b).abs() < 1e-3 + 1e-3 * a.abs(),
            "row {} col {}: y(x)={a} but y(-x)={b} (stale Q8 activations?)",
            i / n_out,
            i % n_out
        );
    }
}
