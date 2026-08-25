//! Multi-window attach, exercised without an 80 GiB file: the window size is
//! capped via ALLPAKA_GPU_WINDOW_GIB, so a 5 GiB region maps through several
//! overlapping bytesNoCopy windows exactly like the 235B mapping does on real
//! hardware. Tensors are placed in late windows and inside overlaps; every
//! one must match the f32 CPU reference.
//!
//! Ignored by default: it allocates 5 GiB. Run with
//! `cargo test -p allpaka-backend --test gpu_windows -- --ignored`

#![cfg(target_os = "macos")]

use allpaka_backend::{gpu, QuantMat};
use allpaka_gguf::GgmlType;

#[test]
#[ignore = "allocates 5 GiB; run explicitly"]
fn late_windows_read_the_same_bytes_as_the_cpu() {
    // Must be set before attach; the OnceLock reads it exactly once.
    std::env::set_var("ALLPAKA_GPU_WINDOW_GIB", "2");

    const PAGE: usize = 16384;
    let len: usize = 5 << 30;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    if !gpu::attach(region) {
        eprintln!("SKIP: no Metal device");
        return;
    }

    let (n_out, n_in) = (32usize, 1024usize);
    // Offsets chosen against 2 GiB windows stepping by 1 GiB: inside window
    // overlaps, straddling window starts, and in the final short window.
    let cases = [
        (GgmlType::Q4K, 144usize, 256usize, 500_000_000usize),
        (GgmlType::Q4K, 144, 256, 2_500_000_000),
        (GgmlType::Q8_0, 34, 32, 3_100_000_000),
        (GgmlType::Q2K, 84, 256, 4_500_000_000),
        (GgmlType::Q6K, 210, 256, 4_800_000_000),
    ];

    for (ty, bb, be, off) in cases {
        let bytes = n_out * n_in / be * bb;
        let mut state = off as u64 | 1;
        for b in region[off..off + bytes].iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (state >> 56) as u8;
        }
        let half = 0x211Fu16.to_le_bytes(); // 0.01 in f16
        let scale_offs: &[usize] = match ty {
            GgmlType::Q8_0 => &[0],
            GgmlType::Q2K => &[80, 82],
            GgmlType::Q4K => &[0, 2],
            GgmlType::Q6K => &[208],
            _ => unreachable!(),
        };
        let row_bytes = n_in / be * bb;
        for r in 0..n_out {
            for blk in 0..n_in / be {
                for &so in scale_offs {
                    let at = off + r * row_bytes + blk * bb + so;
                    region[at] = half[0];
                    region[at + 1] = half[1];
                }
            }
        }

        let inside = &region[off..off + bytes];
        let copy = inside.to_vec();
        let on_gpu = QuantMat::new(inside, ty, n_out, n_in).unwrap();
        let on_cpu = QuantMat::new(&copy, ty, n_out, n_in).unwrap();
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 19) as f32 - 9.0) * 0.07).collect();
        let got = on_gpu.matmul(&x, 1).unwrap();
        let mut want = vec![0f32; n_out];
        for (j, w) in want.iter_mut().enumerate() {
            let row = on_cpu.row(j).unwrap();
            *w = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                g.is_finite() && (g - w).abs() < 1e-2 + 2e-3 * w.abs(),
                "{ty:?} at offset {off}: element {i} gpu {g} vs cpu {w}"
            );
        }
        println!("{ty:?} at {:.2} GiB: parity ok", off as f64 / (1u64 << 30) as f64);
    }
}
