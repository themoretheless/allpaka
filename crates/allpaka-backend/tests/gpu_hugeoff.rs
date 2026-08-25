//! The 235B regression: tensors that live past the 4 GiB mark of the mmap.
//!
//! Two failure modes hide there: 32-bit truncation anywhere in the offset
//! arithmetic, and (on files past the device's maxBufferLength) a nil
//! bytesNoCopy buffer. This test allocates a 5 GiB region, attaches it, and
//! checks kernel parity for a tensor placed beyond 2^32. It also prints the
//! device's maxBufferLength, which decides whether an 88 GiB model needs the
//! chunked windows at all.
//!
//! Ignored by default: it allocates 5 GiB. Run with
//! `cargo test -p allpaka-backend --test gpu_hugeoff -- --ignored`

#![cfg(target_os = "macos")]

use allpaka_backend::{gpu, QuantMat};
use allpaka_gguf::GgmlType;

#[test]
#[ignore = "allocates 5 GiB; run explicitly"]
fn kernels_are_correct_past_the_4_gib_offset() {
    let device = metal::Device::system_default().expect("Metal device");
    println!(
        "maxBufferLength = {:.1} GiB",
        device.max_buffer_length() as f64 / (1u64 << 30) as f64
    );

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

    // One matrix per format, all past 2^32.
    let cases = [
        (GgmlType::Q8_0, 34usize, 32usize, 4_296_000_000usize),
        (GgmlType::Q2K, 84, 256, 4_496_000_000),
        (GgmlType::Q4K, 144, 256, 4_696_000_000),
        (GgmlType::Q6K, 210, 256, 4_896_000_000),
    ];
    let (n_out, n_in) = (32usize, 1024usize);

    for (ty, bb, be, off) in cases {
        let bytes = n_out * n_in / be * bb;
        // Pseudo-random quant bytes, then small sane f16 scales.
        let mut state = off as u64 | 1;
        for b in region[off..off + bytes].iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (state >> 56) as u8;
        }
        let half = 0x211Fu16.to_le_bytes(); // 0.01
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
        // f32 dequantise-then-dot reference; the CPU matmul itself quantises
        // activations and cannot arbitrate exact arithmetic.
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
