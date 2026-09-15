//! CUDA matvec vs CPU dequant+dot on a tiny attached weight region.
//!
//! Skips when `ALLPAKA_NO_GPU` is set or no CUDA device is available.

#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use allpaka_backend::gpu;
use allpaka_gguf::GgmlType;

const PAGE: usize = 16384;

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn dequant_q8_0_row(row: &[u8], n_in: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_in];
    let nb = n_in / 32;
    for b in 0..nb {
        let blk = &row[b * 34..(b + 1) * 34];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..32 {
            out[b * 32 + j] = d * (blk[2 + j] as i8 as f32);
        }
    }
    out
}

fn tame_q8_0(region: &mut [u8], off: usize, rows: usize, n_in: usize) {
    let bb = 34usize;
    let be = 32usize;
    let half = 0x211Fu16.to_le_bytes(); // ~0.01 f16
    let row_bytes = n_in / be * bb;
    for r in 0..rows {
        for blk in 0..n_in / be {
            let at = off + r * row_bytes + blk * bb;
            region[at] = half[0];
            region[at + 1] = half[1];
        }
    }
}

fn tame_f32(region: &mut [u8], off: usize, rows: usize, n_in: usize) {
    for (i, c) in region[off..off + rows * n_in * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        c.copy_from_slice(&(((i % 61) as f32 - 30.0) * 0.03).to_le_bytes());
    }
}

#[test]
fn attach_and_matvec_parity() {
    if std::env::var_os("ALLPAKA_NO_GPU").is_some() {
        eprintln!("SKIP: ALLPAKA_NO_GPU set");
        return;
    }

    let (q8_out, q8_in, q8_off) = (8usize, 64usize, 0usize);
    let (f32_out, f32_in, f32_off) = (4usize, 32usize, PAGE);
    let len = PAGE * 4;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null());
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut state = 0xC0FF_EE00_D15C_AFE1u64;
    for b in region.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    tame_q8_0(region, q8_off, q8_out, q8_in);
    tame_f32(region, f32_off, f32_out, f32_in);

    if !gpu::attach(region) {
        eprintln!("SKIP: no CUDA device / attach failed");
        return;
    }
    assert!(gpu::is_attached());

    // Q8_0 matvec vs CPU dequant+dot
    {
        let row_bytes = q8_in / 32 * 34;
        let w = &region[q8_off..q8_off + q8_out * row_bytes];
        let x: Vec<f32> = (0..q8_in)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let got = gpu::matvec(GgmlType::Q8_0, w, q8_in, q8_out, &x).expect("cuda q8_0 matvec");
        assert_eq!(got.len(), q8_out);
        for j in 0..q8_out {
            let row = dequant_q8_0_row(&w[j * row_bytes..(j + 1) * row_bytes], q8_in);
            let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(
                (got[j] - want).abs() < 1e-3 + 1e-3 * want.abs(),
                "Q8_0 row {j}: gpu {} vs cpu {}",
                got[j],
                want
            );
        }
    }

    // F32 matvec vs CPU dot
    {
        let w = &region[f32_off..f32_off + f32_out * f32_in * 4];
        let x: Vec<f32> = (0..f32_in).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        let got = gpu::matvec(GgmlType::F32, w, f32_in, f32_out, &x).expect("cuda f32 matvec");
        assert_eq!(got.len(), f32_out);
        for j in 0..f32_out {
            let mut row = vec![0f32; f32_in];
            for i in 0..f32_in {
                let o = (j * f32_in + i) * 4;
                row[i] = f32::from_le_bytes(w[o..o + 4].try_into().unwrap());
            }
            let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(
                (got[j] - want).abs() < 1e-4 + 1e-5 * want.abs(),
                "F32 row {j}: gpu {} vs cpu {}",
                got[j],
                want
            );
        }
    }
}

fn tame_f16_scale(region: &mut [u8], off: usize, row_bytes: usize, rows: usize, d_off: usize) {
    let half = 0x211Fu16.to_le_bytes(); // ~0.01 f16
    for r in 0..rows {
        let at = off + r * row_bytes + d_off;
        region[at] = half[0];
        region[at + 1] = half[1];
    }
}

#[test]
fn gemm_q4k_q6k_parity() {
    if std::env::var_os("ALLPAKA_NO_GPU").is_some() {
        eprintln!("SKIP: ALLPAKA_NO_GPU set");
        return;
    }

    let n_in = 256usize;
    let n_out = 8usize;
    let m = 32usize;
    let q4_off = PAGE * 2;
    let q6_off = PAGE * 3;
    let q4_rb = n_in / 256 * 144;
    let q6_rb = n_in / 256 * 210;
    let len = PAGE * 5;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null());
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut state = 0xA5A5_5A5A_C0DE_BEEFu64;
    for b in region.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    tame_f16_scale(region, q4_off, q4_rb, n_out, 0);
    tame_f16_scale(region, q4_off, q4_rb, n_out, 2);
    tame_f16_scale(region, q6_off, q6_rb, n_out, 208);

    if !gpu::attach(region) {
        eprintln!("SKIP: no CUDA device / attach failed");
        return;
    }

    let x: Vec<f32> = (0..m * n_in)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
        .collect();

    for (ty, off, rb) in [
        (GgmlType::Q4K, q4_off, q4_rb),
        (GgmlType::Q6K, q6_off, q6_rb),
    ] {
        let w = &region[off..off + n_out * rb];
        let x1: Vec<f32> = (0..n_in)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        let got1 = gpu::matvec(ty, w, n_in, n_out, &x1).expect("cuda matvec m=1");
        let w_f32 = allpaka_gguf::dequant::dequant(ty, w, n_out * n_in).unwrap();
        for j in 0..n_out {
            let mut want = 0f32;
            for k in 0..n_in {
                want += x1[k] * w_f32[j * n_in + k];
            }
            let g = got1[j];
            let tol = 2e-2 + 2e-2 * want.abs();
            assert!(
                (g - want).abs() < tol,
                "{ty:?} matvec c{j}: gpu {g} vs cpu {want}"
            );
        }
        let got = gpu::matvec_batch(&[gpu::MatvecReq {
            ty,
            w,
            n_in,
            n_out,
            x: &x,
            m,
        }])
        .expect("cuda gemm")
        .pop()
        .unwrap();
        assert_eq!(got.len(), m * n_out);
        let w_f32 = allpaka_gguf::dequant::dequant(ty, w, n_out * n_in).unwrap();
        for row in 0..m {
            for j in 0..n_out {
                let mut want = 0f32;
                for k in 0..n_in {
                    want += x[row * n_in + k] * w_f32[j * n_in + k];
                }
                let g = got[row * n_out + j];
                let tol = 2e-2 + 2e-2 * want.abs();
                assert!(
                    (g - want).abs() < tol,
                    "{ty:?} r{row} c{j}: gpu {g} vs cpu {want}"
                );
            }
        }
    }
}
