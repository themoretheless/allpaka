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
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
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
        let x: Vec<f32> = (0..q8_in).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
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
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
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
        let x1: Vec<f32> = (0..n_in).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let got1 = gpu::matvec(ty, w, n_in, n_out, &x1).expect("cuda matvec m=1");
        let w_f32 = allpaka_gguf::dequant::dequant(ty, w, n_out * n_in).unwrap();
        let q8_m1 = matches!(ty, GgmlType::Q4K | GgmlType::Q6K)
            && std::env::var("ALLPAKA_Q8")
                .map_or(true, |v| v != "0" && !v.eq_ignore_ascii_case("false"));
        let recon1: Vec<f32> = if q8_m1 {
            let mut q = vec![0i8; n_in];
            let mut qd = vec![0f32; n_in / 32];
            for blk in 0..n_in / 32 {
                let base = blk * 32;
                let mut amax = 0f32;
                for j in 0..32 {
                    amax = amax.max(x1[base + j].abs());
                }
                let scale = if amax < 1e-8 * 127.0 {
                    1.0
                } else {
                    amax / 127.0
                };
                qd[blk] = scale;
                for j in 0..32 {
                    let qi = (x1[base + j] / scale).round().clamp(-127.0, 127.0) as i32;
                    q[base + j] = qi as i8;
                }
            }
            (0..n_in).map(|i| q[i] as f32 * qd[i / 32]).collect()
        } else {
            x1.clone()
        };
        for j in 0..n_out {
            let mut want = 0f32;
            for k in 0..n_in {
                want += recon1[k] * w_f32[j * n_in + k];
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
        // m >= MM_MIN_M goes through the GEMM path: f16 dequant + cuBLASLt by
        // default, or ggml MMQ (Q8 activations) when that library is loaded.
        // Compare against float*W with a tolerance that also covers Q8
        // activation rounding, which grows with sum(|x*w|), not with |dot|.
        for row in 0..m {
            let xr = &x[row * n_in..(row + 1) * n_in];
            for j in 0..n_out {
                let wr = &w_f32[j * n_in..(j + 1) * n_in];
                let want: f32 = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
                let mag: f32 = xr.iter().zip(wr).map(|(a, b)| (a * b).abs()).sum();
                let g = got[row * n_out + j];
                let tol = 2e-2 + 2e-2 * want.abs() + 2e-3 * mag;
                assert!(
                    (g - want).abs() < tol,
                    "{ty:?} r{row} c{j}: gpu {g} vs cpu {want} (sum|x*w| {mag})"
                );
            }
        }
    }
}

/// Every weight format with a `dq_fmt` code runs the batched GEMM path
/// (`dequant_rows_f16` + cuBLASLt) and must agree with the CPU dequantizer.
/// A wrong code would dequantize with another format's layout and fail here.
#[test]
fn gemm_every_dq_fmt_matches_cpu_dequant() {
    if std::env::var_os("ALLPAKA_NO_GPU").is_some() {
        eprintln!("SKIP: ALLPAKA_NO_GPU set");
        return;
    }
    let (n_in, n_out, m) = (256usize, 8usize, 32usize);
    // (type, block bytes, offsets of f16 scales inside a block)
    let quant: [(GgmlType, usize, &[usize]); 7] = [
        (GgmlType::Q4K, 144, &[0, 2]),
        (GgmlType::Q5K, 176, &[0, 2]),
        (GgmlType::Q6K, 210, &[208]),
        (GgmlType::Q8_0, 34, &[0]),
        (GgmlType::Q2K, 84, &[80, 82]),
        (GgmlType::Q3K, 110, &[108]),
        (GgmlType::Q5_0, 22, &[0]),
    ];
    let slot = PAGE * 2;
    let len = slot * (quant.len() + 3);
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null());
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for b in region.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    let mut cases: Vec<(GgmlType, usize, usize)> = Vec::new();
    for (i, &(ty, bb, scales)) in quant.iter().enumerate() {
        let off = i * slot;
        let be = if bb == 34 || bb == 22 { 32 } else { 256 };
        let rb = n_in / be * bb;
        let half = 0x211Fu16.to_le_bytes(); // ~0.01
        for blk in 0..n_out * n_in / be {
            for &s in scales {
                let at = off + blk * bb + s;
                region[at] = half[0];
                region[at + 1] = half[1];
            }
        }
        cases.push((ty, off, rb));
    }
    let value = |i: usize| ((i % 29) as f32 - 14.0) * 0.02;
    let base = quant.len() * slot;
    for (i, c) in region[base..base + n_out * n_in * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        c.copy_from_slice(&value(i).to_le_bytes());
    }
    cases.push((GgmlType::F32, base, n_in * 4));
    let base = base + slot;
    for (i, c) in region[base..base + n_out * n_in * 2]
        .chunks_exact_mut(2)
        .enumerate()
    {
        c.copy_from_slice(&half::f16::from_f32(value(i)).to_le_bytes());
    }
    cases.push((GgmlType::F16, base, n_in * 2));
    let base = base + slot;
    for (i, c) in region[base..base + n_out * n_in * 2]
        .chunks_exact_mut(2)
        .enumerate()
    {
        c.copy_from_slice(&((value(i).to_bits() >> 16) as u16).to_le_bytes());
    }
    cases.push((GgmlType::BF16, base, n_in * 2));

    if !gpu::attach(region) {
        eprintln!("SKIP: no CUDA device / attach failed");
        return;
    }
    let x: Vec<f32> = (0..m * n_in)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.04)
        .collect();
    for (ty, off, rb) in cases {
        let w = &region[off..off + n_out * rb];
        let got = gpu::matvec_batch(&[gpu::MatvecReq {
            ty,
            w,
            n_in,
            n_out,
            x: &x,
            m,
        }])
        .unwrap_or_else(|| panic!("cuda gemm {ty:?}"))
        .pop()
        .unwrap();
        let w_f32 = allpaka_gguf::dequant::dequant(ty, w, n_out * n_in).unwrap();
        for row in 0..m {
            let xr = &x[row * n_in..(row + 1) * n_in];
            for j in 0..n_out {
                let wr = &w_f32[j * n_in..(j + 1) * n_in];
                let want: f32 = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
                let mag: f32 = xr.iter().zip(wr).map(|(a, b)| (a * b).abs()).sum();
                let g = got[row * n_out + j];
                assert!(
                    (g - want).abs() < 2e-2 + 2e-2 * want.abs() + 2e-3 * mag,
                    "{ty:?} r{row} c{j}: gpu {g} vs cpu {want}"
                );
            }
        }
    }
}
