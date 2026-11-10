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
/// None when the host has no Metal device (GitHub-hosted runners): the GPU
/// tests then report SKIP instead of failing.
fn attached_region() -> Option<&'static [u8]> {
    use std::sync::OnceLock;
    static REGION: OnceLock<Option<&'static [u8]>> = OnceLock::new();
    REGION
        .get_or_init(|| {
            let len = 72 << 20;
            let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!ptr.is_null());
            let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            // Deterministic pseudo-random bytes: valid quant blocks for every
            // format under test (any bit pattern decodes; f16 scales are tamed
            // below per matrix).
            let mut state = 0x243F_6A88_85A3_08D3u64;
            for b in region.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (state >> 56) as u8;
            }
            if !gpu::attach(region) {
                eprintln!("SKIP: no Metal device; parity tests need one");
                return None;
            }
            Some(&*region)
        })
        .as_ref()
        .copied()
}

/// Overwrite every f16 scale field in the region's blocks with small sane
/// values, so dot products stay finite (random f16 bits include inf/NaN).
fn tame_scales(region: &mut [u8], ty: GgmlType, off: usize, rows: usize, n_in: usize) {
    // F32 has no scales; the random bytes themselves would be inf/NaN soup,
    // so the whole tensor is rewritten with small finite values instead.
    if ty == GgmlType::F32 {
        for (i, c) in region[off..off + rows * n_in * 4]
            .chunks_exact_mut(4)
            .enumerate()
        {
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
    let Some(region) = attached_region() else {
        return;
    };
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

    let x: Vec<f32> = (0..m * n_in)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
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
        if m >= 16 {
            f16::to_f32(f16::scalar_from_f32(v))
        } else {
            v
        }
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
    let Some(region) = attached_region() else {
        return;
    };
    let region_mut =
        unsafe { std::slice::from_raw_parts_mut(region.as_ptr() as *mut u8, region.len()) };
    let (hidden, ffn) = (512usize, 256usize);
    let (gate_off, up_off, down_off) = (12 << 20, 16 << 20, 20 << 20);
    tame_scales(region_mut, GgmlType::Q4K, gate_off, ffn, hidden);
    tame_scales(region_mut, GgmlType::Q4K, up_off, ffn, hidden);
    tame_scales(region_mut, GgmlType::Q6K, down_off, hidden, ffn);

    let gate_bytes = ffn * hidden / 256 * 144;
    let down_bytes = hidden * ffn / 256 * 210;
    let gate = QuantMat::new(
        &region[gate_off..gate_off + gate_bytes],
        GgmlType::Q4K,
        ffn,
        hidden,
    )
    .unwrap();
    let up = QuantMat::new(
        &region[up_off..up_off + gate_bytes],
        GgmlType::Q4K,
        ffn,
        hidden,
    )
    .unwrap();
    let down = QuantMat::new(
        &region[down_off..down_off + down_bytes],
        GgmlType::Q6K,
        hidden,
        ffn,
    )
    .unwrap();

    for m in [1usize, 5] {
        let x: Vec<f32> = (0..m * hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.06)
            .collect();
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

/// End-to-end GPU vs CPU parity for `RopeStyle::Norm` (`rope_norm: true`)
/// across `QuantMat::prefill_attn_block`, `QuantMat::attn_block`, and
/// `gpu::decode_token` (`m = 1` and `m = 3` verify).
#[test]
fn rope_norm_end_to_end_gpu_paths_match_cpu() {
    use allpaka_backend::ops::{self, f16};

    let Some(region) = attached_region() else {
        return;
    };
    let region_mut =
        unsafe { std::slice::from_raw_parts_mut(region.as_ptr() as *mut u8, region.len()) };

    // Use offsets [44 MiB .. 56 MiB] disjoint from all other parity tests.
    // Test both MHA (`n_heads == n_kv == 2`, exercises two-dispatch `qk_prep_batch` + `attend_mm`
    // because `n_heads < 2 * n_kv`) and GQA (`n_heads = 4, n_kv = 2`, exercises fused `attend_mm_fq`).
    let hidden = 512usize;
    let n_heads = 4usize;
    let n_kv = 2usize;
    let hd = 128usize;
    let q_dim = n_heads * hd;
    let kv_dim = n_kv * hd;
    let ffn = 256usize;
    let vocab = 64usize;
    let eps = 1e-6f32;

    let mut cur = 44 << 20;
    let alloc_qmat = |rm: &mut [u8],
                      cur: &mut usize,
                      ty: GgmlType,
                      rows: usize,
                      cols: usize|
     -> (usize, usize) {
        tame_scales(rm, ty, *cur, rows, cols);
        let be = if ty == GgmlType::F32 {
            1
        } else if ty == GgmlType::Q8_0 {
            32
        } else {
            256
        };
        let bb = match ty {
            GgmlType::F32 => 4,
            GgmlType::Q8_0 => 34,
            GgmlType::Q4K => 144,
            GgmlType::Q6K => 210,
            _ => unreachable!(),
        };
        // Set dmin ≈ d / 4 so unsigned 4-bit nibbles (mean 7.5) with 6-bit
        // scales/mins (mean 31.5) are zero-centered instead of creating huge
        // rank-1 outputs (~100).
        if ty == GgmlType::Q4K {
            let half_dmin = 0x191Fu16.to_le_bytes(); // 0.0025 (= d / 4)
            for blk in rm[*cur..*cur + rows * cols / be * bb].chunks_exact_mut(144) {
                blk[2..4].copy_from_slice(&half_dmin);
            }
        }
        let len = rows * cols / be * bb;
        let off = *cur;
        *cur = (*cur + len + 4095) & !4095;
        (off, len)
    };
    let alloc_f32_vec = |rm: &mut [u8], cur: &mut usize, vals: &[f32]| -> (usize, usize) {
        let len = vals.len() * 4;
        let off = *cur;
        for (i, &v) in vals.iter().enumerate() {
            rm[off + i * 4..off + (i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        *cur = (*cur + len + 4095) & !4095;
        (off, len)
    };

    let (wq_off, wq_len) = alloc_qmat(region_mut, &mut cur, GgmlType::Q4K, q_dim, hidden);
    let (wk_off, wk_len) = alloc_qmat(region_mut, &mut cur, GgmlType::Q4K, kv_dim, hidden);
    let (wv_off, wv_len) = alloc_qmat(region_mut, &mut cur, GgmlType::Q6K, kv_dim, hidden);
    let (wo_off, wo_len) = alloc_qmat(region_mut, &mut cur, GgmlType::Q4K, hidden, q_dim);
    let (wg_off, wg_len) = alloc_qmat(region_mut, &mut cur, GgmlType::Q4K, ffn, hidden);
    let (wu_off, wu_len) = alloc_qmat(region_mut, &mut cur, GgmlType::Q4K, ffn, hidden);
    let (wd_off, wd_len) = alloc_qmat(region_mut, &mut cur, GgmlType::Q6K, hidden, ffn);
    let (out_off, out_len) = alloc_qmat(region_mut, &mut cur, GgmlType::Q6K, vocab, hidden);

    let attn_norm_f: Vec<f32> = (0..hidden).map(|i| 0.9 + (i % 31) as f32 * 0.005).collect();
    let ffn_norm_f: Vec<f32> = (0..hidden)
        .map(|i| 0.85 + (i % 29) as f32 * 0.005)
        .collect();
    let out_norm_f: Vec<f32> = (0..hidden)
        .map(|i| 0.95 + (i % 23) as f32 * 0.004)
        .collect();
    let q_norm_f: Vec<f32> = (0..hd).map(|i| 0.88 + i as f32 * 0.001).collect();
    let k_norm_f: Vec<f32> = (0..hd).map(|i| 0.92 + i as f32 * 0.001).collect();
    let (an_off, an_len) = alloc_f32_vec(region_mut, &mut cur, &attn_norm_f);
    let (fn_off, fn_len) = alloc_f32_vec(region_mut, &mut cur, &ffn_norm_f);
    let (on_off, on_len) = alloc_f32_vec(region_mut, &mut cur, &out_norm_f);

    let wq_g = QuantMat::new(
        &region[wq_off..wq_off + wq_len],
        GgmlType::Q4K,
        q_dim,
        hidden,
    )
    .unwrap();
    let wk_g = QuantMat::new(
        &region[wk_off..wk_off + wk_len],
        GgmlType::Q4K,
        kv_dim,
        hidden,
    )
    .unwrap();
    let wv_g = QuantMat::new(
        &region[wv_off..wv_off + wv_len],
        GgmlType::Q6K,
        kv_dim,
        hidden,
    )
    .unwrap();
    let wo_g = QuantMat::new(
        &region[wo_off..wo_off + wo_len],
        GgmlType::Q4K,
        hidden,
        q_dim,
    )
    .unwrap();
    let wg_g = QuantMat::new(&region[wg_off..wg_off + wg_len], GgmlType::Q4K, ffn, hidden).unwrap();
    let wu_g = QuantMat::new(&region[wu_off..wu_off + wu_len], GgmlType::Q4K, ffn, hidden).unwrap();
    let wd_g = QuantMat::new(&region[wd_off..wd_off + wd_len], GgmlType::Q6K, hidden, ffn).unwrap();
    let out_g = QuantMat::new(
        &region[out_off..out_off + out_len],
        GgmlType::Q6K,
        vocab,
        hidden,
    )
    .unwrap();

    // Exact dequantised rows for CPU reference.
    let deq =
        |qm: &QuantMat| -> Vec<Vec<f32>> { (0..qm.n_out).map(|j| qm.row(j).unwrap()).collect() };
    let wq_r = deq(&wq_g);
    let wk_r = deq(&wk_g);
    let wv_r = deq(&wv_g);
    let wo_r = deq(&wo_g);
    let wg_r = deq(&wg_g);
    let wu_r = deq(&wu_g);
    let wd_r = deq(&wd_g);
    let out_r = deq(&out_g);
    let dot_rows = |rows: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
        rows.iter()
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    };
    let round_f16 = |v: f32| f16::to_f32(f16::scalar_from_f32(v));
    let dot_rows_mm = |rows: &[Vec<f32>], x: &[f32]| -> Vec<f32> {
        let xr: Vec<f32> = x.iter().map(|&v| round_f16(v)).collect();
        rows.iter()
            .map(|r| r.iter().zip(&xr).map(|(&a, &b)| round_f16(a) * b).sum())
            .collect()
    };

    // Page-aligned KV cache wrapped for GPU + separate Vec<u16> for CPU.
    let cap = 64usize;
    let kv_elems = 2 * cap * kv_dim;
    let kv_bytes = (kv_elems * 2 + PAGE - 1) & !(PAGE - 1);
    let kv_layout = std::alloc::Layout::from_size_align(kv_bytes, PAGE).unwrap();
    let kv_ptr = unsafe { std::alloc::alloc_zeroed(kv_layout) };
    assert!(!kv_ptr.is_null());
    let kv_slice = unsafe { std::slice::from_raw_parts_mut(kv_ptr, kv_bytes) };
    let gpu_kv = gpu::wrap_region(kv_slice).expect("page-aligned KV must wrap");
    let mut cpu_kv = vec![0u16; kv_elems];
    let k_off = 0usize;
    let v_off = cap * kv_dim;
    let scale = (hd as f32).sqrt().recip();
    let group = n_heads / n_kv;

    let inv_freq = ops::rope_inv_freq(hd, 500_000.0);
    let make_ropes = |base: usize, m: usize| -> Vec<[f32; 2]> {
        let mut out = Vec::with_capacity(m * (hd / 2));
        let mut buf = Vec::new();
        for r in 0..m {
            ops::rope_sin_cos((base + r) as u32, &inv_freq, &mut buf);
            out.extend(buf.iter().map(|&(s, c)| [s, c]));
        }
        out
    };

    let cpu_attn_step =
        |h_in: &[f32], pos: usize, rope: &[[f32; 2]], kv: &mut [u16], mm_half: bool| -> Vec<f32> {
            let proj = |rows: &[Vec<f32>], x: &[f32]| {
                if mm_half {
                    dot_rows_mm(rows, x)
                } else {
                    dot_rows(rows, x)
                }
            };
            let mut q = proj(&wq_r, h_in);
            let mut k = proj(&wk_r, h_in);
            let v = proj(&wv_r, h_in);
            for h in 0..n_heads {
                let sl = &mut q[h * hd..(h + 1) * hd];
                ops::rmsnorm(sl, &q_norm_f, eps);
                ops::rope_norm_cached_from_array(sl, rope);
                if mm_half {
                    for qe in sl.iter_mut() {
                        *qe = round_f16(*qe);
                    }
                }
            }
            for h in 0..n_kv {
                let sl = &mut k[h * hd..(h + 1) * hd];
                ops::rmsnorm(sl, &k_norm_f, eps);
                ops::rope_norm_cached_from_array(sl, rope);
            }
            for d in 0..kv_dim {
                kv[k_off + pos * kv_dim + d] = f16::scalar_from_f32(k[d]);
                kv[v_off + pos * kv_dim + d] = f16::scalar_from_f32(v[d]);
            }
            let n_pos = pos + 1;
            let mut attn = vec![0.0f32; q_dim];
            for qh in 0..n_heads {
                let kvh = qh / group;
                let qv = &q[qh * hd..(qh + 1) * hd];
                let mut sc = vec![0.0f32; n_pos];
                let mut mx = f32::NEG_INFINITY;
                for t in 0..n_pos {
                    let ks =
                        &kv[k_off + t * kv_dim + kvh * hd..k_off + t * kv_dim + (kvh + 1) * hd];
                    let s = qv
                        .iter()
                        .zip(ks)
                        .map(|(&qe, &ke)| qe * f16::to_f32(ke))
                        .sum::<f32>()
                        * scale;
                    sc[t] = s;
                    mx = mx.max(s);
                }
                let mut den = 0.0f32;
                for s in &mut sc {
                    *s = (*s - mx).exp();
                    den += *s;
                }
                let dst = &mut attn[qh * hd..(qh + 1) * hd];
                for (t, &wgt) in sc.iter().enumerate() {
                    let p = wgt / den;
                    let vs =
                        &kv[v_off + t * kv_dim + kvh * hd..v_off + t * kv_dim + (kvh + 1) * hd];
                    for d in 0..hd {
                        dst[d] += p * f16::to_f32(vs[d]);
                    }
                }
            }
            proj(&wo_r, &attn)
        };

    // 1. Prefill (`m = 18`, `rope_norm = true`) via `QuantMat::prefill_attn_block`.
    let m_pf = 18usize;
    let hs_pf: Vec<f32> = (0..m_pf * hidden)
        .map(|i| ((i * 29 + 5) % 67) as f32 * 0.02 - 0.65)
        .collect();
    let ropes_pf = make_ropes(0, m_pf);
    let got_pf = QuantMat::prefill_attn_block(
        &wq_g,
        &wk_g,
        &wv_g,
        &wo_g,
        &hs_pf,
        m_pf,
        Some(&q_norm_f),
        Some(&k_norm_f),
        &ropes_pf,
        hd,
        true,
        false,
        None,
        eps,
        &gpu_kv,
        (k_off, v_off),
        (kv_dim, hd, n_heads, n_kv),
        0,
        scale,
        None,
    )
    .expect("GPU must accept prefill_attn_block with rope_norm=true");

    let mut want_pf = vec![0.0f32; m_pf * hidden];
    for r in 0..m_pf {
        let out_r_vec = cpu_attn_step(
            &hs_pf[r * hidden..(r + 1) * hidden],
            r,
            &ropes_pf[r * (hd / 2)..(r + 1) * (hd / 2)],
            &mut cpu_kv,
            true,
        );
        want_pf[r * hidden..(r + 1) * hidden].copy_from_slice(&out_r_vec);
    }
    // Compare KV cache written by prefill_attn_block first!
    let gpu_kv_u16 =
        unsafe { std::slice::from_raw_parts(kv_slice.as_ptr().cast::<u16>(), kv_elems) };
    for r in 0..m_pf {
        for d in 0..kv_dim {
            let gk = f16::to_f32(gpu_kv_u16[k_off + r * kv_dim + d]);
            let wk = f16::to_f32(cpu_kv[k_off + r * kv_dim + d]);
            assert!(
                (gk - wk).abs() < 2e-2,
                "prefill K[r={r}, d={d}]: gpu {gk} vs cpu {wk}"
            );
            let gv = f16::to_f32(gpu_kv_u16[v_off + r * kv_dim + d]);
            let wv = f16::to_f32(cpu_kv[v_off + r * kv_dim + d]);
            assert!(
                (gv - wv).abs() < 2e-2,
                "prefill V[r={r}, d={d}]: gpu {gv} vs cpu {wv}"
            );
        }
    }
    for (i, (&g, &w)) in got_pf.iter().zip(&want_pf).enumerate() {
        assert!(
            (g - w).abs() < 3e-2 + 1.5e-2 * w.abs(),
            "prefill_attn_block(rope_norm=true)[{i}]: gpu {g} vs cpu {w}"
        );
    }

    // 2. Single-token `QuantMat::attn_block` (`pos = m_pf`, `rope_norm = true`).
    let h_dec: Vec<f32> = (0..hidden)
        .map(|i| ((i * 37 + 11) % 71) as f32 * 0.02 - 0.7)
        .collect();
    let rope_dec = make_ropes(m_pf, 1);
    let got_ab = QuantMat::attn_block(
        &wq_g,
        &wk_g,
        &wv_g,
        &wo_g,
        &h_dec,
        Some(&q_norm_f),
        Some(&k_norm_f),
        &rope_dec,
        true,
        eps,
        &gpu_kv,
        (k_off, v_off),
        (kv_dim, hd, n_heads, n_kv),
        m_pf,
        scale,
    )
    .expect("GPU must accept attn_block with rope_norm=true");
    let want_ab = cpu_attn_step(&h_dec, m_pf, &rope_dec, &mut cpu_kv, false);
    for (i, (&g, &w)) in got_ab.iter().zip(&want_ab).enumerate() {
        assert!(
            (g - w).abs() < 1e-2 + 5e-3 * w.abs(),
            "attn_block(rope_norm=true)[{i}]: gpu {g} vs cpu {w}"
        );
    }

    // 3. Whole-token `gpu::decode_token` (`m = 1` logits and `m = 3` verify) with `rope_norm = true`.
    let layer = gpu::TokenLayer {
        attn_norm: &region[an_off..an_off + an_len],
        ffn_norm: &region[fn_off..fn_off + fn_len],
        gdn: None,
        wq: (GgmlType::Q4K, &region[wq_off..wq_off + wq_len], q_dim),
        wk: (GgmlType::Q4K, &region[wk_off..wk_off + wk_len], kv_dim),
        wv: (GgmlType::Q6K, &region[wv_off..wv_off + wv_len], kv_dim),
        wo: (GgmlType::Q4K, &region[wo_off..wo_off + wo_len], hidden),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        gate_in_q: false,
        q_norm: Some(&q_norm_f),
        k_norm: Some(&k_norm_f),
        k_off,
        v_off,
        ffn: gpu::TokenFfn::Dense {
            gate: (GgmlType::Q4K, &region[wg_off..wg_off + wg_len], ffn),
            up: (GgmlType::Q4K, &region[wu_off..wu_off + wu_len], ffn),
            down: (GgmlType::Q6K, &region[wd_off..wd_off + wd_len], hidden),
        },
    };

    let cpu_full_step = |x_in: &[f32], pos: usize, rope: &[[f32; 2]], kv: &mut [u16]| {
        let mut x = x_in.to_vec();
        let mut h_a = x.clone();
        ops::rmsnorm(&mut h_a, &attn_norm_f, eps);
        let d_a = cpu_attn_step(&h_a, pos, rope, kv, false);
        for (xi, di) in x.iter_mut().zip(&d_a) {
            *xi += di;
        }
        let mut h_f = x.clone();
        ops::rmsnorm(&mut h_f, &ffn_norm_f, eps);
        let g = dot_rows(&wg_r, &h_f);
        let u = dot_rows(&wu_r, &h_f);
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui)
            .collect();
        let d_f = dot_rows(&wd_r, &act);
        for (xi, di) in x.iter_mut().zip(&d_f) {
            *xi += di;
        }
        let mut normed = x;
        ops::rmsnorm(&mut normed, &out_norm_f, eps);
        let logits = dot_rows(&out_r, &normed);
        (normed, logits)
    };

    let pos1 = m_pf + 1;
    let x1: Vec<f32> = (0..hidden)
        .map(|i| ((i * 43 + 3) % 73) as f32 * 0.02 - 0.72)
        .collect();
    let rope1 = make_ropes(pos1, 1);
    let out1 = gpu::decode_token(&gpu::TokenReq {
        layers: std::slice::from_ref(&layer),
        output_norm: &region[on_off..on_off + on_len],
        output: (GgmlType::Q6K, &region[out_off..out_off + out_len], vocab),
        embd: None,
        token_id: None,
        x: &x1,
        m: 1,
        rope: &rope1,
        rot_dim: hd,
        rope_norm: true,
        eps,
        cache: &gpu_kv,
        cache_capacity: cap,
        ssm: None,
        ssm_slots: None,
        kv_dim,
        head_dim: hd,
        n_heads,
        n_kv_heads: n_kv,
        pos: pos1,
        scale,
        argmax: false,
    })
    .expect("GPU decode_token(m=1, rope_norm=true) must succeed");
    let gpu::TokenOut::Logits(got_logits1) = out1 else {
        panic!("expected TokenOut::Logits");
    };
    let (_, want_logits1) = cpu_full_step(&x1, pos1, &rope1, &mut cpu_kv);
    for (i, (&g, &w)) in got_logits1.iter().zip(&want_logits1).enumerate() {
        assert!(
            (g - w).abs() < 2e-2 + 1e-2 * w.abs(),
            "decode_token(m=1, rope_norm=true) logit[{i}]: gpu {g} vs cpu {w}"
        );
    }

    // m = 3 verify path.
    let pos3 = pos1 + 1;
    let m_v = 3usize;
    let x3: Vec<f32> = (0..m_v * hidden)
        .map(|i| ((i * 53 + 17) % 79) as f32 * 0.02 - 0.78)
        .collect();
    let rope3 = make_ropes(pos3, m_v);
    let out3 = gpu::decode_token(&gpu::TokenReq {
        layers: std::slice::from_ref(&layer),
        output_norm: &region[on_off..on_off + on_len],
        output: (GgmlType::Q6K, &region[out_off..out_off + out_len], vocab),
        embd: None,
        token_id: None,
        x: &x3,
        m: m_v,
        rope: &rope3,
        rot_dim: hd,
        rope_norm: true,
        eps,
        cache: &gpu_kv,
        cache_capacity: cap,
        ssm: None,
        ssm_slots: None,
        kv_dim,
        head_dim: hd,
        n_heads,
        n_kv_heads: n_kv,
        pos: pos3,
        scale,
        argmax: true,
    })
    .expect("GPU decode_token(m=3, rope_norm=true) must succeed");
    let gpu::TokenOut::Rows {
        argmax: got_am3,
        hidden: got_hid3,
    } = out3
    else {
        panic!("expected TokenOut::Rows");
    };
    for r in 0..m_v {
        let (want_hid, want_logits) = cpu_full_step(
            &x3[r * hidden..(r + 1) * hidden],
            pos3 + r,
            &rope3[r * (hd / 2)..(r + 1) * (hd / 2)],
            &mut cpu_kv,
        );
        let want_winner = want_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        assert_eq!(
            got_am3[r], want_winner,
            "verify row {r} argmax mismatch (gpu {} vs cpu {})",
            got_am3[r], want_winner
        );
        for d in 0..hidden {
            let g = got_hid3[r * hidden + d];
            let w = want_hid[d];
            assert!(
                (g - w).abs() < 3e-2 + 1e-2 * w.abs(),
                "verify row {r} hidden[{d}]: gpu {g} vs cpu {w}"
            );
        }
    }

    drop(gpu_kv);
    unsafe { std::alloc::dealloc(kv_ptr, kv_layout) };
}

#[test]
fn gpu_embed_decode_token_matches_cpu_for_all_ten_formats() {
    let Some(region) = attached_region() else {
        return;
    };
    let region_mut =
        unsafe { std::slice::from_raw_parts_mut(region.as_ptr() as *mut u8, region.len()) };

    // Use offsets [58 MiB .. 63 MiB] disjoint from all other parity tests.
    let hidden = 512usize;
    let vocab = 64usize;
    let n_tokens = 6usize;
    let norm_off = 58 << 20;
    let out_off = 59 << 20;
    let embd_base_off = 60 << 20;

    // Output norm = finite positive F32 weights.
    for (i, c) in region_mut[norm_off..norm_off + hidden * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        c.copy_from_slice(&(0.75f32 + (i % 7) as f32 * 0.05).to_le_bytes());
    }
    tame_scales(region_mut, GgmlType::Q4K, out_off, vocab, hidden);
    let out_bytes = vocab * hidden / 256 * 144;

    let cache = gpu::wrap_region(&region[0..PAGE]).expect("wrap_region");
    let rope = vec![[0.0f32, 1.0f32]; 64];

    let formats = [
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q5_0,
        GgmlType::F16,
        GgmlType::F32,
        GgmlType::BF16,
    ];

    for (idx, ty) in formats.into_iter().enumerate() {
        let off = embd_base_off + idx * (256 << 10);
        let rb = ty.expected_bytes(hidden).unwrap();
        let total = n_tokens * rb;

        match ty {
            GgmlType::F32 => {
                tame_scales(region_mut, ty, off, n_tokens, hidden);
            }
            GgmlType::F16 => {
                for (i, c) in region_mut[off..off + total].chunks_exact_mut(2).enumerate() {
                    let h = allpaka_dtype::f32_to_f16(((i % 31) as f32 - 15.0) * 0.04);
                    c.copy_from_slice(&h.to_le_bytes());
                }
            }
            GgmlType::BF16 => {
                for (i, c) in region_mut[off..off + total].chunks_exact_mut(2).enumerate() {
                    let b = allpaka_dtype::f32_to_bf16(((i % 29) as f32 - 14.0) * 0.05);
                    c.copy_from_slice(&b.to_le_bytes());
                }
            }
            GgmlType::Q5_0 => {
                let half = 0x211Fu16.to_le_bytes();
                for r in 0..n_tokens {
                    for blk in 0..hidden / 32 {
                        let at = off + r * rb + blk * 22;
                        region_mut[at] = half[0];
                        region_mut[at + 1] = half[1];
                    }
                }
            }
            other => tame_scales(region_mut, other, off, n_tokens, hidden),
        }

        let embd_slice = &region[off..off + total];
        let embd_cpu = QuantMat::new(embd_slice, ty, n_tokens, hidden).unwrap();
        let tok_id = 3u32;
        let x_ref = embd_cpu.row(tok_id as usize).unwrap();

        let run_decode =
            |embd: Option<(GgmlType, &[u8], usize)>, token_id: Option<u32>, x: &[f32]| {
                gpu::decode_token(&gpu::TokenReq {
                    layers: &[],
                    output_norm: &region[norm_off..norm_off + hidden * 4],
                    output: (GgmlType::Q4K, &region[out_off..out_off + out_bytes], vocab),
                    embd,
                    token_id,
                    x,
                    m: 1,
                    rope: &rope,
                    rot_dim: 128,
                    rope_norm: false,
                    eps: 1e-5,
                    cache: &cache,
                    cache_capacity: 8,
                    ssm: None,
                    ssm_slots: None,
                    kv_dim: 256,
                    head_dim: 128,
                    n_heads: 4,
                    n_kv_heads: 2,
                    pos: 0,
                    scale: 1.0 / 128.0f32.sqrt(),
                    argmax: false,
                })
            };

        let Some(gpu::TokenOut::Logits(got)) =
            run_decode(Some((ty, embd_slice, hidden)), Some(tok_id), &[])
        else {
            panic!("GPU decode_token declined gpu embed for {ty:?}");
        };
        let Some(gpu::TokenOut::Logits(want)) = run_decode(None, None, &x_ref) else {
            panic!("GPU decode_token declined cpu-staged x for {ty:?}");
        };

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4 + 1e-4 * w.abs(),
                "gpu embed {ty:?} logit {i}: {g} vs {w}"
            );
        }
    }
}
