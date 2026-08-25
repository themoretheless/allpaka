//! Effective bandwidth of the decode ffn kernels, in isolation.
//!
//! The 235B's decode reads expert weights through matvec_q2_k (gate/up,
//! 1536x4096) and matvec_q3_k (down, 4096x1536) at m = 1. This measures
//! exactly those dispatches by the GPU's own clock and prints GB/s, so a
//! kernel change can be judged in seconds without loading an 83 GiB model.
//!
//! Weights cycle through enough distinct matrices to defeat the last-level
//! cache: a single hot matrix would overstate bandwidth several-fold.
//!
//! Run: `cargo test -p allpaka-backend --test gpu_ffnbench -- --ignored --nocapture`
//! Sweep: prefix with ALLPAKA_LPR_DIV=2 (or 4) for fewer lanes per row.

#![cfg(target_os = "macos")]

use allpaka_backend::{gpu, QuantMat};
use allpaka_gguf::GgmlType;

const PAGE: usize = 16384;

#[test]
#[ignore = "a bandwidth measurement; run explicitly"]
fn ffn_shaped_matvecs_report_effective_bandwidth() {
    let len: usize = 640 << 20;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null());
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for b in region.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    // Scales of random bits include inf/NaN; zero every f16 scale field so
    // the kernels do finite arithmetic (speed is the same either way, but
    // NaN accumulators can change instruction timing on some GPUs).
    assert!(gpu::attach(region), "no Metal device");

    // gate/up shape: Q2_K [1536, 4096]; down shape: Q3_K [4096, 1536].
    let q2_bytes = 1536 * (4096 / 256) * 84;
    let q3_bytes = 4096 * (1536 / 256) * 110;
    let per_pair = q2_bytes + q3_bytes;
    let pairs = len / per_pair - 1;
    let x_gate = vec![0.01f32; 4096];
    let x_down = vec![0.01f32; 1536];

    let mut mats = Vec::new();
    let mut off = 0usize;
    for _ in 0..pairs {
        mats.push((GgmlType::Q2K, off, 1536usize, 4096usize));
        off += q2_bytes;
        mats.push((GgmlType::Q3K, off, 4096usize, 1536usize));
        off += q3_bytes;
    }

    let qmats: Vec<QuantMat> = mats
        .iter()
        .map(|&(ty, off, n_out, n_in)| {
            let bytes = match ty {
                GgmlType::Q2K => n_out * n_in / 256 * 84,
                _ => n_out * n_in / 256 * 110,
            };
            QuantMat::new(&region[off..off + bytes], ty, n_out, n_in).unwrap()
        })
        .collect();

    // Warm up once, then measure several rounds.
    let items: Vec<(&QuantMat, &[f32])> = qmats
        .iter()
        .map(|m| (m, if m.n_in == 4096 { x_gate.as_slice() } else { x_down.as_slice() }))
        .collect();
    QuantMat::matmul_many(&items).unwrap();

    let bytes_per_round: usize = mats
        .iter()
        .map(|&(ty, _, n_out, n_in)| match ty {
            GgmlType::Q2K => n_out * n_in / 256 * 84,
            _ => n_out * n_in / 256 * 110,
        })
        .sum();
    // The same measurement over Q8_0, whose dequant is one multiply: if
    // this runs at markedly more bytes per second, the K-quant kernels are
    // compute-bound on their bit plumbing; if it matches, the memory path
    // itself is the ceiling.
    let q8_bytes = 1536 * (4096 / 32) * 34;
    let q8_count = (len / q8_bytes).min(64) - 1;
    let q8_mats: Vec<QuantMat> = (0..q8_count)
        .map(|i| QuantMat::new(&region[i * q8_bytes..(i + 1) * q8_bytes], GgmlType::Q8_0, 1536, 4096).unwrap())
        .collect();
    let q8_items: Vec<(&QuantMat, &[f32])> =
        q8_mats.iter().map(|m| (m, x_gate.as_slice())).collect();
    QuantMat::matmul_many(&q8_items).unwrap();
    let before8 = gpu::gpu_time_stats();
    for _ in 0..3 {
        QuantMat::matmul_many(&q8_items).unwrap();
    }
    let after8 = gpu::gpu_time_stats();
    let busy8 = (after8.0 - before8.0) as f64 / 1e9;
    println!(
        "q8_0 control: {:.1} GB/s GPU-clock ({} matvecs of {:.1} MB)",
        (q8_bytes * q8_count * 3) as f64 / busy8 / 1e9,
        q8_count,
        q8_bytes as f64 / 1e6,
    );

    let rounds = 5;
    let before = gpu::gpu_time_stats();
    let t0 = std::time::Instant::now();
    for _ in 0..rounds {
        QuantMat::matmul_many(&items).unwrap();
    }
    let wall = t0.elapsed().as_secs_f64();
    let after = gpu::gpu_time_stats();
    let busy = (after.0 - before.0) as f64 / 1e9;
    let total = bytes_per_round as f64 * rounds as f64;
    println!(
        "{} matvecs/round, {:.0} MB/round: GPU-clock {:.1} GB/s, wall {:.1} GB/s",
        mats.len(),
        bytes_per_round as f64 / 1e6,
        total / busy / 1e9,
        total / wall / 1e9,
    );
}

/// The prefill counterpart: tile-matmul (mm) kernels at chunk batch sizes,
/// judged by GPU clock. The shapes are the 235B's expert stages at a
/// realistic per-expert group size and the dense qkv projection at the full
/// chunk. Iterating an mm-kernel change takes seconds here against minutes
/// for a full bench - and, unlike wall clock, the GPU clock does not charge
/// the encode or the arena staging to the kernel.
#[test]
#[ignore = "a bandwidth measurement; run explicitly"]
fn mm_shaped_matmuls_report_effective_bandwidth() {
    let len: usize = 640 << 20;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null());
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut state = 0xB5AD_4ECE_DA1C_E2A9u64;
    for b in region.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    assert!(gpu::attach(region), "no Metal device");

    // Expert stage at a 32-row group: [1536, 4096] Q2_K, m = 32.
    let q2_bytes = 1536 * (4096 / 256) * 84;
    let count = (len / q2_bytes).min(64) - 1;
    let mats: Vec<QuantMat> = (0..count)
        .map(|i| {
            QuantMat::new(&region[i * q2_bytes..(i + 1) * q2_bytes], GgmlType::Q2K, 1536, 4096)
                .unwrap()
        })
        .collect();
    let m = 32usize;
    let x = vec![0.01f32; m * 4096];
    let items: Vec<(&QuantMat, &[f32])> = mats.iter().map(|q| (q, x.as_slice())).collect();
    QuantMat::matmul_many(&items).unwrap();
    let before = gpu::gpu_time_stats();
    let rounds = 5;
    for _ in 0..rounds {
        QuantMat::matmul_many(&items).unwrap();
    }
    let after = gpu::gpu_time_stats();
    let busy = (after.0 - before.0) as f64 / 1e9;
    let bytes = (q2_bytes * count * rounds) as f64;
    let elems = (1536usize * 4096 * count * rounds) as f64;
    println!(
        "mm q2_k m={m}: {:.1} GB/s GPU-clock, {:.2} Tel/s weights-side ({count} mats)",
        bytes / busy / 1e9,
        elems * m as f64 / busy / 1e12,
    );
}

