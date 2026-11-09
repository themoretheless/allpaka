//! Effective bandwidth of dense Q4_K decode matvecs at Qwen3-8B shapes:
//! gate/up [12288x4096], down [4096x12288], qkv-sized [6144x4096], with a
//! Q8_0 control. Decode reads these once per token; llama-bench shows the
//! 8B at ~170 GB/s effective while the in-model decode path here shows
//! ~70-90, so this isolates kernel throughput from orchestration.
//!
//! Run: `cargo test -p allpaka-backend --test gpu_densebench -- --ignored --nocapture`

#![cfg(target_os = "macos")]

use allpaka_backend::{gpu, QuantMat};
use allpaka_gguf::GgmlType;

const PAGE: usize = 16384;

fn q4k_bytes(n_out: usize, n_in: usize) -> usize {
    n_out * (n_in / 256) * 144
}

fn q8_bytes(n_out: usize, n_in: usize) -> usize {
    n_out * (n_in / 32) * 34
}

#[test]
#[ignore = "a bandwidth measurement; run explicitly"]
fn dense_q4k_shapes_report_effective_bandwidth() {
    let len: usize = 640 << 20;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null());
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for b in region.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    if !gpu::attach(region) {
        eprintln!("SKIP: no Metal device");
        return;
    }

    // Enough distinct matrices to defeat the last-level cache.
    let shapes: [(usize, usize); 3] = [
        (12288, 4096), // gate / up
        (4096, 12288), // down
        (6144, 4096),  // qkv-shaped dense projection
    ];
    let mut off = 0usize;
    let mut mats: Vec<(GgmlType, usize, usize, usize)> = Vec::new();
    while off + q4k_bytes(12288, 4096) < len {
        for &(n_out, n_in) in &shapes {
            let bytes = q4k_bytes(n_out, n_in);
            if off + bytes >= len {
                break;
            }
            mats.push((GgmlType::Q4K, off, n_out, n_in));
            off += bytes;
        }
    }
    let qmats: Vec<QuantMat> = mats
        .iter()
        .map(|&(ty, off, n_out, n_in)| {
            QuantMat::new(&region[off..off + q4k_bytes(n_out, n_in)], ty, n_out, n_in).unwrap()
        })
        .collect();
    let x4096 = vec![0.01f32; 4096];
    let x12288 = vec![0.01f32; 12288];
    let items: Vec<(&QuantMat, &[f32])> = qmats
        .iter()
        .map(|m| {
            (
                m,
                if m.n_in == 4096 {
                    x4096.as_slice()
                } else {
                    x12288.as_slice()
                },
            )
        })
        .collect();
    QuantMat::matmul_many(&items).unwrap();

    let bytes_per_round: usize = mats
        .iter()
        .map(|&(_, _, n_out, n_in)| q4k_bytes(n_out, n_in))
        .sum();
    let rounds = 5;
    let before = gpu::gpu_time_stats();
    let t0 = std::time::Instant::now();
    for _ in 0..rounds {
        QuantMat::matmul_many(&items).unwrap();
    }
    let after = gpu::gpu_time_stats();
    let wall = t0.elapsed().as_secs_f64();
    let busy = (after.0 - before.0) as f64 / 1e9;
    let total = (bytes_per_round * rounds) as f64;
    println!(
        "q4_k dense shapes: {:.1} GB/s GPU-clock, {:.1} GB/s wall ({} matvecs/round)",
        total / busy / 1e9,
        total / wall / 1e9,
        mats.len(),
    );

    // Q8_0 control at the gate/up shape: same bytes/element ratio 34/32 vs
    // 144/256 differs, so compare effective weight bytes per second.
    let q8 = q8_bytes(12288, 4096);
    let q8_count = ((len / q8).min(64)) - 1;
    let q8_mats: Vec<QuantMat> = (0..q8_count)
        .map(|i| QuantMat::new(&region[i * q8..(i + 1) * q8], GgmlType::Q8_0, 12288, 4096).unwrap())
        .collect();
    let q8_items: Vec<(&QuantMat, &[f32])> =
        q8_mats.iter().map(|m| (m, x4096.as_slice())).collect();
    QuantMat::matmul_many(&q8_items).unwrap();
    let before8 = gpu::gpu_time_stats();
    for _ in 0..3 {
        QuantMat::matmul_many(&q8_items).unwrap();
    }
    let after8 = gpu::gpu_time_stats();
    let busy8 = (after8.0 - before8.0) as f64 / 1e9;
    println!(
        "q8_0 control: {:.1} GB/s GPU-clock ({} matvecs of {:.1} MB)",
        (q8 * q8_count * 3) as f64 / busy8 / 1e9,
        q8_count,
        q8 as f64 / 1e6,
    );
}
