//! Effective bandwidth of the GLM-4.5-Air decode matvecs, in isolation.
//!
//! The Air's decode reads: expert gate/up Q4_K [1408, 4096], expert down
//! Q8_0 [4096, 1408], shared gate/up Q5_K [1408, 4096], attn q Q4_K
//! [12288, 4096], k Q4_K [1024, 4096], v Q6_K [1024, 4096], wo Q4_K
//! [4096, 12288] and the Q6_K head. This measures each shape's matvec by
//! the GPU's own clock and prints GB/s, so a kernel change can be judged
//! in seconds without loading a 46 GiB model.
//!
//! Run: `cargo test -p allpaka-backend --test gpu_glm_mvbench -- --ignored --nocapture`

#![cfg(target_os = "macos")]

use allpaka_backend::{gpu, QuantMat};
use allpaka_gguf::GgmlType;

const PAGE: usize = 16384;

fn block_bytes(ty: GgmlType, n_in: usize) -> usize {
    match ty {
        GgmlType::Q4K => n_in / 256 * 144,
        GgmlType::Q5K => n_in / 256 * 176,
        GgmlType::Q6K => n_in / 256 * 210,
        GgmlType::Q8_0 => n_in / 32 * 34,
        other => panic!("no block size for {other:?}"),
    }
}

fn bench_shape(region: &[u8], ty: GgmlType, n_out: usize, n_in: usize, label: &str) {
    let bytes = n_out * block_bytes(ty, n_in);
    // Enough distinct matrices to defeat the last-level cache, capped so the
    // head shape still gets several.
    let count = ((640usize << 20) / bytes).clamp(2, 128);
    let mats: Vec<QuantMat> = (0..count)
        .map(|i| QuantMat::new(&region[i * bytes..(i + 1) * bytes], ty, n_out, n_in).unwrap())
        .collect();
    let x = vec![0.01f32; n_in];
    let items: Vec<(&QuantMat, &[f32])> = mats.iter().map(|m| (m, x.as_slice())).collect();
    QuantMat::matmul_many(&items).unwrap();
    let before = gpu::gpu_time_stats();
    let rounds = 5;
    for _ in 0..rounds {
        QuantMat::matmul_many(&items).unwrap();
    }
    let after = gpu::gpu_time_stats();
    let busy = (after.0 - before.0) as f64 / 1e9;
    let total = bytes as f64 * count as f64 * rounds as f64;
    println!(
        "{label:<28} {ty:?} [{n_out},{n_in}] {:>7.2} MB x {count}: {:.1} GB/s GPU-clock",
        bytes as f64 / 1e6,
        total / busy / 1e9,
    );
}

#[test]
#[ignore = "a bandwidth measurement; run explicitly"]
fn glm_decode_matvecs_report_effective_bandwidth() {
    let len: usize = 2 << 30;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null());
    let region = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for b in region.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (state >> 56) as u8;
    }
    // Scales of random bits include inf/NaN; the kernels do finite arithmetic
    // either way, but NaN accumulators can change instruction timing.
    if !gpu::attach(region) {
        eprintln!("SKIP: no Metal device");
        return;
    }

    bench_shape(region, GgmlType::Q4K, 1408, 4096, "expert gate/up");
    bench_shape(region, GgmlType::Q4K, 2048, 4096, "probe n_out 2048");
    bench_shape(region, GgmlType::Q4K, 4096, 4096, "probe n_out 4096");
    bench_shape(region, GgmlType::Q4K, 11264, 4096, "probe 8x1408 rows");
    bench_shape(region, GgmlType::Q8_0, 4096, 1408, "expert down");
    bench_shape(region, GgmlType::Q5K, 1408, 4096, "shared gate/up");
    bench_shape(region, GgmlType::Q8_0, 4096, 1408, "shared down");
    bench_shape(region, GgmlType::Q4K, 12288, 4096, "attn q");
    bench_shape(region, GgmlType::Q4K, 4096, 12288, "attn wo");
    bench_shape(region, GgmlType::Q4K, 1024, 4096, "attn k");
    bench_shape(region, GgmlType::Q6K, 1024, 4096, "attn v");
    bench_shape(region, GgmlType::Q6K, 8192, 4096, "head slice");
}
