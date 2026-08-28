//! Effective bandwidth of the Qwen3.6-35B-A3B decode matvecs, in isolation.
//! Shapes from the GGUF: experts Q4_K [512,2048] gate/up + Q5_K/Q6_K
//! [2048,512] down; GDN Q8_0 qkv [8192,2048], gate [4096,2048], out
//! [2048,4096]; full-attention Q8_0 q [8192,2048], k/v [512,2048], out
//! [2048,4096]; head Q6_K [248320,2048]. Same harness as gpu_glm_mvbench.
//!
//! Run: `cargo test -p allpaka-backend --test gpu_q35_mvbench -- --ignored --nocapture`

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
fn q35_decode_matvecs_report_effective_bandwidth() {
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
    if !gpu::attach(region) {
        eprintln!("SKIP: no Metal device");
        return;
    }

    bench_shape(region, GgmlType::Q4K, 512, 2048, "expert gate/up");
    bench_shape(region, GgmlType::Q4K, 4096, 2048, "probe 8x512 rows");
    bench_shape(region, GgmlType::Q5K, 2048, 512, "expert down q5");
    bench_shape(region, GgmlType::Q6K, 2048, 512, "expert down q6");
    bench_shape(region, GgmlType::Q8_0, 8192, 2048, "gdn qkv / attn q");
    bench_shape(region, GgmlType::Q8_0, 4096, 2048, "gdn gate");
    bench_shape(region, GgmlType::Q8_0, 2048, 4096, "gdn/attn out");
    bench_shape(region, GgmlType::Q8_0, 512, 2048, "attn k/v");
    bench_shape(region, GgmlType::Q6K, 248320, 2048, "head");
}
