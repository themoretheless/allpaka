//! Isolates residual handoff cost; no model inference or GPU kernels are timed.
#[cfg(target_os = "macos")]
use allpaka_backend::{gpu, ops};
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
fn main() {
    let layout = std::alloc::Layout::from_size_align(16384, 16384).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    let mapping = unsafe { std::slice::from_raw_parts(ptr, 16384) };
    assert!(
        gpu::attach(mapping),
        "Metal required: {:?}",
        gpu::init_error()
    );
    let norm = vec![1.0; 4096];
    for bytes in [2 << 20, 8 << 20, 32 << 20] {
        let input = vec![0.125f32; bytes / 4];
        for pair in 0..9 {
            for mode in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                std::env::set_var("ALLPAKA_PREFILL_ZERO_COPY", mode.to_string());
                let mut samples = Vec::new();
                for iteration in 0..110 {
                    gpu::prefill_begin(&input).unwrap();
                    let start = Instant::now();
                    let mut output = gpu::prefill_end_output(input.len()).unwrap();
                    assert_eq!(output[0], 0.125);
                    let last = output.len() - norm.len();
                    ops::rmsnorm(&mut output[last..], &norm, 1e-6);
                    std::hint::black_box(&output[last..]);
                    drop(output);
                    if iteration >= 10 {
                        samples.push(start.elapsed().as_nanos() as u64);
                    }
                }
                samples.sort_unstable();
                println!("{{\"bytes\":{bytes},\"pair\":{pair},\"mode\":{mode},\"median_ns\":{},\"min_ns\":{}}}", samples[samples.len()/2], samples[0]);
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("This benchmark requires Apple Metal");
}
