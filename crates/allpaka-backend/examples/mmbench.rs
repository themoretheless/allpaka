// Microbench: our mm64ll_q4_k vs llama.cpp kernel_mul_mm_q4_K on identical data.
use metal::*;
use std::time::Instant;

#[repr(C)]
struct LlamaMmArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

fn time_kernel(
    queue: &CommandQueue,
    f: impl Fn(&ComputeCommandEncoderRef),
    iters: u64,
) -> f64 {
    let iters = std::env::var("MM_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(iters);
    // warmup
    for _ in 0..3 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        f(enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        for _ in 0..iters {
            f(enc);
        }
        enc.end_encoding();
        let t0 = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let wall = t0.elapsed().as_secs_f64();
        
        best = best.min(wall / iters as f64);
    }
    best
}

fn main() {
    let n_in: usize = std::env::var("MM_NIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
    let n_out: usize = std::env::var("MM_NOUT").ok().and_then(|v| v.parse().ok()).unwrap_or(12288);
    let m: usize = std::env::var("MM_M").ok().and_then(|v| v.parse().ok()).unwrap_or(480);
    let n_par: usize = std::env::var("MM_PAR").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    // MM_WSPAN: total weight pool in MiB; each of the 10 iterations reads a
    // different window (cold streaming, like a real MoE layer).
    let w_span = std::env::var("MM_WSPAN").ok().and_then(|v| v.parse::<usize>().ok())
        .map_or(0, |mb| mb << 20);
    let row_bytes = n_in / 256 * 144;
    let w_len = n_out * row_bytes;
    let w_pool = if w_span > w_len { w_span } else { w_len };
    let w_step = if w_span > w_len { (w_span - w_len) / 10 } else { 0 };

    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let mut w = vec![0u8; w_pool];
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i as u32).wrapping_mul(2654435761).rotate_left(13) as u8;
    }
    // zero out scale bytes a bit to keep floats sane is unnecessary for timing
    let x: Vec<f32> = (0..m * n_in).map(|i| ((i % 97) as f32) * 0.01).collect();

    let w_buf = device.new_buffer_with_data(
        w.as_ptr() as *const _,
        w_pool as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_bufs: Vec<Buffer> = (0..n_par)
        .map(|_| {
            device.new_buffer(
                (m * n_out * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .collect();
    let y_buf = &y_bufs[0];

    // --- ours ---
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();
    for name in ["mmll_q4_k", "mm64ll_q4_k"] {
        let fun = lib.get_function(name, None).unwrap();
        let pipe = device
            .new_compute_pipeline_state_with_function(&fun)
            .unwrap();
        let gx = (n_out as u64).div_ceil(64);
        let gy = (m as u64).div_ceil(32);
        let ni = n_in as u32;
        let no = n_out as u32;
        let wo = 0u64;
        let mm = m as u32;
        let iter_no = std::cell::Cell::new(0u64);
        let t = time_kernel(
            &queue,
            |enc| {
                let wo = iter_no.get() * w_step as u64;
                iter_no.set(iter_no.get() + 1);
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&y_buf), 0);
                enc.set_bytes(3, 4, &ni as *const u32 as *const _);
                enc.set_bytes(4, 4, &no as *const u32 as *const _);
                enc.set_bytes(5, 8, &wo as *const u64 as *const _);
                enc.set_bytes(6, 4, &mm as *const u32 as *const _);
                for p in 0..n_par {
                    enc.set_buffer(2, Some(&y_bufs[p]), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new(gx, gy, 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
            },
            10,
        );
        let fl = 2.0 * (m * n_in * n_out) as f64 * n_par as f64;
        println!(
            "ours {name}: {:.3} ms, {:.2} TFLOPS",
            t * 1e3,
            fl / t / 1e12
        );
    }

    // --- llama ---
    let src = std::fs::read_to_string("/tmp/mm_full.metal").unwrap();
    let llib = device
        .new_library_with_source(&src, &CompileOptions::new())
        .unwrap();
    for name in ["llama_mul_mm_q4_K"] {
        let fun = llib.get_function(name, None).unwrap();
        let pipe = device
            .new_compute_pipeline_state_with_function(&fun)
            .unwrap();
        let args = LlamaMmArgs {
            ne00: n_in as i32,
            ne02: 1,
            nb01: row_bytes as u64,
            nb02: (n_out * row_bytes) as u64,
            nb03: (n_out * row_bytes) as u64,
            ne12: 1,
            nb10: 4,
            nb11: (n_in * 4) as u64,
            nb12: (m * n_in * 4) as u64,
            nb13: (m * n_in * 4) as u64,
            ne0: n_out as i32,
            ne1: m as i32,
            r2: 1,
            r3: 1,
        };
        let gx = (m as u64).div_ceil(32);
        let gy = (n_out as u64).div_ceil(64);
        let t = time_kernel(
            &queue,
            |enc| {
                enc.set_compute_pipeline_state(&pipe);
                enc.set_bytes(0, std::mem::size_of::<LlamaMmArgs>() as u64, &args as *const _ as *const _);
                enc.set_buffer(1, Some(&w_buf), 0);
                enc.set_buffer(2, Some(&x_buf), 0);
                enc.set_buffer(3, Some(&y_buf), 0);
                enc.set_threadgroup_memory_length(0, 8192);
                for p in 0..n_par {
                    enc.set_buffer(3, Some(&y_bufs[p]), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new(gx, gy, 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
            },
            10,
        );
        let fl = 2.0 * (m * n_in * n_out) as f64 * n_par as f64;
        println!(
            "llama {name}: {:.3} ms, {:.2} TFLOPS",
            t * 1e3,
            fl / t / 1e12
        );
    }
}
