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
    // MM_ID=1: bench the grouped (mmid) gate kernel mmllg_id_q4_k instead -
    // MM_EXP experts of which MM_ACT are active with m*ACT/EXP rows each,
    // the rest empty. Measures the low-occupancy regime of MoE prefill.
    let id_mode = std::env::var("MM_ID").is_ok_and(|v| v == "1" || v == "2");
    let n_exp: usize = std::env::var("MM_EXP").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let n_act: usize = std::env::var("MM_ACT").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    // Tokens pick MM_USED experts each, so the (token, expert) pairs are
    // m*USED regardless of how many experts they land on.
    let n_used: usize = std::env::var("MM_USED").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    // MM_STRUCT=1: the attention buffer's mm+barrier skeleton (q,k,v mms,
    // barrier, wo mm) in ONE command buffer per rep - discriminates
    // stage-structure cost from kernel cost.
    let struct_mode = std::env::var("MM_STRUCT").is_ok_and(|v| v == "1");
    // MM_WSPAN: total weight pool in MiB; each of the 10 iterations reads a
    // different window (cold streaming, like a real MoE layer).
    let w_span = std::env::var("MM_WSPAN").ok().and_then(|v| v.parse::<usize>().ok())
        .map_or(0, |mb| mb << 20);
    // MM_FMT: weight format (row_bytes and the kernel list follow it).
    let fmt = std::env::var("MM_FMT").unwrap_or_else(|_| "q4_k".into());
    let row_bytes = match fmt.as_str() {
        "q8_0" => n_in / 32 * 34,
        "q5_k" => n_in / 256 * 176,
        _ => n_in / 256 * 144,
    };
    let w_len = n_out * row_bytes
        * if id_mode { n_exp } else if struct_mode { 3 } else { 1 };
    let w_pool = if w_span > w_len { w_span } else { w_len };
    let w_step = if w_span > w_len { (w_span - w_len) / 10 } else { 0 };

    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let mut w = vec![0u8; w_pool];
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i as u32).wrapping_mul(2654435761).rotate_left(13) as u8;
    }
    // zero out scale bytes a bit to keep floats sane is unnecessary for timing
    let x: Vec<f32> = (0..m * if id_mode { n_used } else { 1 } * n_in)
        .map(|i| ((i % 97) as f32) * 0.01)
        .collect();
    // MM_DENORM=1: fill x with denormals - real activations can carry them,
    // and denormal FMA inputs change the pipeline rate on some GPUs.
    let x: Vec<f32> = if std::env::var("MM_DENORM").is_ok_and(|v| v == "1") {
        vec![1e-42f32; m * n_in]
    } else {
        x
    };

    // MM_MMAP=1: back the weights by an mmap'd file wrapped no-copy, like
    // the engine's GGUF chunks, instead of a driver-allocated buffer.
    let _mmap;
    let w_buf = if std::env::var("MM_MMAP").is_ok_and(|v| v == "1") {
        let path = "/tmp/mmbench_weights.bin";
        std::fs::write(path, &w).unwrap();
        let file = std::fs::File::open(path).unwrap();
        let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let len = (map.len() + 16383) & !16383;
        let buf = unsafe {
            device.new_buffer_with_bytes_no_copy(
                map.as_ptr() as *const _,
                len as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        };
        _mmap = map;
        buf
    } else {
        device.new_buffer_with_data(
            w.as_ptr() as *const _,
            w_pool as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
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
    let kernel_list: Vec<(&str, u64)> = match fmt.as_str() {
        "q8_0" => vec![("mmll_q8_0", 32), ("mmllp_q8_0", 32), ("mm64ll_q8_0", 32)],
        "q5_k" => vec![("mmll_q5_k", 32), ("mm64ll_q5_k", 32), ("mmllp_q5_k", 32)],
        _ => vec![
            ("mmll_q4_k", 32u64),
            ("mm64ll_q4_k", 32),
            ("mmllr64_q4_k", 64),
            ("mmllp_q4_k", 32),
        ],
    };
    for (name, row_tile) in kernel_list {
        let fun = lib.get_function(name, None).unwrap();
        let pipe = device
            .new_compute_pipeline_state_with_function(&fun)
            .unwrap();
        let gx = (n_out as u64).div_ceil(64);
        let gy = (m as u64).div_ceil(row_tile);
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

    // --- mmid (grouped) mode ---
    if id_mode {
        let lib = device
            .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
            .unwrap();
        // [expert, row0, rows] per group; the pairs spread evenly over the
        // active experts (in-model: ACT ~= EXP, ~30 rows each at pp480).
        let rows_per = (m * n_used / n_act).max(1);        let mut table: Vec<[u32; 3]> = vec![[0, 0, 0]; n_exp];
        let mut row0 = 0u32;
        for e in 0..n_act {
            table[e] = [e as u32, row0, rows_per as u32];
            row0 += rows_per as u32;
        }
        let total_rows = row0 as usize;
        let tab_buf = device.new_buffer_with_data(
            table.as_ptr() as *const _,
            (n_exp * 12) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tok: Vec<u32> = (0..total_rows as u32).collect();
        let tok_buf = device.new_buffer_with_data(
            tok.as_ptr() as *const _,
            (total_rows * 4).max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let yid_buf = device.new_buffer(
            (total_rows.max(1) * n_out * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let id_kernel = std::env::var("MM_KERNEL").unwrap_or_else(|_| "mmllg_id_q4_k".into());
        let fun = lib.get_function(&id_kernel, None).unwrap();
        let pipe = device.new_compute_pipeline_state_with_function(&fun).unwrap();
        if std::env::var("MM_ID").is_ok_and(|v| v == "2") {
            // llama.cpp kernel_mul_mm_id_q4_K harness (MIT). Needs
            // /tmp/mm_id_llama.metal assembled from ggml-metal.metal - the
            // extraction never compiled (modern template plumbing) and the
            // question was closed by inspection instead: the kernels are
            // structurally identical. Kept for a future retry.
            let src = std::fs::read_to_string("/tmp/mm_id_llama.metal").unwrap();
            let llib = device
                .new_library_with_source(&src, &CompileOptions::new())
                .unwrap_or_else(|e| panic!("llama mm_id compile: {e}"));
            #[repr(C)]
            struct LlamaIdArgs {
                ne00: i32, ne02: i32, nb01: u64, nb02: u64, nb03: u64,
                ne11: i32, nb10: u64, nb11: u64, nb12: u64, nb13: u64,
                ne20: i32, ne21: i32, ne0: i32, ne1: i32, r2: i16, r3: i16,
            }
            // id = token index (ne20 = 1 flattens the used-slot dim).
            let n_tok = m;
            let mut tpe = vec![0u32; n_exp];
            let mut ids = vec![0i32; n_exp * n_tok];
            for e in 0..n_act {
                tpe[e] = rows_per as u32;
                for j in 0..rows_per.min(n_tok) {
                    ids[e * n_tok + j] = j as i32;
                }
            }
            let tpe_buf = device.new_buffer_with_data(
                tpe.as_ptr() as *const _, (n_exp * 4) as u64,
                MTLResourceOptions::StorageModeShared);
            let ids_buf = device.new_buffer_with_data(
                ids.as_ptr() as *const _, (n_exp * n_tok * 4) as u64,
                MTLResourceOptions::StorageModeShared);
            let dst_buf = device.new_buffer(
                (n_tok * n_out * 4) as u64, MTLResourceOptions::StorageModeShared);
            let largs = LlamaIdArgs {
                ne00: n_in as i32, ne02: n_exp as i32,
                nb01: row_bytes as u64, nb02: (n_out * row_bytes) as u64, nb03: 0,
                ne11: 1, nb10: 4, nb11: 4, nb12: (n_in * 4) as u64, nb13: 0,
                ne20: 1, ne21: n_tok as i32, ne0: n_out as i32, ne1: 1,
                r2: 1, r3: 1,
            };
            let bc = false;
            let consts = FunctionConstantValues::new();
            consts.set_constant_value_at_index(
                &bc as *const bool as *const _, MTLDataType::Bool, 700);
            let lfun = llib
                .get_function("llama_mul_mm_id_q4_K", Some(consts))
                .unwrap_or_else(|e| panic!("llama mm_id fc: {e}"));
            let lpipe = device.new_compute_pipeline_state_with_function(&lfun).unwrap();
            let t = time_kernel(
                &queue,
                |enc| {
                    enc.set_compute_pipeline_state(&lpipe);
                    enc.set_bytes(0, std::mem::size_of::<LlamaIdArgs>() as u64,
                        &largs as *const LlamaIdArgs as *const _);
                    enc.set_buffer(1, Some(&w_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&tpe_buf), 0);
                    enc.set_buffer(4, Some(&ids_buf), 0);
                    enc.set_buffer(5, Some(&dst_buf), 0);
                    enc.set_threadgroup_memory_length(0, 8192);
                    enc.dispatch_thread_groups(
                        MTLSize::new((n_tok as u64).div_ceil(32),
                            (n_out as u64).div_ceil(64), n_exp as u64),
                        MTLSize::new(128, 1, 1),
                    );
                },
                10,
            );
            let fl = 2.0 * (total_rows * n_in * n_out) as f64;
            println!(
                "llama mul_mm_id_q4_K act={n_act}/{n_exp}: {:.3} ms, {:.2} TFLOPS-effective",
                t * 1e3,
                fl / t / 1e12
            );
            return;
        }
        let ni = n_in as u32;
        let no = n_out as u32;
        let est = (n_in / 256 * 144) as u64 * n_out as u64; // per-expert bytes
        let zero_off = 0u64;
        let bstr = n_in as u32;
        let iter_no = std::cell::Cell::new(0u64);
        let t = time_kernel(
            &queue,
            |enc| {
                let wo = iter_no.get() * w_step as u64;
                iter_no.set(iter_no.get() + 1);
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&yid_buf), 0);
                enc.set_bytes(3, 4, &ni as *const u32 as *const _);
                enc.set_bytes(4, 4, &no as *const u32 as *const _);
                enc.set_bytes(5, 8, &wo as *const u64 as *const _);
                enc.set_buffer(6, Some(&tab_buf), 0);
                enc.set_bytes(7, 8, &est as *const u64 as *const _);
                enc.set_buffer(8, Some(&tok_buf), 0);
                enc.set_buffer(9, Some(&x_buf), 0);
                enc.set_buffer(10, Some(&w_buf), 0);
                enc.set_bytes(11, 8, &zero_off as *const u64 as *const _);
                enc.set_bytes(12, 4, &bstr as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new((n_out as u64).div_ceil(64), (m as u64).div_ceil(32), n_exp as u64),
                    MTLSize::new(128, 1, 1),
                );
            },
            10,
        );
        // FLOPs count only the active groups' real rows.
        let fl = 2.0 * (total_rows * n_in * n_out) as f64;
        println!(
            "ours {id_kernel} act={n_act}/{n_exp}: {:.3} ms, {:.2} TFLOPS-effective",
            t * 1e3,
            fl / t / 1e12
        );
        return;
    }

    // --- attention-buffer skeleton mode ---
    if struct_mode {
        let lib = device
            .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
            .unwrap();
        let fun = lib.get_function("mmll_q4_k", None).unwrap();
        let pipe = device.new_compute_pipeline_state_with_function(&fun).unwrap();
        let kv = n_out / 4;
        let row_b = row_bytes as u64;
        let (q_wo, k_wo, v_wo, o_wo) = (
            0u64,
            n_out as u64 * row_b,
            (n_out + kv) as u64 * row_b,
            (n_out + 2 * kv) as u64 * row_b,
        );
        let ys = device.new_buffer(
            (m * (2 * n_out + 2 * kv) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let (q_y, k_y, v_y, o_y) = (
            0u64,
            (m * n_out * 4) as u64,
            (m * (n_out + kv) * 4) as u64,
            (m * (n_out + 2 * kv) * 4) as u64,
        );
        let ni = n_in as u32;
        let no = n_out as u32;
        let nkv = kv as u32;
        let mm32 = m as u32;
        let t = time_kernel(
            &queue,
            |enc| {
                let mm = |enc: &ComputeCommandEncoderRef, no: u32, wo: u64, yo: u64| {
                    enc.set_compute_pipeline_state(&pipe);
                    enc.set_buffer(0, Some(&w_buf), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&ys), yo);
                    enc.set_bytes(3, 4, &ni as *const u32 as *const _);
                    enc.set_bytes(4, 4, &no as *const u32 as *const _);
                    enc.set_bytes(5, 8, &wo as *const u64 as *const _);
                    enc.set_bytes(6, 4, &mm32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((no as u64).div_ceil(64), (m as u64).div_ceil(32), 1),
                        MTLSize::new(128, 1, 1),
                    );
                };
                mm(enc, no, q_wo, q_y);
                mm(enc, nkv, k_wo, k_y);
                mm(enc, nkv, v_wo, v_y);
                enc.memory_barrier_with_resources(&[&ys]);
                mm(enc, no, o_wo, o_y);
            },
            1,
        );
        let fl = 2.0 * (m * n_in * (2 * n_out + 2 * kv)) as f64;
        println!(
            "struct attbuf-skeleton: {:.3} ms, {:.2} TFLOPS",
            t * 1e3,
            fl / t / 1e12
        );
        return;
    }

    // --- attention kernel check ---
    // MM_ATTN=1: attend_mm vs attend_rows_t8 on identical random data,
    // max abs/rel diff. Shapes: MM_M q rows, 32 q heads, 4 kv heads,
    // head_dim 128, positions MM_POS (default MM_M, causal from base 0).
    if std::env::var("MM_ATTN").is_ok_and(|v| v == "1") {
        let lib = device
            .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
            .unwrap();
        let n_heads = 32usize;
        let n_kv = 4usize;
        let hd = 128usize;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let n_pos = m; // attend over the full causal prefix
        let mut rng: u64 = 0x12345678;
        let mut rnd = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as u32) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let q: Vec<f32> = (0..m * q_dim).map(|_| rnd()).collect();
        // f32 -> f16 bits, engine-style (values here are small, no denorm).
        let f16 = |x: f32| -> u16 {
            let b = x.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
            if e <= 0 { return sign; }
            if e >= 31 { return sign | 0x7c00; }
            sign | ((e as u16) << 10) | (((b >> 13) & 0x3ff) as u16)
        };
        let f32_of = |h: u16| -> f32 {
            let sign = ((h & 0x8000) as u32) << 16;
            let e = ((h >> 10) & 0x1f) as i32;
            let m = (h & 0x3ff) as u32;
            if e == 0 {
                return f32::from_bits(sign) * 0.0; // flush denormals; unused here
            }
            f32::from_bits(sign | (((e - 15 + 127) as u32) << 23) | (m << 13))
        };
        let kv: Vec<u16> = (0..2 * n_pos * kv_dim).map(|_| f16(rnd())).collect();
        let k_buf = device.new_buffer_with_data(
            kv.as_ptr() as *const _, (n_pos * kv_dim * 2) as u64, MTLResourceOptions::StorageModeShared);
        let v_buf = device.new_buffer_with_data(
            unsafe { kv.as_ptr().add(n_pos * kv_dim) } as *const _,
            (n_pos * kv_dim * 2) as u64, MTLResourceOptions::StorageModeShared);
        let q_buf = device.new_buffer_with_data(
            q.as_ptr() as *const _, (m * q_dim * 4) as u64, MTLResourceOptions::StorageModeShared);
        let out_a = device.new_buffer((m * q_dim * 4) as u64, MTLResourceOptions::StorageModeShared);
        let out_b = device.new_buffer((m * q_dim * 4) as u64, MTLResourceOptions::StorageModeShared);
        let scale: f32 = 1.0 / (hd as f32).sqrt();
        let mut run = |name: &str, out: &Buffer, grid_y: u64| {
            let fun = lib.get_function(name, None).unwrap();
            let pipe = device.new_compute_pipeline_state_with_function(&fun).unwrap();
            let cmd = queue.new_command_buffer();
            let enc = cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&k_buf), 0);
            enc.set_buffer(1, Some(&v_buf), 0);
            enc.set_buffer(2, Some(&q_buf), 0);
            enc.set_buffer(3, Some(out), 0);
            for (index, value) in [
                (4u64, kv_dim as u32), (5, hd as u32), (6, 0u32), (7, (n_heads / n_kv) as u32),
            ] {
                enc.set_bytes(index, 4, &value as *const u32 as *const _);
            }
            enc.set_bytes(8, 4, &scale as *const f32 as *const _);
            for (index, value) in [(9u64, 0u64), (10, 0u64)] {
                enc.set_bytes(index, 8, &value as *const u64 as *const _);
            }
            enc.set_bytes(11, 4, &(n_heads as u32) as *const u32 as *const _);
            enc.set_bytes(12, 4, &(m as u32) as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n_heads as u64, grid_y, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };
        run("attend_rows_t8", &out_a, (m as u64).div_ceil(8));
        run("attend_mm", &out_b, (m as u64).div_ceil(8));
        // CPU reference for a few (row, head) pairs.
        let cpu_at = |row: usize, head: usize, dim: usize| -> f32 {
            let kvh = head / (n_heads / n_kv);
            let valid = row + 1;
            let mut scores = Vec::with_capacity(valid);
            let mut mx = f32::NEG_INFINITY;
            for t in 0..valid {
                let mut s = 0f32;
                for d in 0..hd {
                    let kb = unsafe { *kv.as_ptr().add(t * kv_dim + kvh * hd + d) };
                    s += q[row * q_dim + head * hd + d] * f32_of(kb);
                }
                scores.push(s);
                mx = mx.max(s);
            }
            let mut num = 0f32;
            let mut den = 0f32;
            for (t, s) in scores.iter().enumerate() {
                let p = ((s - mx) * scale).exp();
                let vb = unsafe { *kv.as_ptr().add((n_pos + t) * kv_dim + kvh * hd + dim) };
                num += p * f32_of(vb);
                den += p;
            }
            num / den
        };
        let a: &[f32] = unsafe {
            std::slice::from_raw_parts(out_a.contents() as *const f32, m * q_dim)
        };
        let b: &[f32] = unsafe {
            std::slice::from_raw_parts(out_b.contents() as *const f32, m * q_dim)
        };
        let mut max_abs = 0f32;
        let mut bad = 0usize;
        let mut bad_head = [0usize; 32];
        let mut bad_row = [0usize; 64];
        let mut bad_dimblk = [0usize; 16];
        let mut bad_rowmod = [0usize; 8];
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            if d > max_abs { max_abs = d; }
            if !y.is_finite() || d > 0.02 {
                let row = i / q_dim;
                let head = (i % q_dim) / hd;
                if bad < 8 {
                    println!(
                        "  bad (row {row}, head {head}, dim {}): cpu={:.5} t8={x:.5} mm={y:.5}",
                        i % hd,
                        cpu_at(row, head, i % hd)
                    );
                }
                bad_head[head] += 1;
                bad_row[row % 64] += 1;
                bad_dimblk[(i % hd) / 8] += 1;
                bad_rowmod[row % 8] += 1;
                bad += 1;
            }
        }
        println!("attend_mm vs t8: max_abs={max_abs:.5} bad={bad}/{}", a.len());
        println!("bad/head: {:?}", bad_head);
        println!("bad/dimblk: {:?}", bad_dimblk);
        println!("bad/row%8: {:?}", bad_rowmod);
        // Spot-check both kernels against the CPU reference.
        for &(row, head) in &[(0usize, 0usize), (0, 2), (4, 0), (7, 31), (33, 5)] {
            if row >= m { continue; }
            let dim = 3;
            let i = row * q_dim + head * hd + dim;
            println!(
                "  (row {row}, head {head}, dim {dim}): cpu={:.5} t8={:.5} mm={:.5}",
                cpu_at(row, head, dim), a[i], b[i]
            );
        }
        // Timings at the same shape (single dispatch per buffer).
        for (name, gy) in [("attend_rows_t8", (m as u64).div_ceil(8)), ("attend_mm", (m as u64).div_ceil(8))] {
            let fun = lib.get_function(name, None).unwrap();
            let pipe = device.new_compute_pipeline_state_with_function(&fun).unwrap();
            let t = time_kernel(
                &queue,
                |enc| {
                    enc.set_compute_pipeline_state(&pipe);
                    enc.set_buffer(0, Some(&k_buf), 0);
                    enc.set_buffer(1, Some(&v_buf), 0);
                    enc.set_buffer(2, Some(&q_buf), 0);
                    enc.set_buffer(3, Some(&out_a), 0);
                    for (index, value) in [
                        (4u64, kv_dim as u32), (5, hd as u32), (6, 0u32), (7, (n_heads / n_kv) as u32),
                    ] {
                        enc.set_bytes(index, 4, &value as *const u32 as *const _);
                    }
                    enc.set_bytes(8, 4, &scale as *const f32 as *const _);
                    for (index, value) in [(9u64, 0u64), (10, 0u64)] {
                        enc.set_bytes(index, 8, &value as *const u64 as *const _);
                    }
                    enc.set_bytes(11, 4, &(n_heads as u32) as *const u32 as *const _);
                    enc.set_bytes(12, 4, &(m as u32) as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(n_heads as u64, gy, 1), MTLSize::new(128, 1, 1));
                },
                10,
            );
            println!("{name}: {:.3} ms/dispatch", t * 1e3);
        }
        return;
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
