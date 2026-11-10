//! Tail-store synchronization against the conservative threadgroup-barrier
//! shader. Uneven/empty expert groups, output-column tails and grid.y=1 force
//! scratch reuse across row tiles, where a missing boundary barrier races.
#![cfg(target_os = "macos")]
use metal::*;

#[test]
fn grouped_tail_stores_match_conservative_barriers() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let source = allpaka_backend::gpu::KERNELS;
    let reference = source.replace(
        "simdgroup_barrier(mem_flags::mem_threadgroup);",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
    );
    assert_ne!(source, reference);
    let libs = [reference.as_str(), source].map(|s| {
        device
            .new_library_with_source(s, &CompileOptions::new())
            .unwrap()
    });
    let groups: [[u32; 3]; 5] = [[0, 0, 0], [1, 0, 1], [2, 1, 31], [3, 32, 33], [4, 65, 65]];
    let rows = 130usize;
    let ni = 256u32;
    let no = 65u32;
    let x: Vec<f32> = (0..rows * ni as usize)
        .map(|i| (i % 97) as f32 * 0.003 - 0.1)
        .collect();
    let tok: Vec<u32> = (0..rows as u32)
        .map(|i| (i * 17 + 3) % rows as u32)
        .collect();
    let buffer = |p, n| device.new_buffer_with_data(p, n, MTLResourceOptions::StorageModeShared);
    let xb = buffer(x.as_ptr().cast(), (x.len() * 4) as u64);
    let tb = buffer(
        groups.as_ptr().cast(),
        std::mem::size_of_val(&groups) as u64,
    );
    let ib = buffer(tok.as_ptr().cast(), (tok.len() * 4) as u64);
    let zero = 0u64;
    for (suffix, block_bytes, scales) in [
        ("q8_0", 34, vec![0]),
        ("q2_k", 84, vec![80, 82]),
        ("q3_k", 110, vec![108]),
        ("q4_k", 144, vec![0, 2]),
        ("q5_k", 176, vec![0, 2]),
        ("q6_k", 210, vec![208]),
    ] {
        let blocks = if suffix == "q8_0" { 8 } else { 1 };
        let stride = no as usize * blocks * block_bytes;
        let mut weights = vec![0u8; stride * groups.len()];
        for (i, v) in weights.iter_mut().enumerate() {
            *v = (i.wrapping_mul(73) + i / 127) as u8;
        }
        for b in weights.chunks_exact_mut(block_bytes) {
            for &offset in &scales {
                b[offset..offset + 2].copy_from_slice(&0x211fu16.to_le_bytes());
            }
        }
        let wb = buffer(weights.as_ptr().cast(), weights.len() as u64);
        let kernels = vec![
            format!("mmll_id_{suffix}"),
            format!("mmllg_id_{suffix}"),
            format!("mmlls_id_{suffix}"),
            format!("mmllpg_id_{suffix}"),
            format!("mmllps_id_{suffix}"),
            format!("mmll_id_{suffix}_bn16"),
            format!("mmllg_id_{suffix}_bn16"),
            format!("mmlls_id_{suffix}_bn16"),
            format!("mmllpg_id_{suffix}_bn16"),
            format!("mmllps_id_{suffix}_bn16"),
        ];
        for name in kernels {
            let outputs = libs.each_ref().map(|_| {
                device.new_buffer(
                    ((rows * no as usize + 64) * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            });
            for (lib, out) in libs.iter().zip(&outputs) {
                let fun = lib.get_function(&name, None).unwrap();
                let pipe = device
                    .new_compute_pipeline_state_with_function(&fun)
                    .unwrap();
                unsafe {
                    std::slice::from_raw_parts_mut(
                        out.contents().cast::<f32>(),
                        rows * no as usize + 64,
                    )
                    .fill(-12345.0);
                }
                let cmd = queue.new_command_buffer();
                let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&wb), 0);
                enc.set_buffer(1, Some(&xb), 0);
                enc.set_buffer(2, Some(out), 0);
                enc.set_bytes(3, 4, (&ni as *const u32).cast());
                enc.set_bytes(4, 4, (&no as *const u32).cast());
                enc.set_bytes(5, 8, (&zero as *const u64).cast());
                enc.set_buffer(6, Some(&tb), 0);
                let stride = stride as u64;
                enc.set_bytes(7, 8, (&stride as *const u64).cast());
                enc.set_buffer(8, Some(&ib), 0);
                enc.set_buffer(9, Some(&xb), 0);
                enc.set_buffer(10, Some(&wb), 0);
                enc.set_bytes(11, 8, (&zero as *const u64).cast());
                enc.set_bytes(12, 4, (&ni as *const u32).cast());
                for _ in 0..8 {
                    enc.dispatch_thread_groups(
                        MTLSize::new(2, 1, groups.len() as u64),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "{name}");
            }
            let output = |i: usize| unsafe {
                std::slice::from_raw_parts(
                    outputs[i].contents().cast::<f32>(),
                    rows * no as usize + 64,
                )
            };
            let (reference, actual) = (output(0), output(1));
            for (i, (&a, &b)) in reference.iter().zip(actual).enumerate() {
                assert!(
                    a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                    "{name}[{i}]: {a} != {b}"
                );
                if i >= rows * no as usize {
                    assert_eq!(b, -12345.0, "{name} wrote beyond output");
                } else {
                    assert_ne!(b, -12345.0, "{name} missed output {i}");
                }
            }
        }

        // Verify BN=16 kernels match BN=32 kernels bit-for-bit across all
        // full-tile (m=16, m=32) and tail-tile (m=1, m=31, m=33, m=65) groups.
        let bn16_pairs = vec![
            (
                format!("mmll_id_{suffix}"),
                format!("mmll_id_{suffix}_bn16"),
            ),
            (
                format!("mmllg_id_{suffix}"),
                format!("mmllg_id_{suffix}_bn16"),
            ),
            (
                format!("mmlls_id_{suffix}"),
                format!("mmlls_id_{suffix}_bn16"),
            ),
            (
                format!("mmllpg_id_{suffix}"),
                format!("mmllpg_id_{suffix}_bn16"),
            ),
            (
                format!("mmllps_id_{suffix}"),
                format!("mmllps_id_{suffix}_bn16"),
            ),
        ];
        for (k32, k16) in bn16_pairs {
            let outputs = [&k32, &k16].map(|kname| {
                let out = device.new_buffer(
                    ((rows * no as usize + 64) * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                unsafe {
                    std::slice::from_raw_parts_mut(
                        out.contents().cast::<f32>(),
                        rows * no as usize + 64,
                    )
                    .fill(-12345.0);
                }
                let fun = libs[1].get_function(kname, None).unwrap();
                let pipe = device
                    .new_compute_pipeline_state_with_function(&fun)
                    .unwrap();
                let cmd = queue.new_command_buffer();
                let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&wb), 0);
                enc.set_buffer(1, Some(&xb), 0);
                enc.set_buffer(2, Some(&out), 0);
                enc.set_bytes(3, 4, (&ni as *const u32).cast());
                enc.set_bytes(4, 4, (&no as *const u32).cast());
                enc.set_bytes(5, 8, (&zero as *const u64).cast());
                enc.set_buffer(6, Some(&tb), 0);
                let stride = stride as u64;
                enc.set_bytes(7, 8, (&stride as *const u64).cast());
                enc.set_buffer(8, Some(&ib), 0);
                enc.set_buffer(9, Some(&xb), 0);
                enc.set_buffer(10, Some(&wb), 0);
                enc.set_bytes(11, 8, (&zero as *const u64).cast());
                enc.set_bytes(12, 4, (&ni as *const u32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(2, 2, groups.len() as u64),
                    MTLSize::new(128, 1, 1),
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "{kname}");
                out
            });
            let out32 = unsafe {
                std::slice::from_raw_parts(
                    outputs[0].contents().cast::<f32>(),
                    rows * no as usize + 64,
                )
            };
            let out16 = unsafe {
                std::slice::from_raw_parts(
                    outputs[1].contents().cast::<f32>(),
                    rows * no as usize + 64,
                )
            };
            for (i, (&a, &b)) in out32.iter().zip(out16).enumerate() {
                assert!(
                    a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                    "{k16} vs {k32} [{i}]: {b} != {a}"
                );
            }
        }
    }
}

#[test]
fn fused_qkv_kernels_match_separate_dispatches() {
    #[repr(C)]
    struct MmQkvArgs {
        n_in: u32,
        n_q: u32,
        n_k: u32,
        n_v: u32,
        wq_off: u64,
        wk_off: u64,
        wv_off: u64,
        q_off: u64,
        k_off: u64,
        v_off: u64,
        m: u32,
        pad: u32,
    }

    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();

    let ni = 256u32;
    let n_q = 128u32;
    let n_k = 64u32;
    let n_v = 64u32;
    let m = 45u32; // exercises both full 32-row tile and 13-row tail tile
    let x: Vec<f32> = (0..(m * ni) as usize)
        .map(|i| (i % 89) as f32 * 0.004 - 0.15)
        .collect();
    let buffer = |p, n| device.new_buffer_with_data(p, n, MTLResourceOptions::StorageModeShared);
    let xb = buffer(x.as_ptr().cast(), (x.len() * 4) as u64);

    for (single_name, qkv_name, block_bytes, blocks, scales) in [
        ("mmll_q8_0", "mmll_qkv_q8_0", 34usize, 8usize, vec![0usize]),
        ("mmllp_q8_0", "mmllp_qkv_q8_0", 34, 8, vec![0]),
        ("mmll_q2_k", "mmll_qkv_q2_k", 84, 1, vec![80, 82]),
        ("mmll_q2_k", "mmllp_qkv_q2_k", 84, 1, vec![80, 82]),
        ("mmllp_q2_k", "mmllp_qkv_q2_k", 84, 1, vec![80, 82]),
        ("mmll_q3_k", "mmll_qkv_q3_k", 110, 1, vec![108]),
        ("mmll_q3_k", "mmllp_qkv_q3_k", 110, 1, vec![108]),
        ("mmllp_q3_k", "mmllp_qkv_q3_k", 110, 1, vec![108]),
        ("mmll_q4_k", "mmll_qkv_q4_k", 144, 1, vec![0, 2]),
        ("mmllp_q4_k", "mmllp_qkv_q4_k", 144, 1, vec![0, 2]),
        ("mmll_q5_k", "mmll_qkv_q5_k", 176, 1, vec![0, 2]),
        ("mmll_q5_k", "mmllp_qkv_q5_k", 176, 1, vec![0, 2]),
        ("mmllp_q5_k", "mmllp_qkv_q5_k", 176, 1, vec![0, 2]),
        ("mmll_q6_k", "mmll_qkv_q6_k", 210, 1, vec![208]),
        ("mmll_q6_k", "mmllp_qkv_q6_k", 210, 1, vec![208]),
        ("mmllp_q6_k", "mmllp_qkv_q6_k", 210, 1, vec![208]),
    ] {
        let total_cols = (n_q + n_k + n_v) as usize;
        let row_bytes = blocks * block_bytes;
        let mut weights = vec![0u8; total_cols * row_bytes];
        for (i, v) in weights.iter_mut().enumerate() {
            *v = (i.wrapping_mul(59) + i / 97) as u8;
        }
        for b in weights.chunks_exact_mut(block_bytes) {
            for &offset in &scales {
                b[offset..offset + 2].copy_from_slice(&0x211fu16.to_le_bytes());
            }
        }
        let wb = buffer(weights.as_ptr().cast(), weights.len() as u64);
        let wq_off = 0u64;
        let wk_off = (n_q as usize * row_bytes) as u64;
        let wv_off = ((n_q + n_k) as usize * row_bytes) as u64;

        let q_elems = (m * n_q) as usize;
        let k_elems = (m * n_k) as usize;
        let v_elems = (m * n_v) as usize;
        let total_out = q_elems + k_elems + v_elems;

        let ref_out = device.new_buffer(
            (total_out * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let fused_out = device.new_buffer(
            (total_out * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let f_single = lib.get_function(single_name, None).unwrap();
        let p_single = device
            .new_compute_pipeline_state_with_function(&f_single)
            .unwrap();
        let f_qkv = lib.get_function(qkv_name, None).unwrap();
        let p_qkv = device
            .new_compute_pipeline_state_with_function(&f_qkv)
            .unwrap();

        let cmd = queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
        for (n_out, w_off, y_off) in [
            (n_q, wq_off, 0u64),
            (n_k, wk_off, (q_elems * 4) as u64),
            (n_v, wv_off, ((q_elems + k_elems) * 4) as u64),
        ] {
            enc.set_compute_pipeline_state(&p_single);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ref_out), y_off);
            enc.set_bytes(3, 4, (&ni as *const u32).cast());
            enc.set_bytes(4, 4, (&n_out as *const u32).cast());
            enc.set_bytes(5, 8, (&w_off as *const u64).cast());
            enc.set_bytes(6, 4, (&m as *const u32).cast());
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64).div_ceil(64), (m as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );
        }

        let args = MmQkvArgs {
            n_in: ni,
            n_q,
            n_k,
            n_v,
            wq_off,
            wk_off,
            wv_off,
            q_off: 0,
            k_off: (q_elems * 4) as u64,
            v_off: ((q_elems + k_elems) * 4) as u64,
            m,
            pad: 0,
        };
        enc.set_compute_pipeline_state(&p_qkv);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&wb), 0);
        enc.set_buffer(2, Some(&wb), 0);
        enc.set_buffer(3, Some(&xb), 0);
        enc.set_buffer(4, Some(&fused_out), 0);
        enc.set_bytes(
            5,
            std::mem::size_of::<MmQkvArgs>() as u64,
            (&args as *const MmQkvArgs).cast(),
        );
        enc.dispatch_thread_groups(
            MTLSize::new(
                ((n_q + n_k + n_v) as u64).div_ceil(64),
                (m as u64).div_ceil(32),
                1,
            ),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(
            cmd.status(),
            MTLCommandBufferStatus::Completed,
            "{qkv_name}"
        );

        let s_ref =
            unsafe { std::slice::from_raw_parts(ref_out.contents().cast::<f32>(), total_out) };
        let s_fused =
            unsafe { std::slice::from_raw_parts(fused_out.contents().cast::<f32>(), total_out) };
        for (i, (&a, &b)) in s_ref.iter().zip(s_fused).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "{qkv_name}[{i}]: {b} != {a}"
            );
        }
    }
}

/// Mixed-format fused QKV: q/k on Q4_K, v on Q6_K (Qwen3-8B's odd layers).
/// Bit-exact against three separate pipelined single-matrix dispatches.
#[test]
fn fused_qkv_mixed_q4k_q6k_matches_separate() {
    #[repr(C)]
    struct MmQkvArgs {
        n_in: u32,
        n_q: u32,
        n_k: u32,
        n_v: u32,
        wq_off: u64,
        wk_off: u64,
        wv_off: u64,
        q_off: u64,
        k_off: u64,
        v_off: u64,
        m: u32,
        pad: u32,
    }

    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();

    let ni = 256u32;
    let n_q = 128u32;
    let n_k = 64u32;
    let n_v = 64u32;
    let m = 45u32; // full 32-row tile + 13-row tail
    let x: Vec<f32> = (0..(m * ni) as usize)
        .map(|i| (i % 89) as f32 * 0.004 - 0.15)
        .collect();
    let buffer = |p, n| device.new_buffer_with_data(p, n, MTLResourceOptions::StorageModeShared);
    let xb = buffer(x.as_ptr().cast(), (x.len() * 4) as u64);

    // One 256-value block per row: q4_k rows are 144 B, q6_k rows 210 B.
    let qk_bytes = 144usize;
    let v_bytes = 210usize;
    let wq_off = 0u64;
    let wk_off = n_q as u64 * qk_bytes as u64;
    let wv_off = wk_off + n_k as u64 * qk_bytes as u64;
    let total_w = wv_off as usize + n_v as usize * v_bytes;
    let mut weights = vec![0u8; total_w];
    for (i, v) in weights.iter_mut().enumerate() {
        *v = (i.wrapping_mul(59) + i / 97) as u8;
    }
    // Sane f16 scales so the outputs stay finite: d/dmin at 0/2 in q4_k,
    // d at 208 in q6_k.
    for r in 0..(n_q + n_k) as usize {
        let b = wq_off as usize + r * qk_bytes;
        weights[b..b + 2].copy_from_slice(&0x211fu16.to_le_bytes());
        weights[b + 2..b + 4].copy_from_slice(&0x211fu16.to_le_bytes());
    }
    for r in 0..n_v as usize {
        let b = wv_off as usize + r * v_bytes;
        weights[b + 208..b + 210].copy_from_slice(&0x211fu16.to_le_bytes());
    }
    let wb = buffer(weights.as_ptr().cast(), weights.len() as u64);

    let q_elems = (m * n_q) as usize;
    let k_elems = (m * n_k) as usize;
    let v_elems = (m * n_v) as usize;
    let total_out = q_elems + k_elems + v_elems;
    let ref_out = device.new_buffer(
        (total_out * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let fused_out = device.new_buffer(
        (total_out * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
    for (name, n_out, w_off, y_off) in [
        ("mmllp_q4_k", n_q, wq_off, 0u64),
        ("mmllp_q4_k", n_k, wk_off, (q_elems * 4) as u64),
        ("mmllp_q6_k", n_v, wv_off, ((q_elems + k_elems) * 4) as u64),
    ] {
        let f = lib.get_function(name, None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&xb), 0);
        enc.set_buffer(2, Some(&ref_out), y_off);
        enc.set_bytes(3, 4, (&ni as *const u32).cast());
        enc.set_bytes(4, 4, (&n_out as *const u32).cast());
        enc.set_bytes(5, 8, (&w_off as *const u64).cast());
        enc.set_bytes(6, 4, (&m as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new((n_out as u64).div_ceil(64), (m as u64).div_ceil(32), 1),
            MTLSize::new(128, 1, 1),
        );
    }

    let args = MmQkvArgs {
        n_in: ni,
        n_q,
        n_k,
        n_v,
        wq_off,
        wk_off,
        wv_off,
        q_off: 0,
        k_off: (q_elems * 4) as u64,
        v_off: ((q_elems + k_elems) * 4) as u64,
        m,
        pad: 0,
    };
    let f_mixed = lib.get_function("mmllp_qkv_q4k_q6k", None).unwrap();
    let p_mixed = device
        .new_compute_pipeline_state_with_function(&f_mixed)
        .unwrap();
    enc.set_compute_pipeline_state(&p_mixed);
    enc.set_buffer(0, Some(&wb), 0);
    enc.set_buffer(1, Some(&wb), 0);
    enc.set_buffer(2, Some(&wb), 0);
    enc.set_buffer(3, Some(&xb), 0);
    enc.set_buffer(4, Some(&fused_out), 0);
    enc.set_bytes(
        5,
        std::mem::size_of::<MmQkvArgs>() as u64,
        (&args as *const MmQkvArgs).cast(),
    );
    enc.dispatch_thread_groups(
        MTLSize::new(
            ((n_q + n_k + n_v) as u64).div_ceil(64),
            (m as u64).div_ceil(32),
            1,
        ),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);

    let s_ref = unsafe { std::slice::from_raw_parts(ref_out.contents().cast::<f32>(), total_out) };
    let s_fused =
        unsafe { std::slice::from_raw_parts(fused_out.contents().cast::<f32>(), total_out) };
    for (i, (&a, &b)) in s_ref.iter().zip(s_fused).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
            "mmllp_qkv_q4k_q6k[{i}]: {b} != {a}"
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RouteParams {
    m: u32,
    n_expert: u32,
    n_used: u32,
    norm: u32,
    has_bias: u32,
    sigmoid: u32,
    shared: u32,
    scale: f32,
    total_rows: u32,
    tile_bound: u32,
    tile_w: u32,
}

#[test]
fn route_fused_matches_three_stage_router() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .expect("KERNELS failed to compile");
    let queue = device.new_command_queue();

    let pipeline = |name: &str| {
        let f = lib.get_function(name, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };
    let buf = |data: &[f32]| {
        device.new_buffer_with_data(
            data.as_ptr().cast(),
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let zbuf = |bytes: u64| {
        let b = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(b.contents().cast::<u8>(), 0, bytes as usize) };
        b
    };

    let p_pick = pipeline("route_pick");
    let p_scan = pipeline("route_scan");
    let p_scat = pipeline("route_scatter");
    let p_fused = pipeline("route_fused");

    let m = 480u32;
    let n_expert = 128u32;
    let n_used = 8u32;
    let shared = 1u32;
    let total_rows = m * n_used;
    let stride = n_used + shared;
    let hit_len = (m * stride) as usize;

    let mut logits = vec![0.0f32; (m * n_expert) as usize];
    for (i, v) in logits.iter_mut().enumerate() {
        *v = (((i * 73 + 19) % 997) as f32 - 498.0) * 0.01;
    }
    let bias = vec![0.0f32; n_expert as usize];
    let params = RouteParams {
        m,
        n_expert,
        n_used,
        norm: 1,
        has_bias: 0,
        sigmoid: 0,
        shared,
        scale: 1.0,
        total_rows,
        tile_bound: n_expert + total_rows / 32,
        tile_w: 32,
    };

    let lg_b = buf(&logits);
    let bias_b = buf(&bias);

    let counts_a = zbuf((n_expert * 4) as u64);
    let counters_a = zbuf((n_expert * 4) as u64);
    let picks_a = zbuf((total_rows * 4) as u64);
    let pickw_a = zbuf((total_rows * 4) as u64);
    let table_a = zbuf((n_expert * 12) as u64);
    let tok_a = zbuf((total_rows * 4) as u64);
    let hit_row_a = zbuf((hit_len * 4) as u64);
    let hit_w_a = zbuf((hit_len * 4) as u64);
    let tok_off_a = zbuf(((m + 1) * 4) as u64);

    let picks_b = zbuf((total_rows * 4) as u64);
    let pickw_b = zbuf((total_rows * 4) as u64);
    let table_b = zbuf((n_expert * 12) as u64);
    let tok_b = zbuf((total_rows * 4) as u64);
    let hit_row_b = zbuf((hit_len * 4) as u64);
    let hit_w_b = zbuf((hit_len * 4) as u64);
    let tok_off_b = zbuf(((m + 1) * 4) as u64);
    let tiles_b = zbuf(((n_expert + total_rows / 32) * 12) as u64);

    let psz = std::mem::size_of::<RouteParams>() as u64;
    let pp = (&params as *const RouteParams).cast();
    let tgs = (m as u64).div_ceil(256);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    enc.set_compute_pipeline_state(&p_pick);
    enc.set_buffer(0, Some(&lg_b), 0);
    enc.set_buffer(1, Some(&bias_b), 0);
    enc.set_buffer(2, Some(&counts_a), 0);
    enc.set_buffer(3, Some(&picks_a), 0);
    enc.set_buffer(4, Some(&pickw_a), 0);
    enc.set_bytes(5, psz, pp);
    enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
    enc.memory_barrier_with_resources(&[&counts_a, &picks_a, &pickw_a]);

    enc.set_compute_pipeline_state(&p_scan);
    enc.set_buffer(0, Some(&counts_a), 0);
    enc.set_buffer(1, Some(&table_a), 0);
    enc.set_buffer(2, Some(&counters_a), 0);
    enc.set_bytes(3, 4, (&n_expert as *const u32).cast());
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
    enc.memory_barrier_with_resources(&[&table_a, &counters_a]);

    enc.set_compute_pipeline_state(&p_scat);
    enc.set_buffer(0, Some(&picks_a), 0);
    enc.set_buffer(1, Some(&pickw_a), 0);
    enc.set_buffer(2, Some(&table_a), 0);
    enc.set_buffer(3, Some(&counters_a), 0);
    enc.set_buffer(4, Some(&tok_a), 0);
    enc.set_buffer(5, Some(&hit_row_a), 0);
    enc.set_buffer(6, Some(&hit_w_a), 0);
    enc.set_buffer(7, Some(&tok_off_a), 0);
    enc.set_bytes(8, psz, pp);
    enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));

    enc.set_compute_pipeline_state(&p_fused);
    enc.set_buffer(0, Some(&lg_b), 0);
    enc.set_buffer(1, Some(&bias_b), 0);
    enc.set_buffer(2, Some(&picks_b), 0);
    enc.set_buffer(3, Some(&pickw_b), 0);
    enc.set_buffer(4, Some(&table_b), 0);
    enc.set_buffer(5, Some(&tok_b), 0);
    enc.set_buffer(6, Some(&hit_row_b), 0);
    enc.set_buffer(7, Some(&hit_w_b), 0);
    enc.set_buffer(8, Some(&tok_off_b), 0);
    enc.set_bytes(9, psz, pp);
    enc.set_buffer(10, Some(&tiles_b), 0);
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);

    let tab_a = unsafe {
        std::slice::from_raw_parts(table_a.contents().cast::<[u32; 3]>(), n_expert as usize)
    };
    let tab_b = unsafe {
        std::slice::from_raw_parts(table_b.contents().cast::<[u32; 3]>(), n_expert as usize)
    };
    assert_eq!(tab_a, tab_b, "group tables differ");

    let off_a =
        unsafe { std::slice::from_raw_parts(tok_off_a.contents().cast::<u32>(), (m + 1) as usize) };
    let off_b =
        unsafe { std::slice::from_raw_parts(tok_off_b.contents().cast::<u32>(), (m + 1) as usize) };
    assert_eq!(off_a, off_b, "tok_off differs");

    let hw_a = unsafe { std::slice::from_raw_parts(hit_w_a.contents().cast::<f32>(), hit_len) };
    let hw_b = unsafe { std::slice::from_raw_parts(hit_w_b.contents().cast::<f32>(), hit_len) };
    for (i, (&a, &b)) in hw_a.iter().zip(hw_b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "hit_w[{i}] differs");
    }

    let tk_a =
        unsafe { std::slice::from_raw_parts(tok_a.contents().cast::<u32>(), total_rows as usize) };
    let tk_b =
        unsafe { std::slice::from_raw_parts(tok_b.contents().cast::<u32>(), total_rows as usize) };
    let hr_a = unsafe { std::slice::from_raw_parts(hit_row_a.contents().cast::<u32>(), hit_len) };
    let hr_b = unsafe { std::slice::from_raw_parts(hit_row_b.contents().cast::<u32>(), hit_len) };

    for (e, entry) in tab_b.iter().enumerate().take(n_expert as usize) {
        let row0 = entry[1] as usize;
        let rows = entry[2] as usize;
        let mut s_a = tk_a[row0..row0 + rows].to_vec();
        let mut s_b = tk_b[row0..row0 + rows].to_vec();
        s_a.sort_unstable();
        s_b.sort_unstable();
        assert_eq!(s_a, s_b, "expert {e} token set differs");
    }

    for t in 0..m as usize {
        let base = t * stride as usize;
        for i in 0..n_used as usize {
            let r_a = hr_a[base + i] as usize;
            let r_b = hr_b[base + i] as usize;
            assert_eq!(tk_a[r_a], t as u32);
            assert_eq!(tk_b[r_b], t as u32);
        }
        assert_eq!(hr_a[base + n_used as usize], hr_b[base + n_used as usize]);
    }

    // Tile descriptors: exact cover of each expert's rows in 32-row chunks,
    // experts in order, then a rows=0 tail up to tile_bound.
    let tile_bound = (n_expert + total_rows / 32) as usize;
    let tiles =
        unsafe { std::slice::from_raw_parts(tiles_b.contents().cast::<[u32; 3]>(), tile_bound) };
    let mut t = 0usize;
    for (e, entry) in tab_b.iter().enumerate().take(n_expert as usize) {
        let row0 = entry[1];
        let rows = entry[2];
        let n_chunks = rows.div_ceil(32) as usize;
        for k in 0..n_chunks {
            assert_eq!(tiles[t][0], e as u32, "tile {t} expert");
            assert_eq!(tiles[t][1], row0 + 32 * k as u32, "tile {t} row0");
            assert_eq!(tiles[t][2], (rows - 32 * k as u32).min(32), "tile {t} rows");
            t += 1;
        }
    }
    for (i, tile) in tiles.iter().enumerate().skip(t) {
        assert_eq!(tile[2], 0, "stale tile {i} resurrected work");
    }
}

/// SWIGLU_X on the Q6_K matvecs (decode SWFUSE for Q4_K_M dense down):
/// the fused silu(gate)*up inside the down matvec must be bit-identical to
/// the standalone swiglu dispatch followed by the plain matvec.
#[test]
fn matvec_q6k_swiglu_fold_matches_separate() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();

    let n_in = 512u32; // two 256-value q6_k blocks per row
    let n_out = 64u32;
    let mut weights = vec![0u8; n_out as usize * (n_in as usize / 256) * 210];
    for (i, v) in weights.iter_mut().enumerate() {
        *v = (i.wrapping_mul(59) + i / 97) as u8;
    }
    for b in weights.chunks_exact_mut(210) {
        b[208..210].copy_from_slice(&0x211fu16.to_le_bytes());
    }
    let gate: Vec<f32> = (0..n_in as usize)
        .map(|i| (i % 89) as f32 * 0.02 - 0.9)
        .collect();
    let up: Vec<f32> = (0..n_in as usize)
        .map(|i| ((i * 7) % 83) as f32 * 0.017 - 0.7)
        .collect();
    let buffer = |p, n| device.new_buffer_with_data(p, n, MTLResourceOptions::StorageModeShared);
    let wb = buffer(weights.as_ptr().cast(), weights.len() as u64);
    let gb = buffer(gate.as_ptr().cast(), (n_in * 4) as u64);
    let ub = buffer(up.as_ptr().cast(), (n_in * 4) as u64);
    let y_ref = device.new_buffer((n_out * 4) as u64, MTLResourceOptions::StorageModeShared);
    let y_fused = device.new_buffer((n_out * 4) as u64, MTLResourceOptions::StorageModeShared);
    let idx_zero = [0u8; 32];
    let w_off = 0u64;

    let pipeline = |name: &str, lpr: u32, swiglu: bool| {
        let consts = FunctionConstantValues::new();
        for (index, value) in [(0u64, 1u32), (1, lpr)] {
            consts.set_constant_value_at_index(
                (&value as *const u32).cast(),
                MTLDataType::UInt,
                index,
            );
        }
        for (index, value) in [
            (2u64, false),
            (3, swiglu),
            (4, false),
            (5, false),
            (6, false),
            (7, false),
            (8, false),
        ] {
            consts.set_constant_value_at_index(
                (&value as *const bool).cast(),
                MTLDataType::Bool,
                index,
            );
        }
        let f = lib.get_function(name, Some(consts)).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };

    for (name, lpr) in [("matvec_q6_k_mv", 16u32), ("matvec_q6_k", 2u32)] {
        // swiglu works in place: each iteration gets a fresh gate copy.
        let act = buffer(gate.as_ptr().cast(), (n_in * 4) as u64);
        let cmd = queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);

        // Reference: standalone swiglu over (act, up), then the plain matvec.
        let f_sw = lib.get_function("swiglu", None).unwrap();
        let p_sw = device
            .new_compute_pipeline_state_with_function(&f_sw)
            .unwrap();
        enc.set_compute_pipeline_state(&p_sw);
        enc.set_buffer(0, Some(&act), 0);
        enc.set_buffer(1, Some(&ub), 0);
        enc.set_bytes(2, 4, (&n_in as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new((n_in as u64).div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        let p_plain = pipeline(name, lpr, false);
        enc.set_compute_pipeline_state(&p_plain);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&act), 0);
        enc.set_buffer(2, Some(&y_ref), 0);
        enc.set_bytes(3, 4, (&n_in as *const u32).cast());
        enc.set_bytes(4, 4, (&n_out as *const u32).cast());
        enc.set_bytes(5, 8, (&w_off as *const u64).cast());
        enc.set_buffer(6, Some(&act), 0);
        enc.set_bytes(7, 32, idx_zero.as_ptr().cast());
        enc.set_buffer(8, Some(&ub), 0);
        enc.dispatch_thread_groups(
            MTLSize::new((n_out as u64 * lpr as u64).div_ceil(128), 1, 1),
            MTLSize::new(128, 1, 1),
        );
        // Fused: x = raw gate, xu = raw up, silu(gate)*up on the fly.
        let p_fused = pipeline(name, lpr, true);
        enc.set_compute_pipeline_state(&p_fused);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&gb), 0);
        enc.set_buffer(2, Some(&y_fused), 0);
        enc.set_bytes(3, 4, (&n_in as *const u32).cast());
        enc.set_bytes(4, 4, (&n_out as *const u32).cast());
        enc.set_bytes(5, 8, (&w_off as *const u64).cast());
        enc.set_buffer(6, Some(&act), 0);
        enc.set_bytes(7, 32, idx_zero.as_ptr().cast());
        enc.set_buffer(8, Some(&ub), 0);
        enc.dispatch_thread_groups(
            MTLSize::new((n_out as u64 * lpr as u64).div_ceil(128), 1, 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);

        let s_ref =
            unsafe { std::slice::from_raw_parts(y_ref.contents().cast::<f32>(), n_out as usize) };
        let s_fused =
            unsafe { std::slice::from_raw_parts(y_fused.contents().cast::<f32>(), n_out as usize) };
        for (i, (&a, &b)) in s_ref.iter().zip(s_fused).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "{name} swiglu-fold [{i}]: fused {b} != separate {a}"
            );
        }
    }
}

/// The fused decode QKV matvec (matvec_qkv_q4k_q6k) must be bit-identical
/// to the three separate matvec_q4_k_mv / matvec_q6_k_mv dispatches.
#[test]
fn matvec_qkv_q4k_q6k_matches_separate() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();

    let n_in = 512u32; // two 256-value blocks per row
    let n_q = 128u32;
    let n_k = 64u32;
    let n_v = 64u32;
    let nb = (n_in / 256) as usize;
    let qk_row = 144 * nb;
    let v_row = 210 * nb;
    let wq_off = 0u64;
    let wk_off = wq_off + (n_q as usize * qk_row) as u64;
    let wv_off = wk_off + (n_k as usize * qk_row) as u64;
    let total_w = wv_off as usize + n_v as usize * v_row;
    let mut weights = vec![0u8; total_w];
    for (i, v) in weights.iter_mut().enumerate() {
        *v = (i.wrapping_mul(59) + i / 97) as u8;
    }
    for (base, row) in [(wq_off, qk_row), (wk_off, qk_row)] {
        for b in weights[base as usize..]
            .chunks_exact_mut(row)
            .take(if base == wq_off {
                n_q as usize
            } else {
                n_k as usize
            })
        {
            for blk in b.chunks_exact_mut(144) {
                blk[0..2].copy_from_slice(&0x211fu16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x211fu16.to_le_bytes());
            }
        }
    }
    for b in weights[wv_off as usize..].chunks_exact_mut(v_row) {
        for blk in b.chunks_exact_mut(210) {
            blk[208..210].copy_from_slice(&0x211fu16.to_le_bytes());
        }
    }
    let x: Vec<f32> = (0..n_in as usize)
        .map(|i| (i % 89) as f32 * 0.02 - 0.9)
        .collect();
    let buffer = |p, n| device.new_buffer_with_data(p, n, MTLResourceOptions::StorageModeShared);
    let wb = buffer(weights.as_ptr().cast(), weights.len() as u64);
    let xb = buffer(x.as_ptr().cast(), (n_in * 4) as u64);
    let total = (n_q + n_k + n_v) as usize;
    let ref_out = device.new_buffer((total * 4) as u64, MTLResourceOptions::StorageModeShared);
    let fused_out = device.new_buffer((total * 4) as u64, MTLResourceOptions::StorageModeShared);
    let idx_zero = [0u8; 32];

    let mv_pipeline = |name: &str| {
        let consts = FunctionConstantValues::new();
        for (index, value) in [(0u64, 1u32), (1, 16u32)] {
            consts.set_constant_value_at_index(
                (&value as *const u32).cast(),
                MTLDataType::UInt,
                index,
            );
        }
        for index in 2u64..=8 {
            consts.set_constant_value_at_index(
                (&false as *const bool).cast(),
                MTLDataType::Bool,
                index,
            );
        }
        let f = lib.get_function(name, Some(consts)).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
    for (name, n_out, w_off, y_off) in [
        ("matvec_q4_k_mv", n_q, wq_off, 0u64),
        ("matvec_q4_k_mv", n_k, wk_off, (n_q * 4) as u64),
        ("matvec_q6_k_mv", n_v, wv_off, ((n_q + n_k) * 4) as u64),
    ] {
        let p = mv_pipeline(name);
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&xb), 0);
        enc.set_buffer(2, Some(&ref_out), y_off);
        enc.set_bytes(3, 4, (&n_in as *const u32).cast());
        enc.set_bytes(4, 4, (&n_out as *const u32).cast());
        enc.set_bytes(5, 8, (&w_off as *const u64).cast());
        enc.set_buffer(6, Some(&xb), 0);
        enc.set_bytes(7, 32, idx_zero.as_ptr().cast());
        enc.set_buffer(8, Some(&xb), 0);
        enc.dispatch_thread_groups(
            MTLSize::new((n_out as u64 * 16).div_ceil(128), 1, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    #[repr(C)]
    struct MvQkvArgs {
        n_in: u32,
        n_q: u32,
        n_k: u32,
        n_v: u32,
        wq_off: u64,
        wk_off: u64,
        wv_off: u64,
        q_off: u64,
        k_off: u64,
        v_off: u64,
    }
    let args = MvQkvArgs {
        n_in,
        n_q,
        n_k,
        n_v,
        wq_off,
        wk_off,
        wv_off,
        q_off: 0,
        k_off: (n_q * 4) as u64,
        v_off: ((n_q + n_k) * 4) as u64,
    };
    let f_fused = lib.get_function("matvec_qkv_q4k_q6k", None).unwrap();
    let p_fused = device
        .new_compute_pipeline_state_with_function(&f_fused)
        .unwrap();
    enc.set_compute_pipeline_state(&p_fused);
    enc.set_buffer(0, Some(&wb), 0);
    enc.set_buffer(1, Some(&wb), 0);
    enc.set_buffer(2, Some(&wb), 0);
    enc.set_buffer(3, Some(&xb), 0);
    enc.set_buffer(4, Some(&fused_out), 0);
    enc.set_bytes(
        5,
        std::mem::size_of::<MvQkvArgs>() as u64,
        (&args as *const MvQkvArgs).cast(),
    );
    enc.dispatch_thread_groups(
        MTLSize::new(((n_q + n_k + n_v) as u64 * 16).div_ceil(128), 1, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);

    let s_ref = unsafe { std::slice::from_raw_parts(ref_out.contents().cast::<f32>(), total) };
    let s_fused = unsafe { std::slice::from_raw_parts(fused_out.contents().cast::<f32>(), total) };
    for (i, (&a, &b)) in s_ref.iter().zip(s_fused).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
            "matvec_qkv_q4k_q6k[{i}]: fused {b} != separate {a}"
        );
    }
}

/// Inner-loop pipelined FlashAttention (`attend_mm_db`, `attend_mm_fq_db`,
/// `attend_mm256_db`) preserves per-accumulator reduction order, so outputs
/// must be bit-identical to the single-chain kernels.
#[test]
fn attend_mm_db_variants_are_bit_identical() {
    #[repr(C)]
    struct FqPrepArgs {
        eps: f32,
        has_qk_norm: u32,
        rot_dim: u32,
        has_bias: u32,
        n_kv_heads: u32,
        wait_base: u32,
    }

    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();
    let pipe = |name: &str| {
        let f = lib.get_function(name, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };
    let buffer = |p, n| device.new_buffer_with_data(p, n, MTLResourceOptions::StorageModeShared);

    // 1. attend_mm vs attend_mm_db (hd=128) and attend_mm256 vs attend_mm256_db (hd=256)
    for (hd, k0, k1) in [
        (128usize, "attend_mm", "attend_mm_db"),
        (256usize, "attend_mm256", "attend_mm256_db"),
    ] {
        let n_q_heads = 8usize;
        let n_kv_heads = 2usize;
        let group = (n_q_heads / n_kv_heads) as u32;
        let base = 19usize;
        let n_rows = 77usize; // spans >1 64-position tile + partial tail row
        let total_pos = base + n_rows;
        let kv_dim = n_kv_heads * hd;
        let q_dim = n_q_heads * hd;
        let scale = 1.0f32 / (hd as f32).sqrt();

        let kv_f32: Vec<f32> = (0..total_pos * kv_dim * 2)
            .map(|i| ((i * 31 + 7) % 113) as f32 * 0.015 - 0.8)
            .collect();
        let mut kv_f16 = vec![0u16; kv_f32.len()];
        allpaka_dtype::f32_to_f16_slice(&kv_f32, &mut kv_f16);
        let q_f32: Vec<f32> = (0..n_rows * q_dim)
            .map(|i| ((i * 19 + 11) % 101) as f32 * 0.018 - 0.9)
            .collect();

        let kv_buf = buffer(kv_f16.as_ptr().cast(), (kv_f16.len() * 2) as u64);
        let q_buf = buffer(q_f32.as_ptr().cast(), (q_f32.len() * 4) as u64);
        let k_base = 0u64;
        let v_base = (total_pos * kv_dim) as u64;
        let kv_dim32 = kv_dim as u32;
        let hd32 = hd as u32;
        let base32 = base as u32;
        let heads32 = n_q_heads as u32;
        let rows32 = n_rows as u32;

        let outputs = [k0, k1].map(|kname| {
            let out = device.new_buffer(
                (n_rows * q_dim * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let p = pipe(kname);
            let cmd = queue.new_command_buffer();
            let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
            enc.set_compute_pipeline_state(&p);
            enc.set_buffer(0, Some(&kv_buf), 0);
            enc.set_buffer(1, Some(&kv_buf), 0);
            enc.set_buffer(2, Some(&q_buf), 0);
            enc.set_buffer(3, Some(&out), 0);
            enc.set_bytes(4, 4, (&kv_dim32 as *const u32).cast());
            enc.set_bytes(5, 4, (&hd32 as *const u32).cast());
            enc.set_bytes(6, 4, (&base32 as *const u32).cast());
            enc.set_bytes(7, 4, (&group as *const u32).cast());
            enc.set_bytes(8, 4, (&scale as *const f32).cast());
            enc.set_bytes(9, 8, (&k_base as *const u64).cast());
            enc.set_bytes(10, 8, (&v_base as *const u64).cast());
            enc.set_bytes(11, 4, (&heads32 as *const u32).cast());
            enc.set_bytes(12, 4, (&rows32 as *const u32).cast());
            enc.dispatch_thread_groups(
                MTLSize::new(n_q_heads as u64, (n_rows as u64).div_ceil(8), 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "{kname}");
            out
        });

        let a_slice = unsafe {
            std::slice::from_raw_parts(outputs[0].contents().cast::<f32>(), n_rows * q_dim)
        };
        let b_slice = unsafe {
            std::slice::from_raw_parts(outputs[1].contents().cast::<f32>(), n_rows * q_dim)
        };
        for (i, (&a, &b)) in a_slice.iter().zip(b_slice).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "{k1} vs {k0} [{i}]: {b} != {a}"
            );
        }
    }

    // 2. attend_mm_fq vs attend_mm_fq_db (hd=128 with qk-norm + rope + producer/consumer sync)
    let hd = 128usize;
    let n_q_heads = 8usize;
    let n_kv_heads = 2usize;
    let group = (n_q_heads / n_kv_heads) as u32;
    let base = 16usize;
    let n_rows = 75usize;
    let total_pos = base + n_rows;
    let kv_dim = n_kv_heads * hd;
    let q_dim = n_q_heads * hd;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let kv_init_f32: Vec<f32> = (0..total_pos * kv_dim * 2)
        .map(|i| ((i * 29 + 5) % 109) as f32 * 0.014 - 0.7)
        .collect();
    let mut kv_init_f16 = vec![0u16; kv_init_f32.len()];
    allpaka_dtype::f32_to_f16_slice(&kv_init_f32, &mut kv_init_f16);
    let q_in: Vec<f32> = (0..n_rows * q_dim)
        .map(|i| ((i * 23 + 3) % 97) as f32 * 0.02 - 0.9)
        .collect();
    let k_in: Vec<f32> = (0..n_rows * kv_dim)
        .map(|i| ((i * 41 + 9) % 89) as f32 * 0.019 - 0.8)
        .collect();
    let v_in: Vec<f32> = (0..n_rows * kv_dim)
        .map(|i| ((i * 37 + 13) % 83) as f32 * 0.017 - 0.7)
        .collect();
    let qw: Vec<f32> = (0..hd).map(|i| 0.85 + (i as f32) * 0.002).collect();
    let kw: Vec<f32> = (0..hd).map(|i| 0.9 + (i as f32) * 0.0015).collect();
    let ropes: Vec<[f32; 2]> = (0..n_rows * (hd / 2))
        .map(|i| {
            let ang = (i as f32) * 0.013;
            [ang.sin(), ang.cos()]
        })
        .collect();

    let q_buf = buffer(q_in.as_ptr().cast(), (q_in.len() * 4) as u64);
    let k_in_buf = buffer(k_in.as_ptr().cast(), (k_in.len() * 4) as u64);
    let v_in_buf = buffer(v_in.as_ptr().cast(), (v_in.len() * 4) as u64);
    let ropes_buf = buffer(ropes.as_ptr().cast(), (ropes.len() * 8) as u64);
    let pa = FqPrepArgs {
        eps: 1e-6,
        has_qk_norm: 1,
        rot_dim: hd as u32,
        has_bias: 0,
        n_kv_heads: n_kv_heads as u32,
        wait_base: 0,
    };
    let k_base = 0u64;
    let v_base = (total_pos * kv_dim) as u64;
    let kv_dim32 = kv_dim as u32;
    let hd32 = hd as u32;
    let base32 = base as u32;
    let heads32 = n_q_heads as u32;
    let rows32 = n_rows as u32;

    let fq_outputs = ["attend_mm_fq", "attend_mm_fq_db"].map(|kname| {
        let kv_buf = buffer(kv_init_f16.as_ptr().cast(), (kv_init_f16.len() * 2) as u64);
        let sync = device.new_buffer(256, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(sync.contents().cast::<u8>(), 0, 256) };
        let out = device.new_buffer(
            (n_rows * q_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let p = pipe(kname);
        let cmd = queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&kv_buf), 0);
        enc.set_buffer(1, Some(&kv_buf), 0);
        enc.set_buffer(2, Some(&q_buf), 0);
        enc.set_buffer(3, Some(&out), 0);
        enc.set_bytes(4, 4, (&kv_dim32 as *const u32).cast());
        enc.set_bytes(5, 4, (&hd32 as *const u32).cast());
        enc.set_bytes(6, 4, (&base32 as *const u32).cast());
        enc.set_bytes(7, 4, (&group as *const u32).cast());
        enc.set_bytes(8, 4, (&scale as *const f32).cast());
        enc.set_bytes(9, 8, (&k_base as *const u64).cast());
        enc.set_bytes(10, 8, (&v_base as *const u64).cast());
        enc.set_bytes(11, 4, (&heads32 as *const u32).cast());
        enc.set_bytes(12, 4, (&rows32 as *const u32).cast());
        enc.set_buffer(13, Some(&k_in_buf), 0);
        enc.set_buffer(14, Some(&v_in_buf), 0);
        enc.set_bytes(
            15,
            std::mem::size_of::<FqPrepArgs>() as u64,
            (&pa as *const FqPrepArgs).cast(),
        );
        enc.set_bytes(16, (hd * 4) as u64, qw.as_ptr().cast());
        enc.set_bytes(17, (hd * 4) as u64, kw.as_ptr().cast());
        enc.set_buffer(18, Some(&ropes_buf), 0);
        enc.set_buffer(19, Some(&kv_buf), 0);
        enc.set_buffer(20, Some(&kv_buf), 0);
        enc.set_buffer(21, Some(&kv_buf), 0);
        enc.set_buffer(22, Some(&sync), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(n_q_heads as u64, (n_rows as u64).div_ceil(8), 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "{kname}");
        out
    });

    let fq_a = unsafe {
        std::slice::from_raw_parts(fq_outputs[0].contents().cast::<f32>(), n_rows * q_dim)
    };
    let fq_b = unsafe {
        std::slice::from_raw_parts(fq_outputs[1].contents().cast::<f32>(), n_rows * q_dim)
    };
    for (i, (&a, &b)) in fq_a.iter().zip(fq_b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
            "attend_mm_fq_db vs attend_mm_fq [{i}]: {b} != {a}"
        );
    }
}

/// Generalized `mmllp_qkv_mixed` bit-exactness across arbitrary combinations
/// of `{Q4_K=0, Q5_K=1, Q6_K=2, Q8_0=3, Q2_K=4, Q3_K=5}`, plus GDN 2-way
/// (`n_v == 0`).
#[test]
fn fused_qkv_mixed_any_formats_match_separate() {
    #[repr(C)]
    struct MmQkvArgs {
        n_in: u32,
        n_q: u32,
        n_k: u32,
        n_v: u32,
        wq_off: u64,
        wk_off: u64,
        wv_off: u64,
        q_off: u64,
        k_off: u64,
        v_off: u64,
        m: u32,
        pad: u32,
    }

    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();
    let f_mixed = lib.get_function("mmllp_qkv_mixed", None).unwrap();
    let p_mixed = device
        .new_compute_pipeline_state_with_function(&f_mixed)
        .unwrap();

    let ni = 512u32; // multiple of 256 so multi-block pipe runs (nblk = 16)
    let m = 45u32;
    let x: Vec<f32> = (0..(m * ni) as usize)
        .map(|i| (i % 89) as f32 * 0.004 - 0.15)
        .collect();
    let buffer = |p, n| device.new_buffer_with_data(p, n, MTLResourceOptions::StorageModeShared);
    let xb = buffer(x.as_ptr().cast(), (x.len() * 4) as u64);

    // (fmt_id, single_kernel, block_bytes, blocks_per_256, scale_offsets)
    let fmt_meta = |fmt: u32| -> (&'static str, usize, usize, Vec<usize>) {
        match fmt {
            0 => ("mmllp_q4_k", 144, 2, vec![0, 2]),
            1 => ("mmllp_q5_k", 176, 2, vec![0, 2]),
            2 => ("mmllp_q6_k", 210, 2, vec![208]),
            3 => ("mmllp_q8_0", 34, 16, vec![0]),
            4 => ("mmllp_q2_k", 84, 2, vec![80, 82]),
            5 => ("mmllp_q3_k", 110, 2, vec![108]),
            _ => unreachable!(),
        }
    };

    for (fq, fk, fv, n_q, n_k, n_v) in [
        (4u32, 5u32, 2u32, 128u32, 64u32, 64u32), // Q2_K + Q3_K + Q6_K (235B-style mixed)
        (0u32, 1u32, 3u32, 128u32, 64u32, 64u32), // Q4_K + Q5_K + Q8_0 (35B-style mixed)
        (3u32, 4u32, 0u32, 192u32, 128u32, 0u32), // GDN 2-way (n_v == 0): Q8_0 + Q2_K
    ] {
        let make_weights = |fmt: u32, rows: u32, seed: usize| -> Buffer {
            let (_, bb, blks, scales) = fmt_meta(fmt);
            let len = (rows as usize * blks * bb).max(256);
            let mut w = vec![0u8; len];
            for (i, v) in w.iter_mut().enumerate() {
                *v = ((i + seed).wrapping_mul(67) + i / 83) as u8;
            }
            for b in w.chunks_exact_mut(bb) {
                for &off in &scales {
                    b[off..off + 2].copy_from_slice(&0x211fu16.to_le_bytes());
                }
            }
            buffer(w.as_ptr().cast(), w.len() as u64)
        };
        let wq_b = make_weights(fq, n_q, 11);
        let wk_b = make_weights(fk, n_k, 37);
        let wv_b = make_weights(fv, n_v, 71);

        let q_elems = (m * n_q) as usize;
        let k_elems = (m * n_k) as usize;
        let v_elems = (m * n_v) as usize;
        let total_out = q_elems + k_elems + v_elems;
        let ref_out = device.new_buffer(
            (total_out * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let fused_out = device.new_buffer(
            (total_out * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
        let zero64 = 0u64;
        for (fmt, wb, n_out, y_off) in [
            (fq, &wq_b, n_q, 0u64),
            (fk, &wk_b, n_k, (q_elems * 4) as u64),
            (fv, &wv_b, n_v, ((q_elems + k_elems) * 4) as u64),
        ] {
            if n_out == 0 {
                continue;
            }
            let (sname, _, _, _) = fmt_meta(fmt);
            let f = lib.get_function(sname, None).unwrap();
            let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
            enc.set_compute_pipeline_state(&p);
            enc.set_buffer(0, Some(wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ref_out), y_off);
            enc.set_bytes(3, 4, (&ni as *const u32).cast());
            enc.set_bytes(4, 4, (&n_out as *const u32).cast());
            enc.set_bytes(5, 8, (&zero64 as *const u64).cast());
            enc.set_bytes(6, 4, (&m as *const u32).cast());
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64).div_ceil(64), (m as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );
        }

        let args = MmQkvArgs {
            n_in: ni,
            n_q,
            n_k,
            n_v,
            wq_off: 0,
            wk_off: 0,
            wv_off: 0,
            q_off: 0,
            k_off: (q_elems * 4) as u64,
            v_off: ((q_elems + k_elems) * 4) as u64,
            m,
            pad: fq | (fk << 8) | (fv << 16),
        };
        enc.set_compute_pipeline_state(&p_mixed);
        enc.set_buffer(0, Some(&wq_b), 0);
        enc.set_buffer(1, Some(&wk_b), 0);
        enc.set_buffer(2, Some(&wv_b), 0);
        enc.set_buffer(3, Some(&xb), 0);
        enc.set_buffer(4, Some(&fused_out), 0);
        enc.set_bytes(
            5,
            std::mem::size_of::<MmQkvArgs>() as u64,
            (&args as *const MmQkvArgs).cast(),
        );
        enc.dispatch_thread_groups(
            MTLSize::new(
                ((n_q + n_k + n_v) as u64).div_ceil(64),
                (m as u64).div_ceil(32),
                1,
            ),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);

        let s_ref =
            unsafe { std::slice::from_raw_parts(ref_out.contents().cast::<f32>(), total_out) };
        let s_fused =
            unsafe { std::slice::from_raw_parts(fused_out.contents().cast::<f32>(), total_out) };
        for (i, (&a, &b)) in s_ref.iter().zip(s_fused).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "mmllp_qkv_mixed({fq},{fk},{fv})[{i}]: {b} != {a}"
            );
        }
    }
}

/// Verify vectorized `matvec_q2_k_mv` (indexed + `SWIGLU_X` + `MV_ID_GRID`)
/// matches `matvec_q2_k` within float reassociation tolerance and benchmark
/// both on 235B MoE shapes (`[1536, 4096]` and `[4096, 1536] × 8` slots).
#[test]
fn matvec_q2_k_mv_matches_reference_and_swiglu_fold() {
    #[repr(C)]
    struct IdxArgs {
        stride: u64,
        slots: u32,
        x_stride: u32,
        n_rows: u32,
        x_row_stride: u32,
        y_row_stride: u32,
        ids_stride: u32,
    }

    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();

    let make_pipe = |name: &str, lpr: u32, indexed: bool, swiglu: bool, id_grid: bool| {
        let consts = FunctionConstantValues::new();
        let tile = 1u32;
        consts.set_constant_value_at_index((&tile as *const u32).cast(), MTLDataType::UInt, 0);
        consts.set_constant_value_at_index((&lpr as *const u32).cast(), MTLDataType::UInt, 1);
        consts.set_constant_value_at_index((&indexed as *const bool).cast(), MTLDataType::Bool, 2);
        consts.set_constant_value_at_index((&swiglu as *const bool).cast(), MTLDataType::Bool, 3);
        let f_false = false;
        for idx in [4u64, 5, 6, 7] {
            consts.set_constant_value_at_index(
                (&f_false as *const bool).cast(),
                MTLDataType::Bool,
                idx,
            );
        }
        consts.set_constant_value_at_index((&id_grid as *const bool).cast(), MTLDataType::Bool, 8);
        let fun = lib.get_function(name, Some(consts)).unwrap();
        device
            .new_compute_pipeline_state_with_function(&fun)
            .unwrap()
    };

    let n_in = 1536usize;
    let n_out = 514usize; // non-multiple of 4 exercises NR0=4 tail handling
    let slots = 8usize;
    let n_experts = 12usize;
    let row_bytes = (n_in / 256) * 84;
    let estride = n_out * row_bytes;
    let mut weights = vec![0u8; n_experts * estride];
    for (i, v) in weights.iter_mut().enumerate() {
        *v = (i.wrapping_mul(43) + i / 113) as u8;
    }
    for b in weights.chunks_exact_mut(84) {
        b[80..82].copy_from_slice(&0x211fu16.to_le_bytes());
        b[82..84].copy_from_slice(&0x2010u16.to_le_bytes());
    }
    let xg: Vec<f32> = (0..slots * n_in)
        .map(|i| ((i * 31 + 7) % 101) as f32 * 0.02 - 1.0)
        .collect();
    let xu: Vec<f32> = (0..slots * n_in)
        .map(|i| ((i * 53 + 11) % 97) as f32 * 0.02 - 0.95)
        .collect();
    let ids: Vec<u32> = [2, 0, 11, 5, 7, 3, 9, 1].to_vec();

    let buffer = |p, n| device.new_buffer_with_data(p, n, MTLResourceOptions::StorageModeShared);
    let wb = buffer(weights.as_ptr().cast(), weights.len() as u64);
    let xgb = buffer(xg.as_ptr().cast(), (xg.len() * 4) as u64);
    let xub = buffer(xu.as_ptr().cast(), (xu.len() * 4) as u64);
    let idsb = buffer(ids.as_ptr().cast(), (ids.len() * 4) as u64);

    for swiglu in [false, true] {
        let p_ref = make_pipe("matvec_q2_k", 8, true, swiglu, false);
        let y_ref = device.new_buffer(
            (slots * n_out * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let idx = IdxArgs {
            slots: slots as u32,
            stride: estride as u64,
            x_stride: n_in as u32,
            n_rows: 1,
            x_row_stride: 0,
            y_row_stride: 0,
            ids_stride: 0,
        };
        let ni32 = n_in as u32;
        let no32 = n_out as u32;
        let zero64 = 0u64;

        let cmd = queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
        enc.set_compute_pipeline_state(&p_ref);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&xgb), 0);
        enc.set_buffer(2, Some(&y_ref), 0);
        enc.set_bytes(3, 4, (&ni32 as *const u32).cast());
        enc.set_bytes(4, 4, (&no32 as *const u32).cast());
        enc.set_bytes(5, 8, (&zero64 as *const u64).cast());
        enc.set_buffer(6, Some(&idsb), 0);
        enc.set_bytes(
            7,
            std::mem::size_of::<IdxArgs>() as u64,
            (&idx as *const IdxArgs).cast(),
        );
        enc.set_buffer(8, Some(&xub), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(((n_out * slots * 8) as u64).div_ceil(128), 1, 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);

        let ref_s =
            unsafe { std::slice::from_raw_parts(y_ref.contents().cast::<f32>(), slots * n_out) };

        for (lpr, id_grid) in [(16u32, true), (8u32, true)] {
            let p_mv = make_pipe("matvec_q2_k_mv", lpr, true, swiglu, id_grid);
            let y_mv = device.new_buffer(
                (slots * n_out * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let cmd = queue.new_command_buffer();
            let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
            enc.set_compute_pipeline_state(&p_mv);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xgb), 0);
            enc.set_buffer(2, Some(&y_mv), 0);
            enc.set_bytes(3, 4, (&ni32 as *const u32).cast());
            enc.set_bytes(4, 4, (&no32 as *const u32).cast());
            enc.set_bytes(5, 8, (&zero64 as *const u64).cast());
            enc.set_buffer(6, Some(&idsb), 0);
            enc.set_bytes(
                7,
                std::mem::size_of::<IdxArgs>() as u64,
                (&idx as *const IdxArgs).cast(),
            );
            enc.set_buffer(8, Some(&xub), 0);
            if id_grid {
                enc.dispatch_thread_groups(
                    MTLSize::new((n_out as u64 * lpr as u64).div_ceil(128), 1, slots as u64),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                enc.dispatch_thread_groups(
                    MTLSize::new(((n_out * slots) as u64 * lpr as u64).div_ceil(128), 1, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);

            let mv_s =
                unsafe { std::slice::from_raw_parts(y_mv.contents().cast::<f32>(), slots * n_out) };
            let mut max_rel = 0.0f32;
            for (i, (&a, &b)) in ref_s.iter().zip(mv_s).enumerate() {
                let err = (a - b).abs();
                let rel = err / a.abs().max(1.0);
                max_rel = max_rel.max(rel);
                assert!(
                    a.is_finite() && b.is_finite() && rel < 1e-5,
                    "q2_k_mv(lpr={lpr},swiglu={swiglu},idg={id_grid})[{i}]: {b} vs {a} (err {err}, rel {rel})"
                );
            }
            assert!(max_rel < 1e-5);
        }
    }
}
