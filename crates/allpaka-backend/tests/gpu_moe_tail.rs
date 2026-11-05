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
        let mut kernels = vec![
            format!("mmll_id_{suffix}"),
            format!("mmllg_id_{suffix}"),
            format!("mmll_id_{suffix}_bn16"),
            format!("mmllg_id_{suffix}_bn16"),
        ];
        if suffix == "q4_k" {
            kernels.extend(
                [
                    "mmlls_id_q4_k",
                    "mmllpg_id_q4_k",
                    "mmllps_id_q4_k",
                    "mmlls_id_q4_k_bn16",
                    "mmllpg_id_q4_k_bn16",
                    "mmllps_id_q4_k_bn16",
                ]
                .map(str::to_owned),
            );
        }
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
        let mut bn16_pairs = vec![
            (format!("mmll_id_{suffix}"), format!("mmll_id_{suffix}_bn16")),
            (
                format!("mmllg_id_{suffix}"),
                format!("mmllg_id_{suffix}_bn16"),
            ),
        ];
        if suffix == "q4_k" {
            bn16_pairs.extend([
                ("mmlls_id_q4_k".to_owned(), "mmlls_id_q4_k_bn16".to_owned()),
                (
                    "mmllpg_id_q4_k".to_owned(),
                    "mmllpg_id_q4_k_bn16".to_owned(),
                ),
                (
                    "mmllps_id_q4_k".to_owned(),
                    "mmllps_id_q4_k_bn16".to_owned(),
                ),
            ]);
        }
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
        ("mmll_q4_k", "mmll_qkv_q4_k", 144, 1, vec![0, 2]),
        ("mmllp_q4_k", "mmllp_qkv_q4_k", 144, 1, vec![0, 2]),
        ("mmll_q5_k", "mmll_qkv_q5_k", 176, 1, vec![0, 2]),
        ("mmll_q6_k", "mmll_qkv_q6_k", 210, 1, vec![208]),
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
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "{qkv_name}");

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
