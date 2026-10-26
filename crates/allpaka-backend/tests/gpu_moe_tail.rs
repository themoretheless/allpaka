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
        let mut kernels = vec![format!("mmll_id_{suffix}"), format!("mmllg_id_{suffix}")];
        if suffix == "q4_k" {
            kernels
                .extend(["mmlls_id_q4_k", "mmllpg_id_q4_k", "mmllps_id_q4_k"].map(str::to_owned));
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
    }
}
