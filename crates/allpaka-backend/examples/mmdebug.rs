// Debug partial-tile store of mmll_q8_0: n_out=64, n_in=1024, m=20.
use metal::*;

fn main() {
    let n_in = 1024usize;
    let n_out = 64usize;
    let m = 20usize;
    // Q8_0 block: 2-byte d (half) + 32 i8 qs. Use d=0.01, qs pattern.
    let row_bytes = n_in / 32 * 34;
    let mut w = vec![0u8; n_out * row_bytes];
    let d = 0x211Fu16.to_le_bytes(); // 0.01 in f16
    for r in 0..n_out {
        for b in 0..n_in / 32 {
            let at = r * row_bytes + b * 34;
            w[at] = d[0];
            w[at + 1] = d[1];
            for q in 0..32 {
                w[at + 2 + q] = ((r + b + q) % 7) as u8; // small ints
            }
        }
    }
    let x: Vec<f32> = (0..m * n_in).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
    // CPU reference (f16 rounding of x like the kernel stages)
    let f16r = |v: f32| -> f32 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u32;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = (b >> 13) & 0x3ff;
        let h = if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7c00
        } else {
            sign | ((exp as u32) << 10) | mant
        } as u16;
        // decode
        let s = (h >> 15) as f32 * -2.0 + 1.0;
        let e = ((h >> 10) & 0x1f) as i32;
        let mm = (h & 0x3ff) as f32 / 1024.0;
        if e == 0 {
            s * mm * 2f32.powi(-14)
        } else if e == 31 {
            f32::INFINITY
        } else {
            s * (1.0 + mm) * 2f32.powi(e - 15)
        }
    };
    let deq = |r: usize, k: usize| -> f32 {
        let b = k / 32;
        let at = r * row_bytes + b * 34;
        w[at + 2 + k % 32] as i8 as f32 * 0.01
    };
    let mut want = vec![0f32; m * n_out];
    for i in 0..m {
        for j in 0..n_out {
            let mut s = 0f32;
            for k in 0..n_in {
                s += deq(j, k) * f16r(x[i * n_in + k]);
            }
            want[i * n_out + j] = s;
        }
    }

    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(allpaka_backend::gpu::KERNELS, &CompileOptions::new())
        .unwrap();
    let fun = lib.get_function("mmll_q8_0", None).unwrap();
    let pipe = device.new_compute_pipeline_state_with_function(&fun).unwrap();
    let w_buf = device.new_buffer_with_data(
        w.as_ptr() as *const _,
        w.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = device.new_buffer((m * n_out * 4) as u64, MTLResourceOptions::StorageModeShared);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&y_buf), 0);
    let ni = n_in as u32;
    let no = n_out as u32;
    let wo = 0u64;
    let mm = m as u32;
    enc.set_bytes(3, 4, &ni as *const u32 as *const _);
    enc.set_bytes(4, 4, &no as *const u32 as *const _);
    enc.set_bytes(5, 8, &wo as *const u64 as *const _);
    enc.set_bytes(6, 4, &mm as *const u32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize::new(n_out as u64 / 64, (m as u64 + 31) / 32, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let got = unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, m * n_out) };
    let mut bad = 0;
    for i in 0..m * n_out {
        if (got[i] - want[i]).abs() > 1e-2 + 2e-3 * want[i].abs() {
            if bad < 12 {
                println!("row {} col {}: gpu {:.4} want {:.4}", i / n_out, i % n_out, got[i], want[i]);
            }
            bad += 1;
        }
    }
    println!("bad = {bad} / {}", m * n_out);
}
