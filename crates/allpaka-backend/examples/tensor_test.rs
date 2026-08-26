// Does the Metal tensor API (mpp::tensor_ops) work on this GPU (M4)?
use metal::*;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let src = std::fs::read_to_string("/tmp/tensor_test.metal").unwrap();
    let opts = CompileOptions::new();
    let lib = match device.new_library_with_source(&src, &opts) {
        Ok(l) => l,
        Err(e) => {
            println!("COMPILE FAIL: {e}");
            return;
        }
    };
    let fun = lib.get_function("dummy_kernel", None).unwrap();
    let pipe = device
        .new_compute_pipeline_state_with_function(&fun)
        .unwrap();

    let n = 16usize;
    // crude f32->f16 bit conversion good enough for small integers
    fn f16(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = ((b >> 13) & 0x3ff) as u16;
        if exp <= 0 { return sign; }
        if exp >= 31 { return sign | 0x7c00; }
        sign | ((exp as u16) << 10) | mant
    }
    let a: Vec<u16> = (0..n * n)
        .map(|i| f16(if i % n == i / n { 1.0 } else { 0.0 }))
        .collect();
    let b: Vec<u16> = (0..n * n)
        .map(|i| f16((i % 7) as f32))
        .collect();
    let a_buf = device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let c_buf = device.new_buffer((n * n * 4) as u64, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    // tensor<> args are passed as buffers
    enc.set_buffer(0, Some(&a_buf), 0);
    enc.set_buffer(1, Some(&b_buf), 0);
    enc.set_buffer(2, Some(&c_buf), 0);
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let c = unsafe {
        std::slice::from_raw_parts(c_buf.contents() as *const f32, n * n)
    };
    // A = identity, so C should equal B
    let ok = (0..n * n).all(|i| (c[i] - ((i % 7) as f32)).abs() < 1e-3);
    println!("tensor API works: {ok}; C[0..4] = {:?}", &c[0..4]);
}
