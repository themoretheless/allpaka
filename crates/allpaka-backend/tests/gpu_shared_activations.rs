//! Ownership and numerical regression for the zero-copy prefill/head boundary.
#![cfg(target_os = "macos")]
use allpaka_backend::{activation::Activation, gpu, ops, QuantMat};
use allpaka_gguf::GgmlType;

#[test]
fn completed_residual_is_owned_and_projects_without_staging() {
    let layout = std::alloc::Layout::from_size_align(16384, 16384).unwrap();
    // GPU weight mappings last until process exit, as in the production mmap.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    let weights = unsafe { std::slice::from_raw_parts_mut(ptr, 16384) };
    let (n_in, n_out) = (32, 64);
    for row in 0..n_out {
        let block = &mut weights[row * 34..(row + 1) * 34];
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = ((row + i) as i8 % 9 - 4) as u8;
        }
    }
    assert!(
        gpu::attach(weights),
        "Metal required: {:?}",
        gpu::init_error()
    );
    let matrix = QuantMat::new(&weights[..n_out * 34], GgmlType::Q8_0, n_out, n_in).unwrap();
    assert!(gpu::prefill_end_output(1).is_none(), "no active chunk");
    for rows in [1, 3, 16, 33] {
        let input: Vec<f32> = (0..(rows + 1) * n_in)
            .map(|i| (i % 31) as f32 * 0.03 - 0.4)
            .collect();
        gpu::prefill_begin(&input).unwrap();
        assert!(
            gpu::prefill_end_output(input.len() + 1).is_none(),
            "reject uninitialized tail"
        );
        let mut output = gpu::prefill_end_output(input.len()).unwrap();
        assert!(
            gpu::prefill_end_output(input.len()).is_none(),
            "handoff is one-shot"
        );
        assert!(
            matches!(&output, Activation::Shared(_)),
            "test requires zero-copy enabled"
        );
        assert_eq!(&*output, &input);
        let saved_pointer = output.as_ptr();
        // A subsequent chunk must not overwrite an output still owned by CPU.
        gpu::prefill_begin(&vec![99.0; input.len() * 2]).unwrap();
        let other = gpu::prefill_end_output(input.len() * 2).unwrap();
        assert_eq!(&*output, &input);
        assert_ne!(saved_pointer, other.as_ptr());
        for row in output.chunks_mut(n_in) {
            ops::rmsnorm(row, &vec![1.0; n_in], 1e-6);
        }
        assert_eq!(
            saved_pointer,
            output.as_ptr(),
            "CPU norm must stay in shared storage"
        );
        // An offset exercises the last-token head and non-power-of-two tiling.
        let expected = matrix.matmul(&output[n_in..], rows).unwrap();
        let actual = matrix.matmul_activation(&output, n_in, rows).unwrap();
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 1e-4, "rows={rows}, diff={max_error}");
        let before = gpu::shared_activation_stats();
        let fallback = matrix.matmul_activation(&output, 1, rows).unwrap();
        let expected = matrix.matmul(&output[1..1 + rows * n_in], rows).unwrap();
        assert_eq!(fallback, expected, "unaligned input must use staging");
        assert_eq!(
            gpu::shared_activation_stats().projection_bytes,
            before.projection_bytes
        );
        assert!(matrix.matmul_activation(&output, usize::MAX, rows).is_err());
        assert!(matrix.matmul_activation(&output, 0, usize::MAX).is_err());
    }
}
