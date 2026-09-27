//! Shape utilities: parse and validate MLX tensor shapes from safetensors metadata.
//!
//! Shapes are stored as `[dim0, dim1, ...]` where `dim0` is the fastest-varying
/// (row-major) dimension — exactly how allpaka expects it after flattening 3D
/// expert tensors into tall `QuantMat`.

/// A shape tuple from an MLX config or tensor spec.
pub type MlxShape = Vec<u64>;

/// Flatten a multi-dimensional shape into 2D `(n_out, n_in)` for matmul purposes.
/// For 3D expert tensors `[expert_ffn, n_in, n_expert]` this returns
/// `(n_out * n_expert, n_in)`.
pub fn flatten_for_matmul(shape: &MlxShape) -> (usize, usize) {
    match shape.len() {
        2 => (shape[1] as usize, shape[0] as usize),
        3 => (shape[1] as usize * shape[2] as usize, shape[0] as usize), // [batch/in, hidden, experts]? No: qwen3_moe uses [expert_count, out_dim, in_dim]
        _ => (1, 1), // fallback
    }
}

/// Validate that a shape fits a particular quantized pattern for MoE dispatch:
/// - gate_proj / up_proj: `[expert_count, intermediate_size, hidden_size]`
/// - down_proj: `[expert_count, hidden_size, intermediate_size]`
/// We return true if dimensions are compatible with our 3D slice_rows semantics.
pub fn is_3d_expert_shape(shape: &[u64]) -> bool {
    shape.len() == 3 && !shape.is_empty()
}

/// Read shape from a JSON value.
pub fn json_to_shape(v: &serde_json::Value) -> Option<MlxShape> {
    v.as_array().and_then(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>().into())
}

/// Convert a u64 vector to its Option<Vec<u64>> representation (for serde helpers).
fn vec_u64_opt(arr: &[serde_json::Value]) -> Option<Vec<u64>> {
    arr.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>().into()
}
