//! `#[repr(C)]` argument blocks shared with the Metal kernels. Field order and
//! width are part of the kernel ABI.

#[repr(C)]
pub(super) struct GpuIdxArgs {
    pub(super) stride: u64,
    pub(super) slots: u32,
    pub(super) x_stride: u32,
    // ROWS (MTP verify) only: per-token-row strides of the ids, x and y
    // blocks, plus the row count. Zero otherwise.
    pub(super) ids_stride: u32,
    pub(super) x_row_stride: u32,
    pub(super) y_row_stride: u32,
    pub(super) n_rows: u32,
}

/// Args of the fused decode QKV matvec (matvec_qkv_q4k_q6k): three weight
/// bases, three byte outputs, one shared input. Mirrors the Metal struct.
#[repr(C)]
pub(super) struct GpuMvQkvArgs {
    pub(super) n_in: u32,
    pub(super) n_q: u32,
    pub(super) n_k: u32,
    pub(super) n_v: u32,
    pub(super) wq_off: u64,
    pub(super) wk_off: u64,
    pub(super) wv_off: u64,
    pub(super) q_off: u64,
    pub(super) k_off: u64,
    pub(super) v_off: u64,
}

#[repr(C)]
pub(super) struct QkPrepArgs {
    pub(super) n_heads: u32,
    pub(super) n_kv_heads: u32,
    pub(super) head_dim: u32,
    pub(super) kv_dim: u32,
    pub(super) eps: f32,
    pub(super) pos: u32,
    pub(super) has_qk_norm: u32,
    pub(super) rot_dim: u32,
    pub(super) k_base: u64,
    pub(super) v_base: u64,
    pub(super) has_bias: u32,
    pub(super) pad2: u32,
}

#[repr(C)]
pub(super) struct FqPrepArgs {
    pub(super) eps: f32,
    pub(super) has_qk_norm: u32,
    pub(super) rot_dim: u32,
    pub(super) has_bias: u32,
    pub(super) n_kv_heads: u32,
    pub(super) wait_base: u32,
}

#[repr(C)]
pub(super) struct QkPrep256Args {
    pub(super) n_heads: u32,
    pub(super) n_kv_heads: u32,
    pub(super) kv_dim: u32,
    pub(super) eps: f32,
    pub(super) pos: u32,
    pub(super) has_qk_norm: u32,
    pub(super) rot_dim: u32,
    pub(super) gate_in_q: u32,
    pub(super) k_base: u64,
    pub(super) v_base: u64,
}

#[repr(C)]
pub(super) struct GdnConvArgs {
    pub(super) channels: u32,
    pub(super) d_conv: u32,
    pub(super) pad0: u32,
    pub(super) pad1: u32,
    pub(super) conv_off: u64,
}

#[repr(C)]
pub(super) struct GdnStepArgs {
    pub(super) heads_k: u32,
    pub(super) heads_v: u32,
    pub(super) d: u32,
    pub(super) key_dim: u32,
    pub(super) eps: f32,
    pub(super) pad0: u32,
    pub(super) state_off: u64,
}

#[repr(C)]
pub(super) struct QkPrepBatch256Args {
    pub(super) n_heads: u32,
    pub(super) n_kv_heads: u32,
    pub(super) kv_dim: u32,
    pub(super) eps: f32,
    pub(super) base: u32,
    pub(super) has_qk_norm: u32,
    pub(super) rot_dim: u32,
    pub(super) gate_in_q: u32,
    pub(super) k_base: u64,
    pub(super) v_base: u64,
}

#[repr(C)]
pub(super) struct GdnConvBatchArgs {
    pub(super) channels: u32,
    pub(super) d_conv: u32,
    pub(super) m: u32,
    pub(super) pad0: u32,
    pub(super) conv_off: u64,
}

#[repr(C)]
pub(super) struct GdnStepBatchArgs {
    pub(super) heads_k: u32,
    pub(super) heads_v: u32,
    pub(super) d: u32,
    pub(super) key_dim: u32,
    pub(super) m: u32,
    pub(super) eps: f32,
    pub(super) state_off: u64,
}
