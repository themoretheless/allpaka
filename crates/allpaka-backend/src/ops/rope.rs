//! Rotary Position Embedding (RoPE) kernels (NORM and NEOX variants).

/// The position-independent half of RoPE: `inv_freq[i] = base^(-2i/d)`.
///
/// Only `head_dim / 2` values exist per model, yet the naive path recomputes
/// `powf` per element per head per layer per token - over 100k calls per
/// decoded token on a 48-layer model. Compute once at load.
pub fn rope_inv_freq(head_dim: usize, freq_base: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| freq_base.powf(-2.0 * i as f32 / head_dim as f32))
        .collect()
}

/// The per-position half of RoPE: `(sin, cos)` of `pos * inv_freq[i]`.
///
/// A token's position is shared by every head of every layer, so this is
/// computed once per position and reused ~1,700 times per token.
pub fn rope_sin_cos(pos: u32, inv_freq: &[f32], out: &mut Vec<(f32, f32)>) {
    out.clear();
    out.reserve(inv_freq.len());
    out.extend(inv_freq.iter().map(|&f| (pos as f32 * f).sin_cos()));
}

/// [`rope_norm`] over a precomputed per-position table. Bit-identical to the
/// naive form: the table holds the same `sin_cos` of the same `f32` products.
pub fn rope_norm_cached(x: &mut [f32], table: &[(f32, f32)]) {
    assert_eq!(x.len(), table.len() * 2);
    for (i, &(sin, cos)) in table.iter().enumerate() {
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * cos - b * sin;
        x[2 * i + 1] = a * sin + b * cos;
    }
}

/// [`rope_neox`] over a precomputed per-position table.
pub fn rope_neox_cached(x: &mut [f32], table: &[(f32, f32)]) {
    assert_eq!(x.len(), table.len() * 2);
    let half = table.len();
    for (i, &(sin, cos)) in table.iter().enumerate() {
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
    }
}

/// [`rope_norm`] over a precomputed per-position table stored as arrays.
pub fn rope_norm_cached_from_array(x: &mut [f32], table: &[[f32; 2]]) {
    assert_eq!(x.len(), table.len() * 2);
    // SAFETY: `[f32; 2]` and `(f32, f32)` have identical size, alignment, and layout.
    let pairs =
        unsafe { std::slice::from_raw_parts(table.as_ptr() as *const (f32, f32), table.len()) };
    rope_norm_cached(x, pairs);
}

/// [`rope_neox`] over a precomputed per-position table stored as arrays.
///
/// The table may cover only a leading slice of the head (partial rotary,
/// GLM): those channels rotate and the tail passes through unchanged.
pub fn rope_neox_cached_from_array(x: &mut [f32], table: &[[f32; 2]]) {
    assert!(x.len() >= table.len() * 2);
    let rot_len = table.len() * 2;
    // SAFETY: `[f32; 2]` and `(f32, f32)` have identical size, alignment, and layout.
    let pairs =
        unsafe { std::slice::from_raw_parts(table.as_ptr() as *const (f32, f32), table.len()) };
    rope_neox_cached(&mut x[..rot_len], pairs);
}

/// Rotary embedding, "NORM" style: adjacent pairs `(x[2i], x[2i+1])` rotate
/// together. Llama 3.x uses this.
///
/// `x` is one head of dimension `head_dim` at sequence position `pos`;
/// `freq_base` is the model's `rope.freq_base` (10000 classically, 500000 for
/// Llama 3). This form recomputes the frequencies per call; the forward pass
/// uses the `_cached` variants above, checked against this one in tests.
pub fn rope_norm(x: &mut [f32], pos: u32, freq_base: f32) {
    let d = x.len();
    assert!(d % 2 == 0);
    for i in 0..d / 2 {
        let theta = pos as f32 * freq_base.powf(-2.0 * i as f32 / d as f32);
        let (sin, cos) = theta.sin_cos();
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * cos - b * sin;
        x[2 * i + 1] = a * sin + b * cos;
    }
}

/// Rotary embedding, "NEOX" style: split pairs `(x[i], x[i + d/2])` rotate
/// together. Qwen and most recent non-Llama models use this.
pub fn rope_neox(x: &mut [f32], pos: u32, freq_base: f32) {
    let d = x.len();
    assert!(d % 2 == 0);
    let half = d / 2;
    for i in 0..half {
        let theta = pos as f32 * freq_base.powf(-2.0 * i as f32 / d as f32);
        let (sin, cos) = theta.sin_cos();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
    }
}
