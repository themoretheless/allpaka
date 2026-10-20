//! A quantised weight matrix and its matmul.
//!
//! The whole reason decode is affordable is that weights stay quantised in
//! memory and are expanded row by row on the way through the dot product. A
//! reference that first dequantised the entire matrix to f32 would need more
//! RAM than the model itself and would not exercise the layout the real
//! kernels use.

use crate::ops::dot;
use allpaka_gguf::{dequant, GgmlType};
use anyhow::{bail, Result};

/// A `[n_out, n_in]` weight matrix over quantised bytes, borrowed straight
/// from the GGUF mmap.
pub struct QuantMat<'a> {
    data: &'a [u8],
    ty: GgmlType,
    pub n_out: usize,
    pub n_in: usize,
    row_bytes: usize,
}

impl<'a> QuantMat<'a> {
    pub fn new(data: &'a [u8], ty: GgmlType, n_out: usize, n_in: usize) -> Result<Self> {
        let (be, bb) = match (ty.block_elements(), ty.block_bytes()) {
            (Some(be), Some(bb)) => (be as usize, bb as usize),
            _ => bail!("unsupported ggml type {ty:?}"),
        };
        if n_in % be != 0 {
            bail!("row length {n_in} is not a whole number of {be}-element blocks");
        }
        let row_bytes = n_in / be * bb;
        let expected = row_bytes
            .checked_mul(n_out)
            .filter(|&total| total == data.len());
        if expected.is_none() {
            bail!(
                "{n_out}x{n_in} of {ty:?} needs {} bytes, got {}",
                row_bytes * n_out,
                data.len()
            );
        }
        Ok(Self {
            data,
            ty,
            n_out,
            n_in,
            row_bytes,
        })
    }

    /// One weight row dequantised to f32.
    pub fn row(&self, j: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.n_in];
        self.row_into(j, &mut out)?;
        Ok(out)
    }

    /// Dequantise weight row `j` directly into `dst` without allocating a new `Vec<f32>`.
    pub fn row_into(&self, j: usize, dst: &mut [f32]) -> Result<()> {
        if j >= self.n_out {
            bail!("row {j} out of {}", self.n_out);
        }
        if dst.len() != self.n_in {
            bail!(
                "row_into destination has {} elements, expected {}",
                dst.len(),
                self.n_in
            );
        }
        let bytes = &self.data[j * self.row_bytes..(j + 1) * self.row_bytes];
        dequant::dequant_into(self.ty, bytes, dst)
    }

    /// A contiguous band of rows as its own matrix, borrowing the same bytes.
    ///
    /// This is how a stacked-expert tensor `[n_in, n_out, n_expert]` is
    /// addressed: expert `e` is `slice_rows(e * n_out, n_out)` of the whole
    /// stack, no copy involved.
    pub fn slice_rows(&self, start: usize, count: usize) -> Result<QuantMat<'a>> {
        if start + count > self.n_out {
            bail!("rows {start}..{} out of {}", start + count, self.n_out);
        }
        Ok(QuantMat {
            data: &self.data[start * self.row_bytes..(start + count) * self.row_bytes],
            ty: self.ty,
            n_out: count,
            n_in: self.n_in,
            row_bytes: self.row_bytes,
        })
    }

    /// `y = x · Wᵀ`: for `x` of shape `[m, n_in]`, produces `[m, n_out]`.
    /// Project owned activations directly from shared Metal storage when possible.
    /// CPU and unsupported-format fallbacks retain the ordinary slice contract.
    pub fn matmul_activation(
        &self,
        x: &crate::activation::Activation,
        offset: usize,
        m: usize,
    ) -> Result<Vec<f32>> {
        let end = offset
            .checked_add(
                m.checked_mul(self.n_in)
                    .ok_or_else(|| anyhow::anyhow!("activation shape overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("activation offset overflow"))?;
        let values = x
            .get(offset..end)
            .ok_or_else(|| anyhow::anyhow!("activation range out of bounds"))?;
        #[cfg(target_os = "macos")]
        if let crate::activation::Activation::Shared(shared) = x {
            if let Some(y) = crate::gpu::matvec_shared(
                self.ty, self.data, self.n_in, self.n_out, shared, offset, m,
            ) {
                return Ok(y);
            }
        }
        self.matmul(values, m)
    }

    pub fn matmul(&self, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if x.len() != m * self.n_in {
            bail!("x has {} values, expected {m} x {}", x.len(), self.n_in);
        }
        if m == 1 {
            return Ok(self.matvec(x));
        }
        // Batched: the GPU reads each weight row once for the whole batch.
        if let Some(y) = crate::gpu::matvec_batch(&[crate::gpu::MatvecReq {
            ty: self.ty,
            w: self.data,
            n_in: self.n_in,
            n_out: self.n_out,
            x,
            m,
        }]) {
            return Ok(y.into_iter().next().unwrap());
        }
        // CPU fallback: evaluate rows in parallel using fused f32 row_dot
        // without allocating a Vec<f32> per weight row.
        const PARALLEL_THRESHOLD_ELEMENTS: usize = 2 << 20;
        let mut y = vec![0f32; m * self.n_out];
        if m * self.n_out * self.n_in < PARALLEL_THRESHOLD_ELEMENTS {
            for i in 0..m {
                let xi = &x[i * self.n_in..(i + 1) * self.n_in];
                let y_row = &mut y[i * self.n_out..(i + 1) * self.n_out];
                for (j, out) in y_row.iter_mut().enumerate() {
                    *out = self.row_dot(j, xi);
                }
            }
            return Ok(y);
        }

        let threads = cpu_threads();
        if m >= threads {
            let rows_per = m.div_ceil(threads).max(1);
            std::thread::scope(|scope| {
                for (chunk_idx, y_chunk) in y.chunks_mut(rows_per * self.n_out).enumerate() {
                    let first_i = chunk_idx * rows_per;
                    scope.spawn(move || {
                        for (local_i, y_row) in y_chunk.chunks_mut(self.n_out).enumerate() {
                            let i = first_i + local_i;
                            let xi = &x[i * self.n_in..(i + 1) * self.n_in];
                            for (j, out) in y_row.iter_mut().enumerate() {
                                *out = self.row_dot(j, xi);
                            }
                        }
                    });
                }
            });
        } else {
            let cols_per = self.n_out.div_ceil(threads).max(1);
            std::thread::scope(|scope| {
                for (i, y_row) in y.chunks_mut(self.n_out).enumerate() {
                    let xi = &x[i * self.n_in..(i + 1) * self.n_in];
                    for (chunk_idx, y_chunk) in y_row.chunks_mut(cols_per).enumerate() {
                        let first_j = chunk_idx * cols_per;
                        scope.spawn(move || {
                            for (local_j, out) in y_chunk.iter_mut().enumerate() {
                                *out = self.row_dot(first_j + local_j, xi);
                            }
                        });
                    }
                }
            });
        }
        Ok(y)
    }

    /// The decode hot path: one activation row against every weight row,
    /// output rows split across all cores.
    ///
    /// For Q8_0/Q4_K/Q6_K the activations are quantised to Q8 once per call
    /// and every row dot runs on i8×i8 integer arithmetic - the same trick
    /// llama.cpp uses, worth several times over the f32-expansion path.
    /// Other formats dequantise row-by-row. Sizes were validated at
    /// construction, so per-row dequantisation cannot fail here.
    fn matvec(&self, x: &[f32]) -> Vec<f32> {
        if self.n_out == 0 || self.n_in == 0 {
            return Vec::new();
        }
        // The GPU takes any matvec whose weights live in the attached mmap
        // and whose format has a kernel; everything else falls through to
        // the CPU reference below.
        if let Some(y) = crate::gpu::matvec(self.ty, self.data, self.n_in, self.n_out, x) {
            return y;
        }
        let acts = self.quantized_acts(x);
        let acts = acts.as_ref().map(|acts| acts.row(0, self.n_in));
        // Threading pays for itself only on big matrices. A MoE decodes
        // through hundreds of small expert matvecs per token, and spawning a
        // thread team for each costs more than the arithmetic; those run
        // single-threaded here and get their parallelism one level up, across
        // experts.
        const PARALLEL_THRESHOLD_ELEMENTS: usize = 2 << 20;
        if self.n_out * self.n_in < PARALLEL_THRESHOLD_ELEMENTS {
            let mut y = vec![0f32; self.n_out];
            for (j, out) in y.iter_mut().enumerate() {
                *out = self.row_dot_dispatch(j, x, acts.as_ref());
            }
            return y;
        }

        let threads = cpu_threads();
        let mut y = vec![0f32; self.n_out];
        let rows_per = self.n_out.div_ceil(threads).max(1);

        std::thread::scope(|scope| {
            for (chunk_index, y_chunk) in y.chunks_mut(rows_per).enumerate() {
                let first_row = chunk_index * rows_per;
                scope.spawn(move || {
                    for (i, out) in y_chunk.iter_mut().enumerate() {
                        *out = self.row_dot_dispatch(first_row + i, x, acts.as_ref());
                    }
                });
            }
        });
        y
    }

    /// Q8 activations for this matrix's integer kernels, when the format has
    /// one and the row length is whole blocks of 32.
    fn quantized_acts(&self, x: &[f32]) -> Option<Q8Acts> {
        match self.ty {
            GgmlType::Q8_0
            | GgmlType::Q5_0
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
                if self.n_in % 32 == 0 =>
            {
                Some(Q8Acts::from_f32(x))
            }
            _ => None,
        }
    }

    fn row_dot_dispatch(&self, j: usize, x: &[f32], acts: Option<&Q8ActsSlice<'_>>) -> f32 {
        match acts {
            Some(a) => {
                let bytes = &self.data[j * self.row_bytes..(j + 1) * self.row_bytes];
                match self.ty {
                    GgmlType::Q8_0 => dot_q8_0_i8(bytes, a),
                    GgmlType::Q5_0 => dot_q5_0_i8(bytes, a),
                    GgmlType::Q2K => dot_q2_k_i8(bytes, a),
                    GgmlType::Q3K => dot_q3_k_i8(bytes, a),
                    GgmlType::Q4K => dot_q4_k_i8(bytes, a),
                    GgmlType::Q5K => dot_q5_k_i8(bytes, a),
                    GgmlType::Q6K => dot_q6_k_i8(bytes, a),
                    _ => unreachable!("acts only exist for integer-kernel types"),
                }
            }
            None => self.row_dot(j, x),
        }
    }

    /// Run several independent matmuls at once. Each item's batch size is
    /// implied by its activation length: `m = x.len() / n_in`.
    ///
    /// On the GPU this is one command buffer and one wait for the whole set -
    /// the difference between the accelerator helping and the accelerator
    /// drowning in dispatch latency. Without a GPU the items run on parallel
    /// CPU threads instead.
    pub fn matmul_many(items: &[(&QuantMat<'_>, &[f32])]) -> Result<Vec<Vec<f32>>> {
        for (m, x) in items {
            if x.is_empty() || x.len() % m.n_in != 0 {
                bail!(
                    "x has {} values, expected a multiple of {}",
                    x.len(),
                    m.n_in
                );
            }
        }
        if crate::gpu::is_attached() {
            if items.len() <= 4 {
                let mut stack_reqs: [std::mem::MaybeUninit<crate::gpu::MatvecReq>; 4] =
                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                for (i, (m, x)) in items.iter().enumerate() {
                    stack_reqs[i].write(crate::gpu::MatvecReq {
                        ty: m.ty,
                        w: m.data,
                        n_in: m.n_in,
                        n_out: m.n_out,
                        x,
                        m: x.len() / m.n_in,
                    });
                }
                let reqs = unsafe {
                    std::slice::from_raw_parts(
                        stack_reqs.as_ptr() as *const crate::gpu::MatvecReq,
                        items.len(),
                    )
                };
                if let Some(out) = crate::gpu::matvec_batch(reqs) {
                    return Ok(out);
                }
            } else {
                let reqs: Vec<crate::gpu::MatvecReq> = items
                    .iter()
                    .map(|(m, x)| crate::gpu::MatvecReq {
                        ty: m.ty,
                        w: m.data,
                        n_in: m.n_in,
                        n_out: m.n_out,
                        x,
                        m: x.len() / m.n_in,
                    })
                    .collect();
                if let Some(out) = crate::gpu::matvec_batch(&reqs) {
                    return Ok(out);
                }
            }
        }
        // CPU fallback: one thread per item, rows expanded once per batch.
        // FFN gate/up projections commonly share the same activation slice,
        // so retain one quantised copy for those items instead of repeating
        // the activation quantisation for every projection.
        let mut cached_acts: Vec<(usize, usize, usize, Q8Acts)> = Vec::with_capacity(items.len());
        let mut act_indices: Vec<Option<usize>> = Vec::with_capacity(items.len());
        for (mat, x) in items {
            let key = (x.as_ptr() as usize, x.len(), mat.n_in);
            if let Some(idx) = cached_acts
                .iter()
                .position(|(ptr, len, n_in, _)| (*ptr, *len, *n_in) == key)
            {
                act_indices.push(Some(idx));
            } else if let Some(quantized) = mat.quantized_acts(x) {
                let idx = cached_acts.len();
                cached_acts.push((key.0, key.1, key.2, quantized));
                act_indices.push(Some(idx));
            } else {
                act_indices.push(None);
            }
        }
        let cached_ref = &cached_acts;
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        std::thread::scope(|scope| {
            for (((mat, x), slot), act_idx) in items.iter().zip(out.iter_mut()).zip(act_indices) {
                let acts_ref = act_idx.map(|idx| &cached_ref[idx].3);
                scope.spawn(move || {
                    let rows = x.len() / mat.n_in;
                    let mut y = vec![0f32; rows * mat.n_out];
                    for i in 0..rows {
                        let xi = &x[i * mat.n_in..(i + 1) * mat.n_in];
                        let row_acts = acts_ref.map(|acts| acts.row(i, mat.n_in));
                        for j in 0..mat.n_out {
                            y[i * mat.n_out + j] = mat.row_dot_dispatch(j, xi, row_acts.as_ref());
                        }
                    }
                    *slot = y;
                });
            }
        });
        Ok(out)
    }

    /// Run whole FFNs `swiglu(x·Gᵀ, x·Uᵀ)·Dᵀ` on the GPU as one command
    /// buffer: gate, up, the elementwise swiglu and down encode together and
    /// pay a single wait. Items are `(gate, up, down, x)` with the batch size
    /// implied by `x.len()`.
    ///
    /// A mixed batch is not all-or-nothing: if the combined GPU dispatch
    /// declines, each item is retried independently and only unsupported
    /// items use the ordinary matmul + swiglu + matmul path.
    pub fn ffn_many(
        items: &[(&QuantMat<'_>, &QuantMat<'_>, &QuantMat<'_>, &[f32])],
    ) -> Option<Vec<Vec<f32>>> {
        if !crate::gpu::is_attached() {
            return None;
        }
        let mut reqs = Vec::with_capacity(items.len());
        for (gate, up, down, x) in items {
            let hidden = gate.n_in;
            let ffn = gate.n_out;
            if up.n_in != hidden
                || up.n_out != ffn
                || down.n_in != ffn
                || down.n_out != hidden
                || x.is_empty()
                || x.len() % hidden != 0
            {
                return None;
            }
            reqs.push(crate::gpu::FfnReq {
                gate_ty: gate.ty,
                gate_w: gate.data,
                up_ty: up.ty,
                up_w: up.data,
                down_ty: down.ty,
                down_w: down.data,
                hidden,
                ffn,
                x,
                m: x.len() / hidden,
            });
        }
        if let Some(out) = crate::gpu::ffn_batch(&reqs) {
            return Some(out);
        }

        let mut out = Vec::with_capacity(items.len());
        for (item, req) in items.iter().zip(&reqs) {
            if let Some(mut one) = crate::gpu::ffn_batch(std::slice::from_ref(req)) {
                out.push(one.pop().expect("one FFN result per request"));
                continue;
            }

            let (gate, up, down, x) = *item;
            let m = x.len() / gate.n_in;
            let mut activated = gate.matmul(x, m).ok()?;
            let up = up.matmul(x, m).ok()?;
            crate::ops::swiglu(&mut activated, &up);
            out.push(down.matmul(&activated, m).ok()?);
        }
        Some(out)
    }

    /// Attention for one decode step followed by this matrix as the output
    /// projection, in one command buffer. `self` is `wo`.
    ///
    /// Returns None when the GPU declines (no device, a head width the kernel
    /// is not written for, foreign bytes), and the caller then runs the CPU
    /// attention and an ordinary matmul.
    pub fn attend_project(&self, req: &crate::gpu::AttnReq) -> Option<Vec<f32>> {
        crate::gpu::attend_project(req, self.ty, self.data, self.n_out)
    }

    /// The raw quantised bytes and их формат - what the GPU paths address.
    pub fn raw(&self) -> (GgmlType, &[u8]) {
        (self.ty, self.data)
    }

    /// One decode layer's whole attention half in one command buffer; see
    /// `gpu::attn_block`. The four matrices are `self`-less because the call
    /// spans all of them equally.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_block(
        wq: &QuantMat,
        wk: &QuantMat,
        wv: &QuantMat,
        wo: &QuantMat,
        x: &[f32],
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        rope: &[[f32; 2]],
        eps: f32,
        cache: &crate::gpu::SharedRegion,
        offs: (usize, usize),
        dims: (usize, usize, usize, usize),
        pos: usize,
        scale: f32,
    ) -> Option<Vec<f32>> {
        let (kv_dim, head_dim, n_heads, n_kv_heads) = dims;
        crate::gpu::attn_block(&crate::gpu::AttnBlockReq {
            wq: (wq.ty, wq.data, wq.n_out),
            wk: (wk.ty, wk.data, wk.n_out),
            wv: (wv.ty, wv.data, wv.n_out),
            wo: (wo.ty, wo.data, wo.n_out),
            x,
            q_norm,
            k_norm,
            rope,
            eps,
            cache,
            k_off: offs.0,
            v_off: offs.1,
            kv_dim,
            head_dim,
            n_heads,
            n_kv_heads,
            pos,
            scale,
        })
    }

    /// A prefill chunk's attention half in one command buffer; see
    /// `gpu::prefill_attn_block`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attn_block(
        wq: &QuantMat,
        wk: &QuantMat,
        wv: &QuantMat,
        wo: &QuantMat,
        hs: &[f32],
        m: usize,
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        ropes: &[[f32; 2]],
        rot_dim: usize,
        gate_in_q: bool,
        attn_bias: Option<(&[u8], &[u8], &[u8])>,
        eps: f32,
        cache: &crate::gpu::SharedRegion,
        offs: (usize, usize),
        dims: (usize, usize, usize, usize),
        base: usize,
        scale: f32,
        fusion: Option<crate::gpu::PrefillFusion>,
    ) -> Option<Vec<f32>> {
        let (kv_dim, head_dim, n_heads, n_kv_heads) = dims;
        crate::gpu::prefill_attn_block(&crate::gpu::PrefillAttnReq {
            wq: (wq.ty, wq.data, wq.n_out),
            wk: (wk.ty, wk.data, wk.n_out),
            wv: (wv.ty, wv.data, wv.n_out),
            wo: (wo.ty, wo.data, wo.n_out),
            hs,
            m,
            q_norm,
            k_norm,
            ropes,
            rot_dim,
            gate_in_q,
            attn_bias,
            eps,
            cache,
            k_off: offs.0,
            v_off: offs.1,
            kv_dim,
            head_dim,
            n_heads,
            n_kv_heads,
            base,
            scale,
            fusion,
        })
    }

    /// A prefill chunk of one gated-delta-net layer; see
    /// `gpu::prefill_gdn_block`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_gdn_block(
        wqkv: &QuantMat,
        zgate: &QuantMat,
        alpha: &QuantMat,
        beta: &QuantMat,
        conv1d_raw: &[u8],
        a: &[f32],
        dt: &[f32],
        ssm_norm: &[f32],
        ssm_out: &QuantMat,
        dims: (usize, usize, usize, usize),
        hidden: usize,
        m: usize,
        eps: f32,
        ssm: &crate::gpu::SharedRegion,
        ssm_slots: Option<(&crate::gpu::SharedRegion, usize)>,
        offs: (usize, usize),
        fusion: Option<crate::gpu::PrefillFusion>,
    ) -> Option<Vec<f32>> {
        let (heads_k, heads_v, d, d_conv) = dims;
        crate::gpu::prefill_gdn_block(&crate::gpu::PrefillGdnReq {
            wqkv: (wqkv.ty, wqkv.data, wqkv.n_out),
            zgate: (zgate.ty, zgate.data, zgate.n_out),
            alpha: alpha.data,
            beta: beta.data,
            conv1d: conv1d_raw,
            a,
            dt,
            ssm_norm,
            ssm_out: (ssm_out.ty, ssm_out.data, ssm_out.n_out),
            heads_k,
            heads_v,
            d,
            d_conv,
            hidden,
            m,
            eps,
            ssm,
            ssm_slots,
            conv_off: offs.0,
            state_off: offs.1,
            fusion,
        })
    }

    /// A prefill chunk's whole routed FFN in one command buffer; see
    /// `gpu::ffn_batch_grouped`. The three mats are the full stacked expert
    /// tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_grouped(
        gate: &QuantMat,
        up: &QuantMat,
        down: &QuantMat,
        n_expert: usize,
        hidden: usize,
        ffn: usize,
        groups: &[[u32; 3]],
        x: &[f32],
        tok: &[u32],
        total_rows: usize,
        fused: Option<crate::gpu::GroupedCombine>,
        shared: Option<(&QuantMat, &QuantMat, &QuantMat, Option<&QuantMat>)>,
        route: Option<crate::gpu::GroupedRoute>,
    ) -> Option<Vec<f32>> {
        let shared = shared.map(|(g, u, d, go)| crate::gpu::GroupedShared {
            gate: (g.ty, g.data),
            up: (u.ty, u.data),
            down: (d.ty, d.data),
            ffn: g.n_out,
            gate_out: go.map(|g| (g.ty, g.data)),
        });
        crate::gpu::ffn_batch_grouped(&crate::gpu::GroupedFfnReq {
            gate: (gate.ty, gate.data),
            up: (up.ty, up.data),
            down: (down.ty, down.data),
            n_expert,
            hidden,
            ffn,
            groups,
            x,
            tok,
            total_rows,
            fused,
            shared,
            route,
        })
    }

    /// One output element: weight row `j` dotted with `x`, via the fused
    /// kernel for the format when one exists.
    fn row_dot(&self, j: usize, x: &[f32]) -> f32 {
        let bytes = &self.data[j * self.row_bytes..(j + 1) * self.row_bytes];
        match self.ty {
            GgmlType::Q8_0 => dot_q8_0(bytes, x),
            GgmlType::Q5_0 => dot_q5_0(bytes, x),
            GgmlType::Q2K => dot_q2_k(bytes, x),
            GgmlType::Q3K => dot_q3_k(bytes, x),
            GgmlType::Q4K => dot_q4_k(bytes, x),
            GgmlType::Q5K => dot_q5_k(bytes, x),
            GgmlType::Q6K => dot_q6_k(bytes, x),
            GgmlType::F16 => dot_f16(bytes, x),
            GgmlType::BF16 => dot_bf16(bytes, x),
            // The router of every MoE layer is F32, and it is on the decode
            // hot path: unpacking it a byte quadruple at a time cost more
            // than the whole routed FFN's CPU share.
            GgmlType::F32 => match as_f32(bytes) {
                Some(row) => dot(row, x),
                None => bytes
                    .chunks_exact(4)
                    .zip(x)
                    .map(|(c, xv)| f32::from_le_bytes(c.try_into().unwrap()) * xv)
                    .sum(),
            },
            _ => {
                let row = dequant::dequant(self.ty, bytes, self.n_in)
                    .expect("row geometry was validated at construction");
                dot(&row, x)
            }
        }
    }
}

#[inline]
fn cpu_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

#[inline(always)]
fn dot_f16(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_endian = "little")]
    {
        let (head, mid, tail) = unsafe { row.align_to::<u16>() };
        if head.is_empty() && tail.is_empty() {
            return crate::ops::f16::dot(x, mid);
        }
    }
    row.chunks_exact(2)
        .zip(x)
        .map(|(c, &xv)| dequant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])) * xv)
        .sum()
}

#[inline(always)]
fn dot_bf16(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        let n = x.len();
        let rp = row.as_ptr();
        let xp = x.as_ptr();
        while i + 8 <= n {
            let bits = vld1q_u16(rp.add(i * 2) as *const u16);
            let f0 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bits)));
            let f1 = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bits));
            acc0 = vfmaq_f32(acc0, f0, vld1q_f32(xp.add(i)));
            acc1 = vfmaq_f32(acc1, f1, vld1q_f32(xp.add(i + 4)));
            i += 8;
        }
        let mut total = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < n {
            let c0 = *rp.add(i * 2);
            let c1 = *rp.add(i * 2 + 1);
            total += dequant::bf16_to_f32(u16::from_le_bytes([c0, c1])) * *xp.add(i);
            i += 1;
        }
        total
    }
    #[cfg(not(target_arch = "aarch64"))]
    row.chunks_exact(2)
        .zip(x)
        .map(|(c, &xv)| dequant::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) * xv)
        .sum()
}

#[inline(always)]
fn dot_q5_0(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(22).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
        let qs = &block[6..22];
        let xs = &x[bi * 32..bi * 32 + 32];
        let mut s = 0f32;
        for j in 0..16 {
            let x0 = ((qs[j] & 0x0f) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
            s += x0 as f32 * xs[j] + x1 as f32 * xs[j + 16];
        }
        total += d * s;
    }
    total
}

#[inline(always)]
fn dot_q3_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(110).enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let packed = &block[96..108];
        let d_all = dequant::f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let xb = &x[bi * 256..bi * 256 + 256];

        let mut scales = [0i32; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            let lo = if i < 8 {
                packed[i] & 0xf
            } else {
                packed[i - 8] >> 4
            };
            let hi = (packed[8 + i % 4] >> (2 * (i / 4))) & 3;
            *s = ((lo | (hi << 4)) as i32) - 32;
        }

        let mut is = 0;
        let mut m: u8 = 1;
        let mut pos = 0;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            for shift in [0u8, 2, 4, 6] {
                for group in 0..2 {
                    let dl = d_all * scales[is] as f32;
                    let mut sub = 0f32;
                    for l in 0..16 {
                        let idx = group * 16 + l;
                        let quant = ((q[idx] >> shift) & 3) as i32
                            - if hmask[idx] & m != 0 { 0 } else { 4 };
                        sub += quant as f32 * xb[pos + l];
                    }
                    total += dl * sub;
                    pos += 16;
                    is += 1;
                }
                m <<= 1;
            }
        }
    }
    total
}

#[inline(always)]
fn dot_q5_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(176).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let xb = &x[bi * 256..bi * 256 + 256];

        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let x_lo = &xb[pair * 64..pair * 64 + 32];
            let x_hi = &xb[pair * 64 + 32..pair * 64 + 64];
            let (sc1, mn1) = dequant::scale_min_k4(pair * 2, packed);
            let (sc2, mn2) = dequant::scale_min_k4(pair * 2 + 1, packed);
            let bit1 = 1u8 << (pair * 2);
            let bit2 = 1u8 << (pair * 2 + 1);
            let mut dot_lo = 0f32;
            let mut sum_lo = 0f32;
            let mut dot_hi = 0f32;
            let mut sum_hi = 0f32;
            for l in 0..32 {
                let hi1 = if qh[l] & bit1 != 0 { 16 } else { 0 };
                let hi2 = if qh[l] & bit2 != 0 { 16 } else { 0 };
                dot_lo += ((q[l] & 0xf) + hi1) as f32 * x_lo[l];
                sum_lo += x_lo[l];
                dot_hi += ((q[l] >> 4) + hi2) as f32 * x_hi[l];
                sum_hi += x_hi[l];
            }
            total += d * sc1 * dot_lo - dmin * mn1 * sum_lo;
            total += d * sc2 * dot_hi - dmin * mn2 * sum_hi;
        }
    }
    total
}

/// Fused Q8_0 dot product: `Σ_blocks d * Σ q_i * x_i`, straight off the
/// quantised bytes.
#[inline(always)]
fn dot_q8_0(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(34).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let xs = &x[bi * 32..bi * 32 + 32];
        #[cfg(target_arch = "aarch64")]
        let s = unsafe {
            use std::arch::aarch64::*;
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let qp = block.as_ptr().add(2) as *const i8;
            let xp = xs.as_ptr();
            for seg in [0usize, 16] {
                let q8 = vld1q_s8(qp.add(seg));
                let q16_lo = vmovl_s8(vget_low_s8(q8));
                let q16_hi = vmovl_high_s8(q8);
                let qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo)));
                let qf1 = vcvtq_f32_s32(vmovl_high_s16(q16_lo));
                let qf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi)));
                let qf3 = vcvtq_f32_s32(vmovl_high_s16(q16_hi));
                acc0 = vfmaq_f32(acc0, qf0, vld1q_f32(xp.add(seg)));
                acc1 = vfmaq_f32(acc1, qf1, vld1q_f32(xp.add(seg + 4)));
                acc0 = vfmaq_f32(acc0, qf2, vld1q_f32(xp.add(seg + 8)));
                acc1 = vfmaq_f32(acc1, qf3, vld1q_f32(xp.add(seg + 12)));
            }
            vaddvq_f32(vaddq_f32(acc0, acc1))
        };
        #[cfg(not(target_arch = "aarch64"))]
        let s = {
            let mut s = 0f32;
            for (q, xv) in block[2..34].iter().zip(xs) {
                s += (*q as i8) as f32 * xv;
            }
            s
        };
        total += d * s;
    }
    total
}

/// Fused Q4_K dot product, following the same 8-sub-block layout as the
/// dequantiser: `Σ d*sc*Σ(q·x) - dmin*mn*Σx` per sub-block, so the min term
/// needs only the activation sum, not a per-element subtraction.
#[inline(always)]
fn dot_q4_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(144).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let qs = &block[16..144];
        let xb = &x[bi * 256..bi * 256 + 256];

        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let x_lo = &xb[pair * 64..pair * 64 + 32];
            let x_hi = &xb[pair * 64 + 32..pair * 64 + 64];
            let (sc1, mn1) = dequant::scale_min_k4(pair * 2, packed);
            let (sc2, mn2) = dequant::scale_min_k4(pair * 2 + 1, packed);

            // Unpack first, reduce second: single-purpose fixed-bound loops
            // are what the auto-vectoriser actually turns into NEON/AVX.
            let mut lo = [0f32; 32];
            let mut hi = [0f32; 32];
            for l in 0..32 {
                lo[l] = (q[l] & 0xf) as f32;
                hi[l] = (q[l] >> 4) as f32;
            }
            let mut dot_lo = 0f32;
            let mut sum_lo = 0f32;
            for l in 0..32 {
                dot_lo += lo[l] * x_lo[l];
                sum_lo += x_lo[l];
            }
            let mut dot_hi = 0f32;
            let mut sum_hi = 0f32;
            for l in 0..32 {
                dot_hi += hi[l] * x_hi[l];
                sum_hi += x_hi[l];
            }
            total += d * sc1 * dot_lo - dmin * mn1 * sum_lo;
            total += d * sc2 * dot_hi - dmin * mn2 * sum_hi;
        }
    }
    total
}

/// Fused Q2_K dot product: `Σ dl*Σ(q·x) - ml*Σx` per 16-element sub-block.
#[inline(always)]
fn dot_q2_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(84).enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let xb = &x[bi * 256..bi * 256 + 256];

        let mut is = 0;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            for shift in [0u8, 2, 4, 6] {
                for group in 0..2 {
                    let sc = scales[is];
                    let dl = d * (sc & 0xf) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    let x0 = half * 128 + (shift as usize / 2) * 32 + group * 16;
                    let xs = &xb[x0..x0 + 16];
                    let mut dotq = 0f32;
                    let mut sum = 0f32;
                    for l in 0..16 {
                        dotq += ((q[group * 16 + l] >> shift) & 3) as f32 * xs[l];
                        sum += xs[l];
                    }
                    total += dl * dotq - ml * sum;
                    is += 1;
                }
            }
        }
    }
    total
}

/// Fused Q6_K dot product, mirroring the dequantiser's half/quarter walk.
#[inline(always)]
fn dot_q6_k(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(210).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let xb = &x[bi * 256..bi * 256 + 256];

        for half in 0..2 {
            let ql = &ql[half * 64..half * 64 + 64];
            let qh = &qh[half * 32..half * 32 + 32];
            let sc = &scales[half * 8..half * 8 + 8];
            let xh = &xb[half * 128..half * 128 + 128];
            // Per-sub-block accumulators: quarter q covers xh[q*32..],
            // sub-block index within the half is q*... scale sc[is + 2q].
            let mut acc = [0f32; 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
                acc[is] += q1 as f32 * xh[l];
                acc[2 + is] += q2 as f32 * xh[l + 32];
                acc[4 + is] += q3 as f32 * xh[l + 64];
                acc[6 + is] += q4 as f32 * xh[l + 96];
            }
            for (g, a) in acc.iter().enumerate() {
                total += d * (sc[g] as i8) as f32 * a;
            }
        }
    }
    total
}

/// A borrowed f32 tensor as a slice, when the mapping is aligned for it.
///
/// GGUF aligns tensor data to 32 bytes and the mapping starts on a page, so
/// this succeeds for every real file; the fallback exists because a slice
/// cast that is wrong about alignment is undefined behaviour, not a wrong
/// answer.
fn as_f32(bytes: &[u8]) -> Option<&[f32]> {
    let (head, mid, tail) = unsafe { bytes.align_to::<f32>() };
    (head.is_empty() && tail.is_empty()).then_some(mid)
}

/// Activations quantised to signed 8-bit in blocks of 32:
/// `x[b*32+l] ≈ d[b] * q[b*32+l]`.
///
/// This is llama.cpp's Q8 trick: the weight side is already integers, so
/// quantising the activation side once per matvec turns every inner product
/// into i8×i8 integer math. `sum[b]` keeps the exact f32 block sum so the
/// Q4_K min term loses nothing to the activation quantisation.
struct Q8Acts {
    d: Vec<f32>,
    sum: Vec<f32>,
    q: Vec<i8>,
}

#[derive(Clone, Copy)]
struct Q8ActsSlice<'a> {
    d: &'a [f32],
    sum: &'a [f32],
    q: &'a [i8],
}

impl Q8Acts {
    fn from_f32(x: &[f32]) -> Self {
        debug_assert_eq!(x.len() % 32, 0);
        let blocks = x.len() / 32;
        let mut d = vec![0f32; blocks];
        let mut sum = vec![0f32; blocks];
        let mut q = vec![0i8; x.len()];
        for b in 0..blocks {
            let xs = &x[b * 32..b * 32 + 32];
            #[cfg(target_arch = "aarch64")]
            let (amax, block_sum) = unsafe {
                use std::arch::aarch64::*;
                let mut vmax0 = vdupq_n_f32(0.0);
                let mut vmax1 = vdupq_n_f32(0.0);
                let mut vsum0 = vdupq_n_f32(0.0);
                let mut vsum1 = vdupq_n_f32(0.0);
                let xp = xs.as_ptr();
                for k in (0..32).step_by(8) {
                    let v0 = vld1q_f32(xp.add(k));
                    let v1 = vld1q_f32(xp.add(k + 4));
                    vmax0 = vmaxq_f32(vmax0, vabsq_f32(v0));
                    vmax1 = vmaxq_f32(vmax1, vabsq_f32(v1));
                    vsum0 = vaddq_f32(vsum0, v0);
                    vsum1 = vaddq_f32(vsum1, v1);
                }
                (
                    vmaxvq_f32(vmaxq_f32(vmax0, vmax1)),
                    vaddvq_f32(vaddq_f32(vsum0, vsum1)),
                )
            };
            #[cfg(not(target_arch = "aarch64"))]
            let (amax, block_sum) = (
                xs.iter().fold(0f32, |m, v| m.max(v.abs())),
                xs.iter().sum::<f32>(),
            );
            let db = if amax.is_finite() { amax / 127.0 } else { 0.0 };
            let id = if db > 0.0 && db.is_finite() {
                1.0 / db
            } else {
                0.0
            };
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let vid = vdupq_n_f32(id);
                let xp = xs.as_ptr();
                let qp = q.as_mut_ptr().add(b * 32);
                for k in (0..32).step_by(16) {
                    let s0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp.add(k)), vid));
                    let s1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp.add(k + 4)), vid));
                    let s2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp.add(k + 8)), vid));
                    let s3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(xp.add(k + 12)), vid));
                    let h01 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
                    let h23 = vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3));
                    let b0123 = vcombine_s8(vqmovn_s16(h01), vqmovn_s16(h23));
                    vst1q_s8(qp.add(k), b0123);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (slot, v) in q[b * 32..b * 32 + 32].iter_mut().zip(xs) {
                *slot = (v * id).round() as i8;
            }
            d[b] = db;
            sum[b] = block_sum;
        }
        Self { d, sum, q }
    }

    #[inline]
    fn row(&self, row: usize, n_in: usize) -> Q8ActsSlice<'_> {
        let blocks = n_in / 32;
        let first = row * blocks;
        Q8ActsSlice {
            d: &self.d[first..first + blocks],
            sum: &self.sum[first..first + blocks],
            q: &self.q[first * 32..(first + blocks) * 32],
        }
    }
}

impl Q8ActsSlice<'_> {
    #[inline]
    fn block(&self, b: usize) -> &[i8] {
        &self.q[b * 32..b * 32 + 32]
    }
}

/// Q8_0 × Q8 activations: `Σ_blocks d_w * d_x * Σ q_w · q_x`.
#[inline(always)]
fn dot_q8_0_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(34).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // SAFETY: i8 and u8 share size and alignment; construct a 32-element i8 slice from pointer.
        let qw: &[i8] =
            unsafe { std::slice::from_raw_parts(block[2..34].as_ptr() as *const i8, 32) };
        total += d * acts.d[bi] * dot_i8(qw, acts.block(bi)) as f32;
    }
    total
}

/// Q5_0 × Q8 activations: 32 5-bit signed values (`q - 16`) per 22-byte block.
#[inline(always)]
fn dot_q5_0_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(22).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
        let qs = &block[6..22];
        let mut qw = [0i8; 32];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let qv = vld1q_u8(qs.as_ptr());
            let mask_0f = vdupq_n_u8(0x0f);
            let bias_16 = vdupq_n_s8(16);
            let bit_mask: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let vbm = vld1q_u8(bit_mask.as_ptr());
            let qh_b0 = vdupq_n_u8((qh & 0xff) as u8);
            let qh_b1 = vdupq_n_u8(((qh >> 8) & 0xff) as u8);
            let qh_b2 = vdupq_n_u8(((qh >> 16) & 0xff) as u8);
            let qh_b3 = vdupq_n_u8(((qh >> 24) & 0xff) as u8);
            let qh_lo_bytes = vcombine_u8(vget_low_u8(qh_b0), vget_low_u8(qh_b1));
            let qh_hi_bytes = vcombine_u8(vget_low_u8(qh_b2), vget_low_u8(qh_b3));
            let h0 = vandq_u8(vtstq_u8(qh_lo_bytes, vbm), vdupq_n_u8(0x10));
            let h1 = vandq_u8(vtstq_u8(qh_hi_bytes, vbm), vdupq_n_u8(0x10));
            let x0 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qv, mask_0f), h0)),
                bias_16,
            );
            let x1 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qv, 4), h1)),
                bias_16,
            );
            vst1q_s8(qw.as_mut_ptr(), x0);
            vst1q_s8(qw.as_mut_ptr().add(16), x1);
        }
        #[cfg(not(target_arch = "aarch64"))]
        for j in 0..16 {
            let x0 = ((qs[j] & 0x0f) | ((((qh >> j) & 1) as u8) << 4)) as i8 - 16;
            let x1 = ((qs[j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)) as i8 - 16;
            qw[j] = x0;
            qw[j + 16] = x1;
        }
        total += d * acts.d[bi] * dot_i8(&qw, acts.block(bi)) as f32;
    }
    total
}

/// Q2_K × Q8 activations: 256 2-bit weights per 84-byte super-block.
#[inline(always)]
fn dot_q2_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(84).enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let mut is = 0;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let q0 = vld1q_u8(q.as_ptr());
                let q1 = vld1q_u8(q.as_ptr().add(16));
                let m3 = vdupq_n_u8(3);
                let planes = [
                    (
                        vreinterpretq_s8_u8(vandq_u8(q0, m3)),
                        vreinterpretq_s8_u8(vandq_u8(q1, m3)),
                    ),
                    (
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 2), m3)),
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 2), m3)),
                    ),
                    (
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 4), m3)),
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 4), m3)),
                    ),
                    (
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 6), m3)),
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 6), m3)),
                    ),
                ];
                let mut qw = [0i8; 32];
                for (s_idx, &(p0, p1)) in planes.iter().enumerate() {
                    let blk = bi * 8 + half * 4 + s_idx;
                    let xq = acts.block(blk);
                    let dx = acts.d[blk];
                    vst1q_s8(qw.as_mut_ptr(), p0);
                    vst1q_s8(qw.as_mut_ptr().add(16), p1);
                    let xv0 = vld1q_s8(xq.as_ptr());
                    let xv1 = vld1q_s8(xq.as_ptr().add(16));
                    let sum0 = vaddlvq_s8(xv0) as f32 * dx;
                    let sum1 = vaddlvq_s8(xv1) as f32 * dx;
                    let dot0 = dot_i8(&qw[0..16], &xq[0..16]) as f32 * dx;
                    let dot1 = dot_i8(&qw[16..32], &xq[16..32]) as f32 * dx;
                    let sc0 = scales[is];
                    let sc1 = scales[is + 1];
                    total += (d * (sc0 & 0xf) as f32) * dot0 - (dmin * (sc0 >> 4) as f32) * sum0;
                    total += (d * (sc1 & 0xf) as f32) * dot1 - (dmin * (sc1 >> 4) as f32) * sum1;
                    is += 2;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (s_idx, shift) in [0u8, 2, 4, 6].into_iter().enumerate() {
                let blk = bi * 8 + half * 4 + s_idx;
                let xq = acts.block(blk);
                let dx = acts.d[blk];
                let mut qw = [0i8; 32];
                for l in 0..32 {
                    qw[l] = ((q[l] >> shift) & 3) as i8;
                }
                for group in 0..2 {
                    let sc = scales[is];
                    let dl = d * (sc & 0xf) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    let dotq = dot_i8(
                        &qw[group * 16..group * 16 + 16],
                        &xq[group * 16..group * 16 + 16],
                    ) as f32
                        * dx;
                    let sum_x = xq[group * 16..group * 16 + 16]
                        .iter()
                        .map(|&v| v as i32)
                        .sum::<i32>() as f32
                        * dx;
                    total += dl * dotq - ml * sum_x;
                    is += 1;
                }
            }
        }
    }
    total
}

/// Q3_K × Q8 activations: 256 signed 3-bit weights per 110-byte super-block.
#[inline(always)]
fn dot_q3_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(110).enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let packed = &block[96..108];
        let d_all = dequant::f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        let mut scales = [0i32; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            let lo = if i < 8 {
                packed[i] & 0xf
            } else {
                packed[i - 8] >> 4
            };
            let hi = (packed[8 + i % 4] >> (2 * (i / 4))) & 3;
            *s = ((lo | (hi << 4)) as i32) - 32;
        }

        let mut is = 0;
        let mut m: u8 = 1;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            for (s_idx, shift) in [0u8, 2, 4, 6].into_iter().enumerate() {
                let blk = bi * 8 + half * 4 + s_idx;
                let xq = acts.block(blk);
                let dx = acts.d[blk];
                let mut qw = [0i8; 32];
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use std::arch::aarch64::*;
                    let m3 = vdupq_n_u8(3);
                    let vm = vdupq_n_u8(m);
                    let four = vdupq_n_u8(4);
                    let vsh = vdupq_n_s8(-(shift as i8));
                    for seg in [0usize, 16] {
                        let qv = vld1q_u8(q.as_ptr().add(seg));
                        let hv = vld1q_u8(hmask.as_ptr().add(seg));
                        let crumbs = vandq_u8(vshlq_u8(qv, vsh), m3);
                        // Where hmask & m == 0, subtract 4 (i.e. vbicq_u8(four, vtstq_u8(hv, vm))).
                        let sub4 = vbicq_u8(four, vtstq_u8(hv, vm));
                        let val = vsubq_s8(vreinterpretq_s8_u8(crumbs), vreinterpretq_s8_u8(sub4));
                        vst1q_s8(qw.as_mut_ptr().add(seg), val);
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                for l in 0..32 {
                    qw[l] = (((q[l] >> shift) & 3) as i8) - if hmask[l] & m != 0 { 0 } else { 4 };
                }
                let dot0 = dot_i8(&qw[0..16], &xq[0..16]) as f32;
                let dot1 = dot_i8(&qw[16..32], &xq[16..32]) as f32;
                total += d_all * dx * (scales[is] as f32 * dot0 + scales[is + 1] as f32 * dot1);
                is += 2;
                m <<= 1;
            }
        }
    }
    total
}

/// Q5_K × Q8 activations: 256 5-bit weights per 176-byte super-block.
#[inline(always)]
fn dot_q5_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(176).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let (sc1, mn1) = dequant::scale_min_k4(pair * 2, packed);
            let (sc2, mn2) = dequant::scale_min_k4(pair * 2 + 1, packed);
            let bit_shift1 = (pair * 2) as u8;
            let bit_shift2 = (pair * 2 + 1) as u8;
            let mut lo = [0i8; 32];
            let mut hi = [0i8; 32];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let mask_0f = vdupq_n_u8(0x0f);
                let mask_10 = vdupq_n_u8(0x10);
                let bm1 = vdupq_n_u8(1u8 << bit_shift1);
                let bm2 = vdupq_n_u8(1u8 << bit_shift2);
                for seg in [0usize, 16] {
                    let qv = vld1q_u8(q.as_ptr().add(seg));
                    let qhv = vld1q_u8(qh.as_ptr().add(seg));
                    let h1 = vandq_u8(vtstq_u8(qhv, bm1), mask_10);
                    let h2 = vandq_u8(vtstq_u8(qhv, bm2), mask_10);
                    let vlo = vorrq_u8(vandq_u8(qv, mask_0f), h1);
                    let vhi = vorrq_u8(vshrq_n_u8(qv, 4), h2);
                    vst1q_s8(lo.as_mut_ptr().add(seg), vreinterpretq_s8_u8(vlo));
                    vst1q_s8(hi.as_mut_ptr().add(seg), vreinterpretq_s8_u8(vhi));
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for l in 0..32 {
                let h1 = ((qh[l] >> bit_shift1) & 1) << 4;
                let h2 = ((qh[l] >> bit_shift2) & 1) << 4;
                lo[l] = ((q[l] & 0x0f) | h1) as i8;
                hi[l] = ((q[l] >> 4) | h2) as i8;
            }
            let b_lo = bi * 8 + pair * 2;
            let b_hi = b_lo + 1;
            let dot_lo = dot_i8(&lo, acts.block(b_lo)) as f32 * acts.d[b_lo];
            let dot_hi = dot_i8(&hi, acts.block(b_hi)) as f32 * acts.d[b_hi];
            total += d * sc1 * dot_lo - dmin * mn1 * acts.sum[b_lo];
            total += d * sc2 * dot_hi - dmin * mn2 * acts.sum[b_hi];
        }
    }
    total
}

/// Q4_K × Q8 activations, same sub-block walk as `dot_q4_k` with the q·x
/// reduction done in integers and the min term taken off the exact block sum.
#[inline(always)]
fn dot_q4_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(144).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = dequant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let packed = &block[4..16];
        let qs = &block[16..144];

        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let (sc1, mn1) = dequant::scale_min_k4(pair * 2, packed);
            let (sc2, mn2) = dequant::scale_min_k4(pair * 2 + 1, packed);

            let mut lo = [0i8; 32];
            let mut hi = [0i8; 32];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let mask = vdupq_n_u8(0x0f);
                let q0 = vld1q_u8(q.as_ptr());
                let q1 = vld1q_u8(q.as_ptr().add(16));
                vst1q_s8(lo.as_mut_ptr(), vreinterpretq_s8_u8(vandq_u8(q0, mask)));
                vst1q_s8(
                    lo.as_mut_ptr().add(16),
                    vreinterpretq_s8_u8(vandq_u8(q1, mask)),
                );
                vst1q_s8(
                    lo_or_hi_ptr(&mut hi, 0),
                    vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)),
                );
                vst1q_s8(
                    lo_or_hi_ptr(&mut hi, 16),
                    vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)),
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            for l in 0..32 {
                lo[l] = (q[l] & 0xf) as i8;
                hi[l] = (q[l] >> 4) as i8;
            }
            let b_lo = bi * 8 + pair * 2;
            let b_hi = b_lo + 1;
            let dot_lo = dot_i8(&lo, acts.block(b_lo)) as f32 * acts.d[b_lo];
            let dot_hi = dot_i8(&hi, acts.block(b_hi)) as f32 * acts.d[b_hi];
            total += d * sc1 * dot_lo - dmin * mn1 * acts.sum[b_lo];
            total += d * sc2 * dot_hi - dmin * mn2 * acts.sum[b_hi];
        }
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn lo_or_hi_ptr(buf: &mut [i8; 32], offset: usize) -> *mut i8 {
    buf.as_mut_ptr().add(offset)
}

/// Q6_K × Q8 activations. Scales cover 16 elements, so each quarter's
/// 32-value dot splits into two 16-lane integer dots against the halves of
/// one activation block.
#[inline(always)]
fn dot_q6_k_i8(row: &[u8], acts: &Q8ActsSlice<'_>) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(210).enumerate() {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        for half in 0..2 {
            let ql = &ql[half * 64..half * 64 + 64];
            let qh = &qh[half * 32..half * 32 + 32];
            let sc = &scales[half * 8..half * 8 + 8];

            let mut quarters = [[0i8; 32]; 4];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let mask_0f = vdupq_n_u8(0x0f);
                let mask_30 = vdupq_n_u8(0x30);
                let bias_32 = vdupq_n_s8(32);
                for seg in [0usize, 16] {
                    let ql_lo = vld1q_u8(ql.as_ptr().add(seg));
                    let ql_hi = vld1q_u8(ql.as_ptr().add(32 + seg));
                    let qh_v = vld1q_u8(qh.as_ptr().add(seg));

                    let q0_u = vorrq_u8(
                        vandq_u8(ql_lo, mask_0f),
                        vandq_u8(vshlq_n_u8(qh_v, 4), mask_30),
                    );
                    let q1_u = vorrq_u8(
                        vandq_u8(ql_hi, mask_0f),
                        vandq_u8(vshlq_n_u8(qh_v, 2), mask_30),
                    );
                    let q2_u = vorrq_u8(vshrq_n_u8(ql_lo, 4), vandq_u8(qh_v, mask_30));
                    let q3_u =
                        vorrq_u8(vshrq_n_u8(ql_hi, 4), vandq_u8(vshrq_n_u8(qh_v, 2), mask_30));

                    vst1q_s8(
                        quarters[0].as_mut_ptr().add(seg),
                        vsubq_s8(vreinterpretq_s8_u8(q0_u), bias_32),
                    );
                    vst1q_s8(
                        quarters[1].as_mut_ptr().add(seg),
                        vsubq_s8(vreinterpretq_s8_u8(q1_u), bias_32),
                    );
                    vst1q_s8(
                        quarters[2].as_mut_ptr().add(seg),
                        vsubq_s8(vreinterpretq_s8_u8(q2_u), bias_32),
                    );
                    vst1q_s8(
                        quarters[3].as_mut_ptr().add(seg),
                        vsubq_s8(vreinterpretq_s8_u8(q3_u), bias_32),
                    );
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for l in 0..32 {
                quarters[0][l] = (((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8) - 32;
                quarters[1][l] = (((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8) - 32;
                quarters[2][l] = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8) - 32;
                quarters[3][l] = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8) - 32;
            }
            for (qi, qq) in quarters.iter().enumerate() {
                let blk = bi * 8 + half * 4 + qi;
                let xq = acts.block(blk);
                let dot0 = dot_i8(&qq[0..16], &xq[0..16]) as f32;
                let dot1 = dot_i8(&qq[16..32], &xq[16..32]) as f32;
                let sc0 = (sc[2 * qi] as i8) as f32;
                let sc1 = (sc[2 * qi + 1] as i8) as f32;
                total += d * acts.d[blk] * (sc0 * dot0 + sc1 * dot1);
            }
        }
    }
    total
}

/// i8×i8 dot product over equal-length slices, accumulated in i32.
///
/// On aarch64 with `dotprod` (ARMv8.2+, all Apple Silicon M1–M4), uses
/// `vdotq_s32` (`SDOT`) to compute 16 `i8×i8` products per instruction.
#[inline(always)]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_feature = "dotprod")
    ))]
    unsafe {
        dot_i8_sdot(a, b)
    }
    #[cfg(all(
        target_arch = "aarch64",
        not(any(target_os = "macos", target_feature = "dotprod"))
    ))]
    // SAFETY: NEON is baseline on aarch64; loads stay within the slices.
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = vdupq_n_s32(0);
        let mut i = 0;
        while i + 16 <= a.len() {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
            acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
            i += 16;
        }
        let mut total = vaddvq_s32(acc);
        while i < a.len() {
            total += a[i] as i32 * b[i] as i32;
            i += 1;
        }
        total
    }
    #[cfg(not(target_arch = "aarch64"))]
    a.iter().zip(b).map(|(&p, &q)| p as i32 * q as i32).sum()
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_feature = "dotprod")
))]
#[target_feature(enable = "dotprod")]
#[allow(clippy::incompatible_msrv)]
#[inline]
unsafe fn dot_i8_sdot(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;
    while i + 16 <= a.len() {
        let va = vld1q_s8(a.as_ptr().add(i));
        let vb = vld1q_s8(b.as_ptr().add(i));
        acc = vdotq_s32(acc, va, vb);
        i += 16;
    }
    let mut total = vaddvq_s32(acc);
    while i < a.len() {
        total += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quantise values to q8_0 in the test itself, so the matmul is checked
    /// against independently computed numbers rather than against the
    /// dequantiser.
    fn q8_0_block(values: &[f32; 32]) -> ([u8; 34], [f32; 32]) {
        let amax = values.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut out = [0u8; 34];
        // f32 -> f16 for the scale, via the crate under test's inverse.
        let bits = half_bits(d);
        out[0..2].copy_from_slice(&bits.to_le_bytes());
        let mut exact = [0f32; 32];
        let d_back = allpaka_gguf::dequant::f16_to_f32(bits);
        for (i, v) in values.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            out[2 + i] = q as u8;
            exact[i] = d_back * q as f32;
        }
        (out, exact)
    }

    /// Minimal f32 -> f16 for test scales (normal values only).
    fn half_bits(x: f32) -> u16 {
        if x == 0.0 {
            return 0;
        }
        let b = x.to_bits();
        let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
        assert!((1..31).contains(&e));
        (((b >> 31) as u16) << 15) | ((e as u16) << 10) | ((b >> 13) & 0x3ff) as u16
    }

    #[test]
    fn a_quantised_matmul_matches_the_hand_computed_product() {
        // Two weight rows of 32, times two activation rows.
        let mut w0 = [0f32; 32];
        let mut w1 = [0f32; 32];
        for i in 0..32 {
            w0[i] = (i as f32) - 16.0;
            w1[i] = 1.0;
        }
        let (b0, e0) = q8_0_block(&w0);
        let (b1, e1) = q8_0_block(&w1);
        let data: Vec<u8> = b0.iter().chain(b1.iter()).copied().collect();

        let mat = QuantMat::new(&data, GgmlType::Q8_0, 2, 32).unwrap();
        let x: Vec<f32> = (0..64).map(|i| (i % 7) as f32 * 0.25).collect();
        let y = mat.matmul(&x, 2).unwrap();

        for i in 0..2 {
            let xi = &x[i * 32..(i + 1) * 32];
            let want0: f32 = e0.iter().zip(xi).map(|(a, b)| a * b).sum();
            let want1: f32 = e1.iter().zip(xi).map(|(a, b)| a * b).sum();
            assert!((y[i * 2] - want0).abs() < 1e-4, "{} vs {want0}", y[i * 2]);
            assert!((y[i * 2 + 1] - want1).abs() < 1e-4);
        }
    }

    #[test]
    fn f32_weights_make_the_matmul_exact() {
        // 3x2 weight in "f32 quantisation": y = x . Wt.
        let w = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mat = QuantMat::new(&bytes, GgmlType::F32, 3, 2).unwrap();
        let y = mat.matmul(&[1.0, 1.0], 1).unwrap();
        assert_eq!(y, vec![3.0, 7.0, 11.0]);
    }

    /// The fused Q4_K kernel must agree with dequantise-then-dot to float
    /// precision, on blocks with nontrivial scales, mins and both nibbles.
    #[test]
    fn fused_q4_k_dot_matches_dequant_then_dot() {
        // Build one 144-byte block with varied contents.
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&half_bits(0.25).to_le_bytes()); // d
        block[2..4].copy_from_slice(&half_bits(0.5).to_le_bytes()); // dmin
        for j in 0..12 {
            block[4 + j] = (7 + j as u8 * 13) & 0x3f; // packed scales/mins
        }
        for (i, b) in block[16..144].iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }

        let mat = QuantMat::new(&block, GgmlType::Q4K, 1, 256).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i % 13) as f32 - 6.0) * 0.3).collect();

        let fused = mat.matmul(&x, 1).unwrap()[0];
        let row = mat.row(0).unwrap();
        let reference: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        // The matvec path quantises activations to Q8, so the comparison
        // against the f32 reference carries that quantisation error too.
        assert!(
            (fused - reference).abs()
                < q8_act_error_bound(&row, &x) + 1e-3 * (1.0 + reference.abs()),
            "{fused} vs {reference}"
        );
    }

    #[test]
    fn fused_q2_k_dot_matches_dequant_then_dot() {
        let mut block = vec![0u8; 84];
        block[80..82].copy_from_slice(&half_bits(0.5).to_le_bytes()); // d
        block[82..84].copy_from_slice(&half_bits(0.25).to_le_bytes()); // dmin
        for (i, b) in block[0..80].iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(41).wrapping_add(3);
        }

        let mat = QuantMat::new(&block, GgmlType::Q2K, 1, 256).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i % 9) as f32 - 4.0) * 0.4).collect();

        let fused = mat.matmul(&x, 1).unwrap()[0];
        let row = mat.row(0).unwrap();
        let reference: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        // The matvec path quantises activations to Q8, so the comparison
        // against the f32 reference carries that quantisation error too.
        assert!(
            (fused - reference).abs()
                < q8_act_error_bound(&row, &x) + 1e-3 * (1.0 + reference.abs()),
            "{fused} vs {reference}"
        );
    }

    /// Worst-case error from rounding `x` to Q8 blocks of 32: half a
    /// quantisation step per element, weighted by |w|.
    fn q8_act_error_bound(w: &[f32], x: &[f32]) -> f32 {
        let mut bound = 0f32;
        for (wb, xb) in w.chunks(32).zip(x.chunks(32)) {
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let half_step = amax / 127.0 / 2.0;
            bound += half_step * wb.iter().map(|v| v.abs()).sum::<f32>();
        }
        bound
    }

    #[test]
    fn fused_q6_k_dot_matches_dequant_then_dot() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&half_bits(0.125).to_le_bytes());
        for (i, b) in block[0..192].iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(53).wrapping_add(7);
        }
        for (i, s) in block[192..208].iter_mut().enumerate() {
            *s = ((i as i32 * 17 % 97) - 48) as i8 as u8; // signed scales
        }

        let mat = QuantMat::new(&block, GgmlType::Q6K, 1, 256).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();

        let fused = mat.matmul(&x, 1).unwrap()[0];
        let row = mat.row(0).unwrap();
        let reference: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!(
            (fused - reference).abs()
                < q8_act_error_bound(&row, &x) + 1e-3 * (1.0 + reference.abs()),
            "{fused} vs {reference}"
        );
    }

    /// The integer Q8_0 matvec must stay within the activation-quantisation
    /// error bound of the exact dequantised product.
    #[test]
    fn integer_q8_0_matvec_matches_reference_within_bound() {
        let mut w0 = [0f32; 32];
        let mut w1 = [0f32; 32];
        for i in 0..32 {
            w0[i] = ((i as f32) - 16.0) * 0.11;
            w1[i] = ((i * 7 % 13) as f32 - 6.0) * 0.3;
        }
        let (b0, e0) = q8_0_block(&w0);
        let (b1, e1) = q8_0_block(&w1);
        let data: Vec<u8> = b0.iter().chain(b1.iter()).copied().collect();
        let mat = QuantMat::new(&data, GgmlType::Q8_0, 2, 32).unwrap();

        let x: Vec<f32> = (0..32).map(|i| ((i % 9) as f32 - 4.0) * 0.17).collect();
        let y = mat.matmul(&x, 1).unwrap();

        for (yi, exact) in y.iter().zip([&e0, &e1]) {
            let want: f32 = exact.iter().zip(&x).map(|(a, b)| a * b).sum();
            let bound = q8_act_error_bound(&exact[..], &x) + 1e-4;
            assert!((yi - want).abs() < bound, "{yi} vs {want} (bound {bound})");
        }
    }

    #[test]
    fn size_mismatches_are_errors_not_silent_wrap() {
        let bytes = vec![0u8; 34];
        assert!(
            QuantMat::new(&bytes, GgmlType::Q8_0, 2, 32).is_err(),
            "too few bytes"
        );
        assert!(
            QuantMat::new(&bytes, GgmlType::Q8_0, 1, 33).is_err(),
            "ragged row"
        );
        let mat = QuantMat::new(&bytes, GgmlType::Q8_0, 1, 32).unwrap();
        assert!(mat.matmul(&[0.0; 16], 1).is_err(), "x shorter than a row");
        assert!(mat.row(1).is_err());
    }
}
