//! A quantised weight matrix and its matmul.
//!
//! The whole reason decode is affordable is that weights stay quantised in
//! memory and are expanded row by row on the way through the dot product. A
//! reference that first dequantised the entire matrix to f32 would need more
//! RAM than the model itself and would not exercise the layout the real
//! kernels use.

pub(crate) mod dot_f32;
#[cfg(target_arch = "aarch64")]
pub(crate) mod dot_neon;
pub(crate) mod q8_acts;

use crate::ops::dot;
use allpaka_dtype::GgmlType;
use allpaka_gguf::dequant;
use anyhow::{bail, Result};
use dot_f32::{
    as_f32, dot_bf16, dot_f16, dot_q2_k, dot_q3_k, dot_q4_k, dot_q5_0, dot_q5_k, dot_q6_k, dot_q8_0,
};
use q8_acts::{
    dot_q2_k_i8, dot_q3_k_i8, dot_q4_k_i8, dot_q5_0_i8, dot_q5_k_i8, dot_q6_k_i8, dot_q8_0_i8,
    Q8Acts, Q8ActsSlice,
};

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

        let threads = crate::thread_pool::cpu_threads();
        let y_addr = y.as_mut_ptr() as usize;
        if m >= threads {
            let n_out = self.n_out;
            let n_in = self.n_in;
            crate::thread_pool::par_for_chunks(m, threads, |_worker, start_i, end_i| {
                let y_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        (y_addr as *mut f32).add(start_i * n_out),
                        (end_i - start_i) * n_out,
                    )
                };
                for (local_i, y_row) in y_slice.chunks_mut(n_out).enumerate() {
                    let i = start_i + local_i;
                    let xi = &x[i * n_in..(i + 1) * n_in];
                    for (j, out) in y_row.iter_mut().enumerate() {
                        *out = self.row_dot(j, xi);
                    }
                }
            });
        } else {
            let total_outputs = m * self.n_out;
            let n_out = self.n_out;
            let n_in = self.n_in;
            crate::thread_pool::par_for_chunks(total_outputs, threads, |_worker, start, end| {
                let y_slice = unsafe {
                    std::slice::from_raw_parts_mut((y_addr as *mut f32).add(start), end - start)
                };
                for (offset, out) in y_slice.iter_mut().enumerate() {
                    let flat = start + offset;
                    let i = flat / n_out;
                    let j = flat % n_out;
                    let xi = &x[i * n_in..(i + 1) * n_in];
                    *out = self.row_dot(j, xi);
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
        let threads = crate::thread_pool::cpu_threads();
        if self.n_out * self.n_in < PARALLEL_THRESHOLD_ELEMENTS || threads <= 1 {
            let mut y = vec![0f32; self.n_out];
            for (j, out) in y.iter_mut().enumerate() {
                *out = self.row_dot_dispatch(j, x, acts.as_ref());
            }
            return y;
        }

        let mut y = vec![0f32; self.n_out];
        let y_addr = y.as_mut_ptr() as usize;
        crate::thread_pool::par_for_chunks(self.n_out, threads, |_worker, start_row, end_row| {
            let y_chunk = unsafe {
                std::slice::from_raw_parts_mut(
                    (y_addr as *mut f32).add(start_row),
                    end_row - start_row,
                )
            };
            for (i, out) in y_chunk.iter_mut().enumerate() {
                *out = self.row_dot_dispatch(start_row + i, x, acts.as_ref());
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
        // CPU fallback: persistent worker pool across items, rows expanded once per batch.
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
        let act_indices_ref = &act_indices;
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        let out_addr = out.as_mut_ptr() as usize;
        let threads = crate::thread_pool::cpu_threads().min(items.len().max(1));
        crate::thread_pool::par_for_chunks(items.len(), threads, |_worker, start, end| {
            let slots = unsafe {
                std::slice::from_raw_parts_mut((out_addr as *mut Vec<f32>).add(start), end - start)
            };
            for (local_idx, slot) in slots.iter_mut().enumerate() {
                let idx = start + local_idx;
                let (mat, x) = items[idx];
                let acts_ref = act_indices_ref[idx].map(|a_idx| &cached_ref[a_idx].3);
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
