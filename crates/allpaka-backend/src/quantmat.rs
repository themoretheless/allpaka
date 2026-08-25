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
        Ok(Self { data, ty, n_out, n_in, row_bytes })
    }

    /// One weight row dequantised to f32.
    pub fn row(&self, j: usize) -> Result<Vec<f32>> {
        if j >= self.n_out {
            bail!("row {j} out of {}", self.n_out);
        }
        let bytes = &self.data[j * self.row_bytes..(j + 1) * self.row_bytes];
        dequant::dequant(self.ty, bytes, self.n_in)
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
        // CPU reference: expand each row once, dot it with every batch row.
        let mut y = vec![0f32; m * self.n_out];
        for j in 0..self.n_out {
            let w = self.row(j)?;
            for i in 0..m {
                let xi = &x[i * self.n_in..(i + 1) * self.n_in];
                y[i * self.n_out + j] = dot(&w, xi);
            }
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
        // The GPU takes any matvec whose weights live in the attached mmap
        // and whose format has a kernel; everything else falls through to
        // the CPU reference below.
        if let Some(y) = crate::gpu::matvec(self.ty, self.data, self.n_in, self.n_out, x) {
            return y;
        }
        let acts = self.quantized_acts(x);
        let acts = acts.as_ref();
        // Threading pays for itself only on big matrices. A MoE decodes
        // through hundreds of small expert matvecs per token, and spawning a
        // thread team for each costs more than the arithmetic; those run
        // single-threaded here and get their parallelism one level up, across
        // experts.
        const PARALLEL_THRESHOLD_ELEMENTS: usize = 2 << 20;
        if self.n_out * self.n_in < PARALLEL_THRESHOLD_ELEMENTS {
            let mut y = vec![0f32; self.n_out];
            for (j, out) in y.iter_mut().enumerate() {
                *out = self.row_dot_dispatch(j, x, acts);
            }
            return y;
        }

        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let mut y = vec![0f32; self.n_out];
        let rows_per = self.n_out.div_ceil(threads).max(1);

        std::thread::scope(|scope| {
            for (chunk_index, y_chunk) in y.chunks_mut(rows_per).enumerate() {
                let first_row = chunk_index * rows_per;
                scope.spawn(move || {
                    for (i, out) in y_chunk.iter_mut().enumerate() {
                        *out = self.row_dot_dispatch(first_row + i, x, acts);
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
            GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K if self.n_in % 32 == 0 => {
                Some(Q8Acts::from_f32(x))
            }
            _ => None,
        }
    }

    fn row_dot_dispatch(&self, j: usize, x: &[f32], acts: Option<&Q8Acts>) -> f32 {
        match acts {
            Some(a) => {
                let bytes = &self.data[j * self.row_bytes..(j + 1) * self.row_bytes];
                match self.ty {
                    GgmlType::Q8_0 => dot_q8_0_i8(bytes, a),
                    GgmlType::Q4K => dot_q4_k_i8(bytes, a),
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
                bail!("x has {} values, expected a multiple of {}", x.len(), m.n_in);
            }
        }
        if crate::gpu::is_attached() {
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
        // CPU fallback: one thread per item, rows expanded once per batch.
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        std::thread::scope(|scope| {
            for ((mat, x), slot) in items.iter().zip(out.iter_mut()) {
                scope.spawn(move || {
                    let rows = x.len() / mat.n_in;
                    let mut y = vec![0f32; rows * mat.n_out];
                    for i in 0..rows {
                        let xi = &x[i * mat.n_in..(i + 1) * mat.n_in];
                        let acts = mat.quantized_acts(xi);
                        for j in 0..mat.n_out {
                            y[i * mat.n_out + j] = mat.row_dot_dispatch(j, xi, acts.as_ref());
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
    /// Returns None when the GPU declines (no device, foreign bytes, a
    /// format without a kernel, or mismatched shapes) - the caller then runs
    /// its usual matmul + swiglu + matmul path.
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
        crate::gpu::ffn_batch(&reqs)
    }

    /// Attention for one decode step followed by this matrix as the output
    /// projection, in one command buffer. `self` is `wo`.
    ///
    /// Returns None when the GPU declines (no device, a head width the kernel
    /// is not written for, foreign bytes), and the caller then runs the CPU
    /// attention and an ordinary matmul.
    #[cfg(target_os = "macos")]
    pub fn attend_project(&self, req: &crate::gpu::AttnReq) -> Option<Vec<f32>> {
        crate::gpu::attend_project(req, self.ty, self.data, self.n_out)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn attend_project(&self, _req: &crate::gpu::AttnReq) -> Option<Vec<f32>> {
        None
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
        shared: Option<(&QuantMat, &QuantMat, &QuantMat)>,
    ) -> Option<Vec<f32>> {
        let shared = shared.map(|(g, u, d)| crate::gpu::GroupedShared {
            gate: (g.ty, g.data),
            up: (u.ty, u.data),
            down: (d.ty, d.data),
            ffn: g.n_out,
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
        })
    }

    /// One output element: weight row `j` dotted with `x`, via the fused
    /// kernel for the format when one exists.
    fn row_dot(&self, j: usize, x: &[f32]) -> f32 {
        let bytes = &self.data[j * self.row_bytes..(j + 1) * self.row_bytes];
        match self.ty {
            GgmlType::Q8_0 => dot_q8_0(bytes, x),
            GgmlType::Q2K => dot_q2_k(bytes, x),
            GgmlType::Q4K => dot_q4_k(bytes, x),
            GgmlType::Q6K => dot_q6_k(bytes, x),
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

/// Fused Q8_0 dot product: `Σ_blocks d * Σ q_i * x_i`, straight off the
/// quantised bytes.
fn dot_q8_0(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(34).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let xs = &x[bi * 32..bi * 32 + 32];
        let mut s = 0f32;
        for (q, xv) in block[2..34].iter().zip(xs) {
            s += (*q as i8) as f32 * xv;
        }
        total += d * s;
    }
    total
}

/// Fused Q4_K dot product, following the same 8-sub-block layout as the
/// dequantiser: `Σ d*sc*Σ(q·x) - dmin*mn*Σx` per sub-block, so the min term
/// needs only the activation sum, not a per-element subtraction.
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

impl Q8Acts {
    fn from_f32(x: &[f32]) -> Self {
        debug_assert_eq!(x.len() % 32, 0);
        let blocks = x.len() / 32;
        let mut d = vec![0f32; blocks];
        let mut sum = vec![0f32; blocks];
        let mut q = vec![0i8; x.len()];
        for b in 0..blocks {
            let xs = &x[b * 32..b * 32 + 32];
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            let db = amax / 127.0;
            let id = if db > 0.0 { 1.0 / db } else { 0.0 };
            for (slot, v) in q[b * 32..b * 32 + 32].iter_mut().zip(xs) {
                *slot = (v * id).round() as i8;
            }
            d[b] = db;
            sum[b] = xs.iter().sum();
        }
        Self { d, sum, q }
    }

    #[inline]
    fn block(&self, b: usize) -> &[i8] {
        &self.q[b * 32..b * 32 + 32]
    }
}

/// Q8_0 × Q8 activations: `Σ_blocks d_w * d_x * Σ q_w · q_x`.
fn dot_q8_0_i8(row: &[u8], acts: &Q8Acts) -> f32 {
    let mut total = 0f32;
    for (bi, block) in row.chunks_exact(34).enumerate() {
        let d = dequant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // SAFETY: i8 and u8 share size and alignment; bytes are reinterpreted.
        let qw: &[i8] = unsafe { std::mem::transmute(&block[2..34]) };
        total += d * acts.d[bi] * dot_i8(qw, acts.block(bi)) as f32;
    }
    total
}

/// Q4_K × Q8 activations, same sub-block walk as `dot_q4_k` with the q·x
/// reduction done in integers and the min term taken off the exact block sum.
fn dot_q4_k_i8(row: &[u8], acts: &Q8Acts) -> f32 {
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

/// Q6_K × Q8 activations. Scales cover 16 elements, so each quarter's
/// 32-value dot splits into two 16-lane integer dots against the halves of
/// one activation block.
fn dot_q6_k_i8(row: &[u8], acts: &Q8Acts) -> f32 {
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
/// On aarch64 this is the NEON widening-multiply chain (`smull`/`sadalp`),
/// which is the whole point of quantising the activations: sixteen products
/// per instruction instead of one f32 FMA lane per element.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
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
            (fused - reference).abs() < q8_act_error_bound(&row, &x) + 1e-3 * (1.0 + reference.abs()),
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
            (fused - reference).abs() < q8_act_error_bound(&row, &x) + 1e-3 * (1.0 + reference.abs()),
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
        assert!(QuantMat::new(&bytes, GgmlType::Q8_0, 2, 32).is_err(), "too few bytes");
        assert!(QuantMat::new(&bytes, GgmlType::Q8_0, 1, 33).is_err(), "ragged row");
        let mat = QuantMat::new(&bytes, GgmlType::Q8_0, 1, 32).unwrap();
        assert!(mat.matmul(&[0.0; 16], 1).is_err(), "x shorter than a row");
        assert!(mat.row(1).is_err());
    }
}
