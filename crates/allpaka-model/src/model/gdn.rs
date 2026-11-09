//! Gated Delta Net (`GDN`) linear-attention recurrence layer (`qwen35moe`).

use super::{gdn_zero_enabled, trace, Model, Session};
use allpaka_backend::{ops, quantmat::QuantMat};
use anyhow::Result;

pub(crate) struct GdnLayer<'a> {
    pub(crate) wqkv: QuantMat<'a>,
    pub(crate) zgate: QuantMat<'a>,
    pub(crate) alpha: QuantMat<'a>,
    pub(crate) beta: QuantMat<'a>,
    /// Pre-transposed `[d_conv][channels]` conv1d weights for contiguous SIMD FMA.
    pub(crate) conv1d_t: Vec<f32>,
    pub(crate) conv1d_raw: &'a [u8],
    pub(crate) a: Vec<f32>,
    pub(crate) dt: Vec<f32>,
    pub(crate) ssm_norm: Vec<f32>,
    pub(crate) ssm_out: QuantMat<'a>,
}

impl<'a> Model<'a> {
    /// The gated-delta-net branch of one layer over `m` consecutive tokens,
    /// updating the layer's conv window and deltanet state in the session.
    /// `hs` is the attn-normed input rows; returns the branch output
    /// (before the residual add), `m * hidden`.
    pub(crate) fn gdn_forward(
        &self,
        li: usize,
        g: &GdnLayer<'a>,
        hs: &[f32],
        m: usize,
        s: &mut Session,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        // Debug kill switch: zero the GDN branch entirely to isolate
        // attention/MoE bugs from deltanet bugs.
        if gdn_zero_enabled() {
            return Ok(vec![0f32; m * c.hidden as usize]);
        }
        let ssm = c.ssm.as_ref().expect("gdn layer without ssm config");
        let d_state = ssm.d_state as usize;
        let heads_k = ssm.n_group as usize;
        let heads_v = ssm.dt_rank as usize;
        let key_dim = heads_k * d_state;
        let value_dim = heads_v * d_state;
        let channels = key_dim * 2 + value_dim;
        let d_conv = ssm.d_conv as usize;
        let eps = c.rms_eps;

        // The two Q8_0 projections batch into one GPU submission; the F32
        // alpha/beta run on the CPU (the grouped GPU path excludes F32).
        let mut parts = QuantMat::matmul_many(&[(&g.wqkv, hs), (&g.zgate, hs)])?;
        let z = parts.pop().expect("z");
        let mut qkv = parts.pop().expect("qkv");
        let beta = g.beta.matmul(hs, m)?;
        let alpha = g.alpha.matmul(hs, m)?;
        trace("qkv_mixed", li, &qkv);
        trace("z", li, &z);
        trace("beta_raw", li, &beta);
        trace("alpha_raw", li, &alpha);

        let st = s.ssm.as_mut().expect("gdn session without ssm cache");
        let n_conv = (st.d_conv - 1) * st.conv_channels;
        let n_state = st.dt_rank * st.d_state * st.d_state;
        // MTP rollback slots: when armed, every row's window and state are
        // also mirrored into the row's slot (same pointers the slices below
        // alias; the copy happens once per row inside the loop).
        let slot_ptrs = st.slot_write_ptrs(li);
        let ssm_slot = st.slot_of(li);
        let (conv_all, state_all) = st.both_mut();
        let conv = &mut conv_all[ssm_slot * n_conv..(ssm_slot + 1) * n_conv];
        let state = &mut state_all[ssm_slot * n_state..(ssm_slot + 1) * n_state];
        let mut out = vec![0f32; m * value_dim];
        let mut c_out = vec![0f32; channels];
        let mut beta_i = vec![0f32; heads_v];
        let mut gate_i = vec![0f32; heads_v];
        for i in 0..m {
            // Depthwise causal conv over pre-transposed `g.conv1d_t` (`[d_conv][channels]`):
            // each tap is a contiguous vector of length `channels`, turning strided scalar
            // loads into sequential SIMD FMA passes.
            let cur = &qkv[i * channels..(i + 1) * channels];
            c_out.fill(0.0);
            for j in 0..d_conv - 1 {
                let win = &conv[j * channels..(j + 1) * channels];
                let w_tap = &g.conv1d_t[j * channels..(j + 1) * channels];
                for ((dst, &x), &w) in c_out.iter_mut().zip(win).zip(w_tap) {
                    *dst += x * w;
                }
            }
            let w_last = &g.conv1d_t[(d_conv - 1) * channels..d_conv * channels];
            for ((dst, &x), &w) in c_out.iter_mut().zip(cur).zip(w_last) {
                *dst = ops::silu(*dst + x * w);
            }
            // Shift the window one row left, append the raw (pre-conv) qkv
            // row: the cache holds the conv's *inputs*, not its outputs.
            conv.copy_within(channels.., 0);
            conv[(d_conv - 2) * channels..].copy_from_slice(cur);

            let row = &mut qkv[i * channels..(i + 1) * channels];
            row.copy_from_slice(&c_out);
            let (qk, v) = row.split_at_mut(2 * key_dim);
            let (q, k) = qk.split_at_mut(key_dim);
            for h in q.chunks_mut(d_state) {
                ops::l2_norm(h, eps);
            }
            for h in k.chunks_mut(d_state) {
                ops::l2_norm(h, eps);
            }
            for (dst, &v) in beta_i.iter_mut().zip(&beta[i * heads_v..(i + 1) * heads_v]) {
                *dst = 1.0 / (1.0 + (-v).exp());
            }
            for (((dst, &x), &dt), &a) in gate_i
                .iter_mut()
                .zip(&alpha[i * heads_v..(i + 1) * heads_v])
                .zip(&g.dt)
                .zip(&g.a)
            {
                *dst = ops::softplus(x + dt) * a;
            }
            ops::gated_deltanet_step(
                state,
                q,
                k,
                v,
                &gate_i,
                &beta_i,
                &mut out[i * value_dim..(i + 1) * value_dim],
            );
            if i == 0 {
                trace("conv_silu", li, &c_out);
                trace("gate", li, &gate_i);
                trace("beta_sig", li, &beta_i);
                trace("delta_out", li, &out[..value_dim]);
            }
            // Gated per-group RMS norm over the recurrence output, times
            // silu(z) (llama.cpp build_norm_gated).
            for h in 0..heads_v {
                let y = &mut out[i * value_dim + h * d_state..i * value_dim + (h + 1) * d_state];
                ops::rmsnorm(y, &g.ssm_norm, eps);
                let z_head = &z[i * value_dim + h * d_state..][..d_state];
                for (y, &zv) in y.iter_mut().zip(z_head) {
                    *y *= ops::silu(zv);
                }
            }
            // MTP rollback slot: this row's window and state, mirrored.
            if let Some((cp, sp, dp, conv_off, state_off, ncv, nst, total)) = slot_ptrs {
                unsafe {
                    let slot = dp.add(i * total);
                    std::ptr::copy_nonoverlapping(cp, slot.add(conv_off), ncv);
                    std::ptr::copy_nonoverlapping(sp, slot.add(state_off), nst);
                }
            }
            if i == 0 {
                trace("final_out", li, &out[..value_dim]);
            }
        }
        let projected = g.ssm_out.matmul(&out, m)?;
        trace("linear_attn_out", li, &projected);
        Ok(projected)
    }
}
