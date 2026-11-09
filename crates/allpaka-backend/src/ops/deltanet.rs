//! Gated DeltaNet recurrent step kernel (`gated_deltanet_step`).

/// One gated-delta-net step over `heads` value-heads of `d`-dimensional state
/// (llama.cpp `build_delta_net_autoregressive` /
/// `kernel_gated_delta_net_impl` semantics):
///
/// ```text
/// S *= exp(g);  sk = S . k;  d = (v - sk) * beta;  S += k . d;  y = (S . q) / sqrt(d)
/// ```
///
/// `state` is `heads` row-major `d x d` matrices; q and k carry `heads_k`
/// heads, mapped onto the value heads as `h % heads_k` (qwen35moe: 32
/// value heads over 16 key heads).
#[allow(clippy::too_many_arguments)]
pub fn gated_deltanet_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    out: &mut [f32],
) {
    let heads = gate.len();
    let d = v.len() / heads;
    let heads_k = k.len() / d;
    assert_eq!(state.len(), heads * d * d);
    assert_eq!(q.len(), heads_k * d);
    assert_eq!(out.len(), v.len());
    let scale = 1.0 / (d as f32).sqrt();
    for h in 0..heads {
        let kh = h % heads_k;
        let g = gate[h].exp();
        let b = beta[h];
        let k_h = &k[kh * d..(kh + 1) * d];
        let q_h = &q[kh * d..(kh + 1) * d];
        let v_h = &v[h * d..(h + 1) * d];
        let out_h = &mut out[h * d..(h + 1) * d];
        let s = &mut state[h * d * d..(h + 1) * d * d];
        for i in 0..d {
            let row = &mut s[i * d..(i + 1) * d];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let vg = vdupq_n_f32(g);
                let mut vsk0 = vdupq_n_f32(0.0);
                let mut vsk1 = vdupq_n_f32(0.0);
                let mut j = 0usize;
                let rp = row.as_mut_ptr();
                let kp = k_h.as_ptr();
                let qp = q_h.as_ptr();
                while j + 8 <= d {
                    let r0 = vmulq_f32(vld1q_f32(rp.add(j)), vg);
                    let r1 = vmulq_f32(vld1q_f32(rp.add(j + 4)), vg);
                    vst1q_f32(rp.add(j), r0);
                    vst1q_f32(rp.add(j + 4), r1);
                    vsk0 = vfmaq_f32(vsk0, r0, vld1q_f32(kp.add(j)));
                    vsk1 = vfmaq_f32(vsk1, r1, vld1q_f32(kp.add(j + 4)));
                    j += 8;
                }
                let mut sk = vaddvq_f32(vaddq_f32(vsk0, vsk1));
                while j < d {
                    let rj = *rp.add(j) * g;
                    *rp.add(j) = rj;
                    sk += rj * *kp.add(j);
                    j += 1;
                }
                let delta = (v_h[i] - sk) * b;
                let vdelta = vdupq_n_f32(delta);
                let mut vy0 = vdupq_n_f32(0.0);
                let mut vy1 = vdupq_n_f32(0.0);
                j = 0;
                while j + 8 <= d {
                    let r0 = vfmaq_f32(vld1q_f32(rp.add(j)), vld1q_f32(kp.add(j)), vdelta);
                    let r1 = vfmaq_f32(vld1q_f32(rp.add(j + 4)), vld1q_f32(kp.add(j + 4)), vdelta);
                    vst1q_f32(rp.add(j), r0);
                    vst1q_f32(rp.add(j + 4), r1);
                    vy0 = vfmaq_f32(vy0, r0, vld1q_f32(qp.add(j)));
                    vy1 = vfmaq_f32(vy1, r1, vld1q_f32(qp.add(j + 4)));
                    j += 8;
                }
                let mut y = vaddvq_f32(vaddq_f32(vy0, vy1));
                while j < d {
                    let rj = *rp.add(j) + *kp.add(j) * delta;
                    *rp.add(j) = rj;
                    y += rj * *qp.add(j);
                    j += 1;
                }
                out_h[i] = y * scale;
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut sk = 0.0f32;
                for (rj, &kj) in row.iter_mut().zip(k_h) {
                    *rj *= g;
                    sk += *rj * kj;
                }
                let delta = (v_h[i] - sk) * b;
                let mut y = 0.0f32;
                for ((rj, &kj), &qj) in row.iter_mut().zip(k_h).zip(q_h) {
                    *rj += kj * delta;
                    y += *rj * qj;
                }
                out_h[i] = y * scale;
            }
        }
    }
}
