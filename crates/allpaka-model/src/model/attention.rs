//! CPU/GPU causal grouped-query attention (`GQA`), QK-norm, RoPE, and KV store.

use super::{
    cpu_attn_enabled, cpu_threads, loader::add_bias_rows, trace, AttnBias, AttnWeights, Model,
};
use crate::config::{Config, RopeStyle};
use crate::kv::KvCache;
use crate::profile;
use allpaka_backend::{ops, thread_pool::par_chunks_mut};
use anyhow::Result;

impl<'a> Model<'a> {
    /// Rotate one head by a per-position table from [`ops::rope_sin_cos`].
    #[inline]
    pub(crate) fn rope_from_arrays(&self, head: &mut [f32], table: &[[f32; 2]]) {
        match self.config.rope_style {
            RopeStyle::Norm => ops::rope_norm_cached_from_array(head, table),
            RopeStyle::Neox => ops::rope_neox_cached_from_array(head, table),
        }
    }

    /// Unified CPU/hybrid attention execution from projected `(q, k, v)` rows:
    /// 1. Deinterleaves `gate_in_q` if enabled (`qwen35moe`).
    /// 2. Adds optional QKV projection biases (`GLM`).
    /// 3. Applies per-head `q_norm` / `k_norm` and RoPE rotation.
    /// 4. Stores `(k, v)` into `kv` (`store_at` or `store_at_rewrite`).
    /// 5. Runs fused/batched GPU attention if allowed and available, falling back
    ///    to parallel grouped online-softmax CPU attention on the persistent thread pool.
    /// 6. Applies the `gate_in_q` sigmoid gate and `wo` output projection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_cpu_attention_from_qkv(
        &self,
        aw: &AttnWeights<'a>,
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        bias: Option<&AttnBias<'a>>,
        mut q: Vec<f32>,
        mut k: Vec<f32>,
        mut v: Vec<f32>,
        m: usize,
        base_pos: usize,
        rope_flat: &[[f32; 2]],
        kv: &mut KvCache,
        kv_layer: usize,
        rewrite_kv: bool,
        allow_gpu_attend: bool,
        trace_layer: Option<usize>,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let head_dim = c.head_dim as usize;
        let q_dim = c.q_dim();
        let kv_dim = c.kv_dim();
        let half = c.rope_dim as usize / 2;

        let gate: Vec<f32> = if aw.gate_in_q {
            let mut gq = Vec::with_capacity(m * q_dim);
            let mut gg = Vec::with_capacity(m * q_dim);
            for row in q.chunks(2 * q_dim) {
                for h in 0..c.n_heads as usize {
                    gq.extend_from_slice(&row[h * 2 * head_dim..h * 2 * head_dim + head_dim]);
                    gg.extend_from_slice(&row[h * 2 * head_dim + head_dim..(h + 1) * 2 * head_dim]);
                }
            }
            q = gq;
            gg
        } else {
            Vec::new()
        };

        if let Some(b) = bias {
            add_bias_rows(&mut q, &b.q.0);
            add_bias_rows(&mut k, &b.k.0);
            add_bias_rows(&mut v, &b.v.0);
        }

        {
            let _s = profile::span(profile::Phase::QkNormRope);
            if let Some(w) = q_norm {
                for (i, row) in q.chunks_mut(q_dim).enumerate() {
                    let rope = &rope_flat[i * half..(i + 1) * half];
                    for head in row.chunks_mut(head_dim) {
                        ops::rmsnorm(head, w, c.rms_eps);
                        self.rope_from_arrays(head, rope);
                    }
                }
            } else {
                for (i, row) in q.chunks_mut(q_dim).enumerate() {
                    let rope = &rope_flat[i * half..(i + 1) * half];
                    for head in row.chunks_mut(head_dim) {
                        self.rope_from_arrays(head, rope);
                    }
                }
            }
            if let Some(w) = k_norm {
                for (i, row) in k.chunks_mut(kv_dim).enumerate() {
                    let rope = &rope_flat[i * half..(i + 1) * half];
                    for head in row.chunks_mut(head_dim) {
                        ops::rmsnorm(head, w, c.rms_eps);
                        self.rope_from_arrays(head, rope);
                    }
                }
            } else {
                for (i, row) in k.chunks_mut(kv_dim).enumerate() {
                    let rope = &rope_flat[i * half..(i + 1) * half];
                    for head in row.chunks_mut(head_dim) {
                        self.rope_from_arrays(head, rope);
                    }
                }
            }
        }

        if let Some(li) = trace_layer {
            trace("Qcur", li, &q[..q_dim]);
            trace("Kcur", li, &k[..kv_dim]);
            trace("Vcur", li, &v[..kv_dim]);
        }

        let cache_synced = {
            let _s = profile::span(profile::Phase::KvStore);
            if rewrite_kv {
                for i in 0..m {
                    kv.store_at_rewrite(
                        kv_layer,
                        base_pos + i,
                        &k[i * kv_dim..(i + 1) * kv_dim],
                        &v[i * kv_dim..(i + 1) * kv_dim],
                    );
                }
                false
            } else if m == 1 && trace_layer.is_none() {
                kv.store(kv_layer, &k, &v);
                true
            } else {
                for i in 0..m {
                    kv.store_at(
                        kv_layer,
                        base_pos + i,
                        &k[i * kv_dim..(i + 1) * kv_dim],
                        &v[i * kv_dim..(i + 1) * kv_dim],
                    );
                }
                kv.sync_gpu_range(kv_layer, base_pos..base_pos + m)
            }
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let cpu_only = !allow_gpu_attend || cpu_attn_enabled();

        // Single-token decode path: try fused attend + wo projection on GPU when !gate_in_q.
        if m == 1 && trace_layer.is_none() && !cpu_only && !aw.gate_in_q {
            let fused = {
                let _s = profile::span(profile::Phase::Attend);
                kv.gpu_view_ref(kv_layer).and_then(|(cache, k_off, v_off)| {
                    aw.wo.attend_project(&allpaka_backend::gpu::AttnReq {
                        cache,
                        k_off,
                        v_off,
                        q: &q,
                        kv_dim,
                        head_dim,
                        n_q_heads: c.n_heads as usize,
                        group: c.group_size(),
                        n_pos: base_pos + 1,
                        scale,
                    })
                })
            };
            if let Some(projected) = fused {
                return Ok(projected);
            }
        }

        let mut attn_out = vec![0.0f32; m * q_dim];
        let q_ref = &q;
        {
            let _s = profile::span(profile::Phase::Attend);
            let on_gpu = if m > 1 && !cpu_only && cache_synced {
                kv.gpu_view(kv_layer).and_then(|(cache, k_off, v_off)| {
                    let reqs: Vec<allpaka_backend::gpu::AttnReq> = (0..m)
                        .map(|i| allpaka_backend::gpu::AttnReq {
                            cache,
                            k_off,
                            v_off,
                            q: &q_ref[i * q_dim..(i + 1) * q_dim],
                            kv_dim,
                            head_dim,
                            n_q_heads: c.n_heads as usize,
                            group: c.group_size(),
                            n_pos: base_pos + i + 1,
                            scale,
                        })
                        .collect();
                    allpaka_backend::gpu::attend_batch(&reqs)
                })
            } else {
                None
            };

            match on_gpu {
                Some(rows) => {
                    for (out, row) in attn_out.chunks_mut(q_dim).zip(rows) {
                        out.copy_from_slice(&row);
                    }
                }
                None if m == 1 => {
                    let group_span = c.group_size() * head_dim;
                    let kv_ref = &*kv;
                    par_chunks_mut(&mut attn_out, group_span, |kv_head, _, out_group| {
                        attend_group(
                            c,
                            kv_ref,
                            kv_layer,
                            base_pos,
                            &q_ref[kv_head * group_span..(kv_head + 1) * group_span],
                            kv_head,
                            out_group,
                            scale,
                        );
                    });
                }
                None => {
                    let threads = cpu_threads();
                    let rows_per = m.div_ceil(threads).max(1);
                    let kv_ref = &*kv;
                    par_chunks_mut(&mut attn_out, rows_per * q_dim, |ci, _, out_chunk| {
                        let first = ci * rows_per;
                        for (i, out_row) in out_chunk.chunks_mut(q_dim).enumerate() {
                            let row = first + i;
                            attend_one(
                                c,
                                kv_ref,
                                kv_layer,
                                base_pos + row,
                                &q_ref[row * q_dim..(row + 1) * q_dim],
                                out_row,
                                scale,
                            );
                        }
                    });
                }
            }
        }

        if let Some(li) = trace_layer {
            trace("attn_pregate", li, &attn_out[..q_dim]);
        }
        if aw.gate_in_q {
            for (o, &gv) in attn_out.iter_mut().zip(&gate) {
                *o *= 1.0 / (1.0 + (-gv).exp());
            }
            if let Some(li) = trace_layer {
                trace("attn_gated", li, &attn_out[..q_dim]);
            }
        }
        let _s = profile::span(profile::Phase::AttnOut);
        aw.wo.matmul(&attn_out, m)
    }
}

/// One kv head attending for its whole GQA group in a single K/V pass.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_group(
    c: &Config,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    q_group: &[f32],
    kv_head: usize,
    out_group: &mut [f32],
    scale: f32,
) {
    let head_dim = c.head_dim as usize;
    let group = c.group_size();
    debug_assert_eq!(q_group.len(), group * head_dim);
    debug_assert_eq!(out_group.len(), group * head_dim);

    let mut stack_maxes = [f32::NEG_INFINITY; 32];
    let mut stack_denoms = [0f32; 32];
    let mut heap_maxes;
    let mut heap_denoms;
    let (maxes, denoms): (&mut [f32], &mut [f32]) = if group <= 32 {
        (&mut stack_maxes[..group], &mut stack_denoms[..group])
    } else {
        heap_maxes = vec![f32::NEG_INFINITY; group];
        heap_denoms = vec![0f32; group];
        (&mut heap_maxes, &mut heap_denoms)
    };

    for t in 0..=pos {
        let kt = kv.k_at(layer, t, kv_head, head_dim);
        let vt = kv.v_at(layer, t, kv_head, head_dim);

        for h in 0..group {
            let qh = &q_group[h * head_dim..(h + 1) * head_dim];
            let s = ops::f16::dot(qh, kt) * scale;

            let acc_start = h * head_dim;
            let acc = &mut out_group[acc_start..acc_start + head_dim];
            if s > maxes[h] {
                if denoms[h] > 0.0 {
                    let rescale = (maxes[h] - s).exp();
                    for a in acc.iter_mut() {
                        *a *= rescale;
                    }
                    denoms[h] *= rescale;
                }
                maxes[h] = s;
            }

            let p = (s - maxes[h]).exp();
            denoms[h] += p;
            ops::f16::axpy(acc, p, vt);
        }
    }

    for (h, &denom) in denoms.iter().enumerate() {
        let inv_d = 1.0 / denom.max(f32::MIN_POSITIVE);
        let acc = &mut out_group[h * head_dim..(h + 1) * head_dim];
        for a in acc.iter_mut() {
            *a *= inv_d;
        }
    }
}

/// All heads of one token row attending over the cache: one group pass per kv head.
pub(crate) fn attend_one(
    c: &Config,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    q_row: &[f32],
    out_row: &mut [f32],
    scale: f32,
) {
    let head_dim = c.head_dim as usize;
    let group = c.group_size();
    for kv_head in 0..c.n_kv_heads as usize {
        let at = kv_head * group * head_dim;
        attend_group(
            c,
            kv,
            layer,
            pos,
            &q_row[at..at + group * head_dim],
            kv_head,
            &mut out_row[at..at + group * head_dim],
            scale,
        );
    }
}

/// The naive reference: one q head, one full pass, buffered softmax. Kept
/// only to pin `attend_group` against an independently simple formulation.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn attend_head(
    c: &Config,
    kv: &KvCache,
    layer: usize,
    pos: usize,
    qh: &[f32],
    qi: usize,
    out: &mut [f32],
    scale: f32,
) {
    let head_dim = c.head_dim as usize;
    let kv_head = qi / c.group_size();
    let mut scores: Vec<f32> = (0..=pos)
        .map(|t| {
            let kt = kv.k_at(layer, t, kv_head, head_dim);
            qh.iter()
                .zip(kt)
                .map(|(a, &b)| a * ops::f16::to_f32(b))
                .sum::<f32>()
                * scale
        })
        .collect();
    ops::softmax(&mut scores);
    for (t, &p) in scores.iter().enumerate() {
        let vt = kv.v_at(layer, t, kv_head, head_dim);
        for (o, &vv) in out.iter_mut().zip(vt) {
            *o += p * ops::f16::to_f32(vv);
        }
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use crate::config::RopeStyle;

    fn attach_a_device_for_tests() -> bool {
        use std::sync::OnceLock;
        static REGION: OnceLock<bool> = OnceLock::new();
        *REGION.get_or_init(|| {
            const PAGE: usize = 16384;
            let layout = std::alloc::Layout::from_size_align(PAGE, PAGE).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            let region: &'static [u8] = unsafe { std::slice::from_raw_parts(ptr, PAGE) };
            allpaka_backend::gpu::attach(region)
        })
    }

    #[test]
    fn gpu_attention_matches_the_cpu_path() {
        if !attach_a_device_for_tests() {
            eprintln!("SKIP: no Metal device");
            return;
        }
        check_gpu_attention(8, 2, 71);
        check_gpu_attention(16, 4, 544);
    }

    fn check_gpu_attention(n_heads: u32, n_kv_heads: u32, positions: usize) {
        let head_dim = 128usize;
        let c = Config {
            architecture: "test".into(),
            n_layers: 1,
            hidden: n_heads * head_dim as u32,
            n_heads,
            n_kv_heads,
            head_dim: head_dim as u32,
            ffn_hidden: 8,
            vocab: 8,
            rms_eps: 1e-6,
            rope_freq_base: 10000.0,
            rope_style: RopeStyle::Neox,
            rope_sections: [0, 0, 0, 0],
            full_attention_interval: 0,
            ssm: None,
            has_qk_norm: false,
            moe: None,
            has_attn_bias: false,
            rope_dim: head_dim as u32,
            nextn: false,
        };
        attach_a_device_for_tests();

        let kv_dim = c.kv_dim();
        let mut kv = KvCache::new(1, kv_dim, positions);
        let mut state = 0x0fed_cba9_8765_4321u64;
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..positions {
            let k: Vec<f32> = (0..kv_dim).map(|_| rnd()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| rnd()).collect();
            kv.store(0, &k, &v);
            kv.advance();
        }
        let q: Vec<f32> = (0..c.q_dim()).map(|_| rnd()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let pos = positions - 1;

        let Some((cache, k_off, v_off)) = kv.gpu_view(0) else {
            panic!("no Metal device, or the cache was not wrapped for it");
        };
        let Some(got) = allpaka_backend::gpu::attend(&allpaka_backend::gpu::AttnReq {
            cache,
            k_off,
            v_off,
            q: &q,
            kv_dim,
            head_dim,
            n_q_heads: n_heads as usize,
            group: c.group_size(),
            n_pos: positions,
            scale,
        }) else {
            panic!("a GPU is attached but the kernel declined a 128-wide head");
        };

        let mut want = vec![0f32; c.q_dim()];
        attend_one(&c, &kv, 0, pos, &q, &mut want, scale);

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4 * (1.0 + w.abs()),
                "element {i}: gpu {g}, cpu {w}"
            );
        }
    }

    #[test]
    fn attend_group_matches_the_naive_reference() {
        let (n_heads, n_kv_heads, head_dim, positions) = (8u32, 2u32, 16usize, 33usize);
        let c = Config {
            architecture: "test".into(),
            n_layers: 1,
            hidden: n_heads * head_dim as u32,
            n_heads,
            n_kv_heads,
            head_dim: head_dim as u32,
            ffn_hidden: 8,
            vocab: 8,
            rms_eps: 1e-6,
            rope_freq_base: 10000.0,
            rope_style: RopeStyle::Neox,
            rope_sections: [0, 0, 0, 0],
            full_attention_interval: 0,
            ssm: None,
            has_qk_norm: false,
            moe: None,
            has_attn_bias: false,
            rope_dim: head_dim as u32,
            nextn: false,
        };
        let kv_dim = c.kv_dim();
        let mut kv = KvCache::new(1, kv_dim, positions);
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..positions {
            let k: Vec<f32> = (0..kv_dim).map(|_| rnd()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| rnd()).collect();
            kv.store(0, &k, &v);
            kv.advance();
        }
        let q: Vec<f32> = (0..c.q_dim()).map(|_| rnd()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let pos = positions - 1;

        let mut got = vec![0f32; c.q_dim()];
        attend_one(&c, &kv, 0, pos, &q, &mut got, scale);

        let mut want = vec![0f32; c.q_dim()];
        for qi in 0..n_heads as usize {
            attend_head(
                &c,
                &kv,
                0,
                pos,
                &q[qi * head_dim..(qi + 1) * head_dim],
                qi,
                &mut want[qi * head_dim..(qi + 1) * head_dim],
                scale,
            );
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-5 * (1.0 + w.abs()),
                "element {i}: {g} vs {w}"
            );
        }
    }
}
