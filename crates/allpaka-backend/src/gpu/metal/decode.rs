//! Whole-token decode: one command buffer carries every layer from the
//! embedded token to logits or argmax, plus the on-device greedy chain.

use super::*;

/// A whole decode token - every layer's attention AND FFN, ending with the
/// output projection - as ONE command buffer with ONE wait.
///
/// The CPU's remaining role per token is: stage the embedding, encode, wait,
/// argmax. Routing runs on the GPU (softmax_topk), and the expert matmuls
/// dispatch INDEXED - the grid covers `n_used` expert slots whose ids only
/// ever exist in GPU memory. By the GPU's own clock, the per-buffer driver
/// scheduling plus round-trip idle this replaces was ~24 ms of a 74 ms
/// 235B token.
/// A resolved GDN branch's weights for the whole-token encoders
/// (decode_token and the speculative multi-token verify).
#[derive(Clone)]
pub(super) struct GdnRefs {
    pub(super) mats: [MatRef; 5], // wqkv, zgate, alpha, beta, ssm_out
    pub(super) states: [ComputePipelineState; 5],
    pub(super) conv1d: NormRef,
    pub(super) heads_k: u32,
    pub(super) heads_v: u32,
    pub(super) d: u32,
    pub(super) d_conv: u32,
    pub(super) conv_off: u64,
    pub(super) state_off: u64,
    pub(super) channels: usize,
    pub(super) value_dim: usize,
}

/// A resolved decode layer for the whole-token encoders.
#[derive(Clone)]
pub(super) struct LayerRefs {
    pub(super) attn_norm: NormRef,
    pub(super) ffn_norm: NormRef,
    /// None on GDN layers (no attention mats).
    pub(super) mats: Option<[MatRef; 4]>,
    pub(super) mat_states: Option<[ComputePipelineState; 4]>,
    pub(super) q_bias: Option<NormRef>,
    pub(super) k_bias: Option<NormRef>,
    pub(super) v_bias: Option<NormRef>,
    pub(super) gdn: Option<GdnRefs>,
    pub(super) ffn: FfnRefs,
    pub(super) normflag: bool,
}

#[derive(Clone)]
pub(super) struct CachedDecodeLayers {
    pub(super) key: u64,
    pub(super) layers: std::sync::Arc<[LayerRefs]>,
    pub(super) max_ffn: usize,
    pub(super) max_slots: usize,
    pub(super) max_expert: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_decode_layers(
    gpu: &mut Gpu,
    req: &TokenReq,
    hidden: usize,
    q_dim: usize,
    kv: usize,
    span: usize,
    allow_normflag: bool,
    dbg: bool,
) -> Option<CachedDecodeLayers> {
    use std::hash::{Hash, Hasher};
    macro_rules! why {
        ($cond:expr, $msg:expr) => {
            if $cond {
                if dbg {
                    eprintln!("tokenbuf declined: {}", $msg);
                }
                return None;
            }
        };
    }

    for l in req.layers {
        why!(
            (l.k_off + span) * 2 > req.cache.len || (l.v_off + span) * 2 > req.cache.len,
            "cache span"
        );
    }

    let sw_fuse_env = std::env::var("ALLPAKA_SWFUSE").is_ok_and(|v| v == "1");
    let runtime_normflag = crate::runtime::get().normflag;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hidden.hash(&mut hasher);
    q_dim.hash(&mut hasher);
    kv.hash(&mut hasher);
    allow_normflag.hash(&mut hasher);
    runtime_normflag.hash(&mut hasher);
    sw_fuse_env.hash(&mut hasher);
    mega_enabled().hash(&mut hasher);
    gufuse().hash(&mut hasher);
    shared_tail().hash(&mut hasher);
    gpu.chunks.len().hash(&mut hasher);
    req.layers.len().hash(&mut hasher);
    for l in req.layers {
        (l.attn_norm.as_ptr() as usize, l.attn_norm.len()).hash(&mut hasher);
        (l.ffn_norm.as_ptr() as usize, l.ffn_norm.len()).hash(&mut hasher);
        (
            std::mem::discriminant(&l.wq.0),
            l.wq.1.as_ptr() as usize,
            l.wq.2,
        )
            .hash(&mut hasher);
        (
            std::mem::discriminant(&l.wk.0),
            l.wk.1.as_ptr() as usize,
            l.wk.2,
        )
            .hash(&mut hasher);
        (
            std::mem::discriminant(&l.wv.0),
            l.wv.1.as_ptr() as usize,
            l.wv.2,
        )
            .hash(&mut hasher);
        (
            std::mem::discriminant(&l.wo.0),
            l.wo.1.as_ptr() as usize,
            l.wo.2,
        )
            .hash(&mut hasher);
        l.gate_in_q.hash(&mut hasher);
        l.q_bias.map(|b| b.as_ptr() as usize).hash(&mut hasher);
        l.k_bias.map(|b| b.as_ptr() as usize).hash(&mut hasher);
        l.v_bias.map(|b| b.as_ptr() as usize).hash(&mut hasher);
        if let Some(g) = &l.gdn {
            (
                std::mem::discriminant(&g.wqkv.0),
                g.wqkv.1.as_ptr() as usize,
                g.wqkv.2,
            )
                .hash(&mut hasher);
            (g.conv_off, g.state_off).hash(&mut hasher);
        }
        match &l.ffn {
            TokenFfn::Dense { gate, up, down } => {
                0u8.hash(&mut hasher);
                (
                    std::mem::discriminant(&gate.0),
                    gate.1.as_ptr() as usize,
                    gate.2,
                )
                    .hash(&mut hasher);
                (std::mem::discriminant(&up.0), up.1.as_ptr() as usize, up.2).hash(&mut hasher);
                (
                    std::mem::discriminant(&down.0),
                    down.1.as_ptr() as usize,
                    down.2,
                )
                    .hash(&mut hasher);
            }
            TokenFfn::Moe {
                router,
                gate,
                up,
                down,
                expert_ffn,
                n_used,
                sigmoid,
                ..
            } => {
                1u8.hash(&mut hasher);
                (router.1.as_ptr() as usize, router.2).hash(&mut hasher);
                (
                    std::mem::discriminant(&gate.0),
                    gate.1.as_ptr() as usize,
                    gate.1.len(),
                )
                    .hash(&mut hasher);
                (
                    std::mem::discriminant(&up.0),
                    up.1.as_ptr() as usize,
                    up.1.len(),
                )
                    .hash(&mut hasher);
                (
                    std::mem::discriminant(&down.0),
                    down.1.as_ptr() as usize,
                    down.1.len(),
                )
                    .hash(&mut hasher);
                (*expert_ffn, *n_used, *sigmoid).hash(&mut hasher);
            }
        }
    }
    let key = hasher.finish();
    if let Some(cached) = &gpu.decode_cache {
        if cached.key == key && cached.layers.len() == req.layers.len() {
            return Some(cached.clone());
        }
    }

    let resolve_or_decline =
        |label: &str, ty: GgmlType, w: &[u8], dbg: bool, gpu: &mut Gpu| -> Option<MatRef> {
            let Some(mat) = resolve(gpu, ty, w) else {
                if dbg {
                    eprintln!("tokenbuf declined: {label} resolve");
                }
                return None;
            };
            Some(mat)
        };
    let resolve_f32_or_decline =
        |label: &str, b: &[u8], n: usize, dbg: bool, gpu: &mut Gpu| -> Option<MatRef> {
            let Some(norm) = resolve_f32(gpu, b, n) else {
                if dbg {
                    eprintln!("tokenbuf declined: {label} norm resolve");
                }
                return None;
            };
            Some(norm)
        };

    let mut layers = Vec::with_capacity(req.layers.len());
    let mut max_ffn = 0usize;
    let mut max_slots = 1usize;
    let mut max_expert = 0usize;
    for l in req.layers {
        let gdn = match &l.gdn {
            None => None,
            Some(g) => {
                let key_dim = g.heads_k * g.d;
                let value_dim = g.heads_v * g.d;
                let channels = key_dim * 2 + value_dim;
                why!(
                    g.wqkv.2 != channels || g.zgate.2 != value_dim || g.ssm_out.2 != hidden,
                    "gdn dims"
                );
                why!(
                    g.alpha.len() != hidden * g.heads_v * 4
                        || g.beta.len() != hidden * g.heads_v * 4,
                    "gdn ab dims"
                );
                why!(g.conv1d.len() != channels * g.d_conv * 4, "gdn conv dims");
                why!(
                    g.a.len() != g.heads_v || g.dt.len() != g.heads_v || g.ssm_norm.len() != g.d,
                    "gdn vec dims"
                );
                let mats = [
                    resolve_or_decline("gdn.wqkv", g.wqkv.0, g.wqkv.1, dbg, gpu)?,
                    resolve_or_decline("gdn.zgate", g.zgate.0, g.zgate.1, dbg, gpu)?,
                    resolve_f32_or_decline("gdn.alpha", g.alpha, hidden, dbg, gpu)?,
                    resolve_f32_or_decline("gdn.beta", g.beta, hidden, dbg, gpu)?,
                    resolve_or_decline("gdn.ssm_out", g.ssm_out.0, g.ssm_out.1, dbg, gpu)?,
                ];
                let mut states = Vec::with_capacity(5);
                for (mat, n_in) in mats.iter().zip([hidden, hidden, hidden, hidden, value_dim]) {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    states.push(gpu.pipeline(mat.kernel, 1, lpr)?.to_owned());
                }
                let conv1d = {
                    let addr = g.conv1d.as_ptr() as usize;
                    if addr % 4 != 0 {
                        return None;
                    }
                    let chunk = gpu.chunk_for(addr, g.conv1d.len())?;
                    NormRef {
                        chunk,
                        off: (addr - gpu.chunks[chunk].start) as u64,
                    }
                };
                Some(GdnRefs {
                    mats,
                    states: states.try_into().ok()?,
                    conv1d,
                    heads_k: g.heads_k as u32,
                    heads_v: g.heads_v as u32,
                    d: g.d as u32,
                    d_conv: g.d_conv as u32,
                    conv_off: g.conv_off as u64,
                    state_off: g.state_off as u64,
                    channels,
                    value_dim,
                })
            }
        };
        let wq_out = if l.gate_in_q { 2 * q_dim } else { q_dim };
        let mats_pre = if gdn.is_none() {
            let dims = [
                (hidden, wq_out),
                (hidden, l.wk.2),
                (hidden, l.wv.2),
                (q_dim, l.wo.2),
            ];
            why!(
                l.wq.2 != wq_out || l.wk.2 != kv || l.wv.2 != kv || l.wo.2 != hidden,
                "attn dims"
            );
            let mats = [
                resolve_or_decline("attn.wq", l.wq.0, l.wq.1, dbg, gpu)?,
                resolve_or_decline("attn.wk", l.wk.0, l.wk.1, dbg, gpu)?,
                resolve_or_decline("attn.wv", l.wv.0, l.wv.1, dbg, gpu)?,
                resolve_or_decline("attn.wo", l.wo.0, l.wo.1, dbg, gpu)?,
            ];
            Some((mats, dims))
        } else {
            None
        };
        let normflag = allow_normflag
            && runtime_normflag
            && mats_pre.as_ref().is_some_and(|(m, _)| {
                m.iter()
                    .take(3)
                    .all(|m| matches!(m.kernel, "matvec_q2_k_mv" | "matvec_q4_k_mv"))
            })
            && matches!(l.ffn, TokenFfn::Moe { .. });
        let (mats, mat_states) = match mats_pre {
            Some((mats, dims)) => {
                let mut mat_states = Vec::with_capacity(4);
                for (i, (mat, &(n_in, _))) in mats.iter().zip(&dims).enumerate() {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    let wait = normflag && i < 3;
                    mat_states.push(
                        gpu.pipeline_wait(mat.kernel, 1, lpr, false, false, wait)?
                            .to_owned(),
                    );
                }
                (Some(mats), Some(mat_states.try_into().ok()?))
            }
            None => (None, None),
        };
        let ffn = match &l.ffn {
            TokenFfn::Dense { gate, up, down } => {
                if gate.2 != up.2 || down.2 != hidden {
                    return None;
                }
                let mats = [
                    resolve_or_decline("ffn.gate", gate.0, gate.1, dbg, gpu)?,
                    resolve_or_decline("ffn.up", up.0, up.1, dbg, gpu)?,
                    resolve_or_decline("ffn.down", down.0, down.1, dbg, gpu)?,
                ];
                let sw_capable = |k: &str| {
                    matches!(
                        k,
                        "matvec_q2_k"
                            | "matvec_q2_k_mv"
                            | "matvec_q3_k_mv"
                            | "matvec_q4_k_mv"
                            | "matvec_q5_k_mv"
                            | "matvec_q6_k"
                            | "matvec_q6_k_mv"
                            | "matvec_q8_0"
                            | "matvec_q8_0_mv"
                    )
                };
                let sw_fused = sw_capable(mats[2].kernel) && sw_fuse_env;
                let mut states = Vec::with_capacity(3);
                for (i, (mat, n_in)) in mats.iter().zip([hidden, hidden, gate.2]).enumerate() {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    let swiglu = i == 2 && sw_fused;
                    if swiglu && debug_qkv() {
                        eprintln!("debug-qkv: decode swiglu-fold on {}", mat.kernel);
                    }
                    states.push(
                        gpu.pipeline_full(mat.kernel, 1, lpr, false, swiglu)?
                            .to_owned(),
                    );
                }
                max_ffn = max_ffn.max(gate.2);
                FfnRefs::Dense {
                    mats,
                    states: states.try_into().ok()?,
                    ffn_dim: gate.2,
                    sw_fused,
                }
            }
            TokenFfn::Moe {
                router,
                router_bias,
                gate,
                up,
                down,
                expert_ffn,
                n_used,
                sigmoid,
                shared,
                shared_gate,
            } => {
                why!(
                    router.0 != GgmlType::F32 || router.2 > 256 || *n_used > 16,
                    "router shape"
                );
                why!(
                    shared_gate.is_some() && shared.is_none(),
                    "shared gate without shared expert"
                );
                let router_ref = resolve_f32_or_decline("router", router.1, hidden, dbg, gpu)?;
                let router_state = gpu
                    .pipeline_wait(
                        "matvec_f32",
                        1,
                        lanes_per_row(GgmlType::F32, hidden),
                        false,
                        false,
                        normflag,
                    )?
                    .to_owned();
                let mats = [
                    resolve_or_decline("ffn.gate", gate.0, gate.1, dbg, gpu)?,
                    resolve_or_decline("ffn.up", up.0, up.1, dbg, gpu)?,
                    resolve_or_decline("ffn.down", down.0, down.1, dbg, gpu)?,
                ];
                let n_expert = router.2;
                let strides = [
                    (gate.1.len() / n_expert) as u64,
                    (up.1.len() / n_expert) as u64,
                    (down.1.len() / n_expert) as u64,
                ];
                let mut states = Vec::with_capacity(3);
                let n_outs = [*expert_ffn, *expert_ffn, hidden];
                let mut sw_fused = false;
                let mut gu_kernels = [""; 2];
                let sw_capable = |k: &str| {
                    matches!(
                        k,
                        "matvec_q2_k"
                            | "matvec_q2_k_mv"
                            | "matvec_q3_k_mv"
                            | "matvec_q4_k_mv"
                            | "matvec_q5_k_mv"
                            | "matvec_q6_k"
                            | "matvec_q6_k_mv"
                            | "matvec_q8_0"
                            | "matvec_q8_0_mv"
                    )
                };
                let shared_down_sw = match shared {
                    Some(sh) => resolve_or_decline("ffn.shared.down", sh[2].0, sh[2].1, dbg, gpu)
                        .is_some_and(|m| sw_capable(m.kernel)),
                    None => true,
                };
                for (i, ((mat, n_in), n_out)) in mats
                    .iter()
                    .zip([hidden, hidden, *expert_ffn])
                    .zip(n_outs)
                    .enumerate()
                {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    let nr0 = (32 / lpr).max(1);
                    let kernel = match mat.kernel {
                        "matvec_q2_k_mv" if n_out % nr0 != 0 => "matvec_q2_k",
                        "matvec_q3_k_mv" if n_out % 2 != 0 => "matvec_q3_k",
                        "matvec_q4_k_mv" if n_out % nr0 != 0 => "matvec_q4_k",
                        "matvec_q5_k_mv" if n_out % nr0 != 0 => "matvec_q5_k",
                        "matvec_q6_k_mv" if n_out % 2 != 0 => "matvec_q6_k",
                        "matvec_q8_0_mv" if n_out % q8_nr0() != 0 => "matvec_q8_0",
                        k => k,
                    };
                    if i < 2 {
                        gu_kernels[i] = kernel;
                    }
                    let swiglu = i == 2 && sw_capable(kernel) && shared_down_sw && sw_fuse_env;
                    if swiglu {
                        sw_fused = true;
                    }
                    states.push(gpu.pipeline_full(kernel, 1, lpr, true, swiglu)?.to_owned());
                }
                max_ffn = max_ffn.max(*expert_ffn);
                let shared_refs = match shared {
                    Some(sh) => {
                        why!(
                            sh[0].2 != *expert_ffn || sh[1].2 != *expert_ffn || sh[2].2 != hidden,
                            "shared dims"
                        );
                        let smats = [
                            resolve_or_decline("ffn.shared.gate", sh[0].0, sh[0].1, dbg, gpu)?,
                            resolve_or_decline("ffn.shared.up", sh[1].0, sh[1].1, dbg, gpu)?,
                            resolve_or_decline("ffn.shared.down", sh[2].0, sh[2].1, dbg, gpu)?,
                        ];
                        let mut sstates = Vec::with_capacity(3);
                        for (i, (mat, n_in)) in
                            smats.iter().zip([hidden, hidden, *expert_ffn]).enumerate()
                        {
                            let lpr = lanes_per_row(mat.ty, n_in);
                            let swiglu = i == 2 && sw_fused && sw_capable(mat.kernel);
                            sstates.push(
                                gpu.pipeline_full(mat.kernel, 1, lpr, false, swiglu)?
                                    .to_owned(),
                            );
                        }
                        max_slots = max_slots.max(*n_used + 1);
                        Some((smats, sstates.try_into().ok()?))
                    }
                    None => {
                        max_slots = max_slots.max(*n_used);
                        None
                    }
                };
                max_expert = max_expert.max(n_expert);
                let mega = if allow_normflag && mega_enabled() && n_expert <= 256 {
                    let gf = mega_fmt(mats[0].ty);
                    let uf = mega_fmt(mats[1].ty);
                    let df = mega_fmt(mats[2].ty);
                    let (sgf, suf, sdf) = match &shared_refs {
                        Some((smats, _)) => (
                            mega_fmt(smats[0].ty),
                            mega_fmt(smats[1].ty),
                            mega_fmt(smats[2].ty),
                        ),
                        None => (Some(0), Some(0), Some(0)),
                    };
                    match (gf, uf, df, sgf, suf, sdf) {
                        (Some(g), Some(u), Some(d), Some(sg), Some(su), Some(sd))
                            if g == u && sg == su =>
                        {
                            gpu.mega_pipeline(g, d, sg, sd).map(|p| p.to_owned())
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                let gu_dual_lpr = lanes_per_row(mats[0].ty, hidden);
                let gu_dual_nr0 = (32 / gu_dual_lpr.max(1)).max(1);
                let gu_dual = if gufuse()
                    && gu_kernels[0] == gu_kernels[1]
                    && matches!(
                        gu_kernels[0],
                        "matvec_q2_k_mv"
                            | "matvec_q3_k_mv"
                            | "matvec_q4_k_mv"
                            | "matvec_q5_k_mv"
                            | "matvec_q6_k_mv"
                            | "matvec_q8_0_mv"
                    )
                    && strides[0] == strides[1]
                    && (*expert_ffn * *n_used) % gu_dual_nr0 == 0
                {
                    gpu.pipeline_dual(gu_kernels[0], 1, gu_dual_lpr)
                        .map(|p| p.to_owned())
                } else {
                    None
                };
                let down_tail = match &shared_refs {
                    Some((smats, _))
                        if shared_tail()
                            && mats[2].kernel == "matvec_q8_0_mv"
                            && smats[2].kernel == "matvec_q8_0_mv"
                            && mats[2].ty == smats[2].ty
                            && (hidden * (*n_used + 1)) % 2 == 0 =>
                    {
                        let lpr = lanes_per_row(mats[2].ty, *expert_ffn);
                        gpu.pipeline_shared_tail("matvec_q8_0_mv", lpr, sw_fused)
                            .map(|p| p.to_owned())
                    }
                    _ => None,
                };
                FfnRefs::Moe {
                    router: router_ref,
                    router_state,
                    router_bias: match router_bias {
                        Some(b) => Some(resolve_norm(gpu, b, n_expert)?),
                        None => None,
                    },
                    n_expert,
                    mats,
                    states: states.try_into().ok()?,
                    strides,
                    expert_ffn: *expert_ffn,
                    n_used: *n_used,
                    sw_fused,
                    mega,
                    sigmoid: *sigmoid,
                    shared: shared_refs,
                    shared_gate: match shared_gate {
                        Some(sg) => {
                            why!(sg.len() != hidden * 4, "shared gate dims");
                            let mat = resolve_f32_or_decline(
                                "ffn.shared.gate_out",
                                sg,
                                hidden,
                                dbg,
                                gpu,
                            )?;
                            let state = gpu
                                .pipeline("matvec_f32", 1, lanes_per_row(GgmlType::F32, hidden))?
                                .to_owned();
                            Some((mat, state))
                        }
                        None => None,
                    },
                    gu_dual,
                    down_tail,
                }
            }
        };
        let Some(attn_norm_ref) = resolve_norm(gpu, l.attn_norm, hidden) else {
            why!(true, "attn norm resolve");
            return None;
        };
        let Some(ffn_norm_ref) = resolve_norm(gpu, l.ffn_norm, hidden) else {
            why!(true, "ffn norm resolve");
            return None;
        };
        let bias_ref = |raw: Option<&[u8]>, n: usize, gpu: &Gpu| -> Option<Option<NormRef>> {
            match raw {
                Some(b) => Some(Some(resolve_norm(gpu, b, n)?)),
                None => Some(None),
            }
        };
        layers.push(LayerRefs {
            attn_norm: attn_norm_ref,
            ffn_norm: ffn_norm_ref,
            mats,
            mat_states,
            q_bias: bias_ref(l.q_bias, q_dim, gpu)?,
            k_bias: bias_ref(l.k_bias, kv, gpu)?,
            v_bias: bias_ref(l.v_bias, kv, gpu)?,
            gdn,
            ffn,
            normflag,
        });
    }

    let cached = CachedDecodeLayers {
        key,
        layers: std::sync::Arc::from(layers),
        max_ffn,
        max_slots,
        max_expert,
    };
    gpu.decode_cache = Some(cached.clone());
    Some(cached)
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub(super) enum FfnRefs {
    Dense {
        mats: [MatRef; 3],
        states: [ComputePipelineState; 3],
        ffn_dim: usize,
        /// Down reads silu(gate)*up via SWIGLU_X (same as MoE).
        sw_fused: bool,
    },
    Moe {
        router: MatRef,
        router_state: ComputePipelineState,
        router_bias: Option<NormRef>,
        n_expert: usize,
        mats: [MatRef; 3],
        states: [ComputePipelineState; 3],
        strides: [u64; 3],
        expert_ffn: usize,
        n_used: usize,
        sw_fused: bool,
        mega: Option<ComputePipelineState>,
        sigmoid: bool,
        shared: Option<([MatRef; 3], [ComputePipelineState; 3])>,
        /// qwen35moe's shared-expert gate projection (F32 [hidden]);
        /// its raw logit becomes the shared slot's weight.
        shared_gate: Option<(MatRef, ComputePipelineState)>,
        /// gate + up as one dual-output indexed dispatch
        /// (ALLPAKA_DECODE_GUFUSE).
        gu_dual: Option<ComputePipelineState>,
        /// Indexed down with SHARED_TAIL: routed + shared Q8_0 down in one
        /// launch (`ALLPAKA_SHARED_TAIL=1`).
        down_tail: Option<ComputePipelineState>,
    },
}

/// Diagnostic-preserving entry point for production fast paths.
pub fn decode_token_checked(req: &TokenReq) -> Result<TokenOut, DecodeDecline> {
    use std::sync::atomic::Ordering::Relaxed;
    DECODE_ATTEMPTS.fetch_add(1, Relaxed);
    match decode_token(req) {
        Some(out) => {
            DECODE_SUCCESSES.fetch_add(1, Relaxed);
            Ok(out)
        }
        None => {
            DECODE_DECLINES.fetch_add(1, Relaxed);
            Err(DecodeDecline {
                stage: "backend-decode",
                reason: "request shape, tensor format, or Metal pipeline is unsupported",
            })
        }
    }
}

pub fn decode_token(req: &TokenReq) -> Option<TokenOut> {
    let dbg = crate::runtime::get().tokenbuf_debug;
    macro_rules! why {
        ($cond:expr, $msg:expr) => {
            if $cond {
                if dbg {
                    eprintln!("tokenbuf declined: {}", $msg);
                }
                return None;
            }
        };
    }
    let m = req.m;
    why!(m == 0 || m > 8, "m");
    let hidden = if req.x.is_empty() {
        why!(m != 1, "gpu embed m");
        let Some((_, _, h)) = req.embd else {
            why!(true, "gpu embed missing embd");
            return None;
        };
        why!(req.token_id.is_none(), "gpu embed missing token_id");
        h
    } else {
        why!(req.x.len() % m != 0, "m");
        req.x.len() / m
    };
    let hd = req.head_dim;
    // 128 everywhere; 256 on qwen35moe's full-attention layers (own kernels).
    why!(
        hd != 128 && hd != 256 || hidden == 0 || hidden % 4 != 0,
        "shape"
    );
    // Full rotary, GLM-style half rotary, qwen35moe's quarter rotary at 256.
    why!(req.rope.len() != m * req.rot_dim / 2, "rope table");
    why!(
        req.rot_dim != hd && req.rot_dim * 2 != hd && !(hd == 256 && req.rot_dim == 64),
        "rot dim"
    );
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let span = (req.pos + m) * req.kv_dim;
    let vocab = req.output.2;
    // GDN layers need the shared SSM region; attention layers the KV cache.
    let has_gdn = req.layers.iter().any(|l| l.gdn.is_some());
    why!(has_gdn && req.ssm.is_none(), "no ssm region");

    let Some(Some(cell)) = GPU.get().map(|c| c.as_ref()) else {
        why!(true, "no gpu");
        return None;
    };
    let Ok(mut gpu) = cell.lock() else {
        why!(true, "lock");
        return None;
    };
    let CachedDecodeLayers {
        layers,
        max_ffn,
        max_slots,
        max_expert,
        ..
    } = resolve_decode_layers(&mut gpu, req, hidden, q_dim, kv, span, true, dbg)?;
    let out_norm = resolve_norm(&gpu, req.output_norm, hidden)?;
    let out_mat = resolve(&gpu, req.output.0, req.output.1)?;
    let out_state = gpu
        .pipeline(out_mat.kernel, 1, lanes_per_row(out_mat.ty, hidden))?
        .to_owned();
    let attend_state = gpu.pipelines[&(attend_kernel(), 1, 1)].to_owned();
    let attend_split_state = gpu.pipelines[&("attend_split", 1, 1)].to_owned();
    let attend_merge_state = gpu.pipelines[&("attend_merge", 1, 1)].to_owned();
    let qk_prep_state = gpu.pipelines[&("qk_prep", 1, 1)].to_owned();
    let resnorm_state = gpu.pipelines[&("residual_norm", 1, 1)].to_owned();
    let resnorm_sig_state = gpu.pipelines[&("residual_norm_sig", 1, 1)].to_owned();
    // head_dim-256 attention (qwen35moe) and the GDN branch.
    let qk_prep256_state = gpu.pipelines[&("qk_prep256", 1, 1)].to_owned();
    let attend256_state = gpu.pipelines[&("attend_s32_256", 1, 1)].to_owned();
    let attend_split256_state = gpu.pipelines[&("attend_split_256", 1, 1)].to_owned();
    let gate_mul_state = gpu.pipelines[&("attn_gate_mul", 1, 1)].to_owned();
    let gdn_conv_state = gpu.pipelines[&("gdn_conv", 1, 1)].to_owned();
    let gdn_step_state = gpu.pipelines[&("gdn_step", 1, 1)].to_owned();
    let gdn_norm_state = gpu.pipelines[&("gdn_out_norm", 1, 1)].to_owned();
    let sigmoid_gating = req
        .layers
        .iter()
        .any(|l| matches!(l.ffn, TokenFfn::Moe { sigmoid: true, .. }));
    let topk_state = gpu.pipelines[&(
        if sigmoid_gating {
            "sigmoid_topk"
        } else {
            "softmax_topk"
        },
        1,
        1,
    )]
        .to_owned();
    let rtopk_state = gpu.pipelines[&("router_topk", 1, 1)].to_owned();
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();
    let combine_state = gpu.pipelines[&("moe_combine", 1, 1)].to_owned();
    let combine_resnorm_state = gpu.pipelines[&("combine_resnorm", 1, 1)].to_owned();
    let resnorm_router_state = gpu.pipelines[&("resnorm_router", 1, 1)].to_owned();
    let qkv_mv_state = gpu.pipelines[&("matvec_qkv_q4k_q6k", 1, 1)].to_owned();
    let argmax_state = gpu.pipelines[&("argmax_f32", 1, 1)].to_owned();
    let argmax_final_state = gpu.pipelines[&("argmax_final", 1, 1)].to_owned();
    let embed_state = gpu.pipelines[&("embed_row_f32_dtoken", 1, 1)].to_owned();
    let gpu_embed = if req.x.is_empty() {
        let (embd_ty, embd_bytes, embd_h) = req.embd?;
        why!(embd_h != hidden, "gpu embed hidden");
        let tok_id = req.token_id?;
        let embd_addr = embd_bytes.as_ptr() as usize;
        let Some(embd_chunk) = gpu.chunk_for(embd_addr, embd_bytes.len()) else {
            why!(true, "gpu embed chunk");
            return None;
        };
        let embd_base = (embd_addr - gpu.chunks[embd_chunk].start) as u64;
        let Some(rb) = row_bytes(embd_ty, hidden) else {
            why!(true, "gpu embed row_bytes");
            return None;
        };
        let Some(fmt) = dq_fmt(embd_ty) else {
            why!(true, "gpu embed dq_fmt");
            return None;
        };
        Some((embd_chunk, embd_base, rb as u32, fmt, tok_id))
    } else {
        None
    };

    // The speculative multi-token verify: same resolved layers, one command
    // buffer, batched projections (TILE = m) and per-token attention/MoE.
    if m > 1 {
        return encode_verify_tokens(req, &mut gpu, &layers, sigmoid_gating);
    }

    // Arena layout, element offsets over f32 (ids are u32-in-f32 slots).
    let align = |n: usize| (n + 63) & !63;
    // Raw wq output can be 2x q_dim when the gate rides inside it; the GDN
    // scratch pads size off the largest deltanet branch.
    let wq_max = req
        .layers
        .iter()
        .filter(|l| l.gdn.is_none())
        .map(|l| l.wq.2)
        .max()
        .unwrap_or(q_dim)
        .max(q_dim);
    let (mut max_channels, mut max_value_dim, mut max_heads_v) = (0usize, 0usize, 1usize);
    for l in req.layers {
        if let Some(g) = &l.gdn {
            max_channels = max_channels.max(g.heads_k * g.d * 2 + g.heads_v * g.d);
            max_value_dim = max_value_dim.max(g.heads_v * g.d);
            max_heads_v = max_heads_v.max(g.heads_v);
        }
    }
    let x_at = 0usize;
    let delta_at = x_at + align(hidden);
    let h_at = delta_at + align(hidden);
    let q_at = h_at + align(hidden);
    // Deinterleaved normed q and the raw gate rows (head_dim-256 path).
    let qn_at = q_at + align(wq_max);
    let qgate_at = qn_at + align(q_dim);
    let k_at = qgate_at + align(q_dim);
    let v_at = k_at + align(kv);
    let attn_at = v_at + align(kv);
    // GDN scratch: the conv-mixed qkv row, z gate, alpha/beta, deltanet out.
    let gqkv_at = attn_at + align(q_dim);
    let gz_at = gqkv_at + align(max_channels);
    let gab_at = gz_at + align(max_value_dim);
    let gy_at = gab_at + align(2 * max_heads_v);
    let logits_at = gy_at + align(max_value_dim);
    let ids_at = logits_at + align(max_expert.max(1));
    let wts_at = ids_at + align(max_slots);
    let gate_at = wts_at + align(max_slots);
    let up_at = gate_at + align(max_slots * max_ffn);
    let downo_at = up_at + align(max_slots * max_ffn);
    let out_logits_at = downo_at + align(max_slots * hidden);
    let flag_at = out_logits_at + align(vocab);
    let ctr_at = flag_at + align(1);
    // Flash-decoding split fan-out: past a few hundred cached positions one
    // threadgroup per q head walks the KV cache too serially (measured 86 us
    // per layer at 544 tokens against ~2 us of cache bytes). Split the
    // positions across nsplit threadgroups and merge the partials.
    // ALLPAKA_ATTN_SPLIT forces the fan-out, 1 disables the split path.
    let n_pos = req.pos + 1;
    let forced_split = crate::runtime::get().attention_split;
    // On M4 Max, twelve slices are consistently faster than sixteen once the
    // cache is long enough to saturate memory-level parallelism. Keep the
    // larger range available to explicit overrides for other GPUs.
    let nsplit = forced_split
        .unwrap_or_else(|| (n_pos / 128).clamp(1, 12))
        .clamp(1, 16)
        .min(n_pos);
    let sp_acc_at = ctr_at + align(1);
    let sp_md_at = sp_acc_at + align(req.n_heads * nsplit * hd);
    // The greedy argmax: 64 (value, index) partial pairs plus the winner.
    let amax_at = sp_md_at + align(req.n_heads * nsplit * 2);
    let tok_at = amax_at + align(64 * 2 + 1);
    let total = tok_at + align(1);
    gpu.ensure_arenas(4096, total * 4);

    // Stage x (or token_id for GPU embedding lookup) and zero the first delta.
    unsafe {
        let yp = gpu.y_arena.contents() as *mut f32;
        if let Some((_, _, _, _, tok_id)) = gpu_embed {
            *(yp.add(tok_at) as *mut u32) = tok_id;
        } else {
            std::ptr::copy_nonoverlapping(req.x.as_ptr(), yp.add(x_at), hidden);
        }
        std::ptr::write_bytes(yp.add(delta_at), 0, hidden);
        std::ptr::write_bytes(yp.add(flag_at), 0, 1);
        std::ptr::write_bytes(yp.add(ctr_at), 0, 1);
        // The shared expert's combine weight is a constant 1 in the slot past
        // the router-written ones - unless the layer carries qwen35moe's
        // gate projection, whose matvec overwrites it during the token.
        for l in &*layers {
            if let FfnRefs::Moe {
                n_used,
                shared: Some(_),
                shared_gate: None,
                ..
            } = &l.ffn
            {
                *yp.add(wts_at + *n_used) = 1.0;
            }
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let command_buffer = gpu.queue.new_command_buffer();
        let cmd = crate::command::CommandScope::new(command_buffer);
        // ALLPAKA_DECODE_SERIAL=1: serial encoder, implicit ordering, all
        // explicit barriers skipped - probes whether the driver's own
        // hazard tracking beats our barrier drains per stage boundary.
        // Megakernel multi-TG sync busy-waits on a device atomic; concurrent
        // unrelated dispatches can starve sibling TGs and deadlock. Force
        // serial whenever any layer uses MEGA (still honor explicit SERIAL=1).
        let any_mega = layers
            .iter()
            .any(|l| matches!(&l.ffn, FfnRefs::Moe { mega: Some(_), .. }));
        let serial = crate::runtime::get().decode_serial || any_mega;
        if any_mega && mega_debug() {
            eprintln!(
                "mega: encode path n_tg={} serial={} layers={}",
                mega_tg(),
                serial,
                layers.len()
            );
        }
        // ALLPAKA_DECODE_SPLIT=1: sample the GPU timestamp counter at existing
        // stage boundaries. Profile mode uses multiple compute encoders inside
        // the same command buffer; production mode remains one concurrent
        // encoder. Both modes keep one submit and one wait per token.
        let split = crate::runtime::get().decode_split;
        const MAX_SPLIT_SAMPLES: u64 = 2048;
        let split_counter = if split
            && gpu
                .device
                .supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
        {
            let desc = metal::CounterSampleBufferDescriptor::new();
            desc.set_storage_mode(metal::MTLStorageMode::Shared);
            desc.set_sample_count(MAX_SPLIT_SAMPLES);
            gpu.device
                .counter_sets()
                .iter()
                .find(|set| set.name() == "timestamp")
                .and_then(|set| {
                    desc.set_counter_set(set);
                    gpu.device
                        .new_counter_sample_buffer_with_descriptor(&desc)
                        .ok()
                })
        } else {
            None
        };
        let split_resolved = split_counter.as_ref().map(|_| {
            gpu.device.new_buffer(
                MAX_SPLIT_SAMPLES * std::mem::size_of::<u64>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        });
        let mut split_labels: Vec<&'static str> = Vec::new();
        let mut split_index = 0u64;
        let mut split_cpu_start = 0u64;
        let mut split_gpu_start = 0u64;
        let profile_encoder = |counter: &metal::CounterSampleBufferRef, start: u64, end: u64| {
            let desc = metal::ComputePassDescriptor::new();
            let attachment = desc
                .sample_buffer_attachments()
                .object_at(0)
                .expect("Metal compute counter attachment 0");
            attachment.set_sample_buffer(counter);
            attachment.set_start_of_encoder_sample_index(start);
            attachment.set_end_of_encoder_sample_index(end);
            cmd.compute_command_encoder_with_descriptor(desc)
        };
        let mut enc = if let Some(counter) = split_counter.as_ref() {
            gpu.device
                .sample_timestamps(&mut split_cpu_start, &mut split_gpu_start);
            split_index = 2;
            profile_encoder(counter, 0, 1)
        } else {
            cmd.compute_command_encoder_with_dispatch_type(if serial {
                metal::MTLDispatchType::Serial
            } else {
                metal::MTLDispatchType::Concurrent
            })
        };
        macro_rules! split_here {
            ($label:expr) => {
                if let Some(counter) = split_counter.as_ref() {
                    if split_index + 1 < MAX_SPLIT_SAMPLES {
                        enc.end_encoding();
                        split_labels.push($label);
                        enc = profile_encoder(counter, split_index, split_index + 1);
                        split_index += 2;
                    }
                }
            };
        }
        let y = &gpu.y_arena;
        // Probe: ALLPAKA_SKIP=bar drops every barrier; barn/bara/barf drop
        // the norm / attention-chain / ffn-chain subsets (values race,
        // timing shows what each sync class costs).
        let no_bar = probe_skip("bar");
        let no_barn = probe_skip("barn");
        let no_bara = probe_skip("bara");
        let no_barf = probe_skip("barf");
        let bar_c = |enc: &metal::ComputeCommandEncoderRef, class: u8| {
            let skip = serial
                || no_bar
                || (class == b'n' && no_barn)
                || (class == b'a' && no_bara)
                || (class == b'f' && no_barf);
            if !skip {
                enc.memory_barrier_with_resources(&[y]);
            }
        };
        let bar = |enc: &metal::ComputeCommandEncoderRef| bar_c(enc, b'x');
        let e = |off: usize| (off * 4) as u64;

        // One plain matvec: in and out are arena element offsets.
        let matvec = |enc: &metal::ComputeCommandEncoderRef,
                      state: &ComputePipelineState,
                      mat: &MatRef,
                      n_in: usize,
                      n_out: usize,
                      x_off: usize,
                      y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            dispatch_mv(enc, mat.ty, n_in, n_out);
        };
        // The indexed form: n_out is per expert, the grid covers all slots.
        let matvec_idx = |enc: &metal::ComputeCommandEncoderRef,
                          state: &ComputePipelineState,
                          mat: &MatRef,
                          n_in: usize,
                          n_out: usize,
                          x_off: usize,
                          y_off: usize,
                          stride: u64,
                          slots: usize,
                          x_stride: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(y), e(ids_at));
            let idx = GpuIdxArgs {
                stride,
                slots: slots as u32,
                x_stride: x_stride as u32,
                ids_stride: 0,
                x_row_stride: 0,
                y_row_stride: 0,
                n_rows: 0,
            };
            enc.set_bytes(
                7,
                std::mem::size_of::<GpuIdxArgs>() as u64,
                &idx as *const GpuIdxArgs as *const _,
            );
            dispatch_mv_indexed(enc, mat.ty, n_in, n_out, slots);
        };
        let resnorm = |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef| {
            enc.set_compute_pipeline_state(&resnorm_state);
            enc.set_buffer(0, Some(y), e(x_at));
            enc.set_buffer(1, Some(y), e(delta_at));
            enc.set_buffer(2, Some(y), e(h_at));
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        };
        // The MoE combine: delta = sum over slots of wts[s] * down[s].
        // sig_last: the last slot's weight is a raw gate logit (qwen35moe).
        let combine = |enc: &metal::ComputeCommandEncoderRef, slots: u32, sig_last: u32| {
            enc.set_compute_pipeline_state(&combine_state);
            enc.set_buffer(0, Some(y), e(downo_at));
            enc.set_buffer(1, Some(y), e(wts_at));
            enc.set_buffer(2, Some(y), e(delta_at));
            let n = hidden as u32;
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &slots as *const u32 as *const _);
            enc.set_bytes(5, 4, &sig_last as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((hidden as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        };
        // residual_norm with the combine folded in: one dispatch and one
        // barrier less per MoE layer boundary.
        let combine_resnorm =
            |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef, slots: u32, sig_last: u32| {
                enc.set_compute_pipeline_state(&combine_resnorm_state);
                enc.set_buffer(0, Some(y), e(x_at));
                enc.set_buffer(1, Some(y), e(delta_at));
                enc.set_buffer(2, Some(y), e(h_at));
                enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
                let n = hidden as u32;
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                enc.set_buffer(6, Some(y), e(downo_at));
                enc.set_buffer(7, Some(y), e(wts_at));
                enc.set_bytes(8, 4, &slots as *const u32 as *const _);
                enc.set_bytes(9, 4, &sig_last as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            };
        // The signalling form: publishes `epoch` to the flag word instead of
        // relying on the barrier the caller then skips.
        let resnorm_sig = |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef, epoch: u32| {
            enc.set_compute_pipeline_state(&resnorm_sig_state);
            enc.set_buffer(0, Some(y), e(x_at));
            enc.set_buffer(1, Some(y), e(delta_at));
            enc.set_buffer(2, Some(y), e(h_at));
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.set_buffer(6, Some(y), e(flag_at));
            enc.set_bytes(7, 4, &epoch as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        };

        let mut dispatched = 0u64;
        if let Some((embd_chunk, embd_base, rb_u, fmt, _)) = gpu_embed {
            enc.set_compute_pipeline_state(&embed_state);
            enc.set_buffer(0, Some(&gpu.chunks[embd_chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_at));
            let h_u = hidden as u32;
            enc.set_bytes(2, 4, &h_u as *const u32 as *const _);
            enc.set_bytes(3, 8, &embd_base as *const u64 as *const _);
            enc.set_bytes(4, 4, &rb_u as *const u32 as *const _);
            enc.set_bytes(5, 4, &fmt as *const u32 as *const _);
            enc.set_buffer(6, Some(y), e(tok_at));
            let tok_idx = 0u32;
            enc.set_bytes(7, 4, &tok_idx as *const u32 as *const _);
            let n_tg = if fmt <= 5 {
                (hidden as u64).div_ceil(256)
            } else {
                (hidden as u64).div_ceil(32)
            };
            enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(32, 1, 1));
            bar(&enc);
            dispatched += 1;
        }
        // MoE combine work deferred into the next norm: Some((slot count,
        // sig_last)) between the down stage of one MoE layer and the
        // residual norm that absorbs it (the next layer's attention norm, or
        // the final norm).
        let mut pending_combine: Option<(u32, u32)> = None;
        for (li, (l, refs)) in req.layers.iter().zip(&*layers).enumerate() {
            let ep_attn = (li as u32) * 2 + 1;
            let ep_ffn = (li as u32) * 2 + 2;
            // h = rmsnorm(x + delta) * attn_norm; x absorbs delta. A pending
            // combine folds into the norm kernel; the normflag path keeps
            // the separate combine (its norm has no fused sig variant).
            let fused = match pending_combine.take() {
                Some((slots, sig_last)) if !refs.normflag => {
                    combine_resnorm(&enc, &refs.attn_norm, slots, sig_last);
                    bar_c(&enc, b'n');
                    true
                }
                Some((slots, sig_last)) => {
                    combine(&enc, slots, sig_last);
                    bar(&enc);
                    false
                }
                None => false,
            };
            if !fused {
                if refs.normflag {
                    resnorm_sig(&enc, &refs.attn_norm, ep_attn);
                    enc.set_buffer(9, Some(y), e(flag_at));
                    enc.set_bytes(10, 4, &ep_attn as *const u32 as *const _);
                    if probe_skip("keepbar") {
                        bar_c(&enc, b'n');
                    }
                } else {
                    resnorm(&enc, &refs.attn_norm);
                    bar_c(&enc, b'n');
                }
            }
            split_here!("resnorm");

            // The gated-delta-net branch replaces the whole attention half.
            if let Some(g) = &refs.gdn {
                let tg = &l.gdn.as_ref().expect("gdn refs without gdn layer");
                let ssm_buf = &req.ssm.expect("gdn layer without ssm region").buf;
                if !probe_skip("qkv") {
                    matvec(
                        &enc,
                        &g.states[0],
                        &g.mats[0],
                        hidden,
                        g.channels,
                        h_at,
                        gqkv_at,
                    );
                    matvec(
                        &enc,
                        &g.states[1],
                        &g.mats[1],
                        hidden,
                        g.value_dim,
                        h_at,
                        gz_at,
                    );
                    matvec(
                        &enc,
                        &g.states[2],
                        &g.mats[2],
                        hidden,
                        g.heads_v as usize,
                        h_at,
                        gab_at,
                    );
                    matvec(
                        &enc,
                        &g.states[3],
                        &g.mats[3],
                        hidden,
                        g.heads_v as usize,
                        h_at,
                        gab_at + g.heads_v as usize,
                    );
                }
                bar_c(&enc, b'a');
                split_here!("qkv");
                if !probe_skip("attend") {
                    // Depthwise conv (+silu), updating the window in place.
                    enc.set_compute_pipeline_state(&gdn_conv_state);
                    enc.set_buffer(0, Some(y), e(gqkv_at));
                    enc.set_buffer(1, Some(ssm_buf), 0);
                    enc.set_buffer(2, Some(&gpu.chunks[g.conv1d.chunk].buf), g.conv1d.off);
                    let cargs = GdnConvArgs {
                        channels: g.channels as u32,
                        d_conv: g.d_conv,
                        pad0: 0,
                        pad1: 0,
                        conv_off: g.conv_off,
                    };
                    enc.set_bytes(
                        3,
                        std::mem::size_of::<GdnConvArgs>() as u64,
                        &cargs as *const GdnConvArgs as *const _,
                    );
                    enc.dispatch_thread_groups(
                        MTLSize::new((g.channels as u64).div_ceil(256), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // The recurrence, l2 norms folded in, state in place.
                    enc.set_compute_pipeline_state(&gdn_step_state);
                    enc.set_buffer(0, Some(ssm_buf), 0);
                    enc.set_buffer(1, Some(y), e(gqkv_at));
                    enc.set_buffer(2, Some(y), e(gab_at));
                    let sargs = GdnStepArgs {
                        heads_k: g.heads_k,
                        heads_v: g.heads_v,
                        d: g.d,
                        key_dim: g.heads_k * g.d,
                        eps: req.eps,
                        pad0: 0,
                        state_off: g.state_off,
                    };
                    enc.set_bytes(
                        3,
                        std::mem::size_of::<GdnStepArgs>() as u64,
                        &sargs as *const GdnStepArgs as *const _,
                    );
                    enc.set_bytes(4, (tg.a.len() * 4) as u64, tg.a.as_ptr() as *const _);
                    enc.set_bytes(5, (tg.dt.len() * 4) as u64, tg.dt.as_ptr() as *const _);
                    enc.set_buffer(6, Some(y), e(gy_at));
                    enc.dispatch_thread_groups(
                        MTLSize::new((g.d / 4) as u64, g.heads_v as u64, 1),
                        MTLSize::new(32, 4, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // Gated rmsnorm over the head outputs, times silu(z).
                    enc.set_compute_pipeline_state(&gdn_norm_state);
                    enc.set_buffer(0, Some(y), e(gy_at));
                    enc.set_buffer(1, Some(y), e(gz_at));
                    enc.set_bytes(
                        2,
                        (tg.ssm_norm.len() * 4) as u64,
                        tg.ssm_norm.as_ptr() as *const _,
                    );
                    let hv = g.heads_v;
                    let dd = g.d;
                    enc.set_bytes(3, 4, &hv as *const u32 as *const _);
                    enc.set_bytes(4, 4, &dd as *const u32 as *const _);
                    enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(g.heads_v as u64, 1, 1),
                        MTLSize::new(g.d as u64, 1, 1),
                    );
                }
                split_here!("attend");
                bar_c(&enc, b'a');
                // ssm_out straight into delta.
                if !probe_skip("qkv") {
                    matvec(
                        &enc,
                        &g.states[4],
                        &g.mats[4],
                        g.value_dim,
                        hidden,
                        gy_at,
                        delta_at,
                    );
                }
                bar_c(&enc, b'a');
                split_here!("wo");
            } else {
                // qkv from h.
                let amats = refs.mats.as_ref().expect("attention layer without mats");
                let astates = refs
                    .mat_states
                    .as_ref()
                    .expect("attention layer without mat states");
                if !probe_skip("qkv") {
                    // One dispatch for wq+wk+wv when the layout is the
                    // Q4_K_M one (q4k/q4k/q6k, even row splits); opt-in -
                    // see qkv_mv_fuse.
                    let fused_qkv = qkv_mv_fuse()
                        && !refs.normflag
                        && amats[0].kernel == "matvec_q4_k_mv"
                        && amats[1].kernel == "matvec_q4_k_mv"
                        && amats[2].kernel == "matvec_q6_k_mv"
                        && l.wq.2 % 2 == 0
                        && kv % 2 == 0;
                    if fused_qkv {
                        if li == 0 && debug_qkv() {
                            eprintln!("debug-qkv: decode fused qkv matvec_qkv_q4k_q6k");
                        }
                        let args = GpuMvQkvArgs {
                            n_in: hidden as u32,
                            n_q: l.wq.2 as u32,
                            n_k: kv as u32,
                            n_v: kv as u32,
                            wq_off: amats[0].w_off,
                            wk_off: amats[1].w_off,
                            wv_off: amats[2].w_off,
                            q_off: e(q_at),
                            k_off: e(k_at),
                            v_off: e(v_at),
                        };
                        enc.set_compute_pipeline_state(&qkv_mv_state);
                        enc.set_buffer(0, Some(&gpu.chunks[amats[0].chunk].buf), 0);
                        enc.set_buffer(1, Some(&gpu.chunks[amats[1].chunk].buf), 0);
                        enc.set_buffer(2, Some(&gpu.chunks[amats[2].chunk].buf), 0);
                        enc.set_buffer(3, Some(y), e(h_at));
                        enc.set_buffer(4, Some(y), 0);
                        enc.set_bytes(
                            5,
                            std::mem::size_of::<GpuMvQkvArgs>() as u64,
                            &args as *const GpuMvQkvArgs as *const _,
                        );
                        let rows = (l.wq.2 + 2 * kv) as u64;
                        enc.dispatch_thread_groups(
                            MTLSize::new((rows * 16).div_ceil(mv_tg()), 1, 1),
                            MTLSize::new(mv_tg(), 1, 1),
                        );
                    } else {
                        matvec(&enc, &astates[0], &amats[0], hidden, l.wq.2, h_at, q_at);
                        matvec(&enc, &astates[1], &amats[1], hidden, kv, h_at, k_at);
                        matvec(&enc, &astates[2], &amats[2], hidden, kv, h_at, v_at);
                    }
                }
                bar_c(&enc, b'a');
                split_here!("qkv");
                // Norm + rope + cache store.
                if !probe_skip("attend") {
                    if hd == 256 {
                        // head_dim-256 path: qk_prep256 deinterleaves the fused gate and
                        // writes normed q to qn_at; attend_s32_256/attend_split_256 run
                        // the scores; the sigmoid gate lands before wo.
                        enc.set_compute_pipeline_state(&qk_prep256_state);
                        enc.set_buffer(0, Some(y), e(q_at));
                        enc.set_buffer(1, Some(y), e(k_at));
                        enc.set_buffer(2, Some(y), e(v_at));
                        enc.set_buffer(3, Some(&req.cache.buf), 0);
                        let args = QkPrep256Args {
                            n_heads: req.n_heads as u32,
                            n_kv_heads: req.n_kv_heads as u32,
                            kv_dim: req.kv_dim as u32,
                            eps: req.eps,
                            pos: req.pos as u32,
                            has_qk_norm: l.q_norm.is_some() as u32,
                            rot_dim: req.rot_dim as u32,
                            gate_in_q: l.gate_in_q as u32,
                            k_base: l.k_off as u64,
                            v_base: l.v_off as u64,
                        };
                        enc.set_bytes(
                            4,
                            std::mem::size_of::<QkPrep256Args>() as u64,
                            &args as *const QkPrep256Args as *const _,
                        );
                        let ones = [1.0f32; 256];
                        let qw = l.q_norm.unwrap_or(&ones);
                        let kw = l.k_norm.unwrap_or(&ones);
                        enc.set_bytes(5, (256 * 4) as u64, qw.as_ptr() as *const _);
                        enc.set_bytes(6, (256 * 4) as u64, kw.as_ptr() as *const _);
                        enc.set_bytes(
                            7,
                            (req.rope.len() * 8) as u64,
                            req.rope.as_ptr() as *const _,
                        );
                        enc.set_buffer(8, Some(y), e(qn_at));
                        enc.set_buffer(9, Some(y), e(qgate_at));
                        enc.dispatch_thread_groups(
                            MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                            MTLSize::new(32, 1, 1),
                        );
                        enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                        let group = (req.n_heads / req.n_kv_heads.max(1)) as u32;
                        enc.set_buffer(0, Some(&req.cache.buf), 0);
                        enc.set_buffer(1, Some(&req.cache.buf), 0);
                        enc.set_buffer(2, Some(y), e(qn_at));
                        for (index, value) in [
                            (4u64, req.kv_dim as u32),
                            (5, 256u32),
                            (6, (req.pos + 1) as u32),
                            (7, group),
                        ] {
                            enc.set_bytes(index, 4, &value as *const u32 as *const _);
                        }
                        enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                        for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                            enc.set_bytes(index, 8, &value as *const u64 as *const _);
                        }
                        if nsplit > 1 {
                            enc.set_compute_pipeline_state(&attend_split256_state);
                            enc.set_buffer(3, Some(y), e(sp_acc_at));
                            enc.set_buffer(11, Some(y), e(sp_md_at));
                            enc.set_bytes(12, 4, &(nsplit as u32) as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((req.n_heads * nsplit) as u64, 1, 1),
                                MTLSize::new(512, 1, 1),
                            );
                            enc.memory_barrier_with_resources(&[y]);
                            enc.set_compute_pipeline_state(&attend_merge_state);
                            enc.set_buffer(0, Some(y), e(sp_acc_at));
                            enc.set_buffer(1, Some(y), e(sp_md_at));
                            enc.set_buffer(2, Some(y), e(attn_at));
                            enc.set_bytes(3, 4, &(nsplit as u32) as *const u32 as *const _);
                            enc.set_bytes(4, 4, &(hd as u32) as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        } else {
                            enc.set_compute_pipeline_state(&attend256_state);
                            enc.set_buffer(3, Some(y), e(attn_at));
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(512, 1, 1),
                            );
                        }
                        if l.gate_in_q {
                            enc.memory_barrier_with_resources(&[y]);
                            enc.set_compute_pipeline_state(&gate_mul_state);
                            enc.set_buffer(0, Some(y), e(attn_at));
                            enc.set_buffer(1, Some(y), e(qgate_at));
                            let n = (q_dim) as u32;
                            enc.set_bytes(2, 4, &n as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((q_dim as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                    } else {
                        enc.set_compute_pipeline_state(&qk_prep_state);
                        enc.set_buffer(0, Some(y), e(q_at));
                        enc.set_buffer(1, Some(y), e(k_at));
                        enc.set_buffer(2, Some(y), e(v_at));
                        enc.set_buffer(3, Some(&req.cache.buf), 0);
                        let args = QkPrepArgs {
                            n_heads: req.n_heads as u32,
                            n_kv_heads: req.n_kv_heads as u32,
                            head_dim: hd as u32,
                            kv_dim: req.kv_dim as u32,
                            eps: req.eps,
                            pos: req.pos as u32,
                            has_qk_norm: l.q_norm.is_some() as u32,
                            rot_dim: req.rot_dim as u32,
                            k_base: l.k_off as u64,
                            v_base: l.v_off as u64,
                            has_bias: refs.q_bias.is_some() as u32,
                            pad2: 0,
                        };
                        enc.set_bytes(
                            4,
                            std::mem::size_of::<QkPrepArgs>() as u64,
                            &args as *const QkPrepArgs as *const _,
                        );
                        let ones = [1.0f32; ATTN_HEAD_DIM];
                        let qw = l.q_norm.unwrap_or(&ones);
                        let kw = l.k_norm.unwrap_or(&ones);
                        enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
                        enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
                        enc.set_bytes(
                            7,
                            (req.rope.len() * 8) as u64,
                            req.rope.as_ptr() as *const _,
                        );
                        // Bias buffers: real F32 regions when the model has them, any
                        // valid buffer otherwise (the kernel skips them on has_bias == 0).
                        let dummy = &gpu.chunks[refs.attn_norm.chunk].buf;
                        for (index, b) in
                            [(8u64, &refs.q_bias), (9, &refs.k_bias), (10, &refs.v_bias)]
                        {
                            match b {
                                Some(r) => {
                                    enc.set_buffer(index, Some(&gpu.chunks[r.chunk].buf), r.off)
                                }
                                None => enc.set_buffer(index, Some(dummy), 0),
                            }
                        }
                        enc.dispatch_thread_groups(
                            MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                            MTLSize::new(32, 1, 1),
                        );
                        enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                        // Attention.
                        let attn_req = AttnReq {
                            cache: req.cache,
                            k_off: l.k_off,
                            v_off: l.v_off,
                            q: &[],
                            kv_dim: req.kv_dim,
                            head_dim: hd,
                            n_q_heads: req.n_heads,
                            group: req.n_heads / req.n_kv_heads.max(1),
                            n_pos: req.pos + 1,
                            scale: req.scale,
                        };
                        enc.set_buffer(0, Some(&req.cache.buf), 0);
                        enc.set_buffer(1, Some(&req.cache.buf), 0);
                        enc.set_buffer(2, Some(y), e(q_at));
                        for (index, value) in [
                            (4u64, attn_req.kv_dim as u32),
                            (5, attn_req.head_dim as u32),
                            (6, attn_req.n_pos as u32),
                            (7, attn_req.group as u32),
                        ] {
                            enc.set_bytes(index, 4, &value as *const u32 as *const _);
                        }
                        enc.set_bytes(8, 4, &attn_req.scale as *const f32 as *const _);
                        for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                            enc.set_bytes(index, 8, &value as *const u64 as *const _);
                        }
                        if nsplit > 1 {
                            // Flash-decoding: partial attention per position slice, then
                            // a merge. The scalar bindings 4..=10 set above are identical;
                            // only the outputs change.
                            enc.set_compute_pipeline_state(&attend_split_state);
                            enc.set_buffer(3, Some(y), e(sp_acc_at));
                            enc.set_buffer(11, Some(y), e(sp_md_at));
                            enc.set_bytes(12, 4, &(nsplit as u32) as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((req.n_heads * nsplit) as u64, 1, 1),
                                MTLSize::new(512, 1, 1),
                            );
                            enc.memory_barrier_with_resources(&[y]);
                            enc.set_compute_pipeline_state(&attend_merge_state);
                            enc.set_buffer(0, Some(y), e(sp_acc_at));
                            enc.set_buffer(1, Some(y), e(sp_md_at));
                            enc.set_buffer(2, Some(y), e(attn_at));
                            enc.set_bytes(3, 4, &(nsplit as u32) as *const u32 as *const _);
                            enc.set_bytes(4, 4, &(hd as u32) as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        } else {
                            enc.set_compute_pipeline_state(&attend_state);
                            enc.set_buffer(3, Some(y), e(attn_at));
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(attend_tg(), 1, 1),
                            );
                        }
                    }
                }
                split_here!("attend");
                bar_c(&enc, b'a');
                // Output projection straight into delta.
                if !probe_skip("qkv") {
                    matvec(
                        &enc,
                        &astates[3],
                        &amats[3],
                        q_dim,
                        hidden,
                        attn_at,
                        delta_at,
                    );
                }
                bar_c(&enc, b'a');
                split_here!("wo");
            }
            // The megakernel absorbs the whole FFN half, resnorm included.
            if let FfnRefs::Moe {
                mega: Some(mega_state),
                router,
                router_bias,
                mats,
                strides,
                n_expert,
                expert_ffn,
                n_used,
                sigmoid,
                shared,
                shared_gate,
                ..
            } = &refs.ffn
            {
                // The megakernel covers softmax (Qwen) and sigmoid+bias
                // gating with a shared-expert slot (GLM) when the quant
                // formats have cores in the kernel.
                {
                    enc.set_compute_pipeline_state(mega_state);
                    enc.set_buffer(0, Some(y), 0);
                    enc.set_buffer(1, Some(&gpu.chunks[mats[0].chunk].buf), 0);
                    enc.set_buffer(2, Some(&gpu.chunks[mats[1].chunk].buf), 0);
                    enc.set_buffer(3, Some(&gpu.chunks[mats[2].chunk].buf), 0);
                    enc.set_buffer(4, Some(&gpu.chunks[router.chunk].buf), 0);
                    enc.set_buffer(
                        5,
                        Some(&gpu.chunks[refs.ffn_norm.chunk].buf),
                        refs.ffn_norm.off,
                    );
                    let ntg = mega_tg();
                    let margs = GpuMegaArgs {
                        hidden: hidden as u32,
                        ffn: *expert_ffn as u32,
                        n_expert: *n_expert as u32,
                        n_used: *n_used as u32,
                        x_at: x_at as u32,
                        delta_at: delta_at as u32,
                        h_at: h_at as u32,
                        logits_at: logits_at as u32,
                        ids_at: ids_at as u32,
                        wts_at: wts_at as u32,
                        gate_at: gate_at as u32,
                        up_at: up_at as u32,
                        downo_at: downo_at as u32,
                        ctr_at: ctr_at as u32,
                        n_tg: ntg,
                        ctr_base: (li as u32) * 6 * ntg,
                        eps: req.eps,
                        _pad: 0,
                        gate_off: mats[0].w_off,
                        up_off: mats[1].w_off,
                        down_off: mats[2].w_off,
                        router_off: router.w_off,
                        gate_stride: strides[0],
                        up_stride: strides[1],
                        down_stride: strides[2],
                        sigmoid: *sigmoid as u32,
                        has_shared: match (shared.is_some(), shared_gate.is_some()) {
                            (false, _) => 0,
                            (true, false) => 1,
                            (true, true) => 2,
                        },
                        sh_gate_off: shared.as_ref().map_or(0, |(m, _)| m[0].w_off),
                        sh_up_off: shared.as_ref().map_or(0, |(m, _)| m[1].w_off),
                        sh_down_off: shared.as_ref().map_or(0, |(m, _)| m[2].w_off),
                        sh_gout_off: shared_gate.as_ref().map_or(0, |(m, _)| m.w_off),
                        _tail: 0,
                    };
                    enc.set_bytes(
                        6,
                        std::mem::size_of::<GpuMegaArgs>() as u64,
                        &margs as *const GpuMegaArgs as *const _,
                    );
                    // Buffers 7-11: router bias, shared expert weights, optional
                    // shared-gate projection; dummies when unused.
                    match router_bias {
                        Some(rb) => enc.set_buffer(7, Some(&gpu.chunks[rb.chunk].buf), rb.off),
                        None => enc.set_buffer(7, Some(y), 0),
                    }
                    match shared {
                        Some((smats, _)) => {
                            enc.set_buffer(8, Some(&gpu.chunks[smats[0].chunk].buf), 0);
                            enc.set_buffer(9, Some(&gpu.chunks[smats[1].chunk].buf), 0);
                            enc.set_buffer(10, Some(&gpu.chunks[smats[2].chunk].buf), 0);
                        }
                        None => {
                            enc.set_buffer(8, Some(y), 0);
                            enc.set_buffer(9, Some(y), 0);
                            enc.set_buffer(10, Some(y), 0);
                        }
                    }
                    match shared_gate {
                        Some((sg, _)) => enc.set_buffer(11, Some(&gpu.chunks[sg.chunk].buf), 0),
                        None => enc.set_buffer(11, Some(y), 0),
                    }
                    enc.dispatch_thread_groups(
                        MTLSize::new(ntg as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    bar_c(&enc, b'f');
                    dispatched += 1;
                    continue;
                }
            }
            // h = rmsnorm(x + delta) * ffn_norm. On the plain MoE path the
            // norm fuses with the router into ONE dispatch below
            // (resnorm_router), so it runs there instead.
            // `ALLPAKA_RFUSE=1` enables; kept off under default `normflag`
            // (max-performance): fused path loses ~20–27% decode on GLM / 30B.
            let moe_plain = !refs.normflag && rfuse() && matches!(&refs.ffn, FfnRefs::Moe { .. });
            if !moe_plain {
                if refs.normflag {
                    resnorm_sig(&enc, &refs.ffn_norm, ep_ffn);
                    enc.set_buffer(9, Some(y), e(flag_at));
                    enc.set_bytes(10, 4, &ep_ffn as *const u32 as *const _);
                    if probe_skip("keepbar") {
                        bar_c(&enc, b'n');
                    }
                } else {
                    resnorm(&enc, &refs.ffn_norm);
                    bar_c(&enc, b'n');
                }
            }
            split_here!("resnorm");

            match &refs.ffn {
                FfnRefs::Dense {
                    mats,
                    states,
                    ffn_dim,
                    sw_fused,
                } => {
                    matvec(&enc, &states[0], &mats[0], hidden, *ffn_dim, h_at, gate_at);
                    matvec(&enc, &states[1], &mats[1], hidden, *ffn_dim, h_at, up_at);
                    split_here!("gate_up");
                    bar(&enc);
                    if !*sw_fused {
                        enc.set_compute_pipeline_state(&swiglu_state);
                        enc.set_buffer(0, Some(y), e(gate_at));
                        enc.set_buffer(1, Some(y), e(up_at));
                        let n = *ffn_dim as u32;
                        enc.set_bytes(2, 4, &n as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((n as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        split_here!("swiglu");
                        bar(&enc);
                        matvec(
                            &enc, &states[2], &mats[2], *ffn_dim, hidden, gate_at, delta_at,
                        );
                        dispatched += 4;
                    } else {
                        enc.set_buffer(8, Some(y), e(up_at));
                        matvec(
                            &enc, &states[2], &mats[2], *ffn_dim, hidden, gate_at, delta_at,
                        );
                        dispatched += 3;
                    }
                    split_here!("down");
                }
                FfnRefs::Moe {
                    router,
                    router_state,
                    router_bias,
                    n_expert,
                    mats,
                    states,
                    strides,
                    expert_ffn,
                    n_used,
                    sw_fused,
                    mega: _,
                    sigmoid,
                    shared,
                    shared_gate,
                    gu_dual,
                    down_tail,
                } => {
                    let n_slots = *n_used + shared.is_some() as usize;
                    let sig_last = if shared_gate.is_some() {
                        1u32
                    } else if shared.is_some() {
                        2u32
                    } else {
                        0u32
                    };
                    let mut ffn_dispatches = 0u64;
                    if !probe_skip("router") {
                        if moe_plain {
                            // FFN residual norm + router matvec + gating + top-k
                            // in ONE dispatch: both halves are single-threadgroup
                            // kernels back to back, so the boundary between them
                            // was pure launch and drain latency. Not under
                            // normflag: that path orders the norm by spin-flag.
                            // Measured 2026-09-12: under normflag, RFUSE costs
                            // ~20–27% decode (GLM / qwen3-30b) despite fewer
                            // dispatches — keep gated off.
                            enc.set_compute_pipeline_state(&resnorm_router_state);
                            enc.set_buffer(0, Some(y), e(x_at));
                            enc.set_buffer(1, Some(y), e(delta_at));
                            enc.set_buffer(2, Some(y), e(h_at));
                            enc.set_buffer(
                                3,
                                Some(&gpu.chunks[refs.ffn_norm.chunk].buf),
                                refs.ffn_norm.off,
                            );
                            let hd = hidden as u32;
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                            enc.set_buffer(6, Some(&gpu.chunks[router.chunk].buf), 0);
                            enc.set_bytes(7, 8, &router.w_off as *const u64 as *const _);
                            enc.set_buffer(8, Some(y), e(ids_at));
                            enc.set_buffer(9, Some(y), e(wts_at));
                            let n = *n_expert as u32;
                            let k = *n_used as u32;
                            enc.set_bytes(10, 4, &n as *const u32 as *const _);
                            enc.set_bytes(11, 4, &k as *const u32 as *const _);
                            match router_bias {
                                Some(rb) => {
                                    enc.set_buffer(12, Some(&gpu.chunks[rb.chunk].buf), rb.off)
                                }
                                None => enc.set_buffer(12, Some(&gpu.chunks[router.chunk].buf), 0),
                            }
                            let sg = *sigmoid as u32;
                            let hb = router_bias.is_some() as u32;
                            enc.set_bytes(13, 4, &sg as *const u32 as *const _);
                            enc.set_bytes(14, 4, &hb as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            ffn_dispatches += 1;
                        } else if !refs.normflag && rtopk_fused() {
                            // Router matvec + gating + top-k in ONE dispatch: the
                            // two tiny dispatches and the drain between them were
                            // pure stage-boundary latency. Not under normflag:
                            // that path orders the norm by spin-flag, which only
                            // the router matvec's WAIT variant can wait on.
                            enc.set_compute_pipeline_state(&rtopk_state);
                            enc.set_buffer(0, Some(&gpu.chunks[router.chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(h_at));
                            enc.set_buffer(2, Some(y), e(ids_at));
                            enc.set_buffer(3, Some(y), e(wts_at));
                            let hd = hidden as u32;
                            let n = *n_expert as u32;
                            let k = *n_used as u32;
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &n as *const u32 as *const _);
                            enc.set_bytes(6, 4, &k as *const u32 as *const _);
                            enc.set_bytes(7, 8, &router.w_off as *const u64 as *const _);
                            match router_bias {
                                Some(rb) => {
                                    enc.set_buffer(8, Some(&gpu.chunks[rb.chunk].buf), rb.off)
                                }
                                None => enc.set_buffer(8, Some(&gpu.chunks[router.chunk].buf), 0),
                            }
                            let sg = *sigmoid as u32;
                            let hb = router_bias.is_some() as u32;
                            enc.set_bytes(9, 4, &sg as *const u32 as *const _);
                            enc.set_bytes(10, 4, &hb as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            ffn_dispatches += 1;
                        } else {
                            matvec(
                                &enc,
                                router_state,
                                router,
                                hidden,
                                *n_expert,
                                h_at,
                                logits_at,
                            );
                            bar(&enc);
                            enc.set_compute_pipeline_state(&topk_state);
                            enc.set_buffer(0, Some(y), e(logits_at));
                            enc.set_buffer(1, Some(y), e(ids_at));
                            enc.set_buffer(2, Some(y), e(wts_at));
                            let n = *n_expert as u32;
                            let k = *n_used as u32;
                            enc.set_bytes(3, 4, &n as *const u32 as *const _);
                            enc.set_bytes(4, 4, &k as *const u32 as *const _);
                            // Sigmoid gating may carry a router bias (GLM exp_probs_b);
                            // the softmax kernel never reads past buffer 4.
                            match router_bias {
                                Some(rb) => {
                                    enc.set_buffer(5, Some(&gpu.chunks[rb.chunk].buf), rb.off);
                                    let one = 1u32;
                                    enc.set_bytes(6, 4, &one as *const u32 as *const _);
                                }
                                None => {
                                    enc.set_buffer(5, Some(&gpu.chunks[router.chunk].buf), 0);
                                    let zero = 0u32;
                                    enc.set_bytes(6, 4, &zero as *const u32 as *const _);
                                }
                            }
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(32, 1, 1),
                            );
                            ffn_dispatches += 2;
                        }
                    }
                    split_here!("router");
                    // Barrier after router/ids. shared_gate only writes
                    // wts[n_used] and rides with gate/up (same as verify);
                    // serial-before-barrier measured as noise on GLM (2026-09-12).
                    bar_c(&enc, b'f');
                    if !probe_skip("experts") {
                        if let Some((sg_mat, sg_state)) = shared_gate {
                            matvec(&enc, sg_state, sg_mat, hidden, 1, h_at, wts_at + *n_used);
                            ffn_dispatches += 1;
                        }
                        match gu_dual {
                            // gate + up in ONE dual-output indexed dispatch.
                            Some(st) => {
                                enc.set_compute_pipeline_state(st);
                                enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
                                enc.set_buffer(1, Some(y), e(h_at));
                                enc.set_buffer(2, Some(y), e(gate_at));
                                let a = hidden as u32;
                                let b = *expert_ffn as u32;
                                enc.set_bytes(3, 4, &a as *const u32 as *const _);
                                enc.set_bytes(4, 4, &b as *const u32 as *const _);
                                enc.set_bytes(5, 8, &mats[0].w_off as *const u64 as *const _);
                                enc.set_buffer(6, Some(y), e(ids_at));
                                let idx = GpuIdxArgs {
                                    stride: strides[0],
                                    slots: *n_used as u32,
                                    x_stride: 0,
                                    ids_stride: 0,
                                    x_row_stride: 0,
                                    y_row_stride: 0,
                                    n_rows: 0,
                                };
                                enc.set_bytes(
                                    7,
                                    std::mem::size_of::<GpuIdxArgs>() as u64,
                                    &idx as *const GpuIdxArgs as *const _,
                                );
                                enc.set_buffer(11, Some(&gpu.chunks[mats[1].chunk].buf), 0);
                                enc.set_bytes(12, 8, &mats[1].w_off as *const u64 as *const _);
                                enc.set_buffer(13, Some(y), e(up_at));
                                let lpr = lanes_per_row(mats[0].ty, hidden) as u64;
                                let tg = mv_tg();
                                enc.dispatch_thread_groups(
                                    MTLSize::new(
                                        ((*expert_ffn * *n_used * 2) as u64 * lpr).div_ceil(tg),
                                        1,
                                        1,
                                    ),
                                    MTLSize::new(tg, 1, 1),
                                );
                                ffn_dispatches += 1;
                            }
                            None => {
                                matvec_idx(
                                    &enc,
                                    &states[0],
                                    &mats[0],
                                    hidden,
                                    *expert_ffn,
                                    h_at,
                                    gate_at,
                                    strides[0],
                                    *n_used,
                                    0,
                                );
                                matvec_idx(
                                    &enc,
                                    &states[1],
                                    &mats[1],
                                    hidden,
                                    *expert_ffn,
                                    h_at,
                                    up_at,
                                    strides[1],
                                    *n_used,
                                    0,
                                );
                                ffn_dispatches += 2;
                            }
                        }
                        // Default: shared gate/up concurrent with expert gate/up.
                        // Stagger (GLM): defer shared Q5 until after the barrier so
                        // it rides with expert Q8 down instead of fighting Q4 BW.
                        let stagger = shared.is_some() && shared_stagger() && down_tail.is_none();
                        if let Some((smats, sstates)) = shared {
                            if !stagger {
                                matvec(
                                    &enc,
                                    &sstates[0],
                                    &smats[0],
                                    hidden,
                                    *expert_ffn,
                                    h_at,
                                    gate_at + *n_used * *expert_ffn,
                                );
                                matvec(
                                    &enc,
                                    &sstates[1],
                                    &smats[1],
                                    hidden,
                                    *expert_ffn,
                                    h_at,
                                    up_at + *n_used * *expert_ffn,
                                );
                                ffn_dispatches += 2;
                            }
                        }
                        split_here!("gate_up");
                        bar_c(&enc, b'f');
                        if !*sw_fused {
                            // Under stagger the shared gate/up slots are still
                            // empty here — only silu the routed experts.
                            let swiglu_n = if stagger { *n_used } else { n_slots };
                            enc.set_compute_pipeline_state(&swiglu_state);
                            enc.set_buffer(0, Some(y), e(gate_at));
                            enc.set_buffer(1, Some(y), e(up_at));
                            let n32 = (swiglu_n * *expert_ffn) as u32;
                            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            ffn_dispatches += 1;
                            split_here!("swiglu");
                            bar_c(&enc, b'f');
                        } else {
                            // The down kernel reads raw gate (x) and raw up
                            // (buffer 8) and applies swiglu on load.
                            enc.set_buffer(8, Some(y), e(up_at));
                        }
                        match (down_tail, shared) {
                            (Some(tail), Some((smats, _))) => {
                                // Routed + shared Q8_0 down in one INDEXED launch.
                                enc.set_compute_pipeline_state(tail);
                                enc.set_buffer(0, Some(&gpu.chunks[mats[2].chunk].buf), 0);
                                enc.set_buffer(1, Some(y), e(gate_at));
                                enc.set_buffer(2, Some(y), e(downo_at));
                                let a = *expert_ffn as u32;
                                let b = hidden as u32;
                                enc.set_bytes(3, 4, &a as *const u32 as *const _);
                                enc.set_bytes(4, 4, &b as *const u32 as *const _);
                                enc.set_bytes(5, 8, &mats[2].w_off as *const u64 as *const _);
                                enc.set_buffer(6, Some(y), e(ids_at));
                                let slots = (*n_used + 1) as u32;
                                let idx = GpuIdxArgs {
                                    stride: strides[2],
                                    slots,
                                    x_stride: *expert_ffn as u32,
                                    ids_stride: 0,
                                    x_row_stride: 0,
                                    y_row_stride: 0,
                                    n_rows: 0,
                                };
                                enc.set_bytes(
                                    7,
                                    std::mem::size_of::<GpuIdxArgs>() as u64,
                                    &idx as *const GpuIdxArgs as *const _,
                                );
                                if *sw_fused {
                                    enc.set_buffer(8, Some(y), e(up_at));
                                }
                                enc.set_buffer(11, Some(&gpu.chunks[smats[2].chunk].buf), 0);
                                enc.set_bytes(12, 8, &smats[2].w_off as *const u64 as *const _);
                                dispatch_mv_indexed(
                                    &enc,
                                    mats[2].ty,
                                    *expert_ffn,
                                    hidden,
                                    *n_used + 1,
                                );
                                ffn_dispatches += 1;
                            }
                            _ => {
                                matvec_idx(
                                    &enc,
                                    &states[2],
                                    &mats[2],
                                    *expert_ffn,
                                    hidden,
                                    gate_at,
                                    downo_at,
                                    strides[2],
                                    *n_used,
                                    *expert_ffn,
                                );
                                ffn_dispatches += 1;
                                if let Some((smats, sstates)) = shared {
                                    if stagger {
                                        // Shared Q5 gate/up overlaps expert Q8 down
                                        // (disjoint slots of gate/up / different
                                        // weight chunks).
                                        matvec(
                                            &enc,
                                            &sstates[0],
                                            &smats[0],
                                            hidden,
                                            *expert_ffn,
                                            h_at,
                                            gate_at + *n_used * *expert_ffn,
                                        );
                                        matvec(
                                            &enc,
                                            &sstates[1],
                                            &smats[1],
                                            hidden,
                                            *expert_ffn,
                                            h_at,
                                            up_at + *n_used * *expert_ffn,
                                        );
                                        ffn_dispatches += 2;
                                        split_here!("shared_gu");
                                        bar_c(&enc, b'f');
                                        if !*sw_fused {
                                            enc.set_compute_pipeline_state(&swiglu_state);
                                            enc.set_buffer(
                                                0,
                                                Some(y),
                                                e(gate_at + *n_used * *expert_ffn),
                                            );
                                            enc.set_buffer(
                                                1,
                                                Some(y),
                                                e(up_at + *n_used * *expert_ffn),
                                            );
                                            let n32 = *expert_ffn as u32;
                                            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                                            enc.dispatch_thread_groups(
                                                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                                                MTLSize::new(256, 1, 1),
                                            );
                                            ffn_dispatches += 1;
                                            bar_c(&enc, b'f');
                                        }
                                    }
                                    // The shared down reads its own raw up slot.
                                    if *sw_fused {
                                        enc.set_buffer(
                                            8,
                                            Some(y),
                                            e(up_at + *n_used * *expert_ffn),
                                        );
                                    }
                                    matvec(
                                        &enc,
                                        &sstates[2],
                                        &smats[2],
                                        *expert_ffn,
                                        hidden,
                                        gate_at + *n_used * *expert_ffn,
                                        downo_at + *n_used * hidden,
                                    );
                                    ffn_dispatches += 1;
                                }
                            }
                        }
                    }
                    split_here!("down");
                    bar_c(&enc, b'f');
                    if cfuse() {
                        // No standalone combine: it folds into the next
                        // residual norm (combine_resnorm), which the barrier
                        // above already orders against the down writes.
                        pending_combine = Some((n_slots as u32, sig_last));
                    } else {
                        combine(&enc, n_slots as u32, sig_last);
                        ffn_dispatches += 1;
                    }
                    split_here!("combine");
                    dispatched += ffn_dispatches;
                }
            }
            // The fused combine+norm is ordered by the down stage's barrier;
            // only the plain paths still need the layer-end drain.
            if pending_combine.is_none() {
                bar(&enc);
            }
            dispatched += 8;
        }
        // Final norm and the output projection, still in the same buffer; a
        // pending combine from the last MoE layer folds into the final norm.
        match pending_combine.take() {
            Some((slots, sig_last)) => combine_resnorm(&enc, &out_norm, slots, sig_last),
            None => resnorm(&enc, &out_norm),
        }
        split_here!("resnorm");
        bar(&enc);
        if !probe_skip("head") {
            matvec(
                &enc,
                &out_state,
                &out_mat,
                hidden,
                vocab,
                h_at,
                out_logits_at,
            );
        }
        split_here!("head");
        if req.argmax {
            // Greedy: the winner reduces on the GPU; the readback below is
            // one 32-bit word instead of the whole vocabulary.
            bar(&enc);
            enc.set_compute_pipeline_state(&argmax_state);
            enc.set_buffer(0, Some(y), e(out_logits_at));
            let v32 = vocab as u32;
            enc.set_bytes(1, 4, &v32 as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(amax_at));
            enc.dispatch_thread_groups(MTLSize::new(64, 1, 1), MTLSize::new(256, 1, 1));
            bar(&enc);
            enc.set_compute_pipeline_state(&argmax_final_state);
            enc.set_buffer(0, Some(y), e(amax_at));
            let np = 64u32;
            enc.set_bytes(1, 4, &np as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(amax_at + 64 * 2));
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            split_here!("argmax");
        }
        enc.end_encoding();
        if let (Some(counter), Some(resolved)) = (split_counter.as_ref(), split_resolved.as_ref()) {
            let blit = cmd.new_blit_command_encoder();
            blit.resolve_counters(counter, metal::NSRange::new(0, split_index), resolved, 0);
            blit.end_encoding();
        }
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        if any_mega && mega_debug() {
            eprintln!(
                "mega: encoded in {:.1} ms, committing ({} mega layers)",
                t_encode.elapsed().as_secs_f64() * 1e3,
                layers
                    .iter()
                    .filter(|l| matches!(&l.ffn, FfnRefs::Moe { mega: Some(_), .. }))
                    .count()
            );
        }
        let cmd = cmd.commit_and_wait();
        if any_mega && mega_debug() {
            eprintln!(
                "mega: GPU wait {:.1} ms",
                t_wait.elapsed().as_secs_f64() * 1e3
            );
        }
        note_gpu_times(cmd);
        if let Some(resolved) = split_resolved.as_ref() {
            let mut cpu_end = 0u64;
            let mut gpu_end = 0u64;
            gpu.device.sample_timestamps(&mut cpu_end, &mut gpu_end);
            let samples = unsafe {
                std::slice::from_raw_parts(resolved.contents() as *const u64, split_index as usize)
            };
            let scale = if gpu_end > split_gpu_start {
                (cpu_end - split_cpu_start) as f64 / (gpu_end - split_gpu_start) as f64
            } else {
                1.0
            };
            let mut split_acc: Vec<(&'static str, f64)> = Vec::new();
            for (stage, label) in split_labels.iter().enumerate() {
                let begin = samples[stage * 2];
                let end = samples[stage * 2 + 1];
                let ms = end.saturating_sub(begin) as f64 * scale / 1e6;
                match split_acc.iter_mut().find(|(known, _)| known == label) {
                    Some(entry) => entry.1 += ms,
                    None => split_acc.push((*label, ms)),
                }
            }
            let total: f64 = split_acc.iter().map(|entry| entry.1).sum();
            for &(label, ms) in &split_acc {
                let phase = match label {
                    "resnorm" => crate::telemetry::Phase::Norm,
                    "qkv" => crate::telemetry::Phase::Qkv,
                    "attend" | "wo" => crate::telemetry::Phase::Attention,
                    "router" => crate::telemetry::Phase::Router,
                    "gate_up" | "swiglu" | "down" | "combine" => crate::telemetry::Phase::Experts,
                    _ => crate::telemetry::Phase::Other,
                };
                crate::telemetry::record_global(
                    phase,
                    std::time::Duration::from_secs_f64(ms / 1e3),
                );
            }
            static PRINTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !PRINTED.swap(true, Ordering::Relaxed) {
                eprintln!("decode profile, ms/token (GPU counters, one submit):");
                for (label, ms) in split_acc {
                    eprintln!("  {label:<10} {ms:6.3} ({:4.1}%)", ms / total * 100.0);
                }
            }
        }
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(dispatched + 2, Ordering::Relaxed);

        if req.argmax {
            let next = unsafe {
                ((gpu.y_arena.contents() as *const f32).add(amax_at + 64 * 2) as *const u32).read()
            };
            TokenOut::Argmax(next)
        } else {
            let mut out = vec![0f32; vocab];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (gpu.y_arena.contents() as *const f32).add(out_logits_at),
                    out.as_mut_ptr(),
                    vocab,
                );
            }
            TokenOut::Logits(out)
        }
    });
    Some(out)
}

pub(super) fn dq_fmt(ty: GgmlType) -> Option<u32> {
    Some(match ty {
        GgmlType::Q4K => 0,
        GgmlType::Q5K => 1,
        GgmlType::Q6K => 2,
        GgmlType::Q8_0 => 3,
        GgmlType::Q2K => 4,
        GgmlType::Q3K => 5,
        GgmlType::BF16 => 6,
        GgmlType::F16 => 7,
        GgmlType::F32 => 8,
        _ => return None,
    })
}

pub(super) fn row_bytes(ty: GgmlType, n_in: usize) -> Option<usize> {
    let be = ty.block_elements()? as usize;
    let bb = ty.block_bytes()? as usize;
    if n_in % be != 0 {
        return None;
    }
    Some(n_in / be * bb)
}

/// After a successful greedy decode, run `n_more` further tokens on-device:
/// embed(hist[step]) -> layers 0..N -> output_head -> argmax -> hist[step+1],
/// committed in one command buffer with exactly ONE wait at the end.
pub fn decode_greedy_continue(req: &TokenReq, n_more: usize) -> Option<Vec<u32>> {
    if n_more == 0 {
        return Some(Vec::new());
    }
    if !is_attached() || !req.argmax {
        return None;
    }
    let (embd_ty, embd_bytes, hidden) = req.embd?;
    let seed_token = req.token_id?;
    if hidden == 0 || hidden % 4 != 0 {
        return None;
    }
    let hd = req.head_dim;
    if (hd != 128 && hd != 256)
        || req.rot_dim != hd && req.rot_dim * 2 != hd && !(hd == 256 && req.rot_dim == 64)
    {
        return None;
    }
    let span_end = (req.pos + n_more) * req.kv_dim;
    if req.pos + n_more > req.cache_capacity {
        return None;
    }
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let vocab = req.output.2;
    let rp = req.rot_dim / 2;

    let Some(Some(cell)) = GPU.get().map(|c| c.as_ref()) else {
        return None;
    };
    let Ok(mut gpu) = cell.lock() else {
        return None;
    };

    let embd_addr = embd_bytes.as_ptr() as usize;
    let embd_chunk = gpu.chunk_for(embd_addr, embd_bytes.len())?;
    let embd_base = (embd_addr - gpu.chunks[embd_chunk].start) as u64;
    let rb = row_bytes(embd_ty, hidden)?;
    let fmt = dq_fmt(embd_ty)?;

    if req.layers.iter().any(|l| l.gdn.is_some()) {
        return None;
    }
    let CachedDecodeLayers {
        layers,
        max_ffn,
        max_slots,
        max_expert,
        ..
    } = resolve_decode_layers(&mut gpu, req, hidden, q_dim, kv, span_end, false, false)?;

    let out_norm = resolve_norm(&gpu, req.output_norm, hidden)?;
    let out_mat = resolve(&gpu, req.output.0, req.output.1)?;
    let out_state = gpu
        .pipeline(out_mat.kernel, 1, lanes_per_row(out_mat.ty, hidden))?
        .to_owned();

    let attend_state = gpu.pipelines[&(attend_kernel(), 1, 1)].to_owned();
    let attend_split_state = gpu.pipelines[&("attend_split", 1, 1)].to_owned();
    let attend_merge_state = gpu.pipelines[&("attend_merge", 1, 1)].to_owned();
    let qk_prep_state = gpu.pipelines[&("qk_prep", 1, 1)].to_owned();
    let resnorm_state = gpu.pipelines[&("residual_norm", 1, 1)].to_owned();
    let qk_prep256_state = gpu.pipelines[&("qk_prep256", 1, 1)].to_owned();
    let attend256_state = gpu.pipelines[&("attend_s32_256", 1, 1)].to_owned();
    let attend_split256_state = gpu.pipelines[&("attend_split_256", 1, 1)].to_owned();
    let gate_mul_state = gpu.pipelines[&("attn_gate_mul", 1, 1)].to_owned();

    let sigmoid_gating = req
        .layers
        .iter()
        .any(|l| matches!(l.ffn, TokenFfn::Moe { sigmoid: true, .. }));
    let topk_state = gpu.pipelines[&(
        if sigmoid_gating {
            "sigmoid_topk"
        } else {
            "softmax_topk"
        },
        1,
        1,
    )]
        .to_owned();
    let rtopk_state = gpu.pipelines[&("router_topk", 1, 1)].to_owned();
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();
    let combine_state = gpu.pipelines[&("moe_combine", 1, 1)].to_owned();
    let combine_resnorm_state = gpu.pipelines[&("combine_resnorm", 1, 1)].to_owned();
    let argmax_state = gpu.pipelines[&("argmax_f32", 1, 1)].to_owned();
    let argmax_final_state = gpu.pipelines[&("argmax_final", 1, 1)].to_owned();
    let embed_state = gpu.pipelines[&("embed_row_f32_dtoken", 1, 1)].to_owned();
    let zero_state = gpu.pipelines[&("zero_f32", 1, 1)].to_owned();
    let store_state = gpu.pipelines[&("store_u32_at", 1, 1)].to_owned();

    let align = |n: usize| (n + 63) & !63;
    let wq_max = req
        .layers
        .iter()
        .map(|l| l.wq.2)
        .max()
        .unwrap_or(q_dim)
        .max(q_dim);
    let x_at = 0usize;
    let delta_at = x_at + align(hidden);
    let h_at = delta_at + align(hidden);
    let q_at = h_at + align(hidden);
    let qn_at = q_at + align(wq_max);
    let qgate_at = qn_at + align(q_dim);
    let k_at = qgate_at + align(q_dim);
    let v_at = k_at + align(kv);
    let attn_at = v_at + align(kv);
    let logits_at = attn_at + align(q_dim);
    let ids_at = logits_at + align(max_expert.max(1));
    let wts_at = ids_at + align(max_slots);
    let gate_at = wts_at + align(max_slots);
    let up_at = gate_at + align(max_slots * max_ffn);
    let downo_at = up_at + align(max_slots * max_ffn);
    let out_logits_at = downo_at + align(max_slots * hidden);
    let flag_at = out_logits_at + align(vocab);
    let ctr_at = flag_at + align(1);
    let max_n_pos = req.pos + n_more;
    let forced_split = crate::runtime::get().attention_split;
    let max_nsplit = forced_split
        .unwrap_or_else(|| (max_n_pos / 128).clamp(1, 12))
        .clamp(1, 16)
        .min(max_n_pos);
    let sp_acc_at = ctr_at + align(1);
    let sp_md_at = sp_acc_at + align(req.n_heads * max_nsplit * hd);
    let amax_at = sp_md_at + align(req.n_heads * max_nsplit * 2);
    let total = amax_at + align(64 * 2 + 1);
    gpu.ensure_arenas(4096, total * 4);

    let mut rope_tables: Vec<[f32; 2]> = Vec::new();
    if req.rope.len() < n_more * rp {
        if let Ok(lock) = ROPE_INV_FREQ.lock() {
            if lock.len() >= rp {
                rope_tables.reserve(n_more * rp);
                for step in 0..n_more {
                    let p = (req.pos + step) as f32;
                    for &freq in &lock[..rp] {
                        let (s, c) = (p * freq).sin_cos();
                        rope_tables.push([s, c]);
                    }
                }
            }
        }
    }
    if req.rope.len() < n_more * rp && rope_tables.len() < n_more * rp {
        return None;
    }

    let hist_buf = gpu.device.new_buffer(
        ((n_more + 1) * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        *(hist_buf.contents() as *mut u32) = seed_token;
        let yp = gpu.y_arena.contents() as *mut f32;
        for l in &*layers {
            if let FfnRefs::Moe {
                n_used,
                shared: Some(_),
                shared_gate: None,
                ..
            } = &l.ffn
            {
                *yp.add(wts_at + *n_used) = 1.0;
            }
        }
    }

    objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let command_buffer = gpu.queue.new_command_buffer();
        let cmd = crate::command::CommandScope::new(command_buffer);
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let y = &gpu.y_arena;
        let bar_c = |enc: &metal::ComputeCommandEncoderRef| enc.memory_barrier_with_resources(&[y]);
        let bar = |enc: &metal::ComputeCommandEncoderRef| enc.memory_barrier_with_resources(&[y]);
        let e = |off: usize| (off * 4) as u64;

        let matvec = |enc: &metal::ComputeCommandEncoderRef,
                      state: &ComputePipelineState,
                      mat: &MatRef,
                      n_in: usize,
                      n_out: usize,
                      x_off: usize,
                      y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            dispatch_mv(enc, mat.ty, n_in, n_out);
        };

        let matvec_idx = |enc: &metal::ComputeCommandEncoderRef,
                          state: &ComputePipelineState,
                          mat: &MatRef,
                          n_in: usize,
                          n_out: usize,
                          x_off: usize,
                          y_off: usize,
                          stride: u64,
                          slots: usize,
                          x_stride: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(y), e(ids_at));
            let idx = GpuIdxArgs {
                stride,
                slots: slots as u32,
                x_stride: x_stride as u32,
                ids_stride: 0,
                x_row_stride: 0,
                y_row_stride: 0,
                n_rows: 0,
            };
            enc.set_bytes(
                7,
                std::mem::size_of::<GpuIdxArgs>() as u64,
                &idx as *const GpuIdxArgs as *const _,
            );
            dispatch_mv_indexed(enc, mat.ty, n_in, n_out, slots);
        };

        let resnorm = |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef| {
            enc.set_compute_pipeline_state(&resnorm_state);
            enc.set_buffer(0, Some(y), e(x_at));
            enc.set_buffer(1, Some(y), e(delta_at));
            enc.set_buffer(2, Some(y), e(h_at));
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        };

        let combine = |enc: &metal::ComputeCommandEncoderRef, slots: u32, sig_last: u32| {
            enc.set_compute_pipeline_state(&combine_state);
            enc.set_buffer(0, Some(y), e(downo_at));
            enc.set_buffer(1, Some(y), e(wts_at));
            enc.set_buffer(2, Some(y), e(delta_at));
            let n = hidden as u32;
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &slots as *const u32 as *const _);
            enc.set_bytes(5, 4, &sig_last as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((hidden as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        };

        let combine_resnorm =
            |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef, slots: u32, sig_last: u32| {
                enc.set_compute_pipeline_state(&combine_resnorm_state);
                enc.set_buffer(0, Some(y), e(x_at));
                enc.set_buffer(1, Some(y), e(delta_at));
                enc.set_buffer(2, Some(y), e(h_at));
                enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
                let n = hidden as u32;
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                enc.set_buffer(6, Some(y), e(downo_at));
                enc.set_buffer(7, Some(y), e(wts_at));
                enc.set_bytes(8, 4, &slots as *const u32 as *const _);
                enc.set_bytes(9, 4, &sig_last as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            };

        for step in 0..n_more {
            let pos = req.pos + step;
            let n_pos = pos + 1;
            let nsplit = forced_split
                .unwrap_or_else(|| (n_pos / 128).clamp(1, 12))
                .clamp(1, 16)
                .min(n_pos);

            enc.set_compute_pipeline_state(&embed_state);
            enc.set_buffer(0, Some(&gpu.chunks[embd_chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_at));
            let h_u = hidden as u32;
            enc.set_bytes(2, 4, &h_u as *const u32 as *const _);
            enc.set_bytes(3, 8, &embd_base as *const u64 as *const _);
            let rb_u = rb as u32;
            enc.set_bytes(4, 4, &rb_u as *const u32 as *const _);
            enc.set_bytes(5, 4, &fmt as *const u32 as *const _);
            enc.set_buffer(6, Some(&hist_buf), 0);
            let tok_idx = step as u32;
            enc.set_bytes(7, 4, &tok_idx as *const u32 as *const _);
            let n_tg = if fmt <= 5 {
                (hidden as u64).div_ceil(256)
            } else {
                (hidden as u64).div_ceil(32)
            };
            enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(32, 1, 1));

            enc.set_compute_pipeline_state(&zero_state);
            enc.set_buffer(0, Some(y), e(delta_at));
            let n_zero = hidden as u32;
            enc.set_bytes(1, 4, &n_zero as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((hidden as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );

            enc.memory_barrier_with_resources(&[y, &hist_buf]);

            let rope_slice = if req.rope.len() >= (step + 1) * rp {
                &req.rope[step * rp..(step + 1) * rp]
            } else if rope_tables.len() >= (step + 1) * rp {
                &rope_tables[step * rp..(step + 1) * rp]
            } else {
                &[]
            };

            let mut pending_combine: Option<(u32, u32)> = None;
            for (l, refs) in req.layers.iter().zip(&*layers) {
                let fused = match pending_combine.take() {
                    Some((slots, sig_last)) => {
                        combine_resnorm(&enc, &refs.attn_norm, slots, sig_last);
                        bar_c(&enc);
                        true
                    }
                    None => false,
                };
                if !fused {
                    resnorm(&enc, &refs.attn_norm);
                    bar_c(&enc);
                }

                let amats = refs.mats.as_ref().unwrap();
                let astates = refs.mat_states.as_ref().unwrap();
                matvec(&enc, &astates[0], &amats[0], hidden, l.wq.2, h_at, q_at);
                matvec(&enc, &astates[1], &amats[1], hidden, kv, h_at, k_at);
                matvec(&enc, &astates[2], &amats[2], hidden, kv, h_at, v_at);
                bar_c(&enc);

                if hd == 256 {
                    enc.set_compute_pipeline_state(&qk_prep256_state);
                    enc.set_buffer(0, Some(y), e(q_at));
                    enc.set_buffer(1, Some(y), e(k_at));
                    enc.set_buffer(2, Some(y), e(v_at));
                    enc.set_buffer(3, Some(&req.cache.buf), 0);
                    let args = QkPrep256Args {
                        n_heads: req.n_heads as u32,
                        n_kv_heads: req.n_kv_heads as u32,
                        kv_dim: req.kv_dim as u32,
                        eps: req.eps,
                        pos: pos as u32,
                        has_qk_norm: l.q_norm.is_some() as u32,
                        rot_dim: req.rot_dim as u32,
                        gate_in_q: l.gate_in_q as u32,
                        k_base: l.k_off as u64,
                        v_base: l.v_off as u64,
                    };
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<QkPrep256Args>() as u64,
                        &args as *const QkPrep256Args as *const _,
                    );
                    let ones = [1.0f32; 256];
                    let qw = l.q_norm.unwrap_or(&ones);
                    let kw = l.k_norm.unwrap_or(&ones);
                    enc.set_bytes(5, (256 * 4) as u64, qw.as_ptr() as *const _);
                    enc.set_bytes(6, (256 * 4) as u64, kw.as_ptr() as *const _);
                    enc.set_bytes(
                        7,
                        (rope_slice.len() * 8) as u64,
                        rope_slice.as_ptr() as *const _,
                    );
                    enc.set_buffer(8, Some(y), e(qn_at));
                    enc.set_buffer(9, Some(y), e(qgate_at));
                    enc.dispatch_thread_groups(
                        MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                        MTLSize::new(32, 1, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                    let group = (req.n_heads / req.n_kv_heads.max(1)) as u32;
                    enc.set_buffer(0, Some(&req.cache.buf), 0);
                    enc.set_buffer(1, Some(&req.cache.buf), 0);
                    enc.set_buffer(2, Some(y), e(qn_at));
                    for (index, value) in [
                        (4u64, req.kv_dim as u32),
                        (5, 256u32),
                        (6, n_pos as u32),
                        (7, group),
                    ] {
                        enc.set_bytes(index, 4, &value as *const u32 as *const _);
                    }
                    enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                    for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                        enc.set_bytes(index, 8, &value as *const u64 as *const _);
                    }
                    if nsplit > 1 {
                        enc.set_compute_pipeline_state(&attend_split256_state);
                        enc.set_buffer(3, Some(y), e(sp_acc_at));
                        enc.set_buffer(11, Some(y), e(sp_md_at));
                        enc.set_bytes(12, 4, &(nsplit as u32) as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((req.n_heads * nsplit) as u64, 1, 1),
                            MTLSize::new(512, 1, 1),
                        );
                        enc.memory_barrier_with_resources(&[y]);
                        enc.set_compute_pipeline_state(&attend_merge_state);
                        enc.set_buffer(0, Some(y), e(sp_acc_at));
                        enc.set_buffer(1, Some(y), e(sp_md_at));
                        enc.set_buffer(2, Some(y), e(attn_at));
                        enc.set_bytes(3, 4, &(nsplit as u32) as *const u32 as *const _);
                        let hd_u = hd as u32;
                        enc.set_bytes(4, 4, &hd_u as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(req.n_heads as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    } else {
                        enc.set_compute_pipeline_state(&attend256_state);
                        enc.set_buffer(3, Some(y), e(attn_at));
                        enc.dispatch_thread_groups(
                            MTLSize::new(req.n_heads as u64, 1, 1),
                            MTLSize::new(512, 1, 1),
                        );
                    }
                    if l.gate_in_q {
                        enc.memory_barrier_with_resources(&[y]);
                        enc.set_compute_pipeline_state(&gate_mul_state);
                        enc.set_buffer(0, Some(y), e(attn_at));
                        enc.set_buffer(1, Some(y), e(qgate_at));
                        let n = q_dim as u32;
                        enc.set_bytes(2, 4, &n as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((q_dim as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                    }
                } else {
                    enc.set_compute_pipeline_state(&qk_prep_state);
                    enc.set_buffer(0, Some(y), e(q_at));
                    enc.set_buffer(1, Some(y), e(k_at));
                    enc.set_buffer(2, Some(y), e(v_at));
                    enc.set_buffer(3, Some(&req.cache.buf), 0);
                    let args = QkPrepArgs {
                        n_heads: req.n_heads as u32,
                        n_kv_heads: req.n_kv_heads as u32,
                        head_dim: hd as u32,
                        kv_dim: req.kv_dim as u32,
                        eps: req.eps,
                        pos: pos as u32,
                        has_qk_norm: l.q_norm.is_some() as u32,
                        rot_dim: req.rot_dim as u32,
                        k_base: l.k_off as u64,
                        v_base: l.v_off as u64,
                        has_bias: refs.q_bias.is_some() as u32,
                        pad2: 0,
                    };
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<QkPrepArgs>() as u64,
                        &args as *const QkPrepArgs as *const _,
                    );
                    let ones = [1.0f32; ATTN_HEAD_DIM];
                    let qw = l.q_norm.unwrap_or(&ones);
                    let kw = l.k_norm.unwrap_or(&ones);
                    enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
                    enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
                    enc.set_bytes(
                        7,
                        (rope_slice.len() * 8) as u64,
                        rope_slice.as_ptr() as *const _,
                    );
                    let dummy = &gpu.chunks[refs.attn_norm.chunk].buf;
                    for (index, b) in [(8u64, &refs.q_bias), (9, &refs.k_bias), (10, &refs.v_bias)]
                    {
                        match b {
                            Some(rb) => {
                                enc.set_buffer(index, Some(&gpu.chunks[rb.chunk].buf), rb.off)
                            }
                            None => enc.set_buffer(index, Some(dummy), 0),
                        }
                    }
                    enc.dispatch_thread_groups(
                        MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                        MTLSize::new(32, 1, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                    let group = (req.n_heads / req.n_kv_heads.max(1)) as u32;
                    enc.set_buffer(0, Some(&req.cache.buf), 0);
                    enc.set_buffer(1, Some(&req.cache.buf), 0);
                    enc.set_buffer(2, Some(y), e(q_at));
                    for (index, value) in [
                        (4u64, req.kv_dim as u32),
                        (5, hd as u32),
                        (6, n_pos as u32),
                        (7, group),
                    ] {
                        enc.set_bytes(index, 4, &value as *const u32 as *const _);
                    }
                    enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                    for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                        enc.set_bytes(index, 8, &value as *const u64 as *const _);
                    }
                    if nsplit > 1 {
                        enc.set_compute_pipeline_state(&attend_split_state);
                        enc.set_buffer(3, Some(y), e(sp_acc_at));
                        enc.set_buffer(11, Some(y), e(sp_md_at));
                        enc.set_bytes(12, 4, &(nsplit as u32) as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((req.n_heads * nsplit) as u64, 1, 1),
                            MTLSize::new(512, 1, 1),
                        );
                        enc.memory_barrier_with_resources(&[y]);
                        enc.set_compute_pipeline_state(&attend_merge_state);
                        enc.set_buffer(0, Some(y), e(sp_acc_at));
                        enc.set_buffer(1, Some(y), e(sp_md_at));
                        enc.set_buffer(2, Some(y), e(attn_at));
                        enc.set_bytes(3, 4, &(nsplit as u32) as *const u32 as *const _);
                        let hd_u = hd as u32;
                        enc.set_bytes(4, 4, &hd_u as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(req.n_heads as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    } else {
                        enc.set_compute_pipeline_state(&attend_state);
                        enc.set_buffer(3, Some(y), e(attn_at));
                        enc.dispatch_thread_groups(
                            MTLSize::new(req.n_heads as u64, 1, 1),
                            MTLSize::new(attend_tg(), 1, 1),
                        );
                    }
                }
                bar_c(&enc);

                matvec(
                    &enc,
                    &astates[3],
                    &amats[3],
                    q_dim,
                    hidden,
                    attn_at,
                    delta_at,
                );
                bar_c(&enc);

                resnorm(&enc, &refs.ffn_norm);
                bar_c(&enc);

                match &refs.ffn {
                    FfnRefs::Dense {
                        mats,
                        states,
                        ffn_dim,
                        sw_fused,
                    } => {
                        matvec(&enc, &states[0], &mats[0], hidden, *ffn_dim, h_at, gate_at);
                        matvec(&enc, &states[1], &mats[1], hidden, *ffn_dim, h_at, up_at);
                        bar_c(&enc);
                        if !*sw_fused {
                            enc.set_compute_pipeline_state(&swiglu_state);
                            enc.set_buffer(0, Some(y), e(gate_at));
                            enc.set_buffer(1, Some(y), e(up_at));
                            let n = *ffn_dim as u32;
                            enc.set_bytes(2, 4, &n as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((n as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            bar_c(&enc);
                            matvec(
                                &enc, &states[2], &mats[2], *ffn_dim, hidden, gate_at, delta_at,
                            );
                        } else {
                            enc.set_buffer(8, Some(y), e(up_at));
                            matvec(
                                &enc, &states[2], &mats[2], *ffn_dim, hidden, gate_at, delta_at,
                            );
                        }
                    }
                    FfnRefs::Moe {
                        router,
                        router_state,
                        router_bias,
                        n_expert,
                        mats,
                        states,
                        strides,
                        expert_ffn,
                        n_used,
                        sw_fused,
                        sigmoid,
                        shared,
                        shared_gate,
                        gu_dual,
                        down_tail,
                        ..
                    } => {
                        let n_slots = *n_used + shared.is_some() as usize;
                        let sig_last = if shared_gate.is_some() {
                            1u32
                        } else if shared.is_some() {
                            2u32
                        } else {
                            0u32
                        };
                        if rtopk_fused() {
                            enc.set_compute_pipeline_state(&rtopk_state);
                            enc.set_buffer(0, Some(&gpu.chunks[router.chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(h_at));
                            enc.set_buffer(2, Some(y), e(ids_at));
                            enc.set_buffer(3, Some(y), e(wts_at));
                            let hd_u = hidden as u32;
                            let n_u = *n_expert as u32;
                            let k_u = *n_used as u32;
                            enc.set_bytes(4, 4, &hd_u as *const u32 as *const _);
                            enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
                            enc.set_bytes(6, 4, &k_u as *const u32 as *const _);
                            enc.set_bytes(7, 8, &router.w_off as *const u64 as *const _);
                            match router_bias {
                                Some(rb) => {
                                    enc.set_buffer(8, Some(&gpu.chunks[rb.chunk].buf), rb.off)
                                }
                                None => enc.set_buffer(8, Some(&gpu.chunks[router.chunk].buf), 0),
                            }
                            let sg_u = *sigmoid as u32;
                            let hb_u = router_bias.is_some() as u32;
                            enc.set_bytes(9, 4, &sg_u as *const u32 as *const _);
                            enc.set_bytes(10, 4, &hb_u as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        } else {
                            matvec(
                                &enc,
                                router_state,
                                router,
                                hidden,
                                *n_expert,
                                h_at,
                                logits_at,
                            );
                            bar(&enc);
                            enc.set_compute_pipeline_state(&topk_state);
                            enc.set_buffer(0, Some(y), e(logits_at));
                            enc.set_buffer(1, Some(y), e(ids_at));
                            enc.set_buffer(2, Some(y), e(wts_at));
                            let n_u = *n_expert as u32;
                            let k_u = *n_used as u32;
                            enc.set_bytes(3, 4, &n_u as *const u32 as *const _);
                            enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
                            match router_bias {
                                Some(rb) => {
                                    enc.set_buffer(5, Some(&gpu.chunks[rb.chunk].buf), rb.off);
                                    let one = 1u32;
                                    enc.set_bytes(6, 4, &one as *const u32 as *const _);
                                }
                                None => {
                                    enc.set_buffer(5, Some(&gpu.chunks[router.chunk].buf), 0);
                                    let zero = 0u32;
                                    enc.set_bytes(6, 4, &zero as *const u32 as *const _);
                                }
                            }
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(32, 1, 1),
                            );
                        }
                        bar_c(&enc);

                        if let Some((sg_mat, sg_state)) = shared_gate {
                            matvec(&enc, sg_state, sg_mat, hidden, 1, h_at, wts_at + *n_used);
                        }

                        match gu_dual {
                            Some(st) => {
                                enc.set_compute_pipeline_state(st);
                                enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
                                enc.set_buffer(1, Some(y), e(h_at));
                                enc.set_buffer(2, Some(y), e(gate_at));
                                let a = hidden as u32;
                                let b = *expert_ffn as u32;
                                enc.set_bytes(3, 4, &a as *const u32 as *const _);
                                enc.set_bytes(4, 4, &b as *const u32 as *const _);
                                enc.set_bytes(5, 8, &mats[0].w_off as *const u64 as *const _);
                                enc.set_buffer(6, Some(y), e(ids_at));
                                let idx = GpuIdxArgs {
                                    stride: strides[0],
                                    slots: *n_used as u32,
                                    x_stride: 0,
                                    ids_stride: 0,
                                    x_row_stride: 0,
                                    y_row_stride: 0,
                                    n_rows: 0,
                                };
                                enc.set_bytes(
                                    7,
                                    std::mem::size_of::<GpuIdxArgs>() as u64,
                                    &idx as *const GpuIdxArgs as *const _,
                                );
                                enc.set_buffer(11, Some(&gpu.chunks[mats[1].chunk].buf), 0);
                                enc.set_bytes(12, 8, &mats[1].w_off as *const u64 as *const _);
                                enc.set_buffer(13, Some(y), e(up_at));
                                let lpr = lanes_per_row(mats[0].ty, hidden) as u64;
                                let tg = mv_tg();
                                enc.dispatch_thread_groups(
                                    MTLSize::new(
                                        ((*expert_ffn * *n_used * 2) as u64 * lpr).div_ceil(tg),
                                        1,
                                        1,
                                    ),
                                    MTLSize::new(tg, 1, 1),
                                );
                            }
                            None => {
                                matvec_idx(
                                    &enc,
                                    &states[0],
                                    &mats[0],
                                    hidden,
                                    *expert_ffn,
                                    h_at,
                                    gate_at,
                                    strides[0],
                                    *n_used,
                                    0,
                                );
                                matvec_idx(
                                    &enc,
                                    &states[1],
                                    &mats[1],
                                    hidden,
                                    *expert_ffn,
                                    h_at,
                                    up_at,
                                    strides[1],
                                    *n_used,
                                    0,
                                );
                            }
                        }
                        if let Some((smats, sstates)) = shared {
                            matvec(
                                &enc,
                                &sstates[0],
                                &smats[0],
                                hidden,
                                *expert_ffn,
                                h_at,
                                gate_at + *n_used * *expert_ffn,
                            );
                            matvec(
                                &enc,
                                &sstates[1],
                                &smats[1],
                                hidden,
                                *expert_ffn,
                                h_at,
                                up_at + *n_used * *expert_ffn,
                            );
                        }
                        bar_c(&enc);

                        if !*sw_fused {
                            enc.set_compute_pipeline_state(&swiglu_state);
                            enc.set_buffer(0, Some(y), e(gate_at));
                            enc.set_buffer(1, Some(y), e(up_at));
                            let n32 = (n_slots * *expert_ffn) as u32;
                            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            bar_c(&enc);
                        } else {
                            enc.set_buffer(8, Some(y), e(up_at));
                        }

                        match (down_tail, shared) {
                            (Some(tail), Some((smats, _))) => {
                                enc.set_compute_pipeline_state(tail);
                                enc.set_buffer(0, Some(&gpu.chunks[mats[2].chunk].buf), 0);
                                enc.set_buffer(1, Some(y), e(gate_at));
                                enc.set_buffer(2, Some(y), e(downo_at));
                                let a = *expert_ffn as u32;
                                let b = hidden as u32;
                                enc.set_bytes(3, 4, &a as *const u32 as *const _);
                                enc.set_bytes(4, 4, &b as *const u32 as *const _);
                                enc.set_bytes(5, 8, &mats[2].w_off as *const u64 as *const _);
                                enc.set_buffer(6, Some(y), e(ids_at));
                                let slots = (*n_used + 1) as u32;
                                let idx = GpuIdxArgs {
                                    stride: strides[2],
                                    slots,
                                    x_stride: *expert_ffn as u32,
                                    ids_stride: 0,
                                    x_row_stride: 0,
                                    y_row_stride: 0,
                                    n_rows: 0,
                                };
                                enc.set_bytes(
                                    7,
                                    std::mem::size_of::<GpuIdxArgs>() as u64,
                                    &idx as *const GpuIdxArgs as *const _,
                                );
                                if *sw_fused {
                                    enc.set_buffer(8, Some(y), e(up_at));
                                }
                                enc.set_buffer(11, Some(&gpu.chunks[smats[2].chunk].buf), 0);
                                enc.set_bytes(12, 8, &smats[2].w_off as *const u64 as *const _);
                                dispatch_mv_indexed(
                                    &enc,
                                    mats[2].ty,
                                    *expert_ffn,
                                    hidden,
                                    *n_used + 1,
                                );
                            }
                            _ => {
                                matvec_idx(
                                    &enc,
                                    &states[2],
                                    &mats[2],
                                    *expert_ffn,
                                    hidden,
                                    gate_at,
                                    downo_at,
                                    strides[2],
                                    *n_used,
                                    *expert_ffn,
                                );
                                if let Some((smats, sstates)) = shared {
                                    if *sw_fused {
                                        enc.set_buffer(
                                            8,
                                            Some(y),
                                            e(up_at + *n_used * *expert_ffn),
                                        );
                                    }
                                    matvec(
                                        &enc,
                                        &sstates[2],
                                        &smats[2],
                                        *expert_ffn,
                                        hidden,
                                        gate_at + *n_used * *expert_ffn,
                                        downo_at + *n_used * hidden,
                                    );
                                }
                            }
                        }
                        bar_c(&enc);
                        if cfuse() {
                            pending_combine = Some((n_slots as u32, sig_last));
                        } else {
                            combine(&enc, n_slots as u32, sig_last);
                        }
                    }
                }
                if pending_combine.is_none() {
                    bar(&enc);
                }
            }

            match pending_combine.take() {
                Some((slots, sig_last)) => combine_resnorm(&enc, &out_norm, slots, sig_last),
                None => resnorm(&enc, &out_norm),
            }
            bar(&enc);
            matvec(
                &enc,
                &out_state,
                &out_mat,
                hidden,
                vocab,
                h_at,
                out_logits_at,
            );
            bar(&enc);

            enc.set_compute_pipeline_state(&argmax_state);
            enc.set_buffer(0, Some(y), e(out_logits_at));
            let v32 = vocab as u32;
            enc.set_bytes(1, 4, &v32 as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(amax_at));
            enc.dispatch_thread_groups(MTLSize::new(64, 1, 1), MTLSize::new(256, 1, 1));
            bar(&enc);

            enc.set_compute_pipeline_state(&argmax_final_state);
            enc.set_buffer(0, Some(y), e(amax_at));
            let np = 64u32;
            enc.set_bytes(1, 4, &np as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(amax_at + 64 * 2));
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            bar(&enc);

            enc.set_compute_pipeline_state(&store_state);
            enc.set_buffer(0, Some(&hist_buf), 0);
            enc.set_buffer(1, Some(y), e(amax_at + 64 * 2));
            let dst_idx = (step + 1) as u32;
            enc.set_bytes(2, 4, &dst_idx as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            enc.memory_barrier_with_resources(&[y, &hist_buf]);
        }

        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_wait = std::time::Instant::now();
        let cmd = cmd.commit_and_wait();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add((n_more * (layers.len() * 8 + 4)) as u64, Ordering::Relaxed);
        DECODE_ATTEMPTS.fetch_add(n_more as u64, Ordering::Relaxed);
        DECODE_SUCCESSES.fetch_add(n_more as u64, Ordering::Relaxed);

        let mut results = vec![0u32; n_more];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (hist_buf.contents() as *const u32).add(1),
                results.as_mut_ptr(),
                n_more,
            );
        }
        Some(results)
    })
}
