//! Batched verification pass: scores a run of draft tokens in one command
//! buffer for speculative decoding.

use super::*;

/// The speculative multi-token verify: `m` tokens through the already
/// resolved layers in ONE command buffer with ONE wait. Projections batch
/// all m rows (TILE = m matvec pipelines, one weight pass); attention and
/// MoE run per token; the GDN recurrence takes the batch kernels, with
/// rollback slots written when armed. The matvec f32 dot per output row is
/// the decode path's numerics, so the arbitration rows stay bit-compatible
/// with plain greedy decode - unlike the mm (half-staged) prefill path,
/// whose drift flips near-tie argmaxes (measured: stream divergence).
pub(super) fn encode_verify_tokens(
    req: &TokenReq,
    gpu: &mut Gpu,
    layers: &[LayerRefs],
    _sigmoid_gating: bool,
) -> Option<TokenOut> {
    let m = req.m;
    let hidden = req.x.len() / m;
    let hd = req.head_dim;
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let vocab = req.output.2;
    let rp = req.rot_dim / 2;

    let out_norm = resolve_norm(gpu, req.output_norm, hidden)?;
    let out_mat = resolve(gpu, req.output.0, req.output.1)?;

    let resnorm_state = gpu.pipelines[&("residual_norm", 1, 1)].to_owned();
    let norm_rows_state = gpu.pipelines[&("norm_rows", 1, 1)].to_owned();
    let combine_rows_state = gpu.pipelines[&("moe_combine_rows", 1, 1)].to_owned();
    let resnorm_router_rows_state = gpu.pipelines[&("resnorm_router_rows", 1, 1)].to_owned();
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();
    let attend_state = gpu.pipelines[&(attend_kernel(), 1, 1)].to_owned();
    let qk_prep_state = gpu.pipelines[&("qk_prep", 1, 1)].to_owned();
    let qk_prep256_state = gpu.pipelines[&("qk_prep256", 1, 1)].to_owned();
    let attend256_state = gpu.pipelines[&("attend_s32_256", 1, 1)].to_owned();
    let qk_prep_batch256_state = gpu.pipelines[&("qk_prep_batch256", 1, 1)].to_owned();
    let attend_rows256_state = gpu.pipelines[&("attend_rows256", 1, 1)].to_owned();
    let gate_mul_state = gpu.pipelines[&("attn_gate_mul", 1, 1)].to_owned();
    let conv_state = gpu.pipelines[&("gdn_conv_batch", 1, 1)].to_owned();
    let step_state = gpu.pipelines[&("gdn_step_batch", 1, 1)].to_owned();
    let norm_out_state = gpu.pipelines[&("gdn_out_norm_batch", 1, 1)].to_owned();
    let argmax_state = gpu.pipelines[&("argmax_f32", 1, 1)].to_owned();
    let argmax_final_state = gpu.pipelines[&("argmax_final", 1, 1)].to_owned();

    // TILE=m pipelines for the batched projections (created up front: the
    // registry needs &mut Gpu).
    let tstate = |gpu: &mut Gpu, mat: &MatRef, n_in: usize| -> Option<ComputePipelineState> {
        let lpr = lanes_per_row(mat.ty, n_in);
        Some(gpu.pipeline(mat.kernel, m, lpr)?.to_owned())
    };
    let mut t_attn: Vec<Option<[ComputePipelineState; 4]>> = Vec::with_capacity(layers.len());
    let mut t_gdn: Vec<Option<[ComputePipelineState; 5]>> = Vec::with_capacity(layers.len());
    // TILE=m pipelines for the shared expert / its gate logit (MoE layers).
    let mut t_sgate: Vec<Option<ComputePipelineState>> = Vec::with_capacity(layers.len());
    let mut t_shared: Vec<Option<[ComputePipelineState; 3]>> = Vec::with_capacity(layers.len());
    for l in layers {
        match &l.ffn {
            FfnRefs::Moe {
                shared,
                shared_gate,
                sw_fused,
                expert_ffn,
                ..
            } => {
                t_sgate.push(match shared_gate {
                    Some((m0, _)) => Some(tstate(gpu, m0, hidden)?),
                    None => None,
                });
                t_shared.push(match shared {
                    Some((smats, _)) => Some([
                        tstate(gpu, &smats[0], hidden)?,
                        tstate(gpu, &smats[1], hidden)?,
                        {
                            let lpr = lanes_per_row(smats[2].ty, *expert_ffn);
                            gpu.pipeline_full(smats[2].kernel, m, lpr, false, *sw_fused)?
                                .to_owned()
                        },
                    ]),
                    None => None,
                });
            }
            _ => {
                t_sgate.push(None);
                t_shared.push(None);
            }
        }
    }
    // ROWS expert pipelines: one dispatch per matrix covering every row's
    // expert slots. Only the kernels whose source carries the ROWS mapping
    // qualify (the per-output math there is the single-row kernel's
    // verbatim); anything else falls back to the per-row dispatches below.
    let mut t_moe_rows: Vec<Option<[ComputePipelineState; 3]>> = Vec::with_capacity(layers.len());
    for l in layers {
        t_moe_rows.push(match &l.ffn {
            FfnRefs::Moe {
                mats,
                expert_ffn,
                sw_fused,
                ..
            } => {
                let n_outs = [*expert_ffn, *expert_ffn, hidden];
                let mut v: Vec<ComputePipelineState> = Vec::with_capacity(3);
                for (i, (mat, n_in)) in mats.iter().zip([hidden, hidden, *expert_ffn]).enumerate() {
                    // The same _mv fallback decode applies per matrix.
                    let kernel = match mat.kernel {
                        "matvec_q2_k_mv" if n_outs[i] % 4 != 0 => "matvec_q2_k",
                        "matvec_q3_k_mv" if n_outs[i] % 2 != 0 => "matvec_q3_k",
                        "matvec_q4_k_mv" if n_outs[i] % 2 != 0 => "matvec_q4_k",
                        "matvec_q5_k_mv" if n_outs[i] % 2 != 0 => "matvec_q5_k",
                        "matvec_q6_k_mv" if n_outs[i] % 2 != 0 => "matvec_q6_k",
                        "matvec_q8_0_mv" if n_outs[i] % q8_nr0() != 0 => "matvec_q8_0",
                        k => k,
                    };
                    if !matches!(
                        kernel,
                        "matvec_q2_k_mv"
                            | "matvec_q3_k_mv"
                            | "matvec_q4_k_mv"
                            | "matvec_q5_k"
                            | "matvec_q5_k_mv"
                            | "matvec_q6_k"
                            | "matvec_q6_k_mv"
                            | "matvec_q8_0"
                            | "matvec_q8_0_mv"
                    ) {
                        v.clear();
                        break;
                    }
                    let lpr = lanes_per_row(mat.ty, n_in);
                    match gpu.pipeline_rows(kernel, lpr, i == 2 && *sw_fused) {
                        Some(p) => v.push(p.to_owned()),
                        None => {
                            v.clear();
                            break;
                        }
                    }
                }
                if v.len() == 3 {
                    Some([v[0].clone(), v[1].clone(), v[2].clone()])
                } else {
                    None
                }
            }
            _ => None,
        });
    }
    for l in layers {
        t_attn.push(match &l.mats {
            Some(mats) => Some([
                tstate(gpu, &mats[0], hidden)?,
                tstate(gpu, &mats[1], hidden)?,
                tstate(gpu, &mats[2], hidden)?,
                tstate(gpu, &mats[3], q_dim)?,
            ]),
            None => None,
        });
        t_gdn.push(match &l.gdn {
            Some(g) => Some([
                tstate(gpu, &g.mats[0], hidden)?,
                tstate(gpu, &g.mats[1], hidden)?,
                tstate(gpu, &g.mats[2], hidden)?,
                tstate(gpu, &g.mats[3], hidden)?,
                tstate(gpu, &g.mats[4], g.value_dim)?,
            ]),
            None => None,
        });
    }
    let t_out = tstate(gpu, &out_mat, hidden)?;

    // Arena layout, element offsets over f32 (ids are u32-in-f32 slots).
    let align = |n: usize| (n + 63) & !63;
    let wq_max = req
        .layers
        .iter()
        .filter(|l| l.gdn.is_none())
        .map(|l| l.wq.2)
        .max()
        .unwrap_or(q_dim)
        .max(q_dim);
    let mut max_dense_ffn = 1usize;
    let mut max_moe_ffn = 1usize;
    let mut max_slots = 1usize;
    let mut max_expert = 1usize;
    let (mut max_channels, mut max_value_dim, mut max_heads_v) = (0usize, 0usize, 1usize);
    for l in layers {
        if let Some(g) = &l.gdn {
            max_channels = max_channels.max(g.channels);
            max_value_dim = max_value_dim.max(g.value_dim);
            max_heads_v = max_heads_v.max(g.heads_v as usize);
        }
        if let FfnRefs::Dense { ffn_dim, .. } = &l.ffn {
            max_dense_ffn = max_dense_ffn.max(*ffn_dim);
        }
        if let FfnRefs::Moe {
            expert_ffn,
            n_used,
            shared,
            n_expert,
            ..
        } = &l.ffn
        {
            max_moe_ffn = max_moe_ffn.max(*expert_ffn);
            max_slots = max_slots.max(*n_used + shared.is_some() as usize);
            max_expert = max_expert.max(*n_expert);
        }
    }
    let max_ffn = max_moe_ffn;
    // The batched MoE below packs the per-row expert scratch contiguously
    // and the row-batched kernels index it by the global maxima, so every
    // MoE layer must share one expert shape; anything else declines (the
    // caller falls back to the batch path). The router kernels are
    // single-threadgroup per row: n_expert <= 256.
    for l in layers {
        if let FfnRefs::Moe {
            expert_ffn,
            n_used,
            shared,
            n_expert,
            ..
        } = &l.ffn
        {
            if *n_used + shared.is_some() as usize != max_slots
                || *expert_ffn != max_ffn
                || *n_expert > 256
            {
                return None;
            }
        }
    }
    let gate_span = (m * max_slots * max_ffn).max(max_dense_ffn);
    let x_at = 0usize;
    let delta_at = x_at + align(m * hidden);
    let h_at = delta_at + align(m * hidden);
    let q_at = h_at + align(m * hidden);
    let qn_at = q_at + align(m * wq_max);
    let qgate_at = qn_at + align(m * q_dim);
    let k_at = qgate_at + align(m * q_dim);
    let v_at = k_at + align(m * kv);
    let attn_at = v_at + align(m * kv);
    let gqkv_at = attn_at + align(m * q_dim);
    let gqkc_at = gqkv_at + align(m * max_channels);
    let gz_at = gqkc_at + align(m * max_channels);
    let gab_at = gz_at + align(m * max_value_dim);
    let gy_at = gab_at + align(2 * m * max_heads_v);
    let logits_at = gy_at + align(m * max_value_dim);
    let ids_at = logits_at + align(m * max_expert);
    let wts_at = ids_at + align(m * max_slots);
    // The expert FFN scratch is per-row: every token's gate/up/down blocks
    // live side by side, so the whole MoE half dispatches stage-by-stage
    // (one barrier per stage) instead of token-by-token.
    let gate_at = wts_at + align(m * max_slots);
    let up_at = gate_at + align(gate_span);
    let downo_at = up_at + align(gate_span);
    // The shared expert runs as plain TILE=m matvecs over ALL rows (one
    // weight pass), so it gets its own contiguous regions instead of a slot
    // in the per-row blocks.
    let sgate_at = downo_at + align(m * max_slots * hidden);
    let sg_gate_at = sgate_at + align(m);
    let sg_up_at = sg_gate_at + align(m * max_ffn);
    let sg_down_at = sg_up_at + align(m * max_ffn);
    let out_logits_at = sg_down_at + align(m * hidden);
    let flag_at = out_logits_at + align(m * vocab);
    let ctr_at = flag_at + align(1);
    let amax_at = ctr_at + align(1);
    let total = amax_at + align(m * (64 * 2 + 1));
    gpu.ensure_arenas(4096, total * 4);

    unsafe {
        let yp = gpu.y_arena.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(req.x.as_ptr(), yp.add(x_at), req.x.len());
        std::ptr::write_bytes(yp.add(delta_at), 0, m * hidden);
        std::ptr::write_bytes(yp.add(flag_at), 0, 1);
        std::ptr::write_bytes(yp.add(ctr_at), 0, 1);
        for l in layers {
            if let FfnRefs::Moe {
                n_used,
                shared: Some(_),
                shared_gate: None,
                ..
            } = &l.ffn
            {
                for i in 0..m {
                    *yp.add(wts_at + i * max_slots + *n_used) = 1.0;
                }
            }
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let mut cmd = gpu.queue.new_command_buffer();
        let mut enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let y = &gpu.y_arena;
        let bar = |enc: &metal::ComputeCommandEncoderRef| enc.memory_barrier_with_resources(&[y]);
        let e = |off: usize| (off * 4) as u64;

        // One plain TILE=m matvec over all m rows: one weight pass.
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
        // Single-row matvec for the per-token expert/shared projections.
        let resnorm = |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef, i: usize| {
            enc.set_compute_pipeline_state(&resnorm_state);
            enc.set_buffer(0, Some(y), e(x_at + i * hidden));
            enc.set_buffer(1, Some(y), e(delta_at + i * hidden));
            enc.set_buffer(2, Some(y), e(h_at + i * hidden));
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        };
        let matvec_row = |enc: &metal::ComputeCommandEncoderRef,
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

        let mut dispatched = 0u64;
        let vdbg = std::env::var_os("ALLPAKA_VERIFY_DEBUG").is_some();
        for (li, (l, refs)) in req.layers.iter().zip(layers).enumerate() {
            if vdbg {
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let yp = gpu.y_arena.contents() as *const f32;
                unsafe {
                    let x0 = std::slice::from_raw_parts(yp.add(x_at), hidden);
                    let d0 = std::slice::from_raw_parts(yp.add(delta_at), hidden);
                    let sx: f64 = x0.iter().map(|&v| v as f64).sum();
                    let sd: f64 = d0.iter().map(|&v| v as f64).sum();
                    eprintln!("verify layer {li} in: sum_x={sx:.6} sum_delta={sd:.6}");
                    if li == 1 {
                        let wts = std::slice::from_raw_parts(yp.add(wts_at), 16);
                        eprintln!("  wts[..]: {:?}", &wts[..9]);
                        let do0 = std::slice::from_raw_parts(yp.add(downo_at), 9 * hidden);
                        for srow in 0..9 {
                            let r = &do0[srow * hidden..(srow + 1) * hidden];
                            let s: f64 = r.iter().map(|&v| v as f64).sum();
                            eprintln!("  downo[{srow}] sum={s:.6} head={:.6}", r[0]);
                        }
                        let h0 = std::slice::from_raw_parts(yp.add(h_at), hidden);
                        let hs: f64 = h0.iter().map(|&v| v as f64).sum();
                        eprintln!("  h_ffn_in sum={hs:.6}");
                        std::fs::write(
                            "/tmp/h0.bin",
                            std::slice::from_raw_parts(h0.as_ptr() as *const u8, hidden * 4),
                        )
                        .ok();
                        eprintln!(
                            "  h_ffn_in head=[{:.6} {:.6} {:.6}] tail=[{:.6} {:.6} {:.6}]",
                            h0[0], h0[1], h0[2], h0[2045], h0[2046], h0[2047]
                        );
                        let lg = std::slice::from_raw_parts(yp.add(logits_at), 256);
                        let ls: f64 = lg.iter().map(|&v| v as f64).sum();
                        eprintln!("  router logits sum={ls:.6}");
                        eprintln!(
                            "  router logits head=[{:.6} {:.6} {:.6} {:.6}]",
                            lg[0], lg[1], lg[2], lg[3]
                        );
                    }
                }
                cmd = gpu.queue.new_command_buffer();
                enc = cmd
                    .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
            }
            // h = rmsnorm(x + delta) * attn_norm over ALL rows in ONE
            // dispatch (norm_rows is residual_norm's body with the row from
            // the grid, bit-identical per row).
            enc.set_compute_pipeline_state(&norm_rows_state);
            enc.set_buffer(0, Some(y), e(x_at));
            enc.set_buffer(1, Some(y), e(delta_at));
            enc.set_buffer(2, Some(y), e(h_at));
            enc.set_buffer(
                3,
                Some(&gpu.chunks[refs.attn_norm.chunk].buf),
                refs.attn_norm.off,
            );
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            let one = 1u32;
            enc.set_bytes(6, 4, &one as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
            dispatched += 1;
            bar(enc);

            if let Some(g) = &refs.gdn {
                // Debug kill switch, mirrors the CPU path's gdn_forward:
                // zero the branch output (into delta) to isolate bugs.
                if std::env::var_os("ALLPAKA_GDN_ZERO").is_some() {
                    unsafe {
                        std::ptr::write_bytes(
                            (gpu.y_arena.contents() as *mut f32).add(delta_at),
                            0,
                            m * hidden,
                        );
                    }
                } else {
                    let tg = &l.gdn.as_ref().expect("gdn refs without gdn layer");
                    let ts = t_gdn[li].as_ref().expect("gdn tile states");
                    let ssm_buf = &req.ssm.expect("gdn layer without ssm region").buf;
                    // The four projections over all m rows, two weight passes.
                    matvec(enc, &ts[0], &g.mats[0], hidden, g.channels, h_at, gqkv_at);
                    matvec(enc, &ts[1], &g.mats[1], hidden, g.value_dim, h_at, gz_at);
                    matvec(
                        enc,
                        &ts[2],
                        &g.mats[2],
                        hidden,
                        g.heads_v as usize,
                        h_at,
                        gab_at,
                    );
                    matvec(
                        enc,
                        &ts[3],
                        &g.mats[3],
                        hidden,
                        g.heads_v as usize,
                        h_at,
                        gab_at + m * g.heads_v as usize,
                    );
                    bar(enc);
                    // Depthwise conv over the chunk (with rollback slots), then
                    // the session window commit.
                    enc.set_compute_pipeline_state(&conv_state);
                    enc.set_buffer(0, Some(y), e(gqkv_at));
                    enc.set_buffer(1, Some(y), e(gqkc_at));
                    enc.set_buffer(2, Some(ssm_buf), 0);
                    enc.set_buffer(3, Some(&gpu.chunks[g.conv1d.chunk].buf), g.conv1d.off);
                    let cargs = GdnConvBatchArgs {
                        channels: g.channels as u32,
                        d_conv: g.d_conv,
                        m: m as u32,
                        pad0: 0,
                        conv_off: g.conv_off,
                    };
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<GdnConvBatchArgs>() as u64,
                        &cargs as *const GdnConvBatchArgs as *const _,
                    );
                    let (slots_buf, slot_total) = match req.ssm_slots {
                        Some((r, total)) => (r, total as u32),
                        None => (req.ssm.expect("ssm region"), 0u32),
                    };
                    enc.set_buffer(5, Some(&slots_buf.buf), 0);
                    enc.set_bytes(6, 4, &slot_total as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((g.channels as u64).div_ceil(256), m as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // The last d_conv-1 raw rows become the next chunk's window.
                    enc.set_compute_pipeline_state(&gpu.pipelines[&("copy_f32", 1, 1)]);
                    let window_rows = g.d_conv as usize - 1;
                    let window_src = if m >= window_rows {
                        gqkv_at + (m - window_rows) * g.channels
                    } else {
                        let old = (window_rows - m) * g.channels;
                        enc.set_buffer(
                            0,
                            Some(ssm_buf),
                            (g.conv_off + m as u64 * g.channels as u64) * 4,
                        );
                        enc.set_buffer(1, Some(y), e(gate_at));
                        let old32 = old as u32;
                        enc.set_bytes(2, 4, &old32 as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((old as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        enc.set_buffer(0, Some(y), e(gqkv_at));
                        enc.set_buffer(1, Some(y), e(gate_at + old));
                        let new32 = (m * g.channels) as u32;
                        enc.set_bytes(2, 4, &new32 as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((new32 as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        enc.memory_barrier_with_resources(&[y, ssm_buf]);
                        gate_at
                    };
                    enc.set_buffer(0, Some(y), e(window_src));
                    enc.set_buffer(1, Some(ssm_buf), g.conv_off * 4);
                    let nwin = (window_rows * g.channels) as u32;
                    enc.set_bytes(2, 4, &nwin as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((nwin as u64).div_ceil(256), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // The recurrence over the chunk, slots per row when armed.
                    enc.set_compute_pipeline_state(&step_state);
                    enc.set_buffer(0, Some(ssm_buf), 0);
                    enc.set_buffer(1, Some(y), e(gqkc_at));
                    enc.set_buffer(2, Some(y), e(gab_at));
                    enc.set_buffer(3, Some(y), e(gab_at + m * g.heads_v as usize));
                    let sargs = GdnStepBatchArgs {
                        heads_k: g.heads_k,
                        heads_v: g.heads_v,
                        d: g.d,
                        key_dim: g.heads_k * g.d,
                        m: m as u32,
                        eps: req.eps,
                        state_off: g.state_off,
                    };
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<GdnStepBatchArgs>() as u64,
                        &sargs as *const GdnStepBatchArgs as *const _,
                    );
                    enc.set_bytes(5, (tg.a.len() * 4) as u64, tg.a.as_ptr() as *const _);
                    enc.set_bytes(6, (tg.dt.len() * 4) as u64, tg.dt.as_ptr() as *const _);
                    enc.set_buffer(7, Some(y), e(gy_at));
                    enc.set_buffer(8, Some(&slots_buf.buf), 0);
                    enc.set_bytes(9, 4, &slot_total as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((g.d / 4) as u64, g.heads_v as u64, 1),
                        MTLSize::new(32, 4, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // Gated rmsnorm over the head outputs, times silu(z).
                    enc.set_compute_pipeline_state(&norm_out_state);
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
                        MTLSize::new(g.heads_v as u64, m as u64, 1),
                        MTLSize::new(g.d as u64, 1, 1),
                    );
                    bar(enc);
                    matvec(
                        enc,
                        &ts[4],
                        &g.mats[4],
                        g.value_dim,
                        hidden,
                        gy_at,
                        delta_at,
                    );
                    bar(enc);
                    dispatched += 11;
                }
            } else {
                let amats = refs.mats.as_ref().expect("attention layer without mats");
                let ts = t_attn[li].as_ref().expect("attention tile states");
                let wq_out = if l.gate_in_q { 2 * q_dim } else { q_dim };
                matvec(enc, &ts[0], &amats[0], hidden, wq_out, h_at, q_at);
                matvec(enc, &ts[1], &amats[1], hidden, kv, h_at, k_at);
                matvec(enc, &ts[2], &amats[2], hidden, kv, h_at, v_at);
                bar(enc);
                if hd == 256 && wq_max == wq_out {
                    // Batched qk_prep: one threadgroup per (head, row); the
                    // per-row math is qk_prep256's verbatim (prefill's
                    // kernel), so every row is bit-identical to decode.
                    enc.set_compute_pipeline_state(&qk_prep_batch256_state);
                    enc.set_buffer(0, Some(y), e(q_at));
                    enc.set_buffer(1, Some(y), e(k_at));
                    enc.set_buffer(2, Some(y), e(v_at));
                    enc.set_buffer(3, Some(&req.cache.buf), 0);
                    let args = QkPrepBatch256Args {
                        n_heads: req.n_heads as u32,
                        n_kv_heads: req.n_kv_heads as u32,
                        kv_dim: req.kv_dim as u32,
                        eps: req.eps,
                        base: req.pos as u32,
                        has_qk_norm: l.q_norm.is_some() as u32,
                        rot_dim: req.rot_dim as u32,
                        gate_in_q: l.gate_in_q as u32,
                        k_base: l.k_off as u64,
                        v_base: l.v_off as u64,
                        rope_norm: req.rope_norm as u32,
                        pad0: 0,
                    };
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<QkPrepBatch256Args>() as u64,
                        &args as *const QkPrepBatch256Args as *const _,
                    );
                    let ones = [1.0f32; 256];
                    let qw = l.q_norm.unwrap_or(&ones);
                    let kw = l.k_norm.unwrap_or(&ones);
                    enc.set_bytes(5, (256 * 4) as u64, qw.as_ptr() as *const _);
                    enc.set_bytes(6, (256 * 4) as u64, kw.as_ptr() as *const _);
                    enc.set_bytes(7, (rp * 8) as u64, req.rope.as_ptr() as *const _);
                    enc.set_buffer(8, Some(y), e(qn_at));
                    enc.set_buffer(9, Some(y), e(qgate_at));
                    enc.dispatch_thread_groups(
                        MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, m as u64, 1),
                        MTLSize::new(32, 1, 1),
                    );
                } else {
                    for i in 0..m {
                        let pos = req.pos + i;
                        if hd == 256 {
                            enc.set_compute_pipeline_state(&qk_prep256_state);
                            enc.set_buffer(0, Some(y), e(q_at + i * wq_max));
                            enc.set_buffer(1, Some(y), e(k_at + i * kv));
                            enc.set_buffer(2, Some(y), e(v_at + i * kv));
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
                                rope_norm: req.rope_norm as u32,
                                pad0: 0,
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
                                (rp * 8) as u64,
                                req.rope[i * rp..].as_ptr() as *const _,
                            );
                            enc.set_buffer(8, Some(y), e(qn_at + i * q_dim));
                            enc.set_buffer(9, Some(y), e(qgate_at + i * q_dim));
                            enc.dispatch_thread_groups(
                                MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                                MTLSize::new(32, 1, 1),
                            );
                        } else {
                            enc.set_compute_pipeline_state(&qk_prep_state);
                            enc.set_buffer(0, Some(y), e(q_at + i * wq_max));
                            enc.set_buffer(1, Some(y), e(k_at + i * kv));
                            enc.set_buffer(2, Some(y), e(v_at + i * kv));
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
                                rope_norm: req.rope_norm as u32,
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
                                (rp * 8) as u64,
                                req.rope[i * rp..].as_ptr() as *const _,
                            );
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
                        }
                    }
                }
                enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                if hd == 256 && wq_max == wq_out {
                    // Batched attend: one threadgroup per (head, row), row r
                    // covers base + r + 1 positions; attend_rows256 is
                    // attend_s32_256's body with the row from grid.y.
                    enc.set_compute_pipeline_state(&attend_rows256_state);
                    enc.set_buffer(0, Some(&req.cache.buf), 0);
                    enc.set_buffer(1, Some(&req.cache.buf), 0);
                    enc.set_buffer(2, Some(y), e(qn_at));
                    enc.set_buffer(3, Some(y), e(attn_at));
                    for (index, value) in [
                        (4u64, req.kv_dim as u32),
                        (5, hd as u32),
                        (6, req.pos as u32),
                        (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
                    ] {
                        enc.set_bytes(index, 4, &value as *const u32 as *const _);
                    }
                    enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                    for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                        enc.set_bytes(index, 8, &value as *const u64 as *const _);
                    }
                    let heads32 = req.n_heads as u32;
                    enc.set_bytes(11, 4, &heads32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(req.n_heads as u64, m as u64, 1),
                        MTLSize::new(512, 1, 1),
                    );
                } else {
                    for i in 0..m {
                        let n_pos = (req.pos + i + 1) as u32;
                        enc.set_buffer(0, Some(&req.cache.buf), 0);
                        enc.set_buffer(1, Some(&req.cache.buf), 0);
                        let q_off = if hd == 256 {
                            qn_at + i * q_dim
                        } else {
                            q_at + i * wq_max
                        };
                        enc.set_buffer(2, Some(y), e(q_off));
                        enc.set_buffer(3, Some(y), e(attn_at + i * q_dim));
                        for (index, value) in [
                            (4u64, req.kv_dim as u32),
                            (5, hd as u32),
                            (6, n_pos),
                            (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
                        ] {
                            enc.set_bytes(index, 4, &value as *const u32 as *const _);
                        }
                        enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                        for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                            enc.set_bytes(index, 8, &value as *const u64 as *const _);
                        }
                        if hd == 256 {
                            enc.set_compute_pipeline_state(&attend256_state);
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(512, 1, 1),
                            );
                        } else {
                            enc.set_compute_pipeline_state(&attend_state);
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(attend_tg(), 1, 1),
                            );
                        }
                    }
                }
                bar(enc);
                if l.gate_in_q && hd == 256 {
                    enc.set_compute_pipeline_state(&gate_mul_state);
                    enc.set_buffer(0, Some(y), e(attn_at));
                    enc.set_buffer(1, Some(y), e(qgate_at));
                    let n32 = (m * q_dim) as u32;
                    enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    bar(enc);
                }
                // Debug kill switch, mirrors the CPU path: zero the
                // attention branch output (into delta).
                if std::env::var_os("ALLPAKA_ATTN_ZERO").is_some() {
                    unsafe {
                        std::ptr::write_bytes(
                            (gpu.y_arena.contents() as *mut f32).add(delta_at),
                            0,
                            m * hidden,
                        );
                    }
                } else {
                    matvec(enc, &ts[3], &amats[3], q_dim, hidden, attn_at, delta_at);
                }
                bar(enc);
                dispatched += 6 + 2 * m as u64;
            }

            // The FFN half: resnorm+router+top-k fused per token
            // (resnorm_router - decode_token's default moe_plain path, so
            // the router logits are bit-identical to plain decode; a
            // matvec_f32+topk pair rounds differently and flips near-tie
            // expert picks). The barrier is load-bearing. Dense layers take
            // the plain resnorm first (resnorm_router is MoE-only).
            if matches!(&refs.ffn, FfnRefs::Dense { .. }) {
                for i in 0..m {
                    resnorm(enc, &refs.ffn_norm, i);
                }
                bar(enc);
            }
            // Debug kill switch, mirrors the CPU path's MOE_ZERO: zero the
            // FFN output (into delta), skipping the whole stage.
            if std::env::var_os("ALLPAKA_MOE_ZERO").is_some() {
                unsafe {
                    std::ptr::write_bytes(
                        (gpu.y_arena.contents() as *mut f32).add(delta_at),
                        0,
                        m * hidden,
                    );
                }
                continue;
            }
            match &refs.ffn {
                FfnRefs::Dense {
                    mats,
                    states,
                    ffn_dim,
                    sw_fused,
                } => {
                    for i in 0..m {
                        matvec_row(
                            enc,
                            &states[0],
                            &mats[0],
                            hidden,
                            *ffn_dim,
                            h_at + i * hidden,
                            gate_at,
                        );
                        matvec_row(
                            enc,
                            &states[1],
                            &mats[1],
                            hidden,
                            *ffn_dim,
                            h_at + i * hidden,
                            up_at,
                        );
                        bar(enc);
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
                            bar(enc);
                            matvec_row(
                                enc,
                                &states[2],
                                &mats[2],
                                *ffn_dim,
                                hidden,
                                gate_at,
                                delta_at + i * hidden,
                            );
                            dispatched += 4;
                        } else {
                            enc.set_buffer(8, Some(y), e(up_at));
                            matvec_row(
                                enc,
                                &states[2],
                                &mats[2],
                                *ffn_dim,
                                hidden,
                                gate_at,
                                delta_at + i * hidden,
                            );
                            dispatched += 3;
                        }
                        bar(enc);
                    }
                }
                FfnRefs::Moe {
                    router,
                    router_state: _,
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
                    gu_dual: _,
                    down_tail: _,
                } => {
                    let n_slots = *n_used + shared.is_some() as usize;
                    // Stage 1: FFN resnorm + router matvec + gating top-k over
                    // ALL rows in ONE dispatch (resnorm_router_rows is
                    // resnorm_router's body with the row from grid.y, so every
                    // row is bit-identical to decode's fused router).
                    enc.set_compute_pipeline_state(&resnorm_router_rows_state);
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
                        Some(rb) => enc.set_buffer(12, Some(&gpu.chunks[rb.chunk].buf), rb.off),
                        None => enc.set_buffer(12, Some(&gpu.chunks[router.chunk].buf), 0),
                    }
                    let sg = *sigmoid as u32;
                    let hb = router_bias.is_some() as u32;
                    enc.set_bytes(13, 4, &sg as *const u32 as *const _);
                    enc.set_bytes(14, 4, &hb as *const u32 as *const _);
                    enc.set_bytes(15, 4, &(max_slots as u32) as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, m as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    bar(enc);
                    // Stage 2: gate/up. Every row is independent (per-row
                    // scratch blocks), so the rows dispatch back to back with
                    // ONE barrier after the whole stage. The shared expert
                    // and its gate logit run as TILE matvecs over ALL rows
                    // (one weight pass each).
                    if let (Some((sg_mat, _)), Some(tsg)) = (shared_gate, t_sgate[li].as_ref()) {
                        matvec(enc, tsg, sg_mat, hidden, 1, h_at, sgate_at);
                    }
                    if let (Some((smats, _)), Some(tsh)) = (shared, t_shared[li].as_ref()) {
                        matvec(
                            enc,
                            &tsh[0],
                            &smats[0],
                            hidden,
                            *expert_ffn,
                            h_at,
                            sg_gate_at,
                        );
                        matvec(enc, &tsh[1], &smats[1], hidden, *expert_ffn, h_at, sg_up_at);
                    }
                    let a = hidden as u32;
                    let b = *expert_ffn as u32;
                    if std::env::var_os("ALLPAKA_ROWS_DEBUG").is_some() && li == 0 {
                        eprintln!("verify rows pipelines: {}", t_moe_rows[li].is_some());
                    }
                    if let Some(rs) = &t_moe_rows[li] {
                        // ROWS: ONE dispatch per matrix covering every row's
                        // expert slots (per-output math identical to the
                        // per-row indexed dispatches).
                        for (mi, y_off) in [gate_at, up_at].into_iter().enumerate() {
                            enc.set_compute_pipeline_state(&rs[mi]);
                            enc.set_buffer(0, Some(&gpu.chunks[mats[mi].chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(h_at));
                            enc.set_buffer(2, Some(y), e(y_off));
                            enc.set_bytes(3, 4, &a as *const u32 as *const _);
                            enc.set_bytes(4, 4, &b as *const u32 as *const _);
                            enc.set_bytes(5, 8, &mats[mi].w_off as *const u64 as *const _);
                            enc.set_buffer(6, Some(y), e(ids_at));
                            let idx = GpuIdxArgs {
                                stride: strides[mi],
                                slots: *n_used as u32,
                                x_stride: 0,
                                ids_stride: max_slots as u32,
                                x_row_stride: hidden as u32,
                                y_row_stride: (max_slots * max_ffn) as u32,
                                n_rows: m as u32,
                            };
                            enc.set_bytes(
                                7,
                                std::mem::size_of::<GpuIdxArgs>() as u64,
                                &idx as *const GpuIdxArgs as *const _,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(
                                    ((*expert_ffn * *n_used * m) as u64
                                        * lanes_per_row(mats[mi].ty, hidden) as u64)
                                        .div_ceil(128),
                                    1,
                                    1,
                                ),
                                MTLSize::new(128, 1, 1),
                            );
                        }
                    } else {
                        for i in 0..m {
                            let ids_off = ids_at + i * max_slots;
                            let gate_row = gate_at + i * max_slots * max_ffn;
                            let up_row = up_at + i * max_slots * max_ffn;
                            let x_off = h_at + i * hidden;
                            enc.set_compute_pipeline_state(&states[0]);
                            enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(x_off));
                            enc.set_buffer(2, Some(y), e(gate_row));
                            enc.set_bytes(3, 4, &a as *const u32 as *const _);
                            enc.set_bytes(4, 4, &b as *const u32 as *const _);
                            enc.set_bytes(5, 8, &mats[0].w_off as *const u64 as *const _);
                            enc.set_buffer(6, Some(y), e(ids_off));
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
                            dispatch_mv_indexed(enc, mats[0].ty, hidden, *expert_ffn, *n_used);
                            enc.set_compute_pipeline_state(&states[1]);
                            enc.set_buffer(0, Some(&gpu.chunks[mats[1].chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(x_off));
                            enc.set_buffer(2, Some(y), e(up_row));
                            enc.set_bytes(3, 4, &a as *const u32 as *const _);
                            enc.set_bytes(4, 4, &b as *const u32 as *const _);
                            enc.set_bytes(5, 8, &mats[1].w_off as *const u64 as *const _);
                            enc.set_buffer(6, Some(y), e(ids_off));
                            let idx = GpuIdxArgs {
                                stride: strides[1],
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
                            dispatch_mv_indexed(enc, mats[1].ty, hidden, *expert_ffn, *n_used);
                        }
                    }
                    bar(enc);
                    if !*sw_fused {
                        // One swiglu over ALL expert rows (contiguous uniform
                        // blocks), and one over the shared expert's rows.
                        enc.set_compute_pipeline_state(&swiglu_state);
                        enc.set_buffer(0, Some(y), e(gate_at));
                        enc.set_buffer(1, Some(y), e(up_at));
                        let n32 = (m * n_slots * *expert_ffn) as u32;
                        enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        if shared.is_some() {
                            enc.set_buffer(0, Some(y), e(sg_gate_at));
                            enc.set_buffer(1, Some(y), e(sg_up_at));
                            let nsg = (m * *expert_ffn) as u32;
                            enc.set_bytes(2, 4, &nsg as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((nsg as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                        bar(enc);
                    }
                    // Stage 3: down projections (the fused-swiglu variants
                    // read the up block through buffer 8).
                    let a2 = *expert_ffn as u32;
                    let b2 = hidden as u32;
                    if let Some(rs) = &t_moe_rows[li] {
                        if *sw_fused {
                            enc.set_buffer(8, Some(y), e(up_at));
                        }
                        enc.set_compute_pipeline_state(&rs[2]);
                        enc.set_buffer(0, Some(&gpu.chunks[mats[2].chunk].buf), 0);
                        enc.set_buffer(1, Some(y), e(gate_at));
                        enc.set_buffer(2, Some(y), e(downo_at));
                        enc.set_bytes(3, 4, &a2 as *const u32 as *const _);
                        enc.set_bytes(4, 4, &b2 as *const u32 as *const _);
                        enc.set_bytes(5, 8, &mats[2].w_off as *const u64 as *const _);
                        enc.set_buffer(6, Some(y), e(ids_at));
                        let idx = GpuIdxArgs {
                            stride: strides[2],
                            slots: *n_used as u32,
                            x_stride: *expert_ffn as u32,
                            ids_stride: max_slots as u32,
                            x_row_stride: (max_slots * max_ffn) as u32,
                            y_row_stride: (max_slots * hidden) as u32,
                            n_rows: m as u32,
                        };
                        enc.set_bytes(
                            7,
                            std::mem::size_of::<GpuIdxArgs>() as u64,
                            &idx as *const GpuIdxArgs as *const _,
                        );
                        enc.dispatch_thread_groups(
                            MTLSize::new(
                                ((hidden * *n_used * m) as u64
                                    * lanes_per_row(mats[2].ty, *expert_ffn) as u64)
                                    .div_ceil(128),
                                1,
                                1,
                            ),
                            MTLSize::new(128, 1, 1),
                        );
                    } else {
                        for i in 0..m {
                            let ids_off = ids_at + i * max_slots;
                            let gate_row = gate_at + i * max_slots * max_ffn;
                            let up_row = up_at + i * max_slots * max_ffn;
                            let downo_row = downo_at + i * max_slots * hidden;
                            if *sw_fused {
                                enc.set_buffer(8, Some(y), e(up_row));
                            }
                            enc.set_compute_pipeline_state(&states[2]);
                            enc.set_buffer(0, Some(&gpu.chunks[mats[2].chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(gate_row));
                            enc.set_buffer(2, Some(y), e(downo_row));
                            enc.set_bytes(3, 4, &a2 as *const u32 as *const _);
                            enc.set_bytes(4, 4, &b2 as *const u32 as *const _);
                            enc.set_bytes(5, 8, &mats[2].w_off as *const u64 as *const _);
                            enc.set_buffer(6, Some(y), e(ids_off));
                            let idx = GpuIdxArgs {
                                stride: strides[2],
                                slots: *n_used as u32,
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
                            dispatch_mv_indexed(enc, mats[2].ty, *expert_ffn, hidden, *n_used);
                        }
                    }
                    // The shared expert's down projection as one TILE matvec
                    // over all rows (the fused-swiglu variant reads the up
                    // rows through buffer 8).
                    if let (Some((smats, _)), Some(tsh)) = (shared, t_shared[li].as_ref()) {
                        if *sw_fused {
                            enc.set_buffer(8, Some(y), e(sg_up_at));
                        }
                        matvec(
                            enc,
                            &tsh[2],
                            &smats[2],
                            *expert_ffn,
                            hidden,
                            sg_gate_at,
                            sg_down_at,
                        );
                    }
                    bar(enc);
                    // Stage 4: the combine over ALL rows in ONE dispatch
                    // (moe_combine_rows; it OVERWRITES delta with the FFN
                    // output - the wo in delta is already absorbed into x,
                    // folding it again double-counts). The shared expert rides
                    // in its own regions: sig_last 0 = none, 1 = sigmoid gate,
                    // 2 = weight 1.
                    let comb_shared = if shared.is_none() {
                        0u32
                    } else if shared_gate.is_some() {
                        1
                    } else {
                        2
                    };
                    enc.set_compute_pipeline_state(&combine_rows_state);
                    enc.set_buffer(0, Some(y), e(downo_at));
                    enc.set_buffer(1, Some(y), e(wts_at));
                    enc.set_buffer(2, Some(y), e(delta_at));
                    let n = hidden as u32;
                    enc.set_bytes(3, 4, &n as *const u32 as *const _);
                    enc.set_bytes(4, 4, &(*n_used as u32) as *const u32 as *const _);
                    enc.set_bytes(5, 4, &comb_shared as *const u32 as *const _);
                    enc.set_bytes(6, 4, &(max_slots as u32) as *const u32 as *const _);
                    enc.set_buffer(7, Some(y), e(sgate_at));
                    enc.set_buffer(8, Some(y), e(sg_down_at));
                    enc.dispatch_thread_groups(
                        MTLSize::new((hidden as u64).div_ceil(256), m as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    bar(enc);
                    dispatched += 4 + 3 * m as u64;
                }
            }
        }
        // Final norm over ALL rows in one dispatch (norm_rows), then the
        // head over all rows and a per-row argmax.
        enc.set_compute_pipeline_state(&norm_rows_state);
        enc.set_buffer(0, Some(y), e(x_at));
        enc.set_buffer(1, Some(y), e(delta_at));
        enc.set_buffer(2, Some(y), e(h_at));
        enc.set_buffer(3, Some(&gpu.chunks[out_norm.chunk].buf), out_norm.off);
        let n = hidden as u32;
        enc.set_bytes(4, 4, &n as *const u32 as *const _);
        enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
        let one = 1u32;
        enc.set_bytes(6, 4, &one as *const u32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
        dispatched += 1;
        bar(enc);
        matvec(enc, &t_out, &out_mat, hidden, vocab, h_at, out_logits_at);
        bar(enc);
        for i in 0..m {
            let slot = amax_at + i * (64 * 2 + 1);
            enc.set_compute_pipeline_state(&argmax_state);
            enc.set_buffer(0, Some(y), e(out_logits_at + i * vocab));
            let v32 = vocab as u32;
            enc.set_bytes(1, 4, &v32 as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(slot));
            enc.dispatch_thread_groups(MTLSize::new(64, 1, 1), MTLSize::new(256, 1, 1));
            bar(enc);
            enc.set_compute_pipeline_state(&argmax_final_state);
            enc.set_buffer(0, Some(y), e(slot));
            let np = 64u32;
            enc.set_bytes(1, 4, &np as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(slot + 64 * 2));
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            bar(enc);
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(dispatched + 2, Ordering::Relaxed);

        let mut argmax = Vec::with_capacity(m);
        let mut hidden_rows = vec![0f32; m * hidden];
        unsafe {
            let yp = gpu.y_arena.contents() as *const f32;
            for i in 0..m {
                argmax.push((yp.add(amax_at + i * (64 * 2 + 1) + 64 * 2) as *const u32).read());
            }
            std::ptr::copy_nonoverlapping(yp.add(h_at), hidden_rows.as_mut_ptr(), m * hidden);
        }
        TokenOut::Rows {
            argmax,
            hidden: hidden_rows,
        }
    });
    Some(out)
}
