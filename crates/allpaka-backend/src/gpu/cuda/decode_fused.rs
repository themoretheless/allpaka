//! The fused decode token: every layer launched back to back on one stream,
//! optionally captured into a CUDA graph and replayed.

use super::*;

/// Shapes, arena layout and graph mode of one fused decode token, derived
/// from the request alone before the GPU is touched.
#[derive(Clone, Copy, Debug)]
struct FusedPlan {
    // Model shapes.
    hidden: usize,
    hd: usize,
    q_dim: usize,
    kv: usize,
    group: usize,
    rot_dim: usize,
    rope_pairs: usize,
    ffn_dim: usize,
    vocab: usize,
    // `y_arena` offsets (f32 elements): activations.
    x_at: usize,
    h_at: usize,
    q_at: usize,
    k_at: usize,
    v_at: usize,
    attn_at: usize,
    gate_at: usize,
    up_at: usize,
    out_logits_at: usize,
    y_elems: usize,
    // `x_arena` offsets (f32 elements): rope tables, output norm, then
    // per layer attn_norm[H], ffn_norm[H], q_norm[hd], k_norm[hd] and,
    // when any layer has attn bias, q_bias[q_dim], k_bias[kv], v_bias[kv].
    has_attn_bias: bool,
    rope_q_at: usize,
    rope_k_at: usize,
    out_norm_at: usize,
    norms_base: usize,
    per_layer_norms: usize,
    x_elems: usize,
    /// Identity of everything a captured graph bakes in; a change re-captures.
    graph_key: u64,
    // Attention / graph mode, from the runtime policy.
    use_fa: bool,
    fa_inline: bool,
    native_shallow: bool,
    hybrid_fa: bool,
    whole_graph: bool,
}

impl FusedPlan {
    /// `None` when the request is not a uniformly shaped dense stack.
    fn new(req: &TokenReq, x_in: &[f32], rope_in: &[[f32; 2]], pos: usize) -> Option<Self> {
        let hidden = if x_in.is_empty() {
            req.embd?.2
        } else {
            x_in.len()
        };
        let hd = req.head_dim;
        let q_dim = req.n_heads * hd;
        let kv = req.n_kv_heads * hd;
        let group = req.n_heads / req.n_kv_heads.max(1);
        let rot_dim = req.rot_dim.min(hd);
        let rope_pairs = rot_dim / 2;
        let use_freq_rope_host = rope_in.is_empty() && has_device_rope_freq();
        if !use_freq_rope_host && rope_in.len() < rope_pairs {
            return None;
        }
        let mut ffn_dim = 0usize;
        let mut wq_max = q_dim;
        for layer in req.layers {
            if layer.wq.2 != q_dim || layer.wk.2 != kv || layer.wv.2 != kv || layer.wo.2 != hidden {
                return None;
            }
            wq_max = wq_max.max(layer.wq.2);
            if let TokenFfn::Dense { gate, up, down } = &layer.ffn {
                if gate.2 != up.2 || down.2 != hidden {
                    return None;
                }
                ffn_dim = ffn_dim.max(gate.2);
            } else {
                return None;
            }
        }
        let vocab = req.output.2;

        // Arena layout (f32 elems), Metal-style dense subset.
        let x_at = 0usize;
        let delta_at = x_at + align64(hidden);
        let h_at = delta_at + align64(hidden);
        let q_at = h_at + align64(hidden);
        let k_at = q_at + align64(wq_max);
        let v_at = k_at + align64(kv);
        let attn_at = v_at + align64(kv);
        let gate_at = attn_at + align64(q_dim);
        let up_at = gate_at + align64(ffn_dim);
        let out_logits_at = up_at + align64(ffn_dim);
        let amax_at = out_logits_at + align64(vocab);
        let y_elems = amax_at + align64(2);

        // x_arena: rope + output norm + per-layer norms (stable for CUDA graph).
        // Per layer: attn_norm[H], ffn_norm[H], q_norm[hd], k_norm[hd],
        // and when any layer has attn bias: q_bias[q_dim], k_bias[kv], v_bias[kv].
        let has_attn_bias = req.layers.iter().any(|l| l.q_bias.is_some());
        let rope_q_at = 0usize;
        let rope_k_at = rope_q_at + req.n_heads * rope_pairs * 2;
        let out_norm_at = rope_k_at + req.n_kv_heads * rope_pairs * 2;
        let norms_base = out_norm_at + hidden;
        let bias_stride = if has_attn_bias { q_dim + 2 * kv } else { 0 };
        let per_layer_norms = 2 * hidden + 2 * hd + bias_stride;
        let x_elems = norms_base + req.layers.len() * per_layer_norms;

        let graph_key = {
            let mut h = 0xcbf29ce484222325u64;
            // Include FA pow2 KV bucket so hybrid FA graphs invalidate on resize.
            let mut fa_bucket = 256u64;
            let need = (pos as u64).saturating_add(1);
            while fa_bucket < need {
                fa_bucket *= 2;
            }
            let cache_id = req.cache as *const SharedRegion as usize as u64;
            let ssm_id = req
                .ssm
                .map_or(0u64, |s| s as *const SharedRegion as usize as u64);
            let w0_id = req
                .layers
                .first()
                .map_or(0u64, |l| l.wq.1.as_ptr() as usize as u64);
            let out_id = req.output.1.as_ptr() as usize as u64;
            for &v in &[
                req.layers.len() as u64,
                hidden as u64,
                q_dim as u64,
                kv as u64,
                vocab as u64,
                ffn_dim as u64,
                req.kv_dim as u64,
                hd as u64,
                fa_bucket,
                cache_id,
                ssm_id,
                w0_id,
                out_id,
            ] {
                h ^= v;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        // Opt-in: ALLPAKA_CUDA_GRAPH=1. With ggml FA, default is hybrid segments
        // around attend. ALLPAKA_FA_INLINE=1 captures FA into one whole-token graph
        // (direct fattn is stream-capturable). Without FA, one whole-token graph.
        // Drain ggml *before* capture — sync during capture corrupts the graph.
        let use_graph = crate::runtime::get().cuda_graph;
        let use_fa = crate::gpu::cuda::ggml::enabled() && !crate::runtime::get().cuda_no_fa;
        let fa_inline = use_fa && crate::runtime::get().cuda_fa_inline; // default on with ggml FA + graphs
                                                                        // Shallow-ctx whole graph: skip FA (pads to ≥256) and use native dpos attend.
                                                                        // Opt-in only — default FA wins tg on 5090 (native-auto regressed ~66.4→64.4).
        let native_shallow = crate::runtime::get().cuda_native_attend;
        let fa_inline = fa_inline && !native_shallow;
        let hybrid_fa = use_graph && use_fa && !fa_inline && !native_shallow;
        let whole_graph = use_graph && (!use_fa || fa_inline || native_shallow);

        Some(Self {
            hidden,
            hd,
            q_dim,
            kv,
            group,
            rot_dim,
            rope_pairs,
            ffn_dim,
            vocab,
            x_at,
            h_at,
            q_at,
            k_at,
            v_at,
            attn_at,
            gate_at,
            up_at,
            out_logits_at,
            y_elems,
            has_attn_bias,
            rope_q_at,
            rope_k_at,
            out_norm_at,
            norms_base,
            per_layer_norms,
            x_elems,
            graph_key,
            use_fa,
            fa_inline,
            native_shallow,
            hybrid_fa,
            whole_graph,
        })
    }
}

/// Whole-token dense decode on one stream: activations stay in `y_arena`, one sync.
pub(super) fn decode_token_one_fused(
    req: &TokenReq,
    x_in: &[f32],
    rope_in: &[[f32; 2]],
    pos: usize,
) -> Option<TokenOut> {
    let plan = FusedPlan::new(req, x_in, rope_in, pos)?;
    let FusedPlan {
        hidden,
        hd,
        q_dim,
        kv,
        rope_pairs,
        ffn_dim,
        vocab,
        x_at,
        h_at,
        q_at,
        k_at,
        v_at,
        attn_at,
        gate_at,
        up_at,
        out_logits_at,
        y_elems,
        has_attn_bias,
        rope_q_at,
        rope_k_at,
        out_norm_at,
        norms_base,
        per_layer_norms,
        x_elems,
        graph_key,
        fa_inline,
        hybrid_fa,
        whole_graph,
        ..
    } = plan;
    let rope = if rope_in.len() >= rope_pairs {
        &rope_in[..rope_pairs]
    } else {
        &[][..]
    };

    let out_norm = norm_f32_bytes(req.output_norm, hidden)?;

    with_gpu(|gpu| {
        gpu.ensure_arenas(x_elems * 4, y_elems * 4)?;
        let stream = Arc::clone(&gpu.stream);

        // One-shot: clear any pending ggml work from prefill before cudarc/graph.
        // Force shared stream so prepare_* never StreamSynchronize during capture.
        crate::gpu::cuda::ggml::bind_peer_stream(&stream);
        crate::gpu::cuda::ggml::prepare_for_cudarc();

        let use_freq_rope = gpu.d_rope_freq.is_some()
            && gpu.d_rope_freq_n == rope_pairs
            && gpu.fns.contains_key("rmsnorm_rope_freq_dpos");
        upload_token_inputs(gpu, req, &plan, x_in, rope, pos, use_freq_rope)?;

        if gpu.decode_norms_key != graph_key {
            refresh_layer_norms(gpu, req, &plan, &out_norm)?;
        }

        let t0 = Instant::now();
        let mut launches = 0u64;

        // Arena device addresses are stable until ensure_arenas reallocates
        // (already done above). Drop SyncOnDrop immediately so we can mutably
        // borrow `gpu` for launches; single-stream mode does not event-track.
        let ybase = {
            let (p, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(g);
            p
        };
        let xbase = {
            let (p, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
            drop(g);
            p
        };
        let rope_freq_ptr = if use_freq_rope {
            let (p, g) = DevicePtr::device_ptr(gpu.d_rope_freq.as_ref().unwrap(), &stream);
            drop(g);
            Some(p)
        } else {
            None
        };
        let q8_on = q8_decode_enabled();
        // Allocations are illegal during capture — size Q8 scratch for hidden + FFN now.
        if q8_on {
            let q8_n = hidden.max(ffn_dim).max(q_dim).max(1);
            gpu.ensure_q8(q8_n, 1)?;
        }

        // cudarc's flag enum has no zero variant, and transmuting 0 into it is
        // undefined behaviour (debug builds abort on it). AUTO_FREE_ON_LAUNCH
        // only affects memory-allocation nodes, and captures here allocate
        // nothing (scratch is sized above), so it behaves like no flags.
        let graph_flags = CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

        let n_layers = req.layers.len();
        let capturing_whole =
            whole_graph && (gpu.decode_graph.is_none() || gpu.decode_graph_key != graph_key);
        let capturing_hybrid =
            hybrid_fa && (gpu.decode_fa_graphs.is_none() || gpu.decode_graph_key != graph_key);
        let capturing = capturing_whole; // CaptureGuard / end-of-token for whole-graph only

        let use_qk_rope_table = !capturing_hybrid
            && rope_freq_ptr.is_some()
            && gpu.fns.contains_key("build_rope_freq_dpos")
            && gpu.fns.contains_key("rmsnorm_rope_qk_store_table_dpos");
        let ctx = FusedCtx {
            req,
            plan,
            stream: &stream,
            pos,
            n_layers,
            ybase,
            xbase,
            rope_freq_ptr,
            q8_on,
            capturing_whole,
            use_qk_rope_table,
        };

        if capturing_whole {
            gpu.decode_graph = None;
            gpu.decode_fa_graphs = None;
            gpu.decode_fa_attn = None;
            gpu.decode_fa_kv_offs = None;
            gpu.decode_fa_execs = None;
            // Abort a stale capture left by a prior failed attempt.
            if let Ok(status) = stream.capture_status() {
                use cudarc::driver::sys::CUstreamCaptureStatus::*;
                if matches!(
                    status,
                    CU_STREAM_CAPTURE_STATUS_ACTIVE | CU_STREAM_CAPTURE_STATUS_INVALIDATED
                ) {
                    let _ = stream.end_capture(graph_flags);
                }
            }
            if fa_inline {
                crate::gpu::cuda::ggml::reset_mask_lim();
                if gpu.fns.contains_key("permute_q8_m1") {
                    gpu.ensure_q8(q_dim, 1)?;
                }
            }
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .inspect_err(|&e| {
                    eprintln!("cuda: begin_capture failed: {e}");
                })
                .ok()?;
        } else if whole_graph {
            if let Some(g) = gpu.decode_graph.as_ref() {
                if gpu.decode_graph_key == graph_key {
                    g.launch()
                        .inspect_err(|&e| {
                            eprintln!("cuda: decode graph launch failed: {e}");
                        })
                        .ok()?;
                    launches = 1;
                    let enc = t0.elapsed().as_nanos() as u64;
                    let t1 = Instant::now();
                    gpu.sync()?;
                    note_call(launches, enc, t1.elapsed().as_nanos() as u64);
                    crate::gpu::cuda::ggml::mark_cudarc_dirty();
                    let out = if req.m > 1 {
                        let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
                        let h = stream
                            .clone_dtoh(&gpu.y_arena.slice(h_at..h_at + hidden))
                            .ok()?;
                        TokenOut::Rows {
                            argmax: vec![idx[0]],
                            hidden: h,
                        }
                    } else if req.argmax {
                        let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
                        TokenOut::Argmax(idx[0])
                    } else {
                        let logits = stream
                            .clone_dtoh(&gpu.y_arena.slice(out_logits_at..out_logits_at + vocab))
                            .ok()?;
                        TokenOut::Logits(logits)
                    };
                    return Some(out);
                }
            }
        } else if hybrid_fa {
            let d_pos_p = {
                let (p, g) = DevicePtr::device_ptr(&gpu.d_pos, &stream);
                drop(g);
                p
            };
            // Prefer mega-graph (pre+FA+post+tail as children) when present.
            if gpu.decode_graph_key == graph_key {
                if let Some(mega) = gpu.decode_graph.as_ref() {
                    if mega.launch().is_ok() {
                        launches = 1;
                        let enc = t0.elapsed().as_nanos() as u64;
                        let t1 = Instant::now();
                        gpu.sync()?;
                        note_call(launches, enc, t1.elapsed().as_nanos() as u64);
                        crate::gpu::cuda::ggml::mark_cudarc_dirty();
                        let out = if req.m > 1 {
                            let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
                            let h = stream
                                .clone_dtoh(&gpu.y_arena.slice(h_at..h_at + hidden))
                                .ok()?;
                            TokenOut::Rows {
                                argmax: vec![idx[0]],
                                hidden: h,
                            }
                        } else if req.argmax {
                            let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
                            TokenOut::Argmax(idx[0])
                        } else {
                            let logits = stream
                                .clone_dtoh(
                                    &gpu.y_arena.slice(out_logits_at..out_logits_at + vocab),
                                )
                                .ok()?;
                            TokenOut::Logits(logits)
                        };
                        return Some(out);
                    }
                }
            }
            let replay_ok = ctx.replay_hybrid_segments(gpu, d_pos_p);
            if replay_ok {
                launches = (n_layers * 2 + 1) as u64;
                let enc = t0.elapsed().as_nanos() as u64;
                let t1 = Instant::now();
                gpu.sync()?;
                note_call(launches, enc, t1.elapsed().as_nanos() as u64);
                crate::gpu::cuda::ggml::mark_cudarc_dirty();
                let out = if req.m > 1 {
                    let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
                    let h = stream
                        .clone_dtoh(&gpu.y_arena.slice(h_at..h_at + hidden))
                        .ok()?;
                    TokenOut::Rows {
                        argmax: vec![idx[0]],
                        hidden: h,
                    }
                } else if req.argmax {
                    let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
                    TokenOut::Argmax(idx[0])
                } else {
                    let logits = stream
                        .clone_dtoh(&gpu.y_arena.slice(out_logits_at..out_logits_at + vocab))
                        .ok()?;
                    TokenOut::Logits(logits)
                };
                return Some(out);
            }
            // Key mismatch or empty: fall through to segment capture.
            gpu.decode_fa_graphs = None;
            gpu.decode_fa_attn = None;
            gpu.decode_fa_kv_offs = None;
            gpu.decode_fa_execs = None;
            gpu.decode_graph = None;
        }

        // Always end_capture on exit if we began — leave the stream clean even
        // when a mid-token launch fails and we fall back.
        struct CaptureGuard<'a> {
            stream: &'a Arc<cudarc::driver::CudaStream>,
            active: bool,
            flags: CUgraphInstantiate_flags,
        }
        impl Drop for CaptureGuard<'_> {
            fn drop(&mut self) {
                if self.active {
                    let _ = self.stream.end_capture(self.flags);
                }
            }
        }
        let mut capture_guard = CaptureGuard {
            stream: &stream,
            active: capturing,
            flags: graph_flags,
        };

        // RMSNorm + NeoX RoPE + store K into f16 cache (skips a separate store launch).

        let mut fa_pre: Vec<cudarc::driver::CudaGraph> = Vec::new();
        let mut fa_post: Vec<cudarc::driver::CudaGraph> = Vec::new();
        let mut fa_attn: Vec<cudarc::driver::CudaGraph> = Vec::new();
        let mut fa_attn_ok = capturing_hybrid && !crate::runtime::get().cuda_no_fa_graph;
        let mut fa_warmed = false;
        if capturing_hybrid {
            fa_pre.reserve(n_layers);
            fa_post.reserve(n_layers);
            fa_attn.reserve(n_layers);
        }

        if use_qk_rope_table {
            let f = gpu.func_owned("build_rope_freq_dpos")?;
            let ropep = xbase + (rope_q_at * 4) as u64;
            let fp = rope_freq_ptr?;
            let pairs_u = rope_pairs as u32;
            let cfg = CudaGpu::cfg_1d(pairs_u, 64);
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&ropep)
                    .arg(&fp)
                    .arg(&gpu.d_pos)
                    .arg(&pairs_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            launches += 1;
        }

        for (_li, layer) in req.layers.iter().enumerate() {
            let layer_base = norms_base + _li * per_layer_norms;
            let attn_norm_at = layer_base;
            let ffn_norm_at = layer_base + hidden;
            let q_norm_at = layer_base + 2 * hidden;
            let k_norm_at = q_norm_at + hd;

            if capturing_hybrid {
                stream
                    .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .inspect_err(|&e| {
                        eprintln!("cuda: hybrid pre begin_capture failed: {e}");
                    })
                    .ok()?;
            }

            ctx.launch_rms_into(gpu, h_at, x_at, attn_norm_at, hidden, 1)?;
            launches += 1;

            let (wq_c, wq_o) = resolve_w(gpu, layer.wq.1)?;
            let (wk_c, wk_o) = resolve_w(gpu, layer.wk.1)?;
            let (wv_c, wv_o) = resolve_w(gpu, layer.wv.1)?;
            let (wo_c, wo_o) = resolve_w(gpu, layer.wo.1)?;
            let ggml_decode =
                crate::gpu::cuda::ggml::enabled() && crate::runtime::get().cuda_ggml_decode;
            // Fused QKV: Q4/Q4/Q4 or Q4_K_M Q4/Q4/Q6 (V often Q6_K).
            // Decode: write V straight into f16 KV cache when V has no bias
            // (bias must land in arena first, then store_kv).
            let write_v_cache = layer.v_bias.is_none();
            let qkv_fused = !ggml_decode
                && matches!(
                    (layer.wq.0, layer.wk.0, layer.wv.0),
                    (GgmlType::Q4K, GgmlType::Q4K, GgmlType::Q4K)
                        | (GgmlType::Q4K, GgmlType::Q4K, GgmlType::Q6K)
                )
                && launch_matvec_q4k_qkv(
                    gpu,
                    &QkvReq {
                        q: WeightLoc {
                            chunk: wq_c,
                            off: wq_o,
                        },
                        k: WeightLoc {
                            chunk: wk_c,
                            off: wk_o,
                        },
                        v: WeightLoc {
                            chunk: wv_c,
                            off: wv_o,
                        },
                        v_ty: layer.wv.0,
                        n_in: hidden,
                        n_q: q_dim,
                        n_kv: kv,
                        y_base: ybase,
                        x_off: h_at,
                        q_off: q_at,
                        k_off: k_at,
                        v_off: v_at,
                        m: 1,
                        v_cache: if write_v_cache {
                            Some(VCacheDst {
                                buf: &req.cache.buf,
                                off: layer.v_off as u64,
                                kv_dim: req.kv_dim as u32,
                            })
                        } else {
                            None
                        },
                    },
                )
                .is_some();
            if _li == 0 && crate::runtime::get().cuda_fuse_log {
                eprintln!(
                    "cuda: layer0 qkv_fused={qkv_fused} ggml_decode={ggml_decode} chunks=({wq_c},{wk_c},{wv_c}) types=({:?},{:?},{:?}) q8={} bias={}",
                    layer.wq.0,
                    layer.wk.0,
                    layer.wv.0,
                    crate::gpu::cuda::runtime::q8_decode_enabled(),
                    has_attn_bias
                );
            }
            if !qkv_fused {
                launch_matvec_y_ptr(
                    gpu,
                    DevWeight {
                        ty: layer.wq.0,
                        loc: WeightLoc {
                            chunk: wq_c,
                            off: wq_o,
                        },
                        n_in: hidden,
                        n_out: q_dim,
                    },
                    ArenaMatvec {
                        y_base: ybase,
                        x_off: h_at,
                        y_off: q_at,
                        m: 1,
                        add: false,
                    },
                )?;
                launch_matvec_y_ptr(
                    gpu,
                    DevWeight {
                        ty: layer.wk.0,
                        loc: WeightLoc {
                            chunk: wk_c,
                            off: wk_o,
                        },
                        n_in: hidden,
                        n_out: kv,
                    },
                    ArenaMatvec {
                        y_base: ybase,
                        x_off: h_at,
                        y_off: k_at,
                        m: 1,
                        add: false,
                    },
                )?;
                launch_matvec_y_ptr(
                    gpu,
                    DevWeight {
                        ty: layer.wv.0,
                        loc: WeightLoc {
                            chunk: wv_c,
                            off: wv_o,
                        },
                        n_in: hidden,
                        n_out: kv,
                    },
                    ArenaMatvec {
                        y_base: ybase,
                        x_off: h_at,
                        y_off: v_at,
                        m: 1,
                        add: false,
                    },
                )?;
                launches += 3;
                if ggml_decode {
                    crate::gpu::cuda::ggml::prepare_for_cudarc();
                }
            } else {
                launches += 1;
            }

            // Attn bias after QKV projections (qwen2), before RoPE / KV store.
            if has_attn_bias {
                let qb_at = layer_base + 2 * hidden + 2 * hd;
                let kb_at = qb_at + q_dim;
                let vb_at = kb_at + kv;
                if layer.q_bias.is_some() {
                    ctx.launch_add_bias(gpu, q_at, qb_at, q_dim)?;
                    launches += 1;
                }
                if layer.k_bias.is_some() {
                    ctx.launch_add_bias(gpu, k_at, kb_at, kv)?;
                    launches += 1;
                }
                if layer.v_bias.is_some() {
                    ctx.launch_add_bias(gpu, v_at, vb_at, kv)?;
                    launches += 1;
                }
            }

            let can_fuse_qk_post =
                qkv_fused && layer.q_norm.is_some() && layer.k_norm.is_some() && use_qk_rope_table;
            let qk_post_fused = if can_fuse_qk_post {
                ctx.launch_rms_rope_qk_store(gpu, q_at, k_at, q_norm_at, k_norm_at, layer.k_off)?;
                true
            } else {
                false
            };
            if qk_post_fused {
                launches += 1;
            } else if layer.q_norm.is_some() && gpu.fns.contains_key("rmsnorm_rope_neox") {
                ctx.launch_rms_rope(gpu, q_at, q_norm_at, rope_q_at, req.n_heads)?;
                launches += 1;
            } else {
                if layer.q_norm.is_some() {
                    ctx.launch_rms_inplace(gpu, q_at, q_norm_at, hd, req.n_heads)?;
                    launches += 1;
                }
                ctx.launch_rope(gpu, q_at, rope_q_at, req.n_heads)?;
                launches += 1;
            }

            // When QKV wrote V to cache, fuse K norm+rope+store and skip V store.
            let k_stored = if qk_post_fused {
                true
            } else if qkv_fused
                && layer.k_norm.is_some()
                && gpu.fns.contains_key("rmsnorm_rope_store_dpos")
            {
                ctx.launch_rms_rope_store(
                    gpu,
                    k_at,
                    k_norm_at,
                    rope_k_at,
                    layer.k_off,
                    req.n_kv_heads,
                )?;
                launches += 1;
                true
            } else {
                if layer.k_norm.is_some() && gpu.fns.contains_key("rmsnorm_rope_neox") {
                    ctx.launch_rms_rope(gpu, k_at, k_norm_at, rope_k_at, req.n_kv_heads)?;
                    launches += 1;
                } else {
                    if layer.k_norm.is_some() {
                        ctx.launch_rms_inplace(gpu, k_at, k_norm_at, hd, req.n_kv_heads)?;
                        launches += 1;
                    }
                    ctx.launch_rope(gpu, k_at, rope_k_at, req.n_kv_heads)?;
                    launches += 1;
                }
                false
            };

            if qkv_fused && k_stored && write_v_cache {
                // V already in cache from QKV; K stored by rmsnorm_rope_store_dpos.
            } else if qkv_fused && write_v_cache {
                ctx.launch_store(gpu, k_at, layer.k_off, kv)?;
                launches += 1;
            } else if qkv_fused && k_stored {
                // K fused into cache; V stayed in arena (bias) — store V only.
                ctx.launch_store(gpu, v_at, layer.v_off, kv)?;
                launches += 1;
            } else {
                ctx.launch_store_pair(gpu, k_at, v_at, layer.k_off, layer.v_off, kv)?;
                launches += 1;
            }

            if capturing_hybrid {
                match stream.end_capture(graph_flags) {
                    Ok(Some(g)) => {
                        g.launch().ok()?;
                        fa_pre.push(g);
                    }
                    Ok(None) => {
                        eprintln!("cuda: hybrid pre graph empty");
                        return None;
                    }
                    Err(e) => {
                        eprintln!("cuda: hybrid pre end_capture failed: {e}");
                        return None;
                    }
                }
                // Try capturing FA (mask + fattn + permute) into its own graph.
                let mut captured_fa = false;
                if fa_attn_ok {
                    if !fa_warmed {
                        // First FA outside capture so ggml/fattn cudaMalloc is done.
                        ctx.launch_attend(
                            gpu,
                            q_at,
                            attn_at,
                            layer.k_off,
                            layer.v_off,
                            matches!(layer.wo.0, GgmlType::Q4K | GgmlType::Q6K)
                                && crate::gpu::cuda::runtime::q8_decode_enabled(),
                        )?;
                        launches += 3;
                        fa_warmed = true;
                        crate::gpu::cuda::ggml::reset_mask_lim();
                    }
                    if stream
                        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                        .is_ok()
                    {
                        crate::gpu::cuda::ggml::prepare_from_cudarc(&stream);
                        let (cache_p, cg) = DevicePtr::device_ptr(&req.cache.buf, &stream);
                        drop(cg);
                        let qp = ybase + (q_at * 4) as u64;
                        let op = ybase + (attn_at * 4) as u64;
                        let d_pos_p = {
                            let (p, g) = DevicePtr::device_ptr(&gpu.d_pos, &stream);
                            drop(g);
                            p
                        };
                        let fa_ok = crate::gpu::cuda::ggml::flash_attn(&FlashAttnReq {
                            q: qp,
                            k: cache_p + (layer.k_off as u64) * 2,
                            v: cache_p + (layer.v_off as u64) * 2,
                            out: op,
                            m: 1,
                            shape: FaShape {
                                head_dim: hd,
                                n_q_heads: req.n_heads,
                                n_kv_heads: req.n_kv_heads.max(1),
                                base: pos,
                                kv_dim: req.kv_dim,
                                scale: req.scale,
                            },
                            d_pos: Some(d_pos_p),
                        });
                        match (fa_ok, stream.end_capture(graph_flags)) {
                            (true, Ok(Some(g))) => {
                                g.launch().ok()?;
                                fa_attn.push(g);
                                captured_fa = true;
                            }
                            (_, Ok(None)) | (false, Ok(Some(_))) => {
                                fa_attn_ok = false;
                                fa_attn.clear();
                            }
                            (_, Err(e)) => {
                                eprintln!("cuda: FA graph capture failed ({e}); eager FA fallback");
                                fa_attn_ok = false;
                                fa_attn.clear();
                            }
                        }
                    } else {
                        fa_attn_ok = false;
                        fa_attn.clear();
                    }
                }
                if !captured_fa {
                    ctx.launch_attend(
                        gpu,
                        q_at,
                        attn_at,
                        layer.k_off,
                        layer.v_off,
                        matches!(layer.wo.0, GgmlType::Q4K | GgmlType::Q6K)
                            && crate::gpu::cuda::runtime::q8_decode_enabled(),
                    )?;
                    launches += 3;
                }
            } else {
                ctx.launch_attend(
                    gpu,
                    q_at,
                    attn_at,
                    layer.k_off,
                    layer.v_off,
                    matches!(layer.wo.0, GgmlType::Q4K | GgmlType::Q6K)
                        && crate::gpu::cuda::runtime::q8_decode_enabled(),
                )?;
                launches += 3;
            }

            if capturing_hybrid {
                stream
                    .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .inspect_err(|&e| {
                        eprintln!("cuda: hybrid post begin_capture failed: {e}");
                    })
                    .ok()?;
            }

            launch_matvec_y_ptr(
                gpu,
                DevWeight {
                    ty: layer.wo.0,
                    loc: WeightLoc {
                        chunk: wo_c,
                        off: wo_o,
                    },
                    n_in: q_dim,
                    n_out: hidden,
                },
                ArenaMatvec {
                    y_base: ybase,
                    x_off: attn_at,
                    y_off: x_at,
                    m: 1,
                    add: true,
                },
            )?;
            launches += 1;

            ctx.launch_rms_into(gpu, h_at, x_at, ffn_norm_at, hidden, 1)?;
            launches += 1;

            let TokenFfn::Dense { gate, up, down } = &layer.ffn else {
                return None;
            };
            let (g_c, g_o) = resolve_w(gpu, gate.1)?;
            let (u_c, u_o) = resolve_w(gpu, up.1)?;
            let (d_c, d_o) = resolve_w(gpu, down.1)?;
            // Fused gate+up: Q8 dual-mmvq (SwiGLU in epilogue).
            // Skip when measuring llama MMVQ via ALLPAKA_GGML_DECODE.
            let gu_fused = !ggml_decode
                && matches!((gate.0, up.0), (GgmlType::Q4K, GgmlType::Q4K))
                && gate.2 == up.2
                && launch_matvec_q4k_2(
                    gpu,
                    &GateUpReq {
                        gate: WeightLoc {
                            chunk: g_c,
                            off: g_o,
                        },
                        up: WeightLoc {
                            chunk: u_c,
                            off: u_o,
                        },
                        n_in: hidden,
                        n_out: gate.2,
                        y_base: ybase,
                        x_off: h_at,
                        gate_off: gate_at,
                        up_off: up_at,
                        m: 1,
                    },
                )
                .is_some();
            if !gu_fused {
                launch_matvec_y_ptr(
                    gpu,
                    DevWeight {
                        ty: gate.0,
                        loc: WeightLoc {
                            chunk: g_c,
                            off: g_o,
                        },
                        n_in: hidden,
                        n_out: gate.2,
                    },
                    ArenaMatvec {
                        y_base: ybase,
                        x_off: h_at,
                        y_off: gate_at,
                        m: 1,
                        add: false,
                    },
                )?;
                launch_matvec_y_ptr(
                    gpu,
                    DevWeight {
                        ty: up.0,
                        loc: WeightLoc {
                            chunk: u_c,
                            off: u_o,
                        },
                        n_in: hidden,
                        n_out: up.2,
                    },
                    ArenaMatvec {
                        y_base: ybase,
                        x_off: h_at,
                        y_off: up_at,
                        m: 1,
                        add: false,
                    },
                )?;
                launches += 2;
                if ggml_decode {
                    crate::gpu::cuda::ggml::prepare_for_cudarc();
                }
                let q8_primed = ctx.launch_swiglu(gpu, gate_at, up_at, gate.2)?;
                if !q8_primed {
                    gpu.invalidate_q8();
                }
                launches += 1;
            } else {
                // Dual mmvq wrote silu(gate)*up (+ Q8 when matvec_q4_k_q8_2_q8).
                launches += 1;
                let q8_ready = crate::gpu::cuda::runtime::q8_decode_enabled()
                    && gate.2 % 32 == 0
                    && gpu.q8_src_n == gate.2
                    && gpu.q8_src_m == 1
                    && gpu.q8_src == ybase + (gate_at * 4) as u64;
                if q8_ready {
                    // In-kernel Q8 epilogue already primed the down-proj cache.
                } else if crate::gpu::cuda::runtime::q8_decode_enabled() && gate.2 % 32 == 0 {
                    let gp = ybase + (gate_at * 4) as u64;
                    if down.0 == GgmlType::Q6K {
                        crate::gpu::cuda::runtime::launch_quantize_q8_q6(gpu, gp, gate.2, 1)?;
                    } else {
                        crate::gpu::cuda::runtime::launch_quantize_q8(gpu, gp, gate.2, 1)?;
                    }
                    launches += 1;
                } else {
                    gpu.invalidate_q8();
                }
            }
            launch_matvec_y_ptr(
                gpu,
                DevWeight {
                    ty: down.0,
                    loc: WeightLoc {
                        chunk: d_c,
                        off: d_o,
                    },
                    n_in: gate.2,
                    n_out: hidden,
                },
                ArenaMatvec {
                    y_base: ybase,
                    x_off: gate_at,
                    y_off: x_at,
                    m: 1,
                    add: true,
                },
            )?;
            launches += 1;

            if capturing_hybrid {
                match stream.end_capture(graph_flags) {
                    Ok(Some(g)) => {
                        g.launch().ok()?;
                        fa_post.push(g);
                    }
                    Ok(None) => {
                        eprintln!("cuda: hybrid post graph empty");
                        return None;
                    }
                    Err(e) => {
                        eprintln!("cuda: hybrid post end_capture failed: {e}");
                        return None;
                    }
                }
            }
        }

        if capturing_hybrid {
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .inspect_err(|&e| {
                    eprintln!("cuda: hybrid tail begin_capture failed: {e}");
                })
                .ok()?;
            capture_guard.active = true;
        }

        // Final norm + LM head.
        ctx.launch_rms_into(gpu, h_at, x_at, out_norm_at, hidden, 1)?;
        let (o_c, o_o) = resolve_w(gpu, req.output.1)?;
        launch_matvec_y_ptr(
            gpu,
            DevWeight {
                ty: req.output.0,
                loc: WeightLoc {
                    chunk: o_c,
                    off: o_o,
                },
                n_in: hidden,
                n_out: vocab,
            },
            ArenaMatvec {
                y_base: ybase,
                x_off: h_at,
                y_off: out_logits_at,
                m: 1,
                add: false,
            },
        )?;
        launches += 2;
        if crate::gpu::cuda::ggml::enabled() && crate::runtime::get().cuda_ggml_decode {
            crate::gpu::cuda::ggml::prepare_for_cudarc();
        }

        let out = if req.argmax || req.m > 1 {
            let f = gpu.func_owned("argmax_f32")?;
            let n_u = vocab as u32;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let xp = ybase + (out_logits_at * 4) as u64;
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&xp)
                    .arg(&n_u)
                    .arg(&mut gpu.d_argmax)
                    .launch_pdl(cfg)
            }
            .ok()?;
            launches += 1;
            if capturing_hybrid {
                capture_guard.active = false;
                match stream.end_capture(graph_flags) {
                    Ok(Some(g)) => {
                        g.launch().ok()?;
                        let k_offs: Vec<i32> = req.layers.iter().map(|l| l.k_off as i32).collect();
                        let v_offs: Vec<i32> = req.layers.iter().map(|l| l.v_off as i32).collect();
                        if fa_attn_ok && fa_attn.len() == n_layers {
                            // One parent graph: child launches of pre/FA/post/tail.
                            // Collapses ~193 host launches/token down to 1.
                            let mut mega: Option<cudarc::driver::CudaGraph> = None;
                            if stream
                                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                                .is_ok()
                            {
                                let mut child_ok = true;
                                for li in 0..n_layers {
                                    if fa_pre[li].launch().is_err()
                                        || fa_attn[li].launch().is_err()
                                        || fa_post[li].launch().is_err()
                                    {
                                        child_ok = false;
                                        break;
                                    }
                                }
                                if child_ok {
                                    child_ok = g.launch().is_ok();
                                }
                                match (child_ok, stream.end_capture(graph_flags)) {
                                    (true, Ok(Some(m))) => {
                                        let _ = m.upload();
                                        mega = Some(m);
                                    }
                                    (_, Err(e)) => {
                                        eprintln!(
                                            "cuda: hybrid mega-graph capture failed ({e}); segment replay"
                                        );
                                    }
                                    _ => {
                                        eprintln!(
                                            "cuda: hybrid mega-graph empty/failed; segment replay"
                                        );
                                    }
                                }
                            }
                            if let Some(m) = mega {
                                gpu.decode_graph = Some(m);
                                gpu.decode_fa_attn = Some(fa_attn);
                                gpu.decode_fa_graphs = Some((fa_pre, fa_post, g));
                                gpu.decode_fa_kv_offs = Some((k_offs, v_offs));
                                gpu.decode_fa_execs = None;
                                gpu.decode_graph_key = graph_key;
                                launches = 1;
                                eprintln!(
                                    "cuda: hybrid FA mega-graph captured ({} layers)",
                                    n_layers
                                );
                            } else {
                                gpu.decode_graph = None;
                                for sg in fa_pre.iter().chain(fa_post.iter()).chain(fa_attn.iter())
                                {
                                    let _ = sg.upload();
                                }
                                let _ = g.upload();
                                gpu.decode_fa_attn = Some(fa_attn);
                                gpu.decode_fa_graphs = Some((fa_pre, fa_post, g));
                                gpu.decode_fa_kv_offs = Some((k_offs, v_offs));
                                gpu.decode_fa_execs = None;
                                gpu.decode_graph_key = graph_key;
                                launches = (n_layers * 2 + 1) as u64;
                                eprintln!(
                                    "cuda: hybrid FA graphs captured ({} layers, FA in-graph)",
                                    n_layers
                                );
                            }
                        } else {
                            for sg in fa_pre.iter().chain(fa_post.iter()) {
                                let _ = sg.upload();
                            }
                            let _ = g.upload();
                            gpu.decode_fa_attn = None;
                            gpu.decode_fa_graphs = Some((fa_pre, fa_post, g));
                            gpu.decode_fa_kv_offs = Some((k_offs, v_offs));
                            gpu.decode_fa_execs = None;
                            gpu.decode_graph_key = graph_key;
                            launches = (n_layers * 2 + 1) as u64;
                            eprintln!("cuda: hybrid FA graphs captured ({} layers)", n_layers);
                        }
                    }
                    Ok(None) => {
                        eprintln!("cuda: hybrid tail graph empty");
                        gpu.decode_fa_graphs = None;
                        gpu.decode_fa_attn = None;
                        return None;
                    }
                    Err(e) => {
                        eprintln!("cuda: hybrid tail end_capture failed: {e}");
                        gpu.decode_fa_graphs = None;
                        gpu.decode_fa_attn = None;
                        return None;
                    }
                }
            } else if capturing {
                capture_guard.active = false;
                match stream.end_capture(graph_flags) {
                    Ok(Some(g)) => {
                        // Capture only records — must launch for this token.
                        g.launch().ok()?;
                        gpu.decode_graph = Some(g);
                        gpu.decode_graph_key = graph_key;
                        launches = 1;
                    }
                    Ok(None) => {
                        eprintln!("cuda: decode graph capture produced empty graph");
                        gpu.decode_graph = None;
                        return None;
                    }
                    Err(e) => {
                        eprintln!("cuda: decode graph end_capture failed: {e}");
                        gpu.decode_graph = None;
                        return None;
                    }
                }
            }
            drop(capture_guard);
            let enc = t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            gpu.sync()?;
            note_call(launches, enc, t1.elapsed().as_nanos() as u64);
            crate::gpu::cuda::ggml::mark_cudarc_dirty();
            let idx = stream.clone_dtoh(&gpu.d_argmax).ok()?;
            if req.m > 1 {
                let h = stream
                    .clone_dtoh(&gpu.y_arena.slice(h_at..h_at + hidden))
                    .ok()?;
                TokenOut::Rows {
                    argmax: vec![idx[0]],
                    hidden: h,
                }
            } else {
                TokenOut::Argmax(idx[0])
            }
        } else {
            if capturing_hybrid {
                capture_guard.active = false;
                match stream.end_capture(graph_flags) {
                    Ok(Some(g)) => {
                        g.launch().ok()?;
                        let k_offs: Vec<i32> = req.layers.iter().map(|l| l.k_off as i32).collect();
                        let v_offs: Vec<i32> = req.layers.iter().map(|l| l.v_off as i32).collect();
                        if fa_attn_ok && fa_attn.len() == n_layers {
                            // One parent graph: child launches of pre/FA/post/tail.
                            // Collapses ~193 host launches/token down to 1.
                            let mut mega: Option<cudarc::driver::CudaGraph> = None;
                            if stream
                                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                                .is_ok()
                            {
                                let mut child_ok = true;
                                for li in 0..n_layers {
                                    if fa_pre[li].launch().is_err()
                                        || fa_attn[li].launch().is_err()
                                        || fa_post[li].launch().is_err()
                                    {
                                        child_ok = false;
                                        break;
                                    }
                                }
                                if child_ok {
                                    child_ok = g.launch().is_ok();
                                }
                                match (child_ok, stream.end_capture(graph_flags)) {
                                    (true, Ok(Some(m))) => {
                                        let _ = m.upload();
                                        mega = Some(m);
                                    }
                                    (_, Err(e)) => {
                                        eprintln!(
                                            "cuda: hybrid mega-graph capture failed ({e}); segment replay"
                                        );
                                    }
                                    _ => {
                                        eprintln!(
                                            "cuda: hybrid mega-graph empty/failed; segment replay"
                                        );
                                    }
                                }
                            }
                            if let Some(m) = mega {
                                gpu.decode_graph = Some(m);
                                gpu.decode_fa_attn = Some(fa_attn);
                                gpu.decode_fa_graphs = Some((fa_pre, fa_post, g));
                                gpu.decode_fa_kv_offs = Some((k_offs, v_offs));
                                gpu.decode_fa_execs = None;
                                gpu.decode_graph_key = graph_key;
                                launches = 1;
                                eprintln!(
                                    "cuda: hybrid FA mega-graph captured ({} layers)",
                                    n_layers
                                );
                            } else {
                                gpu.decode_graph = None;
                                for sg in fa_pre.iter().chain(fa_post.iter()).chain(fa_attn.iter())
                                {
                                    let _ = sg.upload();
                                }
                                let _ = g.upload();
                                gpu.decode_fa_attn = Some(fa_attn);
                                gpu.decode_fa_graphs = Some((fa_pre, fa_post, g));
                                gpu.decode_fa_kv_offs = Some((k_offs, v_offs));
                                gpu.decode_fa_execs = None;
                                gpu.decode_graph_key = graph_key;
                                launches = (n_layers * 2 + 1) as u64;
                                eprintln!(
                                    "cuda: hybrid FA graphs captured ({} layers, FA in-graph)",
                                    n_layers
                                );
                            }
                        } else {
                            for sg in fa_pre.iter().chain(fa_post.iter()) {
                                let _ = sg.upload();
                            }
                            let _ = g.upload();
                            gpu.decode_fa_attn = None;
                            gpu.decode_fa_graphs = Some((fa_pre, fa_post, g));
                            gpu.decode_fa_kv_offs = Some((k_offs, v_offs));
                            gpu.decode_fa_execs = None;
                            gpu.decode_graph_key = graph_key;
                            launches = (n_layers * 2 + 1) as u64;
                            eprintln!("cuda: hybrid FA graphs captured ({} layers)", n_layers);
                        }
                    }
                    Ok(None) => {
                        eprintln!("cuda: hybrid tail graph empty");
                        gpu.decode_fa_graphs = None;
                        gpu.decode_fa_attn = None;
                        return None;
                    }
                    Err(e) => {
                        eprintln!("cuda: hybrid tail end_capture failed: {e}");
                        gpu.decode_fa_graphs = None;
                        gpu.decode_fa_attn = None;
                        return None;
                    }
                }
            } else if capturing {
                capture_guard.active = false;
                match stream.end_capture(graph_flags) {
                    Ok(Some(g)) => {
                        g.launch().ok()?;
                        gpu.decode_graph = Some(g);
                        gpu.decode_graph_key = graph_key;
                        launches = 1;
                    }
                    Ok(None) => {
                        eprintln!("cuda: decode graph capture produced empty graph");
                        gpu.decode_graph = None;
                        return None;
                    }
                    Err(e) => {
                        eprintln!("cuda: decode graph end_capture failed: {e}");
                        gpu.decode_graph = None;
                        return None;
                    }
                }
            }
            drop(capture_guard);
            let enc = t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            gpu.sync()?;
            note_call(launches, enc, t1.elapsed().as_nanos() as u64);
            crate::gpu::cuda::ggml::mark_cudarc_dirty();
            let logits = stream
                .clone_dtoh(&gpu.y_arena.slice(out_logits_at..out_logits_at + vocab))
                .ok()?;
            TokenOut::Logits(logits)
        };
        Some(out)
    })
}

/// Stage the token's input row (embedding gather or H2D copy), the host RoPE
/// tables when the device frequency table is not in use, and the position.
fn upload_token_inputs(
    gpu: &mut CudaGpu,
    req: &TokenReq,
    plan: &FusedPlan,
    x_in: &[f32],
    rope: &[[f32; 2]],
    pos: usize,
    use_freq_rope: bool,
) -> Option<()> {
    let FusedPlan {
        hidden,
        rope_pairs,
        x_at,
        rope_q_at,
        rope_k_at,
        ..
    } = *plan;
    let stream = Arc::clone(&gpu.stream);
    if x_in.is_empty() {
        let (ty, w, n_in) = req.embd?;
        let tok = req.token_id?;
        if n_in != hidden {
            return None;
        }
        launch_embed_row(gpu, ty, w, tok, n_in, x_at)?;
    } else {
        stream
            .memcpy_htod(x_in, &mut gpu.y_arena.slice_mut(x_at..x_at + hidden))
            .ok()?;
    }
    if !use_freq_rope {
        if rope.len() < rope_pairs {
            return None;
        }
        let rope_q = rope_replicated(rope, req.n_heads);
        let rope_k = rope_replicated(rope, req.n_kv_heads);
        stream
            .memcpy_htod(
                &rope_q,
                &mut gpu.x_arena.slice_mut(rope_q_at..rope_q_at + rope_q.len()),
            )
            .ok()?;
        stream
            .memcpy_htod(
                &rope_k,
                &mut gpu.x_arena.slice_mut(rope_k_at..rope_k_at + rope_k.len()),
            )
            .ok()?;
    }
    stream.memcpy_htod(&[pos as u32], &mut gpu.d_pos).ok()?;
    Some(())
}

/// Upload the output norm and every layer's norm / QK-norm / bias vectors into
/// the `x_arena` and drop graphs captured against the previous contents.
fn refresh_layer_norms(
    gpu: &mut CudaGpu,
    req: &TokenReq,
    plan: &FusedPlan,
    out_norm: &[f32],
) -> Option<()> {
    let FusedPlan {
        hidden,
        hd,
        q_dim,
        kv,
        has_attn_bias,
        out_norm_at,
        norms_base,
        per_layer_norms,
        graph_key,
        ..
    } = *plan;
    let stream = Arc::clone(&gpu.stream);
    stream
        .memcpy_htod(
            out_norm,
            &mut gpu.x_arena.slice_mut(out_norm_at..out_norm_at + hidden),
        )
        .ok()?;
    for (li, layer) in req.layers.iter().enumerate() {
        let base = norms_base + li * per_layer_norms;
        let attn_w = norm_f32_bytes(layer.attn_norm, hidden)?;
        let ffn_w = norm_f32_bytes(layer.ffn_norm, hidden)?;
        stream
            .memcpy_htod(&attn_w, &mut gpu.x_arena.slice_mut(base..base + hidden))
            .ok()?;
        stream
            .memcpy_htod(
                &ffn_w,
                &mut gpu.x_arena.slice_mut(base + hidden..base + 2 * hidden),
            )
            .ok()?;
        if let Some(wn) = layer.q_norm {
            if wn.len() < hd {
                return None;
            }
            stream
                .memcpy_htod(
                    &wn[..hd],
                    &mut gpu
                        .x_arena
                        .slice_mut(base + 2 * hidden..base + 2 * hidden + hd),
                )
                .ok()?;
        }
        if let Some(wn) = layer.k_norm {
            if wn.len() < hd {
                return None;
            }
            stream
                .memcpy_htod(
                    &wn[..hd],
                    &mut gpu
                        .x_arena
                        .slice_mut(base + 2 * hidden + hd..base + 2 * hidden + 2 * hd),
                )
                .ok()?;
        }
        if has_attn_bias {
            let qb_at = base + 2 * hidden + 2 * hd;
            let kb_at = qb_at + q_dim;
            let vb_at = kb_at + kv;
            if let Some(b) = layer.q_bias {
                let qb = norm_f32_bytes(b, q_dim)?;
                stream
                    .memcpy_htod(&qb, &mut gpu.x_arena.slice_mut(qb_at..qb_at + q_dim))
                    .ok()?;
            }
            if let Some(b) = layer.k_bias {
                let kb = norm_f32_bytes(b, kv)?;
                stream
                    .memcpy_htod(&kb, &mut gpu.x_arena.slice_mut(kb_at..kb_at + kv))
                    .ok()?;
            }
            if let Some(b) = layer.v_bias {
                let vb = norm_f32_bytes(b, kv)?;
                stream
                    .memcpy_htod(&vb, &mut gpu.x_arena.slice_mut(vb_at..vb_at + kv))
                    .ok()?;
            }
        }
    }
    gpu.decode_norms_key = graph_key;
    gpu.decode_graph = None;
    gpu.decode_fa_graphs = None;
    gpu.decode_fa_attn = None;
    gpu.decode_fa_kv_offs = None;
    gpu.decode_fa_execs = None;
    gpu.decode_graph_key = 0;
    Some(())
}

/// Per-call state of one fused decode token shared by every launch: the plan,
/// stream, arena base addresses and the flags fixed before the first launch.
#[derive(Clone, Copy)]
struct FusedCtx<'a> {
    req: &'a TokenReq<'a>,
    plan: FusedPlan,
    stream: &'a Arc<cudarc::driver::CudaStream>,
    pos: usize,
    n_layers: usize,
    /// Device base of `y_arena` (activations) and `x_arena` (norms, rope).
    ybase: u64,
    xbase: u64,
    /// Device RoPE frequency table, when the dpos kernels compute sin/cos.
    rope_freq_ptr: Option<u64>,
    q8_on: bool,
    capturing_whole: bool,
    use_qk_rope_table: bool,
}

impl FusedCtx<'_> {
    fn launch_rms_into(
        &self,
        gpu: &mut CudaGpu,
        dst: usize,
        src: usize,
        w_off: usize,
        n: usize,
        rows: usize,
    ) -> Option<()> {
        let FusedCtx {
            req,
            stream,
            ybase,
            xbase,
            q8_on,
            ..
        } = *self;

        let dst_p = ybase + (dst * 4) as u64;
        let src_p = ybase + (src * 4) as u64;
        let w_p = xbase + (w_off * 4) as u64;
        let n_u = n as u32;
        let rows_u = rows as u32;
        let eps = req.eps;
        let cfg = LaunchConfig {
            grid_dim: (rows_u, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Fuse Q8 pack when decode mmvq is on and n is Q4_K-friendly.
        if q8_on && n % 256 == 0 {
            gpu.ensure_q8(n, rows)?;
            let f = gpu.func_owned("rmsnorm_into_f32_q8")?;
            let mut q = gpu.q8_q.slice_mut(0..(n / 32) * 44 * rows);
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&dst_p)
                    .arg(&src_p)
                    .arg(&w_p)
                    .arg(&mut q)
                    .arg(&n_u)
                    .arg(&eps)
                    .arg(&rows_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            gpu.q8_src = dst_p;
            gpu.q8_src_n = n;
            gpu.q8_src_m = rows;
            gpu.q8_off = 0;
            return Some(());
        }
        let f = gpu.func_owned("rmsnorm_into_f32")?;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&dst_p)
                .arg(&src_p)
                .arg(&w_p)
                .arg(&n_u)
                .arg(&eps)
                .arg(&rows_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        gpu.invalidate_q8();
        Some(())
    }

    fn launch_rms_inplace(
        &self,
        gpu: &mut CudaGpu,
        x_off: usize,
        w_off: usize,
        n: usize,
        rows: usize,
    ) -> Option<()> {
        let FusedCtx {
            req,
            stream,
            ybase,
            xbase,
            ..
        } = *self;

        let f = gpu.func_owned("rmsnorm_f32")?;
        let n_u = n as u32;
        let rows_u = rows as u32;
        let eps = req.eps;
        let cfg = LaunchConfig {
            grid_dim: (rows_u, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let xp = ybase + (x_off * 4) as u64;
        let wp = xbase + (w_off * 4) as u64;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&xp)
                .arg(&wp)
                .arg(&n_u)
                .arg(&eps)
                .arg(&rows_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        Some(())
    }

    fn launch_add_bias(
        &self,
        gpu: &mut CudaGpu,
        x_off: usize,
        bias_off: usize,
        n: usize,
    ) -> Option<()> {
        let FusedCtx {
            stream,
            ybase,
            xbase,
            ..
        } = *self;

        let f = gpu.func_owned("add_bias_f32")?;
        let n_u = n as u32;
        let cfg = CudaGpu::cfg_1d(n_u, 256);
        let xp = ybase + (x_off * 4) as u64;
        let bp = xbase + (bias_off * 4) as u64;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&xp)
                .arg(&bp)
                .arg(&n_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        Some(())
    }

    fn launch_swiglu(&self, gpu: &mut CudaGpu, gate: usize, up: usize, n: usize) -> Option<bool> {
        let FusedCtx { stream, ybase, .. } = *self;

        let gp = ybase + (gate * 4) as u64;
        let up_p = ybase + (up * 4) as u64;
        // Fuse Q8 pack for the following down mmvq when enabled.
        if crate::gpu::cuda::runtime::q8_decode_enabled() && n % 32 == 0 {
            gpu.ensure_q8(n, 1)?;
            let f = gpu.func_owned("swiglu_into_q8")?;
            let n_u = n as u32;
            let rows_u = 1u32;
            let cfg = LaunchConfig {
                grid_dim: (n_u / 32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut q = gpu.q8_q.slice_mut(0..(n / 32) * 36);
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&gp)
                    .arg(&up_p)
                    .arg(&mut q)
                    .arg(&n_u)
                    .arg(&rows_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            gpu.q8_src = gp;
            gpu.q8_src_n = n;
            gpu.q8_src_m = 1;
            gpu.q8_off = 0;
            return Some(true);
        }
        let f = gpu.func_owned("swiglu")?;
        let n_u = n as u32;
        let cfg = CudaGpu::cfg_1d(n_u, 256);
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&gp)
                .arg(&up_p)
                .arg(&n_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        Some(false)
    }

    fn launch_rms_rope(
        &self,
        gpu: &mut CudaGpu,
        x_off: usize,
        w_off: usize,
        rope_off: usize,
        heads: usize,
    ) -> Option<()> {
        let FusedCtx {
            req,
            stream,
            ybase,
            xbase,
            rope_freq_ptr,
            plan: FusedPlan { hd, rot_dim, .. },
            ..
        } = *self;

        let heads_u = heads as u32;
        let hd_u = hd as u32;
        let rot_u = rot_dim as u32;
        let eps = req.eps;
        let cfg = LaunchConfig {
            grid_dim: (heads_u, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let xp = ybase + (x_off * 4) as u64;
        let wp = xbase + (w_off * 4) as u64;
        if let Some(fp) = rope_freq_ptr {
            let f = gpu.func_owned("rmsnorm_rope_freq_dpos")?;
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&xp)
                    .arg(&wp)
                    .arg(&fp)
                    .arg(&gpu.d_pos)
                    .arg(&heads_u)
                    .arg(&hd_u)
                    .arg(&rot_u)
                    .arg(&eps)
                    .launch_pdl(cfg)
            }
            .ok()?;
        } else {
            let f = gpu.func_owned("rmsnorm_rope_neox")?;
            let rp = xbase + (rope_off * 4) as u64;
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&xp)
                    .arg(&wp)
                    .arg(&rp)
                    .arg(&heads_u)
                    .arg(&hd_u)
                    .arg(&rot_u)
                    .arg(&eps)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        Some(())
    }

    fn launch_rms_rope_store(
        &self,
        gpu: &mut CudaGpu,
        x_off: usize,
        w_off: usize,
        rope_off: usize,
        k_cache_off: usize,
        heads: usize,
    ) -> Option<()> {
        let FusedCtx {
            req,
            stream,
            ybase,
            xbase,
            rope_freq_ptr,
            plan: FusedPlan { hd, rot_dim, .. },
            ..
        } = *self;

        let heads_u = heads as u32;
        let hd_u = hd as u32;
        let rot_u = rot_dim as u32;
        let kv_dim_u = req.kv_dim as u32;
        let base_off = k_cache_off as u64;
        let eps = req.eps;
        let cfg = LaunchConfig {
            grid_dim: (heads_u, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let xp = ybase + (x_off * 4) as u64;
        let wp = xbase + (w_off * 4) as u64;
        if let Some(fp) = rope_freq_ptr {
            let f = gpu.func_owned("rmsnorm_rope_store_freq_dpos")?;
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&xp)
                    .arg(&wp)
                    .arg(&fp)
                    .arg(&req.cache.buf)
                    .arg(&base_off)
                    .arg(&gpu.d_pos)
                    .arg(&heads_u)
                    .arg(&hd_u)
                    .arg(&rot_u)
                    .arg(&kv_dim_u)
                    .arg(&eps)
                    .launch_pdl(cfg)
            }
            .ok()?;
        } else {
            let f = gpu.func_owned("rmsnorm_rope_store_dpos")?;
            let rp = xbase + (rope_off * 4) as u64;
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&xp)
                    .arg(&wp)
                    .arg(&rp)
                    .arg(&req.cache.buf)
                    .arg(&base_off)
                    .arg(&gpu.d_pos)
                    .arg(&heads_u)
                    .arg(&hd_u)
                    .arg(&rot_u)
                    .arg(&kv_dim_u)
                    .arg(&eps)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        Some(())
    }

    fn launch_rms_rope_qk_store(
        &self,
        gpu: &mut CudaGpu,
        q_off: usize,
        k_off: usize,
        qw_off: usize,
        kw_off: usize,
        k_cache_off: usize,
    ) -> Option<()> {
        let FusedCtx {
            req,
            stream,
            ybase,
            xbase,
            use_qk_rope_table,
            plan:
                FusedPlan {
                    hd,
                    rot_dim,
                    rope_q_at,
                    ..
                },
            ..
        } = *self;

        if !use_qk_rope_table {
            return None;
        }
        let f = gpu.func_owned("rmsnorm_rope_qk_store_table_dpos")?;
        let q_heads_u = req.n_heads as u32;
        let kv_heads_u = req.n_kv_heads as u32;
        let hd_u = hd as u32;
        let rot_u = rot_dim as u32;
        let kv_dim_u = req.kv_dim as u32;
        let base_off = k_cache_off as u64;
        let eps = req.eps;
        let qp = ybase + (q_off * 4) as u64;
        let kp = ybase + (k_off * 4) as u64;
        let qwp = xbase + (qw_off * 4) as u64;
        let kwp = xbase + (kw_off * 4) as u64;
        let ropep = xbase + (rope_q_at * 4) as u64;
        let cfg = LaunchConfig {
            grid_dim: (q_heads_u + kv_heads_u, 1, 1),
            block_dim: (hd.next_power_of_two().clamp(32, 256) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&qp)
                .arg(&kp)
                .arg(&qwp)
                .arg(&kwp)
                .arg(&ropep)
                .arg(&req.cache.buf)
                .arg(&base_off)
                .arg(&gpu.d_pos)
                .arg(&q_heads_u)
                .arg(&kv_heads_u)
                .arg(&hd_u)
                .arg(&rot_u)
                .arg(&kv_dim_u)
                .arg(&eps)
                .launch_pdl(cfg)
        }
        .ok()?;
        Some(())
    }

    fn launch_rope(
        &self,
        gpu: &mut CudaGpu,
        x_off: usize,
        rope_off: usize,
        heads: usize,
    ) -> Option<()> {
        let FusedCtx {
            stream,
            ybase,
            xbase,
            rope_freq_ptr,
            plan: FusedPlan { hd, rot_dim, .. },
            ..
        } = *self;

        let heads_u = heads as u32;
        let hd_u = hd as u32;
        let rot_u = rot_dim as u32;
        let cfg = LaunchConfig {
            grid_dim: (heads_u, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let xp = ybase + (x_off * 4) as u64;
        if let Some(fp) = rope_freq_ptr {
            let f = gpu.func_owned("rope_neox_freq_dpos")?;
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&xp)
                    .arg(&fp)
                    .arg(&gpu.d_pos)
                    .arg(&heads_u)
                    .arg(&hd_u)
                    .arg(&rot_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
        } else {
            let f = gpu.func_owned("rope_neox")?;
            let rp = xbase + (rope_off * 4) as u64;
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&xp)
                    .arg(&rp)
                    .arg(&heads_u)
                    .arg(&hd_u)
                    .arg(&rot_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        Some(())
    }

    fn launch_store(&self, gpu: &mut CudaGpu, src: usize, off: usize, n: usize) -> Option<()> {
        let FusedCtx {
            req, stream, ybase, ..
        } = *self;

        let f = gpu.func_owned("store_kv_f16_dpos")?;
        let n_u = n as u32;
        let base_off = off as u64;
        let kv_dim_u = req.kv_dim as u32;
        let cfg = CudaGpu::cfg_1d(n_u, 256);
        let sp = ybase + (src * 4) as u64;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&req.cache.buf)
                .arg(&sp)
                .arg(&base_off)
                .arg(&gpu.d_pos)
                .arg(&kv_dim_u)
                .arg(&n_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        Some(())
    }

    fn launch_store_pair(
        &self,
        gpu: &mut CudaGpu,
        k_src: usize,
        v_src: usize,
        k_off: usize,
        v_off: usize,
        n: usize,
    ) -> Option<()> {
        let FusedCtx {
            req, stream, ybase, ..
        } = *self;

        if let Some(f) = gpu.func_owned("store_kv_pair_f16_dpos") {
            let n_u = n as u32;
            let k_off_u = k_off as u64;
            let v_off_u = v_off as u64;
            let kv_dim_u = req.kv_dim as u32;
            let cfg = CudaGpu::cfg_1d(n_u, 256);
            let kp = ybase + (k_src * 4) as u64;
            let vp = ybase + (v_src * 4) as u64;
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&req.cache.buf)
                    .arg(&kp)
                    .arg(&vp)
                    .arg(&k_off_u)
                    .arg(&v_off_u)
                    .arg(&gpu.d_pos)
                    .arg(&kv_dim_u)
                    .arg(&n_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            return Some(());
        }
        self.launch_store(gpu, k_src, k_off, n)?;
        self.launch_store(gpu, v_src, v_off, n)?;
        Some(())
    }

    fn launch_attend(
        &self,
        gpu: &mut CudaGpu,
        q_off: usize,
        out_off: usize,
        k_off: usize,
        v_off: usize,
        q8_only: bool,
    ) -> Option<()> {
        let FusedCtx {
            req,
            stream,
            pos,
            ybase,
            capturing_whole,
            plan:
                FusedPlan {
                    hd,
                    group,
                    use_fa,
                    fa_inline,
                    native_shallow,
                    ..
                },
            ..
        } = *self;

        // ggml FA when available. Skip during whole-token graph capture
        // unless ALLPAKA_FA_INLINE (direct fattn is stream-capturable).
        // ALLPAKA_NATIVE_ATTEND forces native dpos attend (better at short KV).
        if use_fa && !native_shallow && (!capturing_whole || fa_inline) {
            crate::gpu::cuda::ggml::prepare_from_cudarc(stream);
            let (cache_p, cg) = DevicePtr::device_ptr(&req.cache.buf, stream);
            drop(cg);
            let qp = ybase + (q_off * 4) as u64;
            let op = ybase + (out_off * 4) as u64;
            let k_dev = cache_p + (k_off as u64) * 2;
            let v_dev = cache_p + (v_off as u64) * 2;
            let q_n = hd * req.n_heads;
            let q8_only = q8_only && gpu.fns.contains_key("permute_q8_m1_qonly");
            let fuse_q8 = q8_only || gpu.fns.contains_key("permute_q8_m1");
            if fuse_q8 {
                gpu.ensure_q8(q_n, 1)?;
                // Non-null flag: DLL skips permute; NVRTC does permute+Q8.
                crate::gpu::cuda::ggml::fa_set_q8(1);
            }
            if crate::gpu::cuda::ggml::flash_attn(&FlashAttnReq {
                q: qp,
                k: k_dev,
                v: v_dev,
                out: op,
                m: 1,
                shape: FaShape {
                    head_dim: hd,
                    n_q_heads: req.n_heads,
                    n_kv_heads: req.n_kv_heads.max(1),
                    base: pos,
                    kv_dim: req.kv_dim,
                    scale: req.scale,
                },
                d_pos: {
                    let (p, g) = DevicePtr::device_ptr(&gpu.d_pos, stream);
                    drop(g);
                    Some(p)
                },
            }) {
                if fuse_q8 && crate::gpu::cuda::ggml::fa_q8_consumed() {
                    {
                        let fa = crate::gpu::cuda::ggml::fa_last_src()?;
                        let wo_fa_q8 =
                            q8_only && crate::runtime::get().cuda_wo_fa_q8 && q_n == 8192;
                        if wo_fa_q8 {
                            // Skip permute_q8_m1_qonly; WO mmvq packs from FA buffer.
                            gpu.wo_fa_x = fa;
                        } else if q8_only {
                            crate::gpu::cuda::runtime::launch_permute_q8_m1_qonly(
                                gpu,
                                fa,
                                op,
                                hd,
                                req.n_heads,
                            )?;
                        } else {
                            crate::gpu::cuda::runtime::launch_permute_q8_m1(
                                gpu,
                                fa,
                                op,
                                hd,
                                req.n_heads,
                            )?;
                        }
                    }
                }
                crate::gpu::cuda::ggml::mark_cudarc_dirty();
                return Some(());
            }
        }
        let f = if attend_use_kv_fused(hd, group) {
            gpu.func_owned("attend_gqa_dpos_kv")?
        } else {
            gpu.func_owned("attend_gqa_dpos")?
        };
        let k_off_u = k_off as u64;
        let v_off_u = v_off as u64;
        let kv_dim_u = req.kv_dim as u32;
        let head_dim_u = hd as u32;
        let n_q = req.n_heads as u32;
        let group_u = group as u32;
        let scale = req.scale;
        let grid_x = if attend_use_kv_fused(hd, group) {
            n_q / group_u
        } else {
            n_q
        };
        let cfg = if attend_use_kv_fused(hd, group) {
            attend_gqa_kv_launch_cfg(grid_x, 1, group_u, head_dim_u)
        } else {
            attend_gqa_launch_cfg(grid_x, 1)
        };
        let qp = ybase + (q_off * 4) as u64;
        let op = ybase + (out_off * 4) as u64;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&qp)
                .arg(&req.cache.buf)
                .arg(&op)
                .arg(&k_off_u)
                .arg(&v_off_u)
                .arg(&kv_dim_u)
                .arg(&head_dim_u)
                .arg(&n_q)
                .arg(&group_u)
                .arg(&gpu.d_pos)
                .arg(&scale)
                .launch_pdl(cfg)
        }
        .ok()?;
        Some(())
    }

    /// Replay last token's hybrid segment graphs (pre / FA / post per layer,
    /// then the tail) when they match this token's graph key.
    fn replay_hybrid_segments(&self, gpu: &CudaGpu, d_pos_p: u64) -> bool {
        let FusedCtx {
            req,
            stream,
            pos,
            n_layers,
            ybase,
            plan:
                FusedPlan {
                    hd,
                    q_at,
                    attn_at,
                    graph_key,
                    ..
                },
            ..
        } = *self;
        if gpu.decode_graph_key == graph_key {
            if let (Some((pre, post, tail)), Some((k_offs, v_offs))) = (
                gpu.decode_fa_graphs.as_ref(),
                gpu.decode_fa_kv_offs.as_ref(),
            ) {
                if pre.len() == n_layers && post.len() == n_layers && k_offs.len() == n_layers {
                    let use_cpp = crate::runtime::get().cuda_hybrid_replay;
                    if use_cpp {
                        let (cache_p, cg) = DevicePtr::device_ptr(&req.cache.buf, stream);
                        drop(cg);
                        let qp = ybase + (q_at * 4) as u64;
                        let op = ybase + (attn_at * 4) as u64;
                        let pre_e: Vec<*mut std::ffi::c_void> = pre
                            .iter()
                            .map(|g| g.cu_graph_exec() as *mut std::ffi::c_void)
                            .collect();
                        let post_e: Vec<*mut std::ffi::c_void> = post
                            .iter()
                            .map(|g| g.cu_graph_exec() as *mut std::ffi::c_void)
                            .collect();
                        crate::gpu::cuda::ggml::hybrid_replay(
                            stream,
                            &HybridReplay {
                                pre_execs: &pre_e,
                                post_execs: &post_e,
                                tail_exec: tail.cu_graph_exec() as *mut std::ffi::c_void,
                                q: qp,
                                out: op,
                                cache_base: cache_p,
                                k_off: k_offs,
                                v_off: v_offs,
                                shape: FaShape {
                                    head_dim: hd,
                                    n_q_heads: req.n_heads,
                                    n_kv_heads: req.n_kv_heads.max(1),
                                    base: pos,
                                    kv_dim: req.kv_dim,
                                    scale: req.scale,
                                },
                                d_pos: d_pos_p,
                            },
                        )
                    } else if stream.context().bind_to_thread().is_err() {
                        false
                    } else {
                        let cu_stream = stream.cu_stream();
                        let (cache_p, cg) = DevicePtr::device_ptr(&req.cache.buf, stream);
                        drop(cg);
                        let qp = ybase + (q_at * 4) as u64;
                        let op = ybase + (attn_at * 4) as u64;
                        let shared = crate::gpu::cuda::ggml::shared_stream();
                        let fa_graphs = gpu.decode_fa_attn.as_ref();
                        let use_fa_graphs = fa_graphs.map(|g| g.len() == n_layers).unwrap_or(false);
                        let mut ok = true;
                        for li in 0..n_layers {
                            if unsafe {
                                cudarc::driver::result::graph::launch(
                                    pre[li].cu_graph_exec(),
                                    cu_stream,
                                )
                            }
                            .is_err()
                            {
                                ok = false;
                                break;
                            }
                            if use_fa_graphs {
                                if unsafe {
                                    cudarc::driver::result::graph::launch(
                                        fa_graphs.unwrap()[li].cu_graph_exec(),
                                        cu_stream,
                                    )
                                }
                                .is_err()
                                {
                                    ok = false;
                                    break;
                                }
                            } else {
                                if !shared {
                                    crate::gpu::cuda::ggml::prepare_from_cudarc(stream);
                                }
                                let k_dev = cache_p + (k_offs[li] as u64) * 2;
                                let v_dev = cache_p + (v_offs[li] as u64) * 2;
                                if !crate::gpu::cuda::ggml::flash_attn(&FlashAttnReq {
                                    q: qp,
                                    k: k_dev,
                                    v: v_dev,
                                    out: op,
                                    m: 1,
                                    shape: FaShape {
                                        head_dim: hd,
                                        n_q_heads: req.n_heads,
                                        n_kv_heads: req.n_kv_heads.max(1),
                                        base: pos,
                                        kv_dim: req.kv_dim,
                                        scale: req.scale,
                                    },
                                    d_pos: Some(d_pos_p),
                                }) {
                                    ok = false;
                                    break;
                                }
                            }
                            if unsafe {
                                cudarc::driver::result::graph::launch(
                                    post[li].cu_graph_exec(),
                                    cu_stream,
                                )
                            }
                            .is_err()
                            {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            ok = unsafe {
                                cudarc::driver::result::graph::launch(
                                    tail.cu_graph_exec(),
                                    cu_stream,
                                )
                            }
                            .is_ok();
                        }
                        if ok && !shared {
                            crate::gpu::cuda::ggml::mark_cudarc_dirty();
                        }
                        ok
                    }
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    }
}
