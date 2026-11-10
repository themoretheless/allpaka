//! Fused GPU-resident prefill: the chunk residual stream stays on the device
//! while attention and GDN blocks run one command buffer per layer.

use super::*;

/// Upload the chunk's residual stream for the fused prefill path.
pub fn prefill_begin(xs: &[f32]) -> Option<()> {
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    gpu.prefill_drain();
    gpu.cstamps_begin();
    // One-buffer segments arm lazily: the first eligible layer of the
    // chunk opens the shared command buffer (see prefill_attn_block), an
    // ineligible layer seals it again (pf_obuf_seal). Nothing to arm here.
    gpu.pf_obuf_cmd = None;
    gpu.pf_hs_normed.set(false);
    gpu.ensure_prefill(xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), gpu.pf_x.contents() as *mut f32, xs.len());
    }
    gpu.pf_active_len = Some(xs.len());
    Some(())
}

/// Download the residual stream after the last fused layer.
/// Drain GPU writers before exposing CPU slices of the residual stream.
pub(super) fn prefill_complete(gpu: &mut Gpu) {
    let mut cal = None;
    if let Some(cmd) = gpu.pf_obuf_cmd.take() {
        // The one-buffer chunk commits here: the only wait of the prefill.
        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        use objc::{msg_send, sel, sel_impl};
        let (gs, ge): (f64, f64) =
            unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
        cal = Some((gs, ge));
        note_gpu_times(&cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
    }
    gpu.prefill_drain();
    if let (Some((tab, n_ex)), true) = (gpu.dbg_route.get(), crate::runtime::get().rows_debug) {
        let rb = gpu.route_buf.contents() as *const u32;
        let mut rows: Vec<u32> = (0..n_ex)
            .map(|i| unsafe { *rb.add(tab + i * 3 + 2) })
            .collect();
        rows.sort_unstable_by(|a, b| b.cmp(a));
        let total: u32 = rows.iter().sum();
        let nz: Vec<u32> = rows.iter().copied().filter(|&r| r > 0).collect();
        let tiles: u32 = nz.iter().map(|&r| r.div_ceil(32)).sum();
        eprintln!(
            "rows/expert (sorted, top12): {:?}{} | nonzero {}/{} total {} tiles {} waste {:.1}%",
            &nz[..nz.len().min(12)],
            if nz.len() > 12 { " ..." } else { "" },
            nz.len(),
            n_ex,
            total,
            tiles,
            100.0 * (tiles * 32 - total) as f64 / (tiles * 32).max(1) as f64
        );
    }
    gpu.cstamps_report(cal);
}

pub fn prefill_end(xs: &mut [f32]) -> Option<()> {
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    prefill_complete(&mut gpu);
    if gpu.pf_active_len != Some(xs.len()) || xs.len().checked_mul(4)? > gpu.pf_x_cap {
        return None;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(gpu.pf_x.contents() as *const f32, xs.as_mut_ptr(), xs.len());
    }
    gpu.pf_active_len = None;
    Some(())
}

pub(super) static SHARED_OUTPUTS: AtomicU64 = AtomicU64::new(0);
pub(super) static SHARED_HANDOFF_BYTES: AtomicU64 = AtomicU64::new(0);
pub(super) static SHARED_PROJECTION_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn shared_activation_stats() -> crate::activation::SharedStats {
    crate::activation::SharedStats {
        outputs: SHARED_OUTPUTS.load(Ordering::Relaxed),
        handoff_bytes: SHARED_HANDOFF_BYTES.load(Ordering::Relaxed),
        projection_bytes: SHARED_PROJECTION_BYTES.load(Ordering::Relaxed),
    }
}

/// A completed, exclusively owned shared allocation. No GPU arena retains this
/// buffer. Borrowing it for a synchronous dispatch prevents CPU mutation until
/// that dispatch completes. Dropping it returns storage to a bounded spare slot.
pub struct SharedVector {
    pub(super) buf: Option<Buffer>,
    pub(super) len: usize,
    pub(super) spare: std::sync::Arc<Mutex<Option<Buffer>>>,
}

impl std::ops::Deref for SharedVector {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // Only constructed after prefill_complete; storage outlives the slice.
        unsafe {
            std::slice::from_raw_parts(self.buf.as_ref().unwrap().contents().cast(), self.len)
        }
    }
}
impl std::ops::DerefMut for SharedVector {
    fn deref_mut(&mut self) -> &mut [f32] {
        unsafe {
            std::slice::from_raw_parts_mut(self.buf.as_ref().unwrap().contents().cast(), self.len)
        }
    }
}
impl Drop for SharedVector {
    fn drop(&mut self) {
        if let Ok(mut spare) = self.spare.lock() {
            if spare.is_none() {
                *spare = self.buf.take();
            }
        }
    }
}

/// Hand the completed residual allocation to its next consumer without memcpy.
/// The old copying path remains available for controlled A/B measurements.
pub fn prefill_end_output(len: usize) -> Option<crate::activation::Activation> {
    if !crate::runtime::get().prefill_zero_copy {
        let mut out = vec![0.0; len];
        prefill_end(&mut out)?;
        return Some(out.into());
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    prefill_complete(&mut gpu);
    if gpu.pf_active_len != Some(len) || len.checked_mul(4)? > gpu.pf_x_cap {
        return None;
    }
    let spare = gpu.pf_spare.lock().ok()?.take();
    let replacement = spare
        .filter(|b| b.length() as usize == gpu.pf_x_cap)
        .unwrap_or_else(|| {
            gpu.device
                .new_buffer(gpu.pf_x_cap as u64, MTLResourceOptions::StorageModeShared)
        });
    // Residency sets must not retain every output ever handed to a caller.
    if let Some(set) = gpu.residency {
        use objc::runtime::Object;
        use objc::{msg_send, sel, sel_impl};
        unsafe {
            let _: () =
                msg_send![set as *mut Object, removeAllocation: gpu.pf_x.as_ptr() as *mut Object];
        }
    }
    gpu.pin_buffer_in_residency(&replacement);
    let buf = std::mem::replace(&mut gpu.pf_x, replacement);
    gpu.pf_active_len = None;
    SHARED_OUTPUTS.fetch_add(1, Ordering::Relaxed);
    SHARED_HANDOFF_BYTES.fetch_add((len * 4) as u64, Ordering::Relaxed);
    Some(crate::activation::Activation::Shared(SharedVector {
        buf: Some(buf),
        len,
        spare: gpu.pf_spare.clone(),
    }))
}

/// Wait out deferred prefill buffers after a mid-chunk fallback: the CPU
/// path that takes over would otherwise race buffers still in flight.
pub fn prefill_abort() {
    if let Some(Some(cell)) = GPU.get().map(|c| c.as_ref()) {
        if let Ok(mut gpu) = cell.lock() {
            // A partial one-buffer chunk was never committed; dropping the
            // retained handle discards it.
            gpu.pf_obuf_cmd = None;
            gpu.pf_hs_normed.set(false);
            gpu.prefill_drain();
            gpu.pf_active_len = None;
        }
    }
}

pub fn prefill_attn_block(req: &PrefillAttnReq) -> Option<Vec<f32>> {
    let hidden = req.hs.len() / req.m.max(1);
    let hd = req.head_dim;
    // 128 everywhere; 256 on qwen35moe's full-attention layers. Below the
    // tile-matmul threshold the projections would need per-row matvec tiling
    // this encoder does not carry; short tail chunks decline to the
    // step-by-step path instead. (Caught by decode_paths_agree: a TILE=1
    // pipeline silently computes only the first row.)
    let rot_ok = (hd == ATTN_HEAD_DIM && (req.rot_dim == hd || req.rot_dim * 2 == hd))
        || (hd == 256 && req.rot_dim == 64);
    if !(hd == ATTN_HEAD_DIM || hd == 256)
        || !rot_ok
        || req.ropes.len() != req.m * req.rot_dim / 2
        || req.hs.len() != req.m * hidden
        || req.m < MM_MIN_M
    {
        return None;
    }
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let wq_out = if req.gate_in_q { 2 * q_dim } else { q_dim };
    if req.wq.2 != wq_out || req.wk.2 != kv || req.wv.2 != kv || req.wo.2 != hidden {
        return None;
    }
    let span = (req.base + req.m) * req.kv_dim;
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    if (req.k_off + span) * 2 > req.cache.len || (req.v_off + span) * 2 > req.cache.len {
        return None;
    }

    let mats = [
        resolve(&gpu, req.wq.0, req.wq.1)?,
        resolve(&gpu, req.wk.0, req.wk.1)?,
        resolve(&gpu, req.wv.0, req.wv.1)?,
        resolve(&gpu, req.wo.0, req.wo.1)?,
    ];
    #[allow(unused_variables)]
    let mut states = Vec::with_capacity(4);
    for mat in &mats {
        states.push(
            gpu.pipeline(mm_kernel_for(mat.ty, req.m)?.0, 1, 1)?
                .to_owned(),
        );
    }
    let (qkv_state, qkv_pad) = if wq_out % 64 == 0 && kv % 64 == 0 {
        if mats[0].ty == mats[1].ty
            && mats[1].ty == mats[2].ty
            && (mats[0].ty == GgmlType::Q8_0 || hidden % 256 == 0)
        {
            (
                mm_qkv_kernel_for(mats[0].ty)
                    .and_then(|name| gpu.pipeline(name, 1, 1).map(|p| p.to_owned())),
                0u32,
            )
        } else if mm_ll() && !mm_k64() && mm_pipe() && !crate::runtime::get().no_fused_qkv {
            match (
                super::decode::dq_fmt(mats[0].ty),
                super::decode::dq_fmt(mats[1].ty),
                super::decode::dq_fmt(mats[2].ty),
            ) {
                (Some(fq), Some(fk), Some(fv))
                    if fq <= 5
                        && fk <= 5
                        && fv <= 5
                        && ((fq == 3 && fk == 3 && fv == 3) || hidden % 256 == 0) =>
                {
                    if debug_qkv() {
                        eprintln!(
                            "debug-qkv: mmllp_qkv_mixed selected ({:?}/{:?}/{:?})",
                            mats[0].ty, mats[1].ty, mats[2].ty
                        );
                    }
                    let pad = fq | (fk << 8) | (fv << 16);
                    (
                        gpu.pipeline("mmllp_qkv_mixed", 1, 1).map(|p| p.to_owned()),
                        pad,
                    )
                }
                _ => (None, 0),
            }
        } else {
            (None, 0)
        }
    } else {
        (None, 0)
    };
    let prep_state = gpu.pipelines[&("qk_prep_batch", 1, 1)].to_owned();
    let attend_state = gpu.pipelines[&(attend_rows_kernel(), 1, 1)].to_owned();

    // Fusion: norms and the router resolved up front; the fused path reads
    // xs from pf_x, so the hs upload disappears. A leading Dense layer
    // (GLM's layer 0) fuses the norms but has no router: n_expert == 0.
    struct FusionRefs {
        attn_norm: NormRef,
        ffn_norm: NormRef,
        router: Option<(MatRef, ComputePipelineState)>,
        n_expert: usize,
    }
    let fusion = match &req.fusion {
        Some(f) => {
            if gpu.pf_x_cap < req.m * hidden * 4 || gpu.pf_hs_cap < req.m * hidden * 4 {
                return None;
            }
            let attn_norm = resolve_norm(&gpu, f.attn_norm, hidden)?;
            let ffn_norm = resolve_norm(&gpu, f.ffn_norm, hidden)?;
            let router = if f.n_expert > 0 {
                let router = resolve_f32(&gpu, f.router, hidden)?;
                let router_state = gpu
                    .pipeline(mm_kernel_for(GgmlType::F32, req.m)?.0, 1, 1)?
                    .to_owned();
                Some((router, router_state))
            } else {
                None
            };
            Some(FusionRefs {
                attn_norm,
                ffn_norm,
                router,
                n_expert: f.n_expert,
            })
        }
        None => None,
    };
    // One-buffer note: a layer that cannot join the chunk's shared
    // command buffer no longer declines here. Resolve failures above
    // already returned None (the chunk aborts and the fallback recomputes
    // it); a merely non-onebuf layer (CPU routing, PF_SPLIT) seals the
    // open segment at the encode site below and runs on its own buffer.
    let norm_state = gpu.pipelines[&("norm_rows", 1, 1)].to_owned();
    let prep256_state = gpu.pipelines[&("qk_prep_batch256", 1, 1)].to_owned();
    let attend256_state = gpu.pipelines[&("attend_rows256", 1, 1)].to_owned();
    let attend_mm256_state = gpu.pipelines[&(attend_mm256_kernel(), 1, 1)].to_owned();
    let gate_mul_state = gpu.pipelines[&("attn_gate_mul", 1, 1)].to_owned();
    // GLM's q/k/v biases resolved outside the encoder closure (`?` needs
    // the function's Option, not the closure's Vec).
    let bias_refs = match req.attn_bias {
        Some((qb, kb, vb)) => Some((
            resolve_f32(&gpu, qb, 0)?,
            resolve_f32(&gpu, kb, 0)?,
            resolve_f32(&gpu, vb, 0)?,
        )),
        None => None,
    };

    // Arenas: x holds hs then the rope tables; y holds q (raw, possibly
    // double-wide with the gate), k, v, the deinterleaved q and gate rows,
    // attention output and the projection.
    let align = |n: usize| (n + 63) & !63;
    let hs_at = 0usize;
    let ropes_at = hs_at + align(req.m * hidden);
    let x_total = ropes_at + align(req.ropes.len() * 2);
    let q_at = 0usize;
    let k_at = q_at + align(req.m * wq_out);
    let v_at = k_at + align(req.m * kv);
    let qn_at = v_at + align(req.m * kv);
    let qgate_at = qn_at + align(req.m * q_dim);
    let attn_at = qgate_at + align(req.m * q_dim);
    let out_at = attn_at + align(req.m * q_dim);
    let router_at = out_at + align(req.m * hidden);
    let n_expert = fusion.as_ref().map_or(0, |f| f.n_expert);
    let y_total = router_at + align(req.m * n_expert);
    gpu.ensure_arenas(x_total * 4, y_total * 4);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut f32;
        if fusion.is_none() {
            std::ptr::copy_nonoverlapping(req.hs.as_ptr(), xp.add(hs_at), req.hs.len());
        }
        std::ptr::copy_nonoverlapping(
            req.ropes.as_ptr() as *const f32,
            xp.add(ropes_at),
            req.ropes.len() * 2,
        );
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        // Cloned queue handle: command buffers borrow it, leaving `gpu`
        // free for the deferred commit bookkeeping.
        let queue = gpu.queue.clone();
        let mut cmd = queue.new_command_buffer();
        let policy = crate::runtime::get();
        // ALLPAKA_PF_SPLIT=1: commit after every stage and print its GPU
        // time. Debug-only; serialises the stages, so wall time degrades.
        let split = policy.pf_split;
        // ALLPAKA_PF_CHAIN=1: a commit per stage like PF_SPLIT, but the
        // stage buffers are only queued, never waited - the in-order queue
        // runs them back-to-back at full clock, so the per-stage
        // GPUStart/EndTime deltas (printed at the end of the layer) are
        // free of the DVFS sag that makes PF_SPLIT numbers unusable.
        let chain = !split && policy.pf_chain;
        // One-buffer chunk (ALLPAKA_PF_ONEBUF=1, default): encode into the
        // chunk's shared command buffer - a sequential encoder per stage,
        // the commit happens once in prefill_end. The segment arms lazily
        // on the first eligible layer; a layer that stays off it seals the
        // open segment first, so its own buffer stays ordered after the
        // work already encoded there.
        let onebuf = fusion.is_some() && gpu_route() && pf_onebuf() && !split && !chain;
        if !onebuf {
            gpu.pf_obuf_seal();
        } else {
            gpu.pf_obuf_arm();
        }
        // Deferred commit: chain behind the previous layer's buffer through
        // the shared event instead of a CPU wait. Only the fused GPU-routing
        // path qualifies - it reads nothing back. PF_SPLIT keeps the old
        // waits (its per-stage commits are a different buffer shape anyway).
        let defer = !onebuf && fusion.is_some() && gpu_route() && pf_defer() && !split && !chain;
        if defer {
            cmd.encode_wait_for_event(&gpu.pf_ev, gpu.pf_ev_val);
        }
        let mut staged: Vec<(&'static str, metal::CommandBuffer)> = Vec::new();
        let mut enc = if onebuf {
            gpu.pf_obuf_cmd
                .as_deref()
                .expect("onebuf armed")
                .compute_command_encoder_with_dispatch_type(serial_dispatch())
        } else {
            cmd.compute_command_encoder_with_dispatch_type(serial_dispatch())
        };
        macro_rules! split_here {
            ($label:expr) => {
                if split || chain {
                    use objc::{msg_send, sel, sel_impl};
                    enc.end_encoding();
                    cmd.commit();
                    if chain {
                        staged.push(($label, cmd.to_owned()));
                    } else {
                        cmd.wait_until_completed();
                        let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
                        let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
                        eprintln!("attn {}: {:.3} ms", $label, (ge - gs) * 1e3);
                    }
                    cmd = queue.new_command_buffer();
                    enc = cmd.compute_command_encoder_with_dispatch_type(
                        metal::MTLDispatchType::Concurrent,
                    );
                }
            };
        }
        let y = &gpu.y_arena;
        let e = |off: usize| (off * 4) as u64;

        let matmul = |enc: &metal::ComputeCommandEncoderRef,
                      state: &ComputePipelineState,
                      mat: &MatRef,
                      n_in: usize,
                      n_out: usize,
                      x_buf: &Buffer,
                      x_off: usize,
                      y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            let m32 = req.m as u32;
            enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
            let (bm, bn) = mm_kernel_for(mat.ty, req.m).map_or((32, 32), |(_, bm, bn)| (bm, bn));
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (n_out as u64).div_ceil(bm as u64),
                    (req.m as u64).div_ceil(bn as u64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        };

        let norm_rows = |enc: &metal::ComputeCommandEncoderRef,
                         norm: &NormRef,
                         use_delta: u32,
                         delta_off: usize| {
            enc.set_compute_pipeline_state(&norm_state);
            enc.set_buffer(0, Some(&gpu.pf_x), 0);
            enc.set_buffer(1, Some(y), e(delta_off));
            enc.set_buffer(2, Some(&gpu.pf_hs), 0);
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.set_bytes(6, 4, &use_delta as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(req.m as u64, 1, 1), MTLSize::new(256, 1, 1));
        };
        let (hs_buf, hs_off): (&Buffer, usize) = if let Some(f) = &fusion {
            if onebuf {
                // The previous layer's FFN stages (pf_x combine, y writes)
                // live on the same command buffer now - order them.
                enc.memory_barrier_with_resources(&[y, &gpu.pf_x, &gpu.pf_hs, &gpu.route_buf]);
            }
            // The FFN combine may already have written the attn-normed rows
            // (combine_rows_resnorm); only the onebuf barrier above orders
            // them then.
            if gpu.pf_hs_normed.take() {
                cstamp!(gpu, enc, "anorm_fused");
            } else {
                norm_rows(enc, &f.attn_norm, 0, out_at);
                cstamp!(gpu, enc, "anorm");
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
                }
            }
            (&gpu.pf_hs, 0)
        } else {
            (&gpu.x_arena, hs_at)
        };
        if pf_skip("qkv") {
            // Priced by absence: buffer structure unchanged, stage gone.
        } else if let Some(qkv_pipe) = &qkv_state {
            enc.set_compute_pipeline_state(qkv_pipe);
            enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
            enc.set_buffer(1, Some(&gpu.chunks[mats[1].chunk].buf), 0);
            enc.set_buffer(2, Some(&gpu.chunks[mats[2].chunk].buf), 0);
            enc.set_buffer(3, Some(hs_buf), e(hs_off));
            enc.set_buffer(4, Some(y), 0);
            let qkv_args = MmQkvArgs {
                n_in: hidden as u32,
                n_q: wq_out as u32,
                n_k: kv as u32,
                n_v: kv as u32,
                wq_off: mats[0].w_off,
                wk_off: mats[1].w_off,
                wv_off: mats[2].w_off,
                q_off: e(q_at),
                k_off: e(k_at),
                v_off: e(v_at),
                m: req.m as u32,
                pad: qkv_pad,
            };
            enc.set_bytes(
                5,
                std::mem::size_of::<MmQkvArgs>() as u64,
                &qkv_args as *const MmQkvArgs as *const _,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(
                    ((wq_out + 2 * kv) as u64).div_ceil(64),
                    (req.m as u64).div_ceil(32),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
            cstamp!(gpu, enc, "qkv");
        } else {
            matmul(
                enc, &states[0], &mats[0], hidden, wq_out, hs_buf, hs_off, q_at,
            );
            cstamp!(gpu, enc, "wq");
            matmul(enc, &states[1], &mats[1], hidden, kv, hs_buf, hs_off, k_at);
            cstamp!(gpu, enc, "wk");
            matmul(enc, &states[2], &mats[2], hidden, kv, hs_buf, hs_off, v_at);
            cstamp!(gpu, enc, "wv");
        }
        split_here!("qkv");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }

        // head_dim-256 path: batched qk_prep256 splits the fused gate out of
        // the q rows and writes normed q to qn_at; attend_rows256 runs one
        // threadgroup per (head, row); the sigmoid gate lands before wo.
        if hd == 256 {
            enc.set_compute_pipeline_state(&prep256_state);
            enc.set_buffer(0, Some(y), e(q_at));
            enc.set_buffer(1, Some(y), e(k_at));
            enc.set_buffer(2, Some(y), e(v_at));
            enc.set_buffer(3, Some(&req.cache.buf), 0);
            let args = QkPrepBatch256Args {
                n_heads: req.n_heads as u32,
                n_kv_heads: req.n_kv_heads as u32,
                kv_dim: req.kv_dim as u32,
                eps: req.eps,
                base: req.base as u32,
                has_qk_norm: req.q_norm.is_some() as u32,
                rot_dim: req.rot_dim as u32,
                gate_in_q: req.gate_in_q as u32,
                k_base: req.k_off as u64,
                v_base: req.v_off as u64,
                rope_norm: req.rope_norm as u32,
                pad0: 0,
            };
            enc.set_bytes(
                4,
                std::mem::size_of::<QkPrepBatch256Args>() as u64,
                &args as *const QkPrepBatch256Args as *const _,
            );
            let ones = [1.0f32; 256];
            let qw = req.q_norm.unwrap_or(&ones);
            let kw = req.k_norm.unwrap_or(&ones);
            enc.set_bytes(5, (256 * 4) as u64, qw.as_ptr() as *const _);
            enc.set_bytes(6, (256 * 4) as u64, kw.as_ptr() as *const _);
            enc.set_buffer(7, Some(&gpu.x_arena), e(ropes_at));
            enc.set_buffer(8, Some(y), e(qn_at));
            enc.set_buffer(9, Some(y), e(qgate_at));
            enc.dispatch_thread_groups(
                MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, req.m as u64, 1),
                MTLSize::new(32, 1, 1),
            );
            cstamp!(gpu, enc, "prep");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
            }
            split_here!("prep");

            enc.set_compute_pipeline_state(&attend256_state);
            enc.set_buffer(0, Some(&req.cache.buf), 0);
            enc.set_buffer(1, Some(&req.cache.buf), 0);
            enc.set_buffer(2, Some(y), e(qn_at));
            enc.set_buffer(3, Some(y), e(attn_at));
            for (index, value) in [
                (4u64, req.kv_dim as u32),
                (5, 256u32),
                (6, req.base as u32),
                (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
            ] {
                enc.set_bytes(index, 4, &value as *const u32 as *const _);
            }
            enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
            for (index, value) in [(9u64, req.k_off as u64), (10, req.v_off as u64)] {
                enc.set_bytes(index, 8, &value as *const u64 as *const _);
            }
            let heads32 = req.n_heads as u32;
            enc.set_bytes(11, 4, &heads32 as *const u32 as *const _);
            if attend_mm() {
                // MMA tiles: same bindings, 8 rows per threadgroup.
                enc.set_compute_pipeline_state(&attend_mm256_state);
                let rows32 = req.m as u32;
                enc.set_bytes(12, 4, &rows32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(req.n_heads as u64, (req.m as u64).div_ceil(8), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                enc.dispatch_thread_groups(
                    MTLSize::new(req.n_heads as u64, req.m as u64, 1),
                    MTLSize::new(512, 1, 1),
                );
            }
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
            if req.gate_in_q {
                enc.set_compute_pipeline_state(&gate_mul_state);
                enc.set_buffer(0, Some(y), e(attn_at));
                enc.set_buffer(1, Some(y), e(qgate_at));
                let n32 = (req.m * q_dim) as u32;
                enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[y]);
                }
            }
            cstamp!(gpu, enc, "attend");
            split_here!("attend");
        } else {
            // Fused prep+attention (attend_mm_fq): one dispatch does the
            // qk-norm/rope/cache prep and the MMA attention, producer groups
            // synced to consumers by a device counter instead of a pipeline
            // barrier. gate_in_q (GLM) keeps the two-dispatch path: its q
            // rows are double-wide, which this kernel does not handle.
            let tiles = (req.m as u32).div_ceil(8);
            let fused_qk = attend_mm()
                && crate::runtime::get().pf_fused_qk
                && !req.gate_in_q
                && req.n_heads >= 2 * req.n_kv_heads
                && tiles <= 64;
            if fused_qk && !pf_skip("attend") {
                // The producer/consumer sync counters are per row-block and
                // must start at zero: consumers wait for this dispatch's
                // producer count, so stale counts from an earlier chunk or
                // session would shortcut the wait (or deadlock it).
                enc.set_compute_pipeline_state(&gpu.pipelines[&("zero_f32", 1, 1)]);
                enc.set_buffer(0, Some(&gpu.pf_attn_sync), 0);
                let zw = 64u32;
                enc.set_bytes(1, 4, &zw as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(64, 1, 1));
                enc.memory_barrier_with_resources(&[&gpu.pf_attn_sync]);
                enc.set_compute_pipeline_state(&gpu.pipelines[&(attend_mm_fq_kernel(), 1, 1)]);
                enc.set_buffer(0, Some(&req.cache.buf), 0);
                enc.set_buffer(1, Some(&req.cache.buf), 0);
                enc.set_buffer(2, Some(y), e(q_at));
                enc.set_buffer(3, Some(y), e(attn_at));
                for (index, value) in [
                    (4u64, req.kv_dim as u32),
                    (5, hd as u32),
                    (6, req.base as u32),
                    (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
                ] {
                    enc.set_bytes(index, 4, &value as *const u32 as *const _);
                }
                enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                for (index, value) in [(9u64, req.k_off as u64), (10, req.v_off as u64)] {
                    enc.set_bytes(index, 8, &value as *const u64 as *const _);
                }
                let heads32 = req.n_heads as u32;
                enc.set_bytes(11, 4, &heads32 as *const u32 as *const _);
                let rows32 = req.m as u32;
                enc.set_bytes(12, 4, &rows32 as *const u32 as *const _);
                enc.set_buffer(13, Some(y), e(k_at));
                enc.set_buffer(14, Some(y), e(v_at));
                let fargs = FqPrepArgs {
                    eps: req.eps,
                    has_qk_norm: req.q_norm.is_some() as u32,
                    rot_dim: req.rot_dim as u32,
                    has_bias: req.attn_bias.is_some() as u32,
                    n_kv_heads: req.n_kv_heads as u32,
                    wait_base: 0,
                    rope_norm: req.rope_norm as u32,
                    pad0: 0,
                };
                enc.set_bytes(
                    15,
                    std::mem::size_of::<FqPrepArgs>() as u64,
                    &fargs as *const FqPrepArgs as *const _,
                );
                let ones = [1.0f32; ATTN_HEAD_DIM];
                let qw = req.q_norm.unwrap_or(&ones);
                let kw = req.k_norm.unwrap_or(&ones);
                enc.set_bytes(16, (hd * 4) as u64, qw.as_ptr() as *const _);
                enc.set_bytes(17, (hd * 4) as u64, kw.as_ptr() as *const _);
                enc.set_buffer(18, Some(&gpu.x_arena), e(ropes_at));
                match &bias_refs {
                    Some((qb, kb, vb)) => {
                        enc.set_buffer(19, Some(&gpu.chunks[qb.chunk].buf), qb.w_off);
                        enc.set_buffer(20, Some(&gpu.chunks[kb.chunk].buf), kb.w_off);
                        enc.set_buffer(21, Some(&gpu.chunks[vb.chunk].buf), vb.w_off);
                    }
                    None => {
                        for index in 19u64..=21 {
                            enc.set_buffer(index, Some(&req.cache.buf), 0);
                        }
                    }
                }
                enc.set_buffer(22, Some(&gpu.pf_attn_sync), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(req.n_heads as u64, (req.m as u64).div_ceil(8), 1),
                    MTLSize::new(128, 1, 1),
                );
                cstamp!(gpu, enc, "attend");
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                }
                split_here!("attend");
            }
            if !fused_qk && !pf_skip("attend") {
                enc.set_compute_pipeline_state(&prep_state);
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
                    pos: req.base as u32,
                    has_qk_norm: req.q_norm.is_some() as u32,
                    rot_dim: req.rot_dim as u32,
                    k_base: req.k_off as u64,
                    v_base: req.v_off as u64,
                    has_bias: req.attn_bias.is_some() as u32,
                    rope_norm: req.rope_norm as u32,
                };
                enc.set_bytes(
                    4,
                    std::mem::size_of::<QkPrepArgs>() as u64,
                    &args as *const QkPrepArgs as *const _,
                );
                let ones = [1.0f32; ATTN_HEAD_DIM];
                let qw = req.q_norm.unwrap_or(&ones);
                let kw = req.k_norm.unwrap_or(&ones);
                enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
                enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
                enc.set_buffer(7, Some(&gpu.x_arena), e(ropes_at));
                // Bias buffers: resolved F32 regions for GLM, any valid buffer
                // otherwise (the kernel skips them on has_bias == 0).
                match &bias_refs {
                    Some((qb, kb, vb)) => {
                        enc.set_buffer(8, Some(&gpu.chunks[qb.chunk].buf), qb.w_off);
                        enc.set_buffer(9, Some(&gpu.chunks[kb.chunk].buf), kb.w_off);
                        enc.set_buffer(10, Some(&gpu.chunks[vb.chunk].buf), vb.w_off);
                    }
                    None => {
                        for index in 8u64..=10 {
                            enc.set_buffer(index, Some(&req.cache.buf), 0);
                        }
                    }
                }
                enc.dispatch_thread_groups(
                    MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, req.m as u64, 1),
                    MTLSize::new(32, 1, 1),
                );
                cstamp!(gpu, enc, "prep");
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                }
                split_here!("prep");

                // Causal attention over every row in ONE dispatch: rows in grid.y,
                // each over its own prefix.
                enc.set_compute_pipeline_state(&attend_state);
                enc.set_buffer(0, Some(&req.cache.buf), 0);
                enc.set_buffer(1, Some(&req.cache.buf), 0);
                enc.set_buffer(2, Some(y), e(q_at));
                enc.set_buffer(3, Some(y), e(attn_at));
                for (index, value) in [
                    (4u64, req.kv_dim as u32),
                    (5, hd as u32),
                    (6, req.base as u32),
                    (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
                ] {
                    enc.set_bytes(index, 4, &value as *const u32 as *const _);
                }
                enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                for (index, value) in [(9u64, req.k_off as u64), (10, req.v_off as u64)] {
                    enc.set_bytes(index, 8, &value as *const u64 as *const _);
                }
                let heads32 = req.n_heads as u32;
                enc.set_bytes(11, 4, &heads32 as *const u32 as *const _);
                if attend_mm() {
                    // MMA tiles: same bindings as attend_rows, 8 rows per
                    // threadgroup.
                    enc.set_compute_pipeline_state(&gpu.pipelines[&(attend_mm_kernel(), 1, 1)]);
                    let rows32 = req.m as u32;
                    enc.set_bytes(12, 4, &rows32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(req.n_heads as u64, (req.m as u64).div_ceil(8), 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else {
                    let tile = match attend_rows_kernel() {
                        "attend_rows_t8" => 8u64,
                        "attend_rows_t4" => 4,
                        _ => 1,
                    };
                    if tile > 1 {
                        let rows32 = req.m as u32;
                        enc.set_bytes(12, 4, &rows32 as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(req.n_heads as u64, (req.m as u64).div_ceil(tile), 1),
                            MTLSize::new(128, 1, 1),
                        );
                    } else {
                        enc.dispatch_thread_groups(
                            MTLSize::new(req.n_heads as u64, req.m as u64, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    }
                }
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[y]);
                }
                cstamp!(gpu, enc, "attend");
                split_here!("attend");
            }
        }

        if !pf_skip("wo") {
            matmul(
                enc,
                &states[3],
                &mats[3],
                q_dim,
                hidden,
                &gpu.y_arena,
                attn_at,
                out_at,
            );
        }
        cstamp!(gpu, enc, "wo");
        split_here!("wo");
        if let Some(f) = &fusion {
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
            // xs += wo output, then the FFN norm - one fused pass - and the
            // router matmul, all before the single wait.
            if !pf_skip("fnorm") {
                norm_rows(enc, &f.ffn_norm, 1, out_at);
            }
            cstamp!(gpu, enc, "fnorm");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
            }
            if let Some((router, router_state)) = &f.router {
                matmul(
                    enc,
                    router_state,
                    router,
                    hidden,
                    f.n_expert,
                    &gpu.pf_hs,
                    0,
                    router_at,
                );
                cstamp!(gpu, enc, "rmm");
            }
            split_here!("ffnnorm_router");
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        if onebuf {
            // Nothing: the chunk commits once, in prefill_end.
        } else if defer {
            gpu.commit_chained(cmd);
        } else if chain {
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
            use objc::{msg_send, sel, sel_impl};
            let mut prev_ke: Option<f64> = None;
            for (label, c) in &staged {
                let c: &metal::CommandBufferRef = c;
                let ks: f64 = unsafe { msg_send![c, kernelStartTime] };
                let ke: f64 = unsafe { msg_send![c, kernelEndTime] };
                let gap = prev_ke.map_or(0.0, |p| (ks - p).max(0.0) * 1e3);
                eprintln!("attn {}: {:.3} ms (gap {:.3})", label, (ke - ks) * 1e3, gap);
                prev_ke = Some(ke);
            }
            let ks: f64 = unsafe { msg_send![cmd, kernelStartTime] };
            let ke: f64 = unsafe { msg_send![cmd, kernelEndTime] };
            let gap = prev_ke.map_or(0.0, |p| (ks - p).max(0.0) * 1e3);
            eprintln!("attn tail: {:.3} ms (gap {:.3})", (ke - ks) * 1e3, gap);
        } else {
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
        }
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !defer && !onebuf && policy.ffn_time {
            use objc::{msg_send, sel, sel_impl};
            let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
            let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
            eprintln!("attbuf: {:.3} ms", (ge - gs) * 1e3);
        }
        if !onebuf {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        DISPATCHES.fetch_add(6, Ordering::Relaxed);

        // Fused: the caller needs only the router logits; the projected
        // rows already live in pf_x. Plain: the projected rows come back.
        // GPU routing leaves the logits in y_arena and reports the offset.
        if fusion.is_some() && gpu_route() {
            PF_ROUTER_AT.store(router_at as u64, Ordering::Relaxed);
            return Vec::new();
        }
        let (src_at, out_len) = if fusion.is_some() {
            (router_at, req.m * n_expert)
        } else {
            (out_at, req.m * hidden)
        };
        let mut out = vec![0f32; out_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const f32).add(src_at),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    });
    Some(out)
}

pub fn prefill_gdn_block(req: &PrefillGdnReq) -> Option<Vec<f32>> {
    let hidden = req.hidden;
    let key_dim = req.heads_k * req.d;
    let value_dim = req.heads_v * req.d;
    let channels = key_dim * 2 + value_dim;
    // The recurrence kernel keeps one state row per simdgroup: 32 lanes x 4
    // columns, so d must be 128. Short chunks take the CPU path (the mm
    // pipeline refuses them anyway).
    if req.d != 128
        || req.wqkv.2 != channels
        || req.zgate.2 != value_dim
        || req.ssm_out.2 != hidden
        || req.m < MM_MIN_M
        || req.alpha.len() != hidden * req.heads_v * 4
        || req.beta.len() != hidden * req.heads_v * 4
        || req.conv1d.len() != channels * req.d_conv * 4
        || req.a.len() != req.heads_v
        || req.dt.len() != req.heads_v
        || req.ssm_norm.len() != req.d
    {
        return None;
    }
    let conv_span = req.conv_off + (req.d_conv - 1) * channels;
    let state_span = req.state_off + req.heads_v * req.d * req.d;
    if conv_span * 4 > req.ssm.len || state_span * 4 > req.ssm.len {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    let mats = [
        resolve(&gpu, req.wqkv.0, req.wqkv.1)?,
        resolve(&gpu, req.zgate.0, req.zgate.1)?,
        resolve_f32(&gpu, req.alpha, hidden)?,
        resolve_f32(&gpu, req.beta, hidden)?,
        resolve(&gpu, req.ssm_out.0, req.ssm_out.1)?,
    ];
    let mut states = Vec::with_capacity(5);
    for mat in &mats {
        states.push(
            gpu.pipeline(mm_kernel_for(mat.ty, req.m)?.0, 1, 1)?
                .to_owned(),
        );
    }
    let (qkvz_state, qkvz_pad) = if channels % 64 == 0 && value_dim % 64 == 0 {
        if mats[0].ty == mats[1].ty && (mats[0].ty == GgmlType::Q8_0 || hidden % 256 == 0) {
            (
                mm_qkv_kernel_for(mats[0].ty)
                    .and_then(|name| gpu.pipeline(name, 1, 1).map(|p| p.to_owned())),
                0u32,
            )
        } else if mm_ll() && !mm_k64() && mm_pipe() && !crate::runtime::get().no_fused_qkv {
            match (
                super::decode::dq_fmt(mats[0].ty),
                super::decode::dq_fmt(mats[1].ty),
            ) {
                (Some(fq), Some(fk))
                    if fq <= 5 && fk <= 5 && ((fq == 3 && fk == 3) || hidden % 256 == 0) =>
                {
                    (
                        gpu.pipeline("mmllp_qkv_mixed", 1, 1).map(|p| p.to_owned()),
                        fq | (fk << 8),
                    )
                }
                _ => (None, 0),
            }
        } else {
            (None, 0)
        }
    } else {
        (None, 0)
    };
    let conv1d = {
        let addr = req.conv1d.as_ptr() as usize;
        if addr % 4 != 0 {
            return None;
        }
        let chunk = gpu.chunk_for(addr, req.conv1d.len())?;
        NormRef {
            chunk,
            off: (addr - gpu.chunks[chunk].start) as u64,
        }
    };

    struct FusionRefs {
        attn_norm: NormRef,
        ffn_norm: NormRef,
        router: Option<(MatRef, ComputePipelineState)>,
        n_expert: usize,
    }
    let fusion = match &req.fusion {
        Some(f) => {
            if gpu.pf_x_cap < req.m * hidden * 4 || gpu.pf_hs_cap < req.m * hidden * 4 {
                return None;
            }
            let attn_norm = resolve_norm(&gpu, f.attn_norm, hidden)?;
            let ffn_norm = resolve_norm(&gpu, f.ffn_norm, hidden)?;
            let router = if f.n_expert > 0 {
                let router = resolve_f32(&gpu, f.router, hidden)?;
                let router_state = gpu
                    .pipeline(mm_kernel_for(GgmlType::F32, req.m)?.0, 1, 1)?
                    .to_owned();
                Some((router, router_state))
            } else {
                None
            };
            Some(FusionRefs {
                attn_norm,
                ffn_norm,
                router,
                n_expert: f.n_expert,
            })
        }
        None => None,
    };
    // One-buffer note: same as prefill_attn_block - no decline here;
    // a non-onebuf layer seals the open segment at the encode site below.
    let norm_state = gpu.pipelines[&("norm_rows", 1, 1)].to_owned();
    let conv_state = gpu.pipelines[&("gdn_conv_batch", 1, 1)].to_owned();
    let step_state = gpu.pipelines[&("gdn_step_batch", 1, 1)].to_owned();
    let norm_out_state = gpu.pipelines[&("gdn_out_norm_batch", 1, 1)].to_owned();

    // y layout: raw qkv rows, conv output, z gate, alpha|beta, deltanet out,
    // the ssm_out projection, the router logits.
    let align = |n: usize| (n + 63) & !63;
    let qkv_at = 0usize;
    let qkc_at = qkv_at + align(req.m * channels);
    let z_at = qkc_at + align(req.m * channels);
    let ab_at = z_at + align(req.m * value_dim);
    let gy_at = ab_at + align(2 * req.m * req.heads_v);
    let out_at = gy_at + align(req.m * value_dim);
    let router_at = out_at + align(req.m * hidden);
    let n_expert = fusion.as_ref().map_or(0, |f| f.n_expert);
    let window_at = router_at + align(req.m * n_expert);
    gpu.ensure_arenas(4096, (window_at + align((req.d_conv - 1) * channels)) * 4);

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let queue = gpu.queue.clone();
        let mut cmd = queue.new_command_buffer();
        let split = crate::runtime::get().pf_split;
        // One-buffer segment: lazily armed on the first eligible layer,
        // sealed by a layer that stays off it (see prefill_attn_block).
        let onebuf = fusion.is_some() && gpu_route() && pf_onebuf() && !split;
        if !onebuf {
            gpu.pf_obuf_seal();
        } else {
            gpu.pf_obuf_arm();
        }
        let defer = !onebuf && fusion.is_some() && gpu_route() && pf_defer() && !split;
        if defer {
            cmd.encode_wait_for_event(&gpu.pf_ev, gpu.pf_ev_val);
        }
        let mut enc = if onebuf {
            gpu.pf_obuf_cmd
                .as_deref()
                .expect("onebuf armed")
                .compute_command_encoder_with_dispatch_type(serial_dispatch())
        } else {
            cmd.compute_command_encoder_with_dispatch_type(serial_dispatch())
        };
        macro_rules! split_here {
            ($label:expr) => {
                if split {
                    use objc::{msg_send, sel, sel_impl};
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
                    let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
                    eprintln!("gdn {}: {:.3} ms", $label, (ge - gs) * 1e3);
                    cmd = queue.new_command_buffer();
                    enc = cmd.compute_command_encoder_with_dispatch_type(
                        metal::MTLDispatchType::Concurrent,
                    );
                }
            };
        }
        let y = &gpu.y_arena;
        let e = |off: usize| (off * 4) as u64;

        let matmul = |enc: &metal::ComputeCommandEncoderRef,
                      state: &ComputePipelineState,
                      mat: &MatRef,
                      n_in: usize,
                      n_out: usize,
                      x_buf: &Buffer,
                      x_off: usize,
                      y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            let m32 = req.m as u32;
            enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
            let (bm, bn) = mm_kernel_for(mat.ty, req.m).map_or((32, 32), |(_, bm, bn)| (bm, bn));
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (n_out as u64).div_ceil(bm as u64),
                    (req.m as u64).div_ceil(bn as u64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        };
        let norm_rows = |enc: &metal::ComputeCommandEncoderRef,
                         norm: &NormRef,
                         use_delta: u32,
                         delta_off: usize| {
            enc.set_compute_pipeline_state(&norm_state);
            enc.set_buffer(0, Some(&gpu.pf_x), 0);
            enc.set_buffer(1, Some(y), e(delta_off));
            enc.set_buffer(2, Some(&gpu.pf_hs), 0);
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.set_bytes(6, 4, &use_delta as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(req.m as u64, 1, 1), MTLSize::new(256, 1, 1));
        };

        let f = fusion.as_ref().expect("gdn prefill without fusion");
        if onebuf {
            // The previous layer's stages live on the same command buffer.
            enc.memory_barrier_with_resources(&[
                y,
                &gpu.pf_x,
                &gpu.pf_hs,
                &gpu.route_buf,
                &req.ssm.buf,
            ]);
        }
        // h = rmsnorm(xs) * attn_norm, then the four projections off it.
        // The previous layer's fused combine may already have written those
        // rows (combine_rows_resnorm).
        if gpu.pf_hs_normed.take() {
            cstamp!(gpu, enc, "anorm_fused");
        } else {
            norm_rows(enc, &f.attn_norm, 0, out_at);
            cstamp!(gpu, enc, "anorm");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
            }
        }
        if let Some(qkvz_pipe) = &qkvz_state {
            enc.set_compute_pipeline_state(qkvz_pipe);
            enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
            enc.set_buffer(1, Some(&gpu.chunks[mats[1].chunk].buf), 0);
            enc.set_buffer(2, Some(&gpu.chunks[mats[1].chunk].buf), 0);
            enc.set_buffer(3, Some(&gpu.pf_hs), 0);
            enc.set_buffer(4, Some(y), 0);
            let qkv_args = MmQkvArgs {
                n_in: hidden as u32,
                n_q: channels as u32,
                n_k: value_dim as u32,
                n_v: 0,
                wq_off: mats[0].w_off,
                wk_off: mats[1].w_off,
                wv_off: 0,
                q_off: e(qkv_at),
                k_off: e(z_at),
                v_off: 0,
                m: req.m as u32,
                pad: qkvz_pad,
            };
            enc.set_bytes(
                5,
                std::mem::size_of::<MmQkvArgs>() as u64,
                &qkv_args as *const MmQkvArgs as *const _,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(
                    ((channels + value_dim) as u64).div_ceil(64),
                    (req.m as u64).div_ceil(32),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
            cstamp!(gpu, enc, "wqkv");
        } else {
            matmul(
                enc, &states[0], &mats[0], hidden, channels, &gpu.pf_hs, 0, qkv_at,
            );
            cstamp!(gpu, enc, "wqkv");
            matmul(
                enc, &states[1], &mats[1], hidden, value_dim, &gpu.pf_hs, 0, z_at,
            );
        }
        matmul(
            enc,
            &states[2],
            &mats[2],
            hidden,
            req.heads_v,
            &gpu.pf_hs,
            0,
            ab_at,
        );
        matmul(
            enc,
            &states[3],
            &mats[3],
            hidden,
            req.heads_v,
            &gpu.pf_hs,
            0,
            ab_at + req.m * req.heads_v,
        );
        split_here!("qkv");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }

        // Depthwise conv over the chunk, updating the session window.
        enc.set_compute_pipeline_state(&conv_state);
        enc.set_buffer(0, Some(y), e(qkv_at));
        enc.set_buffer(1, Some(y), e(qkc_at));
        enc.set_buffer(2, Some(&req.ssm.buf), 0);
        enc.set_buffer(3, Some(&gpu.chunks[conv1d.chunk].buf), conv1d.off);
        let cargs = GdnConvBatchArgs {
            channels: channels as u32,
            d_conv: req.d_conv as u32,
            m: req.m as u32,
            pad0: 0,
            conv_off: req.conv_off as u64,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GdnConvBatchArgs>() as u64,
            &cargs as *const GdnConvBatchArgs as *const _,
        );
        let (slots_buf, slot_total) = match req.ssm_slots {
            Some((r, total)) => (r, total as u32),
            None => (req.ssm, 0u32),
        };
        enc.set_buffer(5, Some(&slots_buf.buf), 0);
        enc.set_bytes(6, 4, &slot_total as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((channels as u64).div_ceil(256), req.m as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        cstamp!(gpu, enc, "conv");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y, &req.ssm.buf]);
        }
        // The session window update, now that every row's read is done: the
        // last d_conv-1 RAW qkv rows become the next chunk's window.
        enc.set_compute_pipeline_state(&gpu.pipelines[&("copy_f32", 1, 1)]);
        let window_rows = req.d_conv - 1;
        let window_src = if req.m >= window_rows {
            qkv_at + (req.m - window_rows) * channels
        } else {
            let old = (window_rows - req.m) * channels;
            enc.set_buffer(
                0,
                Some(&req.ssm.buf),
                ((req.conv_off + req.m * channels) * 4) as u64,
            );
            enc.set_buffer(1, Some(y), e(window_at));
            let old32 = old as u32;
            enc.set_bytes(2, 4, &old32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((old as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.set_buffer(0, Some(y), e(qkv_at));
            enc.set_buffer(1, Some(y), e(window_at + old));
            let new32 = (req.m * channels) as u32;
            enc.set_bytes(2, 4, &new32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((new32 as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.memory_barrier_with_resources(&[y, &req.ssm.buf]);
            window_at
        };
        enc.set_buffer(0, Some(y), e(window_src));
        enc.set_buffer(1, Some(&req.ssm.buf), (req.conv_off * 4) as u64);
        let nwin = (window_rows * channels) as u32;
        enc.set_bytes(2, 4, &nwin as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((nwin as u64).div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y, &req.ssm.buf]);
        }
        split_here!("conv");

        // The recurrence over the chunk, state updated in place.
        enc.set_compute_pipeline_state(&step_state);
        enc.set_buffer(0, Some(&req.ssm.buf), 0);
        enc.set_buffer(1, Some(y), e(qkc_at));
        enc.set_buffer(2, Some(y), e(ab_at));
        enc.set_buffer(3, Some(y), e(ab_at + req.m * req.heads_v));
        let sargs = GdnStepBatchArgs {
            heads_k: req.heads_k as u32,
            heads_v: req.heads_v as u32,
            d: req.d as u32,
            key_dim: key_dim as u32,
            m: req.m as u32,
            eps: req.eps,
            state_off: req.state_off as u64,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GdnStepBatchArgs>() as u64,
            &sargs as *const GdnStepBatchArgs as *const _,
        );
        enc.set_bytes(5, (req.a.len() * 4) as u64, req.a.as_ptr() as *const _);
        enc.set_bytes(6, (req.dt.len() * 4) as u64, req.dt.as_ptr() as *const _);
        enc.set_buffer(7, Some(y), e(gy_at));
        enc.set_buffer(8, Some(&slots_buf.buf), 0);
        enc.set_bytes(9, 4, &slot_total as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((req.d / 4) as u64, req.heads_v as u64, 1),
            MTLSize::new(32, 4, 1),
        );
        cstamp!(gpu, enc, "step");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y, &req.ssm.buf]);
        }
        split_here!("step");

        // Gated rmsnorm over the head outputs, times silu(z).
        enc.set_compute_pipeline_state(&norm_out_state);
        enc.set_buffer(0, Some(y), e(gy_at));
        enc.set_buffer(1, Some(y), e(z_at));
        enc.set_bytes(
            2,
            (req.ssm_norm.len() * 4) as u64,
            req.ssm_norm.as_ptr() as *const _,
        );
        let hv = req.heads_v as u32;
        let dd = req.d as u32;
        enc.set_bytes(3, 4, &hv as *const u32 as *const _);
        enc.set_bytes(4, 4, &dd as *const u32 as *const _);
        enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(req.heads_v as u64, req.m as u64, 1),
            MTLSize::new(req.d as u64, 1, 1),
        );
        cstamp!(gpu, enc, "onorm");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        split_here!("onorm");

        matmul(
            enc, &states[4], &mats[4], value_dim, hidden, y, gy_at, out_at,
        );
        cstamp!(gpu, enc, "wo");
        split_here!("wo");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        // xs += ssm_out output, then the FFN norm and the router matmul -
        // one fused dispatch when the router is present.
        norm_rows(enc, &f.ffn_norm, 1, out_at);
        cstamp!(gpu, enc, "fnorm");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
        }
        if let Some((router, router_state)) = &f.router {
            matmul(
                enc,
                router_state,
                router,
                hidden,
                f.n_expert,
                &gpu.pf_hs,
                0,
                router_at,
            );
            cstamp!(gpu, enc, "rmm");
        }
        split_here!("ffnnorm_router");
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        if onebuf {
            // Nothing: the chunk commits once, in prefill_end.
        } else if defer {
            gpu.commit_chained(cmd);
        } else {
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
        }
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !onebuf {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        DISPATCHES.fetch_add(8, Ordering::Relaxed);

        // Same contract as the attention block: GPU routing leaves the
        // logits in y_arena and reports the offset; a CPU-routing caller
        // gets them downloaded.
        if fusion.is_some() && gpu_route() {
            PF_ROUTER_AT.store(router_at as u64, Ordering::Relaxed);
            return Vec::new();
        }
        let mut out = vec![0f32; req.m * n_expert];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const f32).add(router_at),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    });
    Some(out)
}
