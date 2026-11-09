//! Grouped MoE feed-forward for prefill: every routed expert of a chunk in
//! one buffer, with optional on-device routing, shared expert and combine.

use super::*;

pub fn ffn_batch_grouped(req: &GroupedFfnReq) -> Option<Vec<f32>> {
    let route_mode = req.route.is_some();
    if route_mode {
        let r = req.route.as_ref()?;
        // route_pick holds 8 candidates per token and the fused buffers are
        // the only ones it can read logits from; anything else keeps CPU
        // routing. total_rows is exactly m * n_used in this mode.
        if r.n_used > 8 || req.fused.as_ref().map_or(0, |c| c.m) * r.n_used != req.total_rows {
            return None;
        }
    } else if req.groups.is_empty() || req.tok.len() != req.total_rows {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    // One-buffer note: a non-routing FFN (a leading Dense layer, GLM) no
    // longer declines the chunk - it seals the open one-buffer segment
    // at the encode site below and runs fused on an event-chained buffer
    // of its own; the next MoE layer re-arms.
    if req.shared.is_some() && req.fused.is_none() {
        // The shared expert reads pf_hs and lands in the combine region;
        // both exist only on the fused path. Callers on the CPU path run
        // the shared expert themselves.
        return None;
    }
    let mats = [
        resolve(&gpu, req.gate.0, req.gate.1)?,
        resolve(&gpu, req.up.0, req.up.1)?,
        resolve(&gpu, req.down.0, req.down.1)?,
    ];
    let m_fused = req.fused.as_ref().map_or(0, |c| c.m);
    let sh_mats = match &req.shared {
        Some(sh) => Some([
            resolve(&gpu, sh.gate.0, sh.gate.1)?,
            resolve(&gpu, sh.up.0, sh.up.1)?,
            resolve(&gpu, sh.down.0, sh.down.1)?,
        ]),
        None => None,
    };
    // qwen35moe's shared-expert gate projection (F32 [hidden]); the patch
    // kernel after route_scatter folds its sigmoid into the CSR weights.
    let sh_gate_out = match &req.shared {
        Some(sh) => match &sh.gate_out {
            Some((ty, bytes)) => {
                if *ty != GgmlType::F32 || bytes.len() != req.hidden * 4 {
                    return None;
                }
                let mat = resolve_f32(&gpu, bytes, req.hidden)?;
                let state = gpu
                    .pipeline(mm_kernel_for(GgmlType::F32, m_fused.max(MM_MIN_M))?.0, 1, 1)?
                    .to_owned();
                Some((mat, state))
            }
            None => None,
        },
        None => None,
    };
    let strides = [
        (req.gate.1.len() / req.n_expert) as u64,
        (req.up.1.len() / req.n_expert) as u64,
        (req.down.1.len() / req.n_expert) as u64,
    ];
    let n_active_experts = if route_mode {
        req.n_expert.max(1)
    } else {
        req.groups.len().max(1)
    };
    let avg_rows = req.total_rows / n_active_experts;
    let bn16 = mmid_use_bn16(avg_rows);
    let bn = if bn16 { 16usize } else { 32usize };
    let mut states = Vec::with_capacity(3);
    let mut bms = [32u64; 3];
    let mut fused_swiglu = false;
    for (i, mat) in mats.iter().enumerate() {
        let (name, bm) = if i == 2 {
            match mmid_swiglu_kernel_for(mat.ty, bn16) {
                Some(nb) => {
                    fused_swiglu = true;
                    nb
                }
                None => mmid_kernel_for(mat.ty, false, bn16)?,
            }
        } else {
            mmid_kernel_for(mat.ty, true, bn16)?
        };
        bms[i] = bm as u64;
        states.push(gpu.pipeline(name, 1, 1)?.to_owned());
    }
    // Plain tile-matmul pipelines for the shared expert (no gather).
    let mut sh_states = Vec::with_capacity(3);
    if let Some(sh_mats) = &sh_mats {
        for mat in sh_mats {
            sh_states.push(
                gpu.pipeline(mm_kernel_for(mat.ty, m_fused)?.0, 1, 1)?
                    .to_owned(),
            );
        }
    }
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();

    // Fused gate+up as ONE dual-weight dispatch: same expert stride, 64-wide
    // tiles never straddle the gate/up boundary (ffn % 64 == 0), q4_k only.
    // Measured SLOWER than the two-dispatch path on M4 Max (317 vs 331 tok/s
    // pp480, GLM-4.5-Air) - one kernel doing 2x work per threadgroup loses to
    // two back-to-back dispatches. Kept behind ALLPAKA_DUAL=1 for reference.
    let dual_state = if !bn16
        && fused_swiglu
        && strides[0] == strides[1]
        && req.ffn % 64 == 0
        && std::env::var_os("ALLPAKA_DUAL").is_some()
    {
        gpu.pipeline("mmllgd_id_q4_k", 1, 1).map(|p| p.to_owned())
    } else {
        None
    };
    let dual = dual_state.is_some();

    let use_route_fused = route_mode
        && req.n_expert <= 256
        && std::env::var_os("ALLPAKA_NO_ROUTE_FUSED").is_none();
    // Tile-descriptor grid (llama-style): route_fused emits one MmGroup per
    // non-empty tile_w-row chunk (32, or 16 for bn16), so the grid covers
    // real work exactly - at pp480/128exp this is ~250 z-slices instead of
    // 15 row-tiles x 128 experts of mostly empty threadgroups. The chunk
    // width must equal the consumer kernel's row stride.
    // ALLPAKA_MMID_TILES=0 reverts.
    let tiles_mode = use_route_fused && crate::runtime::get().mmid_tiles;
    let tile_w: usize = if bn16 { 16 } else { 32 };
    let (row_tiles, n_groups) = if tiles_mode {
        (1, (req.n_expert + req.total_rows / tile_w) as u64)
    } else if route_mode {
        // Worst-case grid: measured FASTER than a tight one - empty
        // threadgroups retire for free, while a small grid serialises the
        // hot experts through the striding loop. ALLPAKA_MMID_RTILES pins
        // the grid for A/B tests.
        let auto = (m_fused as u64).div_ceil(bn as u64).max(1);
        let rt = std::env::var("ALLPAKA_MMID_RTILES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(auto, |n| n.max(1));
        (rt, req.n_expert as u64)
    } else {
        let max_rows = req.groups.iter().map(|g| g[2]).max()? as usize;
        (max_rows.div_ceil(bn) as u64, req.groups.len() as u64)
    };

    // Arenas: x rows then the group table in x; gate, up, down out in y.
    // GPU routing keeps the table, gather and CSR in route_buf instead.
    let align = |n: usize| (n + 63) & !63;
    let x_at = 0usize;
    let x_len = if req.fused.is_some() { 0 } else { req.x.len() };
    let table_at = x_at + align(x_len);
    let tok_at = table_at + align(if route_mode { 0 } else { req.groups.len() * 3 });
    let off_at = tok_at + align(if route_mode { 0 } else { req.total_rows });
    let (n_off, n_hits) = if route_mode {
        (0, 0)
    } else {
        req.fused
            .as_ref()
            .map_or((0, 0), |c| (c.tok_off.len(), c.hit_row.len()))
    };
    let hrow_at = off_at + align(n_off);
    let hw_at = hrow_at + align(n_hits);
    let x_total = hw_at + align(n_hits);
    let gate_at = 0usize;
    let up_at = gate_at + align(req.total_rows * req.ffn);
    // Dual dispatch writes gate and up interleaved per row (stride 2*ffn);
    // the separate up region exists only in the two-dispatch fallback.
    let out_at = if dual {
        gate_at + align(req.total_rows * 2 * req.ffn)
    } else {
        up_at + align(req.total_rows * req.ffn)
    };
    // The shared expert's down rows live right after the expert down rows so
    // the combine can reference them as ordinary hit rows.
    let sh_rows = if req.shared.is_some() { m_fused } else { 0 };
    let sh_ffn = req.shared.as_ref().map_or(0, |s| s.ffn);
    let shg_at = out_at + align((req.total_rows + sh_rows) * req.hidden);
    let shu_at = shg_at + align(m_fused * sh_ffn);
    // qwen35moe's shared-expert gate logits, one float per token.
    let shw_at = shu_at + align(m_fused * sh_ffn);
    let y_total = shw_at + align(m_fused);
    // GPU routing: size route_buf, then rescue the logits out of y_arena
    // BEFORE ensure_arenas can reallocate it (steady state binds y_arena
    // directly - zero copies). Counts/counters must start at zero.
    let rl = req
        .route
        .as_ref()
        .map(|r| {
            route_layout(
                m_fused,
                req.n_expert,
                r.n_used,
                req.shared.is_some(),
                if bn16 { 16 } else { 32 },
            )
        });
    let mut logits_at: Option<(&'static str, usize)> = None;
    // One-buffer rescue: (old y_arena, logits offset, logits target, count).
    let mut obuf_rescue: Option<(Buffer, usize, usize, usize)> = None;
    if let Some(rl) = &rl {
        gpu.ensure_route(rl.total);
        let r_at = PF_ROUTER_AT.load(Ordering::Relaxed) as usize;
        if r_at == usize::MAX {
            return None; // the attention block did not leave logits behind
        }
        let n_logits = m_fused * req.n_expert;
        // Counts/counters self-clean on the GPU (route_scan zeroes them
        // after every layer), so staging once per allocation/bias is enough;
        // per-layer CPU writes would race the deferred buffers.
        let bptr = req
            .route
            .as_ref()
            .and_then(|r| r.bias)
            .map_or(0, |b| b.as_ptr() as usize);
        if gpu.route_staged != Some(bptr) {
            unsafe {
                let rb = gpu.route_buf.contents() as *mut f32;
                std::ptr::write_bytes(rb, 0, rl.table); // counts + counters
                if let Some(b) = req.route.as_ref().and_then(|r| r.bias) {
                    std::ptr::copy_nonoverlapping(b.as_ptr(), rb.add(rl.bias), req.n_expert);
                }
            }
            gpu.route_staged = Some(bptr);
        }
        if y_total * 4 > gpu.y_cap {
            if gpu.pf_obuf_cmd.is_some() && route_mode {
                // One-buffer mode: nothing has committed yet, a CPU
                // readback would see stale data. Rescue on the GPU instead,
                // encoded in order at the head of the FFN stages.
                obuf_rescue = Some((gpu.y_arena.clone(), r_at, rl.logits, n_logits));
                logits_at = Some(("route", rl.logits));
            } else {
                // The logits leave y_arena before it can grow; with deferred
                // commits the buffer that wrote them may still be in flight.
                gpu.prefill_drain();
                unsafe {
                    let rb = gpu.route_buf.contents() as *mut f32;
                    let src = (gpu.y_arena.contents() as *const f32).add(r_at);
                    std::ptr::copy_nonoverlapping(src, rb.add(rl.logits), n_logits);
                }
                logits_at = Some(("route", rl.logits));
            }
        }
        if logits_at.is_none() {
            logits_at = Some(("y", r_at));
        }
    }
    gpu.ensure_arenas(x_total * 4, y_total * 4);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut f32;
        if req.fused.is_none() {
            std::ptr::copy_nonoverlapping(req.x.as_ptr(), xp.add(x_at), req.x.len());
        }
        if !route_mode {
            std::ptr::copy_nonoverlapping(
                req.groups.as_ptr() as *const f32,
                xp.add(table_at),
                req.groups.len() * 3,
            ); // [u32;3] is layout-guaranteed; the arena type is just bytes.
            std::ptr::copy_nonoverlapping(
                req.tok.as_ptr() as *const f32,
                xp.add(tok_at),
                req.tok.len(),
            );
            if let Some(c) = &req.fused {
                std::ptr::copy_nonoverlapping(
                    c.tok_off.as_ptr() as *const f32,
                    xp.add(off_at),
                    c.tok_off.len(),
                );
                std::ptr::copy_nonoverlapping(
                    c.hit_row.as_ptr() as *const f32,
                    xp.add(hrow_at),
                    c.hit_row.len(),
                );
                std::ptr::copy_nonoverlapping(c.hit_w.as_ptr(), xp.add(hw_at), c.hit_w.len());
            }
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        // ALLPAKA_FFN_SPLIT=1: commit after every stage and print its GPU
        // time. Debug-only; serialises the stages, so wall time degrades.
        let split = std::env::var_os("ALLPAKA_FFN_SPLIT").is_some();
        // ALLPAKA_PF_CHAIN=1 (shared with the attention block): per-stage
        // commits queued back-to-back without waits - per-stage GPU times
        // without the DVFS sag of the serialising FFN_SPLIT.
        let chain = !split && std::env::var_os("ALLPAKA_PF_CHAIN").is_some();
        // One-buffer chunk (ALLPAKA_PF_ONEBUF=1, default): encode into the
        // chunk's shared command buffer, lazily armed by the first
        // eligible layer; the commit happens once in prefill_end. A layer
        // that stays off the shared buffer (CPU routing, FFN_SPLIT) seals
        // the open segment first, so its own buffer stays ordered after
        // the work already encoded there.
        let onebuf = (route_mode || (dense_obuf() && req.route.is_none() && req.fused.is_some()))
            && pf_onebuf()
            && !split
            && !chain;
        if !onebuf {
            gpu.pf_obuf_seal();
        } else {
            gpu.pf_obuf_arm();
        }
        // Deferred commit: chain behind the previous layer's buffer through
        // the shared event instead of a CPU wait. Any fused-path buffer
        // qualifies - it reads nothing back (the combine lands in pf_x).
        // FFN_SPLIT keeps the old waits.
        let defer = !onebuf && req.fused.is_some() && pf_defer() && !split && !chain;
        // Cloned queue handle: command buffers borrow it, leaving `gpu`
        // free for the deferred commit bookkeeping.
        let queue = gpu.queue.clone();
        let cmd0 = queue.new_command_buffer();
        let mut cmd = cmd0;
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
                        eprintln!("ffn {}: {:.3} ms", $label, (ge - gs) * 1e3);
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
        if onebuf {
            // The attention stages of this layer live on the same command
            // buffer: order the router logits (y), the normed activations
            // (pf_hs) and the residual stream (pf_x) against the FFN stages.
            match &obuf_rescue {
                Some((oy, ..)) => enc.memory_barrier_with_resources(&[
                    y,
                    &gpu.pf_x,
                    &gpu.pf_hs,
                    &gpu.route_buf,
                    oy,
                ]),
                None => {
                    enc.memory_barrier_with_resources(&[y, &gpu.pf_x, &gpu.pf_hs, &gpu.route_buf])
                }
            }
            if let Some((oy, r_at, lg_off, n_logits)) = &obuf_rescue {
                // GPU-side logits rescue out of the pre-growth y_arena.
                enc.set_compute_pipeline_state(&gpu.pipelines[&("copy_f32", 1, 1)]);
                enc.set_buffer(0, Some(oy), (*r_at * 4) as u64);
                enc.set_buffer(1, Some(&gpu.route_buf), (*lg_off * 4) as u64);
                let n32 = *n_logits as u32;
                enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new((*n_logits as u64).div_ceil(256), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.memory_barrier_with_resources(&[&gpu.route_buf]);
            }
        }
        // Group table, gather and CSR: route_buf on the GPU-routing path,
        // x_arena otherwise.
        let (tab_buf, tab_off, tk_buf, tk_off) = match &rl {
            Some(rl) if tiles_mode => (&gpu.route_buf, rl.tiles, &gpu.route_buf, rl.tok),
            Some(rl) => (&gpu.route_buf, rl.table, &gpu.route_buf, rl.tok),
            None => (&gpu.x_arena, table_at, &gpu.x_arena, tok_at),
        };
        let (off_buf, off_off, hr_buf, hr_off, hw_buf, hw_off) = match &rl {
            Some(rl) => (
                &gpu.route_buf,
                rl.tok_off,
                &gpu.route_buf,
                rl.hit_row,
                &gpu.route_buf,
                rl.hit_w,
            ),
            None => (
                &gpu.x_arena,
                off_at,
                &gpu.x_arena,
                hrow_at,
                &gpu.x_arena,
                hw_at,
            ),
        };
        let stage = |enc: &metal::ComputeCommandEncoderRef,
                     si: usize,
                     n_in: usize,
                     n_out: usize,
                     x_buf: &Buffer,
                     x_off: usize,
                     y_off: usize,
                     sw_up: Option<(usize, usize)>| {
            enc.set_compute_pipeline_state(&states[si]);
            enc.set_buffer(0, Some(&gpu.chunks[mats[si].chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mats[si].w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(tab_buf), e(tab_off));
            enc.set_bytes(7, 8, &strides[si] as *const u64 as *const _);
            enc.set_buffer(8, Some(tk_buf), e(tk_off));
            if let Some((uo, bstr)) = sw_up {
                enc.set_buffer(9, Some(y), e(uo));
                let bs = bstr as u32;
                enc.set_bytes(12, 4, &bs as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64).div_ceil(bms[si]), row_tiles, n_groups),
                MTLSize::new(128, 1, 1),
            );
        };

        // GPU-side routing: pick top-k per token, prefix-sum the per-expert
        // counts into the group table, scatter token ids and CSR hits.
        if let (Some(r), Some(rl), Some((lg_tag, lg_off))) = (req.route.as_ref(), &rl, logits_at) {
            #[repr(C)]
            struct RouteParams {
                m: u32,
                n_expert: u32,
                n_used: u32,
                norm: u32,
                has_bias: u32,
                sigmoid: u32,
                shared: u32,
                scale: f32,
                total_rows: u32,
                tile_bound: u32,
                tile_w: u32,
            }
            let params = RouteParams {
                m: m_fused as u32,
                n_expert: req.n_expert as u32,
                n_used: r.n_used as u32,
                norm: r.norm as u32,
                has_bias: r.bias.is_some() as u32,
                sigmoid: r.sigmoid as u32,
                shared: req.shared.is_some() as u32,
                scale: r.scale,
                total_rows: req.total_rows as u32,
                tile_bound: rl.tile_bound as u32,
                tile_w: tile_w as u32,
            };
            let pp = &params as *const RouteParams as *const _;
            let psz = std::mem::size_of::<RouteParams>() as u64;
            let rb = &gpu.route_buf;
            let tgs = (m_fused as u64).div_ceil(256);
            let lg_buf = if lg_tag == "route" { rb } else { y };
            if use_route_fused {
                enc.set_compute_pipeline_state(&gpu.pipelines[&("route_fused", 1, 1)]);
                enc.set_buffer(0, Some(lg_buf), e(lg_off));
                enc.set_buffer(1, Some(rb), e(rl.bias));
                enc.set_buffer(2, Some(rb), e(rl.picks));
                enc.set_buffer(3, Some(rb), e(rl.pickw));
                enc.set_buffer(4, Some(rb), e(rl.table));
                enc.set_buffer(5, Some(rb), e(rl.tok));
                enc.set_buffer(6, Some(rb), e(rl.hit_row));
                enc.set_buffer(7, Some(rb), e(rl.hit_w));
                enc.set_buffer(8, Some(rb), e(rl.tok_off));
                enc.set_bytes(9, psz, pp);
                enc.set_buffer(10, Some(rb), e(rl.tiles));
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                cstamp!(gpu, enc, "rc");
                gpu.dbg_route.set(Some((rl.table, req.n_expert)));
            } else {
                enc.set_compute_pipeline_state(&gpu.pipelines[&("route_pick", 1, 1)]);
                enc.set_buffer(0, Some(lg_buf), e(lg_off));
                enc.set_buffer(1, Some(rb), e(rl.bias));
                enc.set_buffer(2, Some(rb), e(rl.counts));
                enc.set_buffer(3, Some(rb), e(rl.picks));
                enc.set_buffer(4, Some(rb), e(rl.pickw));
                enc.set_bytes(5, psz, pp);
                enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
                cstamp!(gpu, enc, "rp");
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[rb]);
                }
                enc.set_compute_pipeline_state(&gpu.pipelines[&("route_scan", 1, 1)]);
                enc.set_buffer(0, Some(rb), e(rl.counts));
                enc.set_buffer(1, Some(rb), e(rl.table));
                enc.set_buffer(2, Some(rb), e(rl.counters));
                let ne32 = req.n_expert as u32;
                enc.set_bytes(3, 4, &ne32 as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
                cstamp!(gpu, enc, "rs");
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[rb]);
                }
                enc.set_compute_pipeline_state(&gpu.pipelines[&("route_scatter", 1, 1)]);
                enc.set_buffer(0, Some(rb), e(rl.picks));
                enc.set_buffer(1, Some(rb), e(rl.pickw));
                enc.set_buffer(2, Some(rb), e(rl.table));
                enc.set_buffer(3, Some(rb), e(rl.counters));
                enc.set_buffer(4, Some(rb), e(rl.tok));
                enc.set_buffer(5, Some(rb), e(rl.hit_row));
                enc.set_buffer(6, Some(rb), e(rl.hit_w));
                enc.set_buffer(7, Some(rb), e(rl.tok_off));
                enc.set_bytes(8, psz, pp);
                enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
                cstamp!(gpu, enc, "rc");
            }
            // The gate stage reads the table/tok from route_buf and route_pick
            // read the logits from y - order both against what follows.
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[rb, y]);
            }
        }

        let (xb, xo): (&Buffer, usize) = if req.fused.is_some() {
            (&gpu.pf_hs, 0)
        } else {
            (&gpu.x_arena, x_at)
        };
        split_here!("route");
        // The shared expert's plain tile matmuls (borrowed by the phases
        // below): they depend only on pf_hs, so they fill the same
        // barrier-free windows as the expert gate/up and down stages.
        let mm = |enc: &metal::ComputeCommandEncoderRef,
                  si: usize,
                  n_in: usize,
                  n_out: usize,
                  x_buf: &Buffer,
                  x_off: usize,
                  y_off: usize| {
            let sh_mats = sh_mats.as_ref().expect("shared mm without mats");
            enc.set_compute_pipeline_state(&sh_states[si]);
            enc.set_buffer(0, Some(&gpu.chunks[sh_mats[si].chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &sh_mats[si].w_off as *const u64 as *const _);
            let m32 = m_fused as u32;
            enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
            let (bm, bn) =
                mm_kernel_for(sh_mats[si].ty, m_fused).map_or((32, 32), |(_, bm, bn)| (bm, bn));
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (n_out as u64).div_ceil(bm as u64),
                    (m_fused as u64).div_ceil(bn as u64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        };
        // Phase 1: everything reading pf_hs - expert gate/up plus the shared
        // expert's gate/up. No barriers between them, they run concurrently.
        if pf_skip("gate_up") {
            // Priced by absence.
        } else if dual {
            let ds = dual_state.as_ref().unwrap();
            enc.set_compute_pipeline_state(ds);
            enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
            enc.set_buffer(1, Some(xb), e(xo));
            enc.set_buffer(2, Some(y), e(gate_at));
            let a = req.hidden as u32;
            let b = (req.ffn * 2) as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mats[0].w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(tab_buf), e(tab_off));
            enc.set_bytes(7, 8, &strides[0] as *const u64 as *const _);
            enc.set_buffer(8, Some(tk_buf), e(tk_off));
            enc.set_buffer(10, Some(&gpu.chunks[mats[1].chunk].buf), 0);
            enc.set_bytes(11, 8, &mats[1].w_off as *const u64 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(((req.ffn * 2) as u64).div_ceil(64), row_tiles, n_groups),
                MTLSize::new(128, 1, 1),
            );
        } else {
            stage(enc, 0, req.hidden, req.ffn, xb, xo, gate_at, None);
            cstamp!(gpu, enc, "g");
            stage(enc, 1, req.hidden, req.ffn, xb, xo, up_at, None);
            cstamp!(gpu, enc, "u");
        }
        split_here!("gate");
        if let Some(sh) = &req.shared {
            mm(enc, 0, req.hidden, sh.ffn, &gpu.pf_hs, 0, shg_at);
            mm(enc, 1, req.hidden, sh.ffn, &gpu.pf_hs, 0, shu_at);
            // qwen35moe: the shared expert's gate projection, [m] logits.
            if let Some((sgmat, sgstate)) = &sh_gate_out {
                enc.set_compute_pipeline_state(sgstate);
                enc.set_buffer(0, Some(&gpu.chunks[sgmat.chunk].buf), 0);
                enc.set_buffer(1, Some(&gpu.pf_hs), 0);
                enc.set_buffer(2, Some(y), e(shw_at));
                let a = req.hidden as u32;
                let b = 1u32;
                enc.set_bytes(3, 4, &a as *const u32 as *const _);
                enc.set_bytes(4, 4, &b as *const u32 as *const _);
                enc.set_bytes(5, 8, &sgmat.w_off as *const u64 as *const _);
                let m32 = m_fused as u32;
                enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
                let (_bm, bn) =
                    mm_kernel_for(GgmlType::F32, m_fused).map_or((32, 32), |(_, bm, bn)| (bm, bn));
                enc.dispatch_thread_groups(
                    MTLSize::new(1, (m_fused as u64).div_ceil(bn as u64), 1),
                    MTLSize::new(128, 1, 1),
                );
            }
        }
        if !no_barrier() && !no_stage_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        split_here!("up");
        // Phase 2: the down stage (fused swiglu in its B staging, or the
        // standalone pass first) plus the shared expert's swiglu.
        if !fused_swiglu {
            enc.set_compute_pipeline_state(&swiglu_state);
            enc.set_buffer(0, Some(y), e(gate_at));
            enc.set_buffer(1, Some(y), e(up_at));
            let n32 = (req.total_rows * req.ffn) as u32;
            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            cstamp!(gpu, enc, "sw");
            if !no_barrier() && !no_stage_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
        }
        split_here!("swiglu");
        // The fused down kernel applies swiglu while staging B, so it reads
        // the raw gate and up regions; the plain kernel reads post-swiglu.
        if !pf_skip("down") {
            stage(
                enc,
                2,
                req.ffn,
                req.hidden,
                &gpu.y_arena,
                gate_at,
                out_at,
                if dual {
                    Some((gate_at + req.ffn, 2 * req.ffn))
                } else if fused_swiglu {
                    Some((up_at, req.ffn))
                } else {
                    None
                },
            );
        }
        cstamp!(gpu, enc, "dn");
        if let Some(sh) = &req.shared {
            enc.set_compute_pipeline_state(&swiglu_state);
            enc.set_buffer(0, Some(y), e(shg_at));
            enc.set_buffer(1, Some(y), e(shu_at));
            let n32 = (m_fused * sh.ffn) as u32;
            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
        // Only the shared expert's down projection consumes the swiglu
        // written above. Without a shared expert, the combine barrier below
        // already orders all expert down outputs; draining the same arena
        // here would add a second barrier without an intervening dispatch.
        if req.shared.is_some() && !no_barrier() && !no_stage_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        split_here!("down");
        // Phase 3: the shared expert's down, into the rows right after the
        // expert rows.
        if let Some(sh) = &req.shared {
            mm(
                enc,
                2,
                sh.ffn,
                req.hidden,
                &gpu.y_arena,
                shg_at,
                out_at + req.total_rows * req.hidden,
            );
        }
        split_here!("shared");
        if let Some(c) = &req.fused {
            if !no_barrier() && !no_stage_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
            // qwen35moe: fold sigmoid(shared gate logit) into the shared
            // slot's CSR weight (route_scatter wrote a constant 1). Runs
            // after the phase-1 gate projection; the barrier above covers it.
            if sh_gate_out.is_some() && route_mode {
                #[repr(C)]
                struct RouteParams {
                    m: u32,
                    n_expert: u32,
                    n_used: u32,
                    norm: u32,
                    has_bias: u32,
                    sigmoid: u32,
                    shared: u32,
                    scale: f32,
                    total_rows: u32,
                    tile_bound: u32,
                    tile_w: u32,
                }
                let r = req.route.as_ref().expect("route mode");
                let params = RouteParams {
                    m: m_fused as u32,
                    n_expert: req.n_expert as u32,
                    n_used: r.n_used as u32,
                    norm: r.norm as u32,
                    has_bias: r.bias.is_some() as u32,
                    sigmoid: r.sigmoid as u32,
                    shared: req.shared.is_some() as u32,
                    scale: r.scale,
                    total_rows: req.total_rows as u32,
                    tile_bound: rl.as_ref().map_or(0, |l| l.tile_bound as u32),
                    tile_w: tile_w as u32,
                };
                let rl = rl.as_ref().expect("route mode layout");
                enc.set_compute_pipeline_state(&gpu.pipelines[&("route_patch_shared_w", 1, 1)]);
                enc.set_buffer(0, Some(&gpu.route_buf), e(rl.hit_w));
                enc.set_buffer(1, Some(y), e(shw_at));
                enc.set_bytes(
                    2,
                    std::mem::size_of::<RouteParams>() as u64,
                    &params as *const RouteParams as *const _,
                );
                enc.dispatch_thread_groups(
                    MTLSize::new((m_fused as u64).div_ceil(32), 1, 1),
                    MTLSize::new(32, 1, 1),
                );
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[&gpu.route_buf]);
                }
            }
            // With the next layer's attention-norm weight known, the combine
            // also emits the normed rows for it; the attention block then
            // skips its norm_rows dispatch.
            let next_norm = c
                .next_norm
                .filter(|_| crate::runtime::get().pf_combine_resnorm)
                .and_then(|(raw, eps)| resolve_norm(&gpu, raw, req.hidden).map(|n| (n, eps)));
            if pf_skip("combine") {
                // Priced by absence; the next layer re-does its own norm.
            } else if let Some((nref, eps)) = next_norm {
                enc.set_compute_pipeline_state(&gpu.pipelines[&("combine_rows_resnorm", 1, 1)]);
                enc.set_buffer(0, Some(&gpu.pf_x), 0);
                enc.set_buffer(1, Some(y), e(out_at));
                enc.set_buffer(2, Some(off_buf), e(off_off));
                enc.set_buffer(3, Some(hr_buf), e(hr_off));
                enc.set_buffer(4, Some(hw_buf), e(hw_off));
                let h32 = req.hidden as u32;
                enc.set_bytes(5, 4, &h32 as *const u32 as *const _);
                enc.set_buffer(6, Some(&gpu.pf_hs), 0);
                enc.set_buffer(7, Some(&gpu.chunks[nref.chunk].buf), nref.off);
                enc.set_bytes(8, 4, &eps as *const f32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(c.m as u64, 1, 1), MTLSize::new(256, 1, 1));
                gpu.pf_hs_normed.set(true);
            } else {
                enc.set_compute_pipeline_state(&gpu.pipelines[&("combine_rows", 1, 1)]);
                enc.set_buffer(0, Some(&gpu.pf_x), 0);
                enc.set_buffer(1, Some(y), e(out_at));
                enc.set_buffer(2, Some(off_buf), e(off_off));
                enc.set_buffer(3, Some(hr_buf), e(hr_off));
                enc.set_buffer(4, Some(hw_buf), e(hw_off));
                let h32 = req.hidden as u32;
                enc.set_bytes(5, 4, &h32 as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(c.m as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            cstamp!(gpu, enc, "cb");
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
                eprintln!("ffn {}: {:.3} ms (gap {:.3})", label, (ke - ks) * 1e3, gap);
                prev_ke = Some(ke);
            }
            let ks: f64 = unsafe { msg_send![cmd, kernelStartTime] };
            let ke: f64 = unsafe { msg_send![cmd, kernelEndTime] };
            let gap = prev_ke.map_or(0.0, |p| (ks - p).max(0.0) * 1e3);
            eprintln!("ffn tail: {:.3} ms (gap {:.3})", (ke - ks) * 1e3, gap);
        } else {
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
        }
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !defer && !onebuf && std::env::var_os("ALLPAKA_FFN_TIME").is_some() {
            use objc::{msg_send, sel, sel_impl};
            let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
            let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
            eprintln!("ffnbuf: {:.3} ms", (ge - gs) * 1e3);
        }
        if !onebuf {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        DISPATCHES.fetch_add(4, Ordering::Relaxed);

        // Fused: the combine already landed in pf_x; skip the download.
        let out_rows = if req.fused.is_some() {
            0
        } else {
            req.total_rows
        };
        let mut out = vec![0f32; out_rows * req.hidden];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const f32).add(out_at),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    });
    Some(out)
}
