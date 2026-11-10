//! Attention over the shared KV region: single-query and batched attend,
//! output projection, and the fused decode attention block.

use super::*;

/// The kernel assumes a lane owns four dimensions of a 128-wide head.
pub(super) const ATTN_HEAD_DIM: usize = 128;

/// Attention for one token followed by the output projection, as ONE command
/// buffer with ONE wait.
///
/// The two are adjacent and dependent - `wo` consumes exactly what attention
/// produces - so fusing them costs nothing and saves a round trip. Encoded
/// separately, attention's own submission ate most of what the kernel won:
/// 48 extra command buffers per token on a 48-layer model, at the ~83 us
/// each one costs, against a phase that took 7 ms in total.
///
/// The intermediate never leaves the GPU arena, so the attention output is
/// not copied to the CPU at all.
pub fn attend_project(req: &AttnReq, wo_ty: GgmlType, wo: &[u8], n_out: usize) -> Option<Vec<f32>> {
    if !attn_shape_ok(req) {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    let mat = resolve(&gpu, wo_ty, wo)?;
    let n_in = req.n_q_heads * req.head_dim;
    if n_in % 4 != 0 || n_out % 4 != 0 {
        return None;
    }
    let lpr = lanes_per_row(wo_ty, n_in);
    let state = gpu.pipeline(mat.kernel, 1, lpr)?.to_owned();

    let align = |n: usize| (n + 255) & !255;
    let attn_bytes = align(n_in * 4);
    gpu.ensure_arenas(req.q.len() * 4, attn_bytes + align(n_out * 4));
    unsafe {
        std::ptr::copy_nonoverlapping(
            req.q.as_ptr() as *const u8,
            gpu.x_arena.contents() as *mut u8,
            req.q.len() * 4,
        );
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_attend(enc, &gpu, req, 0);
        // The compute encoder is serial, so the projection sees attention's
        // writes without any barrier of ours.
        enc.set_compute_pipeline_state(&state);
        enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
        enc.set_buffer(1, Some(&gpu.y_arena), 0);
        enc.set_buffer(2, Some(&gpu.y_arena), attn_bytes as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
        let per_group = 128u64;
        let total_threads = n_out as u64 * lpr as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(total_threads.div_ceil(per_group), 1, 1),
            MTLSize::new(per_group, 1, 1),
        );
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(2, Ordering::Relaxed);

        let mut out = vec![0f32; n_out];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const u8).add(attn_bytes),
                out.as_mut_ptr() as *mut u8,
                n_out * 4,
            );
        }
        out
    }))
}

/// Shapes the attention kernel is written for.
pub(super) fn attn_shape_ok(req: &AttnReq) -> bool {
    req.head_dim == ATTN_HEAD_DIM
        && req.n_pos > 0
        && req.group > 0
        && req.q.len() == req.n_q_heads * req.head_dim
        && (req.k_off + req.n_pos * req.kv_dim) * 2 <= req.cache.len
        && (req.v_off + req.n_pos * req.kv_dim) * 2 <= req.cache.len
}

/// Encode attention into an open encoder, q already staged in the x arena and
/// the result landing at `y_off` in the y arena.
pub(super) fn encode_attend(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    req: &AttnReq,
    y_off: usize,
) {
    encode_attend_at(enc, gpu, req, 0, y_off)
}

pub(super) fn encode_attend_at(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    req: &AttnReq,
    x_off: usize,
    y_off: usize,
) {
    encode_attend_from(enc, gpu, req, &gpu.x_arena, x_off, y_off)
}

/// The general form: q can live in either arena - the fused attention block
/// reads it from the y arena, where the qkv matmul just wrote it.
pub(super) fn encode_attend_from(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    req: &AttnReq,
    q_buf: &Buffer,
    q_off: usize,
    y_off: usize,
) {
    enc.set_compute_pipeline_state(&gpu.pipelines[&(attend_kernel(), 1, 1)]);
    enc.set_buffer(0, Some(&req.cache.buf), 0);
    enc.set_buffer(1, Some(&req.cache.buf), 0);
    enc.set_buffer(2, Some(q_buf), q_off as u64);
    enc.set_buffer(3, Some(&gpu.y_arena), y_off as u64);
    for (index, value) in [
        (4u64, req.kv_dim as u32),
        (5, req.head_dim as u32),
        (6, req.n_pos as u32),
        (7, req.group as u32),
    ] {
        enc.set_bytes(index, 4, &value as *const u32 as *const _);
    }
    enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
    for (index, value) in [(9u64, req.k_off as u64), (10, req.v_off as u64)] {
        enc.set_bytes(index, 8, &value as *const u64 as *const _);
    }
    enc.dispatch_thread_groups(
        MTLSize::new(req.n_q_heads as u64, 1, 1),
        MTLSize::new(attend_tg(), 1, 1),
    );
}

pub fn attn_block(req: &AttnBlockReq) -> Option<Vec<f32>> {
    let hd = req.head_dim;
    let rot_dim = req.rope.len() * 2;
    if hd != ATTN_HEAD_DIM || (rot_dim != hd && rot_dim * 2 != hd) {
        return None;
    }
    let hidden = req.x.len();
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    if req.wq.2 != q_dim || req.wk.2 != kv || req.wv.2 != kv || req.wo.2 != hidden {
        return None;
    }
    let span = (req.pos + 1) * req.kv_dim;
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
    let dims = [(hidden, q_dim), (hidden, kv), (hidden, kv), (q_dim, hidden)];
    let mut states = Vec::with_capacity(4);
    for (mat, &(n_in, _)) in mats.iter().zip(&dims) {
        let lpr = lanes_per_row(mat.ty, n_in);
        states.push(gpu.pipeline(mat.kernel, 1, lpr)?.to_owned());
    }

    // Arena plan: q, k, v, attention out and the projection, all in y.
    let align = |n: usize| (n + 255) & !255;
    let q_off = 0usize;
    let k_arena = align(q_dim * 4);
    let v_arena = k_arena + align(kv * 4);
    let attn_off = v_arena + align(kv * 4);
    let out_off = attn_off + align(q_dim * 4);
    gpu.ensure_arenas(align(hidden * 4), out_off + align(hidden * 4));
    unsafe {
        std::ptr::copy_nonoverlapping(
            req.x.as_ptr() as *const u8,
            gpu.x_arena.contents() as *mut u8,
            hidden * 4,
        );
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Concurrent, with a barrier at each stage boundary: q, k and v are
        // independent and small, and the serial encoder ran them one by one.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);

        // qkv projections out of the x arena.
        for (i, y_at) in [(0usize, q_off), (1, k_arena), (2, v_arena)] {
            let (n_in, n_out) = dims[i];
            enc.set_compute_pipeline_state(&states[i]);
            enc.set_buffer(0, Some(&gpu.chunks[mats[i].chunk].buf), 0);
            enc.set_buffer(1, Some(&gpu.x_arena), 0);
            enc.set_buffer(2, Some(&gpu.y_arena), y_at as u64);
            let n_in32 = n_in as u32;
            let n_out32 = n_out as u32;
            enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
            enc.set_bytes(5, 8, &mats[i].w_off as *const u64 as *const _);
            let lpr = lanes_per_row(mats[i].ty, n_in) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64 * lpr).div_ceil(128), 1, 1),
                MTLSize::new(128, 1, 1),
            );
        }

        enc.memory_barrier_with_resources(&[&gpu.y_arena]);

        // Norm + rope + cache store.
        let args = QkPrepArgs {
            n_heads: req.n_heads as u32,
            n_kv_heads: req.n_kv_heads as u32,
            head_dim: hd as u32,
            kv_dim: req.kv_dim as u32,
            eps: req.eps,
            pos: req.pos as u32,
            has_qk_norm: req.q_norm.is_some() as u32,
            rot_dim: rot_dim as u32,
            k_base: req.k_off as u64,
            v_base: req.v_off as u64,
            has_bias: 0,
            rope_norm: req.rope_norm as u32,
        };
        enc.set_compute_pipeline_state(&gpu.pipelines[&("qk_prep", 1, 1)]);
        enc.set_buffer(0, Some(&gpu.y_arena), q_off as u64);
        enc.set_buffer(1, Some(&gpu.y_arena), k_arena as u64);
        enc.set_buffer(2, Some(&gpu.y_arena), v_arena as u64);
        enc.set_buffer(3, Some(&req.cache.buf), 0);
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
        enc.set_bytes(
            7,
            (req.rope.len() * 8) as u64,
            req.rope.as_ptr() as *const _,
        );
        for index in 8u64..=10 {
            enc.set_buffer(index, Some(&req.cache.buf), 0);
        }
        enc.dispatch_thread_groups(
            MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );

        enc.memory_barrier_with_resources(&[&gpu.y_arena, &req.cache.buf]);

        // Attention over the cache the previous dispatch just extended.
        let attn_req = AttnReq {
            cache: req.cache,
            k_off: req.k_off,
            v_off: req.v_off,
            q: &[],
            kv_dim: req.kv_dim,
            head_dim: hd,
            n_q_heads: req.n_heads,
            group: req.n_heads / req.n_kv_heads.max(1),
            n_pos: req.pos + 1,
            scale: req.scale,
        };
        encode_attend_from(enc, &gpu, &attn_req, &gpu.y_arena, q_off, attn_off);
        enc.memory_barrier_with_resources(&[&gpu.y_arena]);

        // Output projection.
        let (n_in, n_out) = dims[3];
        enc.set_compute_pipeline_state(&states[3]);
        enc.set_buffer(0, Some(&gpu.chunks[mats[3].chunk].buf), 0);
        enc.set_buffer(1, Some(&gpu.y_arena), attn_off as u64);
        enc.set_buffer(2, Some(&gpu.y_arena), out_off as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mats[3].w_off as *const u64 as *const _);
        let lpr = lanes_per_row(mats[3].ty, n_in) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new((n_out as u64 * lpr).div_ceil(128), 1, 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(6, Ordering::Relaxed);

        let mut out = vec![0f32; hidden];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const u8).add(out_off),
                out.as_mut_ptr() as *mut u8,
                hidden * 4,
            );
        }
        out
    }))
}

/// A whole prefill chunk's attention as ONE command buffer with ONE wait.
///
/// The requests are one per chunk position, each attending over the cache up
/// to its own position - causality is in `n_pos`, not in any mask. They
/// reuse the decode kernel unchanged: a dispatch per position costs ~1 us to
/// encode, against a CPU pass that was nearly half of prefill wall time.
///
/// All-or-nothing like the matmul batches: one undispatchable request sends
/// the whole chunk to the CPU path.
pub fn attend_batch(reqs: &[AttnReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    if reqs.iter().any(|r| !attn_shape_ok(r)) {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    let align = |n: usize| (n + 255) & !255;
    let mut x_need = 0usize;
    let mut y_need = 0usize;
    let mut offs = Vec::with_capacity(reqs.len());
    for r in reqs {
        offs.push((x_need, y_need));
        x_need += align(r.q.len() * 4);
        y_need += align(r.n_q_heads * r.head_dim * 4);
    }
    gpu.ensure_arenas(x_need, y_need);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut u8;
        for (r, (x_off, _)) in reqs.iter().zip(&offs) {
            std::ptr::copy_nonoverlapping(r.q.as_ptr() as *const u8, xp.add(*x_off), r.q.len() * 4);
        }
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Every position attends independently over an already-written
        // cache: no hazards, full concurrency.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        for (r, (x_off, y_off)) in reqs.iter().zip(&offs) {
            encode_attend_at(enc, &gpu, r, *x_off, *y_off);
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(reqs.len() as u64, Ordering::Relaxed);

        let yp = gpu.y_arena.contents() as *const u8;
        reqs.iter()
            .zip(&offs)
            .map(|(r, (_, y_off))| {
                let n = r.n_q_heads * r.head_dim;
                let mut v = vec![0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(yp.add(*y_off), v.as_mut_ptr() as *mut u8, n * 4);
                }
                v
            })
            .collect()
    }))
}

/// Attention for one token on the GPU. None means "do it on the CPU": no
/// device, a head width the kernel is not written for, or an empty cache.
pub fn attend(req: &AttnReq) -> Option<Vec<f32>> {
    if !attn_shape_ok(req) {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    let out_len = req.n_q_heads * req.head_dim;
    gpu.ensure_arenas(req.q.len() * 4, out_len * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(
            req.q.as_ptr() as *const u8,
            gpu.x_arena.contents() as *mut u8,
            req.q.len() * 4,
        );
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        // The same encoding the fused path uses; two copies of these
        // bindings is exactly how the half switch broke one of them.
        encode_attend(enc, &gpu, req, 0);
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(1, Ordering::Relaxed);

        let mut out = vec![0f32; out_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                gpu.y_arena.contents() as *const u8,
                out.as_mut_ptr() as *mut u8,
                out_len * 4,
            );
        }
        out
    }))
}
