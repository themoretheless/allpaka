//! Dequant matvec and tile-matmul dispatch: lane and tile geometry, kernel
//! choice per tensor format, and the batched matvec entry points.

use super::*;

/// How many lanes cooperate on one output row, given how many quant blocks
/// that row holds. The lanes each take whole blocks, so more lanes than
/// blocks is pure waste; fewer means a lane walks several blocks, which is
/// fine. Rounded up to a power of two so the shuffle reduction stays inside
/// its lane group, and capped at the SIMD width.
pub(super) fn lanes_per_row(ty: GgmlType, n_in: usize) -> usize {
    // The llama-structure q2_k kernel has a fixed geometry: 8 threads per
    // output row (one 32-lane SIMD group per 4 rows), independent of the
    // block count. The old kernel also runs correctly at LPR 8, so the tile
    // fallbacks in pipeline_ex need no geometry special-casing.
    if ty == GgmlType::Q2K && q2_mv() && std::env::var_os("ALLPAKA_Q2_ILV").is_none() {
        // 32 / NR0 threads per row; NR0 sweepable (occupancy vs activation
        // reuse). ffnbench picked 2 rows per SIMD group on M4 Max.
        static NR0: OnceLock<usize> = OnceLock::new();
        let nr0 = *NR0.get_or_init(|| {
            std::env::var("ALLPAKA_Q2_NR0")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n: &usize| [1, 2, 4].contains(n))
                .unwrap_or(2)
        });
        return 32 / nr0;
    }
    // The q3/q4/q5/q6/q8 llama-structure kernels use NR0 rows per SIMD
    // group via LPR (NR0 = 32/LPR). Q4/Q5 are sweepable for GLM 1408-wide
    // expert/shared shapes; others stay at the port default (NR0=2).
    if ty == GgmlType::Q3K && q3_mv() {
        return 16;
    }
    if ty == GgmlType::Q4K && q4_mv() {
        static NR0: OnceLock<usize> = OnceLock::new();
        let nr0 = *NR0.get_or_init(|| {
            std::env::var("ALLPAKA_Q4_NR0")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n: &usize| [1, 2, 4].contains(n))
                .unwrap_or(2)
        });
        return 32 / nr0;
    }
    if ty == GgmlType::Q5K && q5_mv() {
        static NR0: OnceLock<usize> = OnceLock::new();
        let nr0 = *NR0.get_or_init(|| {
            std::env::var("ALLPAKA_Q5_NR0")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n: &usize| [1, 2, 4].contains(n))
                .unwrap_or(2)
        });
        return 32 / nr0;
    }
    if ty == GgmlType::Q6K && q6_mv() {
        return 16;
    }
    if ty == GgmlType::Q8_0 && q8_mv() {
        return 32 / q8_nr0();
    }
    let be = ty.block_elements().unwrap_or(32) as usize;
    let blocks = (n_in / be.max(1)).max(1);
    let mut base = blocks.next_power_of_two().min(32);
    // The interleaved q2_k variant owns TWO blocks per lane at half the
    // lanes; the pipeline swap happens in the kernel-name maps below.
    static ILV: OnceLock<bool> = OnceLock::new();
    if ty == GgmlType::Q2K
        && *ILV.get_or_init(|| std::env::var_os("ALLPAKA_Q2_ILV").is_some())
        && base >= 2
    {
        base /= 2;
    }
    // Sweep knob: fewer lanes per row means more rows per SIMD group and
    // more quant blocks in flight per lane - the occupancy-vs-ILP tradeoff
    // the ffn microbench measures. Power of two, so the shuffle reduction
    // stays intact.
    static DIV: OnceLock<usize> = OnceLock::new();
    let div = *DIV.get_or_init(|| {
        std::env::var("ALLPAKA_LPR_DIV")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|d: &usize| d.is_power_of_two())
            .unwrap_or(1)
    });
    (base / div).max(1)
}

/// Threadgroup size for matvec dispatch. Default 128; sweep with
/// `ALLPAKA_MV_TG` ∈ {64,128,256} (geometry only — NR0 stays 2).
pub(super) fn mv_tg() -> u64 {
    crate::runtime::get().mv_tg
}

/// Llama-style `mul_mv_id` grid for INDEXED decode matvecs (`grid.z = slots`).
/// Default ON. `ALLPAKA_MV_ID=0` restores flat `n_out * slots` 1D packing.
pub(super) fn mv_id_grid() -> bool {
    crate::runtime::get().mv_id_grid
}

/// Dispatch geometry for a matvec pipeline (LPR × rows, TG via `mv_tg`).
pub(super) fn dispatch_mv(enc: &metal::ComputeCommandEncoderRef, ty: GgmlType, n_in: usize, n_rows: usize) {
    let lpr = lanes_per_row(ty, n_in) as u64;
    let tg = mv_tg();
    enc.dispatch_thread_groups(
        MTLSize::new((n_rows as u64 * lpr).div_ceil(tg), 1, 1),
        MTLSize::new(tg, 1, 1),
    );
}

/// Indexed MoE matvec: either llama `mul_mv_id` (`z = slots`) or flat 1D.
pub(super) fn dispatch_mv_indexed(
    enc: &metal::ComputeCommandEncoderRef,
    ty: GgmlType,
    n_in: usize,
    n_out: usize,
    slots: usize,
) {
    let uses_mv = match ty {
        GgmlType::Q2K => q2_kernel() == "matvec_q2_k_mv",
        GgmlType::Q3K => q3_kernel() == "matvec_q3_k_mv",
        GgmlType::Q4K => q4_kernel() == "matvec_q4_k_mv",
        GgmlType::Q5K => q5_kernel() == "matvec_q5_k_mv",
        GgmlType::Q6K => q6_kernel() == "matvec_q6_k_mv",
        GgmlType::Q8_0 => q8_kernel() == "matvec_q8_0_mv",
        _ => false,
    };
    if mv_id_grid() && uses_mv {
        let lpr = lanes_per_row(ty, n_in) as u64;
        let tg = mv_tg();
        enc.dispatch_thread_groups(
            MTLSize::new((n_out as u64 * lpr).div_ceil(tg), 1, slots as u64),
            MTLSize::new(tg, 1, 1),
        );
    } else {
        dispatch_mv(enc, ty, n_in, n_out * slots);
    }
}

/// Batch size from which the tile-matmul kernels beat the tiled matvecs.
/// Below this the [32 x 32] output tile is mostly padding and the matvec
/// path's exact tiles waste nothing.
pub(super) const MM_MIN_M: usize = 16;

/// The tile-matmul kernel for a format, and its output-tile height.
///
/// Two compiled heights exist, 32 and 64 rows; 64 reads each staged
/// activation tile against twice the weight rows. `ALLPAKA_MM64` switches at
/// startup so the two can be swept as whole-bench A/B runs.
/// `(kernel name, BM, BN)`. Three compiled tile shapes: 32x32, 64x32
/// (`ALLPAKA_MM64`, measured worse) and 32x64 (`ALLPAKA_MM_BN64`), the
/// wide-batch tile that halves how often the weights are re-read per pass.
pub(super) fn mm_kernel_for(ty: GgmlType, m: usize) -> Option<(&'static str, usize, usize)> {
    #[derive(Clone, Copy, PartialEq)]
    enum Shape {
        Base,
        Tall,
        Wide,
    }
    static SHAPE: OnceLock<Option<Shape>> = OnceLock::new();
    let forced = *SHAPE.get_or_init(|| {
        if std::env::var_os("ALLPAKA_MM64").is_some() {
            Some(Shape::Tall)
        } else if std::env::var_os("ALLPAKA_MM_BN64").is_some() {
            Some(Shape::Wide)
        } else if std::env::var_os("ALLPAKA_MM_BN32").is_some() {
            Some(Shape::Base)
        } else {
            None
        }
    });
    // 32x32 unconditionally: both alternatives MEASURED WORSE on this GPU,
    // even where their theory was strongest. Tall (64x32) lost 9% outright;
    // wide (64 columns) lost 6% forced everywhere AND ~3% when selected
    // per-m for the dense m=256 projections only - with concurrent encoders
    // the device is occupancy-bound, and fewer threadgroups hurt more than
    // halved weight re-reads help. The env pins stay for re-sweeping on
    // other hardware.
    let _ = m;
    // The llama.cpp-structure mm port (simdgroup MMA over half-staged
    // operands, 64x32 tile): `ALLPAKA_MM_LL=1` routes every quant format to
    // it; F32 (router only) keeps the old kernel.
    if mm_ll() {
        let k64 = mm_k64();
        let name = match (ty, k64) {
            (GgmlType::Q8_0, false) => {
                // The pipelined K-loop measured +9-11% over mmll_q8_0 on the
                // 35B forms (wqkv/zgate/ssm_out); ALLPAKA_MM_PIPE=0 reverts.
                if mm_pipe() {
                    "mmllp_q8_0"
                } else {
                    "mmll_q8_0"
                }
            }
            // Plain (non-indexed) Q5_0 mm: GLM shared / dense projections that
            // are not expert-indexed. Indexed MoE already had mmll_id_q5_0.
            (GgmlType::Q5_0, false) => "mmll_q5_0",
            (GgmlType::Q2K, false) => "mmll_q2_k",
            (GgmlType::Q3K, false) => "mmll_q3_k",
            (GgmlType::Q4K, false) => {
                // Software-pipelined double-buffer K-loop: mmbench on the
                // 30B forms measured 12.4 vs 11.4 TFLOPS loop and 8.9 vs
                // 8.0 one-shot against mmll_q4_k. `ALLPAKA_MM_PIPE=0`
                // reverts.
                if mm_pipe() {
                    "mmllp_q4_k"
                } else {
                    "mmll_q4_k"
                }
            }
            (GgmlType::Q5K, false) => "mmll_q5_k",
            (GgmlType::Q6K, false) => "mmll_q6_k",
            (GgmlType::Q8_0, true) => "mm64ll_q8_0",
            (GgmlType::Q5_0, true) => "mm64ll_q5_0",
            (GgmlType::Q2K, true) => "mm64ll_q2_k",
            (GgmlType::Q3K, true) => "mm64ll_q3_k",
            (GgmlType::Q4K, true) => "mm64ll_q4_k",
            (GgmlType::Q5K, true) => "mm64ll_q5_k",
            (GgmlType::Q6K, true) => "mm64ll_q6_k",
            // The router: f32 weights staged to half like everything else.
            // Top-8 expert selection tolerates the rounding; the f32_router
            // parity test and verify are the arbiters. Own kill switch for
            // clean A/B: ALLPAKA_MM_LL_F32=0.
            (GgmlType::F32, _) => {
                static F32LL: OnceLock<bool> = OnceLock::new();
                if *F32LL.get_or_init(|| std::env::var("ALLPAKA_MM_LL_F32").is_ok_and(|v| v == "1"))
                {
                    "mmll_f32"
                } else {
                    return Some(("matmul_f32", 32, 32));
                }
            }
            _ => return None,
        };
        return Some((name, 64, 32));
    }
    let shape = forced.unwrap_or(Shape::Base);
    let name = match (ty, shape) {
        (GgmlType::Q8_0, Shape::Base) => "matmul_q8_0",
        (GgmlType::Q2K, Shape::Base) => "matmul_q2_k",
        (GgmlType::Q3K, Shape::Base) => "matmul_q3_k",
        (GgmlType::Q4K, Shape::Base) => "matmul_q4_k",
        (GgmlType::Q5K, Shape::Base) => "matmul_q5_k",
        (GgmlType::Q6K, Shape::Base) => "matmul_q6_k",
        (GgmlType::F32, Shape::Base) => "matmul_f32",
        (GgmlType::Q8_0, Shape::Tall) => "matmul64_q8_0",
        (GgmlType::Q2K, Shape::Tall) => "matmul64_q2_k",
        (GgmlType::Q3K, Shape::Tall) => "matmul64_q3_k",
        (GgmlType::Q4K, Shape::Tall) => "matmul64_q4_k",
        (GgmlType::Q5K, Shape::Tall) => "matmul64_q5_k",
        (GgmlType::Q6K, Shape::Tall) => "matmul64_q6_k",
        (GgmlType::F32, Shape::Tall) => "matmul64_f32",
        (GgmlType::Q8_0, Shape::Wide) => "matmulw_q8_0",
        (GgmlType::Q2K, Shape::Wide) => "matmulw_q2_k",
        (GgmlType::Q3K, Shape::Wide) => "matmulw_q3_k",
        (GgmlType::Q4K, Shape::Wide) => "matmulw_q4_k",
        (GgmlType::Q5K, Shape::Wide) => "matmulw_q5_k",
        (GgmlType::Q6K, Shape::Wide) => "matmulw_q6_k",
        (GgmlType::F32, Shape::Wide) => "matmulw_f32",
        _ => return None,
    };
    let (bm, bn) = match shape {
        Shape::Base => (32, 32),
        Shape::Tall => (64, 32),
        Shape::Wide => (32, 64),
    };
    Some((name, bm, bn))
}

#[repr(C)]
pub(super) struct MmQkvArgs {
    pub(super) n_in: u32,
    pub(super) n_q: u32,
    pub(super) n_k: u32,
    pub(super) n_v: u32,
    pub(super) wq_off: u64,
    pub(super) wk_off: u64,
    pub(super) wv_off: u64,
    pub(super) q_off: u64,
    pub(super) k_off: u64,
    pub(super) v_off: u64,
    pub(super) m: u32,
    pub(super) pad: u32,
}

/// Fused QKV (or dual wqkv + zgate) prefill kernel when projections share the
/// same quantized format and their output widths are multiples of 64.
pub(super) fn mm_qkv_kernel_for(ty: GgmlType) -> Option<&'static str> {
    static NO_QKV: OnceLock<bool> = OnceLock::new();
    if !mm_ll()
        || mm_k64()
        || *NO_QKV.get_or_init(|| std::env::var_os("ALLPAKA_NO_FUSED_QKV").is_some())
    {
        return None;
    }
    Some(match ty {
        GgmlType::Q8_0 => {
            if mm_pipe() {
                "mmllp_qkv_q8_0"
            } else {
                "mmll_qkv_q8_0"
            }
        }
        GgmlType::Q5_0 => "mmll_qkv_q5_0",
        GgmlType::Q2K => "mmll_qkv_q2_k",
        GgmlType::Q3K => "mmll_qkv_q3_k",
        GgmlType::Q4K => {
            if mm_pipe() {
                "mmllp_qkv_q4_k"
            } else {
                "mmll_qkv_q4_k"
            }
        }
        GgmlType::Q5K => "mmll_qkv_q5_k",
        GgmlType::Q6K => "mmll_qkv_q6_k",
        _ => return None,
    })
}

/// The compiled TILE specialisations, largest last. A batch is split into
/// tiles from this set, so every dispatch covers exactly the rows it claims.
pub(super) const TILES: &[usize] = &[1, 2, 4, 8];

/// Split `m` rows into `(start, tile)` dispatches, taking the largest
/// available tile first. Exact by construction: the tiles are powers of two
/// down to 1, so the remainder is always coverable and no dispatch ever
/// touches a row past `m`.
pub(super) fn tiles_for(m: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < m {
        let left = m - start;
        let tile = TILES
            .iter()
            .rev()
            .copied()
            .find(|&t| t <= left)
            .expect("TILES contains 1");
        out.push((start, tile));
        start += tile;
    }
    out
}

/// Run one matvec on the GPU. Returns None to mean "do it on the CPU".
pub fn matvec(ty: GgmlType, w: &[u8], n_in: usize, n_out: usize, x: &[f32]) -> Option<Vec<f32>> {
    matvec_batch(&[MatvecReq {
        ty,
        w,
        n_in,
        n_out,
        x,
        m: 1,
    }])
    .map(|mut v| v.pop().unwrap())
}

/// Run several independent matvecs as ONE command buffer with ONE wait.
///
/// This is where the GPU path earns its keep. Decode issues over a thousand
/// small matvecs per token, and a synchronous wait per dispatch costs more
/// than the arithmetic; batching the independent ones (q/k/v together, all
/// experts' gate+up together) collapses those waits to a handful per layer.
///
/// All-or-nothing: if any request has no kernel or foreign bytes, the whole
/// batch declines and the caller runs it on the CPU.
pub fn matvec_batch(reqs: &[MatvecReq]) -> Option<Vec<Vec<f32>>> {
    matvec_batch_inner(reqs, None)
}

pub fn matvec_shared(
    ty: GgmlType,
    w: &[u8],
    n_in: usize,
    n_out: usize,
    x: &SharedVector,
    offset: usize,
    m: usize,
) -> Option<Vec<f32>> {
    // Kernels use float4 loads. The staging path realigns arbitrary CPU
    // slices, but a direct binding must already satisfy that alignment.
    if offset % 4 != 0 || n_in % 4 != 0 || m == 0 {
        return None;
    }
    let end = offset.checked_add(m.checked_mul(n_in)?)?;
    let values = x.get(offset..end)?;
    let out = matvec_batch_inner(
        &[MatvecReq {
            ty,
            w,
            n_in,
            n_out,
            x: values,
            m,
        }],
        Some((x.buf.as_ref()?, offset.checked_mul(4)?)),
    )?
    .pop()?;
    SHARED_PROJECTION_BYTES.fetch_add((values.len() * 4) as u64, Ordering::Relaxed);
    Some(out)
}

pub(super) fn matvec_batch_inner(
    reqs: &[MatvecReq],
    shared: Option<(&Buffer, usize)>,
) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    // Validate everything up front: all-or-nothing keeps CPU/GPU results
    // from interleaving within one logical operation.
    let mut kernels = Vec::with_capacity(reqs.len());
    for r in reqs {
        // A format needs a matvec kernel below the mm threshold, an mm
        // kernel above it. F32 has only the latter: it exists for the MoE
        // router, whose prefill batches are large and whose decode matvec is
        // a rounding error on the CPU.
        let kernel = match r.ty {
            GgmlType::Q5_0 => Some("matvec_q5_0"),
            GgmlType::Q8_0 => Some(q8_kernel()),
            GgmlType::Q2K => Some(q2_kernel()),
            GgmlType::Q3K => Some(q3_kernel()),
            GgmlType::Q4K => Some(q4_kernel()),
            GgmlType::Q5K => Some(q5_kernel()),
            GgmlType::Q6K => Some(q6_kernel()),
            _ => None,
        };
        let routable_mm = r.m >= MM_MIN_M && mm_kernel_for(r.ty, r.m).is_some();
        if kernel.is_none() && !routable_mm {
            return None;
        }
        if r.x.len() != r.m * r.n_in {
            return None;
        }
        let addr = r.w.as_ptr() as usize;
        // Not inside any attached mapping (e.g. a test matrix on the heap)
        // sends the whole batch to the CPU.
        let chunk = gpu.chunk_for(addr, r.w.len())?;
        kernels.push((kernel, chunk, (addr - gpu.chunks[chunk].start) as u64));
    }

    // Bump-plan every dispatch's arena slices up front, then grow the arenas
    // once. Offsets are 256-aligned, which satisfies setBuffer:offset:.
    struct Slot {
        ri: usize,
        start_row: usize,
        rows: usize,
        x_off: usize,
        y_off: usize,
        /// Route this slot to the tile-matmul kernel instead of the matvec.
        mm: bool,
    }
    let align = |n: usize| (n + 255) & !255;
    let mut x_need = 0usize;
    let mut y_need = 0usize;
    let mut slots = Vec::new();
    for (ri, r) in reqs.iter().enumerate() {
        // Big batches go whole to the tile-matmul kernel; small ones become
        // matvec dispatches, each running the tile specialisation that
        // matches its row count exactly.
        if r.m >= MM_MIN_M && mm_kernel_for(r.ty, r.m).is_some() {
            slots.push(Slot {
                ri,
                start_row: 0,
                rows: r.m,
                x_off: x_need,
                y_off: y_need,
                mm: true,
            });
            x_need += align(r.m * r.n_in * 4);
            y_need += align(r.m * r.n_out * 4);
            continue;
        }
        for (start_row, rows) in tiles_for(r.m) {
            slots.push(Slot {
                ri,
                start_row,
                rows,
                x_off: x_need,
                y_off: y_need,
                mm: false,
            });
            x_need += align(rows * r.n_in * 4);
            y_need += align(rows * r.n_out * 4);
        }
    }
    gpu.ensure_arenas(x_need, y_need);

    // Specialise every dispatch's pipeline before encoding: the lookup may
    // compile a new variant and needs &mut, while the encoder below holds the
    // rest of `gpu` immutably.
    let mut states = Vec::with_capacity(slots.len());
    for s in &slots {
        if s.mm {
            states.push(
                gpu.pipeline(mm_kernel_for(reqs[s.ri].ty, s.rows)?.0, 1, 1)?
                    .to_owned(),
            );
        } else {
            let lpr = lanes_per_row(reqs[s.ri].ty, reqs[s.ri].n_in);
            states.push(gpu.pipeline(kernels[s.ri].0?, s.rows, lpr)?.to_owned());
        }
    }

    // Stage the activations into the shared arena: the only per-call copy.
    unsafe {
        let xp = gpu.x_arena.contents() as *mut u8;
        for s in &slots {
            if shared.is_some() {
                continue;
            }
            let r = &reqs[s.ri];
            let xs = &r.x[s.start_row * r.n_in..(s.start_row + s.rows) * r.n_in];
            std::ptr::copy_nonoverlapping(xs.as_ptr() as *const u8, xp.add(s.x_off), xs.len() * 4);
        }
    }

    // The autorelease pool matters: command buffers and encoders are
    // autoreleased ObjC objects, and without a pool on this thread they pile
    // up until some outer drain - slow and leaky at 200 calls per token.
    let out = objc::rc::autoreleasepool(|| {
        // Encode + wait; see the identical structure in ffn_batch.
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Concurrent: the slots read shared weights and disjoint x slices
        // and write disjoint y slices, so they can fill the GPU together. A
        // single expert-sized matvec is ~25k threads - a fraction of the
        // device - and the serial encoder was running them ONE AT A TIME:
        // that, not arithmetic, was the measured ceiling of the decode ffn.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        for (si, s) in slots.iter().enumerate() {
            let r = &reqs[s.ri];
            let (_, chunk, w_off) = &kernels[s.ri];
            enc.set_compute_pipeline_state(&states[si]);
            enc.set_buffer(0, Some(&gpu.chunks[*chunk].buf), 0);
            let (input, input_offset) = shared
                .map_or((&gpu.x_arena, s.x_off), |(buffer, offset)| {
                    (buffer, offset + s.start_row * r.n_in * 4)
                });
            enc.set_buffer(1, Some(input), input_offset as u64);
            enc.set_buffer(2, Some(&gpu.y_arena), s.y_off as u64);
            let n_in32 = r.n_in as u32;
            let n_out32 = r.n_out as u32;
            enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
            enc.set_bytes(5, 8, w_off as *const u64 as *const _);

            if s.mm {
                // One threadgroup per [BM x BN] output tile.
                let (bm, bn) = mm_kernel_for(r.ty, s.rows).map_or((32, 32), |(_, bm, bn)| (bm, bn));
                let m32 = s.rows as u32;
                enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        (r.n_out as u64).div_ceil(bm as u64),
                        (s.rows as u64).div_ceil(bn as u64),
                        1,
                    ),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                // One lane group per output row, threadgroups of four SIMD
                // groups (see dispatch_mv).
                dispatch_mv(enc, r.ty, r.n_in, r.n_out);
            }
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(slots.len() as u64, Ordering::Relaxed);

        let mut out: Vec<Vec<f32>> = reqs
            .iter()
            .map(|r| Vec::with_capacity(r.m * r.n_out))
            .collect();
        let yp = gpu.y_arena.contents() as *const u8;
        for s in &slots {
            let n = s.rows * reqs[s.ri].n_out;
            let at = out[s.ri].len();
            out[s.ri].resize(at + n, 0.0);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    yp.add(s.y_off),
                    out[s.ri][at..].as_mut_ptr() as *mut u8,
                    n * 4,
                );
            }
        }
        out
    });
    Some(out)
}
