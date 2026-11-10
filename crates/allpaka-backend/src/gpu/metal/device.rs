//! The device singleton: weight windows over the mmap, residency, arenas,
//! lazily specialised pipelines, and the attach and capability entry points.

use super::*;

/// One `bytesNoCopy` window over the weight mapping. Chunks overlap by more
/// than any single tensor, so every request fits wholly inside some chunk.
pub(super) struct WeightChunk {
    pub(super) buf: Buffer,
    /// Absolute address of this chunk's first byte. Chunks from different
    /// attached mappings coexist in one list; an absolute range is what lets
    /// a weight slice find its window no matter which model it belongs to.
    pub(super) start: usize,
    pub(super) len: usize,
}

pub struct Gpu {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    /// One pipeline per (kernel, tile, lanes-per-row): both are function
    /// constants, so each shape is its own specialisation. The full cross
    /// product is 4 tiles x 6 lane counts x 6 formats, and a given model uses
    /// a handful of those, so they are compiled on first use rather than at
    /// attach. `swiglu` ignores both and is stored under (1, 1).
    pub(super) pipelines: HashMap<(&'static str, usize, usize), ComputePipelineState>,
    pub(super) library: metal::Library,
    pub(super) chunks: Vec<WeightChunk>,
    /// The queue's MTLResidencySet, retained as a raw pointer (usize keeps
    /// the struct Send; access is serialised by the surrounding Mutex).
    pub(super) residency: Option<usize>,
    /// Reusable staging for activations in and outputs back: bump-allocated
    /// slices bound by buffer offset, so a dispatch creates no buffer objects.
    pub(super) x_arena: Buffer,
    pub(super) x_cap: usize,
    pub(super) y_arena: Buffer,
    pub(super) y_cap: usize,
    /// Fused-prefill residual stream and normed activations: xs lives here
    /// for the whole chunk, so norms, residual adds and the MoE combine
    /// never round-trip through the CPU.
    pub(super) pf_x: Buffer,
    pub(super) pf_x_cap: usize,
    pub(super) pf_active_len: Option<usize>,
    pub(super) pf_spare: std::sync::Arc<Mutex<Option<Buffer>>>,
    pub(super) pf_hs: Buffer,
    pub(super) pf_hs_cap: usize,
    /// The fused FFN combine already wrote the NEXT layer's attn-normed
    /// rows into pf_hs (combine_rows_resnorm): the next fused attention /
    /// GDN block skips its norm_rows dispatch. Cell because the norm
    /// closures in those blocks hold a shared borrow of `gpu`.
    pub(super) pf_hs_normed: std::cell::Cell<bool>,
    /// attend_mm_fq producer/consumer sync: one per-row-block counter
    /// (producer threadgroups publish one increment each), zeroed before
    /// every fused dispatch. Per-block because a single global count can be
    /// reached by later blocks finishing first, letting consumers read
    /// unwritten K/V.
    pub(super) pf_attn_sync: Buffer,
    /// GPU-side MoE routing scratch: counters, the group table, picks,
    /// gathered token ids and the combine CSR, all produced on-device.
    pub(super) route_buf: Buffer,
    pub(super) route_cap: usize,
    /// Debug: (table word-offset, n_expert) of the last fused routing, for
    /// ALLPAKA_ROWS_DEBUG to dump the per-expert row histogram after prefill.
    pub(super) dbg_route: std::cell::Cell<Option<(usize, usize)>>,
    /// The bias pointer (0 = none) route_buf's counters were last zeroed
    /// and staged for; route_scan self-cleans them on the GPU between
    /// layers, so CPU re-staging is needed only on realloc or a bias change
    /// (per-layer CPU writes would race the deferred buffers).
    pub(super) route_staged: Option<usize>,
    /// Fused-prefill deferred commits: layer buffers chain through this
    /// event instead of one CPU wait per layer. `pf_ev_val` is the last
    /// signalled value; `pf_pending` holds committed buffers until their
    /// GPU clock stats are reaped (bounded run-ahead).
    pub(super) pf_ev: metal::SharedEvent,
    pub(super) pf_ev_val: u64,
    pub(super) pf_pending: std::collections::VecDeque<metal::CommandBuffer>,
    /// ALLPAKA_PF_ONEBUF: one-buffer prefill SEGMENTS. The shared command
    /// buffer is armed lazily by the first eligible layer of the chunk
    /// (chained behind the last committed buffer through `pf_ev`) and
    /// sealed - committed chained - by the first layer that cannot join
    /// it, so e.g. GLM's leading Dense layer splits the chunk into
    /// segments instead of disabling the mode. A still-open segment is
    /// committed once from prefill_end. Each stage still gets its own
    /// sequential encoder.
    pub(super) pf_obuf_cmd: Option<metal::CommandBuffer>,
    /// ALLPAKA_GPU_COUNTERS=1: per-dispatch GPUTimestamp sampling through
    /// the fused prefill (the only counter set this device exposes). Index
    /// and labels are Cell/RefCell so encode closures borrowing `gpu`
    /// immutably can still stamp.
    pub(super) cstamps: Option<metal::CounterSampleBuffer>,
    pub(super) cstamps_dst: Option<metal::Buffer>,
    pub(super) cstamp_idx: std::cell::Cell<usize>,
    pub(super) cstamp_labels: std::cell::RefCell<Vec<&'static str>>,
    pub(super) decode_cache: Option<CachedDecodeLayers>,
}

/// route_buf layout in f32/u32 units (4-byte words).
pub(super) struct RouteLayout {
    pub(super) counts: usize,
    pub(super) counters: usize,
    pub(super) table: usize,
    pub(super) picks: usize,
    pub(super) pickw: usize,
    pub(super) tok: usize,
    pub(super) hit_row: usize,
    pub(super) hit_w: usize,
    pub(super) tok_off: usize,
    pub(super) logits: usize,
    pub(super) bias: usize,
    pub(super) tiles: usize,
    pub(super) tile_bound: usize,
    pub(super) total: usize,
}

pub(super) fn route_layout(
    m: usize,
    n_expert: usize,
    n_used: usize,
    shared: bool,
    tile_w: usize,
) -> RouteLayout {
    let align = |n: usize| (n + 63) & !63;
    let hits = n_used + shared as usize;
    let counts = 0;
    let counters = counts + align(n_expert);
    let table = counters + align(n_expert);
    let picks = table + align(n_expert * 3);
    let pickw = picks + align(m * n_used);
    let tok = pickw + align(m * n_used);
    let hit_row = tok + align(m * n_used);
    let hit_w = hit_row + align(m * hits);
    let tok_off = hit_w + align(m * hits);
    let logits = tok_off + align(m + 1);
    let bias = logits + align(m * n_expert);
    // Tight mmid grid: one descriptor per non-empty 32-row chunk.
    // ceil sum bound: total/32 + one partial chunk per expert.
    let tile_bound = n_expert + (m * n_used) / tile_w;
    let tiles = bias + align(n_expert);
    let total = tiles + align(tile_bound * 3);
    RouteLayout {
        counts,
        counters,
        table,
        picks,
        pickw,
        tok,
        hit_row,
        hit_w,
        tok_off,
        logits,
        bias,
        tiles,
        tile_bound,
        total,
    }
}

// Metal objects are reference-counted ObjC objects; the global below is
// guarded by a Mutex, so cross-thread use is serialised.
unsafe impl Send for Gpu {}

pub(super) static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

/// Wrap a model's mmap for the GPU. Call once per opened file; every later
/// `QuantMat::matmul` whose bytes live inside an attached mapping runs on
/// Metal automatically. Several mappings can be attached - speculative
/// decode runs the draft and the target on the same device - and each call
/// adds windows for its mapping. Returns false when no Metal device exists.
pub fn attach(mapping: &[u8]) -> bool {
    // Diagnostic hook: force the pure CPU path to bisect GPU-vs-engine bugs.
    if std::env::var_os("ALLPAKA_NO_GPU").is_some() {
        return false;
    }
    let Some(cell) = GPU.get_or_init(|| init_device().map(Mutex::new)) else {
        return false;
    };
    let mut gpu = match cell.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if gpu.covers(mapping.as_ptr() as usize, mapping.len()) {
        return true; // already attached (tests attach the same region twice)
    }
    gpu.add_mapping(mapping)
}

pub fn is_attached() -> bool {
    GPU.get().is_some_and(Option::is_some)
}

/// Active physical memory architecture for the Metal backend.
///
/// Apple Silicon GPUs use Unified Memory Architecture (`UMA`): CPU and GPU
/// share physical RAM, and `StorageModeShared` buffers created with
/// `newBufferWithBytesNoCopy` alias host allocations directly without H2D/D2H
/// copies.
#[inline]
pub fn memory_architecture() -> crate::memory::MemoryArchitecture {
    if is_attached() {
        crate::memory::MemoryArchitecture::Uma
    } else {
        crate::memory::MemoryArchitecture::HostOnly
    }
}

pub(super) static INIT_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Why the Metal device was not attached: no device, a kernel compile error
/// or a missing pipeline. Tests report it; the first failure is otherwise
/// only printed into whichever test happened to initialise the device.
pub fn init_error() -> Option<String> {
    INIT_ERROR.lock().ok().and_then(|e| e.clone())
}

pub(super) fn init_failed(message: String) {
    eprintln!("metal: {message}");
    if let Ok(mut slot) = INIT_ERROR.lock() {
        slot.get_or_insert(message);
    }
}

/// Runtime state that determines whether large mapped models avoid Metal's
/// per-submit residency scan. The values are reported by benchmarks so a
/// missing residency set cannot masquerade as slow kernels.
pub fn residency_status() -> (usize, bool) {
    let Some(Some(cell)) = GPU.get() else {
        return (0, false);
    };
    cell.lock()
        .map(|gpu| (gpu.chunks.len(), gpu.residency.is_some()))
        .unwrap_or((0, false))
}

/// Total bytes allocated in persistent GPU activation, prefill, and routing
/// arenas on unified memory.
pub fn arena_bytes() -> usize {
    let Some(Some(cell)) = GPU.get() else {
        return 0;
    };
    cell.lock()
        .map(|gpu| gpu.x_cap + gpu.y_cap + gpu.pf_x_cap + gpu.pf_hs_cap + gpu.route_cap)
        .unwrap_or(0)
}

pub(super) static ROPE_INV_FREQ: Mutex<Vec<f32>> = Mutex::new(Vec::new());

pub fn set_rope_inv_freq(inv_freq: &[f32]) -> bool {
    if inv_freq.is_empty() {
        return false;
    }
    if let Ok(mut lock) = ROPE_INV_FREQ.lock() {
        *lock = inv_freq.to_vec();
        true
    } else {
        false
    }
}

pub fn has_device_rope_freq() -> bool {
    ROPE_INV_FREQ.lock().is_ok_and(|f| !f.is_empty())
}

pub fn prefer_gpu_embed() -> bool {
    is_attached()
        && !std::env::var("ALLPAKA_GPU_EMBED")
            .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
}

pub fn can_embed_on_gpu(ty: GgmlType, bytes: &[u8], n_in: usize) -> bool {
    if !prefer_gpu_embed()
        || n_in == 0
        || n_in % 4 != 0
        || dq_fmt(ty).is_none()
        || row_bytes(ty, n_in).is_none()
    {
        return false;
    }
    let Some(Some(cell)) = GPU.get().map(|c| c.as_ref()) else {
        return false;
    };
    let Ok(gpu) = cell.lock() else {
        return false;
    };
    gpu.chunk_for(bytes.as_ptr() as usize, bytes.len())
        .is_some()
}

pub fn prefill_replay(_xs: &[f32]) -> Option<Vec<f32>> {
    None
}

/// Largest request a chunked mapping can serve; chunks overlap by this much.
/// Tensors in real GGUFs top out in the hundreds of megabytes (the 235B's
/// biggest is ~0.5 GiB), so 2 GiB of overlap leaves a wide margin.
pub(super) const CHUNK_OVERLAP: usize = 2 << 30;

pub(super) fn init_device() -> Option<Gpu> {
    let Some(device) = Device::system_default() else {
        init_failed("no system default Metal device".into());
        return None;
    };
    let library = device
        .new_library_with_source(KERNELS, &metal::CompileOptions::new())
        .map_err(|e| init_failed(format!("kernel compile failed, staying on CPU: {e}")))
        .ok()?;
    let mut pipelines = HashMap::new();
    // These two take no specialisation; they are stored under the unit shape.
    for name in [
        "swiglu",
        "attend",
        "attend_s8",
        "attend_s16",
        "attend_s32",
        "attend_mv",
        "attend_split",
        "attend_merge",
        "qk_prep",
        "qk_prep_batch",
        "attend_rows",
        "attend_rows_t4",
        "attend_rows_t8",
        "attend_mm",
        "attend_mm_db",
        "attend_mm_fq",
        "attend_mm_fq_db",
        "residual_norm",
        "residual_norm_sig",
        "norm_rows",
        "combine_rows",
        "combine_rows_resnorm",
        "softmax_topk",
        "sigmoid_topk",
        "router_topk",
        "moe_combine",
        "moe_combine_rows",
        "combine_resnorm",
        "resnorm_router",
        "resnorm_router_rows",
        "mmllr64_q4_k",
        "mmllp_q4_k",
        "mmllp_q8_0",
        "mmllp_q6_k",
        "mmllp_qkv_q4_k",
        "mmllp_qkv_q8_0",
        "mmllp_qkv_q4k_q6k",
        "matvec_qkv_q4k_q6k",
        "mmllpg_id_q4_k",
        "mmllps_id_q4_k",
        "mmllps_id_q6_k",
        "mmllpg_id_q4_k_bn16",
        "mmllps_id_q4_k_bn16",
        "route_pick",
        "route_scan",
        "route_scatter",
        "route_fused",
        "argmax_f32",
        "argmax_final",
        "embed_row_f32_dtoken",
        "zero_f32",
        "store_u32_at",
        "copy_f32",
        "gdn_conv",
        "gdn_step",
        "gdn_out_norm",
        "qk_prep256",
        "attend_s32_256",
        "attend_split_256",
        "attn_gate_mul",
        "gdn_conv_batch",
        "gdn_step_batch",
        "gdn_out_norm_batch",
        "qk_prep_batch256",
        "attend_rows256",
        "attend_mm256",
        "route_patch_shared_w",
    ] {
        let f = library
            .get_function(name, None)
            .map_err(|e| init_failed(format!("get_function {name} failed: {e}")))
            .ok()?;
        let p = device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| init_failed(format!("pipeline {name} failed: {e}")))
            .ok()?;
        pipelines.insert((name, 1, 1), p);
    }

    // Arenas start big enough for a decode layer and grow on demand.
    let x_arena = device.new_buffer(1 << 22, MTLResourceOptions::StorageModeShared);
    let y_arena = device.new_buffer(1 << 22, MTLResourceOptions::StorageModeShared);
    let pf_x = device.new_buffer(1 << 12, MTLResourceOptions::StorageModeShared);
    let pf_hs = device.new_buffer(1 << 12, MTLResourceOptions::StorageModeShared);
    let route_buf = device.new_buffer(1 << 12, MTLResourceOptions::StorageModeShared);
    let pf_attn_sync = device.new_buffer(256, MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(pf_attn_sync.contents() as *mut u8, 0, 256) };
    let pf_ev = device.new_shared_event();

    Some(Gpu {
        queue: device.new_command_queue(),
        device,
        pipelines,
        library,
        chunks: Vec::new(),
        residency: None,
        x_arena,
        x_cap: 1 << 22,
        y_arena,
        y_cap: 1 << 22,
        pf_x,
        pf_x_cap: 1 << 12,
        pf_active_len: None,
        pf_spare: Default::default(),
        pf_hs,
        pf_hs_cap: 1 << 12,
        pf_hs_normed: std::cell::Cell::new(false),
        pf_attn_sync,
        route_buf,
        route_cap: 1 << 12,
        dbg_route: std::cell::Cell::new(None),
        route_staged: None,
        pf_ev,
        pf_ev_val: 0,
        pf_pending: std::collections::VecDeque::new(),
        pf_obuf_cmd: None,
        cstamps: None,
        cstamps_dst: None,
        cstamp_idx: std::cell::Cell::new(0),
        cstamp_labels: std::cell::RefCell::new(Vec::new()),
        decode_cache: None,
    })
}

impl Gpu {
    /// Whether `[addr, addr + len)` is wholly inside some attached window.
    pub(super) fn covers(&self, addr: usize, len: usize) -> bool {
        self.chunk_for(addr, len).is_some()
    }

    /// Cover one model's mmap with overlapping `bytesNoCopy` windows.
    ///
    /// The mmap base is page-aligned by definition; lengths must be whole
    /// pages for bytesNoCopy. The kernel maps whole pages anyway, so a
    /// rounded tail is readable zeroes, never touched by valid offsets.
    ///
    /// One buffer cannot exceed the device's maxBufferLength (36 GiB class
    /// on Apple Silicon). newBufferWithBytesNoCopy does NOT error past it -
    /// it returns nil, and kernels then read garbage. Large mappings
    /// therefore get several overlapping chunk windows, and every dispatch
    /// picks the window that wholly contains its tensor.
    pub(super) fn add_mapping(&mut self, mapping: &[u8]) -> bool {
        if mapping.is_empty() {
            return true;
        }
        let page = 16384usize;
        let base = mapping.as_ptr() as usize;
        if base % page != 0 {
            let buf = self.device.new_buffer_with_data(
                mapping.as_ptr() as *const _,
                mapping.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            if buf.as_ptr().is_null() {
                return false;
            }
            let added = vec![WeightChunk {
                buf,
                start: base,
                len: mapping.len(),
            }];
            self.ensure_residency(&added);
            self.chunks.extend(added);
            return true;
        }
        let aligned_len = mapping.len().div_ceil(page) * page;
        // Stay well under the device's maxBufferLength: a buffer AT the
        // limit (80.6 GiB here) comes back non-nil yet reads wrong - the
        // 235B produced garbage logits until its mapping was split into
        // windows this size. Extra windows cost nothing; they alias pages.
        const WINDOW_CAP: usize = 32 << 30;
        let mut max_buf = (self.device.max_buffer_length() as usize / page) * page;
        max_buf = max_buf.min(WINDOW_CAP);
        // Test hook: cap the window size to exercise multi-window mapping on
        // small regions (the real trigger needs an 80+ GiB file).
        if let Ok(cap) = std::env::var("ALLPAKA_GPU_WINDOW_GIB") {
            if let Ok(gib) = cap.parse::<usize>() {
                max_buf = max_buf.min(gib << 30);
            }
        }
        if max_buf == 0 {
            return false;
        }
        let step = if max_buf > CHUNK_OVERLAP {
            max_buf - CHUNK_OVERLAP
        } else {
            max_buf / 2
        };
        let step = (step / page).max(1) * page;

        let mut added = Vec::new();
        let mut start = 0usize;
        loop {
            let len = max_buf.min(aligned_len - start);
            let buf = self.device.new_buffer_with_bytes_no_copy(
                (base + start) as *const _,
                len as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            if buf.as_ptr().is_null() {
                eprintln!(
                    "metal: bytesNoCopy declined a {:.1} GiB window, staying on CPU",
                    len as f64 / (1u64 << 30) as f64
                );
                return false;
            }
            added.push(WeightChunk {
                buf,
                start: base + start,
                len,
            });
            if start + len >= aligned_len {
                break;
            }
            start += step;
        }

        println!(
            "metal: {} attached, {:.1} GiB of weights shared with the GPU{}",
            self.device.name(),
            mapping.len() as f64 / (1u64 << 30) as f64,
            if added.len() > 1 {
                format!(
                    " in {} windows (maxBufferLength {:.1} GiB)",
                    added.len(),
                    max_buf as f64 / (1u64 << 30) as f64
                )
            } else {
                String::new()
            }
        );
        // Pin the new windows in a residency set attached to the queue.
        // Without one, the driver re-evaluates residency for the 83 GiB of
        // bytesNoCopy windows around submissions - llama.cpp pins its
        // buffers the same way (its init log: "use residency sets = true").
        self.ensure_residency(&added);
        self.chunks.extend(added);
        true
    }

    /// Add buffers to the queue's residency set, creating it on first use.
    /// Best effort: on failure the buffers still work, just unpinned.
    pub(super) fn ensure_residency(&mut self, bufs: &[WeightChunk]) {
        use objc::runtime::{Class, Object};
        use objc::{msg_send, sel, sel_impl};
        unsafe {
            if self.residency.is_none() {
                // Residency sets arrived in macOS 15; class! would panic on
                // an older system, which must simply run unpinned.
                let Some(class) = Class::get("MTLResidencySetDescriptor") else {
                    return;
                };
                let desc: *mut Object = msg_send![class, new];
                if desc.is_null() {
                    return;
                }
                let mut err: *mut Object = std::ptr::null_mut();
                let set: *mut Object = msg_send![
                    self.device.as_ptr() as *mut Object,
                    newResidencySetWithDescriptor: desc
                    error: &mut err
                ];
                let _: () = msg_send![desc, release];
                if set.is_null() {
                    return;
                }
                let queue = self.queue.as_ptr() as *mut Object;
                let _: () = msg_send![queue, addResidencySet: set];
                for arena in [
                    &self.x_arena,
                    &self.y_arena,
                    &self.pf_x,
                    &self.pf_hs,
                    &self.route_buf,
                ] {
                    let _: () = msg_send![set, addAllocation: arena.as_ptr() as *mut Object];
                }
                self.residency = Some(set as usize);
            }
            if let Some(set) = self.residency {
                let set = set as *mut Object;
                for c in bufs {
                    let _: () = msg_send![set, addAllocation: c.buf.as_ptr() as *mut Object];
                }
                let _: () = msg_send![set, commit];
                let _: () = msg_send![set, requestResidency];
            }
        }
    }

    /// The pipeline for one (kernel, tile, lanes-per-row) shape, specialised
    /// and cached on first use. Returns None if the shape will not compile,
    /// which sends the whole batch to the CPU rather than half of it.
    pub(super) fn pipeline(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
    ) -> Option<&ComputePipelineState> {
        self.pipeline_full(name, tile, lpr, false, false)
    }

    /// The full form: `indexed` compiles the expert-indexed variant,
    /// `swiglu` the down-projection variant that applies silu(gate)*up on
    /// its activation loads.
    pub(super) fn pipeline_full(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
        indexed: bool,
        swiglu: bool,
    ) -> Option<&ComputePipelineState> {
        self.pipeline_wait(name, tile, lpr, indexed, swiglu, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn pipeline_wait(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
        indexed: bool,
        swiglu: bool,
        wait: bool,
    ) -> Option<&ComputePipelineState> {
        // The llama-structure kernel is single-column; multi-row tiles fall
        // back to the block-per-lane kernel, which is correct at any LPR.
        let name = if tile != 1 {
            match name {
                "matvec_q2_k_mv" => "matvec_q2_k",
                "matvec_q3_k_mv" => "matvec_q3_k",
                "matvec_q4_k_mv" => "matvec_q4_k",
                "matvec_q5_k_mv" => "matvec_q5_k",
                "matvec_q6_k_mv" => "matvec_q6_k",
                "matvec_q8_0_mv" => "matvec_q8_0",
                other => other,
            }
        } else {
            name
        };
        let idg = indexed && tile == 1 && name.ends_with("_mv") && mv_id_grid();
        let key = (
            name,
            tile,
            lpr + if indexed { 1000 } else { 0 }
                + if swiglu { 2000 } else { 0 }
                + if wait { 4000 } else { 0 }
                + if idg { 80000 } else { 0 },
        );
        if !self.pipelines.contains_key(&key) {
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [(0u64, tile as u32), (1, lpr as u32)] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            let ind = indexed;
            consts.set_constant_value_at_index(
                &ind as *const bool as *const _,
                metal::MTLDataType::Bool,
                2,
            );
            let sw = swiglu;
            consts.set_constant_value_at_index(
                &sw as *const bool as *const _,
                metal::MTLDataType::Bool,
                3,
            );
            let wt = wait;
            consts.set_constant_value_at_index(
                &wt as *const bool as *const _,
                metal::MTLDataType::Bool,
                4,
            );
            // NB: an UNSET bool function constant reads as TRUE on this
            // Metal (measured, macOS 26 / M4 Max) - DUAL_GW (5), ROWS (6),
            // SHARED_TAIL (7), and MV_ID_GRID (8) must be pinned explicitly.
            for index in [5u64, 6, 7] {
                let f = false;
                consts.set_constant_value_at_index(
                    &f as *const bool as *const _,
                    metal::MTLDataType::Bool,
                    index,
                );
            }
            consts.set_constant_value_at_index(
                &idg as *const bool as *const _,
                metal::MTLDataType::Bool,
                8,
            );
            let f = self
                .library
                .get_function(name, Some(consts))
                .map_err(|e| eprintln!("metal: {name}<tile {tile}, lanes {lpr}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// The dual gate/up variant of an INDEXED matvec: function constant 5
    /// (DUAL_GW) on top of the plain indexed form.
    pub(super) fn pipeline_dual(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
    ) -> Option<&ComputePipelineState> {
        let key = (name, tile, lpr + 8000);
        if !self.pipelines.contains_key(&key) {
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [(0u64, tile as u32), (1, lpr as u32)] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            for (index, value) in [
                (2u64, true),
                (3, false),
                (4, false),
                (5, true),
                (6, false),
                (7, false),
                (8, false),
            ] {
                consts.set_constant_value_at_index(
                    &value as *const bool as *const _,
                    metal::MTLDataType::Bool,
                    index,
                );
            }
            let f = self
                .library
                .get_function(name, Some(consts))
                .map_err(|e| eprintln!("metal: {name}<dual, lanes {lpr}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// The ROWS variant of an INDEXED matvec (MTP verify): function constant
    /// 6, indexed always on, tile always 1. Only the kernels whose source
    /// carries the ROWS mapping may be requested - the host checks against
    /// ROWS_KERNELS.
    pub(super) fn pipeline_rows(
        &mut self,
        name: &'static str,
        lpr: usize,
        swiglu: bool,
    ) -> Option<&ComputePipelineState> {
        let key = (name, 1, lpr + 16000 + if swiglu { 2000 } else { 0 });
        if !self.pipelines.contains_key(&key) {
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [(0u64, 1u32), (1, lpr as u32)] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            for (index, value) in [
                (2u64, true),
                (3, swiglu),
                (4, false),
                (5, false),
                (6, true),
                (7, false),
                (8, false),
            ] {
                consts.set_constant_value_at_index(
                    &value as *const bool as *const _,
                    metal::MTLDataType::Bool,
                    index,
                );
            }
            let f = self
                .library
                .get_function(name, Some(consts))
                .map_err(|e| eprintln!("metal: {name}<rows, lanes {lpr}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// INDEXED matvec with SHARED_TAIL: last slot reads w2/w2_off (GLM
    /// shared expert down folded into the routed down launch).
    pub(super) fn pipeline_shared_tail(
        &mut self,
        name: &'static str,
        lpr: usize,
        swiglu: bool,
    ) -> Option<&ComputePipelineState> {
        let key = (
            name,
            1,
            lpr + 32000 + if swiglu { 2000 } else { 0 } + if mv_id_grid() { 80000 } else { 0 },
        );
        if !self.pipelines.contains_key(&key) {
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [(0u64, 1u32), (1, lpr as u32)] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            for (index, value) in [
                (2u64, true),
                (3, swiglu),
                (4, false),
                (5, false),
                (6, false),
                (7, true),
                (8, mv_id_grid()),
            ] {
                consts.set_constant_value_at_index(
                    &value as *const bool as *const _,
                    metal::MTLDataType::Bool,
                    index,
                );
            }
            let f = self
                .library
                .get_function(name, Some(consts))
                .map_err(|e| eprintln!("metal: {name}<shared_tail, lanes {lpr}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// cached under the kernel name with the formats folded into the key.
    pub(super) fn mega_pipeline(
        &mut self,
        gate_fmt: u32,
        down_fmt: u32,
        sh_gate_fmt: u32,
        sh_down_fmt: u32,
    ) -> Option<&ComputePipelineState> {
        // Four format nibbles packed into the two key slots the pipelines
        // map allows.
        let key = (
            "moe_ffn_mega",
            (gate_fmt | (sh_gate_fmt << 4)) as usize,
            (down_fmt | (sh_down_fmt << 4)) as usize,
        );
        if !self.pipelines.contains_key(&key) {
            let t0 = std::time::Instant::now();
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [
                (10u64, gate_fmt),
                (11, down_fmt),
                (12, sh_gate_fmt),
                (13, sh_down_fmt),
            ] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            let f = self
                .library
                .get_function("moe_ffn_mega", Some(consts))
                .map_err(|e| eprintln!("metal: moe_ffn_mega<{gate_fmt},{down_fmt},{sh_gate_fmt},{sh_down_fmt}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            if mega_debug() {
                eprintln!(
                    "mega: specialized pipeline gate={gate_fmt} down={down_fmt} in {:.1} ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// Index of the chunk window wholly containing the absolute range
    /// `[addr, addr + len)`, across every attached mapping.
    pub(super) fn chunk_for(&self, addr: usize, len: usize) -> Option<usize> {
        self.chunks
            .iter()
            .position(|c| addr >= c.start && addr + len <= c.start + c.len)
    }

    pub(super) fn pin_buffer_in_residency(&mut self, buf: &Buffer) {
        use objc::runtime::Object;
        use objc::{msg_send, sel, sel_impl};
        if let Some(set) = self.residency {
            unsafe {
                let set = set as *mut Object;
                let _: () = msg_send![set, addAllocation: buf.as_ptr() as *mut Object];
                let _: () = msg_send![set, commit];
                let _: () = msg_send![set, requestResidency];
            }
        }
    }

    pub(super) fn ensure_prefill(&mut self, bytes: usize) {
        if bytes > self.pf_x_cap {
            self.prefill_drain();
            let cap = bytes.next_power_of_two();
            let buf = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pin_buffer_in_residency(&buf);
            self.pf_x = buf;
            self.pf_x_cap = cap;
        }
        if bytes > self.pf_hs_cap {
            self.prefill_drain();
            let cap = bytes.next_power_of_two();
            let buf = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pin_buffer_in_residency(&buf);
            self.pf_hs = buf;
            self.pf_hs_cap = cap;
        }
    }

    pub(super) fn ensure_arenas(&mut self, x_need: usize, y_need: usize) {
        if x_need > self.x_cap {
            self.prefill_drain();
            let cap = x_need.next_power_of_two();
            let buf = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pin_buffer_in_residency(&buf);
            self.x_arena = buf;
            self.x_cap = cap;
        }
        if y_need > self.y_cap {
            self.prefill_drain();
            let cap = y_need.next_power_of_two();
            let buf = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pin_buffer_in_residency(&buf);
            self.y_arena = buf;
            self.y_cap = cap;
        }
    }

    pub(super) fn ensure_route(&mut self, words: usize) {
        let bytes = words * 4;
        if bytes > self.route_cap {
            self.prefill_drain();
            let cap = bytes.next_power_of_two();
            let buf = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pin_buffer_in_residency(&buf);
            self.route_buf = buf;
            self.route_cap = cap;
            self.route_staged = None;
        }
    }

    /// Deferred commit for the fused prefill: signal the chain event and
    /// reap the oldest pending buffer past the run-ahead window, collecting
    /// its GPU clock stats.
    pub(super) fn commit_chained(&mut self, cmd: &metal::CommandBufferRef) {
        self.pf_ev_val += 1;
        cmd.encode_signal_event(&self.pf_ev, self.pf_ev_val);
        cmd.commit();
        self.pf_pending.push_back(cmd.to_owned());
        while self.pf_pending.len() > 4 {
            let old = self.pf_pending.pop_front().expect("len checked");
            old.wait_until_completed();
            note_gpu_times(&old);
        }
    }

    /// Seal the open one-buffer prefill segment: commit it chained, so a
    /// layer that cannot join the shared buffer (or a split-timing probe)
    /// runs after it in event order, and the next eligible layer lazily
    /// re-arms a fresh segment.
    pub(super) fn pf_obuf_seal(&mut self) {
        if let Some(cmd) = self.pf_obuf_cmd.take() {
            self.commit_chained(&cmd);
        }
    }

    /// Lazily arm the chunk's shared one-buffer command buffer, chained
    /// behind the last committed buffer (the wait is already satisfied
    /// right after `prefill_begin`'s drain). Returns without arming when
    /// a segment is already open.
    pub(super) fn pf_obuf_arm(&mut self) {
        if self.pf_obuf_cmd.is_none() {
            // Retained: the per-stage encoders are autoreleased per call
            // site, the segment's command buffer must outlive them all.
            let cmd = self.queue.new_command_buffer().to_owned();
            cmd.encode_wait_for_event(&self.pf_ev, self.pf_ev_val);
            self.pf_obuf_cmd = Some(cmd);
        }
    }

    /// Wait out every deferred prefill buffer: prefill_end, a mid-chunk
    /// fallback, or any CPU access to arenas the GPU may still be writing.
    pub(super) fn prefill_drain(&mut self) {
        while let Some(old) = self.pf_pending.pop_front() {
            old.wait_until_completed();
            note_gpu_times(&old);
        }
    }

    /// ALLPAKA_GPU_COUNTERS=1: (re)arm the per-dispatch timestamp sampling
    /// for a fused prefill chunk. Lazily creates the sample buffer on the
    /// only counter set this device has.
    pub(super) fn cstamps_begin(&mut self) {
        if !gpu_counters() {
            return;
        }
        if self.cstamps.is_none() {
            if !gpu_counters_supported(&self.device) {
                eprintln!(
                    "gpu counters: atDispatchBoundary sampling unsupported on this device, off"
                );
                return;
            }
            let Some(set) = self
                .device
                .counter_sets()
                .into_iter()
                .find(|s| s.name() == "timestamp")
            else {
                eprintln!("gpu counters: no timestamp counter set");
                return;
            };
            let desc = metal::CounterSampleBufferDescriptor::new();
            desc.set_counter_set(&set);
            desc.set_sample_count(CSTAMP_CAP as u64);
            desc.set_storage_mode(metal::MTLStorageMode::Shared);
            match self.device.new_counter_sample_buffer_with_descriptor(&desc) {
                Ok(buf) => {
                    self.cstamps_dst = Some(self.device.new_buffer(
                        (CSTAMP_CAP * 8) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ));
                    self.cstamps = Some(buf);
                }
                Err(e) => eprintln!("gpu counters: sample buffer failed: {e}"),
            }
        }
        self.cstamp_idx.set(0);
        self.cstamp_labels.borrow_mut().clear();
    }

    /// Resolve the chunk's stamps and print the per-label timing summary.
    /// The interval before stamp i is attributed to the dispatch stamp i-1
    /// names; ns/tick is calibrated by the chunk buffer's GPU start/end when
    /// available (else a 1 GHz assumption).
    pub(super) fn cstamps_report(&mut self, cal: Option<(f64, f64)>) {
        let Some(buf) = &self.cstamps else {
            return;
        };
        let n = self.cstamp_idx.get().min(CSTAMP_CAP);
        if n < 2 {
            return;
        }
        let dst = self.cstamps_dst.as_ref().expect("dst with buf").clone();
        let cmd = self.queue.new_command_buffer().to_owned();
        let be = cmd.new_blit_command_encoder();
        be.resolve_counters(buf, metal::NSRange::new(0, n as u64), &dst, 0);
        be.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ts = unsafe { std::slice::from_raw_parts(dst.contents() as *const u64, n).to_vec() };
        let labels = self.cstamp_labels.borrow();
        let scale = match cal {
            Some((gs, ge)) if ts[n - 1] > ts[0] => (ge - gs) * 1e9 / (ts[n - 1] - ts[0]) as f64,
            _ => 1.0,
        };
        let mut total: std::collections::HashMap<&str, (f64, u32, f64)> =
            std::collections::HashMap::new();
        for i in 1..n {
            let dt = (ts[i] - ts[i - 1]) as f64 * scale / 1000.0;
            let l = labels.get(i - 1).copied().unwrap_or("?");
            let e = total.entry(l).or_default();
            e.0 += dt;
            e.1 += 1;
            e.2 = e.2.max(dt);
        }
        let sum: f64 = total.values().map(|v| v.0).sum();
        eprintln!("gpu counters: {n} stamps, {sum:.2} ms attributed");
        let mut v: Vec<_> = total.into_iter().collect();
        v.sort_by(|a, b| (b.1).0.total_cmp(&(a.1).0));
        for (l, (s, c, mx)) in v {
            eprintln!(
                "  {l:<6} {s:8.3} ms  n={c:<3} mean {:.3} max {:.3}  ({:.1}%)",
                s / c as f64,
                mx,
                s / sum * 100.0
            );
        }
    }
}
