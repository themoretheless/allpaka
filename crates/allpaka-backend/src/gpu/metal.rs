//! The Metal path: fused dequant-matvec kernels on Apple GPUs.
//!
//! Only matrix-vector products go to the GPU - they are where every byte and
//! every FLOP of decode lives. Norms, RoPE and softmax stay on the CPU: they
//! touch kilobytes, and a dispatch costs more than they do.
//!
//! Weights are never copied. The GGUF mmap is wrapped in `bytesNoCopy` shared
//! buffers (Apple Silicon memory is unified; the GPU reads the same pages the
//! page cache holds), and kernels address tensors by byte offset. A mapping
//! larger than the device's `maxBufferLength` is covered by several
//! overlapping chunk buffers - one giant buffer would silently come back nil
//! and every kernel would read garbage, which is exactly how an 88 GiB model
//! produced NaN logits while a 17 GiB one verified clean.
//!
//! The kernels mirror `quantmat.rs` line for line. The CPU implementation is
//! the reference; `verify` against llama.cpp arbitrates both.

#![cfg(target_os = "macos")]

use metal::foreign_types::ForeignType;
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use allpaka_gguf::GgmlType;

/// Where GPU wall time goes, accumulated across every call. The split that
/// matters: `wait_ns` is dead time the CPU spends blocked on the GPU, and its
/// ratio to `calls` is the per-round-trip cost every extra command buffer pays.
static CALLS: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// y_arena f32 offset of the router logits the last fused attention block
/// produced, when GPU-side routing skipped the CPU readback.
static PF_ROUTER_AT: AtomicU64 = AtomicU64::new(u64::MAX);
static ENCODE_NS: AtomicU64 = AtomicU64::new(0);
static WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// GPU-side execution time (GPUEndTime - GPUStartTime summed over buffers)
/// and the scheduling delay before it (GPUStartTime - kernelStartTime).
/// The difference between WAIT_NS and their sum is time the CPU spent
/// blocked while the GPU had nothing of ours to run - the number that says
/// whether decode is bound by kernels or by getting work to them.
static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);
static SCHED_NS: AtomicU64 = AtomicU64::new(0);

/// `(gpu_busy_ns, sched_ns)` since process start.
pub fn gpu_time_stats() -> (u64, u64) {
    (
        GPU_BUSY_NS.load(Ordering::Relaxed),
        SCHED_NS.load(Ordering::Relaxed),
    )
}

/// Effective GB/s of one INDEXED matvec (slots experts × n_out rows).
/// Used by `gpu_glm_mvbench` to A/B `ALLPAKA_MV_ID` geometry vs plain.
/// `w` must hold at least `slots` contiguous expert matrices of shape
/// `[n_out, n_in]` (same layout as decode expert packs).
pub fn indexed_matvec_bandwidth(
    w: &[u8],
    ty: GgmlType,
    n_out: usize,
    n_in: usize,
    slots: usize,
    rounds: u32,
) -> Option<f64> {
    let gpu_m = GPU.get()?.as_ref()?;
    let mut gpu = gpu_m.lock().ok()?;
    let mat = resolve(&gpu, ty, w)?;
    let lpr = lanes_per_row(ty, n_in);
    let state = gpu
        .pipeline_full(mat.kernel, 1, lpr, true, false)?
        .to_owned();
    let expert_bytes = match ty {
        GgmlType::Q4K => n_out * (n_in / 256) * 144,
        GgmlType::Q5K => n_out * (n_in / 256) * 176,
        GgmlType::Q8_0 => n_out * (n_in / 32) * 34,
        _ => return None,
    };
    if w.len() < expert_bytes.saturating_mul(slots.max(1)) {
        return None;
    }
    let stride = expert_bytes as u64;
    // Stage x / ids / y in the shared arenas.
    let x_elems = n_in.max(slots);
    let y_elems = n_out * slots + slots;
    gpu.ensure_arenas(x_elems * 4, y_elems * 4);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut f32;
        for i in 0..n_in {
            *xp.add(i) = 0.01;
        }
        let yp = gpu.y_arena.contents() as *mut u32;
        for s in 0..slots {
            *yp.add(s) = (s % slots.max(1)) as u32;
        }
    }
    let before = gpu_time_stats();
    objc::rc::autoreleasepool(|| {
        for _ in 0..rounds {
            let cmd = gpu.queue.new_command_buffer();
            let enc =
                cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
            enc.set_compute_pipeline_state(&state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(&gpu.x_arena), 0);
            enc.set_buffer(2, Some(&gpu.y_arena), (slots * 4) as u64);
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(&gpu.y_arena), 0);
            let idx = GpuIdxArgs {
                stride,
                slots: slots as u32,
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
            dispatch_mv_indexed(enc, ty, n_in, n_out, slots);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
        }
    });
    let after = gpu_time_stats();
    let busy = (after.0 - before.0) as f64 / 1e9;
    if busy <= 0.0 {
        return None;
    }
    let total = (expert_bytes * slots) as f64 * rounds as f64;
    Some(total / busy / 1e9)
}

/// Read the completed buffer's own timestamps. CFTimeIntervals in seconds on
/// a shared clock, so differences are valid; zeros (no GPU work) are skipped.
fn note_gpu_times(cmd: &metal::CommandBufferRef) {
    use objc::{msg_send, sel, sel_impl};
    let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
    let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
    let ks: f64 = unsafe { msg_send![cmd, kernelStartTime] };
    if ge > gs && gs > 0.0 {
        GPU_BUSY_NS.fetch_add(((ge - gs) * 1e9) as u64, Ordering::Relaxed);
        if gs > ks && ks > 0.0 {
            SCHED_NS.fetch_add(((gs - ks) * 1e9) as u64, Ordering::Relaxed);
        }
    }
}

/// `(calls, dispatches, encode_ns, wait_ns)` since process start.
pub fn stats() -> (u64, u64, u64, u64) {
    (
        CALLS.load(Ordering::Relaxed),
        DISPATCHES.load(Ordering::Relaxed),
        ENCODE_NS.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed),
    )
}

pub use super::metal_kernels::KERNELS;

/// One `bytesNoCopy` window over the weight mapping. Chunks overlap by more
/// than any single tensor, so every request fits wholly inside some chunk.
struct WeightChunk {
    buf: Buffer,
    /// Absolute address of this chunk's first byte. Chunks from different
    /// attached mappings coexist in one list; an absolute range is what lets
    /// a weight slice find its window no matter which model it belongs to.
    start: usize,
    len: usize,
}

pub struct Gpu {
    device: Device,
    queue: CommandQueue,
    /// One pipeline per (kernel, tile, lanes-per-row): both are function
    /// constants, so each shape is its own specialisation. The full cross
    /// product is 4 tiles x 6 lane counts x 6 formats, and a given model uses
    /// a handful of those, so they are compiled on first use rather than at
    /// attach. `swiglu` ignores both and is stored under (1, 1).
    pipelines: HashMap<(&'static str, usize, usize), ComputePipelineState>,
    library: metal::Library,
    chunks: Vec<WeightChunk>,
    /// The queue's MTLResidencySet, retained as a raw pointer (usize keeps
    /// the struct Send; access is serialised by the surrounding Mutex).
    residency: Option<usize>,
    /// Reusable staging for activations in and outputs back: bump-allocated
    /// slices bound by buffer offset, so a dispatch creates no buffer objects.
    x_arena: Buffer,
    x_cap: usize,
    y_arena: Buffer,
    y_cap: usize,
    /// Fused-prefill residual stream and normed activations: xs lives here
    /// for the whole chunk, so norms, residual adds and the MoE combine
    /// never round-trip through the CPU.
    pf_x: Buffer,
    pf_x_cap: usize,
    pf_active_len: Option<usize>,
    pf_spare: std::sync::Arc<Mutex<Option<Buffer>>>,
    pf_hs: Buffer,
    pf_hs_cap: usize,
    /// GPU-side MoE routing scratch: counters, the group table, picks,
    /// gathered token ids and the combine CSR, all produced on-device.
    route_buf: Buffer,
    route_cap: usize,
    /// The bias pointer (0 = none) route_buf's counters were last zeroed
    /// and staged for; route_scan self-cleans them on the GPU between
    /// layers, so CPU re-staging is needed only on realloc or a bias change
    /// (per-layer CPU writes would race the deferred buffers).
    route_staged: Option<usize>,
    /// Fused-prefill deferred commits: layer buffers chain through this
    /// event instead of one CPU wait per layer. `pf_ev_val` is the last
    /// signalled value; `pf_pending` holds committed buffers until their
    /// GPU clock stats are reaped (bounded run-ahead).
    pf_ev: metal::SharedEvent,
    pf_ev_val: u64,
    pf_pending: std::collections::VecDeque<metal::CommandBuffer>,
    /// ALLPAKA_PF_ONEBUF: one-buffer prefill SEGMENTS. The shared command
    /// buffer is armed lazily by the first eligible layer of the chunk
    /// (chained behind the last committed buffer through `pf_ev`) and
    /// sealed - committed chained - by the first layer that cannot join
    /// it, so e.g. GLM's leading Dense layer splits the chunk into
    /// segments instead of disabling the mode. A still-open segment is
    /// committed once from prefill_end. Each stage still gets its own
    /// sequential encoder.
    pf_obuf_cmd: Option<metal::CommandBuffer>,
    /// ALLPAKA_GPU_COUNTERS=1: per-dispatch GPUTimestamp sampling through
    /// the fused prefill (the only counter set this device exposes). Index
    /// and labels are Cell/RefCell so encode closures borrowing `gpu`
    /// immutably can still stamp.
    cstamps: Option<metal::CounterSampleBuffer>,
    cstamps_dst: Option<metal::Buffer>,
    cstamp_idx: std::cell::Cell<usize>,
    cstamp_labels: std::cell::RefCell<Vec<&'static str>>,
    decode_cache: Option<CachedDecodeLayers>,
}

/// route_buf layout in f32/u32 units (4-byte words).
struct RouteLayout {
    counts: usize,
    counters: usize,
    table: usize,
    picks: usize,
    pickw: usize,
    tok: usize,
    hit_row: usize,
    hit_w: usize,
    tok_off: usize,
    logits: usize,
    bias: usize,
    total: usize,
}

fn route_layout(m: usize, n_expert: usize, n_used: usize, shared: bool) -> RouteLayout {
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
    let total = bias + align(n_expert);
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
        total,
    }
}

// Metal objects are reference-counted ObjC objects; the global below is
// guarded by a Mutex, so cross-thread use is serialised.
unsafe impl Send for Gpu {}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

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

static INIT_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Why the Metal device was not attached: no device, a kernel compile error
/// or a missing pipeline. Tests report it; the first failure is otherwise
/// only printed into whichever test happened to initialise the device.
pub fn init_error() -> Option<String> {
    INIT_ERROR.lock().ok().and_then(|e| e.clone())
}

fn init_failed(message: String) {
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

static ROPE_INV_FREQ: Mutex<Vec<f32>> = Mutex::new(Vec::new());

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
const CHUNK_OVERLAP: usize = 2 << 30;

fn init_device() -> Option<Gpu> {
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
        "residual_norm",
        "residual_norm_sig",
        "norm_rows",
        "combine_rows",
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
        "mmllpg_id_q4_k",
        "mmllps_id_q4_k",
        "route_pick",
        "route_scan",
        "route_scatter",
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
        route_buf,
        route_cap: 1 << 12,
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
    fn covers(&self, addr: usize, len: usize) -> bool {
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
    fn add_mapping(&mut self, mapping: &[u8]) -> bool {
        let page = 16384usize;
        let base = mapping.as_ptr() as usize;
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
    fn ensure_residency(&mut self, bufs: &[WeightChunk]) {
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
    fn pipeline(
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
    fn pipeline_full(
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
    fn pipeline_wait(
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
    fn pipeline_dual(
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
    fn pipeline_rows(
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
    fn pipeline_shared_tail(
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
    fn mega_pipeline(
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
    fn chunk_for(&self, addr: usize, len: usize) -> Option<usize> {
        self.chunks
            .iter()
            .position(|c| addr >= c.start && addr + len <= c.start + c.len)
    }

    fn pin_buffer_in_residency(&mut self, buf: &Buffer) {
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

    fn ensure_prefill(&mut self, bytes: usize) {
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

    fn ensure_arenas(&mut self, x_need: usize, y_need: usize) {
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

    fn ensure_route(&mut self, words: usize) {
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
    fn commit_chained(&mut self, cmd: &metal::CommandBufferRef) {
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
    fn pf_obuf_seal(&mut self) {
        if let Some(cmd) = self.pf_obuf_cmd.take() {
            self.commit_chained(&cmd);
        }
    }

    /// Lazily arm the chunk's shared one-buffer command buffer, chained
    /// behind the last committed buffer (the wait is already satisfied
    /// right after `prefill_begin`'s drain). Returns without arming when
    /// a segment is already open.
    fn pf_obuf_arm(&mut self) {
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
    fn prefill_drain(&mut self) {
        while let Some(old) = self.pf_pending.pop_front() {
            old.wait_until_completed();
            note_gpu_times(&old);
        }
    }

    /// ALLPAKA_GPU_COUNTERS=1: (re)arm the per-dispatch timestamp sampling
    /// for a fused prefill chunk. Lazily creates the sample buffer on the
    /// only counter set this device has.
    fn cstamps_begin(&mut self) {
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
    fn cstamps_report(&mut self, cal: Option<(f64, f64)>) {
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
        v.sort_by(|a, b| (b.1).0.partial_cmp(&(a.1).0).unwrap());
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

/// One matmul request inside a batch: `x` holds `m` activation rows.
pub struct MatvecReq<'a> {
    pub ty: GgmlType,
    pub w: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
    pub x: &'a [f32],
    pub m: usize,
}

/// How many lanes cooperate on one output row, given how many quant blocks
/// that row holds. The lanes each take whole blocks, so more lanes than
/// blocks is pure waste; fewer means a lane walks several blocks, which is
/// fine. Rounded up to a power of two so the shuffle reduction stays inside
/// its lane group, and capped at the SIMD width.
fn lanes_per_row(ty: GgmlType, n_in: usize) -> usize {
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
fn mv_tg() -> u64 {
    static TG: OnceLock<u64> = OnceLock::new();
    *TG.get_or_init(|| {
        std::env::var("ALLPAKA_MV_TG")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &u64| matches!(n, 64 | 128 | 256))
            .unwrap_or(128)
    })
}

/// Llama-style `mul_mv_id` grid for INDEXED decode matvecs (`grid.z = slots`).
/// Default ON. `ALLPAKA_MV_ID=0` restores flat `n_out * slots` 1D packing.
fn mv_id_grid() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("ALLPAKA_MV_ID").map_or(true, |v| v != "0"))
}

/// Dispatch geometry for a matvec pipeline (LPR × rows, TG via `mv_tg`).
fn dispatch_mv(enc: &metal::ComputeCommandEncoderRef, ty: GgmlType, n_in: usize, n_rows: usize) {
    let lpr = lanes_per_row(ty, n_in) as u64;
    let tg = mv_tg();
    enc.dispatch_thread_groups(
        MTLSize::new((n_rows as u64 * lpr).div_ceil(tg), 1, 1),
        MTLSize::new(tg, 1, 1),
    );
}

/// Indexed MoE matvec: either llama `mul_mv_id` (`z = slots`) or flat 1D.
fn dispatch_mv_indexed(
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
const MM_MIN_M: usize = 16;

/// The tile-matmul kernel for a format, and its output-tile height.
///
/// Two compiled heights exist, 32 and 64 rows; 64 reads each staged
/// activation tile against twice the weight rows. `ALLPAKA_MM64` switches at
/// startup so the two can be swept as whole-bench A/B runs.
/// `(kernel name, BM, BN)`. Three compiled tile shapes: 32x32, 64x32
/// (`ALLPAKA_MM64`, measured worse) and 32x64 (`ALLPAKA_MM_BN64`), the
/// wide-batch tile that halves how often the weights are re-read per pass.
fn mm_kernel_for(ty: GgmlType, m: usize) -> Option<(&'static str, usize, usize)> {
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

/// The compiled TILE specialisations, largest last. A batch is split into
/// tiles from this set, so every dispatch covers exactly the rows it claims.
const TILES: &[usize] = &[1, 2, 4, 8];

/// Split `m` rows into `(start, tile)` dispatches, taking the largest
/// available tile first. Exact by construction: the tiles are powers of two
/// down to 1, so the remainder is always coverable and no dispatch ever
/// touches a row past `m`.
fn tiles_for(m: usize) -> Vec<(usize, usize)> {
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

fn matvec_batch_inner(
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

/// A CPU allocation the GPU reads in place.
///
/// The KV cache is written by the CPU every step and read by the attention
/// kernel; on unified memory neither needs a copy, so the cache is allocated
/// page-aligned and wrapped here exactly like the weight mapping is. Holding
/// the `Buffer` keeps the wrapper alive: dropping this releases it, and the
/// caller still owns the memory.
pub struct SharedRegion {
    buf: Buffer,
    len: usize,
}

impl SharedRegion {
    /// Memory architecture backing this region (`UMA` zero-copy shared memory).
    #[inline]
    pub const fn architecture(&self) -> crate::memory::MemoryArchitecture {
        crate::memory::MemoryArchitecture::Uma
    }

    /// True because Metal `StorageModeShared` aliases the host allocation in
    /// unified physical memory without separate device VRAM storage.
    #[inline]
    pub const fn is_unified(&self) -> bool {
        true
    }

    /// False on UMA: host and GPU share the same physical bytes.
    #[inline]
    pub const fn is_discrete(&self) -> bool {
        false
    }
}

// The buffer is an ObjC refcounted handle; the region it wraps is owned by
// the caller, who is responsible for not writing it while a kernel reads it
// (the forward pass stores the step, then attends, in that order).
unsafe impl Send for SharedRegion {}
unsafe impl Sync for SharedRegion {}

/// Wrap a page-aligned allocation for the GPU. `region` must start on a page
/// boundary and span whole pages; anything else is declined rather than
/// silently mapped short.
pub fn wrap_region(region: &[u8]) -> Option<SharedRegion> {
    const PAGE: usize = 16384;
    if region.is_empty() || region.as_ptr() as usize % PAGE != 0 || region.len() % PAGE != 0 {
        return None;
    }
    let gpu = GPU.get()?.as_ref()?.lock().ok()?;
    let buf = gpu.device.new_buffer_with_bytes_no_copy(
        region.as_ptr() as *const _,
        region.len() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    if buf.as_ptr().is_null() {
        return None;
    }
    Some(SharedRegion {
        buf,
        len: region.len(),
    })
}

pub fn upload_region_range(region: &mut SharedRegion, offset: usize, data: &[u8]) -> bool {
    if offset.saturating_add(data.len()) > region.len {
        return false;
    }
    if data.is_empty() {
        return true;
    }
    unsafe {
        let dst = (region.buf.contents() as *mut u8).add(offset);
        if !std::ptr::eq(dst, data.as_ptr()) {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }
    true
}

pub fn download_region_range(region: &SharedRegion, offset: usize, dst: &mut [u8]) -> bool {
    if offset.saturating_add(dst.len()) > region.len {
        return false;
    }
    if dst.is_empty() {
        return true;
    }
    unsafe {
        let src = (region.buf.contents() as *const u8).add(offset);
        if !std::ptr::eq(src, dst.as_ptr()) {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
    }
    true
}

pub fn decode_attention_capacity_safe(_capacity: usize) -> bool {
    true
}

pub fn minimum_kv_capacity() -> usize {
    1
}

/// One decode step's attention over the cache, for every query head.
///
/// `k_off` and `v_off` are *element* offsets of this layer's K and V inside
/// the wrapped region of halves; from there the cache is `[pos][kv_dim]`.
pub struct AttnReq<'a> {
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub q: &'a [f32],
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub group: usize,
    pub n_pos: usize,
    pub scale: f32,
}

/// The kernel assumes a lane owns four dimensions of a 128-wide head.
const ATTN_HEAD_DIM: usize = 128;

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
fn attn_shape_ok(req: &AttnReq) -> bool {
    req.head_dim == ATTN_HEAD_DIM
        && req.n_pos > 0
        && req.group > 0
        && req.q.len() == req.n_q_heads * req.head_dim
        && (req.k_off + req.n_pos * req.kv_dim) * 2 <= req.cache.len
        && (req.v_off + req.n_pos * req.kv_dim) * 2 <= req.cache.len
}

/// Encode attention into an open encoder, q already staged in the x arena and
/// the result landing at `y_off` in the y arena.
fn encode_attend(enc: &metal::ComputeCommandEncoderRef, gpu: &Gpu, req: &AttnReq, y_off: usize) {
    encode_attend_at(enc, gpu, req, 0, y_off)
}

fn encode_attend_at(
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
fn encode_attend_from(
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

/// The gated-delta-net branch of a qwen35moe layer (linear attention):
/// replaces the whole attention half of the token (no KV cache, the
/// recurrence state lives in the shared SSM region instead).
pub struct TokenGdn<'a> {
    /// hidden -> conv channels (key_dim*2 + value_dim), Q8_0.
    pub wqkv: (GgmlType, &'a [u8], usize),
    /// hidden -> value_dim (the z gate), Q8_0.
    pub zgate: (GgmlType, &'a [u8], usize),
    /// F32 [hidden -> heads_v] projections, raw bytes in the mmap.
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    /// F32 conv weights [channels * d_conv], raw bytes in the mmap.
    pub conv1d: &'a [u8],
    /// Small F32 vectors handed to the kernel by value.
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    /// value_dim -> hidden, Q8_0.
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    /// Element offsets of this layer's conv window and deltanet state
    /// inside the SSM region.
    pub conv_off: usize,
    pub state_off: usize,
}

/// One decode layer's description for the whole-token encoder.
pub struct TokenLayer<'a> {
    /// Raw F32 norm weights, borrowed from the mmap (16 KB exceeds setBytes).
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// qwen35moe: the sigmoid output gate is fused into the q projection
    /// ([q | gate] per head); split and applied post-attention on the GPU.
    pub gate_in_q: bool,
    /// Raw F32 Q/K/V biases in the mmap (GLM-4); None for bias-free models.
    pub q_bias: Option<&'a [u8]>,
    pub k_bias: Option<&'a [u8]>,
    pub v_bias: Option<&'a [u8]>,
    /// Element offsets of this layer's K and V in the cache region.
    pub k_off: usize,
    pub v_off: usize,
    /// Present on linear-attention layers: the whole attention half is the
    /// gated-delta-net branch, and the attention fields above go unused.
    pub gdn: Option<TokenGdn<'a>>,
    pub ffn: TokenFfn<'a>,
}

pub enum TokenFfn<'a> {
    Dense {
        gate: (GgmlType, &'a [u8], usize),
        up: (GgmlType, &'a [u8], usize),
        down: (GgmlType, &'a [u8], usize),
    },
    Moe {
        /// F32 router `[n_expert, hidden]`.
        router: (GgmlType, &'a [u8], usize),
        /// Raw F32 router bias (GLM `exp_probs_b`); sigmoid gating only.
        router_bias: Option<&'a [u8]>,
        /// Stacked expert tensors; per-expert geometry alongside.
        gate: (GgmlType, &'a [u8]),
        up: (GgmlType, &'a [u8]),
        down: (GgmlType, &'a [u8]),
        expert_ffn: usize,
        n_used: usize,
        /// Sigmoid gating (GLM-4) instead of softmax; weights already
        /// renormalized over the top-k by the caller's contract.
        sigmoid: bool,
        /// Optional shared expert, same ffn width as a routed expert; runs
        /// as an extra combine slot with weight 1.
        shared: Option<[(GgmlType, &'a [u8], usize); 3]>,
        /// qwen35moe: the shared expert's combine weight is not a constant
        /// 1 but sigmoid(gate . h), an F32 [hidden] vector in the mmap. The
        /// raw logit lands in the slot's weight and the combine kernel
        /// applies the sigmoid.
        shared_gate: Option<&'a [u8]>,
    },
}

pub struct TokenReq<'a> {
    /// The embedded input rows, `m * hidden` floats (`m` = 1 for decode).
    pub x: &'a [f32],
    /// Tokens to run through the layers in one buffer. m > 1 is the
    /// speculative-verify path: batched projections (TILE = m), per-token
    /// attention and MoE, the GDN batch kernels.
    pub m: usize,
    pub layers: &'a [TokenLayer<'a>],
    pub cache: &'a SharedRegion,
    pub cache_capacity: usize,
    /// qwen35moe: the SSM region holding every GDN layer's conv window and
    /// deltanet state (f32 elements), wrapped like the KV cache.
    pub ssm: Option<&'a SharedRegion>,
    /// MTP verification: per-row rollback slots written by the GDN batch
    /// kernels (region, slot stride in elements); None for plain decode.
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// First token's position; token i lands at pos + i.
    pub pos: usize,
    pub scale: f32,
    /// `(sin, cos)` pairs per rotary pair per token: `m * rot_dim / 2`.
    pub rope: &'a [[f32; 2]],
    /// Rotary width; equals head_dim for full rope, head_dim/2 for GLM-4,
    /// head_dim/4 for qwen35moe's 64-of-256 partial rope.
    pub rot_dim: usize,
    pub eps: f32,
    pub output_norm: &'a [u8],
    pub output: (GgmlType, &'a [u8], usize),
    /// Greedy caller: run the argmax on the GPU after the output projection
    /// and read back 4 bytes instead of the whole vocabulary.
    pub argmax: bool,
    pub embd: Option<(GgmlType, &'a [u8], usize)>,
    pub token_id: Option<u32>,
}

/// What decode_token produced: the full vocabulary logits, or just the
/// greedy winner when the request asked for the on-GPU argmax. For a
/// multi-token request: the per-row argmax and the output-normed hidden
/// rows (the speculative verifier's arbitration inputs and the next
/// round's draft seed).
pub enum TokenOut {
    Logits(Vec<f32>),
    Argmax(u32),
    Rows { argmax: Vec<u32>, hidden: Vec<f32> },
}

/// A resolved norm bind: which chunk window holds the F32 weights.
#[derive(Clone, Copy)]
struct NormRef {
    chunk: usize,
    off: u64,
}

/// F32 weights for the router: resolve() deliberately excludes F32 so the
/// general matmul paths never route tiny test tensors to a kernel that
/// assumes float4-divisible rows; the whole-token path opts in here, with
/// that assumption checked.
fn resolve_f32(gpu: &Gpu, w: &[u8], n_in: usize) -> Option<MatRef> {
    if n_in % 4 != 0 || w.as_ptr() as usize % 4 != 0 {
        return None;
    }
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some(MatRef {
        kernel: "matvec_f32",
        ty: GgmlType::F32,
        chunk,
        w_off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

fn resolve_norm(gpu: &Gpu, raw: &[u8], hidden: usize) -> Option<NormRef> {
    if raw.len() != hidden * 4 {
        return None;
    }
    let addr = raw.as_ptr() as usize;
    if addr % 4 != 0 {
        return None;
    }
    let chunk = gpu.chunk_for(addr, raw.len())?;
    Some(NormRef {
        chunk,
        off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

#[repr(C)]
struct GpuMegaArgs {
    hidden: u32,
    ffn: u32,
    n_expert: u32,
    n_used: u32,
    x_at: u32,
    delta_at: u32,
    h_at: u32,
    logits_at: u32,
    ids_at: u32,
    wts_at: u32,
    gate_at: u32,
    up_at: u32,
    downo_at: u32,
    ctr_at: u32,
    n_tg: u32,
    ctr_base: u32,
    eps: f32,
    _pad: u32,
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    router_off: u64,
    gate_stride: u64,
    up_stride: u64,
    down_stride: u64,
    sigmoid: u32,
    /// 0 = none, 1 = always-on weight 1, 2 = gated via `sh_gout_off`.
    has_shared: u32,
    sh_gate_off: u64,
    sh_up_off: u64,
    sh_down_off: u64,
    sh_gout_off: u64,
    /// Metal `set_bytes` length should be a multiple of 16; keep trailing pad.
    _tail: u64,
}

/// How many threadgroups the megakernel launches.
///
/// Multi-TG uses a device atomic busy-wait that soft-locks the Metal
/// scheduler and can freeze the whole Mac. Until a non-spinning sync exists,
/// this is hard-capped at 1 (local barrier only). `ALLPAKA_MEGA_TG` is ignored.
fn mega_tg() -> u32 {
    1
}

fn mega_debug() -> bool {
    static M: OnceLock<bool> = OnceLock::new();
    *M.get_or_init(|| std::env::var("ALLPAKA_MEGA_DEBUG").is_ok_and(|v| v == "1"))
}

/// MEGA stays off. A live multi-TG device-atomic `mega_sync` soft-locked the
/// Metal scheduler and froze a Mac; the spin was deleted and `mega_tg` is 1,
/// but the fused path is still unfinished. Do not re-enable via env until a
/// non-spinning redesign + a test that cannot wedge the GPU lands.
fn mega_enabled() -> bool {
    false
}

/// Format code for the megakernel's expert cores; None = unsupported.
fn mega_fmt(ty: GgmlType) -> Option<u32> {
    match ty {
        GgmlType::Q2K => Some(0),
        GgmlType::Q3K => Some(1),
        GgmlType::Q4K => Some(2),
        GgmlType::Q5_0 => Some(3),
        GgmlType::Q5K => Some(4),
        GgmlType::Q8_0 => Some(5),
        _ => None,
    }
}

#[repr(C)]
struct GpuIdxArgs {
    stride: u64,
    slots: u32,
    x_stride: u32,
    // ROWS (MTP verify) only: per-token-row strides of the ids, x and y
    // blocks, plus the row count. Zero otherwise.
    ids_stride: u32,
    x_row_stride: u32,
    y_row_stride: u32,
    n_rows: u32,
}

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
struct GdnRefs {
    mats: [MatRef; 5], // wqkv, zgate, alpha, beta, ssm_out
    states: [ComputePipelineState; 5],
    conv1d: NormRef,
    heads_k: u32,
    heads_v: u32,
    d: u32,
    d_conv: u32,
    conv_off: u64,
    state_off: u64,
    channels: usize,
    value_dim: usize,
}

/// A resolved decode layer for the whole-token encoders.
#[derive(Clone)]
struct LayerRefs {
    attn_norm: NormRef,
    ffn_norm: NormRef,
    /// None on GDN layers (no attention mats).
    mats: Option<[MatRef; 4]>,
    mat_states: Option<[ComputePipelineState; 4]>,
    q_bias: Option<NormRef>,
    k_bias: Option<NormRef>,
    v_bias: Option<NormRef>,
    gdn: Option<GdnRefs>,
    ffn: FfnRefs,
    normflag: bool,
}

#[derive(Clone)]
struct CachedDecodeLayers {
    key: u64,
    layers: std::sync::Arc<[LayerRefs]>,
    max_ffn: usize,
    max_slots: usize,
    max_expert: usize,
}

#[allow(clippy::too_many_arguments)]
fn resolve_decode_layers(
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
            && mats_pre
                .as_ref()
                .is_some_and(|(m, _)| m.iter().take(3).all(|m| m.kernel == "matvec_q4_k_mv"))
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
                            | "matvec_q3_k_mv"
                            | "matvec_q4_k_mv"
                            | "matvec_q5_k_mv"
                            | "matvec_q8_0"
                            | "matvec_q8_0_mv"
                    )
                };
                let sw_fused = sw_capable(mats[2].kernel) && sw_fuse_env;
                let mut states = Vec::with_capacity(3);
                for (i, (mat, n_in)) in mats.iter().zip([hidden, hidden, gate.2]).enumerate() {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    let swiglu = i == 2 && sw_fused;
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
                            | "matvec_q3_k_mv"
                            | "matvec_q4_k_mv"
                            | "matvec_q5_k_mv"
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
                    let kernel = match mat.kernel {
                        "matvec_q2_k_mv" if n_out % 4 != 0 => "matvec_q2_k",
                        "matvec_q3_k_mv" if n_out % 2 != 0 => "matvec_q3_k",
                        "matvec_q4_k_mv" if n_out % 2 != 0 => "matvec_q4_k",
                        "matvec_q5_k_mv" if n_out % 2 != 0 => "matvec_q5_k",
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
enum FfnRefs {
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

/// A whole-token GPU request was rejected before submission. The checked API
/// keeps this distinct from a successful GPU result so callers cannot silently
/// mistake a CPU fallback for GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeDecline {
    pub stage: &'static str,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodePathStats {
    pub attempts: u64,
    pub successes: u64,
    pub declines: u64,
}

static DECODE_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_SUCCESSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_DECLINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn decode_path_stats() -> DecodePathStats {
    use std::sync::atomic::Ordering::Relaxed;
    DecodePathStats {
        attempts: DECODE_ATTEMPTS.load(Relaxed),
        successes: DECODE_SUCCESSES.load(Relaxed),
        declines: DECODE_DECLINES.load(Relaxed),
    }
}

impl std::fmt::Display for DecodeDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stage={} reason={}", self.stage, self.reason)
    }
}

impl std::error::Error for DecodeDecline {}

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

/// Backend-neutral checked decode contract. New callers should use this
/// rather than interpreting `None` from the compatibility API below.
pub fn decode_token_outcome(req: &TokenReq) -> crate::accel::AccelOutcome<TokenOut> {
    match decode_token_checked(req) {
        Ok(out) => crate::accel::AccelOutcome::Executed(out),
        Err(reason) => crate::accel::AccelOutcome::Declined(crate::accel::DeclineReason::Backend {
            operation: "decode-token",
            detail: reason.to_string(),
        }),
    }
}

pub fn decode_token(req: &TokenReq) -> Option<TokenOut> {
    let dbg = std::env::var_os("ALLPAKA_TOKENBUF_DEBUG").is_some();
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
        let split = std::env::var_os("ALLPAKA_DECODE_SPLIT").is_some();
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
                    matvec(&enc, &astates[0], &amats[0], hidden, l.wq.2, h_at, q_at);
                    matvec(&enc, &astates[1], &amats[1], hidden, kv, h_at, k_at);
                    matvec(&enc, &astates[2], &amats[2], hidden, kv, h_at, v_at);
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

fn dq_fmt(ty: GgmlType) -> Option<u32> {
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

fn row_bytes(ty: GgmlType, n_in: usize) -> Option<usize> {
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

/// The speculative multi-token verify: `m` tokens through the already
/// resolved layers in ONE command buffer with ONE wait. Projections batch
/// all m rows (TILE = m matvec pipelines, one weight pass); attention and
/// MoE run per token; the GDN recurrence takes the batch kernels, with
/// rollback slots written when armed. The matvec f32 dot per output row is
/// the decode path's numerics, so the arbitration rows stay bit-compatible
/// with plain greedy decode - unlike the mm (half-staged) prefill path,
/// whose drift flips near-tie argmaxes (measured: stream divergence).
fn encode_verify_tokens(
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
    let mut max_ffn = 1usize;
    let mut max_slots = 1usize;
    let mut max_expert = 1usize;
    let (mut max_channels, mut max_value_dim, mut max_heads_v) = (0usize, 0usize, 1usize);
    for l in layers {
        if let Some(g) = &l.gdn {
            max_channels = max_channels.max(g.channels);
            max_value_dim = max_value_dim.max(g.value_dim);
            max_heads_v = max_heads_v.max(g.heads_v as usize);
        }
        if let FfnRefs::Moe {
            expert_ffn,
            n_used,
            shared,
            n_expert,
            ..
        } = &l.ffn
        {
            max_ffn = max_ffn.max(*expert_ffn);
            max_slots = max_slots.max(*n_used + shared.is_some() as usize);
            max_expert = max_expert.max(*n_expert);
        }
    }
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
    let up_at = gate_at + align(m * max_slots * max_ffn);
    let downo_at = up_at + align(m * max_slots * max_ffn);
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

/// One decode layer's whole attention half as ONE command buffer: the three
/// qkv projections, per-head norm + RoPE, the f16 store of k/v into the
/// cache, attention over it, and the output projection. Five waits become
/// one, and by the GPU's own clock the driver's per-buffer scheduling was as
/// expensive as the round trips themselves.
///
/// NEOX rope only (what every current target model uses); anything else
/// declines to the CPU path.
pub struct AttnBlockReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    /// The normed layer input, `hidden` floats.
    pub x: &'a [f32],
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// `(sin, cos)` per NEOX pair, `head_dim / 2` entries.
    pub rope: &'a [[f32; 2]],
    pub eps: f32,
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub pos: usize,
    pub scale: f32,
}

#[repr(C)]
struct QkPrepArgs {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    eps: f32,
    pos: u32,
    has_qk_norm: u32,
    rot_dim: u32,
    k_base: u64,
    v_base: u64,
    has_bias: u32,
    pad2: u32,
}

#[repr(C)]
struct QkPrep256Args {
    n_heads: u32,
    n_kv_heads: u32,
    kv_dim: u32,
    eps: f32,
    pos: u32,
    has_qk_norm: u32,
    rot_dim: u32,
    gate_in_q: u32,
    k_base: u64,
    v_base: u64,
}

#[repr(C)]
struct GdnConvArgs {
    channels: u32,
    d_conv: u32,
    pad0: u32,
    pad1: u32,
    conv_off: u64,
}

#[repr(C)]
struct GdnStepArgs {
    heads_k: u32,
    heads_v: u32,
    d: u32,
    key_dim: u32,
    eps: f32,
    pad0: u32,
    state_off: u64,
}

#[repr(C)]
struct QkPrepBatch256Args {
    n_heads: u32,
    n_kv_heads: u32,
    kv_dim: u32,
    eps: f32,
    base: u32,
    has_qk_norm: u32,
    rot_dim: u32,
    gate_in_q: u32,
    k_base: u64,
    v_base: u64,
}

#[repr(C)]
struct GdnConvBatchArgs {
    channels: u32,
    d_conv: u32,
    m: u32,
    pad0: u32,
    conv_off: u64,
}

#[repr(C)]
struct GdnStepBatchArgs {
    heads_k: u32,
    heads_v: u32,
    d: u32,
    key_dim: u32,
    m: u32,
    eps: f32,
    state_off: u64,
}

pub fn attn_block(req: &AttnBlockReq) -> Option<Vec<f32>> {
    let hd = req.head_dim;
    if hd != ATTN_HEAD_DIM || req.rope.len() != hd / 2 {
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
            rot_dim: hd as u32,
            k_base: req.k_off as u64,
            v_base: req.v_off as u64,
            has_bias: 0,
            pad2: 0,
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

/// A prefill chunk's whole attention half as ONE command buffer: the three
/// qkv projections over all `m` rows (tile-matmul kernels), per-row norm +
/// rope + f16 cache store (qk_prep_batch), causal attention per row, and
/// the output projection. Mirrors the decode token buffer; the FFN half
/// stays separate because routing groups tokens on the CPU.
/// Extra weights for the fused prefill: with this present (and
/// `prefill_begin` called for the chunk), the block reads the residual
/// stream from GPU memory, norms on the GPU, folds the attention output
/// back in, re-norms for the FFN and finishes with the router matmul -
/// returning router logits instead of the projected rows.
pub struct PrefillFusion<'a> {
    /// F32 raw norm weights, `hidden` floats each.
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    /// F32 raw router weights, `n_expert * hidden` floats.
    pub router: &'a [u8],
    pub n_expert: usize,
}

/// Upload the chunk's residual stream for the fused prefill path.
/// Stamp a GPUTimestamp after a dispatch. No-op unless ALLPAKA_GPU_COUNTERS.
macro_rules! cstamp {
    ($gpu:expr, $enc:expr, $label:expr) => {
        if let Some(buf) = &$gpu.cstamps {
            let i = $gpu.cstamp_idx.get();
            if i < CSTAMP_CAP {
                $enc.sample_counters_in_buffer(buf, i as u64, false);
                $gpu.cstamp_labels.borrow_mut().push($label);
                $gpu.cstamp_idx.set(i + 1);
            }
        }
    };
}

pub fn prefill_begin(xs: &[f32]) -> Option<()> {
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    gpu.prefill_drain();
    gpu.cstamps_begin();
    // One-buffer segments arm lazily: the first eligible layer of the
    // chunk opens the shared command buffer (see prefill_attn_block), an
    // ineligible layer seals it again (pf_obuf_seal). Nothing to arm here.
    gpu.pf_obuf_cmd = None;
    gpu.ensure_prefill(xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), gpu.pf_x.contents() as *mut f32, xs.len());
    }
    gpu.pf_active_len = Some(xs.len());
    Some(())
}

/// Download the residual stream after the last fused layer.
/// Drain GPU writers before exposing CPU slices of the residual stream.
fn prefill_complete(gpu: &mut Gpu) {
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

static SHARED_OUTPUTS: AtomicU64 = AtomicU64::new(0);
static SHARED_HANDOFF_BYTES: AtomicU64 = AtomicU64::new(0);
static SHARED_PROJECTION_BYTES: AtomicU64 = AtomicU64::new(0);

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
    buf: Option<Buffer>,
    len: usize,
    spare: std::sync::Arc<Mutex<Option<Buffer>>>,
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
    if std::env::var("ALLPAKA_PREFILL_ZERO_COPY").is_ok_and(|v| v == "0") {
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
            gpu.prefill_drain();
            gpu.pf_active_len = None;
        }
    }
}

pub struct PrefillAttnReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    /// Normed layer input, `m * hidden` floats.
    pub hs: &'a [f32],
    pub m: usize,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// `(sin, cos)` pairs for every row: `m * rot_dim / 2` entries.
    pub ropes: &'a [[f32; 2]],
    /// Rotary width; == head_dim for full rope, smaller for GLM's partial,
    /// 64 for qwen35moe's 256-wide heads.
    pub rot_dim: usize,
    /// qwen35moe: the sigmoid output gate rides inside the q projection
    /// ([q | gate] per head, wq.n_out == 2 * q_dim); split and applied
    /// post-attention.
    pub gate_in_q: bool,
    /// GLM's per-head q/k/v additive biases as raw F32 GGUF bytes.
    pub attn_bias: Option<(&'a [u8], &'a [u8], &'a [u8])>,
    pub eps: f32,
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// First row's position; row i attends over base + i + 1 positions.
    pub base: usize,
    pub scale: f32,
    pub fusion: Option<PrefillFusion<'a>>,
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
    let attend_mm256_state = gpu.pipelines[&("attend_mm256", 1, 1)].to_owned();
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
        // ALLPAKA_PF_SPLIT=1: commit after every stage and print its GPU
        // time. Debug-only; serialises the stages, so wall time degrades.
        let split = std::env::var_os("ALLPAKA_PF_SPLIT").is_some();
        // One-buffer chunk (ALLPAKA_PF_ONEBUF=1, default): encode into the
        // chunk's shared command buffer - a sequential encoder per stage,
        // the commit happens once in prefill_end. The segment arms lazily
        // on the first eligible layer; a layer that stays off it seals the
        // open segment first, so its own buffer stays ordered after the
        // work already encoded there.
        let onebuf = fusion.is_some() && gpu_route() && pf_onebuf() && !split;
        if !onebuf {
            gpu.pf_obuf_seal();
        } else {
            gpu.pf_obuf_arm();
        }
        // Deferred commit: chain behind the previous layer's buffer through
        // the shared event instead of a CPU wait. Only the fused GPU-routing
        // path qualifies - it reads nothing back. PF_SPLIT keeps the old
        // waits (its per-stage commits are a different buffer shape anyway).
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
                    eprintln!("attn {}: {:.3} ms", $label, (ge - gs) * 1e3);
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
            norm_rows(enc, &f.attn_norm, 0, out_at);
            cstamp!(gpu, enc, "anorm");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
            }
            (&gpu.pf_hs, 0)
        } else {
            (&gpu.x_arena, hs_at)
        };
        matmul(
            enc, &states[0], &mats[0], hidden, wq_out, hs_buf, hs_off, q_at,
        );
        cstamp!(gpu, enc, "wq");
        matmul(enc, &states[1], &mats[1], hidden, kv, hs_buf, hs_off, k_at);
        cstamp!(gpu, enc, "wk");
        matmul(enc, &states[2], &mats[2], hidden, kv, hs_buf, hs_off, v_at);
        cstamp!(gpu, enc, "wv");
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
                pad2: 0,
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
                enc.set_compute_pipeline_state(&gpu.pipelines[&("attend_mm", 1, 1)]);
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
        cstamp!(gpu, enc, "wo");
        split_here!("wo");
        if let Some(f) = &fusion {
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
            // xs += wo output, then the FFN norm - one fused pass - and the
            // router matmul, all before the single wait.
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
        }
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
        if !defer && !onebuf && std::env::var_os("ALLPAKA_FFN_TIME").is_some() {
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

/// A prefill chunk of one gated-delta-net layer (qwen35moe linear
/// attention), mirroring [`prefill_attn_block`]'s fusion contract: reads the
/// residual stream from pf_x, leaves it updated there (branch output folded
/// into the FFN norm), the router logits in y_arena for GPU routing.
pub struct PrefillGdnReq<'a> {
    /// hidden -> conv channels (key_dim*2 + value_dim), Q8_0.
    pub wqkv: (GgmlType, &'a [u8], usize),
    /// hidden -> value_dim (the z gate), Q8_0.
    pub zgate: (GgmlType, &'a [u8], usize),
    /// F32 [hidden -> heads_v] projections, raw bytes in the mmap.
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    /// F32 conv weights [channels * d_conv], raw bytes.
    pub conv1d: &'a [u8],
    /// Small F32 vectors handed to the kernel by value.
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    /// value_dim -> hidden, Q8_0.
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    pub hidden: usize,
    pub m: usize,
    pub eps: f32,
    /// The shared SSM region: conv window and deltanet state, updated in
    /// place (element offsets per layer).
    pub ssm: &'a SharedRegion,
    /// MTP verification: per-row rollback slots (`SsmCache::arm_slots`),
    /// (region, slot stride in elements). The batch kernels then also write
    /// every row's state/window into the slots.
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub conv_off: usize,
    pub state_off: usize,
    pub fusion: Option<PrefillFusion<'a>>,
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
        let split = std::env::var_os("ALLPAKA_PF_SPLIT").is_some();
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
        norm_rows(enc, &f.attn_norm, 0, out_at);
        cstamp!(gpu, enc, "anorm");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
        }
        matmul(
            enc, &states[0], &mats[0], hidden, channels, &gpu.pf_hs, 0, qkv_at,
        );
        cstamp!(gpu, enc, "wqkv");
        matmul(
            enc, &states[1], &mats[1], hidden, value_dim, &gpu.pf_hs, 0, z_at,
        );
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
        // xs += ssm_out output, then the FFN norm and the router matmul.
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

/// A prefill chunk's whole routed FFN - every expert group's gate, up,
/// swiglu and down - as ONE command buffer with SIX dispatches, however many
/// experts were hit. `x` holds the gathered rows (each group's tokens
/// contiguous), `groups` maps each to its expert and row range. Returns the
/// down outputs in the same row layout.
pub struct GroupedFfnReq<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    pub n_expert: usize,
    pub hidden: usize,
    pub ffn: usize,
    /// `[expert, row0, rows]` per hit group; row0 indexes into the flat row
    /// space. An array, not a tuple: this is copied verbatim into GPU memory
    /// and a tuple's layout is not guaranteed.
    pub groups: &'a [[u32; 3]],
    /// The UNGATHERED activations (`m` rows); `tok[row]` names each flat
    /// row's source token. The gather happens inside the gate/up kernels.
    pub x: &'a [f32],
    pub tok: &'a [u32],
    pub total_rows: usize,
    /// Fused prefill: read activations from the pf_hs buffer written by the
    /// preceding fused attention block, and scatter the weighted expert
    /// outputs into pf_x on the GPU (CSR hits per token) instead of
    /// returning them.
    pub fused: Option<GroupedCombine<'a>>,
    /// GLM's shared expert (fused path only): dense gate/up/down applied to
    /// every token. Its down rows land in the down region right after the
    /// `total_rows` expert rows, so the caller can reference them in the CSR
    /// hits with weight 1.0.
    pub shared: Option<GroupedShared<'a>>,
    /// GPU-side routing (fused path only): route_pick/scan/scatter build the
    /// group table, token gather and CSR hits on-device at the head of the
    /// FFN command buffer, from the router logits the attention block left
    /// in y_arena. `groups`/`tok`/CSR slices are ignored in this mode.
    pub route: Option<GroupedRoute<'a>>,
}

pub struct GroupedRoute<'a> {
    pub n_used: usize,
    /// Renormalise the winners to sum to 1 (`norm_topk_prob`).
    pub norm: bool,
    pub scale: f32,
    /// Selection bias (`+bias` shifts only the top-k choice).
    pub bias: Option<&'a [f32]>,
    /// Sigmoid gating (GLM) when true, softmax over all experts (Qwen3)
    /// when false. The softmax case is only exact without a selection bias
    /// and with `norm` set; the caller falls back to the CPU router
    /// otherwise.
    pub sigmoid: bool,
}

pub struct GroupedShared<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    /// Shared-expert FFN width (n_ff_exp * n_expert_shared).
    pub ffn: usize,
    /// qwen35moe: the shared expert's combine weight is sigmoid(gate . h)
    /// per token, an F32 [hidden] projection in the mmap. Its raw logits
    /// land in y; route_patch_shared_w folds them into the CSR weights.
    pub gate_out: Option<(GgmlType, &'a [u8])>,
}

pub struct GroupedCombine<'a> {
    /// CSR offsets, `m + 1` entries.
    pub tok_off: &'a [u32],
    /// Flat-row index and routing weight per hit.
    pub hit_row: &'a [u32],
    pub hit_w: &'a [f32],
    pub m: usize,
}

/// Deferred fused-prefill commits: layer buffers chain through an
/// MTLSharedEvent instead of one CPU wait per layer, so the CPU encodes
/// layer N+1 while the GPU still runs layer N. `ALLPAKA_PF_DEFER=0` reverts
/// to per-layer waits.
fn pf_defer() -> bool {
    crate::runtime::get().prefill_defer
}

/// The whole fused prefill chunk as ONE command buffer (sequential
/// encoders per stage, one commit in prefill_end). Default ON: measured
/// qwen3-30b pp480 1222-1230 vs 1196-1198 tok/s with the event chain
/// (pp1200: 1124 vs 1095) - pure buffer-start overhead removal, GPU
/// executing unchanged. `ALLPAKA_PF_ONEBUF=0` reverts to PF_DEFER chaining.
fn pf_onebuf() -> bool {
    crate::runtime::get().prefill_one_buffer
}

/// Join the shared prefill buffer without GPU-side MoE routing, for a grouped
/// FFN that has no route table and whose combine already lands in `pf_x`
/// (`req.fused.is_some()`: the download is skipped, so nothing is read back).
/// Dense models reach this. The `route_mode` gate guards the router-logits
/// rescue, which a dense layer has no logits for, so it excluded dense from the
/// segment over a hazard that cannot occur there. A CPU-routed MoE keeps the
/// old path: it is not what this was measured on.
/// `ALLPAKA_PF_OBUF_DENSE=0` reverts.
fn dense_obuf() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("ALLPAKA_PF_OBUF_DENSE").map_or(true, |v| v != "0"))
}

/// ALLPAKA_GPU_COUNTERS=1: per-dispatch GPUTimestamp sampling in the fused
/// prefill. This device exposes only the "timestamp" counter set, and only
/// atCommandBoundary sampling - per-dispatch compute-encoder samples assert
/// ("not supported on this device"), so the mode is a silent no-op here.
/// Kept as the probe harness for a device that supports it.
fn gpu_counters() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| std::env::var("ALLPAKA_GPU_COUNTERS").is_ok_and(|v| v == "1"))
}

/// Whether sampleCountersInBuffer works on compute encoders here
/// (MTLCounterSamplingPoint::atDispatchBoundary == 2).
fn gpu_counters_supported(device: &metal::DeviceRef) -> bool {
    use objc::{msg_send, sel, sel_impl};
    unsafe { msg_send![device, supportsCounterSampling: 2u64] }
}

/// Stamp capacity per prefill chunk (~16 dispatches x 48 layers x chunks).
const CSTAMP_CAP: usize = 1024;

/// The llama.cpp-structure q2_k matvec, `ALLPAKA_Q2_MV=1` to enable. OFF by
/// default: measured 95 GB/s against the word-load kernel's ~170 in
/// ffn-shaped decode (the word-load rewrite already beat llama's structure
/// for this format; the port only won for q3_k, whose old kernel still
/// assembled ushort pairs).
fn q2_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q2_MV").is_ok_and(|v| v == "1"))
}

/// Same switch for the q3_k llama-structure kernel: `ALLPAKA_Q3_MV=0` off.
fn q3_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q3_MV").map_or(true, |v| v != "0"))
}

fn q3_kernel() -> &'static str {
    if q3_mv() {
        "matvec_q3_k_mv"
    } else {
        "matvec_q3_k"
    }
}

/// The q4_k llama-structure matvec. Default ON: decode 30B 90 -> 102 tok/s,
/// 235B 22.2 -> 24.3 (q4_k attention weights). `ALLPAKA_Q4_MV=0` reverts.
fn q4_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q4_MV").map_or(true, |v| v != "0"))
}

fn q4_kernel() -> &'static str {
    if q4_mv() {
        "matvec_q4_k_mv"
    } else {
        "matvec_q4_k"
    }
}

/// The q6_k llama-structure matvec. Default ON after alternating 30B A/B on
/// M4 Max measured 133.4 versus 131.6 tok/s with decode parity preserved.
/// `ALLPAKA_Q6_MV=0` restores the reference kernel.
fn q6_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q6_MV").map_or(true, |v| v != "0"))
}

fn q6_kernel() -> &'static str {
    if q6_mv() {
        "matvec_q6_k_mv"
    } else {
        "matvec_q6_k"
    }
}

/// The q5_k llama-structure matvec. Default ON for GLM shared gate/up and
/// qwen35moe expert down (Q5_K). Carries SWIGLU_X so decode SWFUSE can fuse
/// the down projection. `ALLPAKA_Q5_MV=0` restores the reference
/// block-per-lane kernel. Needs Mac A/B confirmation on glm/qwen35 shapes.
fn q5_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q5_MV").map_or(true, |v| v != "0"))
}

fn q5_kernel() -> &'static str {
    if q5_mv() {
        "matvec_q5_k_mv"
    } else {
        "matvec_q5_k"
    }
}

/// The q8_0 llama-structure matvec. Default ON: GLM expert/shared down and
/// qwen35moe GDN/attn projections. Carries SWIGLU_X for decode fusion.
/// `ALLPAKA_Q8_MV=0` restores the packed-load reference kernel.
fn q8_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q8_MV").map_or(true, |v| v != "0"))
}

fn q8_kernel() -> &'static str {
    if q8_mv() {
        "matvec_q8_0_mv"
    } else {
        "matvec_q8_0"
    }
}

/// Prefill attention over simdgroup MMA tiles (attend_mm, llama-style: K/V
/// read directly by the MMAs, online softmax). Default ON: standalone
/// 0.278 vs 0.742 ms/dispatch for attend_rows_t8 at m=480 (qwen3-30b
/// shape), exact to 1e-5. `ALLPAKA_ATTN_MM=0` reverts.
fn attend_mm() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| std::env::var("ALLPAKA_ATTN_MM").map_or(true, |v| v != "0"))
}

/// Prefill attention row tiling: `ALLPAKA_ATTN_T4=0` reverts to the
/// row-per-threadgroup kernel.
fn attend_rows_kernel() -> &'static str {
    static T4: OnceLock<usize> = OnceLock::new();
    match *T4.get_or_init(|| {
        std::env::var("ALLPAKA_ATTN_T4")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    }) {
        8 => "attend_rows_t8",
        0 => "attend_rows",
        _ => "attend_rows_t4",
    }
}

/// The llama-structure decode attention. Default ON: four-position-in-flight
/// flash-attn vec path. `ALLPAKA_ATTN_MV=0` reverts to the position-per-step
/// kernel.
fn attend_kernel() -> &'static str {
    static MV: OnceLock<bool> = OnceLock::new();
    if *MV.get_or_init(|| std::env::var("ALLPAKA_ATTN_MV").map_or(true, |v| v != "0")) {
        return "attend_mv";
    }
    static SG: OnceLock<usize> = OnceLock::new();
    match *SG.get_or_init(|| {
        std::env::var("ALLPAKA_ATTN_S8")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32)
    }) {
        32 => "attend_s32",
        16 => "attend_s16",
        4 | 0 => "attend",
        _ => "attend_s8",
    }
}

/// Threads per attend threadgroup, matching the kernel's simdgroup count.
fn attend_tg() -> u64 {
    match attend_kernel() {
        "attend_s32" => 1024,
        "attend_s16" => 512,
        "attend_s8" => 256,
        _ => 128,
    }
}

/// Probe-before-surgery for decode: `ALLPAKA_SKIP=qkv,attend,experts,router,head`
/// drops those dispatch groups from the whole-token buffer. Values become
/// garbage; the timing is the honest cost of what remains.
fn probe_skip(what: &str) -> bool {
    static SKIP: OnceLock<Vec<String>> = OnceLock::new();
    SKIP.get_or_init(|| {
        std::env::var("ALLPAKA_SKIP")
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default()
    })
    .iter()
    .any(|s| s == what)
}

/// The decode q2_k kernel: the interleaved dual-block variant under its env.
fn q2_kernel() -> &'static str {
    static ILV: OnceLock<bool> = OnceLock::new();
    if *ILV.get_or_init(|| std::env::var_os("ALLPAKA_Q2_ILV").is_some()) {
        "matvec_q2_k_ilv"
    } else if q2_mv() {
        "matvec_q2_k_mv"
    } else {
        "matvec_q2_k"
    }
}

/// Whether the llama-structure mm kernels are routed to (shared switch for
/// mm_kernel_for and mmid_kernel_for). Default ON: measured 235B prefill
/// 111 -> 136 tok/s, verify prefill PASS with a smaller diff than the old
/// kernels. `ALLPAKA_MM_LL=0` reverts.
/// GPU-side MoE routing for the fused prefill: route_pick/scan/scatter run
/// at the head of the FFN command buffer instead of a CPU readback + top-k.
/// `ALLPAKA_GPU_ROUTE=0` reverts to the CPU router.
fn gpu_route() -> bool {
    crate::runtime::get().gpu_route
}

/// The decode-side combine fold into the next residual norm
/// (combine_resnorm). OFF by default: measured SLOWER on M4 Max
/// (qwen3-30b decode 78.5 -> 75.0 tok/s, +20 ms GPU executing over 32
/// tokens) - the single-threadgroup fused kernel walks all expert slots'
/// down rows with one thread's latency chain, which costs more than the
/// dispatch and barrier it removes. `ALLPAKA_CFUSE=1` re-enables.
fn cfuse() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| std::env::var("ALLPAKA_CFUSE").is_ok_and(|v| v == "1"))
}

/// The decode gate+up dual dispatch: one INDEXED launch covers both
/// matrices (DUAL_GW function constant). OFF by default: qwen3-30b measured
/// 105.0 tok/s versus 113.7 for two concurrent dispatches on M4 Max.
/// `ALLPAKA_DECODE_GUFUSE=1` enables the diagnostic variant.
fn gufuse() -> bool {
    static G: OnceLock<bool> = OnceLock::new();
    *G.get_or_init(|| std::env::var("ALLPAKA_DECODE_GUFUSE").is_ok_and(|v| v == "1"))
}

/// The fused single-threadgroup router_topk in decode. OFF by default: the
/// fused kernel walks the experts serially inside one threadgroup and is
/// latency-bound - on qwen3-30b the parallel matvec_f32 + tiny top-k pair
/// (one extra dispatch and barrier, same buffer) measured 113.8 vs 79.2
/// tok/s decode (265 vs 388 ms GPU executing over 32 tokens).
/// `ALLPAKA_RTOPK=1` re-enables the fused kernel.
fn rtopk_fused() -> bool {
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| std::env::var("ALLPAKA_RTOPK").is_ok_and(|v| v == "1"))
}

/// The decode-side fold of the FFN residual norm into the router top-k
/// (resnorm_router). OFF by default: under max-performance `normflag`,
/// enabling costs ~20–27% decode tok/s on GLM-Air / qwen3-30b (2026-09-12)
/// despite fewer dispatches. `ALLPAKA_RFUSE=1` enables only when
/// `normflag` is also off (`moe_plain`).
fn rfuse() -> bool {
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| std::env::var("ALLPAKA_RFUSE").is_ok_and(|v| v == "1"))
}

/// Fold the GLM shared expert's Q8_0 down into the indexed expert-down
/// launch (+1 parallel slot). OFF by default until cool A/B confirms a win;
/// `ALLPAKA_SHARED_TAIL=1` enables.
fn shared_tail() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("ALLPAKA_SHARED_TAIL").is_ok_and(|v| v == "1"))
}

/// Stagger GLM shared Q5 gate/up out of the expert Q4 window: expert
/// gate/up → barrier → expert Q8 down ∥ shared Q5 gate/up → barrier →
/// shared Q8 down. OFF by default until cool Terminal A/B
/// (`scripts/glm-ab-stagger.sh`) confirms a win; `ALLPAKA_SHARED_STAGGER=1`
/// enables. Disabled automatically when `SHARED_TAIL` folds shared down
/// into the indexed expert-down launch (shared gate/up must finish first).
fn shared_stagger() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("ALLPAKA_SHARED_STAGGER").is_ok_and(|v| v == "1"))
}

/// Rows per SIMD group for `matvec_q8_0_mv` (NR0 = 32/LPR). Default 2;
/// sweep with `ALLPAKA_Q8_NR0` ∈ {1,2,4} (GLM expert down is the hot shape).
fn q8_nr0() -> usize {
    static NR0: OnceLock<usize> = OnceLock::new();
    *NR0.get_or_init(|| {
        std::env::var("ALLPAKA_Q8_NR0")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| [1, 2, 4].contains(n))
            .unwrap_or(2)
    })
}

/// K-step 64 for the LL mm kernels: `ALLPAKA_MM_K64=1` to enable.
fn mm_k64() -> bool {
    static K: OnceLock<bool> = OnceLock::new();
    *K.get_or_init(|| std::env::var("ALLPAKA_MM_K64").is_ok_and(|v| v != "0"))
}

/// The software-pipelined double-buffer K-loop in the q4_k mm kernels
/// (mmllp_q4_k, mmllpg/ps_id_q4_k). Default ON. `ALLPAKA_MM_PIPE=0` reverts
/// to the two-barrier llama-structure loop.
fn mm_pipe() -> bool {
    crate::runtime::get().mm_pipeline
}

/// K-step 64 for the mmid (MoE) kernels only: `ALLPAKA_MMID_K64=1`. The FFN
/// K-loop runs 128 iterations at NK=32 with two threadgroup barriers each;
/// halving them costs shared memory but the MoE stages are latency-bound.
fn mmid_k64() -> bool {
    static K: OnceLock<bool> = OnceLock::new();
    *K.get_or_init(|| std::env::var("ALLPAKA_MMID_K64").is_ok_and(|v| v != "0"))
}

fn mm_ll() -> bool {
    static LL: OnceLock<bool> = OnceLock::new();
    *LL.get_or_init(|| std::env::var("ALLPAKA_MM_LL").map_or(true, |v| v != "0"))
}

/// ALLPAKA_NO_BARRIER=1 skips in-buffer memory barriers (FFN path only) to
/// measure their cost. Timing probe only - results are racy garbage.
/// ALLPAKA_NO_STAGE_BARRIER=1 drops only the FFN stage-to-stage barriers
/// (route and combine barriers stay, so the mm stages still do full work
/// on valid tables; the down stage may read torn gate data - timing only).
fn no_stage_barrier() -> bool {
    static NB: OnceLock<bool> = OnceLock::new();
    *NB.get_or_init(|| std::env::var_os("ALLPAKA_NO_STAGE_BARRIER").is_some())
}

fn no_barrier() -> bool {
    static NB: OnceLock<bool> = OnceLock::new();
    *NB.get_or_init(|| {
        std::env::var_os("ALLPAKA_NO_BARRIER").is_some()
            || std::env::var_os("ALLPAKA_SERIAL").is_some()
    })
}

/// ALLPAKA_SERIAL=1: encode the prefill command buffers with
/// MTLDispatchType::Serial and drop the explicit barriers - serial dispatch
/// gives in-order execution (ordering without the full pipeline flush that
/// memory_barrier_with_resources pays). llama.cpp encodes this way.
fn serial_dispatch() -> metal::MTLDispatchType {
    static S: OnceLock<bool> = OnceLock::new();
    if *S.get_or_init(|| std::env::var_os("ALLPAKA_SERIAL").is_some()) {
        metal::MTLDispatchType::Serial
    } else {
        metal::MTLDispatchType::Concurrent
    }
}

/// Down-projection kernel with swiglu folded into B staging: gate and up
/// stay raw in y, the mm applies silu(g)*u while loading. q4_k only;
/// anything else keeps the standalone swiglu pass.
fn mmid_swiglu_kernel_for(ty: GgmlType) -> Option<(&'static str, usize)> {
    if !mm_ll() || mm_k64() {
        return None;
    }
    match ty {
        GgmlType::Q4K => Some((
            if mm_pipe() {
                "mmllps_id_q4_k"
            } else {
                "mmlls_id_q4_k"
            },
            64,
        )),
        _ => None,
    }
}

fn mmid_kernel_for(ty: GgmlType, gather: bool) -> Option<(&'static str, usize)> {
    if mm_ll() {
        let name = match (ty, gather, mm_k64() || mmid_k64()) {
            (GgmlType::Q8_0, false, false) => "mmll_id_q8_0",
            (GgmlType::Q2K, false, false) => "mmll_id_q2_k",
            (GgmlType::Q3K, false, false) => "mmll_id_q3_k",
            (GgmlType::Q4K, false, false) => "mmll_id_q4_k",
            (GgmlType::Q5K, false, false) => "mmll_id_q5_k",
            (GgmlType::Q6K, false, false) => "mmll_id_q6_k",
            (GgmlType::Q5_0, false, false) => "mmll_id_q5_0",
            (GgmlType::Q8_0, true, false) => "mmllg_id_q8_0",
            (GgmlType::Q2K, true, false) => "mmllg_id_q2_k",
            (GgmlType::Q3K, true, false) => "mmllg_id_q3_k",
            (GgmlType::Q4K, true, false) => {
                if mm_pipe() {
                    "mmllpg_id_q4_k"
                } else {
                    "mmllg_id_q4_k"
                }
            }
            (GgmlType::Q5K, true, false) => "mmllg_id_q5_k",
            (GgmlType::Q6K, true, false) => "mmllg_id_q6_k",
            (GgmlType::Q5_0, true, false) => "mmllg_id_q5_0",
            (GgmlType::Q8_0, false, true) => "mm64ll_id_q8_0",
            (GgmlType::Q2K, false, true) => "mm64ll_id_q2_k",
            (GgmlType::Q3K, false, true) => "mm64ll_id_q3_k",
            (GgmlType::Q4K, false, true) => "mm64ll_id_q4_k",
            (GgmlType::Q5K, false, true) => "mm64ll_id_q5_k",
            (GgmlType::Q6K, false, true) => "mm64ll_id_q6_k",
            (GgmlType::Q5_0, false, true) => "mm64ll_id_q5_0",
            (GgmlType::Q8_0, true, true) => "mm64llg_id_q8_0",
            (GgmlType::Q2K, true, true) => "mm64llg_id_q2_k",
            (GgmlType::Q3K, true, true) => "mm64llg_id_q3_k",
            (GgmlType::Q4K, true, true) => "mm64llg_id_q4_k",
            (GgmlType::Q5K, true, true) => "mm64llg_id_q5_k",
            (GgmlType::Q6K, true, true) => "mm64llg_id_q6_k",
            (GgmlType::Q5_0, true, true) => "mm64llg_id_q5_0",
            _ => return None,
        };
        return Some((name, 64));
    }
    if gather {
        // The pre-LL kernels have no gather form; the caller falls back to
        // the CPU-gathered path.
        return None;
    }
    Some((
        match ty {
            GgmlType::Q8_0 => "mmid_q8_0",
            GgmlType::Q2K => "mmid_q2_k",
            GgmlType::Q3K => "mmid_q3_k",
            GgmlType::Q4K => "mmid_q4_k",
            GgmlType::Q5K => "mmid_q5_k",
            GgmlType::Q6K => "mmid_q6_k",
            _ => return None,
        },
        32,
    ))
}

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
    let mut states = Vec::with_capacity(3);
    let mut bms = [32u64; 3];
    let mut fused_swiglu = false;
    for (i, mat) in mats.iter().enumerate() {
        let (name, bm) = if i == 2 {
            match mmid_swiglu_kernel_for(mat.ty) {
                Some(nb) => {
                    fused_swiglu = true;
                    nb
                }
                None => mmid_kernel_for(mat.ty, false)?,
            }
        } else {
            mmid_kernel_for(mat.ty, true)?
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
    let dual_state = if fused_swiglu
        && strides[0] == strides[1]
        && req.ffn % 64 == 0
        && std::env::var_os("ALLPAKA_DUAL").is_some()
    {
        gpu.pipeline("mmllgd_id_q4_k", 1, 1).map(|p| p.to_owned())
    } else {
        None
    };
    let dual = dual_state.is_some();

    let (row_tiles, n_groups) = if route_mode {
        // Group sizes exist only on the GPU. The mmid kernels stride over
        // row tiles (r1 += grid.y * 32), so any grid covers every row;
        // launch ~2x the average to skip a storm of empty threadgroups
        // without looping in the common case.
        let avg = (req.total_rows / req.n_expert.max(1)) as u64;
        let _ = avg;
        // Worst-case grid: measured FASTER than a tight one - empty
        // threadgroups retire for free, while a small grid serialises the
        // hot experts through the striding loop. ALLPAKA_MMID_RTILES pins
        // the grid for A/B tests.
        let auto = (m_fused as u64).div_ceil(32).max(1);
        let rt = std::env::var("ALLPAKA_MMID_RTILES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(auto, |n| n.max(1));
        (rt, req.n_expert as u64)
    } else {
        let max_rows = req.groups.iter().map(|g| g[2]).max()? as usize;
        (max_rows.div_ceil(32) as u64, req.groups.len() as u64)
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
        .map(|r| route_layout(m_fused, req.n_expert, r.n_used, req.shared.is_some()));
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
        // One-buffer chunk (ALLPAKA_PF_ONEBUF=1, default): encode into the
        // chunk's shared command buffer, lazily armed by the first
        // eligible layer; the commit happens once in prefill_end. A layer
        // that stays off the shared buffer (CPU routing, FFN_SPLIT) seals
        // the open segment first, so its own buffer stays ordered after
        // the work already encoded there.
        let onebuf = (route_mode || (dense_obuf() && req.route.is_none() && req.fused.is_some()))
            && pf_onebuf()
            && !split;
        if !onebuf {
            gpu.pf_obuf_seal();
        } else {
            gpu.pf_obuf_arm();
        }
        // Deferred commit: chain behind the previous layer's buffer through
        // the shared event instead of a CPU wait. Any fused-path buffer
        // qualifies - it reads nothing back (the combine lands in pf_x).
        // FFN_SPLIT keeps the old waits.
        let defer = !onebuf && req.fused.is_some() && pf_defer() && !split;
        // Cloned queue handle: command buffers borrow it, leaving `gpu`
        // free for the deferred commit bookkeeping.
        let queue = gpu.queue.clone();
        let cmd0 = queue.new_command_buffer();
        let mut cmd = cmd0;
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
                    eprintln!("ffn {}: {:.3} ms", $label, (ge - gs) * 1e3);
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
            };
            let pp = &params as *const RouteParams as *const _;
            let psz = std::mem::size_of::<RouteParams>() as u64;
            let rb = &gpu.route_buf;
            let tgs = (m_fused as u64).div_ceil(256);
            let lg_buf = if lg_tag == "route" { rb } else { y };
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
        if dual {
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
        if !no_barrier() && !no_stage_barrier() {
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
            enc.set_compute_pipeline_state(&gpu.pipelines[&("combine_rows", 1, 1)]);
            enc.set_buffer(0, Some(&gpu.pf_x), 0);
            enc.set_buffer(1, Some(y), e(out_at));
            enc.set_buffer(2, Some(off_buf), e(off_off));
            enc.set_buffer(3, Some(hr_buf), e(hr_off));
            enc.set_buffer(4, Some(hw_buf), e(hw_off));
            let h32 = req.hidden as u32;
            enc.set_bytes(5, 4, &h32 as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(c.m as u64, 1, 1), MTLSize::new(256, 1, 1));
            cstamp!(gpu, enc, "cb");
        }
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

/// One FFN inside a fused batch: `y = swiglu(x·Gᵀ, x·Uᵀ) · Dᵀ` for `m`
/// activation rows, without the CPU touching the intermediates.
pub struct FfnReq<'a> {
    pub gate_ty: GgmlType,
    pub gate_w: &'a [u8],
    pub up_ty: GgmlType,
    pub up_w: &'a [u8],
    pub down_ty: GgmlType,
    pub down_w: &'a [u8],
    pub hidden: usize,
    pub ffn: usize,
    pub x: &'a [f32],
    pub m: usize,
}

/// A validated weight reference: kernel name plus window-relative offset.
#[derive(Clone, Copy)]
struct MatRef {
    kernel: &'static str,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
}

fn resolve(gpu: &Gpu, ty: GgmlType, w: &[u8]) -> Option<MatRef> {
    let kernel = match ty {
        GgmlType::Q5_0 => "matvec_q5_0",
        GgmlType::Q8_0 => q8_kernel(),
        GgmlType::Q2K => q2_kernel(),
        GgmlType::Q3K => q3_kernel(),
        GgmlType::Q4K => q4_kernel(),
        GgmlType::Q5K => q5_kernel(),
        GgmlType::Q6K => q6_kernel(),
        _ => return None,
    };
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some(MatRef {
        kernel,
        ty,
        chunk,
        w_off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

/// Encode one logical matmul as tiled sub-dispatches. Offsets into the
/// x/y buffers are per sub-batch; row strides keep them 16-byte aligned for
/// the kernels' float4 loads (n_in and n_out are whole quant blocks).
///
/// `states` holds one pipeline per tile, in `tiles_for(m)` order; they are
/// specialised ahead of encoding because that needs `&mut Gpu`.
#[allow(clippy::too_many_arguments)]
fn encode_matvec(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    mat: &MatRef,
    states: &[ComputePipelineState],
    n_in: usize,
    n_out: usize,
    x_buf: &Buffer,
    x_off: usize,
    y_buf: &Buffer,
    y_off: usize,
    m: usize,
) -> u64 {
    let mut dispatched = 0u64;
    if m >= MM_MIN_M {
        // One tile-matmul dispatch for the whole batch; `states` holds the
        // single mm pipeline the caller resolved under the same condition.
        enc.set_compute_pipeline_state(&states[0]);
        enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
        enc.set_buffer(1, Some(x_buf), x_off as u64);
        enc.set_buffer(2, Some(y_buf), y_off as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        let m32 = m as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
        enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
        let (bm, bn) = mm_kernel_for(mat.ty, m).map_or((32, 32), |(_, bm, bn)| (bm, bn));
        enc.dispatch_thread_groups(
            MTLSize::new(
                (n_out as u64).div_ceil(bm as u64),
                (m as u64).div_ceil(bn as u64),
                1,
            ),
            MTLSize::new(128, 1, 1),
        );
        return 1;
    }
    for ((start, _), state) in tiles_for(m).into_iter().zip(states) {
        enc.set_compute_pipeline_state(state);
        enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
        enc.set_buffer(1, Some(x_buf), (x_off + start * n_in * 4) as u64);
        enc.set_buffer(2, Some(y_buf), (y_off + start * n_out * 4) as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
        let per_group = 128u64;
        let total_threads = n_out as u64 * lanes_per_row(mat.ty, n_in) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(total_threads.div_ceil(per_group), 1, 1),
            MTLSize::new(per_group, 1, 1),
        );
        dispatched += 1;
    }
    dispatched
}

/// Run whole FFNs (gate, up, swiglu, down) as ONE command buffer with ONE
/// wait. The compute encoder is serial, so each dispatch sees the previous
/// one's writes; the intermediates never leave the GPU arena.
///
/// Per decoded token on a MoE this replaces two waits (gate+up batch, down
/// batch) with one, and removes the CPU swiglu pass between them.
///
/// All-or-nothing, like matvec_batch: any request without kernels for all
/// three weights declines the whole batch to the CPU path.
pub fn ffn_batch(reqs: &[FfnReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    let mut mats = Vec::with_capacity(reqs.len());
    for r in reqs {
        if r.x.len() != r.m * r.hidden || r.hidden % 4 != 0 || r.ffn % 4 != 0 {
            return None;
        }
        mats.push((
            resolve(&gpu, r.gate_ty, r.gate_w)?,
            resolve(&gpu, r.up_ty, r.up_w)?,
            resolve(&gpu, r.down_ty, r.down_w)?,
        ));
    }

    // Arena plan per request: x in the x arena; gate, up and the final out
    // rows in the y arena, each 256-aligned at the request level.
    struct Plan {
        x_off: usize,
        gate_off: usize,
        up_off: usize,
        out_off: usize,
    }
    let align = |n: usize| (n + 255) & !255;
    let mut x_need = 0usize;
    let mut y_need = 0usize;
    let mut plans = Vec::with_capacity(reqs.len());
    for r in reqs {
        let p = Plan {
            x_off: x_need,
            gate_off: y_need,
            up_off: y_need + align(r.m * r.ffn * 4),
            out_off: y_need + 2 * align(r.m * r.ffn * 4),
        };
        x_need += align(r.m * r.hidden * 4);
        y_need = p.out_off + align(r.m * r.hidden * 4);
        plans.push(p);
    }
    gpu.ensure_arenas(x_need, y_need);

    // One pipeline per (matrix, tile), specialised before encoding starts;
    // see encode_matvec. Gate and up share a shape, down has its own.
    let mut states = Vec::with_capacity(reqs.len());
    for (r, (gate, up, down)) in reqs.iter().zip(&mats) {
        let mut per_mat = Vec::with_capacity(3);
        for (mat, n_in) in [(gate, r.hidden), (up, r.hidden), (down, r.ffn)] {
            // Same routing as matvec_batch: a big batch is one tile-matmul
            // dispatch, a small one is exact matvec tiles.
            if r.m >= MM_MIN_M {
                per_mat.push(vec![gpu
                    .pipeline(mm_kernel_for(mat.ty, r.m)?.0, 1, 1)?
                    .to_owned()]);
            } else {
                let lpr = lanes_per_row(mat.ty, n_in);
                let mut tiles = Vec::new();
                for (_, rows) in tiles_for(r.m) {
                    tiles.push(gpu.pipeline(mat.kernel, rows, lpr)?.to_owned());
                }
                per_mat.push(tiles);
            }
        }
        states.push(per_mat);
    }

    unsafe {
        let xp = gpu.x_arena.contents() as *mut u8;
        for (r, p) in reqs.iter().zip(&plans) {
            std::ptr::copy_nonoverlapping(
                r.x.as_ptr() as *const u8,
                xp.add(p.x_off),
                r.x.len() * 4,
            );
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Concurrent encoder in three barriered waves: every expert's gate
        // AND up together, every swiglu, every down. One expert's matvec is
        // far too small to fill the GPU; eight experts' worth at once is
        // what the serial encoder was quietly forbidding.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let mut dispatched = 0u64;
        for (((r, p), (gate, up, _)), st) in reqs.iter().zip(&plans).zip(&mats).zip(&states) {
            dispatched += encode_matvec(
                enc,
                &gpu,
                gate,
                &st[0],
                r.hidden,
                r.ffn,
                &gpu.x_arena,
                p.x_off,
                &gpu.y_arena,
                p.gate_off,
                r.m,
            );
            dispatched += encode_matvec(
                enc,
                &gpu,
                up,
                &st[1],
                r.hidden,
                r.ffn,
                &gpu.x_arena,
                p.x_off,
                &gpu.y_arena,
                p.up_off,
                r.m,
            );
        }
        enc.memory_barrier_with_resources(&[&gpu.y_arena]);
        for (r, p) in reqs.iter().zip(&plans) {
            // swiglu(gate, up) in place over m*ffn elements.
            enc.set_compute_pipeline_state(&gpu.pipelines[&("swiglu", 1, 1)]);
            enc.set_buffer(0, Some(&gpu.y_arena), p.gate_off as u64);
            enc.set_buffer(1, Some(&gpu.y_arena), p.up_off as u64);
            let n32 = (r.m * r.ffn) as u32;
            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
            let per_group = 256u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n32 as u64).div_ceil(per_group), 1, 1),
                MTLSize::new(per_group, 1, 1),
            );
            dispatched += 1;
        }
        enc.memory_barrier_with_resources(&[&gpu.y_arena]);
        for (((r, p), (_, _, down)), st) in reqs.iter().zip(&plans).zip(&mats).zip(&states) {
            dispatched += encode_matvec(
                enc,
                &gpu,
                down,
                &st[2],
                r.ffn,
                r.hidden,
                &gpu.y_arena,
                p.gate_off,
                &gpu.y_arena,
                p.out_off,
                r.m,
            );
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(dispatched, Ordering::Relaxed);

        let yp = gpu.y_arena.contents() as *const u8;
        reqs.iter()
            .zip(&plans)
            .map(|(r, p)| {
                let mut v = vec![0f32; r.m * r.hidden];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        yp.add(p.out_off),
                        v.as_mut_ptr() as *mut u8,
                        v.len() * 4,
                    );
                }
                v
            })
            .collect()
    });
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{tiles_for, TILES};

    #[test]
    fn tiles_cover_every_row_exactly_once() {
        for m in 0..40 {
            let tiles = tiles_for(m);
            let mut at = 0;
            for &(start, rows) in &tiles {
                assert_eq!(start, at, "tiles for {m} leave a gap or overlap");
                assert!(TILES.contains(&rows), "{rows} has no compiled kernel");
                at += rows;
            }
            assert_eq!(at, m, "tiles for {m} do not reach the last row");
        }
    }

    #[test]
    fn tiles_prefer_the_widest_kernel() {
        // A dispatch per 8 rows plus an exact tail - never 8 one-row passes
        // over the same weights.
        let widest = *TILES.last().unwrap();
        assert_eq!(tiles_for(widest), vec![(0, widest)]);
        assert_eq!(tiles_for(11), vec![(0, 8), (8, 2), (10, 1)]);
        assert_eq!(tiles_for(16), vec![(0, 8), (8, 8)]);
    }
}
