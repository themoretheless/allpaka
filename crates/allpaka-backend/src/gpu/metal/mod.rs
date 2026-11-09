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

pub use super::types::*;

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

mod telemetry;

pub use telemetry::{gpu_time_stats, stats};
use telemetry::*;

mod select;
use select::*;

mod verify;
use verify::*;

mod decode;
pub use decode::*;

mod prefill;
pub use prefill::*;

mod ffn_grouped;
pub use ffn_grouped::*;

/// y_arena f32 offset of the router logits the last fused attention block
/// produced, when GPU-side routing skipped the CPU readback.
static PF_ROUTER_AT: AtomicU64 = AtomicU64::new(u64::MAX);

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
    /// The fused FFN combine already wrote the NEXT layer's attn-normed
    /// rows into pf_hs (combine_rows_resnorm): the next fused attention /
    /// GDN block skips its norm_rows dispatch. Cell because the norm
    /// closures in those blocks hold a shared borrow of `gpu`.
    pf_hs_normed: std::cell::Cell<bool>,
    /// attend_mm_fq producer/consumer sync: a monotonic device counter the
    /// K/V producer threadgroups publish to (one increment per producer per
    /// tile), plus the host-side total of all past increments, passed to the
    /// next dispatch as its wait base. Never reset; u32 wrap needs ~180k
    /// prefills.
    pf_attn_sync: Buffer,
    pf_attn_epoch: std::cell::Cell<u32>,
    /// GPU-side MoE routing scratch: counters, the group table, picks,
    /// gathered token ids and the combine CSR, all produced on-device.
    route_buf: Buffer,
    route_cap: usize,
    /// Debug: (table word-offset, n_expert) of the last fused routing, for
    /// ALLPAKA_ROWS_DEBUG to dump the per-expert row histogram after prefill.
    dbg_route: std::cell::Cell<Option<(usize, usize)>>,
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
    tiles: usize,
    tile_bound: usize,
    total: usize,
}

fn route_layout(m: usize, n_expert: usize, n_used: usize, shared: bool, tile_w: usize) -> RouteLayout {
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
        "attend_mm_fq",
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
        "mmllp_qkv_q4_k",
        "mmllp_qkv_q8_0",
        "mmllpg_id_q4_k",
        "mmllps_id_q4_k",
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
    let pf_attn_sync = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(pf_attn_sync.contents() as *mut u8, 0, 64) };
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
        pf_attn_epoch: std::cell::Cell::new(0),
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
    crate::runtime::get().mv_tg
}

/// Llama-style `mul_mv_id` grid for INDEXED decode matvecs (`grid.z = slots`).
/// Default ON. `ALLPAKA_MV_ID=0` restores flat `n_out * slots` 1D packing.
fn mv_id_grid() -> bool {
    crate::runtime::get().mv_id_grid
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

#[repr(C)]
struct MmQkvArgs {
    n_in: u32,
    n_q: u32,
    n_k: u32,
    n_v: u32,
    wq_off: u64,
    wk_off: u64,
    wv_off: u64,
    q_off: u64,
    k_off: u64,
    v_off: u64,
    m: u32,
    pad: u32,
}

/// Fused QKV (or dual wqkv + zgate) prefill kernel when projections share the
/// same quantized format and their output widths are multiples of 64.
fn mm_qkv_kernel_for(ty: GgmlType) -> Option<&'static str> {
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
    crate::runtime::get().mega_debug
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
struct FqPrepArgs {
    eps: f32,
    has_qk_norm: u32,
    rot_dim: u32,
    has_bias: u32,
    n_kv_heads: u32,
    wait_base: u32,
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
