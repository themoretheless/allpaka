//! ggml-cuda MMQ/MMVQ / flash-attn via `allpaka_ggml.dll`: on by default when
//! the library is present, `ALLPAKA_GGML=0` turns it off.

use cudarc::driver::CudaStream;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

type InitFn = unsafe extern "C" fn(i32) -> i32;
type MulMatFn = unsafe extern "C" fn(
    w_dev: *const c_void,
    x_dev: *const f32,
    y_dev: *mut f32,
    n_in: i32,
    n_out: i32,
    m: i32,
    w_type: i32,
) -> i32;
type MulMatVecFn = unsafe extern "C" fn(
    w_dev: *const c_void,
    x_dev: *const f32,
    y_dev: *mut f32,
    n_in: i32,
    n_out: i32,
    w_type: i32,
) -> i32;
type FlashAttnFn = unsafe extern "C" fn(
    q_dev: *const f32,
    k_dev: *const c_void,
    v_dev: *const c_void,
    out_dev: *mut f32,
    head_dim: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    m: i32,
    base: i32,
    kv_dim: i32,
    scale: f32,
    d_pos_dev: *const u32,
) -> i32;
type SyncFn = unsafe extern "C" fn();
type BindPeerFn = unsafe extern "C" fn(*mut c_void) -> i32;
type PeerFn = unsafe extern "C" fn();
type ClearSharedFn = unsafe extern "C" fn();
type HybridReplayFn = unsafe extern "C" fn(
    pre_execs: *const *mut c_void,
    post_execs: *const *mut c_void,
    tail_exec: *mut c_void,
    n_layers: i32,
    stream: *mut c_void,
    q_dev: *const f32,
    out_dev: *mut f32,
    cache_base: *const c_void,
    k_off: *const i32,
    v_off: *const i32,
    head_dim: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    base: i32,
    kv_dim: i32,
    scale: f32,
    d_pos_dev: *const u32,
) -> i32;
type ResetMaskFn = unsafe extern "C" fn();
type SetQ8Fn = unsafe extern "C" fn(*mut c_void);
type Q8ConsumedFn = unsafe extern "C" fn() -> i32;
type FaLastSrcFn = unsafe extern "C" fn() -> *mut c_void;

struct Api {
    _lib: libloading::Library,
    init: InitFn,
    mul_mat: MulMatFn,
    mul_mat_vec: Option<MulMatVecFn>,
    flash_attn: Option<FlashAttnFn>,
    hybrid_replay: Option<HybridReplayFn>,
    reset_mask: Option<ResetMaskFn>,
    set_q8: Option<SetQ8Fn>,
    q8_consumed: Option<Q8ConsumedFn>,
    fa_last_src: Option<FaLastSrcFn>,
    sync: SyncFn,
    bind_peer: Option<BindPeerFn>,
    wait_peer: Option<PeerFn>,
    signal_peer: Option<PeerFn>,
    clear_shared: Option<ClearSharedFn>,
}

static API: OnceLock<Option<Api>> = OnceLock::new();
static GGML_PENDING: AtomicBool = AtomicBool::new(false);
static CUDARC_DIRTY: AtomicBool = AtomicBool::new(false);
static PEER_BOUND: AtomicBool = AtomicBool::new(false);
static SHARED_STREAM: AtomicBool = AtomicBool::new(false);

fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("ALLPAKA_GGML_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(p) = exe.parent() {
            dirs.push(p.to_path_buf());
            dirs.push(p.join("ggml"));
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    dirs.push(manifest.join("../../third_party/allpaka_ggml/bin"));
    dirs.push(PathBuf::from(
        "D:/Source/allpaka/third_party/allpaka_ggml/bin",
    ));
    dirs
}

fn load() -> Option<&'static Api> {
    API.get_or_init(|| {
        for dir in candidate_dirs() {
            let dll = dir.join("allpaka_ggml.dll");
            if !dll.is_file() {
                continue;
            }
            if let Ok(abs) = std::fs::canonicalize(&dir) {
                let path = std::env::var_os("PATH").unwrap_or_default();
                let mut new_path = abs.as_os_str().to_owned();
                new_path.push(";");
                new_path.push(&path);
                // SAFETY: Windows-only path (a `.dll`, a `;`-joined PATH). There the OS
                // guards the environment block with its own lock and std serialises its own
                // env access, so a concurrent reader sees the old or the new value, not a torn
                // one. This runs once, inside `API.get_or_init`.
                unsafe { std::env::set_var("PATH", new_path) };
            }
            // SAFETY: loading runs the DLL's static initialisers. allpaka_ggml.dll is our
            // own build of third_party/allpaka_ggml (no DllMain of its own, only CRT and
            // ggml-cuda static setup), loaded at most once per candidate inside `get_or_init`.
            let lib = match unsafe { libloading::Library::new(&dll) } {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("cuda: failed to load {}: {e}", dll.display());
                    continue;
                }
            };
            // SAFETY (every `lib.get` below): each symbol is read as the fn-pointer type
            // declared at the top of this file, which mirrors the `extern "C"` prototype in
            // third_party/allpaka_ggml/allpaka_ggml.h; the two must be kept in sync. The
            // copied pointers stay valid because `lib` moves into the same `Api`, which lives
            // in the `API` static until exit; on `continue`, `lib` and every pointer copied
            // out of it are dropped together and never called.
            let init: InitFn = match unsafe { lib.get(b"allpaka_ggml_init\0") } {
                Ok(s) => *s,
                Err(e) => {
                    eprintln!("cuda: allpaka_ggml_init missing: {e}");
                    continue;
                }
            };
            let mul_mat: MulMatFn = match unsafe { lib.get(b"allpaka_ggml_mul_mat\0") } {
                Ok(s) => *s,
                Err(e) => {
                    eprintln!("cuda: allpaka_ggml_mul_mat missing: {e}");
                    continue;
                }
            };
            let mul_mat_vec: Option<MulMatVecFn> =
                unsafe { lib.get(b"allpaka_ggml_mul_mat_vec\0").ok().map(|s| *s) };
            let flash_attn: Option<FlashAttnFn> =
                unsafe { lib.get(b"allpaka_ggml_flash_attn\0").ok().map(|s| *s) };
            let hybrid_replay: Option<HybridReplayFn> =
                unsafe { lib.get(b"allpaka_hybrid_replay\0").ok().map(|s| *s) };
            let reset_mask: Option<ResetMaskFn> =
                unsafe { lib.get(b"allpaka_fa_reset_mask_lim\0").ok().map(|s| *s) };
            let set_q8: Option<SetQ8Fn> =
                unsafe { lib.get(b"allpaka_fa_set_q8\0").ok().map(|s| *s) };
            let q8_consumed: Option<Q8ConsumedFn> =
                unsafe { lib.get(b"allpaka_fa_q8_consumed\0").ok().map(|s| *s) };
            let fa_last_src: Option<FaLastSrcFn> =
                unsafe { lib.get(b"allpaka_fa_last_src\0").ok().map(|s| *s) };
            let sync: SyncFn = match unsafe { lib.get(b"allpaka_ggml_sync\0") } {
                Ok(s) => *s,
                Err(e) => {
                    eprintln!("cuda: allpaka_ggml_sync missing: {e}");
                    continue;
                }
            };
            let bind_peer: Option<BindPeerFn> =
                unsafe { lib.get(b"allpaka_ggml_bind_peer_stream\0").ok().map(|s| *s) };
            let wait_peer: Option<PeerFn> =
                unsafe { lib.get(b"allpaka_ggml_wait_peer\0").ok().map(|s| *s) };
            let signal_peer: Option<PeerFn> =
                unsafe { lib.get(b"allpaka_ggml_signal_peer\0").ok().map(|s| *s) };
            let clear_shared: Option<ClearSharedFn> =
                unsafe { lib.get(b"allpaka_ggml_clear_shared\0").ok().map(|s| *s) };
            let api = Api {
                _lib: lib,
                init,
                mul_mat,
                mul_mat_vec,
                flash_attn,
                hybrid_replay,
                reset_mask,
                set_q8,
                q8_consumed,
                fa_last_src,
                sync,
                bind_peer,
                wait_peer,
                signal_peer,
                clear_shared,
            };
            let rc = unsafe { (api.init)(0) };
            if rc != 0 {
                eprintln!("cuda: allpaka_ggml_init failed ({rc})");
                continue;
            }
            eprintln!(
                "cuda: ggml loaded from {} (flash_attn={})",
                dll.display(),
                api.flash_attn.is_some()
            );
            return Some(api);
        }
        eprintln!("cuda: allpaka_ggml.dll not found (set ALLPAKA_GGML_DIR)");
        None
    })
    .as_ref()
}

/// ggml kernels are used when allowed (on unless `ALLPAKA_GGML=0`) and the
/// library actually loaded; without `allpaka_ggml.dll` every caller keeps
/// the built-in kernels.
pub fn enabled() -> bool {
    crate::runtime::get().cuda_ggml && load().is_some()
}

pub fn fa_kv_floor() -> usize {
    std::env::var("ALLPAKA_FA_KV_FLOOR")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| (16..=8192).contains(&v))
        .unwrap_or(256)
}

/// Bind cudarc's stream so ggml can share it (preferred) or use events.
pub fn bind_peer_stream(stream: &CudaStream) {
    let Some(api) = load() else {
        return;
    };
    let Some(bind) = api.bind_peer else {
        return;
    };
    let ptr = stream.cu_stream() as *mut c_void;
    if ptr.is_null() {
        eprintln!("cuda: ggml peer bind skipped (null/default stream)");
        return;
    }
    let rc = unsafe { bind(ptr) };
    if rc != 0 {
        return;
    }
    PEER_BOUND.store(true, Ordering::Release);
    // Bind with non-null cudarc stream always injects shared mode.
    SHARED_STREAM.store(true, Ordering::Release);
    eprintln!("cuda: ggml shares cudarc stream (no sync)");
}

pub fn shared_stream() -> bool {
    SHARED_STREAM.load(Ordering::Acquire)
}

/// Prefill: drop shared stream so ggml MMQ/FA can overlap cudarc on a peer queue.
pub fn clear_shared_for_prefill() {
    let Some(api) = load() else {
        return;
    };
    if let Some(clear) = api.clear_shared {
        unsafe { clear() };
    }
    SHARED_STREAM.store(false, Ordering::Release);
    // Keep PEER_BOUND so prepare_* still uses events against the last peer.
}

/// Mark that cudarc kernels may have written buffers ggml will read.
pub fn mark_cudarc_dirty() {
    CUDARC_DIRTY.store(true, Ordering::Release);
}

/// Before ggml reads tensors produced on the cudarc stream.
pub fn prepare_from_cudarc(stream: &CudaStream) {
    if SHARED_STREAM.load(Ordering::Acquire) {
        CUDARC_DIRTY.store(false, Ordering::Release);
        return;
    }
    if !CUDARC_DIRTY.swap(false, Ordering::AcqRel) {
        return;
    }
    if PEER_BOUND.load(Ordering::Acquire) {
        if let Some(api) = load() {
            if let Some(wait) = api.wait_peer {
                unsafe { wait() };
                return;
            }
        }
    }
    let _ = stream.synchronize();
}

/// Before cudarc kernels read tensors produced by ggml.
pub fn prepare_for_cudarc() {
    if SHARED_STREAM.load(Ordering::Acquire) {
        GGML_PENDING.store(false, Ordering::Release);
        return;
    }
    if !GGML_PENDING.swap(false, Ordering::AcqRel) {
        return;
    }
    if PEER_BOUND.load(Ordering::Acquire) {
        if let Some(api) = load() {
            if let Some(sig) = api.signal_peer {
                unsafe { sig() };
                return;
            }
        }
    }
    if let Some(api) = load() {
        unsafe { (api.sync)() };
    }
}

/// Drain ggml CUDA work. Safe no-op if unused.
pub fn sync() {
    if !GGML_PENDING.swap(false, Ordering::AcqRel) {
        return;
    }
    if let Some(api) = load() {
        unsafe { (api.sync)() };
    }
}

/// ggml_type: Q4_K=12, Q6_K=14.
pub fn mul_mat(
    w_dev: u64,
    x_dev: u64,
    y_dev: u64,
    n_in: usize,
    n_out: usize,
    m: usize,
    w_type: i32,
) -> bool {
    let Some(api) = load() else {
        return false;
    };
    // SAFETY: the pointers are CUDA device addresses passed through to the DLL and never
    // dereferenced on the host. Its kernel reads `w` and `x` and writes `y`, so they must
    // be live device allocations covering the sizes implied by the shape arguments. This
    // safe fn cannot check that; callers pass offsets into buffers they own (weight
    // windows, the x/y arenas) sized from the same shape.
    let rc = unsafe {
        (api.mul_mat)(
            w_dev as *const c_void,
            x_dev as *const f32,
            y_dev as *mut f32,
            n_in as i32,
            n_out as i32,
            m as i32,
            w_type,
        )
    };
    if rc != 0 {
        eprintln!("cuda: allpaka_ggml_mul_mat failed ({rc})");
        return false;
    }
    GGML_PENDING.store(true, Ordering::Release);
    true
}

/// Direct llama MMVQ for m=1 (no cgraph). Prefer over [`mul_mat`] for decode.
pub fn mul_mat_vec(
    w_dev: u64,
    x_dev: u64,
    y_dev: u64,
    n_in: usize,
    n_out: usize,
    w_type: i32,
) -> bool {
    let Some(api) = load() else {
        return false;
    };
    let Some(mmv) = api.mul_mat_vec else {
        return mul_mat(w_dev, x_dev, y_dev, n_in, n_out, 1, w_type);
    };
    // SAFETY: the pointers are CUDA device addresses passed through to the DLL and never
    // dereferenced on the host. Its kernel reads `w` and `x` and writes `y`, so they must
    // be live device allocations covering the sizes implied by the shape arguments. This
    // safe fn cannot check that; callers pass offsets into buffers they own (weight
    // windows, the x/y arenas) sized from the same shape.
    let rc = unsafe {
        mmv(
            w_dev as *const c_void,
            x_dev as *const f32,
            y_dev as *mut f32,
            n_in as i32,
            n_out as i32,
            w_type,
        )
    };
    if rc != 0 {
        eprintln!("cuda: allpaka_ggml_mul_mat_vec failed ({rc})");
        return false;
    }
    GGML_PENDING.store(true, Ordering::Release);
    true
}

/// Force next decode FA to include mask_from_pos (for CUDA graph capture).
pub fn reset_mask_lim() {
    let Some(api) = load() else {
        return;
    };
    if let Some(f) = api.reset_mask {
        unsafe { f() };
    }
}

/// Pack FA permute into `q8_dev` on next decode flash_attn (m=1). Cleared after one use.
pub fn fa_set_q8(q8_dev: u64) {
    let Some(api) = load() else {
        return;
    };
    let Some(f) = api.set_q8 else {
        return;
    };
    // SAFETY: `q8_dev` is a device address the DLL records and writes into during the
    // next m=1 flash_attn; the host never dereferences it. Callers pass `gpu.q8_q`'s
    // base, which `ensure_q8` sized for the activation before the call.
    unsafe { f(q8_dev as *mut c_void) };
}

/// True if the last flash_attn skipped permute for Q8 fuse.
pub fn fa_q8_consumed() -> bool {
    let Some(api) = load() else {
        return false;
    };
    let Some(f) = api.q8_consumed else {
        return false;
    };
    unsafe { f() != 0 }
}

/// FA device buffer after skip-permute flash_attn ([hd, n_q] F32).
pub fn fa_last_src() -> Option<u64> {
    let api = load()?;
    let f = api.fa_last_src?;
    let p = unsafe { f() };
    if p.is_null() {
        None
    } else {
        Some(p as u64)
    }
}

/// Attention geometry shared by every flash-attention launch of one token.
#[derive(Clone, Copy, Debug)]
pub struct FaShape {
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    /// KV position of the first query row.
    pub base: usize,
    /// Row stride of the f16 KV cache, in elements.
    pub kv_dim: usize,
    pub scale: f32,
}

/// One flash-attention launch over `m` query rows; all addresses are device pointers.
#[derive(Clone, Copy, Debug)]
pub struct FlashAttnReq {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub out: u64,
    pub m: usize,
    pub shape: FaShape,
    /// Device `u32` holding the decode position (graph-replayable), if any.
    pub d_pos: Option<u64>,
}

pub fn flash_attn(req: &FlashAttnReq) -> bool {
    let FlashAttnReq {
        q: q_dev,
        k: k_dev,
        v: v_dev,
        out: out_dev,
        m,
        shape:
            FaShape {
                head_dim,
                n_q_heads,
                n_kv_heads,
                base,
                kv_dim,
                scale,
            },
        d_pos: d_pos_dev,
    } = *req;
    let Some(api) = load() else {
        return false;
    };
    let Some(fa) = api.flash_attn else {
        return false;
    };
    // SAFETY: q/k/v/out/d_pos are device addresses only the DLL's kernels dereference:
    // q and out cover `m * n_q_heads * head_dim` floats, k/v the f16 KV rows up to
    // `base + m` at stride `kv_dim`, d_pos one u32 or null. Callers derive them from the
    // y arena and the attached KV `SharedRegion`; this safe fn cannot check them.
    let rc = unsafe {
        fa(
            q_dev as *const f32,
            k_dev as *const c_void,
            v_dev as *const c_void,
            out_dev as *mut f32,
            head_dim as i32,
            n_q_heads as i32,
            n_kv_heads as i32,
            m as i32,
            base as i32,
            kv_dim as i32,
            scale,
            d_pos_dev
                .map(|p| p as *const u32)
                .unwrap_or(std::ptr::null()),
        )
    };
    if rc != 0 {
        eprintln!("cuda: allpaka_ggml_flash_attn failed ({rc})");
        return false;
    }
    GGML_PENDING.store(true, Ordering::Release);
    true
}

/// One-token hybrid replay: per-layer pre/post segment graphs, a tail graph,
/// and the KV offsets FA needs between them.
#[derive(Clone, Copy, Debug)]
pub struct HybridReplay<'a> {
    pub pre_execs: &'a [*mut c_void],
    pub post_execs: &'a [*mut c_void],
    pub tail_exec: *mut c_void,
    pub q: u64,
    pub out: u64,
    pub cache_base: u64,
    /// Per-layer K/V element offsets into the f16 cache.
    pub k_off: &'a [i32],
    pub v_off: &'a [i32],
    pub shape: FaShape,
    pub d_pos: u64,
}

/// One-token hybrid replay: segment CUDA graphs + FA interleaved (DLL-side loop).
pub fn hybrid_replay(stream: &CudaStream, req: &HybridReplay<'_>) -> bool {
    let HybridReplay {
        pre_execs,
        post_execs,
        tail_exec,
        q: q_dev,
        out: out_dev,
        cache_base,
        k_off,
        v_off,
        shape:
            FaShape {
                head_dim,
                n_q_heads,
                n_kv_heads,
                base,
                kv_dim,
                scale,
            },
        d_pos: d_pos_dev,
    } = *req;
    let Some(api) = load() else {
        return false;
    };
    let Some(hr) = api.hybrid_replay else {
        return false;
    };
    if pre_execs.len() != post_execs.len()
        || pre_execs.len() != k_off.len()
        || k_off.len() != v_off.len()
        || pre_execs.is_empty()
    {
        return false;
    }
    // SAFETY: on the host the DLL reads `pre_execs.len()` entries from `pre_execs`,
    // `post_execs`, `k_off` and `v_off`; the four lengths were checked equal and nonzero
    // above, and the slices stay borrowed for the whole call. The exec handles belong to
    // `CudaGraph`s the caller holds across the call, and the stream is `stream`'s own.
    // q/out/cache/d_pos are device addresses that only the kernels dereference.
    let rc = unsafe {
        hr(
            pre_execs.as_ptr(),
            post_execs.as_ptr(),
            tail_exec,
            pre_execs.len() as i32,
            stream.cu_stream() as *mut c_void,
            q_dev as *const f32,
            out_dev as *mut f32,
            cache_base as *const c_void,
            k_off.as_ptr(),
            v_off.as_ptr(),
            head_dim as i32,
            n_q_heads as i32,
            n_kv_heads as i32,
            base as i32,
            kv_dim as i32,
            scale,
            d_pos_dev as *const u32,
        )
    };
    if rc != 0 {
        eprintln!("cuda: allpaka_hybrid_replay failed ({rc})");
        return false;
    }
    GGML_PENDING.store(true, Ordering::Release);
    true
}

pub fn ggml_type_code(ty: allpaka_dtype::GgmlType) -> Option<i32> {
    match ty {
        allpaka_dtype::GgmlType::Q4K => Some(12),
        allpaka_dtype::GgmlType::Q6K => Some(14),
        _ => None,
    }
}
