//! The CUDA device singleton: context, compiled kernel module, weight windows
//! and scratch arenas, plus initialisation and the locked accessor.

use super::*;

/// Capacity of `CudaGpu::attn_part` (f32) and `attn_cnt` (u32).
pub const ATTN_PART_ELEMS: usize = 1 << 20;
pub const ATTN_CNT_ELEMS: usize = 256;

pub(super) const KERNEL_NAMES: &[&str] = &[
    "rmsnorm_f32",
    "rmsnorm_into_f32",
    "rmsnorm_into_f32_q8",
    "rmsnorm_into_f32_q8_mb",
    "attn_decode_g8_q8",
    "residual_add",
    "swiglu",
    "swiglu_into_q8",
    "geglu_tanh",
    "rmsnorm_add_f32",
    "copy_f32",
    "cast_f32_to_f16",
    "cast_f16_to_f32",
    "rope_neox",
    "rmsnorm_rope_neox",
    "rmsnorm_rope_store_dpos",
    "rmsnorm_rope_freq_dpos",
    "rmsnorm_rope_store_freq_dpos",
    "build_rope_freq_dpos",
    "rmsnorm_rope_qk_store_table_dpos",
    "rope_neox_freq_dpos",
    "rope_neox_batch",
    "prefill_norm_rope",
    "add_rmsnorm_into_f32",
    "argmax_f32",
    "store_kv_f16",
    "store_kv_f16_dpos",
    "store_kv_pair_f16_dpos",
    "store_kv_batch_f16",
    "attend_gqa",
    "attend_gqa_dpos",
    "attend_gqa_batch",
    "attend_gqa_kv",
    "attend_gqa_dpos_kv",
    "attend_gqa_batch_kv",
    "softmax_topk",
    "moe_combine",
    "moe_combine_csr",
    "matvec_f32",
    "matvec_f16",
    "matvec_bf16",
    "matvec_q8_0",
    "matvec_q5_0",
    "matvec_q4_k",
    "matvec_q4_k_q8",
    "matvec_q6_k_q8",
    "matvec_q4_k_q8_2",
    "matvec_q4_k_q8_qkv",
    "matvec_q4_k_2",
    "matvec_q4_k_qkv",
    "quantize_q8_1",
    "quantize_q8_1_q6",
    "permute_q8_m1",
    "permute_q8_m1_qonly",
    "matvec_q5_k",
    "matvec_q6_k",
    "matvec_q2_k",
    "matvec_q3_k",
    "dequant_rows_f16",
    "mm_q4_k",
    "mm_q4_k_q8",
    "mm_q4_k_mma",
    "mm_q4_k_tile",
    "mm_q4_k_mmq",
    "mm_q6_k",
    "add_bias_f32",
    "add_bias_rows_f32",
    "gather_rows_f32",
    "gemm_f16_f32",
    "gdn_conv",
    "gdn_step",
    "gdn_out_norm",
    "gdn_conv_batch",
    "gdn_step_batch",
    "gdn_out_norm_batch",
    "ssm_store_f32",
    "ssm_load_f32",
];

/// One session's captured decode graphs, set aside while another decodes.
pub struct ParkedDecodeGraphs {
    pub key: u64,
    pub graph: Option<CudaGraph>,
    pub fa_graphs: Option<(Vec<CudaGraph>, Vec<CudaGraph>, CudaGraph)>,
    pub fa_attn: Option<Vec<CudaGraph>>,
    pub fa_execs: Option<(
        Vec<*mut std::ffi::c_void>,
        Vec<*mut std::ffi::c_void>,
        *mut std::ffi::c_void,
    )>,
    pub fa_kv_offs: Option<(Vec<i32>, Vec<i32>)>,
}

/// How many sessions' decode graphs are kept besides the active one.
pub const DECODE_GRAPHS_PARKED: usize = 4;

pub struct WeightChunk {
    pub buf: CudaSlice<u8>,
    pub start: usize,
    pub len: usize,
}

pub struct CudaGpu {
    #[allow(dead_code)]
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    #[allow(dead_code)]
    pub module: RawModule,
    /// Holds nvcc MMQ module so `mm_q4_k_mmq` stays valid.
    #[allow(dead_code)]
    pub mmq_module: Option<RawModule>,
    /// Holds nvcc MMVQ module so Q8 matvec kernels stay valid.
    #[allow(dead_code)]
    pub mmvq_module: Option<RawModule>,
    /// Holds NVRTC embed_row module (kept separate from main KERNELS PTX).
    #[allow(dead_code)]
    pub embed_module: Option<RawModule>,
    pub fns: HashMap<&'static str, KernelFn>,
    pub chunks: Vec<WeightChunk>,
    /// Mappings uploaded by `attach_mapping`: (host base, length, owners).
    pub attached: Vec<(usize, usize, usize)>,
    pub blaslt: Option<CudaBlasLT>,
    pub x_arena: CudaSlice<f32>,
    pub x_cap: usize,
    pub y_arena: CudaSlice<f32>,
    pub y_cap: usize,
    /// Fused decode's persistent inputs: RoPE tables, the output norm and
    /// every layer's norm / bias vectors, uploaded once per `decode_norms_key`.
    /// Its own buffer rather than a range of `x_arena`: prefill and the
    /// matvec paths use `x_arena` as scratch, and a short prefill chunk landed
    /// its RoPE table inside the norm weights the next decode token read.
    pub dec_x: CudaSlice<f32>,
    pub dec_x_cap: usize,
    pub f16_arena: CudaSlice<f16>,
    pub f16_cap: usize,
    pub pf_x: CudaSlice<f32>,
    pub pf_x_cap: usize,
    pub pf_hs: CudaSlice<f32>,
    pub pf_hs_cap: usize,
    pub pf_active: bool,
    pub pf_len: usize,
    /// When set, dense FFN was already applied on-device in prefill_attn; skip ffn_grouped.
    pub pf_ffn_done: bool,
    /// Decode token position for graph-replayable store/attend kernels.
    pub d_pos: CudaSlice<u32>,
    /// NeoX RoPE inv_freq[rot_dim/2] on device (set once; kernels use d_pos).
    pub d_rope_freq: Option<CudaSlice<f32>>,
    pub d_rope_freq_n: usize,
    /// Gemma 3's local-layer RoPE frequencies (`set_rope_inv_freq_swa`).
    pub d_rope_freq_swa: Option<CudaSlice<f32>>,
    /// Scratch for argmax index (avoids alloc during graph capture).
    pub d_argmax: CudaSlice<u32>,
    /// Captured whole-token dense decode graph (invalidated on shape change).
    /// For hybrid FA: optional mega-graph that replays segment launches + FA in one go.
    pub decode_graph: Option<CudaGraph>,
    /// Whole-prefill graph. Captured on the second call of a given length.
    pub pf_graph: Option<CudaGraph>,
    pub pf_graph_len: usize,
    pub pf_graph_seen: u32,
    pub pf_capturing: bool,
    /// Identity of the prefill `pf_graph` was captured for (0 = none).
    pub pf_graph_key: u64,
    /// Key the caller announced for the next prefill (`prefill_graph_key`),
    /// consumed by `prefill_replay` / `prefill_begin`.
    pub pf_graph_want: u64,
    /// Set when something reallocated a buffer mid-capture.
    pub pf_capture_bad: bool,
    /// A capture for the current key failed; do not retry until it changes.
    pub pf_graph_failed: bool,
    /// Page-locked staging for the prefill RoPE table (address, capacity in
    /// f32). Pageable H2D copies cannot be stream-captured; a copy from
    /// pinned memory can, and a replay re-reads the same staging bytes.
    pub pf_rope_pinned: (usize, usize),
    /// The same for chunks that are not being captured (see `upload_rope_pinned`).
    pub pf_rope_scratch: (usize, usize),
    /// Batched decode (`MultiRow`): each row's position, page-locked on the
    /// host and copied into `multi_dpos_dev` inside the step, so a captured
    /// step replays with the positions staged before it (`multi_stage`).
    pub multi_dpos_pinned: usize,
    pub multi_dpos_dev: Option<CudaSlice<u32>>,
    /// Hybrid FA decode: per-layer (pre-attend, post-attend) graphs + final tail.
    /// Used when ALLPAKA_CUDA_GRAPH=1 and ggml FA is available.
    pub decode_fa_graphs: Option<(Vec<CudaGraph>, Vec<CudaGraph>, CudaGraph)>,
    /// Per-layer FA(+mask+permute) graphs when direct fattn is stream-capturable.
    pub decode_fa_attn: Option<Vec<CudaGraph>>,
    /// Cached driver exec handles + KV offs for the hybrid replay hot path.
    pub decode_fa_execs: Option<(
        Vec<*mut std::ffi::c_void>,
        Vec<*mut std::ffi::c_void>,
        *mut std::ffi::c_void,
    )>,
    pub decode_fa_kv_offs: Option<(Vec<i32>, Vec<i32>)>,
    pub decode_graph_key: u64,
    pub decode_norms_key: u64,
    /// Decode graphs of other sessions, newest last, kept while another
    /// session decodes so alternating between them replays instead of
    /// recapturing (a recapture per switch cost +186% on Qwen2.5-7B).
    pub decode_graphs_parked: Vec<ParkedDecodeGraphs>,
    /// Persistent scratch for RMSNorm weights (avoids cudaMalloc per layer).
    pub w_scratch: CudaSlice<f32>,
    pub w_scratch_cap: usize,
    /// Small host f32 tensors (norm weights) mirrored on device, keyed by
    /// host address; a lookup re-checks the contents against a host copy.
    pub f32_cache: HashMap<usize, (Vec<f32>, CudaSlice<f32>)>,
    /// Prefill: device address of an `m * hidden` addend not yet folded into
    /// `pf_x` (the dense FFN's down projection). The next attention RMSNorm
    /// folds it in with `add_rmsnorm_into_f32`; anything else that reads or
    /// writes `pf_x` (or may reallocate `y_arena`) calls `pf_flush_add` first.
    pub pf_pending_add: Option<(u64, usize)>,
    pub pf_rope_n: usize,
    /// Host address of the table `pf_rope_n` describes: Gemma 3 alternates
    /// two tables of the same length between layers.
    pub pf_rope_src: usize,
    pub x_f16_n: usize,
    /// Second stream: dequant next W while compute GEMMs the current W.
    pub dq_stream: Arc<CudaStream>,
    pub ev_w0: CudaEvent,
    pub ev_w1: CudaEvent,
    /// Recorded on `stream` after the GEMM that read W slot 0 / 1; the
    /// dequant of the next weight into that slot waits on it (WAR).
    pub ev_c0: CudaEvent,
    pub ev_c1: CudaEvent,
    pub w_ping: usize,
    pub w_pitch: usize,
    /// Reserved f16 elems for X before ping-pong W. Grows to max(m*n_in), never shrinks mid-run.
    pub f16_x_region: usize,
    pub q8_q: CudaSlice<i8>,
    pub q8_q_cap: usize,
    pub q8_d: CudaSlice<f32>,
    /// Last quantized activation pointer (skip re-quantize for Q/K/V share).
    pub q8_src: u64,
    pub q8_src_n: usize,
    pub q8_src_m: usize,
    /// Byte offset into `q8_q` where the current packed activation starts.
    pub q8_off: usize,
    /// Last L2-persist window size applied to `q8_q` (0 = never).
    pub q8_l2_bytes: usize,
    /// FA output device ptr for fused WO f32→Q8 mmvq (`ALLPAKA_WO_FA_Q8`).
    pub wo_fa_x: u64,
    /// Split-KV partials (m, l, pad, pad, acc[128]) for attn_decode_g8_q8.
    pub attn_part: CudaSlice<f32>,
    /// Zeroed per-kv-head completion counters (the kernel resets them).
    pub attn_cnt: CudaSlice<u32>,
}

// SAFETY: what keeps `CudaGpu` from being Send automatically is `CudaGraph` (raw
// CUgraph / CUgraphExec handles) and the raw exec pointers in `decode_fa_execs`. The
// only instance lives inside the `Mutex` in `GPU`, so exactly one thread uses it at a
// time. Driver handles are not thread-affine: they belong to the device context, and
// every cudarc call (and `CudaGraph::drop`) binds that context to the calling thread
// first; the raw `graph::launch` replay path in decode_fused calls `bind_to_thread`
// itself before launching. `decode_fa_execs` is only ever assigned `None`; were it
// filled, its pointers would be exec handles owned by graphs in this same struct.
// `Sync` is deliberately not implemented: shared access goes through the mutex.
unsafe impl Send for CudaGpu {}

pub static GPU: OnceLock<Option<Mutex<CudaGpu>>> = OnceLock::new();

pub fn note_call(dispatches: u64, encode_ns: u64, wait_ns: u64) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    DISPATCHES.fetch_add(dispatches, Ordering::Relaxed);
    ENCODE_NS.fetch_add(encode_ns, Ordering::Relaxed);
    WAIT_NS.fetch_add(wait_ns, Ordering::Relaxed);
}

pub(super) fn init_device() -> Option<CudaGpu> {
    let ctx = CudaContext::new(0)
        .map_err(|e| eprintln!("cuda: no device: {e}"))
        .ok()?;
    // Manual stream sync for dequant/compute overlap (events below).
    unsafe { ctx.disable_event_tracking() };
    // CUDA graph capture and ggml peer inject both need a non-default stream
    // (null cannot be installed into ggml). Prefer that whenever graphs, ggml
    // FA/MMQ, or ggml decode may share the queue. DEFAULT_STREAM=1 keeps the
    // legacy default-stream path (pp-oriented; peer share is skipped).
    let want_graph = crate::runtime::get().cuda_graph;
    let want_ggml = crate::gpu::cuda::ggml::enabled();
    let want_ggml_decode = std::env::var("ALLPAKA_GGML_DECODE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let force_default = std::env::var("ALLPAKA_CUDA_DEFAULT_STREAM").is_ok_and(|v| v == "1");
    let stream = if force_default && !want_ggml_decode {
        ctx.default_stream()
    } else if want_graph || want_ggml || want_ggml_decode {
        ctx.new_stream().ok()?
    } else {
        ctx.default_stream()
    };
    let dq_stream = ctx.new_stream().ok()?;
    let ev_w0 = ctx.new_event(None).ok()?;
    let ev_w1 = ctx.new_event(None).ok()?;
    let ev_c0 = ctx.new_event(None).ok()?;
    let ev_c1 = ctx.new_event(None).ok()?;

    let mut include_paths = Vec::new();
    if let Ok(root) = std::env::var("CUDA_PATH") {
        let inc = std::path::Path::new(&root).join("include");
        if inc.is_dir() {
            include_paths.push(inc.to_string_lossy().into_owned());
        }
    }
    // Fallback well-known install location on Windows.
    let fallback =
        std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\include");
    if include_paths.is_empty() && fallback.is_dir() {
        include_paths.push(fallback.to_string_lossy().into_owned());
    }

    let make_opts = |arch: Option<&'static str>| CompileOptions {
        arch,
        use_fast_math: Some(true),
        include_paths: include_paths.clone(),
        options: vec!["-std=c++17".into()],
        ..Default::default()
    };
    let ptx = match compile_ptx_with_opts(KERNELS, make_opts(Some("compute_120"))) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cuda: NVRTC compute_120 failed ({e}), retrying default arch");
            compile_ptx_with_opts(KERNELS, make_opts(None))
                .map_err(|e2| eprintln!("cuda: NVRTC compile failed: {e2}"))
                .ok()?
        }
    };
    // Modules are loaded through the raw driver API: launches need the
    // `CUfunction` handle, which cudarc does not expose (see `pdl.rs`).
    let load_kfn = |mm: &RawModule, name: &'static str| -> Option<KernelFn> {
        let f = mm.function(name);
        if f.is_none() {
            eprintln!("cuda: load_function {name}: not found");
        }
        f
    };
    let module = RawModule::load_ptx(&ptx.to_src())
        .map_err(|e| eprintln!("cuda: load_module failed: {e}"))
        .ok()?;
    let mut fns = HashMap::new();
    for &name in KERNEL_NAMES {
        if name == "mm_q4_k_mmq" {
            continue; // loaded from nvcc PTX below
        }
        fns.insert(name, load_kfn(&module, name)?);
    }

    // Optional nvcc-compiled Q4_K MMQ (ALLPAKA_MMQ=nvcc). Keep module alive for fns.
    let mmq_ptx_src = include_str!("../../../../cuda/mmq_q4_k.ptx");
    let mut mmq_module = None;
    match RawModule::load_ptx(mmq_ptx_src) {
        Ok(mm) => {
            if let Some(f) = load_kfn(&mm, "mm_q4_k_mmq") {
                // Dynamic smem ≈ 52 KiB (As+Ad+Asum+Wtile+scales+Bw).
                let _ = f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    65_536,
                );
                fns.insert("mm_q4_k_mmq", f);
                println!("cuda: nvcc MMQ kernel loaded");
            }
            mmq_module = Some(mm);
        }
        Err(e) => eprintln!("cuda: mmq PTX load failed: {e}"),
    }

    // nvcc MMVQ overrides NVRTC ports (better codegen on sm_120).
    let mmvq_ptx_src = include_str!("../../../../cuda/mmvq_q8.ptx");
    let mut mmvq_module = None;
    match RawModule::load_ptx(mmvq_ptx_src) {
        Ok(mm) => {
            let mut n = 0usize;
            for name in [
                "matvec_q4_k_q8",
                "matvec_q4_k_q8_k8192",
                "matvec_q4_k_q8_k5120",
                "matvec_q4_k_f32_k8192",
                "matvec_q4_k_q8_n2",
                "matvec_q4_k_q8_r2",
                "matvec_q4_k_q8_n8",
                "matvec_q6_k_q8",
                "matvec_q6_k_q8_n8",
                "matvec_q6_k_q8_n8_k25600",
                "matvec_q6_k_f32_n8",
                "matvec_q4_k_q8_2",
                "matvec_q4_k_q8_2_k5120",
                "matvec_q4_k_q8_2_k5120_bal",
                "matvec_q4_k_q8_2_n8",
                "matvec_q4_k_q8_2_q8",
                "matvec_q4_k_q8_qkv",
                "matvec_q4_q4_q6_q8_qkv",
                "quantize_q8_1",
                "quantize_q8_1_q6",
            ] {
                if let Some(f) = load_kfn(&mm, name) {
                    fns.insert(name, f);
                    n += 1;
                }
            }
            if n > 0 {
                println!("cuda: nvcc MMVQ kernels loaded ({n})");
            }
            mmvq_module = Some(mm);
        }
        Err(e) => eprintln!("cuda: mmvq PTX load failed: {e}"),
    }

    let mut embed_module = None;
    let embed_ptx = match compile_ptx_with_opts(EMBED_KERNELS, make_opts(Some("compute_120"))) {
        Ok(p) => Some(p),
        Err(_) => compile_ptx_with_opts(EMBED_KERNELS, make_opts(None)).ok(),
    };
    if let Some(ptx) = embed_ptx {
        match RawModule::load_ptx(&ptx.to_src()) {
            Ok(mm) => {
                if let Some(f) = load_kfn(&mm, "embed_row_f32") {
                    fns.insert("embed_row_f32", f);
                }
                if let Some(f) = load_kfn(&mm, "embed_row_f32_dtoken") {
                    fns.insert("embed_row_f32_dtoken", f);
                }
                if let Some(f) = load_kfn(&mm, "inc_u32") {
                    fns.insert("inc_u32", f);
                }
                if let Some(f) = load_kfn(&mm, "copy_u32") {
                    fns.insert("copy_u32", f);
                }
                if let Some(f) = load_kfn(&mm, "store_u32_at") {
                    fns.insert("store_u32_at", f);
                }
                if fns.contains_key("embed_row_f32") {
                    println!("cuda: embed chain kernels loaded");
                }
                embed_module = Some(mm);
            }
            Err(e) => eprintln!("cuda: embed module load failed: {e}"),
        }
    } else {
        eprintln!("cuda: embed NVRTC failed");
    }

    let blaslt = CudaBlasLT::new(stream.clone())
        .map_err(|e| eprintln!("cuda: cublasLt unavailable: {e}"))
        .ok();

    let x_arena = stream.alloc_zeros::<f32>(1 << 20).ok()?;
    let y_arena = stream.alloc_zeros::<f32>(1 << 20).ok()?;
    // Allocated here, next to the arenas, rather than grown on the first
    // decode token: with a late allocation decode measured 80.3 against 80.6
    // tok/s (Qwen3-32B) and 303.5 against 305.9 (Qwen2.5-7B), three rounds each.
    let dec_x = stream.alloc_zeros::<f32>(1 << 20).ok()?;
    let f16_arena = stream.alloc_zeros::<f16>(1 << 20).ok()?;
    let pf_x = stream.alloc_zeros::<f32>(1 << 10).ok()?;
    let pf_hs = stream.alloc_zeros::<f32>(1 << 10).ok()?;
    let d_pos = stream.alloc_zeros::<u32>(1).ok()?;
    let d_argmax = stream.alloc_zeros::<u32>(1).ok()?;
    let w_scratch = stream.alloc_zeros::<f32>(1 << 14).ok()?;
    let q8_q = stream.alloc_zeros::<i8>(1 << 16).ok()?;
    let q8_d = stream.alloc_zeros::<f32>(1 << 12).ok()?;
    let attn_part = stream.alloc_zeros::<f32>(ATTN_PART_ELEMS).ok()?;
    let attn_cnt = stream.alloc_zeros::<u32>(ATTN_CNT_ELEMS).ok()?;

    println!("cuda: device attached, kernels compiled");

    if crate::gpu::cuda::ggml::enabled() {
        // Bind early so prefill MMQ/FA and optional decode MMVQ share the stream.
        crate::gpu::cuda::ggml::bind_peer_stream(&stream);
    }

    Some(CudaGpu {
        ctx,
        stream,
        module,
        mmq_module,
        mmvq_module,
        embed_module,
        fns,
        chunks: Vec::new(),
        attached: Vec::new(),
        blaslt,
        x_arena,
        x_cap: 1 << 20,
        y_arena,
        y_cap: 1 << 20,
        dec_x,
        dec_x_cap: 1 << 20,
        f16_arena,
        f16_cap: 1 << 20,
        pf_x,
        pf_x_cap: 1 << 10,
        pf_hs,
        pf_hs_cap: 1 << 10,
        pf_active: false,
        pf_len: 0,
        pf_ffn_done: false,
        d_pos,
        d_rope_freq: None,
        d_rope_freq_n: 0,
        d_rope_freq_swa: None,
        d_argmax,
        decode_graph: None,
        pf_graph: None,
        pf_graph_len: 0,
        pf_graph_seen: 0,
        pf_capturing: false,
        pf_graph_key: 0,
        pf_graph_want: 0,
        pf_capture_bad: false,
        pf_graph_failed: false,
        pf_rope_pinned: (0, 0),
        pf_rope_scratch: (0, 0),
        multi_dpos_pinned: 0,
        multi_dpos_dev: None,
        decode_fa_graphs: None,
        decode_fa_attn: None,
        decode_fa_execs: None,
        decode_fa_kv_offs: None,
        decode_graph_key: 0,
        decode_graphs_parked: Vec::new(),
        decode_norms_key: 0,
        w_scratch,
        w_scratch_cap: 1 << 14,
        f32_cache: HashMap::new(),
        pf_pending_add: None,
        pf_rope_n: 0,
        pf_rope_src: 0,
        x_f16_n: 0,
        dq_stream,
        ev_w0,
        ev_w1,
        ev_c0,
        ev_c1,
        w_ping: 0,
        w_pitch: 0,
        f16_x_region: 0,
        q8_q,
        q8_q_cap: 1 << 16,
        q8_d,
        q8_src: 0,
        q8_src_n: 0,
        q8_src_m: 0,
        q8_off: 0,
        q8_l2_bytes: 0,
        wo_fa_x: 0,
        attn_part,
        attn_cnt,
    })
}

impl CudaGpu {
    pub fn func_owned(&self, name: &str) -> Option<KernelFn> {
        self.fns.get(name).cloned()
    }

    pub fn covers(&self, addr: usize, len: usize) -> bool {
        self.chunk_for(addr, len).is_some()
    }

    pub fn chunk_for(&self, addr: usize, len: usize) -> Option<usize> {
        self.chunks
            .iter()
            .position(|c| addr >= c.start && addr + len <= c.start + c.len)
    }

    /// Upload `mapping` once and count its owners. A sub-range of an already
    /// uploaded mapping needs no copy and is not counted.
    /// The scratch arenas' size in bytes, as allocated now. They are grown
    /// on demand and never shrunk, so this is also their peak.
    pub fn arena_bytes(&self) -> usize {
        let f32s = self.x_arena.len()
            + self.y_arena.len()
            + self.dec_x.len()
            + self.pf_x.len()
            + self.pf_hs.len()
            + self.w_scratch.len()
            + self.q8_d.len()
            + self.attn_part.len();
        let u32s = self.d_pos.len() + self.d_argmax.len() + self.attn_cnt.len();
        (f32s + u32s) * 4 + self.f16_arena.len() * 2 + self.q8_q.len()
    }

    pub fn attach_mapping(&mut self, mapping: &[u8]) -> bool {
        let key = (mapping.as_ptr() as usize, mapping.len());
        if let Some(entry) = self.attached.iter_mut().find(|e| (e.0, e.1) == key) {
            entry.2 += 1;
            return true;
        }
        if self.covers(key.0, key.1) {
            return true;
        }
        if !self.add_mapping(mapping) {
            return false;
        }
        self.attached.push((key.0, key.1, 1));
        true
    }

    /// Drop one owner of `mapping`; the last one frees its windows and every
    /// captured graph, since graphs bake in the windows' device pointers.
    pub fn detach_mapping(&mut self, mapping: &[u8]) {
        let key = (mapping.as_ptr() as usize, mapping.len());
        let Some(at) = self.attached.iter().position(|e| (e.0, e.1) == key) else {
            return;
        };
        self.attached[at].2 -= 1;
        if self.attached[at].2 > 0 {
            return;
        }
        self.attached.swap_remove(at);
        let _ = self.stream.synchronize();
        let _ = self.dq_stream.synchronize();
        let end = key.0 + key.1.div_ceil(16384) * 16384;
        self.chunks.retain(|c| c.start < key.0 || c.start >= end);
        self.drop_graphs();
    }

    /// Forget the captured prefill graph (its baked buffer addresses or
    /// weights are about to change). During a capture the recording is
    /// poisoned instead: `prefill_end` discards it and the chunk reruns.
    pub fn drop_pf_graph(&mut self) {
        self.pf_graph = None;
        if self.pf_capturing {
            self.pf_capture_bad = true;
        }
    }

    /// Forget captured decode graphs and the keys that select them.
    pub fn drop_graphs(&mut self) {
        self.clear_decode_graphs();
        self.drop_pf_graph();
        self.decode_graph_key = 0;
        self.decode_norms_key = 0;
    }

    /// Drop every captured decode graph (whole-token, hybrid segments, FA
    /// graphs and their KV offsets); the selecting keys are left to the caller.
    pub fn clear_decode_graphs(&mut self) {
        self.clear_active_decode_graphs();
        self.decode_graphs_parked.clear();
    }

    /// Drop the active session's decode graphs only, before capturing new
    /// ones for it; graphs other sessions parked stay valid.
    pub fn clear_active_decode_graphs(&mut self) {
        self.decode_graph = None;
        self.decode_fa_graphs = None;
        self.decode_fa_attn = None;
        self.decode_fa_kv_offs = None;
        self.decode_fa_execs = None;
    }

    /// Switch the active decode graphs to those captured for `key`: set the
    /// current ones aside (if any) and bring back `key`'s if they were set
    /// aside earlier. Afterwards `decode_graph_key == key` exactly when graphs
    /// for `key` are active; otherwise the caller captures as before.
    pub fn select_decode_graphs(&mut self, key: u64) {
        if self.decode_graph_key == key {
            return;
        }
        let active = self.decode_graph.is_some()
            || self.decode_fa_graphs.is_some()
            || self.decode_fa_attn.is_some();
        if self.decode_graph_key != 0 && active {
            let parked = ParkedDecodeGraphs {
                key: self.decode_graph_key,
                graph: self.decode_graph.take(),
                fa_graphs: self.decode_fa_graphs.take(),
                fa_attn: self.decode_fa_attn.take(),
                fa_execs: self.decode_fa_execs.take(),
                fa_kv_offs: self.decode_fa_kv_offs.take(),
            };
            self.decode_graphs_parked.retain(|p| p.key != parked.key);
            self.decode_graphs_parked.push(parked);
            if self.decode_graphs_parked.len() > DECODE_GRAPHS_PARKED {
                self.decode_graphs_parked.remove(0);
            }
        } else {
            self.decode_graph = None;
            self.decode_fa_graphs = None;
            self.decode_fa_attn = None;
            self.decode_fa_execs = None;
            self.decode_fa_kv_offs = None;
        }
        self.decode_graph_key = 0;
        if let Some(at) = self.decode_graphs_parked.iter().position(|p| p.key == key) {
            let back = self.decode_graphs_parked.remove(at);
            self.decode_graph = back.graph;
            self.decode_fa_graphs = back.fa_graphs;
            self.decode_fa_attn = back.fa_attn;
            self.decode_fa_execs = back.fa_execs;
            self.decode_fa_kv_offs = back.fa_kv_offs;
            self.decode_graph_key = key;
        }
    }

    pub fn add_mapping(&mut self, mapping: &[u8]) -> bool {
        let page = 16384usize;
        let base = mapping.as_ptr() as usize;
        let aligned_len = mapping.len().div_ceil(page) * page;
        const WINDOW_CAP: usize = 32 << 30;
        let mut max_buf = WINDOW_CAP;
        if let Ok(cap) = std::env::var("ALLPAKA_GPU_WINDOW_GIB") {
            if let Ok(gib) = cap.parse::<usize>() {
                max_buf = max_buf.min(gib << 30);
            }
        }
        let overlap = 2usize << 30;
        let step = if max_buf > overlap {
            max_buf - overlap
        } else {
            max_buf / 2
        }
        .max(page);
        let step = (step / page).max(1) * page;

        let mut added = Vec::new();
        let mut start = 0usize;
        loop {
            let len = max_buf.min(aligned_len - start);
            let host_len = len.min(mapping.len().saturating_sub(start));
            let mut buf = match unsafe { self.stream.alloc::<u8>(len) } {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "cuda: alloc {:.1} GiB window failed: {e}",
                        len as f64 / (1u64 << 30) as f64
                    );
                    return false;
                }
            };
            if host_len > 0 {
                if let Err(e) = self.stream.memcpy_htod(
                    &mapping[start..start + host_len],
                    &mut buf.slice_mut(0..host_len),
                ) {
                    eprintln!("cuda: H2D weight upload failed: {e}");
                    return false;
                }
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
            "cuda: attached {:.1} GiB of weights in {} window(s)",
            mapping.len() as f64 / (1u64 << 30) as f64,
            added.len()
        );
        self.chunks.extend(added);
        if let Ok(total) = self.ctx.total_mem() {
            let weights: usize = self.chunks.iter().map(|c| c.len).sum();
            crate::gpu::cuda::ggml::set_f16_budget(crate::gpu::cuda::ggml::f16_cache_budget(
                total as u64,
                weights as u64,
            ));
        }
        true
    }

    /// Fold a deferred prefill residual (`pf_pending_add`) into `pf_x`.
    pub fn pf_flush_add(&mut self) -> Option<()> {
        let Some((add, len)) = self.pf_pending_add.take() else {
            return Some(());
        };
        let f = self.func_owned("residual_add")?;
        let n = len as u32;
        let cfg = CudaGpu::cfg_1d(n, 256);
        let mut a = self.pf_x.slice_mut(0..len);
        unsafe {
            self.stream
                .launch_builder_pdl(&f)
                .arg(&mut a)
                .arg(&add)
                .arg(&n)
                .launch_pdl(cfg)
        }
        .ok()?;
        Some(())
    }

    pub fn ensure_arenas(&mut self, x_need: usize, y_need: usize) -> Option<()> {
        if y_need > self.y_cap * 4 {
            // The deferred addend lives in y_arena.
            self.pf_flush_add()?;
        }
        let mut grew = false;
        if x_need > self.x_cap * 4 {
            let elems = x_need.div_ceil(4).next_power_of_two();
            self.x_arena = self.stream.alloc_zeros::<f32>(elems).ok()?;
            self.x_cap = elems;
            grew = true;
        }
        if y_need > self.y_cap * 4 {
            let elems = y_need.div_ceil(4).next_power_of_two();
            self.y_arena = self.stream.alloc_zeros::<f32>(elems).ok()?;
            self.y_cap = elems;
            grew = true;
        }
        if grew {
            self.drop_graphs();
            self.pf_rope_n = 0;
            self.x_f16_n = 0;
        }
        Some(())
    }

    /// Size the fused decode's persistent input buffer (`elems` f32). Growing
    /// it moves the buffer, so captured graphs and the uploaded norms go.
    pub fn ensure_dec_x(&mut self, elems: usize) -> Option<()> {
        if elems > self.dec_x_cap {
            let _ = self.stream.synchronize();
            let cap = elems.next_power_of_two();
            self.dec_x = self.stream.alloc_zeros::<f32>(cap).ok()?;
            self.dec_x_cap = cap;
            self.drop_graphs();
        }
        Some(())
    }

    pub fn ensure_f16(&mut self, elems: usize) -> Option<()> {
        if elems > self.f16_cap {
            let _ = self.stream.synchronize();
            let _ = self.dq_stream.synchronize();
            let cap = elems.next_power_of_two();
            self.f16_arena = self.stream.alloc_zeros::<f16>(cap).ok()?;
            self.f16_cap = cap;
            self.x_f16_n = 0;
        }
        Some(())
    }

    pub fn ensure_prefill(&mut self, elems: usize) -> Option<()> {
        if elems > self.pf_x_cap || elems > self.pf_hs_cap {
            self.drop_pf_graph();
        }
        if elems > self.pf_x_cap {
            let cap = elems.next_power_of_two();
            self.pf_x = self.stream.alloc_zeros::<f32>(cap).ok()?;
            self.pf_x_cap = cap;
        }
        if elems > self.pf_hs_cap {
            let cap = elems.next_power_of_two();
            self.pf_hs = self.stream.alloc_zeros::<f32>(cap).ok()?;
            self.pf_hs_cap = cap;
        }
        Some(())
    }

    pub fn invalidate_q8(&mut self) {
        self.q8_src = 0;
        self.q8_src_n = 0;
        self.q8_src_m = 0;
        self.q8_off = 0;
    }

    pub fn ensure_q8(&mut self, n: usize, rows: usize) -> Option<()> {
        // Packed Q8 block: float scale + 32 quants + four Q4_K partial sums.
        self.ensure_q8_bytes((n / 32) * 44 * rows)
    }

    pub fn ensure_q8_bytes(&mut self, q_need: usize) -> Option<()> {
        if q_need > self.q8_q_cap {
            let cap = q_need.next_power_of_two();
            self.q8_q = self.stream.alloc_zeros::<i8>(cap).ok()?;
            self.q8_q_cap = cap;
            self.q8_l2_bytes = 0;
            // New allocation drops prior packed activations.
            self.invalidate_q8();
        }
        Some(())
    }

    pub fn ensure_w_scratch(&mut self, elems: usize) -> Option<()> {
        if elems > self.w_scratch_cap {
            let cap = elems.next_power_of_two();
            self.w_scratch = self.stream.alloc_zeros::<f32>(cap).ok()?;
            self.w_scratch_cap = cap;
        }
        Some(())
    }

    pub fn sync(&self) -> Option<()> {
        crate::gpu::cuda::ggml::sync();
        self.stream.synchronize().ok()
    }

    pub fn cfg_1d(n: u32, threads: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n.div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

pub fn try_init() -> Option<()> {
    // CUDA 13.x on Windows keeps DLLs under bin\x64; ensure the loader sees them.
    if let Ok(root) = std::env::var("CUDA_PATH") {
        let x64 = std::path::Path::new(&root).join("bin").join("x64");
        if x64.is_dir() {
            let mut path = std::env::var_os("PATH").unwrap_or_default();
            let prefix = x64.as_os_str();
            if !path
                .to_string_lossy()
                .to_ascii_lowercase()
                .contains(&x64.to_string_lossy().to_ascii_lowercase())
            {
                let mut new_path = prefix.to_os_string();
                new_path.push(";");
                new_path.push(path);
                // Also keep classic bin/ for nvcc-adjacent tools.
                let bin = std::path::Path::new(&root).join("bin");
                if bin.is_dir() {
                    new_path.push(";");
                    new_path.push(bin.as_os_str());
                }
                path = new_path;
                std::env::set_var("PATH", path);
            }
        }
    }
    let slot = std::panic::catch_unwind(|| GPU.get_or_init(|| init_device().map(Mutex::new)));
    match slot {
        Ok(_) => GPU.get().and_then(|g| g.as_ref()).map(|_| ()),
        Err(_) => {
            eprintln!("cuda: accelerator init panicked (missing NVRTC/CUDA DLLs?)");
            None
        }
    }
}

pub fn with_gpu<R>(f: impl FnOnce(&mut CudaGpu) -> Option<R>) -> Option<R> {
    let cell = GPU.get()?.as_ref()?;
    let mut gpu = cell.lock().ok()?;
    f(&mut gpu)
}
