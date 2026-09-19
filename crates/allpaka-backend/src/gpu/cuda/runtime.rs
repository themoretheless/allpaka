//! CUDA device runtime: context, streams, weight residency, scratch arenas.

use crate::gpu::cuda::kernels::{EMBED_KERNELS, KERNELS};
use crate::gpu::cuda::pdl::{LaunchArgsPdl, StreamPdlExt};
use allpaka_gguf::GgmlType;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use half::f16;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub static CALLS: AtomicU64 = AtomicU64::new(0);
pub static DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub static ENCODE_NS: AtomicU64 = AtomicU64::new(0);
pub static WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);
pub static SCHED_NS: AtomicU64 = AtomicU64::new(0);

pub static DECODE_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
pub static DECODE_SUCCESSES: AtomicU64 = AtomicU64::new(0);
pub static DECODE_DECLINES: AtomicU64 = AtomicU64::new(0);

/// Batch size from which dequant + GEMM beats a loop of matvecs.
pub const MM_MIN_M: usize = 32;

const KERNEL_NAMES: &[&str] = &[
    "rmsnorm_f32",
    "rmsnorm_into_f32",
    "rmsnorm_into_f32_q8",
    "residual_add",
    "swiglu",
    "swiglu_into_q8",
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
    pub module: Arc<CudaModule>,
    /// Holds nvcc MMQ module so `mm_q4_k_mmq` stays valid.
    #[allow(dead_code)]
    pub mmq_module: Option<Arc<CudaModule>>,
    /// Holds nvcc MMVQ module so Q8 matvec kernels stay valid.
    #[allow(dead_code)]
    pub mmvq_module: Option<Arc<CudaModule>>,
    /// Holds NVRTC embed_row module (kept separate from main KERNELS PTX).
    #[allow(dead_code)]
    pub embed_module: Option<Arc<CudaModule>>,
    pub fns: HashMap<&'static str, CudaFunction>,
    pub chunks: Vec<WeightChunk>,
    pub blaslt: Option<CudaBlasLT>,
    pub x_arena: CudaSlice<f32>,
    pub x_cap: usize,
    pub y_arena: CudaSlice<f32>,
    pub y_cap: usize,
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
    /// Scratch for argmax index (avoids alloc during graph capture).
    pub d_argmax: CudaSlice<u32>,
    /// Captured whole-token dense decode graph (invalidated on shape change).
    /// For hybrid FA: optional mega-graph that replays segment launches + FA in one go.
    pub decode_graph: Option<CudaGraph>,
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
    /// Persistent scratch for RMSNorm weights (avoids cudaMalloc per layer).
    pub w_scratch: CudaSlice<f32>,
    pub w_scratch_cap: usize,
    pub pf_rope_n: usize,
    pub x_f16_n: usize,
    /// Second stream: dequant next W while compute GEMMs the current W.
    pub dq_stream: Arc<CudaStream>,
    pub ev_w0: CudaEvent,
    pub ev_w1: CudaEvent,
    pub w_ping: usize,
    pub w_pitch: usize,
    /// Reserved f16 elems for X before ping-pong W. Grows to max(m*n_in), never shrinks mid-run.
    pub f16_x_region: usize,
    pub q8_q: CudaSlice<i8>,
    pub q8_q_cap: usize,
    pub q8_d: CudaSlice<f32>,
    pub q8_d_cap: usize,
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
}

unsafe impl Send for CudaGpu {}

pub static GPU: OnceLock<Option<Mutex<CudaGpu>>> = OnceLock::new();

pub fn note_call(dispatches: u64, encode_ns: u64, wait_ns: u64) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    DISPATCHES.fetch_add(dispatches, Ordering::Relaxed);
    ENCODE_NS.fetch_add(encode_ns, Ordering::Relaxed);
    WAIT_NS.fetch_add(wait_ns, Ordering::Relaxed);
}

fn init_device() -> Option<CudaGpu> {
    let ctx = CudaContext::new(0)
        .map_err(|e| eprintln!("cuda: no device: {e}"))
        .ok()?;
    // Manual stream sync for dequant/compute overlap (events below).
    unsafe { ctx.disable_event_tracking() };
    // CUDA graph capture and ggml peer inject both need a non-default stream
    // (null cannot be installed into ggml). Prefer that whenever graphs, ggml
    // FA/MMQ, or ggml decode may share the queue. DEFAULT_STREAM=1 keeps the
    // legacy default-stream path (pp-oriented; peer share is skipped).
    let want_graph = std::env::var("ALLPAKA_CUDA_GRAPH").map_or(false, |v| v == "1");
    let want_ggml = crate::gpu::cuda::ggml::enabled();
    let want_ggml_decode = std::env::var("ALLPAKA_GGML_DECODE")
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));
    let force_default = std::env::var("ALLPAKA_CUDA_DEFAULT_STREAM").map_or(false, |v| v == "1");
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
    let module = ctx
        .load_module(ptx)
        .map_err(|e| eprintln!("cuda: load_module failed: {e}"))
        .ok()?;
    let mut fns = HashMap::new();
    for &name in KERNEL_NAMES {
        if name == "mm_q4_k_mmq" {
            continue; // loaded from nvcc PTX below
        }
        let f = module
            .load_function(name)
            .map_err(|e| eprintln!("cuda: load_function {name}: {e}"))
            .ok()?;
        fns.insert(name, f);
    }

    // Optional nvcc-compiled Q4_K MMQ (ALLPAKA_MMQ=nvcc). Keep module alive for fns.
    let mmq_ptx_src = include_str!("../../../cuda/mmq_q4_k.ptx");
    let mut mmq_module = None;
    match ctx.load_module(Ptx::from_src(mmq_ptx_src)) {
        Ok(mm) => {
            if let Ok(f) = mm.load_function("mm_q4_k_mmq") {
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
    let mmvq_ptx_src = include_str!("../../../cuda/mmvq_q8.ptx");
    let mut mmvq_module = None;
    match ctx.load_module(Ptx::from_src(mmvq_ptx_src)) {
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
                "matvec_q4_k_q8_2_n8",
                "matvec_q4_k_q8_2_q8",
                "matvec_q4_k_q8_qkv",
                "matvec_q4_q4_q6_q8_qkv",
                "quantize_q8_1",
                "quantize_q8_1_q6",
            ] {
                if let Ok(f) = mm.load_function(name) {
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
        match ctx.load_module(ptx) {
            Ok(mm) => {
                if let Ok(f) = mm.load_function("embed_row_f32") {
                    fns.insert("embed_row_f32", f);
                }
                if let Ok(f) = mm.load_function("embed_row_f32_dtoken") {
                    fns.insert("embed_row_f32_dtoken", f);
                }
                if let Ok(f) = mm.load_function("inc_u32") {
                    fns.insert("inc_u32", f);
                }
                if let Ok(f) = mm.load_function("copy_u32") {
                    fns.insert("copy_u32", f);
                }
                if let Ok(f) = mm.load_function("store_u32_at") {
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
    let f16_arena = stream.alloc_zeros::<f16>(1 << 20).ok()?;
    let pf_x = stream.alloc_zeros::<f32>(1 << 10).ok()?;
    let pf_hs = stream.alloc_zeros::<f32>(1 << 10).ok()?;
    let d_pos = stream.alloc_zeros::<u32>(1).ok()?;
    let d_argmax = stream.alloc_zeros::<u32>(1).ok()?;
    let w_scratch = stream.alloc_zeros::<f32>(1 << 14).ok()?;
    let q8_q = stream.alloc_zeros::<i8>(1 << 16).ok()?;
    let q8_d = stream.alloc_zeros::<f32>(1 << 12).ok()?;

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
        blaslt,
        x_arena,
        x_cap: 1 << 20,
        y_arena,
        y_cap: 1 << 20,
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
        d_argmax,
        decode_graph: None,
        decode_fa_graphs: None,
        decode_fa_attn: None,
        decode_fa_execs: None,
        decode_fa_kv_offs: None,
        decode_graph_key: 0,
        decode_norms_key: 0,
        w_scratch,
        w_scratch_cap: 1 << 14,
        pf_rope_n: 0,
        x_f16_n: 0,
        dq_stream,
        ev_w0,
        ev_w1,
        w_ping: 0,
        w_pitch: 0,
        f16_x_region: 0,
        q8_q,
        q8_q_cap: 1 << 16,
        q8_d,
        q8_d_cap: 1 << 12,
        q8_src: 0,
        q8_src_n: 0,
        q8_src_m: 0,
        q8_off: 0,
        q8_l2_bytes: 0,
        wo_fa_x: 0,
    })
}

impl CudaGpu {
    pub fn func_owned(&self, name: &str) -> Option<CudaFunction> {
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
        true
    }

    pub fn ensure_arenas(&mut self, x_need: usize, y_need: usize) -> Option<()> {
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
            self.decode_graph = None;
            self.decode_fa_graphs = None;
            self.decode_fa_attn = None;
            self.decode_fa_execs = None;
            self.decode_fa_kv_offs = None;
            self.decode_graph_key = 0;
            self.decode_norms_key = 0;
            self.pf_rope_n = 0;
            self.x_f16_n = 0;
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

pub fn matvec_kernel(ty: GgmlType) -> Option<&'static str> {
    Some(match ty {
        GgmlType::F32 => "matvec_f32",
        GgmlType::F16 => "matvec_f16",
        GgmlType::BF16 => "matvec_bf16",
        GgmlType::Q5_0 => "matvec_q5_0",
        GgmlType::Q8_0 => "matvec_q8_0",
        GgmlType::Q2K => "matvec_q2_k",
        GgmlType::Q3K => "matvec_q3_k",
        GgmlType::Q4K => "matvec_q4_k",
        GgmlType::Q5K => "matvec_q5_k",
        GgmlType::Q6K => "matvec_q6_k",
        GgmlType::Other(_) => return None,
    })
}

/// Q4/Q6_K: Metal-style — 4 warps × 4 output rows per block.
/// Q5_K: one warp per output row, several rows per block.
pub fn matvec_warp_rows(ty: GgmlType) -> Option<u32> {
    match ty {
        GgmlType::Q4K | GgmlType::Q6K => Some(4),
        GgmlType::Q5K => Some(8),
        _ => None,
    }
}

pub fn matvec_launch_cfg(ty: GgmlType, n_out: u32, m: u32) -> LaunchConfig {
    match ty {
        // Q4/Q6_K: 4 warps × 4 rows = 16 outs/block (best Metal-style decode).
        GgmlType::Q4K | GgmlType::Q6K => LaunchConfig {
            grid_dim: (n_out.div_ceil(16), m, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        },
        GgmlType::Q5K => LaunchConfig {
            grid_dim: (n_out.div_ceil(8), m, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        },
        _ => {
            let threads = 256u32;
            LaunchConfig {
                grid_dim: (n_out.div_ceil(threads), m, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            }
        }
    }
}

pub fn row_bytes(ty: GgmlType, n_in: usize) -> Option<usize> {
    let be = ty.block_elements()? as usize;
    let bb = ty.block_bytes()? as usize;
    if n_in % be != 0 {
        return None;
    }
    Some(n_in / be * bb)
}

pub fn dq_fmt(ty: GgmlType) -> Option<u32> {
    Some(match ty {
        GgmlType::Q4K => 0,
        GgmlType::Q5K => 1,
        GgmlType::Q6K => 2,
        GgmlType::Q8_0 => 3,
        GgmlType::Q2K => 4,
        GgmlType::Q3K => 5,
        GgmlType::Q5_0 => 6,
        GgmlType::F16 => 7,
        GgmlType::F32 => 8,
        GgmlType::BF16 => 9,
        GgmlType::Other(_) => return None,
    })
}

pub fn resolve_w(gpu: &CudaGpu, w: &[u8]) -> Option<(usize, u64)> {
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some((chunk, (addr - gpu.chunks[chunk].start) as u64))
}

/// Dequant one `token_embd` row into `y_arena[y_off..y_off+n_in]` (no sync).
pub fn launch_embed_row(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    w: &[u8],
    token: u32,
    n_in: usize,
    y_off: usize,
) -> Option<()> {
    let (chunk, base) = resolve_w(gpu, w)?;
    let rb = row_bytes(ty, n_in)?;
    let fmt = dq_fmt(ty)?;
    let w_off = base.checked_add(token as u64 * rb as u64)?;
    // Bounds: token row must lie inside the mapped chunk window.
    let end = w_off.checked_add(rb as u64)?;
    if end > gpu.chunks[chunk].len as u64 {
        return None;
    }
    gpu.ensure_arenas((y_off + n_in) * 4, (y_off + n_in) * 4)?;
    crate::gpu::cuda::ggml::prepare_for_cudarc();
    let f = gpu.func_owned("embed_row_f32")?;
    let n_in_u = n_in as u32;
    let fmt_u = fmt;
    let cfg = if matches!(fmt, 0 | 1 | 2) && n_in % 256 == 0 {
        LaunchConfig {
            grid_dim: (1, n_in_u / 256, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    let stream = Arc::clone(&gpu.stream);
    let wbuf = &gpu.chunks[chunk].buf;
    let (ybase, yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
    drop(yg);
    let out_p = ybase + (y_off * 4) as u64;
    unsafe {
        stream
            .launch_builder_pdl(&f)
            .arg(wbuf)
            .arg(&out_p)
            .arg(&n_in_u)
            .arg(&w_off)
            .arg(&fmt_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

/// Embed from device token id pointer into `y_arena[y_off..]`.
pub fn launch_embed_row_dtoken_ptr(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    w: &[u8],
    n_in: usize,
    y_off: usize,
    d_token_p: u64,
) -> Option<()> {
    let (chunk, base) = resolve_w(gpu, w)?;
    let rb = row_bytes(ty, n_in)?;
    let fmt = dq_fmt(ty)?;
    gpu.ensure_arenas((y_off + n_in) * 4, (y_off + n_in) * 4)?;
    crate::gpu::cuda::ggml::prepare_for_cudarc();
    let f = gpu.func_owned("embed_row_f32_dtoken")?;
    let n_in_u = n_in as u32;
    let fmt_u = fmt;
    let rb_u = rb as u32;
    let w_base = base;
    let cfg = if matches!(fmt, 0 | 1 | 2) && n_in % 256 == 0 {
        LaunchConfig {
            grid_dim: (1, n_in_u / 256, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    let stream = Arc::clone(&gpu.stream);
    let wbuf = &gpu.chunks[chunk].buf;
    let (ybase, yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
    drop(yg);
    let out_p = ybase + (y_off * 4) as u64;
    unsafe {
        stream
            .launch_builder_pdl(&f)
            .arg(wbuf)
            .arg(&out_p)
            .arg(&n_in_u)
            .arg(&w_base)
            .arg(&rb_u)
            .arg(&fmt_u)
            .arg(&d_token_p)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

pub fn launch_inc_u32_ptr(gpu: &mut CudaGpu, p: u64) -> Option<()> {
    let f = gpu.func_owned("inc_u32")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { gpu.stream.launch_builder_pdl(&f).arg(&p).launch_pdl(cfg) }.ok()?;
    Some(())
}

pub fn launch_store_u32_at_ptr(gpu: &mut CudaGpu, dst_p: u64, src_p: u64, idx: u32) -> Option<()> {
    let f = gpu.func_owned("store_u32_at")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&dst_p)
            .arg(&src_p)
            .arg(&idx)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

/// Launch dequant+GEMM into `y_arena[y_off..]`. No synchronize / D2H.
///
/// f16 scratch layout is `[X | W | C]` so Q/K/V (same X, different n_out) can
/// reuse the X cast. `reuse_x_f16` skips that cast when the front of the arena
/// already holds `m * n_in` f16 activations. `x_from_pf_hs` reads X from the
/// prefill residual-norm buffer instead of `x_arena`.
pub fn launch_gemm_dequant(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    m: usize,
    x_already_in_arena: bool,
    x_elems: &[f32],
    y_off: usize,
    reuse_x_f16: bool,
    x_from_pf_hs: bool,
    x_dev_override: Option<u64>,
) -> Option<()> {
    gpu.ensure_arenas(m * n_in * 4, (y_off + m * n_out) * 4)?;
    if !x_already_in_arena {
        gpu.stream
            .memcpy_htod(x_elems, &mut gpu.x_arena.slice_mut(0..m * n_in))
            .ok()?;
    }

    // llama.cpp ggml-cuda MMQ (fast on 5090 vs dequant+cuBLASLt).
    // Opt out with ALLPAKA_GGML_GEMM=0.
    let ggml_gemm = std::env::var("ALLPAKA_GGML_GEMM")
        .map_or(true, |v| !(v == "0" || v.eq_ignore_ascii_case("false")));
    if ggml_gemm && crate::gpu::cuda::ggml::enabled() {
        if let Some(wty) = crate::gpu::cuda::ggml::ggml_type_code(ty) {
            if matches!(ty, GgmlType::Q4K | GgmlType::Q6K) && n_in % 256 == 0 {
                let stream = Arc::clone(&gpu.stream);
                crate::gpu::cuda::ggml::prepare_from_cudarc(&stream);
                let (wp, wg) = DevicePtr::device_ptr(&gpu.chunks[chunk].buf, &stream);
                drop(wg);
                let w_dev = wp + w_off;
                let x_dev = if let Some(p) = x_dev_override {
                    p
                } else if x_from_pf_hs {
                    let (p, g) = DevicePtr::device_ptr(&gpu.pf_hs, &stream);
                    drop(g);
                    p
                } else {
                    let (p, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
                    drop(g);
                    p
                };
                let (ybase, yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
                drop(yg);
                let y_dev = ybase + (y_off * 4) as u64;
                if crate::gpu::cuda::ggml::mul_mat(w_dev, x_dev, y_dev, n_in, n_out, m, wty) {
                    return Some(());
                }
            }
        }
    }
    if x_dev_override.is_some() {
        return None;
    }

    crate::gpu::cuda::ggml::prepare_for_cudarc();

    // Int8 TC MMQ: ALLPAKA_MMQ=1 (NVRTC), =tile (float smem), =nvcc (standalone PTX).
    // Neither beats dequant+cuBLAS yet; all stay opt-in.
    if matches!(ty, GgmlType::Q4K) && n_in % 256 == 0 {
        let mmq = std::env::var("ALLPAKA_MMQ").unwrap_or_default();
        if mmq == "1" || mmq == "tile" || mmq == "nvcc" {
            let stream = Arc::clone(&gpu.stream);
            let use_tile = mmq == "tile";
            let use_nvcc = mmq == "nvcc";
            if !use_tile {
                let x_ptr = if x_from_pf_hs {
                    let (p, g) = DevicePtr::device_ptr(&gpu.pf_hs, &stream);
                    drop(g);
                    p
                } else {
                    let (p, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
                    drop(g);
                    p
                };
                if gpu.q8_src != x_ptr || gpu.q8_src_n != n_in || gpu.q8_src_m != m {
                    launch_quantize_q8(gpu, x_ptr, n_in, m)?;
                }
            }
            let fname = if use_nvcc {
                "mm_q4_k_mmq"
            } else if use_tile {
                "mm_q4_k_tile"
            } else {
                "mm_q4_k_mma"
            };
            let f = gpu.func_owned(fname)?;
            let n_in_u = n_in as u32;
            let n_out_u = n_out as u32;
            let m_u = m as u32;
            let w_off_u = w_off;
            // nvcc: tile 128×64, K-step 256; dynamic smem As..Bw ≈ 52 KiB
            const NVCC_MMQ_SMEM: u32 = 52_224;
            let cfg = if use_nvcc {
                LaunchConfig {
                    grid_dim: (n_out_u.div_ceil(128), m_u.div_ceil(64), 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: NVCC_MMQ_SMEM,
                }
            } else if use_tile {
                LaunchConfig {
                    grid_dim: (n_out_u.div_ceil(64), m_u.div_ceil(32), 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                LaunchConfig {
                    grid_dim: (n_out_u.div_ceil(64), m_u.div_ceil(16), 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 0,
                }
            };
            let (ybase, yg) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(yg);
            let dst_p = ybase + (y_off * 4) as u64;
            let w = &gpu.chunks[chunk].buf;
            if use_tile {
                let xv = if x_from_pf_hs {
                    gpu.pf_hs.slice(0..m * n_in)
                } else {
                    gpu.x_arena.slice(0..m * n_in)
                };
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(w)
                        .arg(&xv)
                        .arg(&dst_p)
                        .arg(&n_in_u)
                        .arg(&n_out_u)
                        .arg(&w_off_u)
                        .arg(&m_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            } else {
                // Opt-in MMQ still expects flat q+d; packed ABI leaves this path stale.
                let q = gpu.q8_q.slice(0..q8_packed_elems(n_in, m));
                let d = gpu.q8_d.slice(0..1);
                unsafe {
                    stream
                        .launch_builder_pdl(&f)
                        .arg(w)
                        .arg(&q)
                        .arg(&d)
                        .arg(&dst_p)
                        .arg(&n_in_u)
                        .arg(&n_out_u)
                        .arg(&w_off_u)
                        .arg(&m_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
            }
            return Some(());
        }
    }

    let rb = row_bytes(ty, n_in)?;
    let fmt = dq_fmt(ty)?;
    let x_elems_n = m * n_in;
    let w_elems = n_out * n_in;
    let c_elems = m * n_out;
    // Ping-pong W on a second stream. Grow the reserved X region / W pitch only
    // after draining both streams so dequant never clobbers an in-flight GEMM.
    if w_elems > gpu.w_pitch || x_elems_n > gpu.f16_x_region {
        let _ = gpu.stream.synchronize();
        let _ = gpu.dq_stream.synchronize();
        gpu.w_pitch = gpu.w_pitch.max(w_elems);
        gpu.f16_x_region = gpu.f16_x_region.max(x_elems_n);
        gpu.w_ping = 0;
        gpu.x_f16_n = 0;
    }
    let x16 = 0usize;
    let ping = gpu.w_ping;
    let w16 = gpu.f16_x_region + ping * gpu.w_pitch;
    let c16 = gpu.f16_x_region + 2 * gpu.w_pitch;
    gpu.ensure_f16(c16 + c_elems)?;
    {
        let f = gpu.func_owned("dequant_rows_f16")?;
        let n_in_u = n_in as u32;
        let n_out_u = n_out as u32;
        let fmt_u = fmt;
        let rb_u = rb as u32;
        let w_off_u = w_off;
        const ROWS: u32 = 16;
        let k_blocks = matches!(fmt, 0 | 1 | 2) && n_in % 256 == 0;
        let cfg = if k_blocks {
            LaunchConfig {
                grid_dim: (n_out_u.div_ceil(ROWS), n_in_u / 256, 1),
                block_dim: (32, ROWS, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (n_out_u.div_ceil(ROWS), 1, 1),
                block_dim: (32, ROWS, 1),
                shared_mem_bytes: 0,
            }
        };
        let w = &gpu.chunks[chunk].buf;
        let mut out = gpu.f16_arena.slice_mut(w16..w16 + w_elems);
        unsafe {
            gpu.dq_stream
                .launch_builder_pdl(&f)
                .arg(w)
                .arg(&mut out)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&rb_u)
                .arg(&fmt_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        let ev = if ping == 0 { &gpu.ev_w0 } else { &gpu.ev_w1 };
        ev.record(&gpu.dq_stream).ok()?;
        gpu.w_ping = 1 - ping;
    }
    if !(reuse_x_f16 && gpu.x_f16_n == x_elems_n) {
        let f = gpu.func_owned("cast_f32_to_f16")?;
        let n = x_elems_n as u32;
        let cfg = CudaGpu::cfg_1d(n, 256);
        let mut dst = gpu.f16_arena.slice_mut(x16..x16 + x_elems_n);
        if x_from_pf_hs {
            let src = gpu.pf_hs.slice(0..x_elems_n);
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&n)
                    .launch_pdl(cfg)
            }
            .ok()?;
        } else {
            let src = gpu.x_arena.slice(0..x_elems_n);
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&n)
                    .launch_pdl(cfg)
            }
            .ok()?;
        }
        gpu.x_f16_n = x_elems_n;
    }
    {
        let ev = if ping == 0 { &gpu.ev_w0 } else { &gpu.ev_w1 };
        gpu.stream.wait(ev).ok()?;
    }
    let mut used_blas = false;
    if gpu.blaslt.is_some() {
        let (head, mut c16s) = gpu.f16_arena.split_at_mut(c16);
        let a = head.slice(x16..x16 + m * n_in);
        let b = head.slice(w16..w16 + w_elems);
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: n_out as u64,
            n: m as u64,
            k: n_in as u64,
            alpha: 1.0,
            lda: n_in as i64,
            ldb: n_in as i64,
            beta: 0.0,
            ldc: n_out as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        let blas_ok = {
            let blas = gpu.blaslt.as_ref().unwrap();
            unsafe { Matmul::matmul(blas, cfg, &b, &a, &mut c16s, None, None) }.is_ok()
        };
        if blas_ok {
            let f = gpu.func_owned("cast_f16_to_f32")?;
            let n = (m * n_out) as u32;
            let lcfg = CudaGpu::cfg_1d(n, 256);
            let stream = Arc::clone(&gpu.stream);
            let (ybase, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
            drop(g);
            let dst_p = ybase + (y_off * 4) as u64;
            let src = gpu.f16_arena.slice(c16..c16 + m * n_out);
            unsafe {
                stream
                    .launch_builder_pdl(&f)
                    .arg(&dst_p)
                    .arg(&src)
                    .arg(&n)
                    .launch_pdl(lcfg)
            }
            .ok()?;
            used_blas = true;
        }
    }
    if !used_blas {
        let f = gpu.func_owned("gemm_f16_f32")?;
        let m_u = m as u32;
        let n_u = n_out as u32;
        let k_u = n_in as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u.div_ceil(16), m_u.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let stream = Arc::clone(&gpu.stream);
        let (ybase, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
        drop(g);
        let dst_p = ybase + (y_off * 4) as u64;
        let a = gpu.f16_arena.slice(x16..x16 + m * n_in);
        let b = gpu.f16_arena.slice(w16..w16 + w_elems);
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&a)
                .arg(&b)
                .arg(&dst_p)
                .arg(&m_u)
                .arg(&n_u)
                .arg(&k_u)
                .launch_pdl(cfg)
        }
        .ok()?;
    }
    crate::gpu::cuda::ggml::mark_cudarc_dirty();
    Some(())
}

/// Dequant weight matrix to f16 then GEMM: y[m,n_out] = x[m,n_in] @ W^T.
pub fn gemm_dequant(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    m: usize,
    x_elems: &[f32],
) -> Option<Vec<f32>> {
    let t0 = Instant::now();
    launch_gemm_dequant(
        gpu, ty, chunk, w_off, n_in, n_out, m, false, x_elems, 0, false, false, None,
    )?;
    let encode_ns = t0.elapsed().as_nanos() as u64;
    let t1 = Instant::now();
    gpu.sync()?;
    note_call(2, encode_ns, t1.elapsed().as_nanos() as u64);
    gpu.stream.clone_dtoh(&gpu.y_arena.slice(0..m * n_out)).ok()
}

pub(crate) fn q8_decode_enabled() -> bool {
    match std::env::var("ALLPAKA_Q8") {
        // Default on: Q4_K+Q6_K dp4a mmvq. Set ALLPAKA_Q8=0 to force float Metal.
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn quantize_q8_cfg(n: u32, rows: u32) -> LaunchConfig {
    let nblk = n / 32;
    let (grid_x, block_y) = if n >= 256 {
        ((nblk + 7) / 8, 8u32)
    } else {
        (nblk, 1u32)
    };
    LaunchConfig {
        grid_dim: (grid_x, rows, 1),
        block_dim: (32, block_y, 1),
        shared_mem_bytes: 0,
    }
}

/// Packed Q8 byte count as i8 elems: (n/32)*44*rows.
#[inline]
fn q8_packed_elems(n: usize, rows: usize) -> usize {
    (n / 32) * 44 * rows
}

fn launch_quantize_q8_kernel(
    gpu: &mut CudaGpu,
    kernel: &str,
    x_ptr: u64,
    n: usize,
    rows: usize,
) -> Option<()> {
    if n % 32 != 0 {
        return None;
    }
    gpu.ensure_q8(n, rows)?;
    let n_u = n as u32;
    let rows_u = rows as u32;
    let f = gpu.func_owned(kernel)?;
    let cfg = quantize_q8_cfg(n_u, rows_u);
    let mut q = gpu.q8_q.slice_mut(0..q8_packed_elems(n, rows));
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&x_ptr)
            .arg(&mut q)
            .arg(&n_u)
            .arg(&rows_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    gpu.q8_src = x_ptr;
    gpu.q8_src_n = n;
    gpu.q8_src_m = rows;
    gpu.q8_off = 0;
    Some(())
}

pub fn launch_quantize_q8(gpu: &mut CudaGpu, x_ptr: u64, n: usize, rows: usize) -> Option<()> {
    launch_quantize_q8_kernel(gpu, "quantize_q8_1", x_ptr, n, rows)
}

pub fn launch_quantize_q8_q6(gpu: &mut CudaGpu, x_ptr: u64, n: usize, rows: usize) -> Option<()> {
    launch_quantize_q8_kernel(gpu, "quantize_q8_1_q6", x_ptr, n, rows)
}

/// Decode: FA [hd,n_q] -> out [n_q,hd] + pack Q8 for o_proj (one kernel).
pub fn launch_permute_q8_m1(
    gpu: &mut CudaGpu,
    fa_src: u64,
    out_ptr: u64,
    hd: usize,
    n_q: usize,
) -> Option<()> {
    let n = hd * n_q;
    if n % 32 != 0 {
        return None;
    }
    gpu.ensure_q8(n, 1)?;
    crate::gpu::cuda::ggml::prepare_for_cudarc();
    let hd_u = hd as u32;
    let nq_u = n_q as u32;
    let f = gpu.func_owned("permute_q8_m1")?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32) / 32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut q = gpu.q8_q.slice_mut(0..q8_packed_elems(n, 1));
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&fa_src)
            .arg(&out_ptr)
            .arg(&mut q)
            .arg(&hd_u)
            .arg(&nq_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    gpu.q8_src = out_ptr;
    gpu.q8_src_n = n;
    gpu.q8_src_m = 1;
    gpu.q8_off = 0;
    Some(())
}

pub fn launch_permute_q8_m1_qonly(
    gpu: &mut CudaGpu,
    fa_src: u64,
    out_identity: u64,
    hd: usize,
    n_q: usize,
) -> Option<()> {
    let n = hd * n_q;
    if n % 32 != 0 {
        return None;
    }
    gpu.ensure_q8(n, 1)?;
    crate::gpu::cuda::ggml::prepare_for_cudarc();
    let hd_u = hd as u32;
    let nq_u = n_q as u32;
    let f = gpu.func_owned("permute_q8_m1_qonly")?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32) / 32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut q = gpu.q8_q.slice_mut(0..q8_packed_elems(n, 1));
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(&fa_src)
            .arg(&mut q)
            .arg(&hd_u)
            .arg(&nq_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    gpu.q8_src = out_identity;
    gpu.q8_src_n = n;
    gpu.q8_src_m = 1;
    gpu.q8_off = 0;
    Some(())
}

fn launch_qk_q8_matvec(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    x_ptr: u64,
    y_ptr: u64,
    m: usize,
    add: bool,
) -> Option<()> {
    let stream = Arc::clone(&gpu.stream);
    launch_qk_q8_matvec_on(
        gpu, &stream, ty, chunk, w_off, n_in, n_out, x_ptr, y_ptr, m, add,
    )
}

fn l2_persist_wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ALLPAKA_L2_PERSIST")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Pin the Q8 activation buffer in the persisting L2 slice so thousands of
/// MMVQ CTAs re-hit X instead of thrashing it with weight traffic.
fn apply_q8_l2_persist(gpu: &mut CudaGpu, stream: &CudaStream) {
    if !l2_persist_wanted() {
        return;
    }
    let nbytes = gpu.q8_q_cap;
    if nbytes == 0 || gpu.q8_l2_bytes == nbytes {
        return;
    }
    use cudarc::driver::sys::{
        cuCtxSetLimit, cuStreamSetAttribute, CUaccessPolicyWindow, CUaccessProperty, CUlimit,
        CUstreamAttrID, CUstreamAttrValue,
    };
    static LIMIT_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LIMIT_LOGGED.swap(true, Ordering::Relaxed) {
        // Keep a small persisting carve-out (Q8 X is tens of KiB).
        let rc = unsafe { cuCtxSetLimit(CUlimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 2 << 20) };
        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("cuda: L2 persist for Q8 X (ALLPAKA_L2_PERSIST=1)");
        } else {
            eprintln!("cuda: L2 persist limit failed ({rc:?}); continuing without");
            return;
        }
    }
    let (base, g) = DevicePtr::device_ptr(&gpu.q8_q, stream);
    drop(g);
    let win = CUaccessPolicyWindow {
        base_ptr: base as usize as *mut std::ffi::c_void,
        num_bytes: nbytes,
        hitRatio: 1.0,
        hitProp: CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING,
        missProp: CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
    };
    let mut val = unsafe { std::mem::zeroed::<CUstreamAttrValue>() };
    unsafe {
        val.accessPolicyWindow = win;
        let rc = cuStreamSetAttribute(
            stream.cu_stream(),
            CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &val,
        );
        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            gpu.q8_l2_bytes = nbytes;
        }
    }
}

fn l2_stream_w_wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ALLPAKA_L2_STREAM_W")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Hint weight traffic as STREAMING so it prefers not to pin L2 (leaves more
/// for the reused Q8 activation). Mutually exclusive with Q8 persist window.
fn apply_w_l2_stream(gpu: &CudaGpu, stream: &CudaStream, chunk: usize) {
    if !l2_stream_w_wanted() || l2_persist_wanted() {
        return;
    }
    use cudarc::driver::sys::{
        cuStreamSetAttribute, CUaccessPolicyWindow, CUaccessProperty, CUstreamAttrID,
        CUstreamAttrValue,
    };
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("cuda: L2 STREAMING hint on weights (ALLPAKA_L2_STREAM_W=1)");
    }
    let (base, g) = DevicePtr::device_ptr(&gpu.chunks[chunk].buf, stream);
    drop(g);
    let nbytes = gpu.chunks[chunk].len;
    if nbytes == 0 {
        return;
    }
    let win = CUaccessPolicyWindow {
        base_ptr: base as usize as *mut std::ffi::c_void,
        num_bytes: nbytes,
        hitRatio: 1.0,
        hitProp: CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
        missProp: CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
    };
    let mut val = unsafe { std::mem::zeroed::<CUstreamAttrValue>() };
    unsafe {
        val.accessPolicyWindow = win;
        let _ = cuStreamSetAttribute(
            stream.cu_stream(),
            CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &val,
        );
    }
}

fn launch_qk_q8_matvec_on(
    gpu: &mut CudaGpu,
    stream: &CudaStream,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    x_ptr: u64,
    y_ptr: u64,
    m: usize,
    add: bool,
) -> Option<()> {
    if n_in % 256 != 0 {
        return None;
    }
    // Prefer nwarps=8 for long-K Q6 (down-proj). Q4 stays at 4.
    // Huge n_out (LM head ~152k) prefers nwarps=4: more CTAs on short K.
    let long_q6 = matches!(ty, GgmlType::Q6K) && n_in >= 4096;
    let huge_out = n_out >= 65536;
    let nwarps_env = std::env::var("ALLPAKA_NWARPS").ok();
    let force2 = nwarps_env.as_deref() == Some("2");
    let force8 = nwarps_env.as_deref() == Some("8");
    let force4 = nwarps_env.as_deref() == Some("4");
    let rows2 = std::env::var("ALLPAKA_MMVQ_ROWS")
        .map(|v| v == "2")
        .unwrap_or(false);
    let out4 = huge_out
        && std::env::var("ALLPAKA_OUT_N4")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true); // nwarps=4 on LM head (~152k rows) beats n8 on 5090
    let auto8 = long_q6 && !force2 && !force4 && !rows2 && !(out4 && !force8);
    // Q6 + f32 X: pack Q8 in smem inside the matvec (skip separate quantize_q8_1).
    // Opt-in only: measured slower than quantize+q8 on 5090 (~43 vs ~64 tg).
    let q6_f32 = matches!(ty, GgmlType::Q6K)
        && auto8
        && gpu.fns.contains_key("matvec_q6_k_f32_n8")
        && std::env::var("ALLPAKA_Q6_F32")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    // FA→WO: consume pending FA f32 ptr with fused pack-in-mmvq (skips permute_q8).
    let wo_fa = matches!(ty, GgmlType::Q4K)
        && m == 1
        && n_in == 8192
        && gpu.wo_fa_x != 0
        && gpu.fns.contains_key("matvec_q4_k_f32_k8192");
    if wo_fa {
        let x_fa = gpu.wo_fa_x;
        gpu.wo_fa_x = 0;
        let f = gpu.func_owned("matvec_q4_k_f32_k8192")?;
        let n_u = n_in as u32;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let w_off_u = w_off;
        let add_u = add as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_out_u, m_u, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let wb = &gpu.chunks[chunk].buf;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(&x_fa)
                .arg(&y_ptr)
                .arg(&n_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .arg(&add_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        gpu.invalidate_q8();
        return Some(());
    }
    if q6_f32 {
        let f = gpu.func_owned("matvec_q6_k_f32_n8")?;
        let n_u = n_in as u32;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let w_off_u = w_off;
        let add_u = add as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_out_u, m_u, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let wb = &gpu.chunks[chunk].buf;
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(&x_ptr)
                .arg(&y_ptr)
                .arg(&n_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .arg(&add_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        // X was consumed as f32; invalidate any cached Q8 of this pointer.
        if gpu.q8_src == x_ptr {
            gpu.invalidate_q8();
        }
        return Some(());
    }
    let (kern, nwarps, grid_x) = match (ty, force8 || auto8, force2, rows2) {
        (GgmlType::Q4K, false, false, true) => {
            ("matvec_q4_k_q8_r2", 4u32, (n_out as u32).div_ceil(2))
        }
        (GgmlType::Q4K, true, _, _) => ("matvec_q4_k_q8_n8", 8u32, n_out as u32),
        (GgmlType::Q4K, false, true, _) => ("matvec_q4_k_q8_n2", 2u32, n_out as u32),
        (GgmlType::Q4K, false, false, false)
            if n_in == 8192
                && std::env::var("ALLPAKA_K8192")
                    .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                    .unwrap_or(true) =>
        {
            ("matvec_q4_k_q8_k8192", 4u32, n_out as u32)
        }
        // Opt-in: measured ~61 tg vs ~70.6 generic on Qwen3-32B / 5090.
        (GgmlType::Q4K, false, false, false)
            if n_in == 5120
                && gpu.fns.contains_key("matvec_q4_k_q8_k5120")
                && std::env::var("ALLPAKA_K5120")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false) =>
        {
            ("matvec_q4_k_q8_k5120", 4u32, n_out as u32)
        }
        (GgmlType::Q4K, false, false, false) => ("matvec_q4_k_q8", 4u32, n_out as u32),
        // Opt-in: soft-pipe unroll regressed ~58 tg vs ~70.6 n8 on 5090.
        (GgmlType::Q6K, true, _, _)
            if n_in == 25600
                && gpu.fns.contains_key("matvec_q6_k_q8_n8_k25600")
                && std::env::var("ALLPAKA_Q6_K25600")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false) =>
        {
            ("matvec_q6_k_q8_n8_k25600", 8u32, n_out as u32)
        }
        (GgmlType::Q6K, true, _, _) => ("matvec_q6_k_q8_n8", 8u32, n_out as u32),
        (GgmlType::Q6K, _, _, _) => ("matvec_q6_k_q8", 4u32, n_out as u32),
        _ => return None,
    };
    let q_elems = q8_packed_elems(n_in, m);
    gpu.ensure_q8_bytes(gpu.q8_off + q_elems)?;
    apply_q8_l2_persist(gpu, stream);
    apply_w_l2_stream(gpu, stream, chunk);
    let n_u = n_in as u32;
    let rows_u = m as u32;
    let need_q = gpu.q8_src != x_ptr || gpu.q8_src_n != n_in || gpu.q8_src_m != m;
    if need_q {
        let f = gpu.func_owned("quantize_q8_1")?;
        let cfg = quantize_q8_cfg(n_u, rows_u);
        let mut q = gpu.q8_q.slice_mut(0..q_elems);
        unsafe {
            stream
                .launch_builder_pdl(&f)
                .arg(&x_ptr)
                .arg(&mut q)
                .arg(&n_u)
                .arg(&rows_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        gpu.q8_src = x_ptr;
        gpu.q8_src_n = n_in;
        gpu.q8_src_m = m;
        gpu.q8_off = 0;
    }
    let f = gpu.func_owned(kern)?;
    let n_out_u = n_out as u32;
    let m_u = m as u32;
    let w_off_u = w_off;
    let add_u = add as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, m_u, 1),
        block_dim: (32, nwarps, 1),
        shared_mem_bytes: 0,
    };
    let wb = &gpu.chunks[chunk].buf;
    let q_off = gpu.q8_off;
    let q = gpu.q8_q.slice(q_off..q_off + q_elems);
    unsafe {
        stream
            .launch_builder_pdl(&f)
            .arg(wb)
            .arg(&q)
            .arg(&y_ptr)
            .arg(&n_u)
            .arg(&n_out_u)
            .arg(&w_off_u)
            .arg(&m_u)
            .arg(&add_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

pub fn run_matvec(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    w: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    m: usize,
) -> Option<Vec<f32>> {
    if x.len() != m * n_in {
        return None;
    }
    let (chunk, w_off) = resolve_w(gpu, w)?;
    let _ = matvec_kernel(ty)?;

    if m >= MM_MIN_M && dq_fmt(ty).is_some() {
        return gemm_dequant(gpu, ty, chunk, w_off, n_in, n_out, m, x);
    }

    gpu.ensure_arenas(m * n_in * 4, m * n_out * 4)?;
    gpu.stream
        .memcpy_htod(x, &mut gpu.x_arena.slice_mut(0..m * n_in))
        .ok()?;

    let t0 = Instant::now();
    if matches!(ty, GgmlType::Q4K | GgmlType::Q6K) && n_in % 256 == 0 && q8_decode_enabled() {
        let stream = Arc::clone(&gpu.stream);
        let (xbase, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
        drop(g);
        let (ybase, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
        drop(g);
        launch_qk_q8_matvec(gpu, ty, chunk, w_off, n_in, n_out, xbase, ybase, m, false)?;
    } else {
        let name = matvec_kernel(ty)?;
        let f = gpu.func_owned(name)?;
        let n_in_u = n_in as u32;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let w_off_u = w_off;
        let add_u = 0u32;
        let cfg = matvec_launch_cfg(ty, n_out_u, m_u);
        let wb = &gpu.chunks[chunk].buf;
        let xv = gpu.x_arena.slice(0..m * n_in);
        let mut yv = gpu.y_arena.slice_mut(0..m * n_out);
        let add_k = matches!(ty, GgmlType::Q4K | GgmlType::Q6K);
        unsafe {
            if add_k {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(wb)
                    .arg(&xv)
                    .arg(&mut yv)
                    .arg(&n_in_u)
                    .arg(&n_out_u)
                    .arg(&w_off_u)
                    .arg(&m_u)
                    .arg(&add_u)
                    .launch_pdl(cfg)
            } else {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(wb)
                    .arg(&xv)
                    .arg(&mut yv)
                    .arg(&n_in_u)
                    .arg(&n_out_u)
                    .arg(&w_off_u)
                    .arg(&m_u)
                    .launch_pdl(cfg)
            }
        }
        .ok()?;
    }
    let encode_ns = t0.elapsed().as_nanos() as u64;
    let t1 = Instant::now();
    gpu.sync()?;
    note_call(1, encode_ns, t1.elapsed().as_nanos() as u64);
    gpu.stream.clone_dtoh(&gpu.y_arena.slice(0..m * n_out)).ok()
}

/// Device-resident matvec: `y_arena[y_off..]` ← W · `y_arena[x_off..]`.
/// No H2D / D2H / synchronize — caller owns stream ordering.
pub fn launch_matvec_y(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    x_off: usize,
    y_off: usize,
    m: usize,
) -> Option<()> {
    let stream = Arc::clone(&gpu.stream);
    let base = {
        let (p, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
        drop(g);
        p
    };
    launch_matvec_y_ptr(
        gpu, ty, chunk, w_off, n_in, n_out, x_off, y_off, m, base, false,
    )
}

/// Like [`launch_matvec_y`] but uses a pre-resolved `y_arena` device base so
/// CUDA graph capture can avoid per-launch `DevicePtr` event recording.
pub fn launch_matvec_y_ptr(
    gpu: &mut CudaGpu,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    x_off: usize,
    y_off: usize,
    m: usize,
    y_base: u64,
    add: bool,
) -> Option<()> {
    let stream = Arc::clone(&gpu.stream);
    launch_matvec_y_ptr_on(
        gpu, &stream, ty, chunk, w_off, n_in, n_out, x_off, y_off, m, y_base, add,
    )
}

/// Device-resident matvec on an explicit stream (for Q/K/V overlap).
pub fn launch_matvec_y_ptr_on(
    gpu: &mut CudaGpu,
    stream: &CudaStream,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
    n_in: usize,
    n_out: usize,
    x_off: usize,
    y_off: usize,
    m: usize,
    y_base: u64,
    add: bool,
) -> Option<()> {
    let xv = y_base + (x_off * std::mem::size_of::<f32>()) as u64;
    let yv = y_base + (y_off * std::mem::size_of::<f32>()) as u64;

    // Prefill-sized batches use ggml MMQ via launch_gemm_dequant.
    // ALLPAKA_GGML_DECODE=1: llama MMVQ for non-add (legacy; disables fusion in decode).
    // ALLPAKA_GGML_MMVQ=1: llama MMVQ for all m=1 including add, keep our QKV/gate fusion.
    let ggml_mmvq = crate::gpu::cuda::ggml::enabled()
        && (std::env::var("ALLPAKA_GGML_MMVQ")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            || (std::env::var("ALLPAKA_GGML_DECODE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
                && !add));
    if m == 1 && ggml_mmvq && matches!(ty, GgmlType::Q4K | GgmlType::Q6K) && n_in % 256 == 0 {
        if let Some(wty) = crate::gpu::cuda::ggml::ggml_type_code(ty) {
            // Prefill clears shared stream; force rebind so MMVQ and residual share one queue.
            if !crate::gpu::cuda::ggml::shared_stream() {
                crate::gpu::cuda::ggml::bind_peer_stream(stream);
            }
            crate::gpu::cuda::ggml::prepare_from_cudarc(stream);
            let (wp, wg) = DevicePtr::device_ptr(&gpu.chunks[chunk].buf, stream);
            drop(wg);
            let w_dev = wp + w_off;
            if add {
                gpu.ensure_w_scratch(n_out)?;
                let (sp, sg) = DevicePtr::device_ptr(&gpu.w_scratch, stream);
                drop(sg);
                if crate::gpu::cuda::ggml::mul_mat_vec(w_dev, xv, sp, n_in, n_out, wty) {
                    // Events path: drain ggml before residual_add reads scratch.
                    crate::gpu::cuda::ggml::prepare_for_cudarc();
                    let f = gpu.func_owned("residual_add")?;
                    let n_u = n_out as u32;
                    let cfg = CudaGpu::cfg_1d(n_u, 256);
                    unsafe {
                        stream
                            .launch_builder_pdl(&f)
                            .arg(&yv)
                            .arg(&sp)
                            .arg(&n_u)
                            .launch_pdl(cfg)
                    }
                    .ok()?;
                    gpu.invalidate_q8();
                    return Some(());
                }
            } else if crate::gpu::cuda::ggml::mul_mat_vec(w_dev, xv, yv, n_in, n_out, wty) {
                crate::gpu::cuda::ggml::prepare_for_cudarc();
                gpu.invalidate_q8();
                return Some(());
            }
        }
    }

    if matches!(ty, GgmlType::Q4K | GgmlType::Q6K) && n_in % 256 == 0 && q8_decode_enabled() {
        let r = launch_qk_q8_matvec_on(gpu, stream, ty, chunk, w_off, n_in, n_out, xv, yv, m, add);
        return r;
    }
    let name = matvec_kernel(ty)?;
    let f = gpu.func_owned(name)?;
    let n_in_u = n_in as u32;
    let n_out_u = n_out as u32;
    let m_u = m as u32;
    let w_off_u = w_off;
    let add_u = add as u32;
    let cfg = matvec_launch_cfg(ty, n_out_u, m_u);
    let wb = &gpu.chunks[chunk].buf;
    let add_k = matches!(ty, GgmlType::Q4K | GgmlType::Q6K);
    unsafe {
        if add_k {
            stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(&xv)
                .arg(&yv)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .arg(&add_u)
                .launch_pdl(cfg)
        } else {
            stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(&xv)
                .arg(&yv)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .launch_pdl(cfg)
        }
    }
    .ok()?;
    Some(())
}

/// Fused gate+up Q4_K matvecs (identical shapes). Returns false if types differ.
pub fn launch_matvec_q4k_2(
    gpu: &mut CudaGpu,
    chunk0: usize,
    chunk1: usize,
    w_off0: u64,
    w_off1: u64,
    n_in: usize,
    n_out: usize,
    x_off: usize,
    y0_off: usize,
    y1_off: usize,
    m: usize,
    y_base: u64,
) -> Option<()> {
    if chunk0 != chunk1 || n_in % 256 != 0 {
        return None;
    }
    // Q8 path: one input quantize (via cache) + fused dual mmvq.
    if q8_decode_enabled() {
        let xv = y_base + (x_off * 4) as u64;
        gpu.ensure_q8(n_in, m)?;
        let n_u = n_in as u32;
        let rows_u = m as u32;
        let need_q =
            gpu.q8_src != xv || gpu.q8_src_n != n_in || gpu.q8_src_m != m || gpu.q8_off != 0;
        if need_q {
            let f = gpu.func_owned("quantize_q8_1")?;
            let cfg = quantize_q8_cfg(n_u, rows_u);
            let mut q = gpu.q8_q.slice_mut(0..q8_packed_elems(n_in, m));
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&xv)
                    .arg(&mut q)
                    .arg(&n_u)
                    .arg(&rows_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            gpu.q8_src = xv;
            gpu.q8_src_n = n_in;
            gpu.q8_src_m = m;
            gpu.q8_off = 0;
        }
        // Opt-in SwiGLU→Q8 epilogue (ALLPAKA_GU_Q8=1). Default keeps parallel
        // matvec_q4_k_q8_2 + separate quantize (faster on 5090 than serial-32 pack).
        if n_out % 32 == 0
            && gpu.fns.contains_key("matvec_q4_k_q8_2_q8")
            && std::env::var("ALLPAKA_GU_Q8")
                .map(|v| v == "1")
                .unwrap_or(false)
        {
            let pack_in = q8_packed_elems(n_in, m);
            let pack_out = q8_packed_elems(n_out, m);
            gpu.ensure_q8_bytes(pack_in + pack_out)?;
            // ensure may have invalidated; re-quantize into slot 0 if needed.
            if gpu.q8_src != xv || gpu.q8_src_n != n_in || gpu.q8_off != 0 {
                let f = gpu.func_owned("quantize_q8_1")?;
                let cfg = quantize_q8_cfg(n_u, rows_u);
                let mut q = gpu.q8_q.slice_mut(0..pack_in);
                unsafe {
                    gpu.stream
                        .launch_builder_pdl(&f)
                        .arg(&xv)
                        .arg(&mut q)
                        .arg(&n_u)
                        .arg(&rows_u)
                        .launch_pdl(cfg)
                }
                .ok()?;
                gpu.q8_src = xv;
                gpu.q8_src_n = n_in;
                gpu.q8_src_m = m;
                gpu.q8_off = 0;
            }
            let f = gpu.func_owned("matvec_q4_k_q8_2_q8")?;
            let n_out_u = n_out as u32;
            let m_u = m as u32;
            let cfg = LaunchConfig {
                grid_dim: (n_out_u / 32, m_u, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            };
            let wb = &gpu.chunks[chunk0].buf;
            let y0 = y_base + (y0_off * 4) as u64;
            let (q8_base, _) = DevicePtr::device_ptr(&gpu.q8_q, &gpu.stream);
            let q_ptr = q8_base;
            let yq_ptr = q8_base + pack_in as u64;
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(wb)
                    .arg(wb)
                    .arg(&q_ptr)
                    .arg(&y0)
                    .arg(&yq_ptr)
                    .arg(&n_u)
                    .arg(&n_out_u)
                    .arg(&w_off0)
                    .arg(&w_off1)
                    .arg(&m_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            gpu.q8_src = y0;
            gpu.q8_src_n = n_out;
            gpu.q8_src_m = m;
            gpu.q8_off = pack_in;
            return Some(());
        }
        // nwarps=4 wins for gate+up on 5090; n8 dual kernel kept for A/B (ALLPAKA_GU_N8=1).
        let use_n8 = n_in >= 4096
            && gpu.fns.contains_key("matvec_q4_k_q8_2_n8")
            && std::env::var("ALLPAKA_GU_N8")
                .map(|v| v == "1")
                .unwrap_or(false);
        // Opt-in only; same regression as single-row k5120 on 5090.
        let use_k5120 = !use_n8
            && n_in == 5120
            && gpu.fns.contains_key("matvec_q4_k_q8_2_k5120")
            && std::env::var("ALLPAKA_K5120")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        let f = gpu.func_owned(if use_n8 {
            "matvec_q4_k_q8_2_n8"
        } else if use_k5120 {
            "matvec_q4_k_q8_2_k5120"
        } else {
            "matvec_q4_k_q8_2"
        })?;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let nwarps = if use_n8 { 8u32 } else { 4u32 };
        let cfg = LaunchConfig {
            grid_dim: (n_out_u, m_u, 1),
            block_dim: (32, nwarps, 1),
            shared_mem_bytes: 0,
        };
        let wb = &gpu.chunks[chunk0].buf;
        let q = gpu.q8_q.slice(0..q8_packed_elems(n_in, m));
        let y0 = y_base + (y0_off * 4) as u64;
        let y1 = y_base + (y1_off * 4) as u64;
        unsafe {
            gpu.stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(wb)
                .arg(&q)
                .arg(&y0)
                .arg(&y1)
                .arg(&n_u)
                .arg(&n_out_u)
                .arg(&w_off0)
                .arg(&w_off1)
                .arg(&m_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        return Some(());
    }
    let f = gpu.func_owned("matvec_q4_k_2")?;
    let n_in_u = n_in as u32;
    let n_out_u = n_out as u32;
    let m_u = m as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_out_u.div_ceil(16), m_u, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };
    let wb = &gpu.chunks[chunk0].buf;
    let xv = y_base + (x_off * 4) as u64;
    let y0 = y_base + (y0_off * 4) as u64;
    let y1 = y_base + (y1_off * 4) as u64;
    unsafe {
        gpu.stream
            .launch_builder_pdl(&f)
            .arg(wb)
            .arg(wb)
            .arg(&xv)
            .arg(&y0)
            .arg(&y1)
            .arg(&n_in_u)
            .arg(&n_out_u)
            .arg(&w_off0)
            .arg(&w_off1)
            .arg(&m_u)
            .launch_pdl(cfg)
    }
    .ok()?;
    Some(())
}

/// Fused Q/K/V matvecs. Supports Q4/Q4/Q4 or Q4_K_M's Q4/Q4/Q6.
/// When `v_cache` is set (decode m=1), V is also written as f16 into the KV cache.
pub fn launch_matvec_q4k_qkv(
    gpu: &mut CudaGpu,
    chunk_q: usize,
    chunk_k: usize,
    chunk_v: usize,
    oq: u64,
    ok: u64,
    ov: u64,
    n_in: usize,
    n_q: usize,
    n_kv: usize,
    x_off: usize,
    q_off: usize,
    k_off: usize,
    v_off: usize,
    m: usize,
    y_base: u64,
    v_ty: GgmlType,
    v_cache: Option<(&cudarc::driver::CudaSlice<u8>, u64, u32)>,
) -> Option<()> {
    if chunk_q != chunk_k || chunk_q != chunk_v || n_in % 256 != 0 {
        if std::env::var("ALLPAKA_FUSE_LOG")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            eprintln!("cuda: qkv fuse skip chunks=({chunk_q},{chunk_k},{chunk_v}) n_in={n_in}");
        }
        return None;
    }
    let kern = match v_ty {
        GgmlType::Q4K => "matvec_q4_k_q8_qkv",
        GgmlType::Q6K => "matvec_q4_q4_q6_q8_qkv",
        _ => return None,
    };
    let xv = y_base + (x_off * 4) as u64;
    let q = y_base + (q_off * 4) as u64;
    let k = y_base + (k_off * 4) as u64;
    let v = y_base + (v_off * 4) as u64;
    let n_in_u = n_in as u32;
    let n_q_u = n_q as u32;
    let n_kv_u = n_kv as u32;
    let m_u = m as u32;

    if !q8_decode_enabled() {
        return None;
    }
    gpu.ensure_q8(n_in, m)?;
    let need_q = gpu.q8_src != xv || gpu.q8_src_n != n_in || gpu.q8_src_m != m || gpu.q8_off != 0;
    if need_q {
        let f = gpu.func_owned("quantize_q8_1")?;
        let rows_u = m as u32;
        let cfg = quantize_q8_cfg(n_in_u, rows_u);
        let mut qq = gpu.q8_q.slice_mut(0..q8_packed_elems(n_in, m));
        unsafe {
            gpu.stream
                .launch_builder_pdl(&f)
                .arg(&xv)
                .arg(&mut qq)
                .arg(&n_in_u)
                .arg(&rows_u)
                .launch_pdl(cfg)
        }
        .ok()?;
        gpu.q8_src = xv;
        gpu.q8_src_n = n_in;
        gpu.q8_src_m = m;
        gpu.q8_off = 0;
    }
    let f = gpu.func_owned(kern)?;
    // GQA split (opt-in): fat Q4/Q4/Q6 kernel only for n_kv rows; thin Q4 for Q tail.
    // Decode m=1 only — thin matvec stride must match contiguous Q[n_kv..).
    let split = matches!(v_ty, GgmlType::Q6K)
        && m == 1
        && n_q > n_kv
        && std::env::var("ALLPAKA_QKV_SPLIT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let grid_x = if split { n_kv_u } else { n_q_u.max(n_kv_u) };
    let cfg = LaunchConfig {
        grid_dim: (grid_x, m_u, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };
    let wb = &gpu.chunks[chunk_q].buf;
    let q_elems = q8_packed_elems(n_in, m);
    let q8_off = gpu.q8_off;
    let qq = gpu.q8_q.slice(q8_off..q8_off + q_elems);
    let null_u64 = 0u64;
    let kv_dim_u = v_cache.map(|(_, _, d)| d).unwrap_or(0);
    let v_cache_off = v_cache.map(|(_, o, _)| o).unwrap_or(0);
    unsafe {
        match v_cache {
            Some((cache_buf, _, _)) => gpu
                .stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(wb)
                .arg(wb)
                .arg(&qq)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&n_in_u)
                .arg(&n_q_u)
                .arg(&n_kv_u)
                .arg(&oq)
                .arg(&ok)
                .arg(&ov)
                .arg(&m_u)
                .arg(cache_buf)
                .arg(&v_cache_off)
                .arg(&gpu.d_pos)
                .arg(&kv_dim_u)
                .launch_pdl(cfg),
            None => gpu
                .stream
                .launch_builder_pdl(&f)
                .arg(wb)
                .arg(wb)
                .arg(wb)
                .arg(&qq)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&n_in_u)
                .arg(&n_q_u)
                .arg(&n_kv_u)
                .arg(&oq)
                .arg(&ok)
                .arg(&ov)
                .arg(&m_u)
                .arg(&null_u64)
                .arg(&v_cache_off)
                .arg(&null_u64)
                .arg(&kv_dim_u)
                .launch_pdl(cfg),
        }
    }
    .ok()?;
    if split {
        let rb4 = (n_in / 256) * 144;
        let q_tail = n_q - n_kv;
        let oq_tail = oq + (n_kv as u64) * (rb4 as u64);
        let q_tail_ptr = q + (n_kv * 4) as u64;
        // Reuse already-quantized Q8 at gpu.q8_off (same xv).
        launch_qk_q8_matvec(
            gpu,
            GgmlType::Q4K,
            chunk_q,
            oq_tail,
            n_in,
            q_tail,
            xv,
            q_tail_ptr,
            m,
            false,
        )?;
        if std::env::var("ALLPAKA_FUSE_LOG")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            eprintln!("cuda: qkv_split n_kv={n_kv} q_tail={q_tail}");
        }
    }
    Some(())
}
