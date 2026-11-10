//! The CUDA device singleton: context, compiled kernel module, weight windows
//! and scratch arenas, plus initialisation and the locked accessor.

use super::*;

pub(super) const KERNEL_NAMES: &[&str] = &[
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
    /// Whole-prefill graph. Captured on the second call of a given length.
    pub pf_graph: Option<CudaGraph>,
    pub pf_graph_len: usize,
    pub pf_graph_seen: u32,
    pub pf_capturing: bool,
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
    #[allow(dead_code)]
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
    let mmq_ptx_src = include_str!("../../../../cuda/mmq_q4_k.ptx");
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
    let mmvq_ptx_src = include_str!("../../../../cuda/mmvq_q8.ptx");
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
                "matvec_q4_k_q8_2_k5120_bal",
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
        pf_graph: None,
        pf_graph_len: 0,
        pf_graph_seen: 0,
        pf_capturing: false,
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
