//! CUDA device runtime: context, streams, weight residency, scratch arenas.

use crate::gpu::cuda::kernels::KERNELS;
use allpaka_gguf::GgmlType;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
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
    "residual_add",
    "swiglu",
    "copy_f32",
    "cast_f32_to_f16",
    "cast_f16_to_f32",
    "rope_neox",
    "rope_neox_batch",
    "argmax_f32",
    "store_kv_f16",
    "store_kv_f16_dpos",
    "store_kv_batch_f16",
    "attend_gqa",
    "attend_gqa_dpos",
    "attend_gqa_batch",
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
    "quantize_q8_1",
    "matvec_q5_k",
    "matvec_q6_k",
    "matvec_q2_k",
    "matvec_q3_k",
    "dequant_rows_f16",
    "mm_q4_k",
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
    /// Scratch for argmax index (avoids alloc during graph capture).
    pub d_argmax: CudaSlice<u32>,
    /// Captured whole-token dense decode graph (invalidated on shape change).
    pub decode_graph: Option<CudaGraph>,
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
    pub q8_q: CudaSlice<i8>,
    pub q8_q_cap: usize,
    pub q8_d: CudaSlice<f32>,
    pub q8_d_cap: usize,
    /// Last quantized activation pointer (skip re-quantize for Q/K/V share).
    pub q8_src: u64,
    pub q8_src_n: usize,
    pub q8_src_m: usize,
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
    let stream = ctx.default_stream();
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
    let fallback = std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\include");
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
        let f = module
            .load_function(name)
            .map_err(|e| eprintln!("cuda: load_function {name}: {e}"))
            .ok()?;
        fns.insert(name, f);
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

    Some(CudaGpu {
        ctx,
        stream,
        module,
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
        d_argmax,
        decode_graph: None,
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
        q8_q,
        q8_q_cap: 1 << 16,
        q8_d,
        q8_d_cap: 1 << 12,
        q8_src: 0,
        q8_src_n: 0,
        q8_src_m: 0,
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
    }

    pub fn ensure_q8(&mut self, n: usize, rows: usize) -> Option<()> {
        let q_need = n * rows;
        let d_need = (n / 32) * rows;
        if q_need > self.q8_q_cap {
            let cap = q_need.next_power_of_two();
            self.q8_q = self.stream.alloc_zeros::<i8>(cap).ok()?;
            self.q8_q_cap = cap;
        }
        if d_need > self.q8_d_cap {
            let cap = d_need.next_power_of_two().max(1);
            self.q8_d = self.stream.alloc_zeros::<f32>(cap).ok()?;
            self.q8_d_cap = cap;
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

/// Q4/Q6_K: llama.cpp mmvq — one output row per block, N warps split K.
/// Q5_K: one warp per output row, several rows per block.
pub fn matvec_warp_rows(ty: GgmlType) -> Option<u32> {
    match ty {
        GgmlType::Q4K | GgmlType::Q6K => Some(8),
        GgmlType::Q5K => Some(8),
        _ => None,
    }
}

pub fn matvec_launch_cfg(ty: GgmlType, n_out: u32, m: u32) -> LaunchConfig {
    match ty {
        GgmlType::Q4K | GgmlType::Q6K => {
            let nwarps = 8u32;
            LaunchConfig {
                grid_dim: (n_out, m, 1),
                block_dim: (32, nwarps, 1),
                shared_mem_bytes: 0,
            }
        }
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
) -> Option<()> {
    gpu.ensure_arenas(m * n_in * 4, (y_off + m * n_out) * 4)?;
    if !x_already_in_arena {
        gpu.stream
            .memcpy_htod(x_elems, &mut gpu.x_arena.slice_mut(0..m * n_in))
            .ok()?;
    }

    // Opt-in quantized MMQ (dp4a): skips full-matrix f16 dequant.
    if matches!(ty, GgmlType::Q4K)
        && n_in % 256 == 0
        && std::env::var_os("ALLPAKA_MMQ").is_some()
        && std::env::var("ALLPAKA_MMQ").map_or(true, |v| v != "0")
    {
        let f = gpu.func_owned("mm_q4_k")?;
        let n_in_u = n_in as u32;
        let n_out_u = n_out as u32;
        let m_u = m as u32;
        let w_off_u = w_off;
        let cfg = LaunchConfig {
            grid_dim: (n_out_u.div_ceil(32), m_u.div_ceil(8), 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = Arc::clone(&gpu.stream);
        let (ybase, _g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
        let dst_p = ybase + (y_off * 4) as u64;
        let w = &gpu.chunks[chunk].buf;
        let xv = if x_from_pf_hs {
            gpu.pf_hs.slice(0..m * n_in)
        } else {
            gpu.x_arena.slice(0..m * n_in)
        };
        unsafe {
            stream
                .launch_builder(&f)
                .arg(w)
                .arg(&xv)
                .arg(&dst_p)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .launch(cfg)
        }
        .ok()?;
        return Some(());
    }

    let rb = row_bytes(ty, n_in)?;
    let fmt = dq_fmt(ty)?;
    let x_elems_n = m * n_in;
    let w_elems = n_out * n_in;
    let c_elems = m * n_out;
    // Layout [X | W | C] on the compute stream. Dual-stream ping-pong produced
    // non-finite logits on full Qwen3-32B prefill (parity only covered tiny GEMMs).
    let x16 = 0usize;
    let w16 = x_elems_n;
    let c16 = x_elems_n + w_elems;
    gpu.ensure_f16(c16 + c_elems)?;
    {
        let f = gpu.func_owned("dequant_rows_f16")?;
        let n_in_u = n_in as u32;
        let n_out_u = n_out as u32;
        let fmt_u = fmt;
        let rb_u = rb as u32;
        let w_off_u = w_off;
        const ROWS: u32 = 8;
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
            gpu.stream
                .launch_builder(&f)
                .arg(w)
                .arg(&mut out)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&rb_u)
                .arg(&fmt_u)
                .launch(cfg)
        }
        .ok()?;
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
                    .launch_builder(&f)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&n)
                    .launch(cfg)
            }
            .ok()?;
        } else {
            let src = gpu.x_arena.slice(0..x_elems_n);
            unsafe {
                gpu.stream
                    .launch_builder(&f)
                    .arg(&mut dst)
                    .arg(&src)
                    .arg(&n)
                    .launch(cfg)
            }
            .ok()?;
        }
        gpu.x_f16_n = x_elems_n;
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
                    .launch_builder(&f)
                    .arg(&dst_p)
                    .arg(&src)
                    .arg(&n)
                    .launch(lcfg)
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
                .launch_builder(&f)
                .arg(&a)
                .arg(&b)
                .arg(&dst_p)
                .arg(&m_u)
                .arg(&n_u)
                .arg(&k_u)
                .launch(cfg)
        }
        .ok()?;
    }
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
        gpu, ty, chunk, w_off, n_in, n_out, m, false, x_elems, 0, false, false,
    )?;
    let encode_ns = t0.elapsed().as_nanos() as u64;
    let t1 = Instant::now();
    gpu.sync()?;
    note_call(2, encode_ns, t1.elapsed().as_nanos() as u64);
    gpu.stream
        .clone_dtoh(&gpu.y_arena.slice(0..m * n_out))
        .ok()
}

fn q8_decode_enabled() -> bool {
    match std::env::var("ALLPAKA_Q8") {
        Ok(v) => v != "0",
        Err(_) => false,
    }
}

fn launch_q4k_q8_matvec(
    gpu: &mut CudaGpu,
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
    gpu.ensure_q8(n_in, m)?;
    let n_u = n_in as u32;
    let rows_u = m as u32;
    let need_q = gpu.q8_src != x_ptr || gpu.q8_src_n != n_in || gpu.q8_src_m != m;
    if need_q {
        let f = gpu.func_owned("quantize_q8_1")?;
        let cfg = LaunchConfig {
            grid_dim: (n_u / 32, rows_u, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut q = gpu.q8_q.slice_mut(0..n_in * m);
        let mut d = gpu.q8_d.slice_mut(0..(n_in / 32) * m);
        unsafe {
            gpu.stream
                .launch_builder(&f)
                .arg(&x_ptr)
                .arg(&mut q)
                .arg(&mut d)
                .arg(&n_u)
                .arg(&rows_u)
                .launch(cfg)
        }
        .ok()?;
        gpu.q8_src = x_ptr;
        gpu.q8_src_n = n_in;
        gpu.q8_src_m = m;
    }
    let f = gpu.func_owned("matvec_q4_k_q8")?;
    let n_out_u = n_out as u32;
    let m_u = m as u32;
    let w_off_u = w_off;
    let add_u = add as u32;
    let cfg = matvec_launch_cfg(GgmlType::Q4K, n_out_u, m_u);
    let wb = &gpu.chunks[chunk].buf;
    let q = gpu.q8_q.slice(0..n_in * m);
    let d = gpu.q8_d.slice(0..(n_in / 32) * m);
    unsafe {
        gpu.stream
            .launch_builder(&f)
            .arg(wb)
            .arg(&q)
            .arg(&d)
            .arg(&y_ptr)
            .arg(&n_u)
            .arg(&n_out_u)
            .arg(&w_off_u)
            .arg(&m_u)
            .arg(&add_u)
            .launch(cfg)
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
    if matches!(ty, GgmlType::Q4K) && n_in % 256 == 0 && q8_decode_enabled() {
        let stream = Arc::clone(&gpu.stream);
        let (xbase, g) = DevicePtr::device_ptr(&gpu.x_arena, &stream);
        drop(g);
        let (ybase, g) = DevicePtr::device_ptr(&gpu.y_arena, &stream);
        drop(g);
        launch_q4k_q8_matvec(gpu, chunk, w_off, n_in, n_out, xbase, ybase, m, false)?;
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
                    .launch_builder(&f)
                    .arg(wb)
                    .arg(&xv)
                    .arg(&mut yv)
                    .arg(&n_in_u)
                    .arg(&n_out_u)
                    .arg(&w_off_u)
                    .arg(&m_u)
                    .arg(&add_u)
                    .launch(cfg)
            } else {
                gpu.stream
                    .launch_builder(&f)
                    .arg(wb)
                    .arg(&xv)
                    .arg(&mut yv)
                    .arg(&n_in_u)
                    .arg(&n_out_u)
                    .arg(&w_off_u)
                    .arg(&m_u)
                    .launch(cfg)
            }
        }
        .ok()?;
    }
    let encode_ns = t0.elapsed().as_nanos() as u64;
    let t1 = Instant::now();
    gpu.sync()?;
    note_call(1, encode_ns, t1.elapsed().as_nanos() as u64);
    gpu.stream
        .clone_dtoh(&gpu.y_arena.slice(0..m * n_out))
        .ok()
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
    launch_matvec_y_ptr(gpu, ty, chunk, w_off, n_in, n_out, x_off, y_off, m, base, false)
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
    let xv = y_base + (x_off * std::mem::size_of::<f32>()) as u64;
    let yv = y_base + (y_off * std::mem::size_of::<f32>()) as u64;
    if matches!(ty, GgmlType::Q4K) && n_in % 256 == 0 && q8_decode_enabled() {
        return launch_q4k_q8_matvec(gpu, chunk, w_off, n_in, n_out, xv, yv, m, add);
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
            gpu.stream
                .launch_builder(&f)
                .arg(wb)
                .arg(&xv)
                .arg(&yv)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .arg(&add_u)
                .launch(cfg)
        } else {
            gpu.stream
                .launch_builder(&f)
                .arg(wb)
                .arg(&xv)
                .arg(&yv)
                .arg(&n_in_u)
                .arg(&n_out_u)
                .arg(&w_off_u)
                .arg(&m_u)
                .launch(cfg)
        }
    }
    .ok()?;
    Some(())
}
