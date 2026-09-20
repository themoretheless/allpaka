// Direct llama MMVQ entry (no ggml cgraph). Linked into allpaka_ggml.dll.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/fattn.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>

static cudaStream_t g_peer = nullptr;
static cudaStream_t g_ggml_private = nullptr;
static cudaEvent_t g_ev_peer = nullptr;
static cudaEvent_t g_ev_ggml = nullptr;
static ggml_backend_cuda_context * g_ctx = nullptr;
static bool g_shared_stream = false;

static void ensure_peer_events() {
    if (!g_ev_peer) {
        cudaEventCreateWithFlags(&g_ev_peer, cudaEventDisableTiming);
    }
    if (!g_ev_ggml) {
        cudaEventCreateWithFlags(&g_ev_ggml, cudaEventDisableTiming);
    }
}

static void reset_stream_helpers(int stream_no) {
    if (!g_ctx) {
        return;
    }
    const int dev = g_ctx->device;
    if (g_ctx->cublas_handles[dev][stream_no]) {
        cublasDestroy(g_ctx->cublas_handles[dev][stream_no]);
        g_ctx->cublas_handles[dev][stream_no] = nullptr;
    }
    g_ctx->cublas_workspaces[dev][stream_no] = nullptr;
    // Drop pool tied to the old stream; ggml recreates on demand.
    g_ctx->pools[dev][stream_no].reset();
}

static void inject_shared_stream() {
    if (!g_ctx || !g_shared_stream || !g_peer) {
        return;
    }
    // Install cudarc's stream into ggml so FA/MMVQ and decode graphs share one queue.
    cudaStream_t & slot = g_ctx->streams[g_ctx->device][0];
    if (slot != nullptr && slot != g_peer) {
        // Prefill dual-queue created a private ggml stream; park it and force peer.
        g_ggml_private = slot;
    }
    if (slot != g_peer) {
        slot = g_peer;
        reset_stream_helpers(0);
    }
    g_ctx->curr_stream_no = 0;
}

extern "C" void allpaka_peer_set_ctx(void * cuda_ctx_v) {
    g_ctx = static_cast<ggml_backend_cuda_context *>(cuda_ctx_v);
    inject_shared_stream();
}

static void drain_for_shared() {
    // Drain dual-queue work before decode graph capture on the cudarc stream.
    if (g_peer) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(g_peer, &cap) == cudaSuccess
            && cap != cudaStreamCaptureStatusNone) {
            return;
        }
        cudaStreamSynchronize(g_peer);
    }
    if (!g_ctx) {
        return;
    }
    const int dev = g_ctx->device;
    for (int i = 0; i < GGML_CUDA_MAX_STREAMS; ++i) {
        cudaStream_t s = g_ctx->streams[dev][i];
        if (s && s != g_peer) {
            cudaStreamSynchronize(s);
        }
    }
    if (g_ggml_private && g_ggml_private != g_peer) {
        cudaStreamSynchronize(g_ggml_private);
    }
    g_ctx->curr_stream_no = 0;
}

extern "C" int allpaka_peer_bind(void * peer_stream) {
    g_peer = static_cast<cudaStream_t>(peer_stream);
    // nullptr = legacy default stream; cannot inject (ggml would recreate).
    if (g_peer != nullptr) {
        if (!g_shared_stream) {
            drain_for_shared();
        }
        g_shared_stream = true;
        inject_shared_stream();
        fprintf(stderr, "allpaka_ggml: shared cudarc stream installed\n");
        return 0;
    }
    g_shared_stream = false;
    return 0;
}

extern "C" void allpaka_peer_clear_shared(void) {
    if (!g_ctx) {
        return;
    }
    cudaStream_t & slot = g_ctx->streams[g_ctx->device][0];
    if (g_shared_stream && slot == g_peer) {
        // Prefer the parked private stream; else null -> create on demand.
        slot = g_ggml_private;
        g_ggml_private = nullptr;
        reset_stream_helpers(0);
    }
    g_shared_stream = false;
    ensure_peer_events();
}

static cudaStream_t ggml_stream_handle() {
    if (!g_ctx) {
        return nullptr;
    }
    if (g_shared_stream && g_peer) {
        return g_peer;
    }
    cudaStream_t & slot = g_ctx->streams[g_ctx->device][g_ctx->curr_stream_no];
    if (slot == nullptr) {
        cudaStreamCreateWithFlags(&slot, cudaStreamNonBlocking);
    }
    return slot;
}

extern "C" int allpaka_peer_is_shared(void) {
    return g_shared_stream ? 1 : 0;
}

extern "C" cudaStream_t allpaka_peer_stream(void) {
    return ggml_stream_handle();
}

extern "C" void allpaka_peer_wait(void) {
    // A/B: ALLPAKA_PEER_NOSYNC=1 skips waits (racy overlap; was ~3050 pp historically).
    if (g_shared_stream || !g_peer || !g_ev_peer) {
        return;
    }
    const char * nosync = std::getenv("ALLPAKA_PEER_NOSYNC");
    if (nosync && (nosync[0] == '1' || nosync[0] == 't' || nosync[0] == 'T')) {
        return;
    }
    cudaStream_t gs = ggml_stream_handle();
    if (!gs || gs == g_peer) {
        return;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(g_peer, &cap) == cudaSuccess
        && cap != cudaStreamCaptureStatusNone) {
        return;
    }
    // GPU wait on ggml private only — never WaitEvent onto g_peer (graph capture).
    cudaEventRecord(g_ev_peer, g_peer);
    cudaStreamWaitEvent(gs, g_ev_peer, 0);
}

extern "C" void allpaka_peer_signal(void) {
    if (g_shared_stream || !g_ctx || !g_ev_ggml || !g_peer) {
        return;
    }
    const char * nosync = std::getenv("ALLPAKA_PEER_NOSYNC");
    if (nosync && (nosync[0] == '1' || nosync[0] == 't' || nosync[0] == 'T')) {
        return;
    }
    cudaStream_t gs = ggml_stream_handle();
    if (!gs) {
        return;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(g_peer, &cap) == cudaSuccess
        && cap != cudaStreamCaptureStatusNone) {
        return;
    }
    // Prefill only (shared=false). Decode rebinds shared before capture so
    // these WaitEvent edges are drained in drain_for_shared and never recorded
    // into the decode graph.
    cudaEventRecord(g_ev_ggml, gs);
    cudaStreamWaitEvent(g_peer, g_ev_ggml, 0);
}

extern "C" int allpaka_direct_mmvq(
    void * cuda_ctx_v,
    const void * w_dev,
    const float * x_dev,
    float * y_dev,
    int32_t n_in,
    int32_t n_out,
    int32_t w_type)
{
    if (!cuda_ctx_v || !w_dev || !x_dev || !y_dev || n_in <= 0 || n_out <= 0) {
        return -2;
    }
    const ggml_type ty = (ggml_type)w_type;
    if (ty != GGML_TYPE_Q4_K && ty != GGML_TYPE_Q6_K) {
        return -3;
    }
    if (n_in % ggml_blck_size(ty) != 0) {
        return -4;
    }

    auto & ctx = *static_cast<ggml_backend_cuda_context *>(cuda_ctx_v);
    g_ctx = &ctx;
    inject_shared_stream();

    ggml_tensor W;
    ggml_tensor X;
    ggml_tensor Y;
    std::memset(&W, 0, sizeof(W));
    std::memset(&X, 0, sizeof(X));
    std::memset(&Y, 0, sizeof(Y));

    W.type = ty;
    W.ne[0] = n_in;
    W.ne[1] = n_out;
    W.ne[2] = 1;
    W.ne[3] = 1;
    W.nb[0] = ggml_type_size(ty);
    W.nb[1] = ggml_row_size(ty, n_in);
    W.nb[2] = W.nb[1] * (size_t)n_out;
    W.nb[3] = W.nb[2];
    W.data = const_cast<void *>(w_dev);

    X.type = GGML_TYPE_F32;
    X.ne[0] = n_in;
    X.ne[1] = 1;
    X.ne[2] = 1;
    X.ne[3] = 1;
    X.nb[0] = sizeof(float);
    X.nb[1] = sizeof(float) * (size_t)n_in;
    X.nb[2] = X.nb[1];
    X.nb[3] = X.nb[1];
    X.data = const_cast<float *>(x_dev);

    Y.type = GGML_TYPE_F32;
    Y.ne[0] = n_out;
    Y.ne[1] = 1;
    Y.ne[2] = 1;
    Y.ne[3] = 1;
    Y.nb[0] = sizeof(float);
    Y.nb[1] = sizeof(float) * (size_t)n_out;
    Y.nb[2] = Y.nb[1];
    Y.nb[3] = Y.nb[1];
    Y.data = y_dev;
    Y.src[0] = &W;
    Y.src[1] = &X;
    Y.op = GGML_OP_MUL_MAT;

    ggml_backend_buffer buf_w{};
    ggml_backend_buffer buf_x{};
    ggml_backend_buffer buf_y{};
    buf_w.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    buf_x.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    buf_y.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    W.buffer = &buf_w;
    X.buffer = &buf_x;
    Y.buffer = &buf_y;

    static int calls = 0;
    if (calls < 3) {
        fprintf(stderr, "allpaka_ggml: direct_mmvq n_in=%d n_out=%d type=%d shared=%d (call %d)\n",
            n_in, n_out, w_type, (int)g_shared_stream, calls);
    }
    calls++;

    ggml_cuda_mul_mat_vec_q(ctx, &W, &X, nullptr, &Y, nullptr);
    if (calls == 1 || calls == 100 || calls == 1000) {
        fprintf(stderr, "allpaka_ggml: direct_mmvq calls=%d\n", calls);
    }
    return 0;
}

// Decode FA without ggml cgraph scheduling (same kernels as graph_compute).
extern "C" int allpaka_fa_launch_direct(void * fa_tensor_v) {
    if (!g_ctx || !fa_tensor_v) {
        return -1;
    }
    auto * fa = static_cast<ggml_tensor *>(fa_tensor_v);
    if (fa->op != GGML_OP_FLASH_ATTN_EXT || !fa->data) {
        return -2;
    }
    ggml_cuda_flash_attn_ext(*g_ctx, fa);
    return 0;
}

#include <cublasLt.h>
#include <cuda_fp8.h>

__global__ void k_f32_to_f16(const float * x, half * y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half(x[i]);
    }
}

__global__ void k_f16_to_f32(const half * x, float * y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __half2float(x[i]);
    }
}

// Cached f16 weights, cublasLt heuristic instead of GemmEx default algo.
extern "C" int allpaka_lt_gemm(
    void * cuda_ctx_v, const void * w_f16, const float * x_f32, float * y_f32,
    int n_in, int n_out, int m)
{
    if (!cuda_ctx_v || !w_f16 || !x_f32 || !y_f32 || n_in <= 0 || n_out <= 0 || m <= 0) {
        return -2;
    }
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(cuda_ctx_v);
    g_ctx = &ctx;
    inject_shared_stream();
    cudaStream_t stream = (g_shared_stream && g_peer) ? g_peer : ctx.streams[ctx.device][0];
    if (!stream) {
        return -3;
    }

    static cublasLtHandle_t lt = nullptr;
    static void * scratch = nullptr;
    static size_t scratch_bytes = 0;
    static void * workspace = nullptr;
    static constexpr size_t k_ws = 64ull << 20;

    struct Key {
        int n_in, n_out, m;
        bool operator==(const Key & o) const {
            return n_in == o.n_in && n_out == o.n_out && m == o.m;
        }
    };
    struct KeyHash {
        size_t operator()(const Key & k) const {
            return ((size_t) k.n_in * 1315423911u) ^ ((size_t) k.n_out << 1) ^ (size_t) k.m;
        }
    };
    static std::unordered_map<Key, cublasLtMatmulAlgo_t, KeyHash> algos;

    if (!lt && cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        return -3;
    }
    if (!workspace && cudaMalloc(&workspace, k_ws) != cudaSuccess) {
        return -4;
    }

    const size_t x_n = (size_t) m * (size_t) n_in;
    const size_t y_n = (size_t) m * (size_t) n_out;
    const size_t need = (x_n + y_n) * sizeof(half);
    if (scratch_bytes < need) {
        if (scratch) {
            cudaFree(scratch);
        }
        if (cudaMalloc(&scratch, need) != cudaSuccess) {
            scratch = nullptr;
            scratch_bytes = 0;
            return -5;
        }
        scratch_bytes = need;
    }
    half * x_f16 = static_cast<half *>(scratch);
    half * y_f16 = x_f16 + x_n;

    k_f32_to_f16<<<(x_n + 255) / 256, 256, 0, stream>>>(x_f32, x_f16, (int) x_n);

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, CUDA_R_16F) != CUBLAS_STATUS_SUCCESS) {
        return -6;
    }
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    // A is W as column-major [n_in, n_out], B is X [n_in, m], C is Y [n_out, m].
    cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, n_in, n_out, n_in);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, n_in, m, n_in);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, n_out, m, n_out);

    Key key{n_in, n_out, m};
    auto it = algos.find(key);
    if (it == algos.end()) {
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &k_ws, sizeof(k_ws));
        cublasLtMatmulHeuristicResult_t heur{};
        int n_found = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            lt, desc, a, b, c, c, pref, 1, &heur, &n_found);
        cublasLtMatmulPreferenceDestroy(pref);
        pref = nullptr;
        if (hs != CUBLAS_STATUS_SUCCESS || n_found < 1) {
            cublasLtMatmulDescDestroy(desc);
            cublasLtMatrixLayoutDestroy(a);
            cublasLtMatrixLayoutDestroy(b);
            cublasLtMatrixLayoutDestroy(c);
            return -7;
        }
        it = algos.emplace(key, heur.algo).first;
        static int logged = 0;
        if (logged < 4) {
            fprintf(stderr, "allpaka_ggml: cublasLt n_in=%d n_out=%d m=%d\n", n_in, n_out, m);
            logged++;
        }
    }

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    cublasStatus_t st = cublasLtMatmul(
        lt, desc,
        &alpha, w_f16, a,
        x_f16, b,
        &beta, y_f16, c,
        y_f16, c,
        &it->second, workspace, k_ws, stream);

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return -8;
    }
    k_f16_to_f32<<<(y_n + 255) / 256, 256, 0, stream>>>(y_f16, y_f32, (int) y_n);
    return 0;
}

__device__ float warp_max_f(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

__global__ void k_amax_f16(const half * x, float * partial, int n) {
    float m = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        m = fmaxf(m, fabsf(__half2float(x[i])));
    }
    m = warp_max_f(m);
    __shared__ float sm[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sm[warp] = m;
    }
    __syncthreads();
    if (warp == 0) {
        m = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.f;
        m = warp_max_f(m);
        if (lane == 0) {
            partial[blockIdx.x] = m;
        }
    }
}

__global__ void k_amax_f32(const float * x, float * partial, int n) {
    float m = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        m = fmaxf(m, fabsf(x[i]));
    }
    m = warp_max_f(m);
    __shared__ float sm[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sm[warp] = m;
    }
    __syncthreads();
    if (warp == 0) {
        m = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.f;
        m = warp_max_f(m);
        if (lane == 0) {
            partial[blockIdx.x] = m;
        }
    }
}

__global__ void k_reduce_max(const float * p, int n, float * out) {
    float m = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        m = fmaxf(m, p[i]);
    }
    m = warp_max_f(m);
    __shared__ float sm[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sm[warp] = m;
    }
    __syncthreads();
    if (warp == 0) {
        m = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.f;
        m = warp_max_f(m);
        if (lane == 0) {
            *out = m;
        }
    }
}

__global__ void k_q_f16_e4m3(const half * x, __nv_fp8_e4m3 * y, int n, const float * amax, float * scale) {
    const float s = fmaxf(*amax, 1e-8f) / 448.f;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *scale = s;
    }
    const float inv = 1.f / s;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = __nv_fp8_e4m3(__half2float(x[i]) * inv);
    }
}

__global__ void k_q_f32_e4m3(const float * x, __nv_fp8_e4m3 * y, int n, const float * amax, float * scale) {
    const float s = fmaxf(*amax, 1e-8f) / 448.f;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *scale = s;
    }
    const float inv = 1.f / s;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = __nv_fp8_e4m3(x[i] * inv);
    }
}

// Prefill GEMM on FP8 tensor cores. Weights quantized once from the f16 cache.
extern "C" int allpaka_fp8_gemm(
    void * cuda_ctx_v, const void * w_f16, const float * x_f32, float * y_f32,
    int n_in, int n_out, int m)
{
    if (!cuda_ctx_v || !w_f16 || !x_f32 || !y_f32 || n_in <= 0 || n_out <= 0 || m <= 0) {
        return -2;
    }
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(cuda_ctx_v);
    g_ctx = &ctx;
    inject_shared_stream();
    cudaStream_t stream = (g_shared_stream && g_peer) ? g_peer : ctx.streams[ctx.device][0];
    if (!stream) {
        return -3;
    }

    static cublasLtHandle_t lt = nullptr;
    static float * partial = nullptr;
    static float * amax = nullptr;
    static void * scratch = nullptr;
    static size_t scratch_bytes = 0;
    static void * workspace = nullptr;
    static constexpr int k_blocks = 128;
    static constexpr size_t k_ws = 64ull << 20;

    struct W8 {
        void * fp8;
        float * scale;
    };
    static std::unordered_map<const void *, W8> weights;

    struct Key {
        int n_in, n_out, m;
        bool operator==(const Key & o) const {
            return n_in == o.n_in && n_out == o.n_out && m == o.m;
        }
    };
    struct KeyHash {
        size_t operator()(const Key & k) const {
            return ((size_t) k.n_in * 1315423911u) ^ ((size_t) k.n_out << 1) ^ (size_t) k.m;
        }
    };
    static std::unordered_map<Key, cublasLtMatmulAlgo_t, KeyHash> algos;

    if (!lt && cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        return -4;
    }
    if (!partial && cudaMalloc(&partial, k_blocks * sizeof(float)) != cudaSuccess) {
        return -5;
    }
    if (!amax && cudaMalloc(&amax, 2 * sizeof(float)) != cudaSuccess) {
        return -5;
    }
    if (!workspace && cudaMalloc(&workspace, k_ws) != cudaSuccess) {
        return -5;
    }

    const int64_t nw = (int64_t) n_in * (int64_t) n_out;
    auto wit = weights.find(w_f16);
    if (wit == weights.end()) {
        W8 w{};
        if (cudaMalloc(&w.fp8, (size_t) nw) != cudaSuccess || cudaMalloc(&w.scale, sizeof(float)) != cudaSuccess) {
            if (w.fp8) {
                cudaFree(w.fp8);
            }
            return -6;
        }
        k_amax_f16<<<k_blocks, 256, 0, stream>>>(static_cast<const half *>(w_f16), partial, (int) nw);
        k_reduce_max<<<1, 128, 0, stream>>>(partial, k_blocks, amax);
        k_q_f16_e4m3<<<k_blocks, 256, 0, stream>>>(
            static_cast<const half *>(w_f16), static_cast<__nv_fp8_e4m3 *>(w.fp8), (int) nw, amax, w.scale);
        wit = weights.emplace(w_f16, w).first;
        static int logged = 0;
        if (logged < 3) {
            fprintf(stderr, "allpaka_ggml: fp8 weight %d MiB\n", (int) (nw >> 20));
            logged++;
        }
    }

    const size_t x_n = (size_t) m * (size_t) n_in;
    const size_t y_n = (size_t) m * (size_t) n_out;
    const size_t need = x_n + y_n * sizeof(half);
    if (scratch_bytes < need) {
        if (scratch) {
            cudaFree(scratch);
        }
        if (cudaMalloc(&scratch, need) != cudaSuccess) {
            scratch = nullptr;
            scratch_bytes = 0;
            return -7;
        }
        scratch_bytes = need;
    }
    __nv_fp8_e4m3 * x8 = static_cast<__nv_fp8_e4m3 *>(scratch);
    half * y16 = reinterpret_cast<half *>(x8 + x_n);
    float * b_scale = amax + 1;

    k_amax_f32<<<k_blocks, 256, 0, stream>>>(x_f32, partial, (int) x_n);
    k_reduce_max<<<1, 128, 0, stream>>>(partial, k_blocks, amax);
    k_q_f32_e4m3<<<k_blocks, 256, 0, stream>>>(x_f32, x8, (int) x_n, amax, b_scale);

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return -8;
    }
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    float * a_scale = wit->second.scale;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale));
    const char * fe = std::getenv("ALLPAKA_FP8_FAST");
    const int8_t fast = (fe && fe[0] == '0') ? 0 : 1;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast));
    cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, n_in, n_out, n_in);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, n_in, m, n_in);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, n_out, m, n_out);

    Key key{n_in, n_out, m};
    auto it = algos.find(key);
    if (it == algos.end()) {
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &k_ws, sizeof(k_ws));
        cublasLtMatmulHeuristicResult_t heur{};
        int n_found = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            lt, desc, a, b, c, c, pref, 1, &heur, &n_found);
        cublasLtMatmulPreferenceDestroy(pref);
        if (hs != CUBLAS_STATUS_SUCCESS || n_found < 1) {
            cublasLtMatmulDescDestroy(desc);
            cublasLtMatrixLayoutDestroy(a);
            cublasLtMatrixLayoutDestroy(b);
            cublasLtMatrixLayoutDestroy(c);
            static int bad = 0;
            if (bad < 2) {
                fprintf(stderr, "allpaka_ggml: fp8 heuristic failed %d n_in=%d n_out=%d m=%d\n",
                        (int) hs, n_in, n_out, m);
                bad++;
            }
            return -9;
        }
        it = algos.emplace(key, heur.algo).first;
        static int logged = 0;
        if (logged < 4) {
            fprintf(stderr, "allpaka_ggml: fp8 gemm n_in=%d n_out=%d m=%d\n", n_in, n_out, m);
            logged++;
        }
    }

    const float alpha = 1.f;
    const float beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(
        lt, desc,
        &alpha, wit->second.fp8, a,
        x8, b,
        &beta, y16, c,
        y16, c,
        &it->second, workspace, k_ws, stream);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    if (st != CUBLAS_STATUS_SUCCESS) {
        static int bad = 0;
        if (bad < 2) {
            fprintf(stderr, "allpaka_ggml: fp8 matmul failed %d\n", (int) st);
            bad++;
        }
        return -10;
    }
    k_f16_to_f32<<<(y_n + 255) / 256, 256, 0, stream>>>(y16, y_f32, (int) y_n);
    return 0;
}

static __device__ void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static __device__ void store_partial(float m, float * partial) {
    m = warp_max_f(m);
    __shared__ float sm[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sm[warp] = m;
    }
    __syncthreads();
    if (warp == 0) {
        m = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.f;
        m = warp_max_f(m);
        if (lane == 0) {
            partial[blockIdx.x] = m;
        }
    }
}

__global__ void k_q4k_amax(const block_q4_K * x, int nblocks, float * partial) {
    float m = 0.f;
    for (int ib = blockIdx.x * blockDim.x + threadIdx.x; ib < nblocks; ib += blockDim.x * gridDim.x) {
        const float dall = __low2half(x[ib].dm);
        const float dmin = __high2half(x[ib].dm);
        for (int j = 0; j < 8; ++j) {
            uint8_t sc, mn;
            scale_min_k4(j, x[ib].scales, sc, mn);
            const float d = dall * (float) sc;
            const float mm = dmin * (float) mn;
            m = fmaxf(m, fabsf(mm));
            m = fmaxf(m, fabsf(15.f * d - mm));
        }
    }
    store_partial(m, partial);
}

__global__ void k_q6k_amax(const block_q6_K * x, int nblocks, float * partial) {
    float m = 0.f;
    for (int ib = blockIdx.x * blockDim.x + threadIdx.x; ib < nblocks; ib += blockDim.x * gridDim.x) {
        const float d = fabsf((float) x[ib].d);
        float ms = 0.f;
        for (int j = 0; j < QK_K / 16; ++j) {
            ms = fmaxf(ms, fabsf((float) x[ib].scales[j]));
        }
        m = fmaxf(m, d * ms * 32.f);
    }
    store_partial(m, partial);
}

// One warp, 256 fp8, coalesced 8-byte stores. inv already applied by the caller via amax.
__global__ void k_q4k_to_fp8(const block_q4_K * x, __nv_fp8_e4m3 * y, int nblocks, const float * amax) {
    const float inv = 448.f / fmaxf(*amax, 1e-8f);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    __shared__ float sm[8][256];
    const int il = lane / 8;
    const int ir = lane % 8;
    const int is = 2 * il;
    const int scat = 64 * il + 4 * ir;
    for (int ib = blockIdx.x * nwarps + warp; ib < nblocks; ib += gridDim.x * nwarps) {
        const block_q4_K & b = x[ib];
        const float dall = __low2half(b.dm);
        const float dmin = __high2half(b.dm);
        const uint8_t * q = b.qs + 32 * il + 4 * ir;
        uint8_t sc, mn;
        scale_min_k4(is, b.scales, sc, mn);
        const float d1 = dall * (float) sc;
        const float m1 = dmin * (float) mn;
        scale_min_k4(is + 1, b.scales, sc, mn);
        const float d2 = dall * (float) sc;
        const float m2 = dmin * (float) mn;
        float * s = sm[warp];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            s[scat + l] = (d1 * (float) (q[l] & 0xF) - m1) * inv;
            s[scat + 32 + l] = (d2 * (float) (q[l] >> 4) - m2) * inv;
        }
        __syncwarp();
        alignas(8) __nv_fp8_e4m3 tmp[8];
        const int o = lane * 8;
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            tmp[l] = __nv_fp8_e4m3(s[o + l]);
        }
        *reinterpret_cast<uint2 *>(reinterpret_cast<unsigned char *>(y) + (int64_t) ib * QK_K + o) =
            *reinterpret_cast<uint2 *>(tmp);
        __syncwarp();
    }
}

__global__ void k_q6k_to_fp8(const block_q6_K * x, __nv_fp8_e4m3 * y, int nblocks, const float * amax) {
    const float inv = 448.f / fmaxf(*amax, 1e-8f);
    __shared__ float s[256];
    const int tid = threadIdx.x;
    const int ip = tid >> 5;
    const int il = tid & 31;
    const int is = 8 * ip + il / 16;
    for (int ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const block_q6_K & b = x[ib];
        const float d = (float) b.d * inv;
        const uint8_t * ql = b.ql + 64 * ip + il;
        const uint8_t qh = b.qh[32 * ip + il];
        const int8_t * sc = b.scales + is;
        const int yo = 128 * ip + il;
        s[yo] = d * (float) sc[0] * (float) ((int) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
        s[yo + 32] = d * (float) sc[2] * (float) ((int) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
        s[yo + 64] = d * (float) sc[4] * (float) ((int) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
        s[yo + 96] = d * (float) sc[6] * (float) ((int) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
        __syncthreads();
        alignas(4) __nv_fp8_e4m3 tmp[4];
        const int o = tid * 4;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            tmp[l] = __nv_fp8_e4m3(s[o + l]);
        }
        *reinterpret_cast<uint32_t *>(reinterpret_cast<unsigned char *>(y) + (int64_t) ib * QK_K + o) =
            *reinterpret_cast<uint32_t *>(tmp);
        __syncthreads();
    }
}

__global__ void k_scale_div448(float * s) {
    if (threadIdx.x == 0) {
        *s = fmaxf(*s, 1e-8f) / 448.f;
    }
}

// Q4_K / Q6_K to FP8 each call. No resident f16 copy.
extern "C" int allpaka_fp8_gemm_q(
    void * cuda_ctx_v, const void * w_q, const float * x_f32, float * y_f32,
    int n_in, int n_out, int m, int w_type)
{
    if (!cuda_ctx_v || !w_q || !x_f32 || !y_f32 || n_in <= 0 || n_out <= 0 || m < 32) {
        return -2;
    }
    if (w_type != GGML_TYPE_Q4_K && w_type != GGML_TYPE_Q6_K) {
        return -3;
    }
    if ((n_in % QK_K) != 0) {
        return -4;
    }
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(cuda_ctx_v);
    g_ctx = &ctx;
    inject_shared_stream();
    cudaStream_t stream = (g_shared_stream && g_peer) ? g_peer : ctx.streams[ctx.device][0];
    if (!stream) {
        return -3;
    }

    static cublasLtHandle_t lt = nullptr;
    static float * partial = nullptr;
    static float * qpartial = nullptr;
    static float * amax = nullptr;
    static float * w_scale2[8] = {};
    static void * w8s[8] = {};
    static int nslots = 0;
    static size_t w8_bytes = 0;
    static void * scratch = nullptr;
    static size_t scratch_bytes = 0;
    static void * workspace = nullptr;
    static cudaStream_t qstream = nullptr;
    static cudaEvent_t qev[2048];
    static int qev_n = 0;
    static int qev_i = 0;
    static int slot_i = 0;
    static int fork_ev = -1;
    static int slot_gemm_ev[8] = {};
    static bool slot_used[8] = {};
    static bool saw_capture = false;
    static cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr;
    static int timed = 0;
    static constexpr int k_blocks = 128;
    static constexpr size_t k_ws = 128ull << 20;

    struct Key {
        int n_in, n_out, m;
        bool operator==(const Key & o) const {
            return n_in == o.n_in && n_out == o.n_out && m == o.m;
        }
    };
    struct KeyHash {
        size_t operator()(const Key & k) const {
            return ((size_t) k.n_in * 1315423911u) ^ ((size_t) k.n_out << 1) ^ (size_t) k.m;
        }
    };
    static std::unordered_map<Key, cublasLtMatmulAlgo_t, KeyHash> algos;

    const int64_t nw = (int64_t) n_in * (int64_t) n_out;
    const int nblocks = (int) (nw / QK_K);
    int m_gemm = m;
    if (m >= 128) {
        const int up = (m + 127) & ~127;
        if (up - m <= 64) {
            m_gemm = up;
        }
    }
    const size_t x_n = (size_t) m * (size_t) n_in;
    const size_t x_pad = (size_t) m_gemm * (size_t) n_in;
    const size_t y_n = (size_t) m * (size_t) n_out;
    const size_t y_pad = (size_t) m_gemm * (size_t) n_out;
    const size_t need_w = (size_t) nw;
    const size_t need_s = x_pad + y_pad * sizeof(half);

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    const bool capturing = cap != cudaStreamCaptureStatusNone;

    if (!lt && cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        return -4;
    }
    if (!partial && cudaMalloc(&partial, k_blocks * sizeof(float)) != cudaSuccess) {
        return -5;
    }
    if (!qpartial && cudaMalloc(&qpartial, k_blocks * sizeof(float)) != cudaSuccess) {
        return -5;
    }
    if (!amax && cudaMalloc(&amax, 2 * sizeof(float)) != cudaSuccess) {
        return -5;
    }
    for (int s = 0; s < 8; ++s) {
        if (!w_scale2[s] && cudaMalloc(&w_scale2[s], sizeof(float)) != cudaSuccess) {
            return -5;
        }
    }
    if (!workspace && cudaMalloc(&workspace, k_ws) != cudaSuccess) {
        return -5;
    }
    if (!qstream && cudaStreamCreateWithFlags(&qstream, cudaStreamNonBlocking) != cudaSuccess) {
        return -5;
    }
    if (qev_n == 0) {
        for (int i = 0; i < 2048; ++i) {
            if (cudaEventCreateWithFlags(&qev[i], cudaEventDisableTiming) != cudaSuccess) {
                return -5;
            }
        }
        qev_n = 2048;
    }
    if (!ev0) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventCreate(&ev2);
    }
    if (!capturing && (w8_bytes < need_w || scratch_bytes < need_s)) {
        cudaStreamSynchronize(stream);
        if (qstream) {
            cudaStreamSynchronize(qstream);
        }
    }
    if (w8_bytes < need_w) {
        if (capturing) {
            return -6;
        }
        for (int s = 0; s < 8; ++s) {
            if (w8s[s]) {
                cudaFree(w8s[s]);
                w8s[s] = nullptr;
            }
        }
        w8_bytes = 0;
        nslots = 0;
        for (int s = 0; s < 2; ++s) {
            if (cudaMalloc(&w8s[s], need_w) != cudaSuccess) {
                break;
            }
            nslots++;
        }
        if (nslots < 1) {
            return -6;
        }
        w8_bytes = need_w;
    }
    if (scratch_bytes < need_s) {
        if (capturing) {
            return -7;
        }
        if (scratch) {
            cudaFree(scratch);
        }
        if (cudaMalloc(&scratch, need_s) != cudaSuccess) {
            scratch = nullptr;
            scratch_bytes = 0;
            return -7;
        }
        scratch_bytes = need_s;
    }

    if (capturing && !saw_capture) {
        qev_i = 0;
        slot_i = 0;
        fork_ev = -1;
        for (int s = 0; s < 8; ++s) {
            slot_used[s] = false;
        }
    } else if (!capturing && qstream
               && cudaStreamQuery(stream) == cudaSuccess
               && cudaStreamQuery(qstream) == cudaSuccess) {
        qev_i = 0;
        slot_i = 0;
        fork_ev = -1;
        for (int s = 0; s < 8; ++s) {
            slot_used[s] = false;
        }
    }
    saw_capture = capturing;

    // Quant of an uncached tensor overlaps the previous GEMM. The fork event
    // is recorded before that GEMM, so the side stream does not wait for it.
    // Two buffers: this quant must not overwrite weights the in-flight GEMM reads.
    void * w8 = w8s[0];
    float * w_scale = w_scale2[0];
    struct W8C {
        void * fp8;
        float * scale;
    };
    static std::unordered_map<const void *, W8C> wcache;
    static int n_cached = 0;
    static size_t cached_bytes = 0;
    bool cached = false;
    bool staging = false;
    auto cit = wcache.find(w_q);
    if (cit != wcache.end()) {
        w8 = cit->second.fp8;
        w_scale = cit->second.scale;
        cached = true;
    } else if (!capturing && need_w >= (1ull << 20)) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        if (free_b > need_w + (512ull << 20)) {
            void * fp8 = nullptr;
            float * sc = nullptr;
            if (cudaMalloc(&fp8, need_w) == cudaSuccess && cudaMalloc(&sc, sizeof(float)) == cudaSuccess) {
                w8 = fp8;
                w_scale = sc;
                wcache.emplace(w_q, W8C{fp8, sc});
                staging = true;
                cached_bytes += need_w;
                n_cached++;
                if (n_cached <= 2 || (n_cached % 48) == 0) {
                    fprintf(stderr, "allpaka_ggml: fp8 cache %d tensors %d MiB free %d MiB\n",
                            n_cached, (int) (cached_bytes >> 20), (int) ((free_b - need_w) >> 20));
                }
            } else if (fp8) {
                cudaFree(fp8);
            }
        } else {
            static int stopped = 0;
            if (stopped++ == 0) {
                fprintf(stderr, "allpaka_ggml: fp8 cache full %d tensors %d MiB free %d MiB\n",
                        n_cached, (int) (cached_bytes >> 20), (int) (free_b >> 20));
            }
        }
    }
    cudaStream_t qs = stream;
    int gemm_ev = -1;
    int slot = 0;
    const bool overlap = !cached && !staging && need_w >= (64ull << 20) && nslots >= 2 && qev_i + 4 < qev_n;
    if (overlap) {
        slot = slot_i % nslots;
        slot_i++;
        w8 = w8s[slot];
        w_scale = w_scale2[slot];
        if (fork_ev < 0) {
            cudaEventRecord(qev[qev_i], stream);
            cudaStreamWaitEvent(qstream, qev[qev_i], 0);
            qev_i++;
        } else {
            cudaStreamWaitEvent(qstream, qev[fork_ev], 0);
        }
        if (slot_used[slot]) {
            cudaStreamWaitEvent(qstream, qev[slot_gemm_ev[slot]], 0);
        }
        qs = qstream;
    }
    const bool time_this = !capturing && !cached && m >= 400 && nw >= 10000000 && timed < 3;
    if (!cached) {
    if (time_this) {
        cudaEventRecord(ev0, qs);
    }
    if (w_type == GGML_TYPE_Q4_K) {
        k_q4k_amax<<<k_blocks, 256, 0, qs>>>((const block_q4_K *) w_q, nblocks, qpartial);
        k_reduce_max<<<1, 128, 0, qs>>>(qpartial, k_blocks, w_scale);
        k_q4k_to_fp8<<<2048, 256, 0, qs>>>((const block_q4_K *) w_q, (__nv_fp8_e4m3 *) w8, nblocks, w_scale);
    } else {
        k_q6k_amax<<<k_blocks, 256, 0, qs>>>((const block_q6_K *) w_q, nblocks, qpartial);
        k_reduce_max<<<1, 128, 0, qs>>>(qpartial, k_blocks, w_scale);
        k_q6k_to_fp8<<<4096, 64, 0, qs>>>((const block_q6_K *) w_q, (__nv_fp8_e4m3 *) w8, nblocks, w_scale);
    }
    k_scale_div448<<<1, 1, 0, qs>>>(w_scale);
    }
    if (qs != stream) {
        cudaEventRecord(qev[qev_i], qs);
        cudaStreamWaitEvent(stream, qev[qev_i], 0);
        qev_i++;
        gemm_ev = qev_i;
        qev_i++;
    }
    if (qev_i + 1 < qev_n) {
        fork_ev = qev_i;
        cudaEventRecord(qev[fork_ev], stream);
        qev_i++;
    }
    if (time_this) {
        cudaEventRecord(ev1, stream);
    }

    __nv_fp8_e4m3 * x8 = static_cast<__nv_fp8_e4m3 *>(scratch);
    half * y16 = reinterpret_cast<half *>(x8 + x_pad);
    float * b_scale = amax + 1;
    k_amax_f32<<<k_blocks, 256, 0, stream>>>(x_f32, partial, (int) x_n);
    k_reduce_max<<<1, 128, 0, stream>>>(partial, k_blocks, amax);
    k_q_f32_e4m3<<<k_blocks, 256, 0, stream>>>(x_f32, x8, (int) x_n, amax, b_scale);
    if (m_gemm > m) {
        cudaMemsetAsync(x8 + x_n, 0, x_pad - x_n, stream);
    }

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return -8;
    }
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scale, sizeof(w_scale));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale));
    const char * fe = std::getenv("ALLPAKA_FP8_FAST");
    const int8_t fast = (fe && fe[0] == '0') ? 0 : 1;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast));
    cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, n_in, n_out, n_in);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, n_in, m_gemm, n_in);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, n_out, m_gemm, n_out);

    Key key{n_in, n_out, m_gemm};
    auto it = algos.find(key);
    if (it == algos.end()) {
        if (capturing) {
            cublasLtMatmulDescDestroy(desc);
            cublasLtMatrixLayoutDestroy(a);
            cublasLtMatrixLayoutDestroy(b);
            cublasLtMatrixLayoutDestroy(c);
            return -9;
        }
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &k_ws, sizeof(k_ws));
        cublasLtMatmulHeuristicResult_t heurs[32]{};
        int n_found = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            lt, desc, a, b, c, c, pref, 32, heurs, &n_found);
        cublasLtMatmulPreferenceDestroy(pref);
        if (hs != CUBLAS_STATUS_SUCCESS || n_found < 1) {
            cublasLtMatmulDescDestroy(desc);
            cublasLtMatrixLayoutDestroy(a);
            cublasLtMatrixLayoutDestroy(b);
            cublasLtMatrixLayoutDestroy(c);
            static int bad = 0;
            if (bad < 2) {
                fprintf(stderr, "allpaka_ggml: fp8q heuristic failed %d n_in=%d n_out=%d m=%d\n",
                        (int) hs, n_in, n_out, m);
                bad++;
            }
            return -9;
        }
        int best_i = 0;
        float best_ms = 1e30f;
        const float alpha_try = 1.f;
        const float beta_try = 0.f;
        const bool tune = m >= 400 && nw >= 10000000;
        if (tune) {
            for (int spin = 0; spin < 30; ++spin) {
                cublasLtMatmul(
                    lt, desc, &alpha_try, w8, a, x8, b, &beta_try, y16, c, y16, c,
                    &heurs[0].algo, workspace, k_ws, stream);
            }
            cudaStreamSynchronize(stream);
        }
        for (int i = 0; i < n_found; ++i) {
            float ms = 1e30f;
            const int reps = tune ? 2 : 1;
            for (int rep = 0; rep < reps; ++rep) {
                cudaEventRecord(ev1, stream);
                cublasStatus_t ts = cublasLtMatmul(
                    lt, desc, &alpha_try, w8, a, x8, b, &beta_try, y16, c, y16, c,
                    &heurs[i].algo, workspace, k_ws, stream);
                cudaEventRecord(ev2, stream);
                cudaEventSynchronize(ev2);
                float one = 1e30f;
                if (ts == CUBLAS_STATUS_SUCCESS) {
                    cudaEventElapsedTime(&one, ev1, ev2);
                }
                if (one < ms) {
                    ms = one;
                }
            }
            if (ms < best_ms) {
                best_ms = ms;
                best_i = i;
            }
        }
        it = algos.emplace(key, heurs[best_i].algo).first;
        if (tune) {
            fprintf(stderr, "allpaka_ggml: fp8q algo n_in=%d n_out=%d m=%d pad=%d pick=%d/%d %.0f us\n",
                    n_in, n_out, m, m_gemm, best_i, n_found, best_ms * 1000.f);
        }
    }

    const float alpha = 1.f;
    const float beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(
        lt, desc,
        &alpha, w8, a,
        x8, b,
        &beta, y16, c,
        y16, c,
        &it->second, workspace, k_ws, stream);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    if (st != CUBLAS_STATUS_SUCCESS) {
        static int bad = 0;
        if (bad < 2) {
            fprintf(stderr, "allpaka_ggml: fp8q matmul failed %d\n", (int) st);
            bad++;
        }
        return -10;
    }
    if (gemm_ev >= 0) {
        cudaEventRecord(qev[gemm_ev], stream);
        slot_gemm_ev[slot] = gemm_ev;
        slot_used[slot] = true;
    }
    k_f16_to_f32<<<(y_n + 255) / 256, 256, 0, stream>>>(y16, y_f32, (int) y_n);
    if (time_this) {
        cudaEventRecord(ev2, stream);
        cudaEventSynchronize(ev2);
        float qms = 0.f, gms = 0.f;
        cudaEventElapsedTime(&qms, ev0, ev1);
        cudaEventElapsedTime(&gms, ev1, ev2);
        fprintf(stderr, "allpaka_ggml: fp8q n_in=%d n_out=%d m=%d quant=%.0f us gemm=%.0f us\n",
                n_in, n_out, m, qms * 1000.f, gms * 1000.f);
        timed++;
    }
    return 0;
}
