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
