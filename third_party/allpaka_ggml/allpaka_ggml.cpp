// Thin ggml-cuda bridge: zero-copy device views; shape-cached graphs; CUDA graphs disabled.

#define GGML_BACKEND_SHARED
#define ALLPAKA_GGML_BUILD

#include "allpaka_ggml.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" void allpaka_fa_permute_launch(
    const float * in, float * out, int hd, int n_q, int m, cudaStream_t stream);

extern "C" void allpaka_fa_permute_q8_launch(
    const float * in, float * out, void * yq, int hd, int n_q, cudaStream_t stream);

extern "C" void allpaka_fa_mask_from_pos_launch(
    void * mask, int n_kv, const unsigned int * d_pos, cudaStream_t stream);

// Optional: skip DLL permute; Rust NVRTC does permute+Q8 (cudarc-owned buffers).
static void * g_fa_q8 = nullptr; // non-null => skip permute this call
static void * g_fa_src = nullptr;
static int g_fa_q8_used = 0;

extern "C" void allpaka_fa_set_q8(void * q8_dev) {
    g_fa_q8 = q8_dev;
    g_fa_q8_used = 0;
    g_fa_src = nullptr;
}

extern "C" int allpaka_fa_q8_consumed(void) {
    int u = g_fa_q8_used;
    g_fa_q8_used = 0;
    g_fa_q8 = nullptr;
    return u;
}

extern "C" void * allpaka_fa_last_src(void) {
    return g_fa_src;
}

static void fa_permute_out(
    const float * in, float * out, int hd, int n_q, int m, cudaStream_t st)
{
    if (g_fa_q8 && m == 1) {
        // Leave FA in `in` (fa_dev); Rust launches permute_q8 into out + q8.
        (void)out;
        (void)hd;
        (void)n_q;
        (void)st;
        g_fa_src = const_cast<float *>(in);
        g_fa_q8_used = 1;
        g_fa_q8 = nullptr;
        return;
    }
    allpaka_fa_permute_launch(in, out, hd, n_q, m, st);
}

extern "C" cudaStream_t allpaka_peer_stream(void);

extern "C" int allpaka_dequant_f16(
    void * cuda_ctx_v, const void * src, void * dst, int64_t n_elem, int w_type);

extern "C" int allpaka_lt_gemm(
    void * cuda_ctx_v, const void * w_f16, const float * x_f32, float * y_f32,
    int n_in, int n_out, int m);

extern "C" int allpaka_fp8_gemm(
    void * cuda_ctx_v, const void * w_f16, const float * x_f32, float * y_f32,
    int n_in, int n_out, int m);

extern "C" int allpaka_fp8_gemm_q(
    void * cuda_ctx_v, const void * w_q, const float * x_f32, float * y_f32,
    int n_in, int n_out, int m, int w_type);

extern "C" int allpaka_fa_launch_direct(void * fa_tensor_v);

namespace {

std::mutex g_mu;
ggml_backend_t g_be = nullptr;
int g_device = 0;

static bool f16_weights_on() {
    const char * e = std::getenv("ALLPAKA_F16_W");
    if (!e || !e[0]) {
        return true;
    }
    return !(e[0] == '0' || e[0] == 'f' || e[0] == 'F');
}

// One-time Q4/Q6 -> f16. Prefill otherwise dequants the same weights on every GEMM.
static std::unordered_map<const void *, void *> g_f16_w;

static const void * f16_weight(const void * w, int32_t n_in, int32_t n_out, int32_t w_type, int32_t * ty_io) {
    if (!f16_weights_on() || !g_be) {
        return w;
    }
    auto it = g_f16_w.find(w);
    if (it != g_f16_w.end()) {
        *ty_io = GGML_TYPE_F16;
        return it->second;
    }
    const int64_t n = (int64_t) n_in * (int64_t) n_out;
    if (n <= 0) {
        return w;
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        return w;
    }
    const size_t need = (size_t) n * sizeof(uint16_t);
    if (free_b < need + (3ull << 30)) {
        return w;
    }
    void * dst = nullptr;
    if (cudaMalloc(&dst, need) != cudaSuccess) {
        return w;
    }
    if (allpaka_dequant_f16(g_be->context, w, dst, n, w_type) != 0) {
        cudaFree(dst);
        return w;
    }
    g_f16_w.emplace(w, dst);
    static int logged = 0;
    if (logged < 3) {
        std::fprintf(stderr, "allpaka_ggml: f16 weight %d MiB (cached %d)\n",
                     (int) (need >> 20), (int) g_f16_w.size());
        logged++;
    }
    *ty_io = GGML_TYPE_F16;
    return dst;
}

struct ViewCtx {
    void * ptr;
};

struct ShapeKey {
    int32_t n_in, n_out, m, w_type;
    bool operator==(const ShapeKey & o) const {
        return n_in == o.n_in && n_out == o.n_out && m == o.m && w_type == o.w_type;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey & k) const {
        size_t h = (size_t)k.n_in;
        h = h * 1315423911u + (size_t)k.n_out;
        h = h * 1315423911u + (size_t)k.m;
        h = h * 1315423911u + (size_t)k.w_type;
        return h;
    }
};

struct Slot {
    ggml_context * ctx = nullptr;
    ggml_tensor * W = nullptr;
    ggml_tensor * X = nullptr;
    ggml_tensor * Y = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_x = nullptr;
    ggml_backend_buffer_t buf_y = nullptr;
    ViewCtx * vw = nullptr;
    ViewCtx * vx = nullptr;
    ViewCtx * vy = nullptr;
};

std::unordered_map<ShapeKey, Slot, ShapeKeyHash> g_slots;

static void view_free_buffer(ggml_backend_buffer_t buffer) { (void)buffer; }
static void * view_get_base(ggml_backend_buffer_t buffer) {
    return static_cast<ViewCtx *>(buffer->context)->ptr;
}
static enum ggml_status view_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}
static void view_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor,
                            const void * data, size_t offset, size_t size) {
    cudaMemcpy(static_cast<char *>(tensor->data) + offset, data, size, cudaMemcpyHostToDevice);
}
static void view_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor,
                            void * data, size_t offset, size_t size) {
    cudaMemcpy(data, static_cast<const char *>(tensor->data) + offset, size, cudaMemcpyDeviceToHost);
}
static void view_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor,
                               uint8_t value, size_t offset, size_t size) {
    cudaMemset(static_cast<char *>(tensor->data) + offset, value, size);
}
static void view_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<ViewCtx *>(buffer->context);
    cudaMemset(ctx->ptr, value, buffer->size);
}

static const ggml_backend_buffer_i view_iface = {
    view_free_buffer, view_get_base, view_init_tensor, view_memset_tensor,
    view_set_tensor, view_get_tensor, nullptr, nullptr, nullptr, view_clear, nullptr,
};

static ggml_backend_buffer_t make_view_buffer(ViewCtx * ctx, size_t size) {
    return ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(g_device), view_iface, ctx, size);
}

static void free_slot(Slot & s) {
    if (s.buf_w) { ggml_backend_buffer_free(s.buf_w); s.buf_w = nullptr; }
    if (s.buf_x) { ggml_backend_buffer_free(s.buf_x); s.buf_x = nullptr; }
    if (s.buf_y) { ggml_backend_buffer_free(s.buf_y); s.buf_y = nullptr; }
    delete s.vw; s.vw = nullptr;
    delete s.vx; s.vx = nullptr;
    delete s.vy; s.vy = nullptr;
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    s.W = s.X = s.Y = nullptr;
    s.gf = nullptr;
}

static Slot * get_or_create_slot(const ShapeKey & key) {
    auto it = g_slots.find(key);
    if (it != g_slots.end()) {
        return &it->second;
    }

    enum ggml_type ty = static_cast<enum ggml_type>(key.w_type);
    Slot s;
    struct ggml_init_params params = {
        ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        nullptr,
        true,
    };
    s.ctx = ggml_init(params);
    if (!s.ctx) {
        return nullptr;
    }
    s.W = ggml_new_tensor_2d(s.ctx, ty, key.n_in, key.n_out);
    s.X = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, key.n_in, key.m);
    s.Y = ggml_mul_mat(s.ctx, s.W, s.X);

    const size_t w_bytes = ggml_row_size(ty, key.n_in) * (size_t)key.n_out;
    const size_t x_bytes = sizeof(float) * (size_t)key.n_in * (size_t)key.m;
    const size_t y_bytes = sizeof(float) * (size_t)key.n_out * (size_t)key.m;

    s.vw = new ViewCtx{nullptr};
    s.vx = new ViewCtx{nullptr};
    s.vy = new ViewCtx{nullptr};
    s.buf_w = make_view_buffer(s.vw, w_bytes);
    s.buf_x = make_view_buffer(s.vx, x_bytes);
    s.buf_y = make_view_buffer(s.vy, y_bytes);
    if (!s.buf_w || !s.buf_x || !s.buf_y) {
        free_slot(s);
        return nullptr;
    }

    s.W->buffer = s.buf_w; s.W->data = nullptr;
    s.X->buffer = s.buf_x; s.X->data = nullptr;
    s.Y->buffer = s.buf_y; s.Y->data = nullptr;
    ggml_backend_buffer_init_tensor(s.buf_w, s.W);
    ggml_backend_buffer_init_tensor(s.buf_x, s.X);
    ggml_backend_buffer_init_tensor(s.buf_y, s.Y);

    s.gf = ggml_new_graph(s.ctx);
    ggml_build_forward_expand(s.gf, s.Y);

    auto [ins, ok] = g_slots.emplace(key, s);
    (void)ok;
    std::fprintf(stderr, "allpaka_ggml: slot n_in=%d n_out=%d m=%d type=%d\n",
                 key.n_in, key.n_out, key.m, key.w_type);
    return &ins->second;
}

struct FaKey {
    int32_t head_dim, n_q, n_kv_h, m, n_kv, kv_dim;
    uint32_t scale_bits;
    bool operator==(const FaKey & o) const {
        return head_dim == o.head_dim && n_q == o.n_q && n_kv_h == o.n_kv_h
            && m == o.m && n_kv == o.n_kv && kv_dim == o.kv_dim && scale_bits == o.scale_bits;
    }
};

struct FaKeyHash {
    size_t operator()(const FaKey & k) const {
        size_t h = (size_t)k.head_dim;
        h = h * 1315423911u + (size_t)k.n_q;
        h = h * 1315423911u + (size_t)k.n_kv_h;
        h = h * 1315423911u + (size_t)k.m;
        h = h * 1315423911u + (size_t)k.n_kv;
        h = h * 1315423911u + (size_t)k.kv_dim;
        h = h * 1315423911u + (size_t)k.scale_bits;
        return h;
    }
};

struct FaSlot {
    ggml_context * ctx = nullptr;
    ggml_tensor * Q = nullptr;
    ggml_tensor * K = nullptr;
    ggml_tensor * V = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * fa = nullptr;
    ggml_tensor * perm = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf_q = nullptr;
    ggml_backend_buffer_t buf_k = nullptr;
    ggml_backend_buffer_t buf_v = nullptr;
    ggml_backend_buffer_t buf_m = nullptr;
    ggml_backend_buffer_t buf_fa = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;
    ggml_backend_buffer_t buf_o = nullptr;
    ViewCtx * vq = nullptr;
    ViewCtx * vk = nullptr;
    ViewCtx * vv = nullptr;
    ViewCtx * vm = nullptr;
    ViewCtx * vfa = nullptr;
    ViewCtx * vperm = nullptr;
    ViewCtx * vo = nullptr;
    void * mask_dev = nullptr;
    void * fa_dev = nullptr;
    void * perm_dev = nullptr;
    size_t mask_bytes = 0;
    size_t fa_bytes = 0;
    int32_t n_kv = 0;
    int32_t m = 0;
    int32_t mask_lim = -1; // decode: last causal limit written into mask
};

std::unordered_map<FaKey, FaSlot, FaKeyHash> g_fa;
// Decode hot path: last m=1 slot (shared across layers).
static FaKey g_fa_decode_key{};
static FaSlot * g_fa_decode_slot = nullptr;
static bool g_fa_decode_valid = false;

static void free_fa_slot(FaSlot & s) {
    if (s.buf_q) { ggml_backend_buffer_free(s.buf_q); s.buf_q = nullptr; }
    if (s.buf_k) { ggml_backend_buffer_free(s.buf_k); s.buf_k = nullptr; }
    if (s.buf_v) { ggml_backend_buffer_free(s.buf_v); s.buf_v = nullptr; }
    if (s.buf_m) { ggml_backend_buffer_free(s.buf_m); s.buf_m = nullptr; }
    if (s.buf_fa) { ggml_backend_buffer_free(s.buf_fa); s.buf_fa = nullptr; }
    if (s.buf_perm) { ggml_backend_buffer_free(s.buf_perm); s.buf_perm = nullptr; }
    if (s.buf_o) { ggml_backend_buffer_free(s.buf_o); s.buf_o = nullptr; }
    delete s.vq; s.vq = nullptr;
    delete s.vk; s.vk = nullptr;
    delete s.vv; s.vv = nullptr;
    delete s.vm; s.vm = nullptr;
    delete s.vfa; s.vfa = nullptr;
    delete s.vperm; s.vperm = nullptr;
    delete s.vo; s.vo = nullptr;
    if (s.mask_dev) { cudaFree(s.mask_dev); s.mask_dev = nullptr; }
    if (s.fa_dev) { cudaFree(s.fa_dev); s.fa_dev = nullptr; }
    if (s.perm_dev) { cudaFree(s.perm_dev); s.perm_dev = nullptr; }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    s.Q = s.K = s.V = s.mask = s.fa = s.perm = s.out = nullptr;
    s.gf = nullptr;
}

static void fill_causal_mask(void * mask_dev, int32_t n_kv, int32_t m, int32_t base) {
    std::vector<ggml_fp16_t> host((size_t)n_kv * (size_t)m);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg = ggml_fp32_to_fp16(-INFINITY);
    for (int32_t t = 0; t < m; t++) {
        const int32_t lim = base + t;
        for (int32_t p = 0; p < n_kv; p++) {
            host[(size_t)t * (size_t)n_kv + (size_t)p] = (p <= lim) ? zero : neg;
        }
    }
    cudaMemcpy(mask_dev, host.data(), host.size() * sizeof(ggml_fp16_t), cudaMemcpyHostToDevice);
}

// Decode m=1: unmask newly visible KV positions (mask_lim+1 .. base) with a
// tiny H2D instead of rewriting the whole capacity row.
static void unmask_causal_range(void * mask_dev, int32_t from, int32_t to_inclusive) {
    if (from > to_inclusive) {
        return;
    }
    const int32_t n = to_inclusive - from + 1;
    std::vector<ggml_fp16_t> host((size_t)n, ggml_fp32_to_fp16(0.0f));
    cudaMemcpy(
        static_cast<ggml_fp16_t *>(mask_dev) + from,
        host.data(),
        (size_t)n * sizeof(ggml_fp16_t),
        cudaMemcpyHostToDevice);
}

static FaSlot * get_or_create_fa(const FaKey & key, float scale) {
    auto it = g_fa.find(key);
    if (it != g_fa.end()) {
        return &it->second;
    }

    const int32_t hd = key.head_dim;
    const int32_t n_q = key.n_q;
    const int32_t n_kv_h = key.n_kv_h;
    const int32_t m = key.m;
    const int32_t kv_dim = key.kv_dim;
    const int32_t n_kv = key.n_kv;
    if (n_q % n_kv_h != 0 || n_kv <= 0 || m <= 0 || kv_dim < n_kv_h * hd) {
        return nullptr;
    }

    FaSlot s;
    struct ggml_init_params params = {
        ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        nullptr,
        true,
    };
    s.ctx = ggml_init(params);
    if (!s.ctx) {
        return nullptr;
    }

    // Q storage [m][n_q][hd] F32 -> logical [hd, m, n_q, 1]
    s.Q = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F32, hd, m, n_q, 1);
    s.Q->nb[0] = sizeof(float);
    s.Q->nb[1] = (size_t)n_q * (size_t)hd * sizeof(float);
    s.Q->nb[2] = (size_t)hd * sizeof(float);
    s.Q->nb[3] = (size_t)m * (size_t)n_q * (size_t)hd * sizeof(float);

    // K/V [n_kv][n_kv_h][hd] F16, pos stride kv_dim -> [hd, n_kv, n_kv_h, 1]
    s.K = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16, hd, n_kv, n_kv_h, 1);
    s.K->nb[0] = sizeof(ggml_fp16_t);
    s.K->nb[1] = (size_t)kv_dim * sizeof(ggml_fp16_t);
    s.K->nb[2] = (size_t)hd * sizeof(ggml_fp16_t);
    s.K->nb[3] = (size_t)n_kv * (size_t)kv_dim * sizeof(ggml_fp16_t);

    s.V = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16, hd, n_kv, n_kv_h, 1);
    s.V->nb[0] = sizeof(ggml_fp16_t);
    s.V->nb[1] = (size_t)kv_dim * sizeof(ggml_fp16_t);
    s.V->nb[2] = (size_t)hd * sizeof(ggml_fp16_t);
    s.V->nb[3] = (size_t)n_kv * (size_t)kv_dim * sizeof(ggml_fp16_t);

    s.mask = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16, n_kv, m, 1, 1);

    s.mask_bytes = (size_t)n_kv * (size_t)m * sizeof(ggml_fp16_t);
    s.fa_bytes = (size_t)hd * (size_t)n_q * (size_t)m * sizeof(float);
    s.n_kv = n_kv;
    s.m = m;
    if (cudaMalloc(&s.mask_dev, s.mask_bytes) != cudaSuccess ||
        cudaMalloc(&s.fa_dev, s.fa_bytes) != cudaSuccess) {
        free_fa_slot(s);
        return nullptr;
    }
    // Prefill needs a permute scratch; decode permutes with a direct kernel.

    // Prefill: exact causal. Decode capacity slot: filled on first use.
    if (m > 1) {
        fill_causal_mask(s.mask_dev, n_kv, m, /*base=*/n_kv - m);
        s.mask_lim = n_kv - 1;
    }

    s.fa = ggml_flash_attn_ext(s.ctx, s.Q, s.K, s.V, s.mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(s.fa, GGML_PREC_F32);

    const size_t q_bytes = (size_t)m * (size_t)n_q * (size_t)hd * sizeof(float);
    const size_t kv_bytes = (size_t)n_kv * (size_t)kv_dim * sizeof(ggml_fp16_t);

    s.vq = new ViewCtx{nullptr};
    s.vk = new ViewCtx{nullptr};
    s.vv = new ViewCtx{nullptr};
    s.vm = new ViewCtx{s.mask_dev};
    s.vfa = new ViewCtx{s.fa_dev};
    s.vo = new ViewCtx{nullptr};
    s.buf_q = make_view_buffer(s.vq, q_bytes);
    s.buf_k = make_view_buffer(s.vk, kv_bytes);
    s.buf_v = make_view_buffer(s.vv, kv_bytes);
    s.buf_m = make_view_buffer(s.vm, s.mask_bytes);
    s.buf_fa = make_view_buffer(s.vfa, s.fa_bytes);
    s.buf_o = make_view_buffer(s.vo, s.fa_bytes);
    if (!s.buf_q || !s.buf_k || !s.buf_v || !s.buf_m || !s.buf_fa || !s.buf_o) {
        free_fa_slot(s);
        return nullptr;
    }

    s.Q->buffer = s.buf_q; s.Q->data = nullptr;
    s.K->buffer = s.buf_k; s.K->data = nullptr;
    s.V->buffer = s.buf_v; s.V->data = nullptr;
    s.mask->buffer = s.buf_m; s.mask->data = s.mask_dev;
    s.fa->buffer = s.buf_fa; s.fa->data = s.fa_dev;
    ggml_backend_buffer_init_tensor(s.buf_q, s.Q);
    ggml_backend_buffer_init_tensor(s.buf_k, s.K);
    ggml_backend_buffer_init_tensor(s.buf_v, s.V);
    ggml_backend_buffer_init_tensor(s.buf_m, s.mask);
    ggml_backend_buffer_init_tensor(s.buf_fa, s.fa);

    s.gf = ggml_new_graph(s.ctx);
    // FA + permute/cont in one graph (decode and prefill). Decode still
    // overwrites out->data with the caller's out_dev each launch.
    s.perm = ggml_permute(s.ctx, s.fa, 2, 1, 0, 3);
    s.out = ggml_cont_4d(s.ctx, s.perm, m, n_q, hd, 1);
    if (cudaMalloc(&s.perm_dev, s.fa_bytes) != cudaSuccess) {
        free_fa_slot(s);
        return nullptr;
    }
    s.vperm = new ViewCtx{s.perm_dev};
    s.buf_perm = make_view_buffer(s.vperm, s.fa_bytes);
    if (!s.buf_perm) {
        free_fa_slot(s);
        return nullptr;
    }
    s.perm->buffer = s.buf_perm; s.perm->data = s.perm_dev;
    s.out->buffer = s.buf_o; s.out->data = nullptr;
    ggml_backend_buffer_init_tensor(s.buf_perm, s.perm);
    ggml_backend_buffer_init_tensor(s.buf_o, s.out);
    ggml_build_forward_expand(s.gf, s.out);

    auto [ins, ok] = g_fa.emplace(key, s);
    (void)ok;
    std::fprintf(stderr, "allpaka_ggml: flash_attn hd=%d n_q=%d n_kv_h=%d m=%d n_kv=%d\n",
                 hd, n_q, n_kv_h, m, n_kv);
    return &ins->second;
}

} // namespace

extern "C" int allpaka_ggml_init(int device) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_be) {
        return 0;
    }
    // Default: disable ggml CUDA graphs (safer with peer stream). Opt-in via
    // ALLPAKA_GGML_GRAPHS=1 for FA/MMQ graph capture inside ggml.
    const char * graphs = std::getenv("ALLPAKA_GGML_GRAPHS");
    const bool want_graphs = graphs && (graphs[0] == '1' || graphs[0] == 't' || graphs[0] == 'T');
    if (!want_graphs) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif
    }
    g_device = device;
    g_be = ggml_backend_cuda_init(device);
    if (!g_be) {
        std::fprintf(stderr, "allpaka_ggml: ggml_backend_cuda_init(%d) failed\n", device);
        return -1;
    }
    std::fprintf(stderr, "allpaka_ggml: ready (device %d, graphs %s)\n",
                 device, want_graphs ? "enabled" : "disabled");
    extern void allpaka_peer_set_ctx(void *);
    allpaka_peer_set_ctx(g_be->context);
    return 0;
}

extern "C" void allpaka_ggml_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    extern void allpaka_peer_clear_shared(void);
    allpaka_peer_clear_shared();
    for (auto & kv : g_slots) {
        free_slot(kv.second);
    }
    g_slots.clear();
    for (auto & kv : g_fa) {
        free_fa_slot(kv.second);
    }
    g_fa.clear();
    g_fa_decode_valid = false;
    g_fa_decode_slot = nullptr;
    if (g_be) {
        ggml_backend_free(g_be);
        g_be = nullptr;
    }
}

extern "C" void allpaka_ggml_sync(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_be) {
        ggml_backend_synchronize(g_be);
    }
}

extern "C" int allpaka_ggml_bind_peer_stream(void * peer_stream) {
    extern int allpaka_peer_bind(void *);
    return allpaka_peer_bind(peer_stream);
}

extern "C" void allpaka_ggml_wait_peer(void) {
    extern void allpaka_peer_wait(void);
    allpaka_peer_wait();
}

extern "C" void allpaka_ggml_signal_peer(void) {
    extern void allpaka_peer_signal(void);
    allpaka_peer_signal();
}

extern "C" int allpaka_ggml_peer_is_shared(void) {
    extern int allpaka_peer_is_shared(void);
    return allpaka_peer_is_shared();
}

extern "C" void allpaka_ggml_clear_shared(void) {
    extern void allpaka_peer_clear_shared(void);
    allpaka_peer_clear_shared();
}

extern "C" void * allpaka_ggml_cuda_stream(void) { return nullptr; }
extern "C" void * allpaka_ggml_done_event(void) { return nullptr; }

extern "C" int allpaka_ggml_mul_mat_vec(
    const void * w_dev,
    const float * x_dev,
    float * y_dev,
    int32_t n_in,
    int32_t n_out,
    int32_t w_type)
{
    if (!w_dev || !x_dev || !y_dev || n_in <= 0 || n_out <= 0) {
        return -2;
    }
    if (allpaka_ggml_init(g_device) != 0) {
        return -1;
    }
    // Direct llama MMVQ (exported from ggml-cuda.dll) — no cgraph host tax.
    extern int allpaka_direct_mmvq(
        void * cuda_ctx_v,
        const void * w_dev,
        const float * x_dev,
        float * y_dev,
        int32_t n_in,
        int32_t n_out,
        int32_t w_type);
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_be || !g_be->context) {
        return -1;
    }
    return allpaka_direct_mmvq(g_be->context, w_dev, x_dev, y_dev, n_in, n_out, w_type);
}

extern "C" int allpaka_ggml_mul_mat(
    const void * w_dev,
    const float * x_dev,
    float * y_dev,
    int32_t n_in,
    int32_t n_out,
    int32_t m,
    int32_t w_type)
{
    if (!w_dev || !x_dev || !y_dev || n_in <= 0 || n_out <= 0 || m <= 0) {
        return -2;
    }
    if (m > 1 && m < 32) {
        return -10;
    }
    if (allpaka_ggml_init(g_device) != 0) {
        return -1;
    }

    enum ggml_type ty = static_cast<enum ggml_type>(w_type);
    if (ty != GGML_TYPE_Q4_K && ty != GGML_TYPE_Q6_K) {
        return -3;
    }
    if (n_in % ggml_blck_size(ty) != 0) {
        return -4;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    const char * fp8_first = std::getenv("ALLPAKA_FP8");
    if (fp8_first && fp8_first[0] == '1' && m >= 32 &&
        allpaka_fp8_gemm_q(g_be->context, w_dev, x_dev, y_dev, n_in, n_out, m, w_type) == 0) {
        return 0;
    }
    int32_t ty_use = w_type;
    const void * w_use = f16_weight(w_dev, n_in, n_out, w_type, &ty_use);
    ShapeKey key{n_in, n_out, m, ty_use};
    Slot * slot = get_or_create_slot(key);
    if (!slot) {
        return -5;
    }

    if (ty_use == GGML_TYPE_F16) {
        const char * fp8 = std::getenv("ALLPAKA_FP8");
        if (fp8 && fp8[0] == '1' &&
            allpaka_fp8_gemm(g_be->context, w_use, x_dev, y_dev, n_in, n_out, m) == 0) {
            return 0;
        }
        const char * lt = std::getenv("ALLPAKA_LT");
        const bool lt_on = !lt || !lt[0] || !(lt[0] == '0' || lt[0] == 'f' || lt[0] == 'F');
        if (lt_on && allpaka_lt_gemm(g_be->context, w_use, x_dev, y_dev, n_in, n_out, m) == 0) {
            return 0;
        }
    }

    slot->vw->ptr = const_cast<void *>(w_use);
    slot->vx->ptr = const_cast<float *>(x_dev);
    slot->vy->ptr = y_dev;
    slot->W->data = const_cast<void *>(w_use);
    slot->X->data = const_cast<float *>(x_dev);
    slot->Y->data = y_dev;

    enum ggml_status st = ggml_backend_graph_compute_async(g_be, slot->gf);
    return st == GGML_STATUS_SUCCESS ? 0 : -7;
}

extern "C" void allpaka_fa_reset_mask_lim(void) {
    if (g_fa_decode_slot) {
        g_fa_decode_slot->mask_lim = -1;
    }
}

extern "C" int allpaka_ggml_flash_attn(
    const float * q_dev,
    const void * k_dev,
    const void * v_dev,
    float * out_dev,
    int32_t head_dim,
    int32_t n_q_heads,
    int32_t n_kv_heads,
    int32_t m,
    int32_t base,
    int32_t kv_dim,
    float scale,
    const uint32_t * d_pos_dev)
{
    if (!q_dev || !k_dev || !v_dev || !out_dev) {
        return -2;
    }
    if (head_dim <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 || m < 1 || base < 0 || kv_dim <= 0) {
        return -3;
    }
    if (n_q_heads % n_kv_heads != 0) {
        return -4;
    }
    // Hot path: skip re-init and avoid holding the global mutex across GPU work.
    // Slots are created under the lock once; compute is single-stream decode.
    if (!g_be) {
        if (allpaka_ggml_init(g_device) != 0) {
            return -1;
        }
    }

    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));

    // Decode (m=1): power-of-two capacity slot. Prefill: exact n_kv = base + m.
    // Default floor 256. Override with ALLPAKA_FA_KV_FLOOR (e.g. 32/64) for A/B.
    const int32_t need = (m == 1) ? (base + 1) : (base + m);
    int32_t n_kv;
    if (m == 1) {
        int32_t floor_kv = 256;
        if (const char * fl = std::getenv("ALLPAKA_FA_KV_FLOOR")) {
            int v = std::atoi(fl);
            if (v >= 16 && v <= 8192) {
                floor_kv = v;
            }
        }
        n_kv = floor_kv;
        while (n_kv < need) {
            n_kv *= 2;
        }
        if (n_kv > 8192) {
            return -6;
        }
    } else {
        n_kv = need;
    }
    FaKey key{head_dim, n_q_heads, n_kv_heads, m, n_kv, kv_dim, scale_bits};

    FaSlot * slot = nullptr;
    if (m == 1 && g_fa_decode_valid && g_fa_decode_key == key && g_fa_decode_slot) {
        slot = g_fa_decode_slot;
    } else {
        std::lock_guard<std::mutex> lock(g_mu);
        slot = get_or_create_fa(key, scale);
        if (slot && m == 1) {
            g_fa_decode_key = key;
            g_fa_decode_slot = slot;
            g_fa_decode_valid = true;
        }
    }
    if (!slot) {
        return -5;
    }

    cudaStream_t st = allpaka_peer_stream();

    if (m == 1 && d_pos_dev) {
        // Once per token: shared decode slot. Subsequent layers skip.
        if (slot->mask_lim != base) {
            allpaka_fa_mask_from_pos_launch(slot->mask_dev, slot->n_kv, d_pos_dev, st);
            slot->mask_lim = base;
        }
    } else if (m == 1) {
        if (slot->mask_lim < 0) {
            fill_causal_mask(slot->mask_dev, slot->n_kv, 1, base);
            slot->mask_lim = base;
        } else if (base > slot->mask_lim) {
            unmask_causal_range(slot->mask_dev, slot->mask_lim + 1, base);
            slot->mask_lim = base;
        } else if (base < slot->mask_lim) {
            fill_causal_mask(slot->mask_dev, slot->n_kv, 1, base);
            slot->mask_lim = base;
        }
    }

    slot->vq->ptr = const_cast<float *>(q_dev);
    slot->vk->ptr = const_cast<void *>(k_dev);
    slot->vv->ptr = const_cast<void *>(v_dev);
    slot->Q->data = const_cast<float *>(q_dev);
    slot->K->data = const_cast<void *>(k_dev);
    slot->V->data = const_cast<void *>(v_dev);
    slot->mask->data = slot->mask_dev;
    slot->fa->data = slot->fa_dev;

    if (m == 1) {
        slot->vo->ptr = out_dev;
        slot->out->data = out_dev;
        // Default: direct fattn + host permute (best FA-inline wait so far).
        // ALLPAKA_FA_GRAPH_COMPUTE=1 uses fused FA+permute ggml graph.
        const char * gcomp = std::getenv("ALLPAKA_FA_GRAPH_COMPUTE");
        const bool want_gcomp = gcomp && (gcomp[0] == '1' || gcomp[0] == 't' || gcomp[0] == 'T');
        if (want_gcomp) {
            enum ggml_status stc = ggml_backend_graph_compute_async(g_be, slot->gf);
            return stc == GGML_STATUS_SUCCESS ? 0 : -7;
        }
        if (allpaka_fa_launch_direct(slot->fa) != 0) {
            return -7;
        }
        fa_permute_out(
            static_cast<const float *>(slot->fa_dev),
            out_dev,
            head_dim,
            n_q_heads,
            1,
            st);
        return 0;
    }

    slot->vo->ptr = out_dev;
    slot->perm->data = slot->perm_dev;
    slot->out->data = out_dev;

    enum ggml_status stc = ggml_backend_graph_compute_async(g_be, slot->gf);
    return stc == GGML_STATUS_SUCCESS ? 0 : -7;
}

extern "C" int allpaka_hybrid_replay(
    void * const * pre_execs,
    void * const * post_execs,
    void * tail_exec,
    int32_t n_layers,
    void * stream_v,
    const float * q_dev,
    float * out_dev,
    const void * cache_base,
    const int32_t * k_off,
    const int32_t * v_off,
    int32_t head_dim,
    int32_t n_q_heads,
    int32_t n_kv_heads,
    int32_t base,
    int32_t kv_dim,
    float scale,
    const uint32_t * d_pos_dev)
{
    if (!pre_execs || !post_execs || !tail_exec || !stream_v || !q_dev || !out_dev
        || !cache_base || !k_off || !v_off || n_layers <= 0) {
        return -2;
    }
    if (!g_be) {
        if (allpaka_ggml_init(g_device) != 0) {
            return -1;
        }
    }

    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
    const int32_t need = base + 1;
    int32_t floor_kv = 256;
    if (const char * fl = std::getenv("ALLPAKA_FA_KV_FLOOR")) {
        int v = std::atoi(fl);
        if (v >= 16 && v <= 8192) {
            floor_kv = v;
        }
    }
    int32_t n_kv = floor_kv;
    while (n_kv < need) {
        n_kv *= 2;
    }
    if (n_kv > 8192) {
        return -6;
    }
    FaKey key{head_dim, n_q_heads, n_kv_heads, 1, n_kv, kv_dim, scale_bits};

    FaSlot * slot = nullptr;
    if (g_fa_decode_valid && g_fa_decode_key == key && g_fa_decode_slot) {
        slot = g_fa_decode_slot;
    } else {
        std::lock_guard<std::mutex> lock(g_mu);
        slot = get_or_create_fa(key, scale);
        if (slot) {
            g_fa_decode_key = key;
            g_fa_decode_slot = slot;
            g_fa_decode_valid = true;
        }
    }
    if (!slot) {
        return -5;
    }

    cudaStream_t st = static_cast<cudaStream_t>(stream_v);
    if (d_pos_dev && slot->mask_lim != base) {
        allpaka_fa_mask_from_pos_launch(slot->mask_dev, slot->n_kv, d_pos_dev, st);
        slot->mask_lim = base;
    }

    slot->vq->ptr = const_cast<float *>(q_dev);
    slot->Q->data = const_cast<float *>(q_dev);
    slot->mask->data = slot->mask_dev;
    slot->fa->data = slot->fa_dev;

    const char * cache_bytes = static_cast<const char *>(cache_base);
    for (int32_t li = 0; li < n_layers; li++) {
        // Driver API: cudarc streams/execs are CUstream / CUgraphExec.
        if (cuGraphLaunch(
                static_cast<CUgraphExec>(pre_execs[li]),
                static_cast<CUstream>(stream_v)) != CUDA_SUCCESS) {
            return -8;
        }

        const void * k_dev = cache_bytes + (size_t)k_off[li] * 2;
        const void * v_dev = cache_bytes + (size_t)v_off[li] * 2;
        slot->vk->ptr = const_cast<void *>(k_dev);
        slot->vv->ptr = const_cast<void *>(v_dev);
        slot->K->data = const_cast<void *>(k_dev);
        slot->V->data = const_cast<void *>(v_dev);

        if (allpaka_fa_launch_direct(slot->fa) != 0) {
            return -7;
        }
        fa_permute_out(
            static_cast<const float *>(slot->fa_dev),
            out_dev,
            head_dim,
            n_q_heads,
            1,
            st);

        if (cuGraphLaunch(
                static_cast<CUgraphExec>(post_execs[li]),
                static_cast<CUstream>(stream_v)) != CUDA_SUCCESS) {
            return -9;
        }
    }

    if (cuGraphLaunch(
            static_cast<CUgraphExec>(tail_exec),
            static_cast<CUstream>(stream_v)) != CUDA_SUCCESS) {
        return -10;
    }
    return 0;
}
