//! CUDA C kernel source compiled at attach time via NVRTC (sm_120 / compute_120).
//!
//! Intentionally free of host C library / CUDA toolkit headers so NVRTC can
//! compile without MSVC `stdint.h` on the include path.

pub const KERNELS: &str = r#"
// Minimal device-side typedefs (no stdint.h / cuda_fp16.h / math.h).
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef int int32_t;
typedef signed short int16_t;

// ---- helpers ----------------------------------------------------------------

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

// IEEE754 binary16 <-> binary32 via hardware CVT (software poly was a
// major tax on every dequant / attend load).
__device__ __forceinline__ float half_bits_to_f32(uint16_t h) {
    float f;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ uint16_t f32_to_half_bits(float f) {
    uint16_t h;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ uint16_t f32_to_f16_bits(float f) {
    return f32_to_half_bits(f);
}

// Packed Q8 activation block with four Q4_K partial sums.
struct block_q8_1 {
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
    int16_t ps[4];
};

__device__ __forceinline__ void store_q8_meta(
    block_q8_1* b, unsigned lane, int qi, float d)
{
    int ps = qi;
    ps += __shfl_xor_sync(0xffffffffu, ps, 2);
    ps += __shfl_xor_sync(0xffffffffu, ps, 1);
    ps += __shfl_xor_sync(0xffffffffu, ps, 16);
    if (lane < 16 && (lane & 3u) == 0) {
        b->ps[lane >> 2] = (int16_t)ps;
    }
    if (lane == 0) {
        b->d = f32_to_f16_bits(d);
        b->s = 0;
    }
}

__device__ __forceinline__ void store_q8_scale(block_q8_1* b, unsigned lane, float d) {
    if (lane == 0) {
        b->d = f32_to_f16_bits(d);
        b->s = 0;
    }
}

__device__ __forceinline__ float scale_min_k4(int j, const uint8_t* packed, float* mn_out) {
    float sc, mn;
    if (j < 4) {
        sc = (float)(packed[j] & 63);
        mn = (float)(packed[j + 4] & 63);
    } else {
        sc = (float)((packed[j + 4] & 0xf) | ((packed[j - 4] >> 6) << 4));
        mn = (float)((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
    }
    *mn_out = mn;
    return sc;
}

// Keep griddepcontrol on sm_90+ — nopdl-dev A/B regressed ~57 vs ~63 tok/s.
__device__ __forceinline__ void pdl_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_lc() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ float warp_sum_f(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ---- elementwise / norms ----------------------------------------------------

extern "C" __global__ void rmsnorm_f32(
    float* __restrict__ x,
    const float* __restrict__ w,
    unsigned n,
    float eps,
    unsigned rows)
{
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    float* xr = x + (size_t)row * n;
    // Two-pass: mean of squares then scale. Block-reduce in shared mem.
    __shared__ float buf[256];
    float local = 0.f;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float mean = buf[0] / (float)n;
    float scale = rsqrtf(mean + eps);
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        xr[i] = xr[i] * scale * w[i];
    }
}

extern "C" __global__ void rmsnorm_into_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ w,
    unsigned n,
    float eps,
    unsigned rows)
{
    unsigned row = blockIdx.x;
    pdl_lc();
    if (row >= rows) return;
    const float* sr = src + (size_t)row * n;
    float* dr = dst + (size_t)row * n;
    __shared__ float buf[256];
    float local = 0.f;
    pdl_sync();
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        float v = sr[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float mean = buf[0] / (float)n;
    float scale = rsqrtf(mean + eps);
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        dr[i] = sr[i] * scale * w[i];
    }
}

// RMSNorm into dst, then pack dst as block_q8_1 for mmvq reuse.
extern "C" __global__ void rmsnorm_into_f32_q8(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ w,
    block_q8_1* __restrict__ y,
    unsigned n,
    float eps,
    unsigned rows)
{
    unsigned row = blockIdx.x;
    pdl_lc();
    if (row >= rows) return;
    const float* sr = src + (size_t)row * n;
    float* dr = dst + (size_t)row * n;
    __shared__ float buf[256];
    float local = 0.f;
    pdl_sync();
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        float v = sr[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float mean = buf[0] / (float)n;
    float scale = rsqrtf(mean + eps);
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        dr[i] = sr[i] * scale * w[i];
    }
    __syncthreads();
    // One warp-worth of lanes quantize 32-groups (blockDim.x should be 256).
    unsigned nblk = n / 32u;
    block_q8_1* yr = y + (size_t)row * nblk;
    for (unsigned blk = threadIdx.x / 32u; blk < nblk; blk += blockDim.x / 32u) {
        unsigned lane = threadIdx.x & 31u;
        unsigned base = blk * 32u;
        float v = dr[base + lane];
        float amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        float d = amax / 127.f;
        if (d < 1e-8f) d = 1.f;
        int qi = __float2int_rn(v / d);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        yr[blk].qs[lane] = (int8_t)qi;
        store_q8_meta(&yr[blk], lane, qi, d);
    }
}

extern "C" __global__ void residual_add(
    float* __restrict__ a,
    const float* __restrict__ b,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

extern "C" __global__ void swiglu(
    float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gate[i] = silu_f(gate[i]) * up[i];
}

// SwiGLU + block_q8_1 pack in one pass (decode down-proj avoids a separate quantize).
extern "C" __global__ void swiglu_into_q8(
    float* __restrict__ gate,
    const float* __restrict__ up,
    block_q8_1* __restrict__ y,
    unsigned n, unsigned rows)
{
    unsigned row = blockIdx.y;
    unsigned blk = blockIdx.x;
    unsigned lane = threadIdx.x;
    if (row >= rows) return;
    unsigned base = blk * 32u;
    if (base >= n) return;
    size_t idx = (size_t)row * n + base + lane;
    float v = silu_f(gate[idx]) * up[idx];
    gate[idx] = v;
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float scale = amax / 127.f;
    if (scale < 1e-8f) scale = 1.f;
    int qi = __float2int_rn(v / scale);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    block_q8_1* b = y + (size_t)row * (n / 32u) + blk;
    b->qs[lane] = (int8_t)qi;
    store_q8_meta(b, lane, qi, scale);
}

extern "C" __global__ void copy_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void cast_f32_to_f16(
    uint16_t* __restrict__ dst,
    const float* __restrict__ src,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f32_to_half_bits(src[i]);
}

// RoPE NeoX: pairs (x[i], x[i+half]) with table[i] = {sin, cos}.
// Applies to `heads` contiguous heads of dimension `head_dim`, only first
// `rot_dim` elements of each head (partial rotary when rot_dim < head_dim).
extern "C" __global__ void rope_neox(
    float* __restrict__ x,
    const float* __restrict__ rope, // [heads, rot_dim/2, 2] as flat sin,cos
    unsigned heads,
    unsigned head_dim,
    unsigned rot_dim)
{
    unsigned h = blockIdx.x;
    if (h >= heads) return;
    unsigned half_n = rot_dim / 2;
    float* xh = x + (size_t)h * head_dim;
    const float* rh = rope + (size_t)h * half_n * 2;
    for (unsigned i = threadIdx.x; i < half_n; i += blockDim.x) {
        float sinv = rh[2 * i];
        float cosv = rh[2 * i + 1];
        float a = xh[i];
        float b = xh[i + half_n];
        xh[i] = a * cosv - b * sinv;
        xh[i + half_n] = a * sinv + b * cosv;
    }
}

// Per-head RMSNorm (weight w[head_dim]) then NeoX RoPE. One block per head.
extern "C" __global__ void rmsnorm_rope_neox(
    float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ rope,
    unsigned heads,
    unsigned head_dim,
    unsigned rot_dim,
    float eps)
{
    unsigned h = blockIdx.x;
    if (h >= heads) return;
    float* xh = x + (size_t)h * head_dim;
    __shared__ float buf[256];
    float local = 0.f;
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xh[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(buf[0] / (float)head_dim + eps);
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xh[i] = xh[i] * scale * w[i];
    }
    __syncthreads();
    unsigned half_n = rot_dim / 2;
    const float* rh = rope + (size_t)h * half_n * 2;
    for (unsigned i = threadIdx.x; i < half_n; i += blockDim.x) {
        float sinv = rh[2 * i];
        float cosv = rh[2 * i + 1];
        float a = xh[i];
        float b = xh[i + half_n];
        xh[i] = a * cosv - b * sinv;
        xh[i + half_n] = a * sinv + b * cosv;
    }
}

// Per-head RMSNorm + NeoX RoPE, then store this head's slice into KV cache (K path).
extern "C" __global__ void rmsnorm_rope_store_dpos(
    float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ rope,
    uint16_t* __restrict__ cache,
    unsigned long long base_off,
    const unsigned* __restrict__ d_pos,
    unsigned heads,
    unsigned head_dim,
    unsigned rot_dim,
    unsigned kv_dim,
    float eps)
{
    unsigned h = blockIdx.x;
    if (h >= heads) return;
    float* xh = x + (size_t)h * head_dim;
    __shared__ float buf[256];
    float local = 0.f;
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xh[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(buf[0] / (float)head_dim + eps);
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xh[i] = xh[i] * scale * w[i];
    }
    __syncthreads();
    unsigned half_n = rot_dim / 2;
    const float* rh = rope + (size_t)h * half_n * 2;
    for (unsigned i = threadIdx.x; i < half_n; i += blockDim.x) {
        float sinv = rh[2 * i];
        float cosv = rh[2 * i + 1];
        float a = xh[i];
        float b = xh[i + half_n];
        xh[i] = a * cosv - b * sinv;
        xh[i + half_n] = a * sinv + b * cosv;
    }
    __syncthreads();
    unsigned pos = d_pos[0];
    unsigned long long dst = base_off + (unsigned long long)pos * kv_dim + (unsigned long long)h * head_dim;
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        cache[dst + i] = f32_to_half_bits(xh[i]);
    }
}

// Per-head RMSNorm + NeoX RoPE. RoPE angles from inv_freq[i]*d_pos (no H2D table).
extern "C" __global__ void rmsnorm_rope_freq_dpos(
    float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ inv_freq,
    const unsigned* __restrict__ d_pos,
    unsigned heads,
    unsigned head_dim,
    unsigned rot_dim,
    float eps)
{
    unsigned h = blockIdx.x;
    if (h >= heads) return;
    float* xh = x + (size_t)h * head_dim;
    __shared__ float buf[256];
    float local = 0.f;
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xh[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(buf[0] / (float)head_dim + eps);
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xh[i] = xh[i] * scale * w[i];
    }
    __syncthreads();
    unsigned half_n = rot_dim / 2;
    float pos = (float)d_pos[0];
    for (unsigned i = threadIdx.x; i < half_n; i += blockDim.x) {
        float angle = pos * inv_freq[i];
        float sinv, cosv;
        __sincosf(angle, &sinv, &cosv);
        float a = xh[i];
        float b = xh[i + half_n];
        xh[i] = a * cosv - b * sinv;
        xh[i + half_n] = a * sinv + b * cosv;
    }
}

// RMSNorm + freq RoPE + store K into f16 cache.
extern "C" __global__ void rmsnorm_rope_store_freq_dpos(
    float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ inv_freq,
    uint16_t* __restrict__ cache,
    unsigned long long base_off,
    const unsigned* __restrict__ d_pos,
    unsigned heads,
    unsigned head_dim,
    unsigned rot_dim,
    unsigned kv_dim,
    float eps)
{
    unsigned h = blockIdx.x;
    if (h >= heads) return;
    float* xh = x + (size_t)h * head_dim;
    __shared__ float buf[256];
    float local = 0.f;
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xh[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(buf[0] / (float)head_dim + eps);
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xh[i] = xh[i] * scale * w[i];
    }
    __syncthreads();
    unsigned half_n = rot_dim / 2;
    float posf = (float)d_pos[0];
    for (unsigned i = threadIdx.x; i < half_n; i += blockDim.x) {
        float angle = posf * inv_freq[i];
        float sinv, cosv;
        __sincosf(angle, &sinv, &cosv);
        float a = xh[i];
        float b = xh[i + half_n];
        xh[i] = a * cosv - b * sinv;
        xh[i + half_n] = a * sinv + b * cosv;
    }
    __syncthreads();
    unsigned pos = d_pos[0];
    unsigned long long dst = base_off + (unsigned long long)pos * kv_dim + (unsigned long long)h * head_dim;
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        cache[dst + i] = f32_to_half_bits(xh[i]);
    }
}

extern "C" __global__ void build_rope_freq_dpos(
    float* __restrict__ rope,
    const float* __restrict__ inv_freq,
    const unsigned* __restrict__ d_pos,
    unsigned pairs)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    float sinv, cosv;
    __sincosf((float)d_pos[0] * inv_freq[i], &sinv, &cosv);
    rope[2 * i] = sinv;
    rope[2 * i + 1] = cosv;
}

extern "C" __global__ void rmsnorm_rope_qk_store_table_dpos(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ qw,
    const float* __restrict__ kw,
    const float* __restrict__ rope,
    uint16_t* __restrict__ cache,
    unsigned long long base_off,
    const unsigned* __restrict__ d_pos,
    unsigned q_heads,
    unsigned kv_heads,
    unsigned head_dim,
    unsigned rot_dim,
    unsigned kv_dim,
    float eps)
{
    unsigned gh = blockIdx.x;
    bool is_k = gh >= q_heads;
    unsigned h = is_k ? gh - q_heads : gh;
    if ((!is_k && h >= q_heads) || (is_k && h >= kv_heads)) return;
    float* xh = (is_k ? k : q) + (size_t)h * head_dim;
    const float* w = is_k ? kw : qw;
    __shared__ float buf[256];
    float local = 0.f;
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xh[i];
        local += v * v;
    }
    local = warp_sum_f(local);
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    if (lane == 0) buf[warp] = local;
    __syncthreads();
    if (warp == 0) {
        unsigned nwarps = blockDim.x >> 5;
        local = lane < nwarps ? buf[lane] : 0.f;
        local = warp_sum_f(local);
        if (lane == 0) buf[0] = local;
    }
    __syncthreads();
    float scale = rsqrtf(buf[0] / (float)head_dim + eps);
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xh[i] * scale * w[i];
        if (is_k) buf[i] = v;
        else xh[i] = v;
    }
    __syncthreads();
    unsigned half_n = rot_dim / 2;
    unsigned long long dst = 0;
    if (is_k) {
        unsigned pos = d_pos[0];
        dst = base_off + (unsigned long long)pos * kv_dim + (unsigned long long)h * head_dim;
    }
    for (unsigned i = threadIdx.x; i < half_n; i += blockDim.x) {
        float sinv = rope[2 * i];
        float cosv = rope[2 * i + 1];
        float a = is_k ? buf[i] : xh[i];
        float b = is_k ? buf[i + half_n] : xh[i + half_n];
        float r0 = a * cosv - b * sinv;
        float r1 = a * sinv + b * cosv;
        if (is_k) {
            cache[dst + i] = f32_to_half_bits(r0);
            cache[dst + i + half_n] = f32_to_half_bits(r1);
        } else {
            xh[i] = r0;
            xh[i + half_n] = r1;
        }
    }
    if (is_k) {
        for (unsigned i = rot_dim + threadIdx.x; i < head_dim; i += blockDim.x) {
            cache[dst + i] = f32_to_half_bits(buf[i]);
        }
    }
}

extern "C" __global__ void rope_neox_freq_dpos(
    float* __restrict__ x,
    const float* __restrict__ inv_freq,
    const unsigned* __restrict__ d_pos,
    unsigned heads,
    unsigned head_dim,
    unsigned rot_dim)
{
    unsigned h = blockIdx.x;
    if (h >= heads) return;
    unsigned half_n = rot_dim / 2;
    float* xh = x + (size_t)h * head_dim;
    float pos = (float)d_pos[0];
    for (unsigned i = threadIdx.x; i < half_n; i += blockDim.x) {
        float angle = pos * inv_freq[i];
        float sinv, cosv;
        __sincosf(angle, &sinv, &cosv);
        float a = xh[i];
        float b = xh[i + half_n];
        xh[i] = a * cosv - b * sinv;
        xh[i + half_n] = a * sinv + b * cosv;
    }
}

extern "C" __global__ void argmax_f32(
    const float* __restrict__ x,
    unsigned n,
    unsigned* __restrict__ out_idx)
{
    // Single-block argmax for vocab-sized vectors.
    __shared__ float smax[256];
    __shared__ unsigned sidx[256];
    float best = (-1.0f/0.0f);
    unsigned bi = 0;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (v > best) { best = v; bi = i; }
    }
    smax[threadIdx.x] = best;
    sidx[threadIdx.x] = bi;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (smax[threadIdx.x + s] > smax[threadIdx.x]) {
                smax[threadIdx.x] = smax[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_idx[0] = sidx[0];
}

// Store one K or V row (kv_dim floats) into the f16 cache at element offset.
extern "C" __global__ void store_kv_f16(
    uint16_t* __restrict__ cache,
    const float* __restrict__ src,
    unsigned long long elem_off,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cache[elem_off + i] = f32_to_half_bits(src[i]);
}

// Like store_kv_f16 but reads the token position from device memory so a
// CUDA graph can be replayed across tokens without rebuilding.
extern "C" __global__ void store_kv_f16_dpos(
    uint16_t* __restrict__ cache,
    const float* __restrict__ src,
    unsigned long long base_off,
    const unsigned* __restrict__ d_pos,
    unsigned kv_dim,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned pos = d_pos[0];
    cache[base_off + (unsigned long long)pos * kv_dim + i] = f32_to_half_bits(src[i]);
}

// Store K and V for one token in one launch (same d_pos / kv_dim).
extern "C" __global__ void store_kv_pair_f16_dpos(
    uint16_t* __restrict__ cache,
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    unsigned long long k_off,
    unsigned long long v_off,
    const unsigned* __restrict__ d_pos,
    unsigned kv_dim,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned pos = d_pos[0];
    unsigned long long row = (unsigned long long)pos * kv_dim + i;
    cache[k_off + row] = f32_to_half_bits(k_src[i]);
    cache[v_off + row] = f32_to_half_bits(v_src[i]);
}

// Store m consecutive KV rows: src[row, kv_dim] → cache[base + (pos0+row)*kv_dim].
extern "C" __global__ void store_kv_batch_f16(
    uint16_t* __restrict__ cache,
    const float* __restrict__ src,
    unsigned long long base_off,
    unsigned kv_dim,
    unsigned m,
    unsigned pos0)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row = blockIdx.y;
    if (row >= m || i >= kv_dim) return;
    unsigned long long dst = base_off + (unsigned long long)(pos0 + row) * kv_dim + i;
    cache[dst] = f32_to_half_bits(src[(size_t)row * kv_dim + i]);
}

// RoPE NeoX over a batch of tokens. x: [m, heads, head_dim];
// rope: [m, rot_dim] as interleaved sin,cos (same table for every head).
extern "C" __global__ void rope_neox_batch(
    float* __restrict__ x,
    const float* __restrict__ rope,
    unsigned m,
    unsigned heads,
    unsigned head_dim,
    unsigned rot_dim)
{
    unsigned h = blockIdx.x;
    unsigned row = blockIdx.y;
    if (h >= heads || row >= m) return;
    unsigned half_n = rot_dim / 2;
    float* xh = x + ((size_t)row * heads + h) * head_dim;
    const float* rh = rope + (size_t)row * half_n * 2;
    for (unsigned i = threadIdx.x; i < half_n; i += blockDim.x) {
        float sinv = rh[2 * i];
        float cosv = rh[2 * i + 1];
        float a = xh[i];
        float b = xh[i + half_n];
        xh[i] = a * cosv - b * sinv;
        xh[i + half_n] = a * sinv + b * cosv;
    }
}

// Online-softmax GQA attention over an f16 KV cache.
// One warp (32 threads) per query head. head_dim in {64,128,256}.
extern "C" __global__ void attend_gqa(
    const float* __restrict__ q,     // [n_q_heads, head_dim]
    const uint16_t* __restrict__ cache,  // whole KV region
    float* __restrict__ out,         // [n_q_heads, head_dim]
    unsigned long long k_off,        // element offset of K base
    unsigned long long v_off,
    unsigned kv_dim,
    unsigned head_dim,
    unsigned n_q_heads,
    unsigned group,                  // n_q_heads / n_kv_heads
    unsigned n_pos,
    float scale)
{
    unsigned qh = blockIdx.x;
    if (qh >= n_q_heads) return;
    unsigned lane = threadIdx.x;
    unsigned kvh = qh / group;
    const float* qh_ptr = q + (size_t)qh * head_dim;
    unsigned nreg = head_dim >> 5; // 2/4/8 for 64/128/256
    float q_r[8], acc_r[8];
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) {
            q_r[i] = qh_ptr[i * 32u + lane];
            acc_r[i] = 0.f;
        }
    }
    float m_i = (-1.0f/0.0f);
    float l_i = 0.f;
    for (unsigned p = 0; p < n_pos; p++) {
        const uint16_t* krow = cache + k_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
        float partial = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < 8; i++) {
            if (i < nreg) partial += q_r[i] * half_bits_to_f32(krow[i * 32u + lane]);
        }
        float score = warp_sum_f(partial) * scale;
        score = __shfl_sync(0xffffffffu, score, 0);
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float w = expf(score - m_new);
        float l_new = alpha * l_i + w;
        const uint16_t* vrow = cache + v_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
        #pragma unroll
        for (unsigned i = 0; i < 8; i++) {
            if (i < nreg) {
                acc_r[i] = acc_r[i] * alpha + w * half_bits_to_f32(vrow[i * 32u + lane]);
            }
        }
        m_i = m_new;
        l_i = l_new;
    }
    float inv_l = 1.0f / l_i;
    float* out_h = out + (size_t)qh * head_dim;
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) out_h[i * 32u + lane] = acc_r[i] * inv_l;
    }
}

// attend_gqa with n_pos = *d_pos + 1 for CUDA-graph replay.
extern "C" __global__ void attend_gqa_dpos(
    const float* __restrict__ q,
    const uint16_t* __restrict__ cache,
    float* __restrict__ out,
    unsigned long long k_off,
    unsigned long long v_off,
    unsigned kv_dim,
    unsigned head_dim,
    unsigned n_q_heads,
    unsigned group,
    const unsigned* __restrict__ d_pos,
    float scale)
{
    unsigned qh = blockIdx.x;
    if (qh >= n_q_heads) return;
    unsigned lane = threadIdx.x;
    unsigned kvh = qh / group;
    unsigned n_pos = d_pos[0] + 1u;
    const float* qh_ptr = q + (size_t)qh * head_dim;
    unsigned nreg = head_dim >> 5;
    float q_r[8], acc_r[8];
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) {
            q_r[i] = qh_ptr[i * 32u + lane];
            acc_r[i] = 0.f;
        }
    }
    float m_i = (-1.0f/0.0f);
    float l_i = 0.f;
    for (unsigned p = 0; p < n_pos; p++) {
        const uint16_t* krow = cache + k_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
        float partial = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < 8; i++) {
            if (i < nreg) partial += q_r[i] * half_bits_to_f32(krow[i * 32u + lane]);
        }
        float score = warp_sum_f(partial) * scale;
        score = __shfl_sync(0xffffffffu, score, 0);
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float w = expf(score - m_new);
        float l_new = alpha * l_i + w;
        const uint16_t* vrow = cache + v_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
        #pragma unroll
        for (unsigned i = 0; i < 8; i++) {
            if (i < nreg) {
                acc_r[i] = acc_r[i] * alpha + w * half_bits_to_f32(vrow[i * 32u + lane]);
            }
        }
        m_i = m_new;
        l_i = l_new;
    }
    float inv_l = 1.0f / l_i;
    float* out_h = out + (size_t)qh * head_dim;
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) out_h[i * 32u + lane] = acc_r[i] * inv_l;
    }
}

// Prefill attend: one warp per (row, q_head). Causal: row r attends to
// base..base+r inclusive (n_pos_r = base + r + 1).
extern "C" __global__ void attend_gqa_batch(
    const float* __restrict__ q,     // [m, n_q_heads, head_dim]
    const uint16_t* __restrict__ cache,
    float* __restrict__ out,         // [m, n_q_heads, head_dim]
    unsigned long long k_off,
    unsigned long long v_off,
    unsigned kv_dim,
    unsigned head_dim,
    unsigned n_q_heads,
    unsigned group,
    unsigned m,
    unsigned base,
    float scale)
{
    unsigned qh = blockIdx.x;
    unsigned row = blockIdx.y;
    if (qh >= n_q_heads || row >= m) return;
    unsigned lane = threadIdx.x;
    unsigned kvh = qh / group;
    unsigned n_pos = base + row + 1;
    const float* qh_ptr = q + ((size_t)row * n_q_heads + qh) * head_dim;
    unsigned nreg = head_dim >> 5;
    float q_r[8], acc_r[8];
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) {
            q_r[i] = qh_ptr[i * 32u + lane];
            acc_r[i] = 0.f;
        }
    }
    float m_i = (-1.0f/0.0f);
    float l_i = 0.f;
    for (unsigned p = 0; p < n_pos; p++) {
        const uint16_t* krow = cache + k_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
        float partial = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < 8; i++) {
            if (i < nreg) partial += q_r[i] * half_bits_to_f32(krow[i * 32u + lane]);
        }
        float score = warp_sum_f(partial) * scale;
        score = __shfl_sync(0xffffffffu, score, 0);
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float w = expf(score - m_new);
        float l_new = alpha * l_i + w;
        const uint16_t* vrow = cache + v_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
        #pragma unroll
        for (unsigned i = 0; i < 8; i++) {
            if (i < nreg) {
                acc_r[i] = acc_r[i] * alpha + w * half_bits_to_f32(vrow[i * 32u + lane]);
            }
        }
        m_i = m_new;
        l_i = l_new;
    }
    float inv_l = 1.0f / l_i;
    float* out_h = out + ((size_t)row * n_q_heads + qh) * head_dim;
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) out_h[i * 32u + lane] = acc_r[i] * inv_l;
    }
}

// Prefill GQA-fused: block=(32, group) per (row, kv_head). K/V tiled into smem
// so each position is loaded once; sync only at tile boundaries.
extern "C" __global__ void attend_gqa_batch_kv(
    const float* __restrict__ q,
    const uint16_t* __restrict__ cache,
    float* __restrict__ out,
    unsigned long long k_off,
    unsigned long long v_off,
    unsigned kv_dim,
    unsigned head_dim,
    unsigned n_q_heads,
    unsigned group,
    unsigned m,
    unsigned base,
    float scale)
{
    unsigned kvh = blockIdx.x;
    unsigned row = blockIdx.y;
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned n_kv = n_q_heads / group;
    if (kvh >= n_kv || row >= m) return;
    unsigned qh = kvh * group + warp;
    unsigned tid = warp * 32u + lane;
    unsigned nthreads = group * 32u;
    unsigned n_pos = base + row + 1u;
    unsigned nreg = head_dim >> 5;

    extern __shared__ uint16_t smem[];
    // Layout: [TILE][head_dim] K then [TILE][head_dim] V. TILE=64.
    const unsigned TILE = 64u;
    uint16_t* k_s = smem;
    uint16_t* v_s = smem + TILE * head_dim;

    const float* qh_ptr = q + ((size_t)row * n_q_heads + qh) * head_dim;
    float q_r[8], acc_r[8];
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) {
            q_r[i] = qh_ptr[i * 32u + lane];
            acc_r[i] = 0.f;
        }
    }
    float m_i = (-1.0f/0.0f);
    float l_i = 0.f;

    for (unsigned p0 = 0; p0 < n_pos; p0 += TILE) {
        unsigned ntile = n_pos - p0;
        if (ntile > TILE) ntile = TILE;
        unsigned tile_elems = ntile * head_dim;
        for (unsigned i = tid; i < tile_elems; i += nthreads) {
            unsigned ti = i / head_dim;
            unsigned di = i - ti * head_dim;
            unsigned p = p0 + ti;
            const uint16_t* krow = cache + k_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
            const uint16_t* vrow = cache + v_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
            k_s[i] = krow[di];
            v_s[i] = vrow[di];
        }
        __syncthreads();

        for (unsigned ti = 0; ti < ntile; ti++) {
            const uint16_t* krow = k_s + ti * head_dim;
            float partial = 0.f;
            #pragma unroll
            for (unsigned i = 0; i < 8; i++) {
                if (i < nreg) partial += q_r[i] * half_bits_to_f32(krow[i * 32u + lane]);
            }
            float score = warp_sum_f(partial) * scale;
            score = __shfl_sync(0xffffffffu, score, 0);
            float m_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_new);
            float w = expf(score - m_new);
            float l_new = alpha * l_i + w;
            const uint16_t* vrow = v_s + ti * head_dim;
            #pragma unroll
            for (unsigned i = 0; i < 8; i++) {
                if (i < nreg) {
                    acc_r[i] = acc_r[i] * alpha + w * half_bits_to_f32(vrow[i * 32u + lane]);
                }
            }
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    float inv_l = 1.0f / l_i;
    float* out_h = out + ((size_t)row * n_q_heads + qh) * head_dim;
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) out_h[i * 32u + lane] = acc_r[i] * inv_l;
    }
}

extern "C" __global__ void attend_gqa_kv(
    const float* __restrict__ q,
    const uint16_t* __restrict__ cache,
    float* __restrict__ out,
    unsigned long long k_off,
    unsigned long long v_off,
    unsigned kv_dim,
    unsigned head_dim,
    unsigned n_q_heads,
    unsigned group,
    unsigned n_pos,
    float scale)
{
    unsigned kvh = blockIdx.x;
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned n_kv = n_q_heads / group;
    if (kvh >= n_kv) return;
    unsigned qh = kvh * group + warp;
    unsigned tid = warp * 32u + lane;
    unsigned nthreads = group * 32u;
    unsigned nreg = head_dim >> 5;

    extern __shared__ uint16_t smem[];
    const unsigned TILE = 64u;
    uint16_t* k_s = smem;
    uint16_t* v_s = smem + TILE * head_dim;

    const float* qh_ptr = q + (size_t)qh * head_dim;
    float q_r[8], acc_r[8];
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) {
            q_r[i] = qh_ptr[i * 32u + lane];
            acc_r[i] = 0.f;
        }
    }
    float m_i = (-1.0f/0.0f);
    float l_i = 0.f;

    for (unsigned p0 = 0; p0 < n_pos; p0 += TILE) {
        unsigned ntile = n_pos - p0;
        if (ntile > TILE) ntile = TILE;
        unsigned tile_elems = ntile * head_dim;
        for (unsigned i = tid; i < tile_elems; i += nthreads) {
            unsigned ti = i / head_dim;
            unsigned di = i - ti * head_dim;
            unsigned p = p0 + ti;
            const uint16_t* krow = cache + k_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
            const uint16_t* vrow = cache + v_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
            k_s[i] = krow[di];
            v_s[i] = vrow[di];
        }
        __syncthreads();

        for (unsigned ti = 0; ti < ntile; ti++) {
            const uint16_t* krow = k_s + ti * head_dim;
            float partial = 0.f;
            #pragma unroll
            for (unsigned i = 0; i < 8; i++) {
                if (i < nreg) partial += q_r[i] * half_bits_to_f32(krow[i * 32u + lane]);
            }
            float score = warp_sum_f(partial) * scale;
            score = __shfl_sync(0xffffffffu, score, 0);
            float m_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_new);
            float w = expf(score - m_new);
            float l_new = alpha * l_i + w;
            const uint16_t* vrow = v_s + ti * head_dim;
            #pragma unroll
            for (unsigned i = 0; i < 8; i++) {
                if (i < nreg) {
                    acc_r[i] = acc_r[i] * alpha + w * half_bits_to_f32(vrow[i * 32u + lane]);
                }
            }
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    float inv_l = 1.0f / l_i;
    float* out_h = out + (size_t)qh * head_dim;
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) out_h[i * 32u + lane] = acc_r[i] * inv_l;
    }
}

extern "C" __global__ void attend_gqa_dpos_kv(
    const float* __restrict__ q,
    const uint16_t* __restrict__ cache,
    float* __restrict__ out,
    unsigned long long k_off,
    unsigned long long v_off,
    unsigned kv_dim,
    unsigned head_dim,
    unsigned n_q_heads,
    unsigned group,
    const unsigned* __restrict__ d_pos,
    float scale)
{
    unsigned kvh = blockIdx.x;
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned n_kv = n_q_heads / group;
    if (kvh >= n_kv) return;
    unsigned qh = kvh * group + warp;
    unsigned tid = warp * 32u + lane;
    unsigned nthreads = group * 32u;
    unsigned n_pos = d_pos[0] + 1u;
    unsigned nreg = head_dim >> 5;

    extern __shared__ uint16_t smem[];
    const unsigned TILE = 64u;
    uint16_t* k_s = smem;
    uint16_t* v_s = smem + TILE * head_dim;

    const float* qh_ptr = q + (size_t)qh * head_dim;
    float q_r[8], acc_r[8];
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) {
            q_r[i] = qh_ptr[i * 32u + lane];
            acc_r[i] = 0.f;
        }
    }
    float m_i = (-1.0f/0.0f);
    float l_i = 0.f;

    for (unsigned p0 = 0; p0 < n_pos; p0 += TILE) {
        unsigned ntile = n_pos - p0;
        if (ntile > TILE) ntile = TILE;
        unsigned tile_elems = ntile * head_dim;
        for (unsigned i = tid; i < tile_elems; i += nthreads) {
            unsigned ti = i / head_dim;
            unsigned di = i - ti * head_dim;
            unsigned p = p0 + ti;
            const uint16_t* krow = cache + k_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
            const uint16_t* vrow = cache + v_off + (size_t)p * kv_dim + (size_t)kvh * head_dim;
            k_s[i] = krow[di];
            v_s[i] = vrow[di];
        }
        __syncthreads();

        for (unsigned ti = 0; ti < ntile; ti++) {
            const uint16_t* krow = k_s + ti * head_dim;
            float partial = 0.f;
            #pragma unroll
            for (unsigned i = 0; i < 8; i++) {
                if (i < nreg) partial += q_r[i] * half_bits_to_f32(krow[i * 32u + lane]);
            }
            float score = warp_sum_f(partial) * scale;
            score = __shfl_sync(0xffffffffu, score, 0);
            float m_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_new);
            float w = expf(score - m_new);
            float l_new = alpha * l_i + w;
            const uint16_t* vrow = v_s + ti * head_dim;
            #pragma unroll
            for (unsigned i = 0; i < 8; i++) {
                if (i < nreg) {
                    acc_r[i] = acc_r[i] * alpha + w * half_bits_to_f32(vrow[i * 32u + lane]);
                }
            }
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    float inv_l = 1.0f / l_i;
    float* out_h = out + (size_t)qh * head_dim;
    #pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        if (i < nreg) out_h[i * 32u + lane] = acc_r[i] * inv_l;
    }
}
// ---- MoE helpers ------------------------------------------------------------

extern "C" __global__ void softmax_topk(
    const float* __restrict__ logits, // [n_expert]
    float* __restrict__ weights,      // [n_used]
    unsigned* __restrict__ ids,       // [n_used]
    unsigned n_expert,
    unsigned n_used,
    int norm,
    int sigmoid)
{
    // Single-thread kernel: n_expert <= 256.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float scores[256];
    unsigned idx[256];
    for (unsigned i = 0; i < n_expert; i++) {
        float v = logits[i];
        scores[i] = sigmoid ? (1.0f / (1.0f + expf(-v))) : v;
        idx[i] = i;
    }
    if (!sigmoid) {
        float m = (-1.0f/0.0f);
        for (unsigned i = 0; i < n_expert; i++) m = fmaxf(m, scores[i]);
        float sum = 0.f;
        for (unsigned i = 0; i < n_expert; i++) {
            scores[i] = expf(scores[i] - m);
            sum += scores[i];
        }
        float inv = 1.0f / sum;
        for (unsigned i = 0; i < n_expert; i++) scores[i] *= inv;
    }
    // Partial selection sort for top-k.
    for (unsigned k = 0; k < n_used; k++) {
        unsigned best = k;
        for (unsigned i = k + 1; i < n_expert; i++) {
            if (scores[i] > scores[best]) best = i;
        }
        float ts = scores[k]; scores[k] = scores[best]; scores[best] = ts;
        unsigned ti = idx[k]; idx[k] = idx[best]; idx[best] = ti;
    }
    if (norm) {
        float s = 0.f;
        for (unsigned k = 0; k < n_used; k++) s += scores[k];
        float inv = 1.0f / s;
        for (unsigned k = 0; k < n_used; k++) scores[k] *= inv;
    }
    for (unsigned k = 0; k < n_used; k++) {
        weights[k] = scores[k];
        ids[k] = idx[k];
    }
}

extern "C" __global__ void moe_combine(
    float* __restrict__ out,          // [hidden]
    const float* __restrict__ downs,  // [slots, hidden]
    const float* __restrict__ weights,// [slots]
    unsigned slots,
    unsigned hidden)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    float acc = 0.f;
    for (unsigned s = 0; s < slots; s++) {
        acc += weights[s] * downs[(size_t)s * hidden + i];
    }
    out[i] = acc;
}

extern "C" __global__ void moe_combine_csr(
    float* __restrict__ xs,           // [m, hidden] residual stream
    const float* __restrict__ downs,  // [total_rows(+shared), hidden]
    const unsigned* __restrict__ tok_off,
    const unsigned* __restrict__ hit_row,
    const float* __restrict__ hit_w,
    unsigned m,
    unsigned hidden)
{
    unsigned row = blockIdx.y;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || i >= hidden) return;
    unsigned a = tok_off[row];
    unsigned b = tok_off[row + 1];
    float acc = 0.f;
    for (unsigned h = a; h < b; h++) {
        acc += hit_w[h] * downs[(size_t)hit_row[h] * hidden + i];
    }
    xs[(size_t)row * hidden + i] += acc;
}

// ---- fused matvecs: one thread per (batch_row, out_row) ---------------------

#define MV_IDX() \
    unsigned out = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned row = blockIdx.y; \
    if (out >= n_out || row >= m) return; \
    const float* xr = x + (size_t)row * n_in; \
    float* yr = y + (size_t)row * n_out; \
    const uint8_t* wb = w + w_off;

extern "C" __global__ void matvec_f32(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    MV_IDX();
    const float* wr = (const float*)(wb + (size_t)out * n_in * 4);
    float acc = 0.f;
    for (unsigned i = 0; i < n_in; i++) acc += wr[i] * xr[i];
    yr[out] = acc;
}

extern "C" __global__ void matvec_f16(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    MV_IDX();
    const uint16_t* wr = (const uint16_t*)(wb + (size_t)out * n_in * 2);
    float acc = 0.f;
    for (unsigned i = 0; i < n_in; i++) acc += half_bits_to_f32(wr[i]) * xr[i];
    yr[out] = acc;
}

extern "C" __global__ void matvec_bf16(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    MV_IDX();
    const uint16_t* wr = (const uint16_t*)(wb + (size_t)out * n_in * 2);
    float acc = 0.f;
    for (unsigned i = 0; i < n_in; i++) {
        acc += __uint_as_float(((uint32_t)wr[i]) << 16) * xr[i];
    }
    yr[out] = acc;
}

extern "C" __global__ void matvec_q8_0(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    MV_IDX();
    const uint8_t* wrow = wb + (size_t)out * (n_in / 32) * 34;
    float acc = 0.f;
    unsigned nb = n_in / 32;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 34;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        const float* xb = xr + b * 32;
        for (unsigned i = 0; i < 32; i++) {
            acc += d * (float)((int8_t)blk[2 + i]) * xb[i];
        }
    }
    yr[out] = acc;
}

extern "C" __global__ void matvec_q5_0(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    MV_IDX();
    const uint8_t* wrow = wb + (size_t)out * (n_in / 32) * 22;
    float acc = 0.f;
    unsigned nb = n_in / 32;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 22;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
            | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
        const uint8_t* qs = blk + 6;
        const float* xb = xr + b * 32;
        for (unsigned j = 0; j < 16; j++) {
            int x0 = (int)((qs[j] & 0x0f) | (((qh >> j) & 1) << 4)) - 16;
            int x1 = (int)((qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
            acc += d * (float)x0 * xb[j];
            acc += d * (float)x1 * xb[j + 16];
        }
    }
    yr[out] = acc;
}

// Metal-style Q4_K mmvq: one warp owns 4 output rows; X stays in registers.
extern "C" __global__ void matvec_q4_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned row = blockIdx.y;
    if (row >= m) return;
    // 4 warps × 4 rows = 16 output rows per block (best measured ~17.9 tok/s).
    unsigned pair0 = blockIdx.x * 16u + warp * 4u;
    if (pair0 >= n_out) return;
    unsigned nrows = n_out - pair0;
    if (nrows > 4u) nrows = 4u;

    const float* xr = x + (size_t)row * n_in;
    unsigned nb = n_in / 256;
    unsigned rb = nb * 144u;
    const uint8_t* row0 = (w + w_off) + (size_t)pair0 * rb;

    unsigned ix = lane / 8u;
    unsigned it = lane % 8u;
    unsigned iq = it / 4u;
    unsigned ir = it % 4u;

    float sumf[4] = {0.f, 0.f, 0.f, 0.f};
    const float* y4 = xr + ix * 256u + 64u * iq + 8u * ir;

    for (unsigned ib = ix; ib < nb; ib += 4u) {
        float yl[16], yh[16];
        float sumy0 = 0.f, sumy1 = 0.f, sumy2 = 0.f, sumy3 = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            yl[i] = y4[i];
            sumy0 += yl[i];
            yl[i + 8] = y4[i + 32];
            sumy1 += yl[i + 8];
            yh[i] = y4[i + 128];
            sumy2 += yh[i];
            yh[i + 8] = y4[i + 160];
            sumy3 += yh[i + 8];
        }

        for (unsigned r = 0; r < nrows; r++) {
            const uint8_t* blk = row0 + r * rb + ib * 144u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint16_t* sc = (const uint16_t*)(blk + 4) + iq;
            const uint16_t* q1 = (const uint16_t*)(blk + 16) + 16u * iq + 4u * ir;
            const uint16_t* q2 = q1 + 32;

            uint16_t sc16[4];
            sc16[0] = (uint16_t)(sc[0] & 0x3f3f);
            sc16[1] = (uint16_t)(sc[2] & 0x3f3f);
            sc16[2] = (uint16_t)(((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2));
            sc16[3] = (uint16_t)(((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2));
            const uint8_t* sc8 = (const uint8_t*)sc16;

            float acc1[4] = {0.f, 0.f, 0.f, 0.f};
            float acc2[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint16_t qq1 = q1[i];
                uint16_t qq2 = q2[i];
                acc1[0] += yl[2 * i + 0] * (float)(qq1 & 0x000F);
                acc1[1] += yl[2 * i + 1] * (float)(qq1 & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (float)(qq1 & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (float)(qq1 & 0xF000);
                acc2[0] += yh[2 * i + 0] * (float)(qq2 & 0x000F);
                acc2[1] += yh[2 * i + 1] * (float)(qq2 & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (float)(qq2 & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (float)(qq2 & 0xF000);
            }

            sumf[r] +=
                d * ((acc1[0] + (1.f / 256.f) * acc1[1]) * (float)sc8[0] +
                     (acc1[2] + (1.f / 256.f) * acc1[3]) * (float)sc8[1] * (1.f / 16.f) +
                     (acc2[0] + (1.f / 256.f) * acc2[1]) * (float)sc8[4] +
                     (acc2[2] + (1.f / 256.f) * acc2[3]) * (float)sc8[5] * (1.f / 16.f))
                - dmin * (sumy0 * (float)sc8[2] + sumy1 * (float)sc8[3] +
                          sumy2 * (float)sc8[6] + sumy3 * (float)sc8[7]);
        }
        y4 += 4 * 256;
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) sumf[r] = warp_sum_f(sumf[r]);
    if (lane == 0) {
        float* yr = y + (size_t)row * n_out;
        for (unsigned r = 0; r < nrows; r++) {
            if (add) yr[pair0 + r] += sumf[r];
            else yr[pair0 + r] = sumf[r];
        }
    }
}

// Pack activations to block_q8_1 (32-wide amax scale). One warp per 32-group.
extern "C" __global__ void quantize_q8_1(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    unsigned n, unsigned rows)
{
    unsigned row = blockIdx.y;
    unsigned blk = blockIdx.x;
    unsigned lane = threadIdx.x;
    if (row >= rows) return;
    unsigned base = blk * 32u;
    if (base >= n) return;
    float v = x[(size_t)row * n + base + lane];
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float scale = amax / 127.f;
    if (scale < 1e-8f) scale = 1.f;
    int qi = __float2int_rn(v / scale);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    block_q8_1* b = y + (size_t)row * (n / 32u) + blk;
    b->qs[lane] = (int8_t)qi;
    store_q8_meta(b, lane, qi, scale);
}

extern "C" __global__ void quantize_q8_1_q6(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    unsigned n, unsigned rows)
{
    unsigned row = blockIdx.y;
    unsigned blk = blockIdx.x;
    unsigned lane = threadIdx.x;
    if (row >= rows) return;
    unsigned base = blk * 32u;
    if (base >= n) return;
    float v = x[(size_t)row * n + base + lane];
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float scale = amax / 127.f;
    if (scale < 1e-8f) scale = 1.f;
    int qi = __float2int_rn(v / scale);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    block_q8_1* b = y + (size_t)row * (n / 32u) + blk;
    b->qs[lane] = (int8_t)qi;
    store_q8_scale(b, lane, scale);
}

// Decode FA [hd,n_q] -> out [n_q,hd] + pack Q8 (replaces permute + quantize_q8_1).
extern "C" __global__ void permute_q8_m1(
    const float* __restrict__ in,
    float* __restrict__ out,
    block_q8_1* __restrict__ yq,
    unsigned hd, unsigned n_q)
{
    unsigned blk = blockIdx.x;
    unsigned lane = threadIdx.x;
    unsigned n = hd * n_q;
    unsigned base = blk * 32u;
    if (base >= n) return;
    unsigned out_i = base + lane;
    unsigned d = out_i % hd;
    unsigned h = out_i / hd;
    float v = in[(size_t)d + (size_t)h * hd];
    out[out_i] = v;
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float scale = amax / 127.f;
    if (scale < 1e-8f) scale = 1.f;
    int qi = __float2int_rn(v / scale);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    block_q8_1* b = yq + blk;
    b->qs[lane] = (int8_t)qi;
    store_q8_meta(b, lane, qi, scale);
}

extern "C" __global__ void permute_q8_m1_qonly(
    const float* __restrict__ in,
    block_q8_1* __restrict__ yq,
    unsigned hd, unsigned n_q)
{
    unsigned blk = blockIdx.x;
    unsigned lane = threadIdx.x;
    unsigned n = hd * n_q;
    unsigned base = blk * 32u;
    if (base >= n) return;
    unsigned out_i = base + lane;
    unsigned d = out_i % hd;
    unsigned h = out_i / hd;
    float v = in[(size_t)d + (size_t)h * hd];
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float scale = amax / 127.f;
    if (scale < 1e-8f) scale = 1.f;
    int qi = __float2int_rn(v / scale);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    block_q8_1* b = yq + blk;
    b->qs[lane] = (int8_t)qi;
    store_q8_meta(b, lane, qi, scale);
}

// llama.cpp ggml_cuda_dp4a — INT8×INT8→INT32 SIMD dot.
__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int out;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    return (int)x16[2 * i32] | ((int)x16[2 * i32 + 1] << 16);
}

__device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// Per-byte saturating subtract (CUDA __vsubss4), no special PTX opcode needed.
__device__ __forceinline__ int vsubss4(int a, int b) {
    (void)b;
    return (a + 0x60606060) ^ 0x80808080;
}

// Port of llama vec_dot_q4_K_q8_1 (MMVQ). iqs ∈ {0,2,…,30}.
// X is packed block_q8_1*; partial qs sums via dp4a (do not use block s).
__device__ __forceinline__ float vec_dot_q4_k_q8(
    const uint8_t* __restrict__ blk,
    const block_q8_1* __restrict__ bq8,
    int iqs)
{
    const int bq8_offset = 2 * ((iqs / 2) / 4); // 0,2,4,6
    const int* q4 = (const int*)(blk + 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    int v0 = __ldg(q4);
    int v1 = __ldg(q4 + 4);

    const uint16_t* scales = (const uint16_t*)(blk + 4);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t)-(int32_t)(j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t)(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t)(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* mn = sc + 2;

    float d = half_bits_to_f32(__ldg((const uint16_t*)blk));
    float dmin = half_bits_to_f32(__ldg((const uint16_t*)blk + 1));

    float sumf_d = 0.f;
    float sumf_m = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const block_q8_1* bq8i = bq8 + bq8_offset + i;
        float d8 = half_bits_to_f32(__ldg(&bq8i->d));
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        int u0 = q8[0];
        int u1 = q8[4];
        int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
        int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
        int dot1 = dp4a_s32(v1i, u1, dp4a_s32(v0i, u0, 0));
        int dot2 = (int)bq8i->ps[(iqs / 2) % 4];
        sumf_d += d8 * (float)(dot1 * (int)sc[i]);
        sumf_m += d8 * (float)(dot2 * (int)mn[i]);
    }
    return d * sumf_d - dmin * sumf_m;
}

// Port of llama vec_dot_q6_K_q8_1 (MMVQ). iqs ∈ {0..31}, VDR=1.
__device__ __forceinline__ float vec_dot_q6_k_q8(
    const uint8_t* __restrict__ blk,
    const block_q8_1* __restrict__ bq8,
    int iqs)
{
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);

    const int vl = get_int_b2(blk, iqs);
    const int vh = get_int_b2(blk + 128, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    const int8_t* scales = (const int8_t*)(blk + 192) + scale_offset;
    float d = half_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));

    float sumf = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const block_q8_1* bq8i = bq8 + bq8_offset + 2 * i;
        float d8 = half_bits_to_f32(__ldg(&bq8i->d));
        const int* q8 = (const int*)bq8i->qs + (iqs % 8);
        int u = q8[0];
        int sc = (int)scales[4 * i];
        int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        int vi = vsubss4(vil | vih, 0x20202020);
        sumf += d8 * (float)(dp4a_s32(vi, u, 0) * sc);
    }
    return d * sumf;
}

// llama.cpp GENERIC MMVQ on Blackwell: nwarps=4, rows_per_block=1.
// blocks_per_iter = vdr * nwarps * 32 / qi; Q4_K: vdr=2,qi=32 -> 8.
#define MMVQ_NWARPS 4u

extern "C" __global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8(
    const uint8_t* __restrict__ w,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned tid = warp * 32u + lane;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    pdl_sync();
    if (out >= n_out || row >= m) { pdl_lc(); return; }

    const block_q8_1* xr = x + (size_t)row * (n_in / 32u);
    const uint8_t* wrow = (w + w_off) + (size_t)out * (n_in / 256u) * 144u;
    unsigned nb = n_in / 256u;

    float acc = 0.f;
    for (unsigned kbx = tid / 16u; kbx < nb; kbx += 8u) {
        int iqs = (int)(2u * (tid % 16u));
        acc += vec_dot_q4_k_q8(wrow + kbx * 144u, xr + kbx * 8u, iqs);
    }
    pdl_lc();
    acc = warp_sum_f(acc);

    __shared__ float wacc[MMVQ_NWARPS];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < MMVQ_NWARPS; i++) s += wacc[i];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
    }
}

extern "C" __global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q6_k_q8(
    const uint8_t* __restrict__ w,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned tid = warp * 32u + lane;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    pdl_sync();
    if (out >= n_out || row >= m) { pdl_lc(); return; }

    const block_q8_1* xr = x + (size_t)row * (n_in / 32u);
    const uint8_t* wrow = (w + w_off) + (size_t)out * (n_in / 256u) * 210u;
    unsigned nb = n_in / 256u;

    float acc = 0.f;
    for (unsigned kbx = tid / 32u; kbx < nb; kbx += 4u) {
        int iqs = (int)(tid % 32u);
        acc += vec_dot_q6_k_q8(wrow + kbx * 210u, xr + kbx * 8u, iqs);
    }
    pdl_lc();
    acc = warp_sum_f(acc);

    __shared__ float wacc[MMVQ_NWARPS];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < MMVQ_NWARPS; i++) s += wacc[i];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
    }
}

extern "C" __global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8_2(
    const uint8_t* __restrict__ w0,
    const uint8_t* __restrict__ w1,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1,
    unsigned n_in, unsigned n_out,
    unsigned long long w_off0, unsigned long long w_off1, unsigned m)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned tid = warp * 32u + lane;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    pdl_sync();
    if (out >= n_out || row >= m) { pdl_lc(); return; }
    (void)y1;

    const block_q8_1* xr = x + (size_t)row * (n_in / 32u);
    unsigned rb = (n_in / 256u) * 144u;
    const uint8_t* row0 = (w0 + w_off0) + (size_t)out * rb;
    const uint8_t* row1 = (w1 + w_off1) + (size_t)out * rb;
    unsigned nb = n_in / 256u;

    float acc0 = 0.f, acc1 = 0.f;
    for (unsigned kbx = tid / 16u; kbx < nb; kbx += 8u) {
        int iqs = (int)(2u * (tid % 16u));
        const block_q8_1* xb = xr + kbx * 8u;
        acc0 += vec_dot_q4_k_q8(row0 + kbx * 144u, xb, iqs);
        acc1 += vec_dot_q4_k_q8(row1 + kbx * 144u, xb, iqs);
    }
    pdl_lc();
    acc0 = warp_sum_f(acc0);
    acc1 = warp_sum_f(acc1);

    __shared__ float wacc0[MMVQ_NWARPS], wacc1[MMVQ_NWARPS];
    if (lane == 0) {
        wacc0[warp] = acc0;
        wacc1[warp] = acc1;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s0 = 0.f, s1 = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < MMVQ_NWARPS; i++) {
            s0 += wacc0[i];
            s1 += wacc1[i];
        }
        float* yr = y0 + (size_t)row * n_out;
        yr[out] = silu_f(s0) * s1;
    }
}

// Fused Q/K/V Q8 mmvq: one X quant, three weight rows (n_q may exceed n_kv).
extern "C" __global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8_qkv(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv,
    const block_q8_1* __restrict__ x,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    unsigned n_in, unsigned n_q, unsigned n_kv,
    unsigned long long oq, unsigned long long ok, unsigned long long ov, unsigned m,
    uint16_t* __restrict__ cache,
    unsigned long long v_cache_off,
    const unsigned* __restrict__ d_pos,
    unsigned kv_dim)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned tid = warp * 32u + lane;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    pdl_sync();
    if (row >= m) { pdl_lc(); return; }
    bool do_q = out < n_q;
    bool do_kv = out < n_kv;
    if (!do_q && !do_kv) { pdl_lc(); return; }

    const block_q8_1* xr = x + (size_t)row * (n_in / 32u);
    unsigned nb = n_in / 256u;
    unsigned rb = nb * 144u;
    const uint8_t* wrow_q = do_q ? (wq + oq) + (size_t)out * rb : nullptr;
    const uint8_t* wrow_k = do_kv ? (wk + ok) + (size_t)out * rb : nullptr;
    const uint8_t* wrow_v = do_kv ? (wv + ov) + (size_t)out * rb : nullptr;

    float acc_q = 0.f, acc_k = 0.f, acc_v = 0.f;
    for (unsigned kbx = tid / 16u; kbx < nb; kbx += 8u) {
        int iqs = (int)(2u * (tid % 16u));
        const block_q8_1* xb = xr + kbx * 8u;
        if (do_q) acc_q += vec_dot_q4_k_q8(wrow_q + kbx * 144u, xb, iqs);
        if (do_kv) {
            acc_k += vec_dot_q4_k_q8(wrow_k + kbx * 144u, xb, iqs);
            acc_v += vec_dot_q4_k_q8(wrow_v + kbx * 144u, xb, iqs);
        }
    }
    pdl_lc();
    acc_q = warp_sum_f(acc_q);
    acc_k = warp_sum_f(acc_k);
    acc_v = warp_sum_f(acc_v);

    __shared__ float sq[MMVQ_NWARPS], sk[MMVQ_NWARPS], sv[MMVQ_NWARPS];
    if (lane == 0) {
        sq[warp] = acc_q;
        sk[warp] = acc_k;
        sv[warp] = acc_v;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float aq = 0.f, ak = 0.f, av = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < MMVQ_NWARPS; i++) {
            aq += sq[i];
            ak += sk[i];
            av += sv[i];
        }
        if (do_q) q[(size_t)row * n_q + out] = aq;
        if (do_kv) {
            k[(size_t)row * n_kv + out] = ak;
            v[(size_t)row * n_kv + out] = av;
            if (cache && m == 1u && d_pos) {
                unsigned pos = d_pos[0];
                cache[v_cache_off + (unsigned long long)pos * kv_dim + out] =
                    f32_to_half_bits(av);
            }
        }
    }
}

extern "C" __global__ void matvec_q5_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    unsigned lane = threadIdx.x;
    unsigned local = threadIdx.y;
    unsigned out = blockIdx.x * blockDim.y + local;
    unsigned row = blockIdx.y;
    bool active = out < n_out && row < m;
    unsigned out_s = active ? out : 0;
    unsigned row_s = row < m ? row : 0;
    const float* xr = x + (size_t)row_s * n_in;
    float* yr = y + (size_t)row_s * n_out;
    const uint8_t* wrow = (w + w_off) + (size_t)out_s * (n_in / 256) * 176;
    unsigned tid = local * 32u + lane;
    __shared__ float xs[256];
    float acc = 0.f;
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        xs[tid] = xr[b * 256 + tid];
        __syncthreads();
        const uint8_t* blk = wrow + b * 176;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t* packed = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* qs = blk + 48;
        float partial = 0.f;
        for (unsigned pair = 0; pair < 4; pair++) {
            float mn1, mn2;
            float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
            float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
            uint8_t bit1 = (uint8_t)(1u << (pair * 2));
            uint8_t bit2 = (uint8_t)(1u << (pair * 2 + 1));
            uint8_t qq = qs[pair * 32 + lane];
            int hi1 = (qh[lane] & bit1) ? 16 : 0;
            int hi2 = (qh[lane] & bit2) ? 16 : 0;
            float xl = xs[pair * 64 + lane];
            float xh = xs[pair * 64 + 32 + lane];
            partial += (d * sc1 * (float)((qq & 0xf) + hi1) - dmin * mn1) * xl;
            partial += (d * sc2 * (float)((qq >> 4) + hi2) - dmin * mn2) * xh;
        }
        acc += warp_sum_f(partial);
        __syncthreads();
    }
    if (lane == 0 && active) yr[out] = acc;
}

// Metal-style Q6_K mmvq: 4 output rows per warp, X in registers.
extern "C" __global__ void matvec_q6_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned row = blockIdx.y;
    if (row >= m) return;
    // 4 warps × 4 rows = 16 output rows per block.
    unsigned pair0 = blockIdx.x * 16u + warp * 4u;
    if (pair0 >= n_out) return;
    unsigned nrows = n_out - pair0;
    if (nrows > 4u) nrows = 4u;

    const float* xr = x + (size_t)row * n_in;
    unsigned nb = n_in / 256;
    unsigned rb = nb * 210u;
    const uint8_t* row0 = (w + w_off) + (size_t)pair0 * rb;

    unsigned t = lane / 2u;
    unsigned ix = lane % 2u;
    unsigned ip = t / 8u;
    unsigned il = t % 8u;
    unsigned l0 = 4u * il;
    unsigned is = 8u * ip + l0 / 16u;
    unsigned y_offset = 128u * ip + l0;
    unsigned q_offset_l = 64u * ip + l0;
    unsigned q_offset_h = 32u * ip + l0;

    float sumf[4] = {0.f, 0.f, 0.f, 0.f};

    for (unsigned ib = ix; ib < nb; ib += 2u) {
        float yl[16];
        const float* yv = xr + ib * 256u + y_offset;
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            yl[4 * l + 0] = yv[l + 0];
            yl[4 * l + 1] = yv[l + 32];
            yl[4 * l + 2] = yv[l + 64];
            yl[4 * l + 3] = yv[l + 96];
        }

        for (unsigned r = 0; r < nrows; r++) {
            const uint8_t* blk = row0 + r * rb + ib * 210u;
            const uint8_t* q1 = blk + q_offset_l;
            const uint8_t* q2 = q1 + 32;
            const uint8_t* qh = blk + 128 + q_offset_h;
            const int8_t* sc = (const int8_t*)(blk + 192) + is;
            float d = half_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));

            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                s0 += yl[4 * l + 0] * (float)((int)((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                s1 += yl[4 * l + 1] * (float)((int)((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                s2 += yl[4 * l + 2] * (float)((int)((q1[l] >> 4) | ((qh[l] & 0x30) << 0)) - 32);
                s3 += yl[4 * l + 3] * (float)((int)((q2[l] >> 4) | ((qh[l] & 0xC0) >> 2)) - 32);
            }
            sumf[r] += d * (s0 * (float)sc[0] + s1 * (float)sc[2]
                          + s2 * (float)sc[4] + s3 * (float)sc[6]);
        }
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) sumf[r] = warp_sum_f(sumf[r]);
    if (lane == 0) {
        float* yr = y + (size_t)row * n_out;
        for (unsigned r = 0; r < nrows; r++) {
            if (add) yr[pair0 + r] += sumf[r];
            else yr[pair0 + r] = sumf[r];
        }
    }
}

extern "C" __global__ void matvec_q2_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    MV_IDX();
    const uint8_t* wrow = wb + (size_t)out * (n_in / 256) * 84;
    float acc = 0.f;
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 84;
        const uint8_t* scales = blk;
        const uint8_t* qs = blk + 16;
        float d = half_bits_to_f32((uint16_t)blk[80] | ((uint16_t)blk[81] << 8));
        float dmin = half_bits_to_f32((uint16_t)blk[82] | ((uint16_t)blk[83] << 8));
        const float* xb = xr + b * 256;
        unsigned is = 0;
        unsigned xo = 0;
        for (unsigned hi = 0; hi < 2; hi++) {
            const uint8_t* q = qs + hi * 32;
            for (unsigned shift = 0; shift < 8; shift += 2) {
                for (unsigned group = 0; group < 2; group++) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0xf);
                    float ml = dmin * (float)(sc >> 4);
                    for (unsigned l = 0; l < 16; l++) {
                        float quant = (float)((q[group * 16 + l] >> shift) & 3);
                        acc += (dl * quant - ml) * xb[xo++];
                    }
                }
            }
        }
    }
    yr[out] = acc;
}

extern "C" __global__ void matvec_q3_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    MV_IDX();
    const uint8_t* wrow = wb + (size_t)out * (n_in / 256) * 110;
    float acc = 0.f;
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 110;
        const uint8_t* hmask = blk;
        const uint8_t* qs = blk + 32;
        const uint8_t* packed = blk + 96;
        float d_all = half_bits_to_f32((uint16_t)blk[108] | ((uint16_t)blk[109] << 8));
        int scales[16];
        for (unsigned i = 0; i < 16; i++) {
            unsigned lo = (i < 8) ? (packed[i] & 0xf) : (packed[i - 8] >> 4);
            unsigned hi = (packed[8 + i % 4] >> (2 * (i / 4))) & 3;
            scales[i] = (int)(lo | (hi << 4)) - 32;
        }
        const float* xb = xr + b * 256;
        unsigned is = 0;
        unsigned xo = 0;
        uint8_t mbit = 1;
        for (unsigned hi = 0; hi < 2; hi++) {
            const uint8_t* q = qs + hi * 32;
            for (unsigned shift = 0; shift < 8; shift += 2) {
                for (unsigned group = 0; group < 2; group++) {
                    float dl = d_all * (float)scales[is++];
                    for (unsigned l = 0; l < 16; l++) {
                        unsigned idx = group * 16 + l;
                        int quant = (int)((q[idx] >> shift) & 3)
                            - ((hmask[idx] & mbit) ? 0 : 4);
                        acc += dl * (float)quant * xb[xo++];
                    }
                }
                mbit <<= 1;
            }
        }
    }
    yr[out] = acc;
}

// ---- dequant rows to f16 for cuBLASLt GEMM ---------------------------------

__device__ __forceinline__ void dq_row_q8_0_f16(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    unsigned nb = n_in / 32;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 34;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        for (unsigned i = 0; i < 32; i++) {
            out[b * 32 + i] = f32_to_half_bits(d * (float)((int8_t)blk[2 + i]));
        }
    }
}

__device__ void dq_row_q4_k_f16(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 144;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t* packed = blk + 4;
        const uint8_t* qs = blk + 16;
        uint16_t* o = out + b * 256;
        for (unsigned pair = 0; pair < 4; pair++) {
            const uint8_t* q = qs + pair * 32;
            float mn1, mn2;
            float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
            float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
            for (unsigned l = 0; l < 32; l++) {
                o[pair * 64 + l] = f32_to_half_bits(d * sc1 * (float)(q[l] & 0xf) - dmin * mn1);
            }
            for (unsigned l = 0; l < 32; l++) {
                o[pair * 64 + 32 + l] = f32_to_half_bits(d * sc2 * (float)(q[l] >> 4) - dmin * mn2);
            }
        }
    }
}

__device__ void dq_row_q5_k_f16(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 176;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t* packed = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* qs = blk + 48;
        uint16_t* o = out + b * 256;
        for (unsigned pair = 0; pair < 4; pair++) {
            const uint8_t* q = qs + pair * 32;
            float mn1, mn2;
            float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
            float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
            uint8_t bit1 = (uint8_t)(1u << (pair * 2));
            uint8_t bit2 = (uint8_t)(1u << (pair * 2 + 1));
            for (unsigned l = 0; l < 32; l++) {
                int hi = (qh[l] & bit1) ? 16 : 0;
                o[pair * 64 + l] = f32_to_half_bits(d * sc1 * (float)((q[l] & 0xf) + hi) - dmin * mn1);
            }
            for (unsigned l = 0; l < 32; l++) {
                int hi = (qh[l] & bit2) ? 16 : 0;
                o[pair * 64 + 32 + l] = f32_to_half_bits(d * sc2 * (float)((q[l] >> 4) + hi) - dmin * mn2);
            }
        }
    }
}

__device__ void dq_row_q6_k_f16(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* scales = (const int8_t*)(blk + 192);
        float d = half_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
        uint16_t* o = out + b * 256;
        for (unsigned halfb = 0; halfb < 2; halfb++) {
            const uint8_t* ql_h = ql + halfb * 64;
            const uint8_t* qh_h = qh + halfb * 32;
            const int8_t* sc = scales + halfb * 8;
            uint16_t* oh = o + halfb * 128;
            for (unsigned l = 0; l < 32; l++) {
                unsigned is = l / 16;
                int q1 = (int)((ql_h[l] & 0xf) | ((qh_h[l] & 3) << 4)) - 32;
                int q2 = (int)((ql_h[l + 32] & 0xf) | (((qh_h[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32;
                oh[l] = f32_to_half_bits(d * (float)sc[is] * (float)q1);
                oh[32 + l] = f32_to_half_bits(d * (float)sc[is + 2] * (float)q2);
                oh[64 + l] = f32_to_half_bits(d * (float)sc[is + 4] * (float)q3);
                oh[96 + l] = f32_to_half_bits(d * (float)sc[is + 6] * (float)q4);
            }
        }
    }
}

__device__ void dq_row_q2_k_f16(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 84;
        const uint8_t* scales = blk;
        const uint8_t* qs = blk + 16;
        float d = half_bits_to_f32((uint16_t)blk[80] | ((uint16_t)blk[81] << 8));
        float dmin = half_bits_to_f32((uint16_t)blk[82] | ((uint16_t)blk[83] << 8));
        uint16_t* o = out + b * 256;
        unsigned is = 0, xo = 0;
        for (unsigned halfb = 0; halfb < 2; halfb++) {
            const uint8_t* q = qs + halfb * 32;
            for (unsigned shift = 0; shift < 8; shift += 2) {
                for (unsigned group = 0; group < 2; group++) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0xf);
                    float ml = dmin * (float)(sc >> 4);
                    for (unsigned l = 0; l < 16; l++) {
                        float quant = (float)((q[group * 16 + l] >> shift) & 3);
                        o[xo++] = f32_to_half_bits(dl * quant - ml);
                    }
                }
            }
        }
    }
}

__device__ void dq_row_q3_k_f16(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 110;
        const uint8_t* hmask = blk;
        const uint8_t* qs = blk + 32;
        const uint8_t* packed = blk + 96;
        float d_all = half_bits_to_f32((uint16_t)blk[108] | ((uint16_t)blk[109] << 8));
        int scales[16];
        for (unsigned i = 0; i < 16; i++) {
            unsigned lo = (i < 8) ? (packed[i] & 0xf) : (packed[i - 8] >> 4);
            unsigned hi = (packed[8 + i % 4] >> (2 * (i / 4))) & 3;
            scales[i] = (int)(lo | (hi << 4)) - 32;
        }
        uint16_t* o = out + b * 256;
        unsigned is = 0, xo = 0;
        uint8_t mbit = 1;
        for (unsigned halfb = 0; halfb < 2; halfb++) {
            const uint8_t* q = qs + halfb * 32;
            for (unsigned shift = 0; shift < 8; shift += 2) {
                for (unsigned group = 0; group < 2; group++) {
                    float dl = d_all * (float)scales[is++];
                    for (unsigned l = 0; l < 16; l++) {
                        unsigned idx = group * 16 + l;
                        int quant = (int)((q[idx] >> shift) & 3)
                            - ((hmask[idx] & mbit) ? 0 : 4);
                        o[xo++] = f32_to_half_bits(dl * (float)quant);
                    }
                }
                mbit <<= 1;
            }
        }
    }
}

__device__ void dq_row_q5_0_f16(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    unsigned nb = n_in / 32;
    for (unsigned b = 0; b < nb; b++) {
        const uint8_t* blk = wrow + b * 22;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
            | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
        const uint8_t* qs = blk + 6;
        uint16_t* o = out + b * 32;
        for (unsigned j = 0; j < 16; j++) {
            int x0 = (int)((qs[j] & 0x0f) | (((qh >> j) & 1) << 4)) - 16;
            int x1 = (int)((qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
            o[j] = f32_to_half_bits(d * (float)x0);
            o[j + 16] = f32_to_half_bits(d * (float)x1);
        }
    }
}

__device__ void dq_row_f16(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    const uint16_t* src = (const uint16_t*)wrow;
    for (unsigned i = 0; i < n_in; i++) out[i] = src[i];
}

__device__ void dq_row_f32(const uint8_t* wrow, unsigned n_in, uint16_t* out) {
    const float* src = (const float*)wrow;
    for (unsigned i = 0; i < n_in; i++) out[i] = f32_to_half_bits(src[i]);
}

// fmt: 0=q4_k 1=q5_k 2=q6_k 3=q8_0 4=q2_k 5=q3_k 6=q5_0 7=f16 8=f32 9=bf16
//
// K-quants (fmt 0..2): grid (ceil(n_out/ROWS), n_in/256), block (32, ROWS).
// One warp owns one 256-wide superblock and consecutive lanes write consecutive
// f16s (coalesced). Older layout assigned a whole superblock to one lane, so
// stores were hundreds of bytes apart and Q6_K fell back to 1 thread/row.
//
// Other fmts: grid (ceil(n_out/ROWS), 1), block (32, ROWS); lane 0 does the row.
extern "C" __global__ void dequant_rows_f16(
    const uint8_t* __restrict__ w,
    uint16_t* __restrict__ out,          // [n_out, n_in]
    unsigned n_in, unsigned n_out,
    unsigned long long w_off,
    unsigned row_bytes,
    unsigned fmt)
{
    const unsigned ROWS = 16u;
    unsigned r = blockIdx.x * ROWS + threadIdx.y;
    unsigned lane = threadIdx.x;
    if (r >= n_out) return;
    const uint8_t* row = w + w_off + (size_t)r * row_bytes;
    uint16_t* o = out + (size_t)r * n_in;

    if (fmt <= 2u) {
        unsigned b = blockIdx.y;
        uint16_t* o256 = o + b * 256u;
        if (fmt == 0u) {
            const uint8_t* blk = row + b * 144u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* packed = blk + 4;
            const uint8_t* qs = blk + 16;
            #pragma unroll
            for (unsigned pair = 0; pair < 4u; pair++) {
                float mn1, mn2;
                float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
                float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
                uint8_t qq = qs[pair * 32u + lane];
                o256[pair * 64u + lane] =
                    f32_to_half_bits(d * sc1 * (float)(qq & 0xf) - dmin * mn1);
                o256[pair * 64u + 32u + lane] =
                    f32_to_half_bits(d * sc2 * (float)(qq >> 4) - dmin * mn2);
            }
            return;
        }
        if (fmt == 1u) {
            const uint8_t* blk = row + b * 176u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* packed = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = blk + 48;
            #pragma unroll
            for (unsigned pair = 0; pair < 4u; pair++) {
                float mn1, mn2;
                float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
                float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
                uint8_t bit1 = (uint8_t)(1u << (pair * 2));
                uint8_t bit2 = (uint8_t)(1u << (pair * 2 + 1));
                uint8_t qq = qs[pair * 32u + lane];
                int hi1 = (qh[lane] & bit1) ? 16 : 0;
                int hi2 = (qh[lane] & bit2) ? 16 : 0;
                o256[pair * 64u + lane] =
                    f32_to_half_bits(d * sc1 * (float)((qq & 0xf) + hi1) - dmin * mn1);
                o256[pair * 64u + 32u + lane] =
                    f32_to_half_bits(d * sc2 * (float)((qq >> 4) + hi2) - dmin * mn2);
            }
            return;
        }
        // Q6_K
        {
            const uint8_t* blk = row + b * 210u;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = (const int8_t*)(blk + 192);
            float d = half_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            #pragma unroll
            for (unsigned hi = 0; hi < 2u; hi++) {
                const uint8_t* ql_h = ql + hi * 64u;
                const uint8_t* qh_h = qh + hi * 32u;
                const int8_t* sc = scales + hi * 8;
                uint16_t* oh = o256 + hi * 128u;
                unsigned is = lane / 16u;
                int q1 = (int)((ql_h[lane] & 0xf) | ((qh_h[lane] & 3) << 4)) - 32;
                int q2 = (int)((ql_h[lane + 32] & 0xf) | (((qh_h[lane] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_h[lane] >> 4) | (((qh_h[lane] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_h[lane + 32] >> 4) | (((qh_h[lane] >> 6) & 3) << 4)) - 32;
                oh[lane] = f32_to_half_bits(d * (float)sc[is] * (float)q1);
                oh[32u + lane] = f32_to_half_bits(d * (float)sc[is + 2] * (float)q2);
                oh[64u + lane] = f32_to_half_bits(d * (float)sc[is + 4] * (float)q3);
                oh[96u + lane] = f32_to_half_bits(d * (float)sc[is + 6] * (float)q4);
            }
        }
        return;
    }

    if (lane != 0) return;
    switch (fmt) {
        case 3: dq_row_q8_0_f16(row, n_in, o); break;
        case 4: dq_row_q2_k_f16(row, n_in, o); break;
        case 5: dq_row_q3_k_f16(row, n_in, o); break;
        case 6: dq_row_q5_0_f16(row, n_in, o); break;
        case 7: dq_row_f16(row, n_in, o); break;
        case 8: dq_row_f32(row, n_in, o); break;
        case 9: {
            const uint16_t* src = (const uint16_t*)row;
            for (unsigned i = 0; i < n_in; i++) {
                o[i] = f32_to_half_bits(__uint_as_float(((uint32_t)src[i]) << 16));
            }
            break;
        }
        default: break;
    }
}

// Apply bias then optional head rmsnorm for Q/K, then rope, store K/V as f16.
// Simplified host-orchestrated helper used from Rust in pieces; kept for
// potential fused paths.
extern "C" __global__ void add_bias_f32(
    float* __restrict__ x,
    const float* __restrict__ bias,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += bias[i];
}

extern "C" __global__ void gather_rows_f32(
    float* __restrict__ dst,       // [total_rows, cols]
    const float* __restrict__ src, // [m, cols]
    const unsigned* __restrict__ tok,
    unsigned total_rows,
    unsigned cols)
{
    unsigned r = blockIdx.y;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total_rows || i >= cols) return;
    dst[(size_t)r * cols + i] = src[(size_t)tok[r] * cols + i];
}

// C[m, n] = A[m, k] @ B[n, k]^T  (A,B f16; C f32). One thread per C element.
extern "C" __global__ void gemm_f16_f32(
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    float* __restrict__ c,
    unsigned m, unsigned n, unsigned k)
{
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.f;
    const uint16_t* ar = a + (size_t)row * k;
    const uint16_t* bc = b + (size_t)col * k;
    for (unsigned i = 0; i < k; i++) {
        acc += half_bits_to_f32(ar[i]) * half_bits_to_f32(bc[i]);
    }
    c[(size_t)row * n + col] = acc;
}

// ---- int8 Tensor Core MMQ for Q4_K (m16n8k32) --------------------------------
// Host pre-quantizes X to Q8 once (avoids ~n_out/BN redundant amax passes).
// Block: 8 warps × 8 cols = 16×64 output tile. Opt-in via ALLPAKA_MMQ=1.

__device__ __forceinline__ unsigned pack_i8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (unsigned)(uint8_t)a
        | ((unsigned)(uint8_t)b << 8)
        | ((unsigned)(uint8_t)c << 16)
        | ((unsigned)(uint8_t)d << 24);
}

__device__ __forceinline__ void mma_m16n8k32_s8(
    int& d0, int& d1, int& d2, int& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern "C" __global__ void mm_q4_k_mma(
    const uint8_t* __restrict__ w,
    const int8_t* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    // grid (ceil(n_out/64), ceil(m/16)), block (32, 8) — 8 warps × 8 cols.
    const unsigned warp = threadIdx.y;
    const unsigned lane = threadIdx.x;
    const unsigned tid = warp * 32u + lane;
    const unsigned col0 = blockIdx.x * 64u + warp * 8u;
    const unsigned row0 = blockIdx.y * 16u;
    if (row0 >= m) return;

    const unsigned group = lane >> 2;
    const unsigned tig = lane & 3u;
    const unsigned nb = n_in / 256u;
    const unsigned rb = nb * 144u;
    const unsigned nd = n_in / 32u;
    const uint8_t* wb = w + w_off;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    __shared__ int8_t As[16 * 256];
    __shared__ float Xd[16 * 8];
    __shared__ float Xsum[16 * 8];
    __shared__ uint8_t Wtile[64 * 144];
    __shared__ int8_t Bs[8 * 32 * 8];
    __shared__ float Wd[64], Wdmin[64], Wsc[64 * 8], Wmn[64 * 8];

    for (unsigned b = 0; b < nb; b++) {
        for (unsigned rg = tid; rg < 16u * 8u; rg += 256u) {
            unsigned r = rg / 8u;
            unsigned g = rg - r * 8u;
            unsigned gr = row0 + r;
            bool ok = gr < m;
            float d = ok ? xd[(size_t)gr * nd + b * 8u + g] : 1.f;
            if (d < 1e-8f) d = 1.f;
            Xd[r * 8u + g] = d;
            int sumq = 0;
            const int8_t* src = ok ? (xq + (size_t)gr * n_in + b * 256u + g * 32u) : 0;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int8_t qi = ok ? src[i] : (int8_t)0;
                As[r * 256u + g * 32u + (unsigned)i] = qi;
                sumq += (int)qi;
            }
            Xsum[r * 8u + g] = d * (float)sumq;
        }

        {
            unsigned base_c = blockIdx.x * 64u;
            for (unsigned off = tid; off < 64u * 144u; off += 256u) {
                unsigned c = off / 144u;
                unsigned o = off - c * 144u;
                unsigned gc = base_c + c;
                Wtile[off] = (gc < n_out) ? wb[(size_t)gc * rb + b * 144u + o] : (uint8_t)0;
            }
        }
        __syncthreads();

        for (unsigned c = tid; c < 64u; c += 256u) {
            const uint8_t* blk = Wtile + c * 144u;
            Wd[c] = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            Wdmin[c] = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                float mn;
                float sc = scale_min_k4(g, blk + 4, &mn);
                Wsc[c * 8 + g] = sc;
                Wmn[c * 8 + g] = mn;
            }
        }
        __syncthreads();

        int8_t* Bsw = Bs + warp * 32u * 8u;
        const unsigned wc_base = warp * 8u;
        #pragma unroll
        for (unsigned g = 0; g < 8u; g++) {
            unsigned pair = g / 2u;
            unsigned shift = (g & 1u) * 4u;
            #pragma unroll
            for (unsigned c = 0; c < 8u; c++) {
                const uint8_t* blk = Wtile + (wc_base + c) * 144u;
                uint8_t qq = blk[16 + pair * 32u + lane];
                Bsw[lane * 8u + c] = (int8_t)((qq >> shift) & 0xf);
            }
            __syncwarp();

            const int8_t* Asg = As + g * 32u;
            int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
            {
                unsigned a0 = pack_i8x4(
                    Asg[group * 256u + 4u * tig + 0],
                    Asg[group * 256u + 4u * tig + 1],
                    Asg[group * 256u + 4u * tig + 2],
                    Asg[group * 256u + 4u * tig + 3]);
                unsigned a1 = pack_i8x4(
                    Asg[(group + 8u) * 256u + 4u * tig + 0],
                    Asg[(group + 8u) * 256u + 4u * tig + 1],
                    Asg[(group + 8u) * 256u + 4u * tig + 2],
                    Asg[(group + 8u) * 256u + 4u * tig + 3]);
                unsigned a2 = pack_i8x4(
                    Asg[group * 256u + 4u * tig + 16],
                    Asg[group * 256u + 4u * tig + 17],
                    Asg[group * 256u + 4u * tig + 18],
                    Asg[group * 256u + 4u * tig + 19]);
                unsigned a3 = pack_i8x4(
                    Asg[(group + 8u) * 256u + 4u * tig + 16],
                    Asg[(group + 8u) * 256u + 4u * tig + 17],
                    Asg[(group + 8u) * 256u + 4u * tig + 18],
                    Asg[(group + 8u) * 256u + 4u * tig + 19]);
                unsigned b0 = pack_i8x4(
                    Bsw[(4u * tig + 0) * 8u + group],
                    Bsw[(4u * tig + 1) * 8u + group],
                    Bsw[(4u * tig + 2) * 8u + group],
                    Bsw[(4u * tig + 3) * 8u + group]);
                unsigned b1 = pack_i8x4(
                    Bsw[(4u * tig + 16) * 8u + group],
                    Bsw[(4u * tig + 17) * 8u + group],
                    Bsw[(4u * tig + 18) * 8u + group],
                    Bsw[(4u * tig + 19) * 8u + group]);
                mma_m16n8k32_s8(dot0, dot1, dot2, dot3, a0, a1, a2, a3, b0, b1);
            }

            {
                float xd0 = Xd[group * 8u + g], xd1 = Xd[(group + 8u) * 8u + g];
                float xs0 = Xsum[group * 8u + g], xs1 = Xsum[(group + 8u) * 8u + g];
                unsigned wc0 = wc_base + 2u * tig;
                unsigned wc1 = wc_base + 2u * tig + 1u;
                acc0 += (Wd[wc0] * Wsc[wc0 * 8u + g] * xd0) * (float)dot0
                    - (Wdmin[wc0] * Wmn[wc0 * 8u + g]) * xs0;
                acc1 += (Wd[wc1] * Wsc[wc1 * 8u + g] * xd0) * (float)dot1
                    - (Wdmin[wc1] * Wmn[wc1 * 8u + g]) * xs0;
                acc2 += (Wd[wc0] * Wsc[wc0 * 8u + g] * xd1) * (float)dot2
                    - (Wdmin[wc0] * Wmn[wc0 * 8u + g]) * xs1;
                acc3 += (Wd[wc1] * Wsc[wc1 * 8u + g] * xd1) * (float)dot3
                    - (Wdmin[wc1] * Wmn[wc1 * 8u + g]) * xs1;
            }
            __syncwarp();
        }
        __syncthreads();
    }

    if (col0 < n_out) {
        unsigned r0 = row0 + group;
        unsigned r1 = row0 + group + 8u;
        unsigned c0 = col0 + 2u * tig;
        unsigned c1 = col0 + 2u * tig + 1u;
        if (r0 < m && c0 < n_out) y[(size_t)r0 * n_out + c0] = acc0;
        if (r0 < m && c1 < n_out) y[(size_t)r0 * n_out + c1] = acc1;
        if (r1 < m && c0 < n_out) y[(size_t)r1 * n_out + c0] = acc2;
        if (r1 < m && c1 < n_out) y[(size_t)r1 * n_out + c1] = acc3;
    }
}

// Metal-style tiled Q4_K GEMM: dequant W tiles into smem (never full f16 W in HBM).
// Tile BM=32 × BN=64 × BK=32; half staging, float accum. Opt-in ALLPAKA_MMQ=1.
extern "C" __global__ void mm_q4_k_tile(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    constexpr unsigned BM = 32u, BN = 64u, BK = 32u;
    const unsigned tid = threadIdx.y * 32u + threadIdx.x;
    const unsigned row0 = blockIdx.y * BM;
    const unsigned col0 = blockIdx.x * BN;
    if (row0 >= m) return;

    const unsigned nb = n_in / 256u;
    const unsigned rb = nb * 144u;
    const uint8_t* wb = w + w_off;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BN * BK];

    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.f;

    for (unsigned k0 = 0; k0 < n_in; k0 += BK) {
        for (unsigned off = tid; off < BM * BK; off += 256u) {
            unsigned i = off / BK;
            unsigned kk = off - i * BK;
            unsigned gr = row0 + i;
            As[off] = (gr < m && (k0 + kk) < n_in) ? x[(size_t)gr * n_in + k0 + kk] : 0.f;
        }

        {
            unsigned g = (k0 / 32u) & 7u;
            unsigned b = k0 / 256u;
            unsigned pair = g / 2u;
            unsigned shift = (g & 1u) * 4u;
            for (unsigned off = tid; off < BN * BK; off += 256u) {
                unsigned c = off / BK;
                unsigned kk = off - c * BK;
                unsigned gc = col0 + c;
                float v = 0.f;
                if (gc < n_out && b < nb) {
                    const uint8_t* blk = wb + (size_t)gc * rb + b * 144u;
                    float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                    float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
                    float mn;
                    float sc = scale_min_k4((int)g, blk + 4, &mn);
                    uint8_t qq = blk[16 + pair * 32u + kk];
                    int qv = (int)((qq >> shift) & 0xf);
                    v = d * sc * (float)qv - dmin * mn;
                }
                Bs[off] = v;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < 8; t++) {
            unsigned idx = tid + (unsigned)t * 256u;
            unsigned i = idx / BN;
            unsigned j = idx - i * BN;
            if (i < BM) {
                float sum = acc[t];
                #pragma unroll
                for (unsigned kk = 0; kk < BK; kk++) {
                    sum += As[i * BK + kk] * Bs[j * BK + kk];
                }
                acc[t] = sum;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < 8; t++) {
        unsigned idx = tid + (unsigned)t * 256u;
        unsigned i = idx / BN;
        unsigned j = idx - i * BN;
        unsigned gr = row0 + i;
        unsigned gc = col0 + j;
        if (i < BM && gr < m && gc < n_out) {
            y[(size_t)gr * n_out + gc] = acc[t];
        }
    }
}

// Tiled Q4_K × Q8(X) GEMM (dp4a, no TC). Kept for debugging; slower than cuBLAS.
extern "C" __global__ void mm_q4_k_q8(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    const unsigned BM = 16u;
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned tid = warp * 32u + lane;
    unsigned col = blockIdx.x * 32u + lane;
    unsigned row0 = blockIdx.y * BM;
    unsigned gr = row0 + warp;
    const uint8_t* wb = w + w_off;
    unsigned nb = n_in / 256u;
    unsigned row_bytes = nb * 144u;

    // 32 Q4_K blocks (one per output col) + BM rows × 8 block_q8_1.
    __shared__ uint8_t wtile[32 * 144];
    __shared__ block_q8_1 xq[16 * 8];

    float acc = 0.f;
    bool row_ok = gr < m;
    bool col_ok = col < n_out;

    for (unsigned b = 0; b < nb; b++) {
        // Cooperative W load: 32 cols × 144 bytes.
        for (unsigned off = tid; off < 32u * 144u; off += BM * 32u) {
            unsigned c = off / 144u;
            unsigned o = off - c * 144u;
            unsigned gcol = blockIdx.x * 32u + c;
            wtile[off] = (gcol < n_out)
                ? wb[(size_t)gcol * row_bytes + b * 144u + o]
                : 0;
        }
        // Each warp quantizes its X row for this 256-block into block_q8_1.
        if (warp < BM) {
            #pragma unroll
            for (unsigned g = 0; g < 8u; g++) {
                float v = row_ok ? x[(size_t)gr * n_in + b * 256u + g * 32u + lane] : 0.f;
                float amax = fabsf(v);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
                }
                float d = amax / 127.f;
                if (d < 1e-8f) d = 1.f;
                int qi = __float2int_rn(v / d);
                if (qi > 127) qi = 127;
                if (qi < -127) qi = -127;
                block_q8_1* bq = &xq[warp * 8u + g];
                bq->qs[lane] = (int8_t)qi;
                store_q8_meta(bq, lane, qi, d);
            }
        }
        __syncthreads();

        if (row_ok && col_ok) {
            const uint8_t* blk = wtile + lane * 144u;
            const block_q8_1* xb = xq + warp * 8u;
            #pragma unroll
            for (int iqs = 0; iqs < 32; iqs += 2) {
                acc += vec_dot_q4_k_q8(blk, xb, iqs);
            }
        }
        __syncthreads();
    }

    if (row_ok && col_ok) {
        y[(size_t)gr * n_out + col] = acc;
    }
}

// Float-X tiled Q4_K GEMM (accurate, slower than cuBLAS). Kept for debugging.
extern "C" __global__ void mm_q4_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    const unsigned BM = 8u;
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned col = blockIdx.x * 32u + lane;
    unsigned row0 = blockIdx.y * BM;
    unsigned gr = row0 + warp;
    const uint8_t* wb = w + w_off;
    unsigned nb = n_in / 256u;
    unsigned row_bytes = nb * 144u;

    __shared__ float xs[8 * 256];

    float acc = 0.f;

    for (unsigned b = 0; b < nb; b++) {
        #pragma unroll
        for (unsigned g = 0; g < 8u; g++) {
            xs[warp * 256u + g * 32u + lane] =
                (gr < m) ? x[(size_t)gr * n_in + b * 256u + g * 32u + lane] : 0.f;
        }
        __syncthreads();

        if (col < n_out && gr < m) {
            const uint8_t* blk = wb + (size_t)col * row_bytes + b * 144u;
            const float* xr = xs + warp * 256u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* packed = blk + 4;
            const uint8_t* qs = blk + 16;
            float partial = 0.f;
            #pragma unroll
            for (unsigned pair = 0; pair < 4u; pair++) {
                float mn1, mn2;
                float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
                float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
                #pragma unroll
                for (unsigned l = 0; l < 32u; l++) {
                    uint8_t qq = qs[pair * 32u + l];
                    float xl = xr[pair * 64u + l];
                    float xh = xr[pair * 64u + 32u + l];
                    partial += (d * sc1 * (float)(qq & 0xf) - dmin * mn1) * xl;
                    partial += (d * sc2 * (float)(qq >> 4) - dmin * mn2) * xh;
                }
            }
            acc += partial;
        }
        __syncthreads();
    }

    if (col < n_out && gr < m) {
        y[(size_t)gr * n_out + col] = acc;
    }
}

// Fused gate+up: Metal-style 16 outs/block, X loaded once for both W matrices.
extern "C" __global__ void matvec_q4_k_2(
    const uint8_t* __restrict__ w0,
    const uint8_t* __restrict__ w1,
    const float* __restrict__ x,
    float* __restrict__ y0,
    float* __restrict__ y1,
    unsigned n_in, unsigned n_out,
    unsigned long long w_off0, unsigned long long w_off1, unsigned m)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned row = blockIdx.y;
    if (row >= m) return;
    (void)y1; // SwiGLU fused into y0; keep arg for launch ABI.
    unsigned pair0 = blockIdx.x * 16u + warp * 4u;
    if (pair0 >= n_out) return;
    unsigned nrows = n_out - pair0;
    if (nrows > 4u) nrows = 4u;

    const float* xr = x + (size_t)row * n_in;
    unsigned nb = n_in / 256;
    unsigned rb = nb * 144u;
    const uint8_t* row0 = (w0 + w_off0) + (size_t)pair0 * rb;
    const uint8_t* row1 = (w1 + w_off1) + (size_t)pair0 * rb;

    unsigned ix = lane / 8u;
    unsigned it = lane % 8u;
    unsigned iq = it / 4u;
    unsigned ir = it % 4u;

    float sum0[4] = {0.f, 0.f, 0.f, 0.f};
    float sum1[4] = {0.f, 0.f, 0.f, 0.f};
    const float* y4 = xr + ix * 256u + 64u * iq + 8u * ir;

    for (unsigned ib = ix; ib < nb; ib += 4u) {
        float yl[16], yh[16];
        float sumy0 = 0.f, sumy1 = 0.f, sumy2 = 0.f, sumy3 = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            yl[i] = y4[i];
            sumy0 += yl[i];
            yl[i + 8] = y4[i + 32];
            sumy1 += yl[i + 8];
            yh[i] = y4[i + 128];
            sumy2 += yh[i];
            yh[i + 8] = y4[i + 160];
            sumy3 += yh[i + 8];
        }

        for (unsigned r = 0; r < nrows; r++) {
            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                const uint8_t* blk = (pass == 0 ? row0 : row1) + r * rb + ib * 144u;
                float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
                const uint16_t* sc = (const uint16_t*)(blk + 4) + iq;
                const uint16_t* q1 = (const uint16_t*)(blk + 16) + 16u * iq + 4u * ir;
                const uint16_t* q2 = q1 + 32;

                uint16_t sc16[4];
                sc16[0] = (uint16_t)(sc[0] & 0x3f3f);
                sc16[1] = (uint16_t)(sc[2] & 0x3f3f);
                sc16[2] = (uint16_t)(((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2));
                sc16[3] = (uint16_t)(((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2));
                const uint8_t* sc8 = (const uint8_t*)sc16;

                float acc1[4] = {0.f, 0.f, 0.f, 0.f};
                float acc2[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    uint16_t qq1 = q1[i];
                    uint16_t qq2 = q2[i];
                    acc1[0] += yl[2 * i + 0] * (float)(qq1 & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (float)(qq1 & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (float)(qq1 & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (float)(qq1 & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (float)(qq2 & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (float)(qq2 & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (float)(qq2 & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (float)(qq2 & 0xF000);
                }

                float contrib =
                    d * ((acc1[0] + (1.f / 256.f) * acc1[1]) * (float)sc8[0] +
                         (acc1[2] + (1.f / 256.f) * acc1[3]) * (float)sc8[1] * (1.f / 16.f) +
                         (acc2[0] + (1.f / 256.f) * acc2[1]) * (float)sc8[4] +
                         (acc2[2] + (1.f / 256.f) * acc2[3]) * (float)sc8[5] * (1.f / 16.f))
                    - dmin * (sumy0 * (float)sc8[2] + sumy1 * (float)sc8[3] +
                              sumy2 * (float)sc8[6] + sumy3 * (float)sc8[7]);
                if (pass == 0) sum0[r] += contrib;
                else sum1[r] += contrib;
            }
        }
        y4 += 4 * 256;
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        sum0[r] = warp_sum_f(sum0[r]);
        sum1[r] = warp_sum_f(sum1[r]);
    }
    if (lane == 0) {
        float* a = y0 + (size_t)row * n_out;
        for (unsigned r = 0; r < nrows; r++) {
            a[pair0 + r] = silu_f(sum0[r]) * sum1[r];
        }
    }
}

// Fused Q/K/V: one X stream, three outputs (n_q may exceed n_kv).
extern "C" __global__ void matvec_q4_k_qkv(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv,
    const float* __restrict__ x,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    unsigned n_in, unsigned n_q, unsigned n_kv,
    unsigned long long oq, unsigned long long ok, unsigned long long ov, unsigned m)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned nwarps = blockDim.y;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    if (row >= m) return;
    bool do_q = out < n_q;
    bool do_kv = out < n_kv;
    if (!do_q && !do_kv) return;
    const float* xr = x + (size_t)row * n_in;
    unsigned nb = n_in / 256;
    unsigned rb = nb * 144u;
    const uint8_t* wrow_q = do_q ? (wq + oq) + (size_t)out * rb : nullptr;
    const uint8_t* wrow_k = do_kv ? (wk + ok) + (size_t)out * rb : nullptr;
    const uint8_t* wrow_v = do_kv ? (wv + ov) + (size_t)out * rb : nullptr;
    float acc_q = 0.f, acc_k = 0.f, acc_v = 0.f;
    for (unsigned b = warp; b < nb; b += nwarps) {
        const float* xb = xr + b * 256;
        float pq = 0.f, pk = 0.f, pv = 0.f;
        #pragma unroll
        for (unsigned pair = 0; pair < 4; pair++) {
            float xl = xb[pair * 64u + lane];
            float xh = xb[pair * 64u + 32u + lane];
            if (do_q) {
                const uint8_t* blk = wrow_q + b * 144;
                float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
                float mn1, mn2;
                float sc1 = scale_min_k4((int)pair * 2, blk + 4, &mn1);
                float sc2 = scale_min_k4((int)pair * 2 + 1, blk + 4, &mn2);
                uint8_t qq = blk[16 + pair * 32 + lane];
                pq += (d * sc1 * (float)(qq & 0xf) - dmin * mn1) * xl;
                pq += (d * sc2 * (float)(qq >> 4) - dmin * mn2) * xh;
            }
            if (do_kv) {
                {
                    const uint8_t* blk = wrow_k + b * 144;
                    float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                    float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
                    float mn1, mn2;
                    float sc1 = scale_min_k4((int)pair * 2, blk + 4, &mn1);
                    float sc2 = scale_min_k4((int)pair * 2 + 1, blk + 4, &mn2);
                    uint8_t qq = blk[16 + pair * 32 + lane];
                    pk += (d * sc1 * (float)(qq & 0xf) - dmin * mn1) * xl;
                    pk += (d * sc2 * (float)(qq >> 4) - dmin * mn2) * xh;
                }
                {
                    const uint8_t* blk = wrow_v + b * 144;
                    float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                    float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
                    float mn1, mn2;
                    float sc1 = scale_min_k4((int)pair * 2, blk + 4, &mn1);
                    float sc2 = scale_min_k4((int)pair * 2 + 1, blk + 4, &mn2);
                    uint8_t qq = blk[16 + pair * 32 + lane];
                    pv += (d * sc1 * (float)(qq & 0xf) - dmin * mn1) * xl;
                    pv += (d * sc2 * (float)(qq >> 4) - dmin * mn2) * xh;
                }
            }
        }
        if (do_q) acc_q += warp_sum_f(pq);
        if (do_kv) { acc_k += warp_sum_f(pk); acc_v += warp_sum_f(pv); }
    }
    __shared__ float sq[16], sk[16], sv[16];
    if (lane == 0) { sq[warp] = acc_q; sk[warp] = acc_k; sv[warp] = acc_v; }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float aq = 0.f, ak = 0.f, av = 0.f;
        for (unsigned i = 0; i < nwarps; i++) {
            aq += sq[i]; ak += sk[i]; av += sv[i];
        }
        if (do_q) q[(size_t)row * n_q + out] = aq;
        if (do_kv) {
            k[(size_t)row * n_kv + out] = ak;
            v[(size_t)row * n_kv + out] = av;
        }
    }
}

extern "C" __global__ void mm_q6_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    const unsigned BM = 8u;
    unsigned lane = threadIdx.x;
    unsigned col = blockIdx.x * 32u + lane;
    unsigned row0 = blockIdx.y * BM;
    bool col_ok = col < n_out;
    __shared__ float xs[8 * 256];
    float accs[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) accs[i] = 0.f;
    const uint8_t* wb = w + w_off;
    unsigned nb = n_in / 256;
    for (unsigned b = 0; b < nb; b++) {
        for (unsigned idx = lane; idx < BM * 256u; idx += 32) {
            unsigned r = idx >> 8;
            unsigned c = idx & 255u;
            unsigned gr = row0 + r;
            xs[idx] = (gr < m) ? x[(size_t)gr * n_in + b * 256u + c] : 0.f;
        }
        __syncthreads();
        if (col_ok) {
            const uint8_t* blk = wb + (size_t)col * nb * 210u + b * 210u;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = (const int8_t*)(blk + 192);
            float d = half_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            for (unsigned hi = 0; hi < 2; hi++) {
                const uint8_t* ql_h = ql + hi * 64;
                const uint8_t* qh_h = qh + hi * 32;
                const int8_t* sc = scales + hi * 8;
                for (unsigned r = 0; r < BM; r++) {
                    const float* xb = xs + r * 256 + hi * 128;
                    float local = 0.f;
                    for (unsigned l = 0; l < 32; l++) {
                        unsigned is = l / 16;
                        int q1 = (int)((ql_h[l] & 0xf) | ((qh_h[l] & 3) << 4)) - 32;
                        int q2 = (int)((ql_h[l + 32] & 0xf) | (((qh_h[l] >> 2) & 3) << 4)) - 32;
                        int q3 = (int)((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32;
                        int q4 = (int)((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32;
                        local += d * (float)sc[is] * (float)q1 * xb[l];
                        local += d * (float)sc[is + 2] * (float)q2 * xb[32 + l];
                        local += d * (float)sc[is + 4] * (float)q3 * xb[64 + l];
                        local += d * (float)sc[is + 6] * (float)q4 * xb[96 + l];
                    }
                    accs[r] += local;
                }
            }
        }
        __syncthreads();
    }
    if (col_ok) {
        for (unsigned r = 0; r < BM; r++) {
            unsigned gr = row0 + r;
            if (gr < m) y[(size_t)gr * n_out + col] = accs[r];
        }
    }
}

extern "C" __global__ void cast_f16_to_f32(
    float* __restrict__ dst,
    const uint16_t* __restrict__ src,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = half_bits_to_f32(src[i]);
}

// ---- Gated Delta Net (qwen35moe linear-attention layers) --------------------
// Port of Metal gdn_conv / gdn_step / gdn_out_norm (+ batch variants).
// gdn_step* requires d == 128 (32 lanes x 4 columns per state row).

__device__ __forceinline__ float softplus_f(float x) {
    return x > 20.0f ? x : logf(1.0f + expf(x));
}

// warp_sum_f defined near top with other helpers.

// One depthwise causal-conv step: out[c] = silu(sum_j w[c][j] * window[j][c]
// + w[c][last] * cur[c]); window shifts left and keeps RAW cur.
extern "C" __global__ void gdn_conv(
    float* __restrict__ qkv,
    float* __restrict__ ssm,
    const float* __restrict__ w,
    unsigned channels,
    unsigned d_conv,
    unsigned long long conv_off)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    float* win = ssm + conv_off;
    float acc = 0.f;
    for (unsigned j = 0; j + 1 < d_conv; j++) {
        acc += win[j * channels + i] * w[i * d_conv + j];
    }
    float cur = qkv[i];
    acc += cur * w[i * d_conv + d_conv - 1];
    for (unsigned j = 0; j + 2 < d_conv; j++) {
        win[j * channels + i] = win[(j + 1) * channels + i];
    }
    win[(d_conv - 2) * channels + i] = cur;
    qkv[i] = silu_f(acc);
}

// Gated deltanet recurrence for one token. Grid (d/4, heads_v), block (32, 4).
// State layout: [head][out_row][key_col]. ab = [alpha | beta_raw].
extern "C" __global__ void gdn_step(
    float* __restrict__ ssm,
    const float* __restrict__ qkv,
    const float* __restrict__ ab,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ out,
    unsigned heads_k,
    unsigned heads_v,
    unsigned d,
    unsigned key_dim,
    float eps,
    unsigned long long state_off)
{
    unsigned tx = threadIdx.x;
    unsigned ty = threadIdx.y;
    unsigned h = blockIdx.y;
    unsigned i20 = blockIdx.x * 4 + ty;
    if (h >= heads_v || i20 >= d) return;
    unsigned kh = h % heads_k;

    const float* q_ptr = qkv + kh * d;
    const float* k_ptr = qkv + key_dim + kh * d;
    const float* v_ptr = qkv + 2 * key_dim + h * d;

    float q0 = q_ptr[tx * 4 + 0], q1 = q_ptr[tx * 4 + 1];
    float q2 = q_ptr[tx * 4 + 2], q3 = q_ptr[tx * 4 + 3];
    float k0 = k_ptr[tx * 4 + 0], k1 = k_ptr[tx * 4 + 1];
    float k2 = k_ptr[tx * 4 + 2], k3 = k_ptr[tx * 4 + 3];
    float qi = rsqrtf(warp_sum_f(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) + eps);
    float ki = rsqrtf(warp_sum_f(k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3) + eps);
    q0 *= qi; q1 *= qi; q2 *= qi; q3 *= qi;
    k0 *= ki; k1 *= ki; k2 *= ki; k3 *= ki;

    float sp = ab[h] + dt_bias[h];
    float g = softplus_f(sp) * a_log[h];
    float g_exp = expf(g);
    float beta = 1.0f / (1.0f + expf(-ab[heads_v + h]));

    float* s_ptr = ssm + state_off + (unsigned long long)h * d * d
        + (unsigned long long)i20 * d;
    float ls0 = s_ptr[tx * 4 + 0] * g_exp;
    float ls1 = s_ptr[tx * 4 + 1] * g_exp;
    float ls2 = s_ptr[tx * 4 + 2] * g_exp;
    float ls3 = s_ptr[tx * 4 + 3] * g_exp;
    float s_k = warp_sum_f(ls0 * k0 + ls1 * k1 + ls2 * k2 + ls3 * k3);
    float dv = (v_ptr[i20] - s_k) * beta;
    ls0 += k0 * dv; ls1 += k1 * dv; ls2 += k2 * dv; ls3 += k3 * dv;
    float y = warp_sum_f(ls0 * q0 + ls1 * q1 + ls2 * q2 + ls3 * q3)
        * rsqrtf((float)d);
    s_ptr[tx * 4 + 0] = ls0;
    s_ptr[tx * 4 + 1] = ls1;
    s_ptr[tx * 4 + 2] = ls2;
    s_ptr[tx * 4 + 3] = ls3;
    if (tx == 0) out[h * d + i20] = y;
}

// Per-head rmsnorm(y) * silu(z). One block per head, d threads (d==128).
extern "C" __global__ void gdn_out_norm(
    float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ w,
    unsigned heads_v,
    unsigned d,
    float eps)
{
    unsigned h = blockIdx.x;
    unsigned tid = threadIdx.x;
    if (h >= heads_v || tid >= d) return;
    __shared__ float buf[128];
    float yv = y[h * d + tid];
    buf[tid] = yv * yv;
    __syncthreads();
    for (unsigned s = d / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(buf[0] / (float)d + eps);
    float zv = z[h * d + tid];
    y[h * d + tid] = yv * scale * w[tid] * silu_f(zv);
}

// Depthwise causal conv over a chunk. Session window is read-only here;
// host/kernel updates it after every row's read completes.
extern "C" __global__ void gdn_conv_batch(
    const float* __restrict__ qkv,
    float* __restrict__ qkc,
    const float* __restrict__ ssm,
    const float* __restrict__ w,
    float* __restrict__ slots,
    unsigned channels,
    unsigned d_conv,
    unsigned m,
    unsigned long long conv_off,
    unsigned slot_total)
{
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row = blockIdx.y;
    if (c >= channels || row >= m) return;
    const float* win = ssm + conv_off;
    unsigned K = d_conv - 1;
    float acc = 0.f;
    for (unsigned j = 0; j < K; j++) {
        int src = (int)row - (int)(K - j);
        float v = src >= 0 ? qkv[(unsigned)src * channels + c]
                           : win[(unsigned)(src + (int)K) * channels + c];
        acc += v * w[c * d_conv + j];
    }
    float cur = qkv[row * channels + c];
    acc += cur * w[c * d_conv + K];
    qkc[row * channels + c] = silu_f(acc);
    if (slot_total != 0) {
        for (unsigned r = row; r < min(row + K, m); r++) {
            unsigned wj = K - 1 - (r - row);
            slots[(unsigned long long)r * slot_total + conv_off + wj * channels + c] = cur;
        }
        if (row == 0) {
            for (unsigned j = 0; j < K; j++) {
                float v = win[j * channels + c];
                for (unsigned r = 0; r < j; r++) {
                    slots[(unsigned long long)r * slot_total + conv_off
                        + (j - 1 - r) * channels + c] = v;
                }
            }
        }
    }
}

// Recurrence over a whole chunk; state kept in registers across tokens.
// Grid (d/4, heads_v), block (32, 4).
extern "C" __global__ void gdn_step_batch(
    float* __restrict__ ssm,
    const float* __restrict__ qkc,
    const float* __restrict__ alpha,
    const float* __restrict__ beta_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ out,
    float* __restrict__ slots,
    unsigned heads_k,
    unsigned heads_v,
    unsigned d,
    unsigned key_dim,
    unsigned m,
    float eps,
    unsigned long long state_off,
    unsigned slot_total)
{
    unsigned tx = threadIdx.x;
    unsigned ty = threadIdx.y;
    unsigned h = blockIdx.y;
    unsigned i20 = blockIdx.x * 4 + ty;
    if (h >= heads_v || i20 >= d) return;
    unsigned kh = h % heads_k;
    unsigned channels = 2 * key_dim + heads_v * d;

    float* s_ptr = ssm + state_off + (unsigned long long)h * d * d
        + (unsigned long long)i20 * d;
    float ls0 = s_ptr[tx * 4 + 0], ls1 = s_ptr[tx * 4 + 1];
    float ls2 = s_ptr[tx * 4 + 2], ls3 = s_ptr[tx * 4 + 3];

    for (unsigned t = 0; t < m; t++) {
        const float* q_ptr = qkc + (unsigned long long)t * channels + kh * d;
        const float* k_ptr = qkc + (unsigned long long)t * channels + key_dim + kh * d;
        const float* v_ptr = qkc + (unsigned long long)t * channels + 2 * key_dim + h * d;

        float q0 = q_ptr[tx * 4 + 0], q1 = q_ptr[tx * 4 + 1];
        float q2 = q_ptr[tx * 4 + 2], q3 = q_ptr[tx * 4 + 3];
        float k0 = k_ptr[tx * 4 + 0], k1 = k_ptr[tx * 4 + 1];
        float k2 = k_ptr[tx * 4 + 2], k3 = k_ptr[tx * 4 + 3];
        float qi = rsqrtf(warp_sum_f(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) + eps);
        float ki = rsqrtf(warp_sum_f(k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3) + eps);
        q0 *= qi; q1 *= qi; q2 *= qi; q3 *= qi;
        k0 *= ki; k1 *= ki; k2 *= ki; k3 *= ki;

        float sp = alpha[(unsigned long long)t * heads_v + h] + dt_bias[h];
        float g = softplus_f(sp) * a_log[h];
        float g_exp = expf(g);
        float beta = 1.0f / (1.0f + expf(-beta_raw[(unsigned long long)t * heads_v + h]));

        ls0 *= g_exp; ls1 *= g_exp; ls2 *= g_exp; ls3 *= g_exp;
        float s_k = warp_sum_f(ls0 * k0 + ls1 * k1 + ls2 * k2 + ls3 * k3);
        float dv = (v_ptr[i20] - s_k) * beta;
        ls0 += k0 * dv; ls1 += k1 * dv; ls2 += k2 * dv; ls3 += k3 * dv;
        float y = warp_sum_f(ls0 * q0 + ls1 * q1 + ls2 * q2 + ls3 * q3)
            * rsqrtf((float)d);
        if (tx == 0) {
            out[(unsigned long long)t * heads_v * d + h * d + i20] = y;
        }
        if (slot_total != 0) {
            float* slot = slots + (unsigned long long)t * slot_total + state_off
                + (unsigned long long)h * d * d + (unsigned long long)i20 * d;
            slot[tx * 4 + 0] = ls0;
            slot[tx * 4 + 1] = ls1;
            slot[tx * 4 + 2] = ls2;
            slot[tx * 4 + 3] = ls3;
        }
    }
    s_ptr[tx * 4 + 0] = ls0;
    s_ptr[tx * 4 + 1] = ls1;
    s_ptr[tx * 4 + 2] = ls2;
    s_ptr[tx * 4 + 3] = ls3;
}

// Batched gated out-norm: one block per (head, token).
extern "C" __global__ void gdn_out_norm_batch(
    float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ w,
    unsigned heads_v,
    unsigned d,
    float eps)
{
    unsigned h = blockIdx.x;
    unsigned t = blockIdx.y;
    unsigned tid = threadIdx.x;
    if (h >= heads_v || tid >= d) return;
    __shared__ float buf[128];
    unsigned row = t * heads_v * d + h * d;
    float yv = y[row + tid];
    buf[tid] = yv * yv;
    __syncthreads();
    for (unsigned s = d / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(buf[0] / (float)d + eps);
    float zv = z[row + tid];
    y[row + tid] = yv * scale * w[tid] * silu_f(zv);
}

// Copy n floats into an SSM (or other) region at an element offset.
extern "C" __global__ void ssm_store_f32(
    float* __restrict__ ssm,
    const float* __restrict__ src,
    unsigned long long elem_off,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ssm[elem_off + i] = src[i];
}

extern "C" __global__ void ssm_load_f32(
    float* __restrict__ dst,
    const float* __restrict__ ssm,
    unsigned long long elem_off,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = ssm[elem_off + i];
}
"#;

/// Separate NVRTC unit so growing embed support does not reshuffle main-kernel PTX.
pub const EMBED_KERNELS: &str = r#"
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;

__device__ __forceinline__ float half_bits_to_f32(uint16_t h) {
    float f;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ float scale_min_k4(int j, const uint8_t* packed, float* mn_out) {
    float sc, mn;
    if (j < 4) {
        sc = (float)(packed[j] & 63);
        mn = (float)(packed[j + 4] & 63);
    } else {
        sc = (float)((packed[j + 4] & 0xf) | ((packed[j - 4] >> 6) << 4));
        mn = (float)((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
    }
    *mn_out = mn;
    return sc;
}

extern "C" __global__ void embed_row_f32(
    const uint8_t* __restrict__ w,
    float* __restrict__ out,
    unsigned n_in,
    unsigned long long w_off,
    unsigned fmt)
{
    unsigned lane = threadIdx.x;
    const uint8_t* row = w + w_off;
    if (fmt <= 2u) {
        unsigned b = blockIdx.y;
        float* o256 = out + b * 256u;
        if (fmt == 0u) {
            const uint8_t* blk = row + b * 144u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* packed = blk + 4;
            const uint8_t* qs = blk + 16;
            #pragma unroll
            for (unsigned pair = 0; pair < 4u; pair++) {
                float mn1, mn2;
                float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
                float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
                uint8_t qq = qs[pair * 32u + lane];
                o256[pair * 64u + lane] = d * sc1 * (float)(qq & 0xf) - dmin * mn1;
                o256[pair * 64u + 32u + lane] = d * sc2 * (float)(qq >> 4) - dmin * mn2;
            }
            return;
        }
        if (fmt == 1u) {
            const uint8_t* blk = row + b * 176u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* packed = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = blk + 48;
            #pragma unroll
            for (unsigned pair = 0; pair < 4u; pair++) {
                float mn1, mn2;
                float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
                float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
                uint8_t bit1 = (uint8_t)(1u << (pair * 2));
                uint8_t bit2 = (uint8_t)(1u << (pair * 2 + 1));
                uint8_t qq = qs[pair * 32u + lane];
                int hi1 = (qh[lane] & bit1) ? 16 : 0;
                int hi2 = (qh[lane] & bit2) ? 16 : 0;
                o256[pair * 64u + lane] =
                    d * sc1 * (float)((qq & 0xf) + hi1) - dmin * mn1;
                o256[pair * 64u + 32u + lane] =
                    d * sc2 * (float)((qq >> 4) + hi2) - dmin * mn2;
            }
            return;
        }
        {
            const uint8_t* blk = row + b * 210u;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = (const int8_t*)(blk + 192);
            float d = half_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            #pragma unroll
            for (unsigned hi = 0; hi < 2u; hi++) {
                const uint8_t* ql_h = ql + hi * 64u;
                const uint8_t* qh_h = qh + hi * 32u;
                const int8_t* sc = scales + hi * 8;
                float* oh = o256 + hi * 128u;
                unsigned is = lane / 16u;
                int q1 = (int)((ql_h[lane] & 0xf) | ((qh_h[lane] & 3) << 4)) - 32;
                int q2 = (int)((ql_h[lane + 32] & 0xf) | (((qh_h[lane] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_h[lane] >> 4) | (((qh_h[lane] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_h[lane + 32] >> 4) | (((qh_h[lane] >> 6) & 3) << 4)) - 32;
                oh[lane] = d * (float)sc[is] * (float)q1;
                oh[32u + lane] = d * (float)sc[is + 2] * (float)q2;
                oh[64u + lane] = d * (float)sc[is + 4] * (float)q3;
                oh[96u + lane] = d * (float)sc[is + 6] * (float)q4;
            }
            return;
        }
    }
    if (lane != 0 || blockIdx.x != 0) return;
    if (fmt == 3u) {
        unsigned nb = n_in / 32u;
        for (unsigned b = 0; b < nb; b++) {
            const uint8_t* blk = row + b * 34u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            for (unsigned i = 0; i < 32u; i++) {
                out[b * 32u + i] = d * (float)((int8_t)blk[2 + i]);
            }
        }
        return;
    }
    if (fmt == 7u) {
        const uint16_t* src = (const uint16_t*)row;
        for (unsigned i = 0; i < n_in; i++) out[i] = half_bits_to_f32(src[i]);
        return;
    }
    if (fmt == 8u) {
        const float* src = (const float*)row;
        for (unsigned i = 0; i < n_in; i++) out[i] = src[i];
    }
}

// Like embed_row_f32 but token id lives on device (greedy chain after argmax).
extern "C" __global__ void embed_row_f32_dtoken(
    const uint8_t* __restrict__ w,
    float* __restrict__ out,
    unsigned n_in,
    unsigned long long w_base,
    unsigned row_bytes,
    unsigned fmt,
    const unsigned* __restrict__ d_token)
{
    unsigned lane = threadIdx.x;
    unsigned long long w_off = w_base + (unsigned long long)d_token[0] * (unsigned long long)row_bytes;
    const uint8_t* row = w + w_off;
    if (fmt <= 2u) {
        unsigned b = blockIdx.y;
        float* o256 = out + b * 256u;
        if (fmt == 0u) {
            const uint8_t* blk = row + b * 144u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* packed = blk + 4;
            const uint8_t* qs = blk + 16;
            #pragma unroll
            for (unsigned pair = 0; pair < 4u; pair++) {
                float mn1, mn2;
                float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
                float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
                uint8_t qq = qs[pair * 32u + lane];
                o256[pair * 64u + lane] = d * sc1 * (float)(qq & 0xf) - dmin * mn1;
                o256[pair * 64u + 32u + lane] = d * sc2 * (float)(qq >> 4) - dmin * mn2;
            }
            return;
        }
        if (fmt == 1u) {
            const uint8_t* blk = row + b * 176u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* packed = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = blk + 48;
            #pragma unroll
            for (unsigned pair = 0; pair < 4u; pair++) {
                float mn1, mn2;
                float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
                float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
                uint8_t bit1 = (uint8_t)(1u << (pair * 2));
                uint8_t bit2 = (uint8_t)(1u << (pair * 2 + 1));
                uint8_t qq = qs[pair * 32u + lane];
                int hi1 = (qh[lane] & bit1) ? 16 : 0;
                int hi2 = (qh[lane] & bit2) ? 16 : 0;
                o256[pair * 64u + lane] =
                    d * sc1 * (float)((qq & 0xf) + hi1) - dmin * mn1;
                o256[pair * 64u + 32u + lane] =
                    d * sc2 * (float)((qq >> 4) + hi2) - dmin * mn2;
            }
            return;
        }
        {
            const uint8_t* blk = row + b * 210u;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = (const int8_t*)(blk + 192);
            float d = half_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
            #pragma unroll
            for (unsigned hi = 0; hi < 2u; hi++) {
                const uint8_t* ql_h = ql + hi * 64u;
                const uint8_t* qh_h = qh + hi * 32u;
                const int8_t* sc = scales + hi * 8;
                float* oh = o256 + hi * 128u;
                unsigned is = lane / 16u;
                int q1 = (int)((ql_h[lane] & 0xf) | ((qh_h[lane] & 3) << 4)) - 32;
                int q2 = (int)((ql_h[lane + 32] & 0xf) | (((qh_h[lane] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_h[lane] >> 4) | (((qh_h[lane] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_h[lane + 32] >> 4) | (((qh_h[lane] >> 6) & 3) << 4)) - 32;
                oh[lane] = d * (float)sc[is] * (float)q1;
                oh[32u + lane] = d * (float)sc[is + 2] * (float)q2;
                oh[64u + lane] = d * (float)sc[is + 4] * (float)q3;
                oh[96u + lane] = d * (float)sc[is + 6] * (float)q4;
            }
            return;
        }
    }
    if (lane != 0 || blockIdx.x != 0) return;
    if (fmt == 3u) {
        unsigned nb = n_in / 32u;
        for (unsigned b = 0; b < nb; b++) {
            const uint8_t* blk = row + b * 34u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            for (unsigned i = 0; i < 32u; i++) {
                out[b * 32u + i] = d * (float)((int8_t)blk[2 + i]);
            }
        }
        return;
    }
    if (fmt == 7u) {
        const uint16_t* src = (const uint16_t*)row;
        for (unsigned i = 0; i < n_in; i++) out[i] = half_bits_to_f32(src[i]);
        return;
    }
    if (fmt == 8u) {
        const float* src = (const float*)row;
        for (unsigned i = 0; i < n_in; i++) out[i] = src[i];
    }
}

extern "C" __global__ void inc_u32(unsigned* __restrict__ p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1u;
}

extern "C" __global__ void copy_u32(
    unsigned* __restrict__ dst,
    const unsigned* __restrict__ src,
    unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void store_u32_at(
    unsigned* __restrict__ dst,
    const unsigned* __restrict__ src,
    unsigned idx)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[idx] = src[0];
}

"#;
