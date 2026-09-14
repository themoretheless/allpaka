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
    if (row >= rows) return;
    const float* sr = src + (size_t)row * n;
    float* dr = dst + (size_t)row * n;
    __shared__ float buf[256];
    float local = 0.f;
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

// RoPE NeoX: pairs (x[i], x[i+uint16_t]) with table[i] = {sin, cos}.
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

extern "C" __global__ void matvec_q4_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    // llama.cpp mmvq layout: one output row per block, several warps split K
    // so Q4_K block loads are coalesced (vs one warp per row in a fat block).
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned nwarps = blockDim.y;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    if (out >= n_out || row >= m) return;
    const float* xr = x + (size_t)row * n_in;
    const uint8_t* wrow = (w + w_off) + (size_t)out * (n_in / 256) * 144;
    unsigned nb = n_in / 256;
    float acc = 0.f;
    for (unsigned b = warp; b < nb; b += nwarps) {
        const uint8_t* blk = wrow + b * 144;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t* packed = blk + 4;
        const uint8_t* qs = blk + 16;
        const float* xb = xr + b * 256;
        float partial = 0.f;
        #pragma unroll
        for (unsigned pair = 0; pair < 4; pair++) {
            float mn1, mn2;
            float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
            float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
            uint8_t qq = qs[pair * 32 + lane];
            float xl = xb[pair * 64u + lane];
            float xh = xb[pair * 64u + 32u + lane];
            partial += (d * sc1 * (float)(qq & 0xf) - dmin * mn1) * xl;
            partial += (d * sc2 * (float)(qq >> 4) - dmin * mn2) * xh;
        }
        acc += warp_sum_f(partial);
    }
    __shared__ float wacc[16];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = 0.f;
        for (unsigned i = 0; i < nwarps; i++) s += wacc[i];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
    }
}

// Pack activations to Q8_1 (32-wide amax scale). One warp per 32-group.
extern "C" __global__ void quantize_q8_1(
    const float* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ d,
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
    if (lane == 0) d[(size_t)row * (n / 32u) + blk] = scale;
    int qi = __float2int_rn(v / scale);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    q[(size_t)row * n + base + lane] = (int8_t)qi;
}

// Q4_K × Q8_1 mmvq: 8 warps split superblocks; int8 X from quantize_q8_1.
extern "C" __global__ void matvec_q4_k_q8(
    const uint8_t* __restrict__ w,
    const int8_t* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned nwarps = blockDim.y;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    if (out >= n_out || row >= m) return;
    const int8_t* xr = xq + (size_t)row * n_in;
    const float* xdr = xd + (size_t)row * (n_in / 32u);
    const uint8_t* wrow = (w + w_off) + (size_t)out * (n_in / 256) * 144;
    unsigned nb = n_in / 256;
    float acc = 0.f;
    for (unsigned b = warp; b < nb; b += nwarps) {
        const uint8_t* blk = wrow + b * 144;
        float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
        const uint8_t* packed = blk + 4;
        const uint8_t* qs = blk + 16;
        const int8_t* xb = xr + b * 256;
        const float* db = xdr + b * 8u;
        float partial = 0.f;
        #pragma unroll
        for (unsigned pair = 0; pair < 4; pair++) {
            float mn1, mn2;
            float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
            float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
            uint8_t qq = qs[pair * 32 + lane];
            int xl = (int)xb[pair * 64u + lane];
            int xh = (int)xb[pair * 64u + 32u + lane];
            float d0 = db[pair * 2u];
            float d1 = db[pair * 2u + 1u];
            int qlo = (int)(qq & 0xf);
            int qhi = (int)(qq >> 4);
            int dot_lo = qlo * xl;
            int dot_hi = qhi * xh;
            partial += (d * sc1 * (float)dot_lo - dmin * mn1 * (float)xl) * d0;
            partial += (d * sc2 * (float)dot_hi - dmin * mn2 * (float)xh) * d1;
        }
        acc += warp_sum_f(partial);
    }
    __shared__ float wacc[16];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = 0.f;
        for (unsigned i = 0; i < nwarps; i++) s += wacc[i];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
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

extern "C" __global__ void matvec_q6_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned nwarps = blockDim.y;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    if (out >= n_out || row >= m) return;
    const float* xr = x + (size_t)row * n_in;
    const uint8_t* wrow = (w + w_off) + (size_t)out * (n_in / 256) * 210;
    unsigned nb = n_in / 256;
    float acc = 0.f;
    for (unsigned b = warp; b < nb; b += nwarps) {
        const uint8_t* blk = wrow + b * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* scales = (const int8_t*)(blk + 192);
        float d = half_bits_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
        const float* xb = xr + b * 256;
        float partial = 0.f;
        for (unsigned hi = 0; hi < 2; hi++) {
            const uint8_t* ql_h = ql + hi * 64;
            const uint8_t* qh_h = qh + hi * 32;
            const int8_t* sc = scales + hi * 8;
            unsigned l = lane;
            unsigned is = l / 16;
            int q1 = (int)((ql_h[l] & 0xf) | ((qh_h[l] & 3) << 4)) - 32;
            int q2 = (int)((ql_h[l + 32] & 0xf) | (((qh_h[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32;
            float x0 = xb[hi * 128 + l];
            float x1 = xb[hi * 128 + 32 + l];
            float x2 = xb[hi * 128 + 64 + l];
            float x3 = xb[hi * 128 + 96 + l];
            partial += d * (float)sc[is] * (float)q1 * x0;
            partial += d * (float)sc[is + 2] * (float)q2 * x1;
            partial += d * (float)sc[is + 4] * (float)q3 * x2;
            partial += d * (float)sc[is + 6] * (float)q4 * x3;
        }
        acc += warp_sum_f(partial);
    }
    __shared__ float wacc[16];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = 0.f;
        for (unsigned i = 0; i < nwarps; i++) s += wacc[i];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
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
    const unsigned ROWS = 8u;
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

// Tiled Q4_K × int8(X) GEMM with dp4a — never materializes full W in f16.
// blockDim=(32,1): one warp = 32 output columns; grid=(ceil(n_out/32), ceil(m/BM)).
extern "C" __global__ void mm_q4_k(
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

    __shared__ int8_t xq[8 * 256];
    __shared__ float xd[8 * 8];

    float accs[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) accs[i] = 0.f;

    const uint8_t* wb = w + w_off;
    unsigned nb = n_in / 256;

    for (unsigned b = 0; b < nb; b++) {
        for (unsigned r = 0; r < BM; r++) {
            unsigned gr = row0 + r;
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                unsigned i = (unsigned)t * 32u + lane;
                float v = (gr < m) ? x[(size_t)gr * n_in + b * 256u + i] : 0.f;
                float amax = fabsf(v);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
                }
                float d = amax / 127.f;
                if (d < 1e-8f) d = 1.f;
                if (lane == 0) xd[r * 8 + (unsigned)t] = d;
                int q = __float2int_rn(v / d);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                xq[r * 256 + i] = (int8_t)q;
            }
        }
        __syncthreads();

        if (col_ok) {
            const uint8_t* blk = wb + (size_t)col * nb * 144u + b * 144u;
            float d = half_bits_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float dmin = half_bits_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            const uint8_t* packed = blk + 4;
            const uint8_t* qs = blk + 16;
            #pragma unroll
            for (unsigned r = 0; r < BM; r++) {
                float local = 0.f;
                for (unsigned pair = 0; pair < 4u; pair++) {
                    float mn1, mn2;
                    float sc1 = scale_min_k4((int)pair * 2, packed, &mn1);
                    float sc2 = scale_min_k4((int)pair * 2 + 1, packed, &mn2);
                    const uint8_t* q = qs + pair * 32u;
                    const int8_t* xr = xq + r * 256 + pair * 64u;
                    float d0 = xd[r * 8 + pair * 2];
                    float d1 = xd[r * 8 + pair * 2 + 1];
                    int dot_lo = 0, dot_hi = 0, sum_lo = 0, sum_hi = 0;
                    #pragma unroll
                    for (unsigned l = 0; l < 32u; l += 4u) {
                        int x4lo = ((int)(int8_t)xr[l] & 0xff)
                            | (((int)(int8_t)xr[l + 1] & 0xff) << 8)
                            | (((int)(int8_t)xr[l + 2] & 0xff) << 16)
                            | (((int)(int8_t)xr[l + 3] & 0xff) << 24);
                        int x4hi = ((int)(int8_t)xr[l + 32] & 0xff)
                            | (((int)(int8_t)xr[l + 33] & 0xff) << 8)
                            | (((int)(int8_t)xr[l + 34] & 0xff) << 16)
                            | (((int)(int8_t)xr[l + 35] & 0xff) << 24);
                        int qpack = (int)q[l] | ((int)q[l + 1] << 8)
                            | ((int)q[l + 2] << 16) | ((int)q[l + 3] << 24);
                        int qlo = qpack & 0x0f0f0f0f;
                        int qhi = (qpack >> 4) & 0x0f0f0f0f;
                        int one = 0x01010101;
                        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                                     : "=r"(dot_lo)
                                     : "r"(qlo), "r"(x4lo), "r"(dot_lo));
                        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                                     : "=r"(dot_hi)
                                     : "r"(qhi), "r"(x4hi), "r"(dot_hi));
                        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                                     : "=r"(sum_lo)
                                     : "r"(one), "r"(x4lo), "r"(sum_lo));
                        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                                     : "=r"(sum_hi)
                                     : "r"(one), "r"(x4hi), "r"(sum_hi));
                    }
                    local += d * sc1 * d0 * (float)dot_lo - dmin * mn1 * d0 * (float)sum_lo
                           + d * sc2 * d1 * (float)dot_hi - dmin * mn2 * d1 * (float)sum_hi;
                }
                accs[r] += local;
            }
        }
        __syncthreads();
    }

    if (col_ok) {
        #pragma unroll
        for (unsigned r = 0; r < BM; r++) {
            unsigned gr = row0 + r;
            if (gr < m) y[(size_t)gr * n_out + col] = accs[r];
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
