#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

struct block_q8_1 {
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// in: [hd, n_q, m] contiguous; out: [m, n_q, hd]
extern "C" __global__ void allpaka_fa_permute(
    const float * __restrict__ in,
    float * __restrict__ out,
    int hd, int n_q, int m)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)hd * (size_t)n_q * (size_t)m;
    if (idx >= n) return;
    int d = (int)(idx % (size_t)hd);
    size_t rem = idx / (size_t)hd;
    int h = (int)(rem % (size_t)n_q);
    int t = (int)(rem / (size_t)n_q);
    size_t in_i = (size_t)d + (size_t)h * (size_t)hd + (size_t)t * (size_t)hd * (size_t)n_q;
    size_t out_i = (size_t)t * (size_t)n_q * (size_t)hd + (size_t)h * (size_t)hd + (size_t)d;
    out[out_i] = in[in_i];
}

// Decode m=1: permute FA [hd,n_q] -> out [n_q,hd] and pack block_q8_1 for o_proj.
// One warp per 32-wide output group. Pack from the same permuted floats o_proj sees.
extern "C" __global__ void allpaka_fa_permute_q8(
    const float * __restrict__ in,
    float * __restrict__ out,
    block_q8_1 * __restrict__ yq,
    int hd, int n_q)
{
    int blk = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    int n = hd * n_q;
    int base = blk * 32;
    if (base >= n) return;

    // Match allpaka_fa_permute for m=1: out[h*hd+d] = in[d+h*hd] (identity index).
    int out_i = base + lane;
    int d = out_i % hd;
    int h = out_i / hd;
    size_t in_i = (size_t)d + (size_t)h * (size_t)hd;
    float v = in[in_i];
    out[out_i] = v;

    // Identical to allpaka quantize_q8_1 (cvt.rn.f16.f32).
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

    block_q8_1 * b = yq + blk;
    b->qs[lane] = (int8_t)qi;
    int qsum = qi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        qsum += __shfl_xor_sync(0xffffffffu, qsum, off);
    }
    if (lane == 0) {
        uint16_t dh, sh;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(dh) : "f"(scale));
        float s = scale * (float)qsum;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(sh) : "f"(s));
        b->d = dh;
        b->s = sh;
    }
}

// Pack out[0..hd*n_q) to block_q8_1 — same math as allpaka quantize_q8_1.
extern "C" __global__ void allpaka_q8_pack(
    const float * __restrict__ x,
    block_q8_1 * __restrict__ yq,
    int n)
{
    int blk = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    int base = blk * 32;
    if (base >= n) return;
    float v = x[base + lane];
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
    block_q8_1 * b = yq + blk;
    b->qs[lane] = (int8_t)qi;
    int qsum = qi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        qsum += __shfl_xor_sync(0xffffffffu, qsum, off);
    }
    if (lane == 0) {
        uint16_t dh, sh;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(dh) : "f"(scale));
        float s = scale * (float)qsum;
        asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(sh) : "f"(s));
        b->d = dh;
        b->s = sh;
    }
}

extern "C" void allpaka_fa_permute_launch(
    const float * in, float * out, int hd, int n_q, int m, cudaStream_t stream)
{
    const int n = hd * n_q * m;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    allpaka_fa_permute<<<blocks, threads, 0, stream>>>(in, out, hd, n_q, m);
}

extern "C" void allpaka_fa_permute_q8_launch(
    const float * in, float * out, void * yq, int hd, int n_q, cudaStream_t stream)
{
    const int n = hd * n_q;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    allpaka_fa_permute<<<blocks, threads, 0, stream>>>(in, out, hd, n_q, 1);
    const int qblocks = (n + 31) / 32;
    allpaka_q8_pack<<<qblocks, 32, 0, stream>>>(
        out, reinterpret_cast<block_q8_1 *>(yq), n);
}

extern "C" void allpaka_q8_pack_launch(
    const float * x, void * yq, int n, cudaStream_t stream)
{
    const int qblocks = (n + 31) / 32;
    allpaka_q8_pack<<<qblocks, 32, 0, stream>>>(
        x, reinterpret_cast<block_q8_1 *>(yq), n);
}

// Decode causal mask from device position (graph-capturable; lim = *d_pos).
extern "C" __global__ void allpaka_fa_mask_from_pos(
    __half * __restrict__ mask, int n_kv, const unsigned int * __restrict__ d_pos)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_kv) return;
    unsigned int lim = *d_pos;
    mask[i] = (i <= (int)lim) ? __float2half(0.0f) : __float2half(-INFINITY);
}

extern "C" void allpaka_fa_mask_from_pos_launch(
    void * mask, int n_kv, const unsigned int * d_pos, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (n_kv + threads - 1) / threads;
    allpaka_fa_mask_from_pos<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half *>(mask), n_kv, d_pos);
}
