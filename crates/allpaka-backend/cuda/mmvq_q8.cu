// Q4_K / Q6_K MMVQ - nvcc -ptx -arch=sm_120 -O3
// Y = packed block_q8_1 (44 B): float d, int8 qs[32], four Q4_K partial sums.

extern "C" {

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed char int8_t;
typedef signed short int16_t;

struct block_q8_1 {
    float d;
    int8_t qs[32];
    int16_t ps[4];
};

__device__ __forceinline__ float half_bits_to_f32(uint16_t h) {
    float f;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ uint16_t f32_to_f16_bits(float f) {
    uint16_t h;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

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
        b->d = d;
    }
}

__device__ __forceinline__ void store_q8_scale(block_q8_1* b, unsigned lane, float d) {
    if (lane == 0) {
        b->d = d;
    }
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_sum_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int out;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x + 2 * i32;
    return (int)__ldcs(x16) | ((int)__ldcs(x16 + 1) << 16);
}

__device__ __forceinline__ int vsubss4(int a, int b) {
    (void)b;
    return (a + 0x60606060) ^ 0x80808080;
}

// Hopper+/Blackwell PDL ops. Even without host PROGRAMMATIC_STREAM_SERIALIZATION
// these help on sm_120 (nopdl-dev A/B: ~57 vs ~63 tok/s). Keep enabled.
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

__device__ __forceinline__ float vec_dot_q4_k_q8(
    const uint8_t* __restrict__ blk,
    const block_q8_1* __restrict__ bq8,
    int iqs)
{
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* q4 = (const int*)(blk + 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    // STREAMING hint on weight traffic (evict-first); Q8 X stays __ldg.
    int v0 = __ldcs(q4);
    int v1 = __ldcs(q4 + 4);

    const uint16_t* scales = (const uint16_t*)(blk + 4);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = __ldcs(scales + jm + 0);
    const uint32_t s2 = __ldcs(scales + jm + 2);
    const uint32_t s4 = __ldcs(scales + jm + 4);
    const uint32_t hi = (uint32_t)-(int32_t)(j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t)(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t)(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* mn = sc + 2;

    const uint32_t dm = __ldcs((const uint32_t*)blk);
    float d = half_bits_to_f32((uint16_t)dm);
    float dmin = half_bits_to_f32((uint16_t)(dm >> 16));

    float sumf_d = 0.f;
    float sumf_m = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const block_q8_1* bq8i = bq8 + bq8_offset + i;
        float d8 = __ldg(&bq8i->d);
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
    float d = half_bits_to_f32(__ldcs((const uint16_t*)(blk + 208)));

    float sumf = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const block_q8_1* bq8i = bq8 + bq8_offset + 2 * i;
        float d8 = __ldg(&bq8i->d);
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

// Same as vec_dot_q6_k_q8 but plain loads (safe for shared-memory Q8 tiles).
__device__ __forceinline__ float vec_dot_q6_k_q8_smem(
    const uint8_t* __restrict__ blk,
    const block_q8_1* bq8,
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
        float d8 = bq8i->d;
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

#define MMVQ_NWARPS 4u
#define MMVQ_NWARPS2 2u

// 8 Q8 blocks / CTA (vs 1 in NVRTC port). Loaded from nvcc PTX and overrides NVRTC.
__global__ void quantize_q8_1(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    unsigned n, unsigned rows)
{
    unsigned row = blockIdx.y;
    unsigned warp = threadIdx.y;
    unsigned lane = threadIdx.x;
    if (row >= rows) return;
    unsigned blk = blockIdx.x * blockDim.y + warp;
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

__global__ void quantize_q8_1_q6(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    unsigned n, unsigned rows)
{
    unsigned row = blockIdx.y;
    unsigned warp = threadIdx.y;
    unsigned lane = threadIdx.x;
    if (row >= rows) return;
    unsigned blk = blockIdx.x * blockDim.y + warp;
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

__global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8(
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

__global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8_k8192(
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
    (void)n_in;

    const block_q8_1* xr = x + (size_t)row * 256u;
    const uint8_t* wrow = (w + w_off) + (size_t)out * 4608u;
    unsigned kbx = tid / 16u;
    int iqs = (int)(2u * (tid % 16u));

    float acc = 0.f;
    #pragma unroll
    for (unsigned i = 0; i < 4u; i++, kbx += 8u) {
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

// Bandwidth-oriented: 2 warps (Blackwell tuning hypothesis from llama wiki).
__global__ void __launch_bounds__(MMVQ_NWARPS2 * 32, 1) matvec_q4_k_q8_n2(
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
    // nwarps=2 → 4 parallel kbx streams (tid/16 in 0..3), stride 4.
    for (unsigned kbx = tid / 16u; kbx < nb; kbx += 4u) {
        int iqs = (int)(2u * (tid % 16u));
        acc += vec_dot_q4_k_q8(wrow + kbx * 144u, xr + kbx * 8u, iqs);
    }
    pdl_lc();
    acc = warp_sum_f(acc);
    __shared__ float wacc[MMVQ_NWARPS2];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = wacc[0] + wacc[1];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
    }
}

// Two weight rows / block: share X traffic (llama-style rows_per_block=2 path).
__global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8_r2(
    const uint8_t* __restrict__ w,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned tid = warp * 32u + lane;
    unsigned out0 = blockIdx.x * 2u;
    unsigned row = blockIdx.y;
    pdl_sync();
    if (out0 >= n_out || row >= m) { pdl_lc(); return; }
    bool do1 = (out0 + 1u) < n_out;

    const block_q8_1* xr = x + (size_t)row * (n_in / 32u);
    unsigned rb = (n_in / 256u) * 144u;
    const uint8_t* w0 = (w + w_off) + (size_t)out0 * rb;
    const uint8_t* w1 = do1 ? w0 + rb : w0;
    unsigned nb = n_in / 256u;

    float acc0 = 0.f, acc1 = 0.f;
    for (unsigned kbx = tid / 16u; kbx < nb; kbx += 8u) {
        int iqs = (int)(2u * (tid % 16u));
        const block_q8_1* xb = xr + kbx * 8u;
        acc0 += vec_dot_q4_k_q8(w0 + kbx * 144u, xb, iqs);
        if (do1) acc1 += vec_dot_q4_k_q8(w1 + kbx * 144u, xb, iqs);
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
        float* yr = y + (size_t)row * n_out;
        if (add) {
            yr[out0] += s0;
            if (do1) yr[out0 + 1u] += s1;
        } else {
            yr[out0] = s0;
            if (do1) yr[out0 + 1u] = s1;
        }
    }
}

// Blackwell GB10: nwarps=8 when K is long enough (llama should_halve_iters).
#define MMVQ_NWARPS8 8u

__global__ void __launch_bounds__(MMVQ_NWARPS8 * 32, 1) matvec_q4_k_q8_n8(
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
    for (unsigned kbx = tid / 16u; kbx < nb; kbx += 16u) {
        int iqs = (int)(2u * (tid % 16u));
        acc += vec_dot_q4_k_q8(wrow + kbx * 144u, xr + kbx * 8u, iqs);
    }
    pdl_lc();
    acc = warp_sum_f(acc);
    __shared__ float wacc[MMVQ_NWARPS8];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < MMVQ_NWARPS8; i++) s += wacc[i];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
    }
}

__global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q6_k_q8(
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

__global__ void __launch_bounds__(MMVQ_NWARPS8 * 32, 1) matvec_q6_k_q8_n8(
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
    for (unsigned kbx = tid / 32u; kbx < nb; kbx += 8u) {
        int iqs = (int)(tid % 32u);
        acc += vec_dot_q6_k_q8(wrow + kbx * 210u, xr + kbx * 8u, iqs);
    }
    pdl_lc();
    acc = warp_sum_f(acc);
    __shared__ float wacc[MMVQ_NWARPS8];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < MMVQ_NWARPS8; i++) s += wacc[i];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
    }
}

// Down-proj: f32 X with per-tile Q8 pack in-smem (skips separate quantize_q8_1).
// Same nwarps=8 K schedule as matvec_q6_k_q8_n8; quantize matches quantize_q8_1.
__global__ void __launch_bounds__(MMVQ_NWARPS8 * 32, 1) matvec_q6_k_f32_n8(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m,
    unsigned add)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned out = blockIdx.x;
    unsigned row = blockIdx.y;
    pdl_sync();
    if (out >= n_out || row >= m) { pdl_lc(); return; }

    const float* xr = x + (size_t)row * n_in;
    const uint8_t* wrow = (w + w_off) + (size_t)out * (n_in / 256u) * 210u;
    unsigned nb = n_in / 256u;

    __shared__ block_q8_1 xq[MMVQ_NWARPS8][8];
    float acc = 0.f;
    for (unsigned kbx = warp; kbx < nb; kbx += MMVQ_NWARPS8) {
        const float* tile = xr + kbx * 256u;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float v = tile[b * 32 + (int)lane];
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
            xq[warp][b].qs[lane] = (int8_t)qi;
            store_q8_meta(&xq[warp][b], lane, qi, scale);
        }
        __syncwarp();
        acc += vec_dot_q6_k_q8_smem(wrow + kbx * 210u, &xq[warp][0], (int)lane);
    }
    pdl_lc();
    acc = warp_sum_f(acc);
    __shared__ float wacc[MMVQ_NWARPS8];
    if (lane == 0) wacc[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < MMVQ_NWARPS8; i++) s += wacc[i];
        float* yr = y + (size_t)row * n_out;
        if (add) yr[out] += s;
        else yr[out] = s;
    }
}

__global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8_2(
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
        y0[(size_t)row * n_out + out] = silu_f(s0) * s1;
    }
}

// nwarps=8 for long-K gate+up (n_in>=4096); stride 16 over Q4_K blocks.
__global__ void __launch_bounds__(MMVQ_NWARPS8 * 32, 1) matvec_q4_k_q8_2_n8(
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
    for (unsigned kbx = tid / 16u; kbx < nb; kbx += 16u) {
        int iqs = (int)(2u * (tid % 16u));
        const block_q8_1* xb = xr + kbx * 8u;
        acc0 += vec_dot_q4_k_q8(row0 + kbx * 144u, xb, iqs);
        acc1 += vec_dot_q4_k_q8(row1 + kbx * 144u, xb, iqs);
    }
    pdl_lc();
    acc0 = warp_sum_f(acc0);
    acc1 = warp_sum_f(acc1);
    __shared__ float wacc0[MMVQ_NWARPS8], wacc1[MMVQ_NWARPS8];
    if (lane == 0) {
        wacc0[warp] = acc0;
        wacc1[warp] = acc1;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float s0 = 0.f, s1 = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < MMVQ_NWARPS8; i++) {
            s0 += wacc0[i];
            s1 += wacc1[i];
        }
        y0[(size_t)row * n_out + out] = silu_f(s0) * s1;
    }
}

// One block = 32 outs. Same 4-warp Q4_K reduction as matvec_q4_k_q8_2 (correct),
// then pack block_q8_1. Serial over outs — slower than parallel _2 + quantize on 5090;
// keep behind ALLPAKA_GU_Q8=1 only.
__global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8_2_q8(
    const uint8_t* __restrict__ w0,
    const uint8_t* __restrict__ w1,
    const block_q8_1* __restrict__ x,
    float* __restrict__ y0,
    block_q8_1* __restrict__ yq,
    unsigned n_in, unsigned n_out,
    unsigned long long w_off0, unsigned long long w_off1, unsigned m)
{
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned tid = warp * 32u + lane;
    unsigned out0 = blockIdx.x * 32u;
    unsigned row = blockIdx.y;
    pdl_sync();
    if (out0 >= n_out || row >= m) { pdl_lc(); return; }

    const block_q8_1* xr = x + (size_t)row * (n_in / 32u);
    unsigned rb = (n_in / 256u) * 144u;
    unsigned nb = n_in / 256u;
    const uint8_t* base0 = w0 + w_off0;
    const uint8_t* base1 = w1 + w_off1;

    __shared__ float vals[32];
    __shared__ float wacc0[MMVQ_NWARPS], wacc1[MMVQ_NWARPS];

    for (unsigned local = 0; local < 32u; local++) {
        unsigned out = out0 + local;
        float acc0 = 0.f, acc1 = 0.f;
        if (out < n_out) {
            const uint8_t* row0 = base0 + (size_t)out * rb;
            const uint8_t* row1 = base1 + (size_t)out * rb;
            for (unsigned kbx = tid / 16u; kbx < nb; kbx += 8u) {
                int iqs = (int)(2u * (tid % 16u));
                const block_q8_1* xb = xr + kbx * 8u;
                acc0 += vec_dot_q4_k_q8(row0 + kbx * 144u, xb, iqs);
                acc1 += vec_dot_q4_k_q8(row1 + kbx * 144u, xb, iqs);
            }
        }
        acc0 = warp_sum_f(acc0);
        acc1 = warp_sum_f(acc1);
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
            float v = (out < n_out) ? silu_f(s0) * s1 : 0.f;
            vals[local] = v;
            if (out < n_out) {
                y0[(size_t)row * n_out + out] = v;
            }
        }
        __syncthreads();
    }
    pdl_lc();

    if (warp == 0) {
        float v = vals[lane];
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
        block_q8_1* b = yq + (size_t)row * (n_out / 32u) + blockIdx.x;
        b->qs[lane] = (int8_t)qi;
        store_q8_meta(b, lane, qi, scale);
    }
}

__global__ void __launch_bounds__(MMVQ_NWARPS * 32, 1) matvec_q4_k_q8_qkv(
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
                    f32_to_f16_bits(av);
            }
        }
    }
}

// Q4_K_M GQA: Q/K are Q4_K, V is Q6_K (210 B/block).
__global__ void __launch_bounds__(MMVQ_NWARPS * 32, 8) matvec_q4_q4_q6_q8_qkv(
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
    unsigned rb4 = nb * 144u;
    unsigned rb6 = nb * 210u;
    const uint8_t* wrow_q = do_q ? (wq + oq) + (size_t)out * rb4 : nullptr;
    const uint8_t* wrow_k = do_kv ? (wk + ok) + (size_t)out * rb4 : nullptr;
    const uint8_t* wrow_v = do_kv ? (wv + ov) + (size_t)out * rb6 : nullptr;

    float acc_q = 0.f, acc_k = 0.f, acc_v = 0.f;
    for (unsigned kbx = tid / 16u; kbx < nb; kbx += 8u) {
        int iqs = (int)(2u * (tid % 16u));
        const block_q8_1* xb = xr + kbx * 8u;
        if (do_q) acc_q += vec_dot_q4_k_q8(wrow_q + kbx * 144u, xb, iqs);
        if (do_kv) {
            acc_k += vec_dot_q4_k_q8(wrow_k + kbx * 144u, xb, iqs);
            acc_v += vec_dot_q6_k_q8(wrow_v + kbx * 210u, xb, iqs);
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
                    f32_to_f16_bits(av);
            }
        }
    }
}

} // extern "C"
