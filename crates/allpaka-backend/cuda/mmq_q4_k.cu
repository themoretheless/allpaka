// Q4_K MMQ — nvcc -ptx -arch=sm_120
// Tile 128×64 (llama Ampere I×J), K-step 256, mma.m16n8k32 int8 TC.
// Host pre-quantizes X to Q8 (qs + amax scales).

extern "C" {

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef signed char int8_t;

__device__ __forceinline__ float half_to_f32(uint16_t h) {
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

__global__ void quantize_q8_rows(
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
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    float scale = amax / 127.f;
    if (scale < 1e-8f) scale = 1.f;
    if (lane == 0) d[(size_t)row * (n / 32u) + blk] = scale;
    int qi = __float2int_rn(v / scale);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    q[(size_t)row * n + base + lane] = (int8_t)qi;
}

// grid (ceil(n_out/128), ceil(m/64)), block (32, 8), dynamic smem
__global__ void mm_q4_k_mmq(
    const uint8_t* __restrict__ w,
    const int8_t* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ y,
    unsigned n_in, unsigned n_out, unsigned long long w_off, unsigned m)
{
    constexpr unsigned I = 128u;
    constexpr unsigned J = 64u;
    const unsigned warp = threadIdx.y;
    const unsigned lane = threadIdx.x;
    const unsigned tid = warp * 32u + lane;
    const unsigned col0 = blockIdx.x * I;
    const unsigned row0 = blockIdx.y * J;
    if (row0 >= m) return;

    const unsigned nb = n_in / 256u;
    const unsigned rb = nb * 144u;
    const unsigned nd = n_in / 32u;
    const uint8_t* wb = w + w_off;
    const unsigned group = lane >> 2;
    const unsigned tig = lane & 3u;
    const unsigned wcol0 = warp * 16u;

    extern __shared__ char smem[];
    int8_t* As = (int8_t*)smem;
    float* Ad = (float*)(As + J * 256);
    float* Asum = Ad + J * 8;
    uint8_t* Wtile = (uint8_t*)(Asum + J * 8);
    float* Wd = (float*)(Wtile + I * 144);
    float* Wdmin = Wd + I;
    float* Wsc = Wdmin + I;
    float* Wmn = Wsc + I * 8;
    int8_t* Bw = (int8_t*)(Wmn + I * 8); // [8][32][16] as flat 8*32*16
    // Dynamic smem: As+Ad+Asum+Wtile+Wd+Wdmin+Wsc+Wmn+Bw ≈ 52 KiB — set max via host.

    // acc[jt][nt][0..3] for d0,d1,d2,d3
    float acc[4][2][4];
    #pragma unroll
    for (int jt = 0; jt < 4; jt++)
        #pragma unroll
        for (int nt = 0; nt < 2; nt++)
            #pragma unroll
            for (int s = 0; s < 4; s++) acc[jt][nt][s] = 0.f;

    for (unsigned b = 0; b < nb; b++) {
        for (unsigned off = tid; off < J * 256u; off += 256u) {
            unsigned r = off / 256u;
            unsigned k = off - r * 256u;
            unsigned gr = row0 + r;
            As[off] = (gr < m) ? xq[(size_t)gr * n_in + b * 256u + k] : (int8_t)0;
        }
        __syncthreads();
        for (unsigned off = tid; off < J * 8u; off += 256u) {
            unsigned r = off / 8u;
            unsigned g = off - r * 8u;
            unsigned gr = row0 + r;
            float d = (gr < m) ? xd[(size_t)gr * nd + b * 8u + g] : 1.f;
            if (d < 1e-8f) d = 1.f;
            Ad[off] = d;
            int sumq = 0;
            #pragma unroll
            for (int i = 0; i < 32; i++)
                sumq += (int)As[r * 256u + g * 32u + (unsigned)i];
            Asum[off] = d * (float)sumq;
        }
        for (unsigned off = tid; off < I * 144u; off += 256u) {
            unsigned c = off / 144u;
            unsigned o = off - c * 144u;
            unsigned gc = col0 + c;
            Wtile[off] = (gc < n_out) ? wb[(size_t)gc * rb + b * 144u + o] : (uint8_t)0;
        }
        __syncthreads();

        for (unsigned c = tid; c < I; c += 256u) {
            const uint8_t* blk = Wtile + c * 144u;
            Wd[c] = half_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            Wdmin[c] = half_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                float mn;
                float sc = scale_min_k4(g, blk + 4, &mn);
                Wsc[c * 8 + g] = sc;
                Wmn[c * 8 + g] = mn;
            }
        }
        __syncthreads();

        #pragma unroll
        for (unsigned g = 0; g < 8u; g++) {
            unsigned pair = g / 2u;
            unsigned shift = (g & 1u) * 4u;
            #pragma unroll
            for (unsigned c = 0; c < 16u; c++) {
                const uint8_t* blk = Wtile + (wcol0 + c) * 144u;
                Bw[(warp * 32u + lane) * 16u + c] = (int8_t)((blk[16 + pair * 32u + lane] >> shift) & 0xf);
            }
            __syncwarp();

            #pragma unroll
            for (unsigned jt = 0; jt < 4u; jt++) {
                unsigned jbase = jt * 16u;
                unsigned a0 = pack_i8x4(
                    As[(jbase + group) * 256u + g * 32u + 4u * tig + 0],
                    As[(jbase + group) * 256u + g * 32u + 4u * tig + 1],
                    As[(jbase + group) * 256u + g * 32u + 4u * tig + 2],
                    As[(jbase + group) * 256u + g * 32u + 4u * tig + 3]);
                unsigned a1 = pack_i8x4(
                    As[(jbase + group + 8u) * 256u + g * 32u + 4u * tig + 0],
                    As[(jbase + group + 8u) * 256u + g * 32u + 4u * tig + 1],
                    As[(jbase + group + 8u) * 256u + g * 32u + 4u * tig + 2],
                    As[(jbase + group + 8u) * 256u + g * 32u + 4u * tig + 3]);
                unsigned a2 = pack_i8x4(
                    As[(jbase + group) * 256u + g * 32u + 4u * tig + 16],
                    As[(jbase + group) * 256u + g * 32u + 4u * tig + 17],
                    As[(jbase + group) * 256u + g * 32u + 4u * tig + 18],
                    As[(jbase + group) * 256u + g * 32u + 4u * tig + 19]);
                unsigned a3 = pack_i8x4(
                    As[(jbase + group + 8u) * 256u + g * 32u + 4u * tig + 16],
                    As[(jbase + group + 8u) * 256u + g * 32u + 4u * tig + 17],
                    As[(jbase + group + 8u) * 256u + g * 32u + 4u * tig + 18],
                    As[(jbase + group + 8u) * 256u + g * 32u + 4u * tig + 19]);

                float xd0 = Ad[(jbase + group) * 8u + g];
                float xd1 = Ad[(jbase + group + 8u) * 8u + g];
                float xs0 = Asum[(jbase + group) * 8u + g];
                float xs1 = Asum[(jbase + group + 8u) * 8u + g];

                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    unsigned nbase = (unsigned)nt * 8u;
                    unsigned b0 = pack_i8x4(
                        Bw[(warp * 32u + (4u * tig + 0)) * 16u + nbase + group],
                        Bw[(warp * 32u + (4u * tig + 1)) * 16u + nbase + group],
                        Bw[(warp * 32u + (4u * tig + 2)) * 16u + nbase + group],
                        Bw[(warp * 32u + (4u * tig + 3)) * 16u + nbase + group]);
                    unsigned b1 = pack_i8x4(
                        Bw[(warp * 32u + (4u * tig + 16)) * 16u + nbase + group],
                        Bw[(warp * 32u + (4u * tig + 17)) * 16u + nbase + group],
                        Bw[(warp * 32u + (4u * tig + 18)) * 16u + nbase + group],
                        Bw[(warp * 32u + (4u * tig + 19)) * 16u + nbase + group]);
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

                    acc[jt][nt][0] += (Wd[wcol0 + nbase + 2u * tig] * Wsc[(wcol0 + nbase + 2u * tig) * 8u + g] * xd0) * (float)d0
                        - (Wdmin[wcol0 + nbase + 2u * tig] * Wmn[(wcol0 + nbase + 2u * tig) * 8u + g]) * xs0;
                    acc[jt][nt][1] += (Wd[wcol0 + nbase + 2u * tig + 1u] * Wsc[(wcol0 + nbase + 2u * tig + 1u) * 8u + g] * xd0) * (float)d1
                        - (Wdmin[wcol0 + nbase + 2u * tig + 1u] * Wmn[(wcol0 + nbase + 2u * tig + 1u) * 8u + g]) * xs0;
                    acc[jt][nt][2] += (Wd[wcol0 + nbase + 2u * tig] * Wsc[(wcol0 + nbase + 2u * tig) * 8u + g] * xd1) * (float)d2
                        - (Wdmin[wcol0 + nbase + 2u * tig] * Wmn[(wcol0 + nbase + 2u * tig) * 8u + g]) * xs1;
                    acc[jt][nt][3] += (Wd[wcol0 + nbase + 2u * tig + 1u] * Wsc[(wcol0 + nbase + 2u * tig + 1u) * 8u + g] * xd1) * (float)d3
                        - (Wdmin[wcol0 + nbase + 2u * tig + 1u] * Wmn[(wcol0 + nbase + 2u * tig + 1u) * 8u + g]) * xs1;
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned jt = 0; jt < 4u; jt++) {
        unsigned jbase = jt * 16u;
        unsigned r0 = row0 + jbase + group;
        unsigned r1 = row0 + jbase + group + 8u;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            unsigned c0 = col0 + wcol0 + (unsigned)nt * 8u + 2u * tig;
            unsigned c1 = c0 + 1u;
            if (r0 < m && c0 < n_out) y[(size_t)r0 * n_out + c0] = acc[jt][nt][0];
            if (r0 < m && c1 < n_out) y[(size_t)r0 * n_out + c1] = acc[jt][nt][1];
            if (r1 < m && c0 < n_out) y[(size_t)r1 * n_out + c0] = acc[jt][nt][2];
            if (r1 < m && c1 < n_out) y[(size_t)r1 * n_out + c1] = acc[jt][nt][3];
        }
    }
}

} // extern "C"
