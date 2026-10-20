pub const KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Device-scope fences (thread_scope_device) arrived in MSL 3.2. On older
// Metal the only memory order is relaxed, and the threadgroup_barrier with
// mem_device that surrounds every wait and publish is the strongest ordering
// available, so the fence compiles away there instead of failing the whole
// library.
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 320
#define DEVICE_FENCE() atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)
#else
#define DEVICE_FENCE() ((void)0)
#endif

// Batched matvec, one SIMD group per weight row: the 32 lanes walk the row's
// quant blocks round-robin, so the group's reads are contiguous and coalesce,
// and lane partial sums fold with simd_sum at the end. Each lane accumulates
// TILE activation rows at once, so the (expensive) weight bytes are read once
// per tile instead of once per token.
//
// TILE is a function constant, not a kernel argument, and that is the whole
// point: with a runtime row count the compiler cannot unroll the inner loops,
// and every `total[i]` becomes a dynamically indexed thread array, which MSL
// backs with thread-local *memory* rather than registers. The q4_k/q5_k
// kernels carry five such arrays through their innermost loop, so the spill
// costs several memory round trips per quant block. Specialising the pipeline
// per row count keeps all of them in registers.
//
// The host compiles one variant per power of two up to 8 and splits any batch
// into exactly-sized tiles, so a dispatch never reads or writes a row that
// does not exist - there is no padding and no tail guard.
// MSL cannot size an array with a function constant, so the accumulators
// are declared at the maximum tile and only the first TILE entries are
// touched. The loops over them have a compile-time bound, so they unroll,
// every index becomes a literal, and the unused tail is dead code the
// compiler drops - which is what keeps the live entries in registers.
constant constexpr uint MAXT = 8;
constant uint TILE [[function_constant(0)]];

// Lanes per output row. The kernels give each lane whole quant blocks of a
// row, so a fixed 32 lanes leave most of the SIMD group idle whenever a row
// holds fewer than 32 blocks - and that is the normal case, not the corner
// one: a 30B expert's down projection is 768 wide, three 256-element blocks,
// so 3 lanes of 32 did the work and the other 29 waited. The host picks LPR
// to match the row and packs 32/LPR rows into each SIMD group instead.
//
// A power of two dividing 32, so the lane groups line up with the SIMD group
// and the partial sums fold with shuffles inside each group.
constant uint LPR [[function_constant(1)]];

// Expert-indexed dispatch: when set, one dispatch covers `idx.slots` experts
// whose ids a previous dispatch (softmax_topk) wrote on the GPU. The row id
// splits into (slot, row-within-expert); the weight pointer advances by the
// slot's expert, and x optionally advances per slot (the down projection
// consumes per-slot activations, gate/up share one x). Compiled out entirely
// when false - the CPU-routed paths bind nothing extra.
constant bool INDEXED [[function_constant(2)]];

// SwiGLU folded into the activation reads of the DOWN matvec: x is the raw
// gate output, buffer 8 the raw up output, and every load computes
// silu(gate) * up on the fly. Removes the tiny standalone swiglu dispatch
// and one barrier per decoded layer.
constant bool SWIGLU_X [[function_constant(3)]];

// Spin-flag ordering instead of an encoder barrier: the producer publishes
// an epoch to a device atomic after a device-scope fence, and each consumer
// threadgroup's first thread spins until the epoch arrives. Removes the
// full-pipeline drain a memory barrier costs; the producer dispatch is
// encoded first, so it is resident before the consumers saturate the GPU.
constant bool WAIT_X [[function_constant(4)]];

// Dual gate/up dispatch: the flat row space covers the gate matrix's rows
// then the up matrix's (same shape, same expert ids, shared x), reading the
// second weights from buffer 11 and writing its rows to buffer 13. One
// launch per layer instead of two. INDEXED only; the host guarantees
// n_out*slots is even so the 2-row simdgroup pairs never straddle the
// gate/up boundary.
constant bool DUAL_GW [[function_constant(5)]];

// ROWS (MTP verify): the grid covers n_out * slots * n_rows work items -
// every token row's experts in ONE dispatch. Work item g maps to token row
// g / (n_out*slots), whose ids/x/y blocks sit ids_stride / x_row_stride /
// y_row_stride elements apart. Off: plain indexed decode.
constant bool ROWS [[function_constant(6)]];

// Last INDEXED slot reads weights from buffer 11 / w2_off instead of
// ids[slot]*stride. Lets the GLM shared expert's Q8_0 down ride in the
// same dispatch as the routed experts (one more parallel slot), dropping
// the separate shared-down launch. Must be pinned false when unused:
// unset bool constants read as TRUE on this Metal.
constant bool SHARED_TAIL [[function_constant(7)]];

// Llama-style mul_mv_id grid for INDEXED decode: slot = tpg.z, rows on x.
// Default ON; ALLPAKA_MV_ID=0 compiles the flat n_out*slots 1D packing.
// Must be pinned explicitly (unset bool == TRUE on this Metal).
constant bool MV_ID_GRID [[function_constant(8)]];

inline float4 sw4(float4 g, float4 u) {
    return u * (g / (1.0f + exp(-g)));
}
inline float sw1(float g, float u) {
    return u * (g / (1.0f + exp(-g)));
}

struct IdxArgs {
    ulong stride;
    uint slots;
    uint x_stride;
    // ROWS only: per-token-row strides of the ids, x and y blocks, plus the
    // token row count. Unused (zero) otherwise.
    uint ids_stride;
    uint x_row_stride;
    uint y_row_stride;
    uint n_rows;
};

// Sum one lane group's partials. simd_shuffle_down works across the whole
// SIMD group, but with LPR a power of two the halving stops before a fold
// would cross into the neighbouring row's lanes.
inline float lane_sum(float v) {
    if (LPR == 32) {
        return simd_sum(v);
    }
    for (uint off = LPR / 2; off > 0; off >>= 1) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

inline float half_at(device const uchar* p) {
    return (float)(*(device const half*)p);
}

kernel void matvec_q8_0(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    device const float* xu [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    uint jt = tid / LPR;
    uint lane = tid % LPR;
    uint slot = INDEXED ? jt / n_out : 0;
    uint j = INDEXED ? jt % n_out : jt;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    // Rows past the end still run - they read row 0 and drop the result -
    // because an early return would take their lanes out of the shuffles
    // the live rows in this SIMD group depend on.
    bool active = INDEXED ? jt < n_out * idx.slots : j < n_out;
    if (!active) j = 0;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
        if (SWIGLU_X) {
            xu += (ulong)slot * idx.x_stride;
        }
    }
    uint blocks = n_in / 32;
    device const uchar* row = w + w_off + (ulong)j * blocks * 34;
    float total[MAXT] = {0};
    for (uint b = lane; b < blocks; b += LPR) {
        device const uchar* blk = row + b * 34;
        float d = half_at(blk);
        // The 34-byte block is never 4-aligned, so quants load as packed
        // char4 (alignment 1): 8 independent 4-byte loads instead of 32
        // dependent byte loads. Same lesson as matvec_q2_k's word loads -
        // decode is latency-bound on exactly these.
        device const packed_char4* qs = (device const packed_char4*)(blk + 2);
        float s[MAXT] = {0};
        for (uint l4 = 0; l4 < 8; l4++) {
            float4 qv = float4(qs[l4]);
            for (uint i = 0; i < TILE; i++) {
                float4 xv = ((device const float4*)(x + i * n_in + b * 32))[l4];
                if (SWIGLU_X) {
                    float4 uv = ((device const float4*)(xu + i * n_in + b * 32))[l4];
                    xv = sw4(xv, uv);
                }
                s[i] += dot(qv, xv);
            }
        }
        for (uint i = 0; i < TILE; i++) total[i] += d * s[i];
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[i * ycols + jt] = s;
    }
}



// Port of llama.cpp's kernel_mul_mv_q8_0_f32 structure (MIT): NR0 rows per
// SIMD group via LPR (default NR0=2). INDEXED requires n_out % NR0 == 0.
// Decode INDEXED uses llama mul_mv_id geometry: grid.z = slot, grid.x covers
// n_out (ALLPAKA_MV_ID=0 keeps flat n_out*slots 1D). SWIGLU_X / SHARED_TAIL
// as before. qs loads are packed_char4 (34-byte blocks).
kernel void matvec_q8_0_mv(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    device const float* xu [[buffer(8)]],
    device const uchar* w2 [[buffer(11)]],
    constant ulong& w2_off [[buffer(12)]],
    device float* y2 [[buffer(13)]],
    uint3 tpg [[thread_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr short NQ = 8;
    const uint NR0 = 32 / LPR;
    // ID grid: one expert slot per z; x walks rows within that expert.
    // ROWS and DUAL_GW stay on the flat 1D packing. MV_ID_GRID opt-out
    // restores flat INDEXED for A/B (ALLPAKA_MV_ID=0).
    const bool id_grid = INDEXED && !ROWS && !DUAL_GW && MV_ID_GRID;
    uint tok = 0;
    uint slot = 0;
    uint j0 = 0;
    uint y_base = 0;
    uint nrows = 0;
    bool any;
    device const uchar* wm = w;
    ulong woff = w_off;
    device float* yout = y;
    if (id_grid) {
        slot = tpg.z;
        j0 = (tpg.x / 32) * NR0;
        any = slot < idx.slots && j0 < n_out;
        if (!any) {
            j0 = 0;
            slot = 0;
        }
        nrows = any ? min(NR0, n_out - j0) : 0u;
        y_base = slot * n_out + j0;
    } else {
        uint gate_rows = INDEXED ? n_out * idx.slots : n_out;
        uint ycols = DUAL_GW ? gate_rows * 2 : gate_rows;
        uint flat = (tpg.x / 32) * NR0;
        any = flat < (ROWS ? gate_rows * idx.n_rows : ycols);
        if (!any) flat = 0;
        tok = ROWS ? flat / gate_rows : 0;
        if (DUAL_GW && flat >= gate_rows) {
            flat -= gate_rows;
            wm = w2;
            woff = w2_off;
            yout = y2;
        }
        uint fr = ROWS ? flat - tok * gate_rows : flat;
        nrows = any ? min(NR0, gate_rows - fr) : 0u;
        slot = INDEXED ? fr / n_out : 0;
        j0 = INDEXED ? fr % n_out : fr;
        y_base = fr;
    }
    if (INDEXED) {
        if (SHARED_TAIL && slot + 1u == idx.slots) {
            wm = w2;
            woff = w2_off;
        } else {
            wm += ids[(ROWS ? tok * idx.ids_stride : 0u) + slot] * idx.stride;
        }
        x += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
        if (SWIGLU_X) {
            xu += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
        }
    }
    uint nb = n_in / 32;
    ulong nb01 = (ulong)nb * 34;
    device const uchar* row0 = wm + woff + (ulong)j0 * nb01;

    const short ix = tiisg / 4;
    const short il = tiisg % 4;
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    device const float* yb = x + (ulong)ix * 32u + (ulong)il * (uint)NQ;
    device const float* ub = xu + (ulong)ix * 32u + (ulong)il * (uint)NQ;

    for (uint ib = ix; ib < nb; ib += 8) {
        float4 a0 = *(device const float4*)(yb + 0);
        float4 a1 = *(device const float4*)(yb + 4);
        if (SWIGLU_X) {
            a0 = sw4(a0, *(device const float4*)(ub + 0));
            a1 = sw4(a1, *(device const float4*)(ub + 4));
        }
        device const uchar* blk = row0 + (ulong)ib * 34u;
        _Pragma("clang loop unroll(full)")
        for (uint row = 0; row < NR0; ++row) {
            device const packed_char4* qs =
                (device const packed_char4*)(blk + 2 + il * NQ);
            float sumq = dot(float4(qs[0]), a0) + dot(float4(qs[1]), a1);
            sumf[row] += sumq * (float)(*(device const half*)blk);
            ulong step = (row + 1 < nrows) ? nb01 : 0;
            blk += step;
        }
        yb += 8 * 32;
        ub += 8 * 32;
    }

    for (uint row = 0; row < NR0; row++) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && row < nrows) {
            yout[(ROWS ? tok * idx.y_row_stride : 0u) + y_base + row] = s;
        }
    }
}

// Q5_0: 22-byte blocks (f16 scale, 32 high bits, 16 nibble bytes); same
// dispatch geometry as matvec_q8_0, dequantising on load like ggml's
/// dequantize_row_q5_0 (high nibbles form the second half of the block).
kernel void matvec_q5_0(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint jt = tid / LPR;
    uint lane = tid % LPR;
    uint slot = INDEXED ? jt / n_out : 0;
    uint j = INDEXED ? jt % n_out : jt;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    bool active = INDEXED ? jt < n_out * idx.slots : j < n_out;
    if (!active) j = 0;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
    }
    uint blocks = n_in / 32;
    device const uchar* row = w + w_off + (ulong)j * blocks * 22;
    float total[MAXT] = {0};
    for (uint b = lane; b < blocks; b += LPR) {
        device const uchar* blk = row + b * 22;
        float d = half_at(blk);
        uint qh = as_type<uint>(uchar4(blk[2], blk[3], blk[4], blk[5]));
        device const packed_uchar4* qs4 = (device const packed_uchar4*)(blk + 6);
        float s[MAXT] = {0};
        #pragma clang loop unroll(full)
        for (uint l4 = 0; l4 < 4; l4++) {
            uchar4 qb = uchar4(qs4[l4]);
            uint shift0 = l4 * 4;
            uint shift1 = shift0 + 16;
            int4 h0 = int4((qh >> shift0) & 1u, (qh >> (shift0 + 1)) & 1u, (qh >> (shift0 + 2)) & 1u, (qh >> (shift0 + 3)) & 1u) << 4;
            int4 h1 = int4((qh >> shift1) & 1u, (qh >> (shift1 + 1)) & 1u, (qh >> (shift1 + 2)) & 1u, (qh >> (shift1 + 3)) & 1u) << 4;
            float4 q0 = float4((int4(qb & (uchar4)0xF) | h0) - 16);
            float4 q1 = float4((int4(qb >> 4) | h1) - 16);
            for (uint i = 0; i < TILE; i++) {
                device const float4* xi = (device const float4*)(x + i * n_in + b * 32);
                s[i] += dot(q0, xi[l4]) + dot(q1, xi[l4 + 4]);
            }
        }
        for (uint i = 0; i < TILE; i++) total[i] += d * s[i];
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[i * ycols + jt] = s;
    }
}


inline void scale_min_k4(uint j, device const uchar* q, thread float& sc, thread float& mn) {
    if (j < 4) {
        sc = (float)(q[j] & 63);
        mn = (float)(q[j + 4] & 63);
    } else {
        sc = (float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

kernel void matvec_q4_k(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint jt = tid / LPR;
    uint lane = tid % LPR;
    uint slot = INDEXED ? jt / n_out : 0;
    uint j = INDEXED ? jt % n_out : jt;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    // Rows past the end still run - they read row 0 and drop the result -
    // because an early return would take their lanes out of the shuffles
    // the live rows in this SIMD group depend on.
    bool active = INDEXED ? jt < n_out * idx.slots : j < n_out;
    if (!active) j = 0;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
    }
    uint blocks = n_in / 256;
    device const uchar* row = w + w_off + (ulong)j * blocks * 144;
    float total[MAXT] = {0};
    for (uint b = lane; b < blocks; b += LPR) {
        device const uchar* blk = row + b * 144;
        float d = half_at(blk);
        float dmin = half_at(blk + 2);
        device const uchar* packed = blk + 4;
        device const uchar* qs = blk + 16;
        float acc_d[MAXT] = {0};
        float acc_m[MAXT] = {0};
        for (uint pair = 0; pair < 4; pair++) {
            device const uchar4* q4 = (device const uchar4*)(qs + pair * 32);
            uint x_lo = b * 256 + pair * 64;
            float sc1, mn1, sc2, mn2;
            scale_min_k4(pair * 2, packed, sc1, mn1);
            scale_min_k4(pair * 2 + 1, packed, sc2, mn2);
            float dot_lo[MAXT] = {0};
            float dot_hi[MAXT] = {0};
            float sum_lo[MAXT] = {0};
            float sum_hi[MAXT] = {0};
            for (uint l4 = 0; l4 < 8; l4++) {
                uchar4 qq = q4[l4];
                float4 q_lo = float4(qq & (uchar4)0x0F);
                float4 q_hi = float4(qq >> 4);
                for (uint i = 0; i < TILE; i++) {
                    device const float4* xi =
                        (device const float4*)(x + i * n_in + x_lo);
                    float4 xl = xi[l4];
                    float4 xh = xi[8 + l4];
                    dot_lo[i] += dot(q_lo, xl);
                    dot_hi[i] += dot(q_hi, xh);
                    sum_lo[i] += xl.x + xl.y + xl.z + xl.w;
                    sum_hi[i] += xh.x + xh.y + xh.z + xh.w;
                }
            }
            for (uint i = 0; i < TILE; i++) {
                acc_d[i] += sc1 * dot_lo[i] + sc2 * dot_hi[i];
                acc_m[i] += mn1 * sum_lo[i] + mn2 * sum_hi[i];
            }
        }
        for (uint i = 0; i < TILE; i++) {
            total[i] += d * acc_d[i] - dmin * acc_m[i];
        }
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[i * ycols + jt] = s;
    }
}

kernel void matvec_q2_k(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    device const float* xu [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    uint jt = tid / LPR;
    uint lane = tid % LPR;
    uint slot = INDEXED ? jt / n_out : 0;
    uint j = INDEXED ? jt % n_out : jt;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    // Rows past the end still run - they read row 0 and drop the result -
    // because an early return would take their lanes out of the shuffles
    // the live rows in this SIMD group depend on.
    bool active = INDEXED ? jt < n_out * idx.slots : j < n_out;
    if (!active) j = 0;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
        if (SWIGLU_X) {
            xu += (ulong)slot * idx.x_stride;
        }
    }
    uint blocks = n_in / 256;
    device const uchar* row = w + w_off + (ulong)j * blocks * 84;
    float total[MAXT] = {0};
    for (uint b = lane; b < blocks; b += LPR) {
        device const uchar* blk = row + b * 84;
        float d = half_at(blk + 80);
        float dmin = half_at(blk + 82);
        // The whole block loads as words, once: 4 uints of scales and 16 of
        // quants (Q2_K blocks start 4-aligned - 84 is a multiple of 4). The
        // previous form read the same quant byte FOUR times, once per 2-bit
        // shift, one byte at a time: ~260 dependent byte loads per block
        // against 20 independent word loads here. Decode is latency-bound on
        // exactly these loads, and this is where the difference to
        // llama.cpp's per-byte throughput lived.
        device const uint* sw = (device const uint*)blk;
        device const uint* qw = (device const uint*)(blk + 16);
        uint sc_w[4] = { sw[0], sw[1], sw[2], sw[3] };
        float acc_d[MAXT] = {0};
        float acc_m[MAXT] = {0};
        for (uint half_i = 0; half_i < 2; half_i++) {
            for (uint group = 0; group < 2; group++) {
                uint q0 = qw[half_i * 8 + group * 4 + 0];
                uint q1 = qw[half_i * 8 + group * 4 + 1];
                uint q2 = qw[half_i * 8 + group * 4 + 2];
                uint q3 = qw[half_i * 8 + group * 4 + 3];
                for (uint sh2 = 0; sh2 < 4; sh2++) {
                    uint shift = sh2 * 2;
                    uint is = half_i * 8 + sh2 * 2 + group;
                    uint sc = (sc_w[is / 4] >> ((is % 4) * 8)) & 0xFF;
                    float sc_lo = (float)(sc & 0xF);
                    float sc_hi = (float)(sc >> 4);
                    float4 qv0 = float4(as_type<uchar4>((q0 >> shift) & 0x03030303u));
                    float4 qv1 = float4(as_type<uchar4>((q1 >> shift) & 0x03030303u));
                    float4 qv2 = float4(as_type<uchar4>((q2 >> shift) & 0x03030303u));
                    float4 qv3 = float4(as_type<uchar4>((q3 >> shift) & 0x03030303u));
                    uint x0 = b * 256 + half_i * 128 + sh2 * 32 + group * 16;
                    for (uint i = 0; i < TILE; i++) {
                        device const float4* xi =
                            (device const float4*)(x + i * n_in + x0);
                        float4 x0v = xi[0];
                        float4 x1v = xi[1];
                        float4 x2v = xi[2];
                        float4 x3v = xi[3];
                        if (SWIGLU_X) {
                            device const float4* ui =
                                (device const float4*)(xu + i * n_in + x0);
                            x0v = sw4(x0v, ui[0]);
                            x1v = sw4(x1v, ui[1]);
                            x2v = sw4(x2v, ui[2]);
                            x3v = sw4(x3v, ui[3]);
                        }
                        float dotq = dot(qv0, x0v) + dot(qv1, x1v)
                                   + dot(qv2, x2v) + dot(qv3, x3v);
                        float4 xs = x0v + x1v + x2v + x3v;
                        acc_d[i] += sc_lo * dotq;
                        acc_m[i] += sc_hi * (xs.x + xs.y + xs.z + xs.w);
                    }
                }
            }
        }
        for (uint i = 0; i < TILE; i++) {
            total[i] += d * acc_d[i] - dmin * acc_m[i];
        }
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[i * ycols + jt] = s;
    }
}

kernel void matvec_q2_k_ilv(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint jt = tid / LPR;
    uint lane = tid % LPR;
    uint slot = INDEXED ? jt / n_out : 0;
    uint j = INDEXED ? jt % n_out : jt;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    // Rows past the end still run - they read row 0 and drop the result -
    // because an early return would take their lanes out of the shuffles
    // the live rows in this SIMD group depend on.
    bool active = INDEXED ? jt < n_out * idx.slots : j < n_out;
    if (!active) j = 0;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
    }
    uint blocks = n_in / 256;
    device const uchar* row = w + w_off + (ulong)j * blocks * 84;
    float total[MAXT] = {0};
    // Two blocks per lane, loads issued together: at LPR halved the lane
    // owns blocks b and b + LPR, and interleaving their word loads doubles
    // the memory-level parallelism a single lane presents. (Halving LPR
    // WITHOUT this interleave was measured slower twice - the win, if any,
    // has to come from the pairing.)
    for (uint b = lane; b < blocks; b += 2 * LPR) {
        uint bB = b + LPR;
        bool have2 = bB < blocks;
        device const uchar* blk = row + b * 84;
        device const uchar* blkB = row + (have2 ? bB : b) * 84;
        float d = half_at(blk + 80);
        float dmin = half_at(blk + 82);
        float dB = have2 ? half_at(blkB + 80) : 0.0f;
        float dminB = have2 ? half_at(blkB + 82) : 0.0f;
        // The whole block loads as words, once: 4 uints of scales and 16 of
        // quants (Q2_K blocks start 4-aligned - 84 is a multiple of 4). The
        // previous form read the same quant byte FOUR times, once per 2-bit
        // shift, one byte at a time: ~260 dependent byte loads per block
        // against 20 independent word loads here. Decode is latency-bound on
        // exactly these loads, and this is where the difference to
        // llama.cpp's per-byte throughput lived.
        device const uint* sw = (device const uint*)blk;
        device const uint* qw = (device const uint*)(blk + 16);
        device const uint* swB = (device const uint*)blkB;
        device const uint* qwB = (device const uint*)(blkB + 16);
        uint sc_w[4] = { sw[0], sw[1], sw[2], sw[3] };
        uint sc_wB[4] = { swB[0], swB[1], swB[2], swB[3] };
        for (uint half_i = 0; half_i < 2; half_i++) {
            for (uint group = 0; group < 2; group++) {
                uint q0 = qw[half_i * 8 + group * 4 + 0];
                uint q1 = qw[half_i * 8 + group * 4 + 1];
                uint q2 = qw[half_i * 8 + group * 4 + 2];
                uint q3 = qw[half_i * 8 + group * 4 + 3];
                uint q0B = qwB[half_i * 8 + group * 4 + 0];
                uint q1B = qwB[half_i * 8 + group * 4 + 1];
                uint q2B = qwB[half_i * 8 + group * 4 + 2];
                uint q3B = qwB[half_i * 8 + group * 4 + 3];
                for (uint sh2 = 0; sh2 < 4; sh2++) {
                    uint shift = sh2 * 2;
                    uint is = half_i * 8 + sh2 * 2 + group;
                    uint sc = (sc_w[is / 4] >> ((is % 4) * 8)) & 0xFF;
                    uint scB = (sc_wB[is / 4] >> ((is % 4) * 8)) & 0xFF;
                    float dl = d * (float)(sc & 0xF);
                    float ml = dmin * (float)(sc >> 4);
                    float dlB = dB * (float)(scB & 0xF);
                    float mlB = dminB * (float)(scB >> 4);
                    float4 qv0 = float4(as_type<uchar4>((q0 >> shift) & 0x03030303u));
                    float4 qv1 = float4(as_type<uchar4>((q1 >> shift) & 0x03030303u));
                    float4 qv2 = float4(as_type<uchar4>((q2 >> shift) & 0x03030303u));
                    float4 qv3 = float4(as_type<uchar4>((q3 >> shift) & 0x03030303u));
                    float4 qv0B = float4(as_type<uchar4>((q0B >> shift) & 0x03030303u));
                    float4 qv1B = float4(as_type<uchar4>((q1B >> shift) & 0x03030303u));
                    float4 qv2B = float4(as_type<uchar4>((q2B >> shift) & 0x03030303u));
                    float4 qv3B = float4(as_type<uchar4>((q3B >> shift) & 0x03030303u));
                    uint x0 = b * 256 + half_i * 128 + sh2 * 32 + group * 16;
                    uint x0B = (b + LPR) * 256 + half_i * 128 + sh2 * 32 + group * 16;
                    for (uint i = 0; i < TILE; i++) {
                        device const float4* xi =
                            (device const float4*)(x + i * n_in + x0);
                        device const float4* xiB =
                            (device const float4*)(x + i * n_in + x0B);
                        float4 x0v = xi[0];
                        float4 x1v = xi[1];
                        float4 x2v = xi[2];
                        float4 x3v = xi[3];
                        float4 x0B_ = have2 ? xiB[0] : float4(0.0f);
                        float4 x1B_ = have2 ? xiB[1] : float4(0.0f);
                        float4 x2B_ = have2 ? xiB[2] : float4(0.0f);
                        float4 x3B_ = have2 ? xiB[3] : float4(0.0f);
                        float dotq = dot(qv0, x0v) + dot(qv1, x1v)
                                   + dot(qv2, x2v) + dot(qv3, x3v);
                        float dotqB = dot(qv0B, x0B_) + dot(qv1B, x1B_)
                                    + dot(qv2B, x2B_) + dot(qv3B, x3B_);
                        float4 xs = x0v + x1v + x2v + x3v;
                        float4 xsB = x0B_ + x1B_ + x2B_ + x3B_;
                        total[i] += dl * dotq - ml * (xs.x + xs.y + xs.z + xs.w);
                        total[i] += dlB * dotqB - mlB * (xsB.x + xsB.y + xsB.z + xsB.w);
                    }
                }
            }
        }
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[i * ycols + jt] = s;
    }
}


// Port of llama.cpp's kernel_mul_mv_q2_K_f32 structure (MIT). The decisive
// differences to matvec_q2_k above: 32 lanes cooperate on the SAME four
// quant blocks (each lane owns a fixed 8-element slice of each 128-element
// half), the activations load into registers ONCE (yl) and are reused for
// FOUR consecutive output rows, and the 2-bit shift is never executed - the
// mask stays in place and the shift folds into the per-scale multipliers
// 1, 1/4, 1/16, 1/64 (with the odd bytes carrying an extra 1/256).
// Activation traffic drops 4x and the quant decode loses all shift ALU.
//
// Geometry contract with the host: exactly 8 threads per output row, i.e.
// lanes_per_row() returns 8 for Q2_K when this kernel is selected, so one
// 32-lane SIMD group covers 4 rows. TILE is single-column only; the host
// substitutes the plain kernel for tile > 1. INDEXED requires n_out % 4 == 0
// so a SIMD group never straddles two experts; the host checks that too.
kernel void matvec_q2_k_mv(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    device const uchar* w2 [[buffer(11)]],
    constant ulong& w2_off [[buffer(12)]],
    device float* y2 [[buffer(13)]],
    uint3 tpg [[thread_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // Rows per SIMD group rides on the LPR function constant (NR0 = 32/LPR)
    // so the host can sweep the occupancy-vs-reuse tradeoff: more rows reuse
    // the registered activations more but launch fewer SIMD groups.
    const uint NR0 = 32 / LPR;
    const bool id_grid = INDEXED && !ROWS && !DUAL_GW && MV_ID_GRID;
    uint tok = 0;
    uint slot = 0;
    uint j0 = 0;
    uint y_base = 0;
    uint nrows = 0;
    bool any;
    device const uchar* wm = w;
    ulong woff = w_off;
    device float* yout = y;
    if (id_grid) {
        slot = tpg.z;
        j0 = (tpg.x / 32) * NR0;
        any = slot < idx.slots && j0 < n_out;
        if (!any) {
            j0 = 0;
            slot = 0;
        }
        nrows = any ? min(NR0, n_out - j0) : 0u;
        y_base = slot * n_out + j0;
    } else {
        uint gate_rows = INDEXED ? n_out * idx.slots : n_out;
        uint ycols = DUAL_GW ? gate_rows * 2 : gate_rows;
        uint flat = (tpg.x / 32) * NR0;
        any = flat < (ROWS ? gate_rows * idx.n_rows : ycols);
        if (!any) flat = 0;
        tok = ROWS ? flat / gate_rows : 0;
        if (DUAL_GW && flat >= gate_rows) {
            flat -= gate_rows;
            wm = w2;
            woff = w2_off;
            yout = y2;
        }
        uint fr = ROWS ? flat - tok * gate_rows : flat;
        nrows = any ? min(NR0, gate_rows - fr) : 0u;
        slot = INDEXED ? fr / n_out : 0;
        j0 = INDEXED ? fr % n_out : fr;
        y_base = fr;
    }
    if (INDEXED) {
        wm += ids[(ROWS ? tok * idx.ids_stride : 0u) + slot] * idx.stride;
        x += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
    }
    uint nb = n_in / 256;
    ulong nb01 = (ulong)nb * 84;
    device const uchar* row0 = wm + woff + (ulong)j0 * nb01;

    float yl[32];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const short ix = tiisg / 8; // block among the 4 in flight
    const short it = tiisg % 8; // slice within the block
    const short iq = it / 4;    // first or second 128-element half
    const short ir = it % 4;    // 8-element chunk within the half
    const short is = (8 * ir) / 16;

    device const float* y4 = x + ix * 256 + 128 * iq + 8 * ir;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i++) {
            yl[i + 0] = y4[i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];
            sumy[1] += yl[i + 8];
            yl[i + 16] = y4[i + 64];
            sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96];
            sumy[3] += yl[i + 24];
        }
        device const uchar* blk = row0 + ib * 84;
        device const uchar* sc = blk + 8 * iq + is;
        device const ushort* qs = (device const ushort*)(blk + 16) + 16 * iq + 4 * ir;
        device const uchar* dh = blk + 80;

        #pragma clang loop unroll(full)
        for (uint row = 0; row < NR0; row++) {
            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            const ushort4 qv = *(device const packed_ushort4*)qs;
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i += 2) {
                const ushort qsi = qv[i / 2];
                acc1[0] += yl[i + 0] * (qsi & 0x0003);
                acc2[0] += yl[i + 1] * (qsi & 0x0300);
                acc1[1] += yl[i + 8] * (qsi & 0x000c);
                acc2[1] += yl[i + 9] * (qsi & 0x0c00);
                acc1[2] += yl[i + 16] * (qsi & 0x0030);
                acc2[2] += yl[i + 17] * (qsi & 0x3000);
                acc1[3] += yl[i + 24] * (qsi & 0x00c0);
                acc2[3] += yl[i + 25] * (qsi & 0xc000);
            }
            half2 dm = *(device const half2*)dh;
            float dall = (float)dm.x;
            float dmin = (float)dm.y * (1.0f / 16.0f);
            sumf[row] +=
                dall * ((acc1[0] + (1.0f / 256.0f) * acc2[0]) * (sc[0] & 0xF) * (1.0f / 1.0f) +
                        (acc1[1] + (1.0f / 256.0f) * acc2[1]) * (sc[2] & 0xF) * (1.0f / 4.0f) +
                        (acc1[2] + (1.0f / 256.0f) * acc2[2]) * (sc[4] & 0xF) * (1.0f / 16.0f) +
                        (acc1[3] + (1.0f / 256.0f) * acc2[3]) * (sc[6] & 0xF) * (1.0f / 64.0f)) -
                dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                        sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            ulong step = (row + 1 < nrows) ? nb01 : 0;
            qs = (device const ushort*)((device const uchar*)qs + step);
            sc += step;
            dh += step;
        }
        y4 += 4 * 256;
    }

    for (uint row = 0; row < NR0; row++) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && row < nrows) {
            yout[(ROWS ? tok * idx.y_row_stride : 0u) + y_base + row] = s;
        }
    }
}

kernel void matvec_q3_k(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint jt = tid / LPR;
    uint lane = tid % LPR;
    uint slot = INDEXED ? jt / n_out : 0;
    uint j = INDEXED ? jt % n_out : jt;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    bool active = INDEXED ? jt < n_out * idx.slots : j < n_out;
    if (!active) j = 0;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
    }
    uint blocks = n_in / 256;
    device const uchar* row = w + w_off + (ulong)j * blocks * 110;
    float total[MAXT] = {0};
    for (uint b = lane; b < blocks; b += LPR) {
        device const uchar* blk = row + b * 110;
        device const uchar* packed = blk + 96;
        float d_all = half_at(blk + 108);
        // A 110-byte block is only 2-aligned, so quants and the high-bit
        // mask load as ushorts and pair up into words in registers. Like the
        // q2_k rewrite: the bytes load ONCE per block and all four 2-bit
        // planes extract from registers, where the old form re-read every
        // byte per plane.
        device const ushort* hm16 = (device const ushort*)blk;
        device const ushort* qs16 = (device const ushort*)(blk + 32);
        float blk_sum[MAXT] = {0};
        for (uint half_i = 0; half_i < 2; half_i++) {
            for (uint group = 0; group < 2; group++) {
                uint qword[4];
                uint hword[4];
                for (uint k = 0; k < 4; k++) {
                    uint qbase = half_i * 16 + group * 8 + k * 2;
                    qword[k] = (uint)qs16[qbase] | ((uint)qs16[qbase + 1] << 16);
                    uint hbase = group * 8 + k * 2;
                    hword[k] = (uint)hm16[hbase] | ((uint)hm16[hbase + 1] << 16);
                }
                for (uint si = 0; si < 4; si++) {
                    uint is = half_i * 8 + si * 2 + group;
                    uint lo = (is < 8) ? (packed[is] & 0xF) : (packed[is - 8] >> 4);
                    uint hi = (packed[8 + (is % 4)] >> (2 * (is / 4))) & 3;
                    float sc = (float)((int)(lo | (hi << 4)) - 32);
                    uint mrep = 0x01010101u * (uint)(1 << (half_i * 4 + si));
                    uint x0 = b * 256 + half_i * 128 + si * 32 + group * 16;
                    for (uint i = 0; i < TILE; i++) {
                        device const float4* xi =
                            (device const float4*)(x + i * n_in + x0);
                        float dotq = 0.0f;
                        for (uint k = 0; k < 4; k++) {
                            // The absent high bit acts as -4 on the 2-bit quant.
                            int4 qi = int4(as_type<uchar4>(
                                    (qword[k] >> (2 * si)) & 0x03030303u))
                                - select(int4(4), int4(0),
                                    as_type<uchar4>(hword[k] & mrep) != uchar4(0));
                            dotq += dot(float4(qi), xi[k]);
                        }
                        blk_sum[i] += sc * dotq;
                    }
                }
            }
        }
        for (uint i = 0; i < TILE; i++) {
            total[i] += d_all * blk_sum[i];
        }
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[i * ycols + jt] = s;
    }
}

// Port of llama.cpp's kernel_mul_mv_q3_K_f32 structure (MIT), same idea as
// matvec_q2_k_mv: the SIMD group cooperates on four blocks, activations load
// once into registers and are reused for TWO rows, shifts fold into
// constants picked per lane role. Geometry contract: 16 threads per output
// row (lanes_per_row returns 16), one SIMD group per 2 rows. Single-column
// only; INDEXED requires n_out % 2 == 0.
kernel void matvec_q3_k_mv(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    device const float* xu [[buffer(8)]],
    device const uchar* w2 [[buffer(11)]],
    constant ulong& w2_off [[buffer(12)]],
    device float* y2 [[buffer(13)]],
    uint3 tpg [[thread_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint NR0 = 2;
    const bool id_grid = INDEXED && !ROWS && !DUAL_GW && MV_ID_GRID;
    uint tok = 0;
    uint slot = 0;
    uint j0 = 0;
    uint y_base = 0;
    uint nrows = 0;
    bool any;
    device const uchar* wm = w;
    ulong woff = w_off;
    device float* yout = y;
    if (id_grid) {
        slot = tpg.z;
        j0 = (tpg.x / 32) * NR0;
        any = slot < idx.slots && j0 < n_out;
        if (!any) {
            j0 = 0;
            slot = 0;
        }
        nrows = any ? min(NR0, n_out - j0) : 0u;
        y_base = slot * n_out + j0;
    } else {
        uint gate_rows = INDEXED ? n_out * idx.slots : n_out;
        uint ycols = DUAL_GW ? gate_rows * 2 : gate_rows;
        uint flat = (tpg.x / 32) * NR0;
        any = flat < (ROWS ? gate_rows * idx.n_rows : ycols);
        if (!any) flat = 0;
        tok = ROWS ? flat / gate_rows : 0;
        if (DUAL_GW && flat >= gate_rows) {
            flat -= gate_rows;
            wm = w2;
            woff = w2_off;
            yout = y2;
        }
        uint fr = ROWS ? flat - tok * gate_rows : flat;
        nrows = any ? min(NR0, gate_rows - fr) : 0u;
        slot = INDEXED ? fr / n_out : 0;
        j0 = INDEXED ? fr % n_out : fr;
        y_base = fr;
    }
    if (INDEXED) {
        wm += ids[(ROWS ? tok * idx.ids_stride : 0u) + slot] * idx.stride;
        x += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
        if (SWIGLU_X) {
            xu += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
        }
    }
    uint nb = n_in / 256;
    ulong nb01 = (ulong)nb * 110;
    device const uchar* row0 = wm + woff + (ulong)j0 * nb01;

    float yl[32];

    const short t  = tiisg / 4;
    const short ix = tiisg % 4;
    const short ip = t / 4;           // 0 or 1
    const short il = 2 * ((t % 4) / 2); // 0 or 2
    const short ir = t % 2;
    const short l0 = 8 * ir;

    // The compiler needs the per-role masks as tables to fold them.
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00},
                        {0x0030, 0x3000, 0x00c0, 0xc000}};
    const ushort4 hm = mm[2 * ip + il / 2];
    const short shift = 2 * il;
    const float v1 = il == 0 ? 4.0f : 64.0f;
    const float v2 = 4.0f * v1;
    const ushort s_shift1 = 4 * ip;
    const ushort s_shift2 = s_shift1 + il;
    const short q_offset = 32 * ip + l0;
    const short y_offset = 128 * ip + 32 * il + l0;

    device const float* y1 = x + ix * 256 + y_offset;
    device const float* u1 = xu + ix * 256 + y_offset;

    uint scales32, aux32;
    thread ushort* scales16 = (thread ushort*)&scales32;
    thread const char* scales = (thread const char*)&scales32;

    float sumf1[2] = {0.0f, 0.0f};
    float sumf2[2] = {0.0f, 0.0f};

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 yl0 = *(device const float4*)(y1 + 0);
        float4 yl1 = *(device const float4*)(y1 + 4);
        float4 yl2 = *(device const float4*)(y1 + 16);
        float4 yl3 = *(device const float4*)(y1 + 20);
        float4 yl4 = *(device const float4*)(y1 + 32);
        float4 yl5 = *(device const float4*)(y1 + 36);
        float4 yl6 = *(device const float4*)(y1 + 48);
        float4 yl7 = *(device const float4*)(y1 + 52);
        if (SWIGLU_X) {
            yl0 = sw4(yl0, *(device const float4*)(u1 + 0));
            yl1 = sw4(yl1, *(device const float4*)(u1 + 4));
            yl2 = sw4(yl2, *(device const float4*)(u1 + 16));
            yl3 = sw4(yl3, *(device const float4*)(u1 + 20));
            yl4 = sw4(yl4, *(device const float4*)(u1 + 32));
            yl5 = sw4(yl5, *(device const float4*)(u1 + 36));
            yl6 = sw4(yl6, *(device const float4*)(u1 + 48));
            yl7 = sw4(yl7, *(device const float4*)(u1 + 52));
        }
        *((thread float4*)(yl + 0)) = yl0;
        *((thread float4*)(yl + 4)) = yl1;
        *((thread float4*)(yl + 8)) = yl2;
        *((thread float4*)(yl + 12)) = yl3;
        *((thread float4*)(yl + 16)) = yl4;
        *((thread float4*)(yl + 20)) = yl5;
        *((thread float4*)(yl + 24)) = yl6;
        *((thread float4*)(yl + 28)) = yl7;
        device const uchar* blk = row0 + ib * 110;
        device const ushort* q = (device const ushort*)(blk + 32 + q_offset);
        device const ushort* h = (device const ushort*)(blk + l0);
        device const ushort* a = (device const ushort*)(blk + 96);
        device const uchar* dh = blk + 108;

        #pragma clang loop unroll(full)
        for (short row = 0; row < 2; ++row) {
            const float d_all = half_at(dh);

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il + 0];
            scales16[1] = a[il + 1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            const ushort4 qv0 = *(device const packed_ushort4*)q;
            const ushort4 hv0 = *(device const packed_ushort4*)h;
            #pragma clang loop unroll(full)
            for (short l = 0; l < 8; l += 2) {
                const int qsv = qv0[l / 2];
                const ushort hsv = hv0[l / 2];
                s1 += yl[l + 0] * (qsv & qm[il / 2][0]);
                s2 += yl[l + 1] * (qsv & qm[il / 2][1]);
                s3 += ((hsv & hm[0]) ? 0.0f : yl[l + 0]) + ((hsv & hm[1]) ? 0.0f : yl[l + 1]);
                s4 += yl[l + 16] * (qsv & qm[il / 2][2]);
                s5 += yl[l + 17] * (qsv & qm[il / 2][3]);
                s6 += ((hsv & hm[2]) ? 0.0f : yl[l + 16]) + ((hsv & hm[3]) ? 0.0f : yl[l + 17]);
            }
            float d1 = d_all * (s1 + (1.0f / 256.0f) * s2 - s3 * v1);
            float d2 = d_all * (s4 + (1.0f / 256.0f) * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            const ushort4 qv1 = *(device const packed_ushort4*)(q + 8);
            const ushort4 hv1 = *(device const packed_ushort4*)(h + 8);
            #pragma clang loop unroll(full)
            for (short l = 0; l < 8; l += 2) {
                const int qsv = qv1[l / 2];
                const ushort hsv = hv1[l / 2];
                s1 += yl[l + 8] * (qsv & qm[il / 2][0]);
                s2 += yl[l + 9] * (qsv & qm[il / 2][1]);
                s3 += ((hsv & hm[0]) ? 0.0f : yl[l + 8]) + ((hsv & hm[1]) ? 0.0f : yl[l + 9]);
                s4 += yl[l + 24] * (qsv & qm[il / 2][2]);
                s5 += yl[l + 25] * (qsv & qm[il / 2][3]);
                s6 += ((hsv & hm[2]) ? 0.0f : yl[l + 24]) + ((hsv & hm[3]) ? 0.0f : yl[l + 25]);
            }
            d1 = d_all * (s1 + (1.0f / 256.0f) * s2 - s3 * v1);
            d2 = d_all * (s4 + (1.0f / 256.0f) * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            ulong step = ((uint)row + 1 < nrows) ? nb01 : 0;
            q = (device const ushort*)((device const uchar*)q + step);
            h = (device const ushort*)((device const uchar*)h + step);
            a = (device const ushort*)((device const uchar*)a + step);
            dh += step;
        }
        y1 += 4 * 256;
        u1 += 4 * 256;
    }

    for (uint row = 0; row < 2; row++) {
        float s = simd_sum((sumf1[row] + 0.25f * sumf2[row]) / (1 << shift));
        if (tiisg == 0 && row < nrows) {
            yout[(ROWS ? tok * idx.y_row_stride : 0u) + y_base + row] = s;
        }
    }
}

// Port of llama.cpp's kernel_mul_mv_q6_K_f32 structure (MIT): 2 rows per
// SIMD group, TWO blocks in flight (16 lanes each), activations registered
// once per block and reused for both rows, 6-bit reconstruction as
// (low nibble | masked-high-bits) - 32 per element. 16 threads per output
// row; INDEXED requires n_out % 2 == 0.
kernel void matvec_q6_k_mv(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    device const uchar* w2 [[buffer(11)]],
    constant ulong& w2_off [[buffer(12)]],
    device float* y2 [[buffer(13)]],
    uint3 tpg [[thread_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;

    const uint NR0 = 2;
    const bool id_grid = INDEXED && !ROWS && !DUAL_GW && MV_ID_GRID;
    uint tok = 0;
    uint slot = 0;
    uint j0 = 0;
    uint y_base = 0;
    uint nrows = 0;
    bool any;
    device const uchar* wm = w;
    ulong woff = w_off;
    device float* yout = y;
    if (id_grid) {
        slot = tpg.z;
        j0 = (tpg.x / 32) * NR0;
        any = slot < idx.slots && j0 < n_out;
        if (!any) {
            j0 = 0;
            slot = 0;
        }
        nrows = any ? min(NR0, n_out - j0) : 0u;
        y_base = slot * n_out + j0;
    } else {
        uint gate_rows = INDEXED ? n_out * idx.slots : n_out;
        uint ycols = DUAL_GW ? gate_rows * 2 : gate_rows;
        uint flat = (tpg.x / 32) * NR0;
        any = flat < (ROWS ? gate_rows * idx.n_rows : ycols);
        if (!any) flat = 0;
        tok = ROWS ? flat / gate_rows : 0;
        if (DUAL_GW && flat >= gate_rows) {
            flat -= gate_rows;
            wm = w2;
            woff = w2_off;
            yout = y2;
        }
        uint fr = ROWS ? flat - tok * gate_rows : flat;
        nrows = any ? min(NR0, gate_rows - fr) : 0u;
        slot = INDEXED ? fr / n_out : 0;
        j0 = INDEXED ? fr % n_out : fr;
        y_base = fr;
    }
    if (INDEXED) {
        wm += ids[(ROWS ? tok * idx.ids_stride : 0u) + slot] * idx.stride;
        x += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
    }
    uint nb = n_in / 256;
    ulong nb01 = (ulong)nb * 210;
    device const uchar* row0 = wm + woff + (ulong)j0 * nb01;

    const short t = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = t / 8;
    const short il = t % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    float yl[16];
    float sumf[2] = {0.0f, 0.0f};

    for (uint ib = ix; ib < nb; ib += 2) {
        device const uchar* blk = row0 + ib * 210;
        device const uchar* q1 = blk + q_offset_l;
        device const uchar* q2 = q1 + 32;
        device const uchar* qh = blk + 128 + q_offset_h;
        device const char* sc = (device const char*)(blk + 192) + is;
        device const uchar* dh = blk + 208;

        device const float* yv = x + ib * 256 + y_offset;
        float4 yv0 = *(device const float4*)(yv + 0);
        float4 yv1 = *(device const float4*)(yv + 32);
        float4 yv2 = *(device const float4*)(yv + 64);
        float4 yv3 = *(device const float4*)(yv + 96);
        _Pragma("clang loop unroll(full)")
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = yv0[l];
            yl[4 * l + 1] = yv1[l];
            yl[4 * l + 2] = yv2[l];
            yl[4 * l + 3] = yv3[l];
        }

        for (uint row = 0; row < 2; ++row) {
            const uchar4 q1_v = uchar4(*(device const packed_uchar4*)q1);
            const uchar4 q2_v = uchar4(*(device const packed_uchar4*)q2);
            const uchar4 qh_v = uchar4(*(device const packed_uchar4*)qh);
            float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
            _Pragma("clang loop unroll(full)")
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((char)((q1_v[l] & 0xF) | ((qh_v[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((char)((q2_v[l] & 0xF) | ((qh_v[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((char)((q1_v[l] >> 4) | ((qh_v[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((char)((q2_v[l] >> 4) | ((qh_v[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += half_at(dh) *
                (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            ulong step = (row + 1 < nrows) ? nb01 : 0;
            q1 += step;
            q2 += step;
            qh += step;
            sc += step;
            dh += step;
        }
    }

    for (uint row = 0; row < 2; row++) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && row < nrows) {
            yout[(ROWS ? tok * idx.y_row_stride : 0u) + y_base + row] = s;
        }
    }
}

// Port of llama.cpp's kernel_mul_mv_q4_K_f32 structure (MIT), same skeleton
// as matvec_q3_k_mv: NR0 rows per SIMD group via LPR (default NR0=2),
// activations registered once (yl low half, yh high half), nibble masks in
// place with 1/16 and 1/256 folded into the scale multipliers. INDEXED
// requires n_out % NR0 == 0 so a group never straddles expert slots.
kernel void matvec_q4_k_mv(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    device const float* xu [[buffer(8)]],
    device atomic_uint* wait_flag [[buffer(9)]],
    constant uint& wait_epoch [[buffer(10)]],
    device const uchar* w2 [[buffer(11)]],
    constant ulong& w2_off [[buffer(12)]],
    device float* y2 [[buffer(13)]],
    uint3 tpg [[thread_position_in_grid]],
    uint3 ltid_v [[thread_position_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    if (WAIT_X) {
        if (ltid_v.x == 0) {
            while (atomic_load_explicit(wait_flag, memory_order_relaxed) < wait_epoch) {
            }
            DEVICE_FENCE();
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    const ushort kmask1 = 0x3f3f;
    const ushort kmask2 = 0x0f0f;
    const ushort kmask3 = 0xc0c0;

    const uint NR0 = 32 / LPR;
    // ID grid for plain INDEXED decode; DUAL_GW / ROWS stay flat 1D.
    const bool id_grid = INDEXED && !ROWS && !DUAL_GW && MV_ID_GRID;
    uint tok = 0;
    uint slot = 0;
    uint j0 = 0;
    uint y_base = 0;
    uint nrows = 0;
    bool any;
    device const uchar* wm = w;
    ulong woff = w_off;
    device float* yout = y;
    if (id_grid) {
        slot = tpg.z;
        j0 = (tpg.x / 32) * NR0;
        any = slot < idx.slots && j0 < n_out;
        if (!any) {
            j0 = 0;
            slot = 0;
        }
        nrows = any ? min(NR0, n_out - j0) : 0u;
        y_base = slot * n_out + j0;
    } else {
        uint gate_rows = INDEXED ? n_out * idx.slots : n_out;
        uint ycols = DUAL_GW ? gate_rows * 2 : gate_rows;
        uint flat = (tpg.x / 32) * NR0;
        any = flat < (ROWS ? gate_rows * idx.n_rows : ycols);
        if (!any) flat = 0;
        tok = ROWS ? flat / gate_rows : 0;
        // The second half of the flat rows belongs to the up matrix.
        if (DUAL_GW && flat >= gate_rows) {
            flat -= gate_rows;
            wm = w2;
            woff = w2_off;
            yout = y2;
        }
        uint fr = ROWS ? flat - tok * gate_rows : flat;
        nrows = any ? min(NR0, gate_rows - fr) : 0u;
        slot = INDEXED ? fr / n_out : 0;
        j0 = INDEXED ? fr % n_out : fr;
        y_base = fr;
    }
    if (INDEXED) {
        wm += ids[(ROWS ? tok * idx.ids_stride : 0u) + slot] * idx.stride;
        x += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
        if (SWIGLU_X) {
            xu += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
        }
    }
    uint nb = n_in / 256;
    ulong nb01 = (ulong)nb * 144;
    device const uchar* row0 = wm + woff + (ulong)j0 * nb01;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float yl[16];
    float yh[16];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;
    device const float* u4 = xu + ix * 256 + 64 * iq + 8 * ir;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        // float4 activation loads (same idea as Q8 `_mv`); weight dequant
        // still walks yl/yh element-wise.
        float4 yl0 = *(device const float4*)(y4 + 0);
        float4 yl1 = *(device const float4*)(y4 + 4);
        float4 yl2 = *(device const float4*)(y4 + 32);
        float4 yl3 = *(device const float4*)(y4 + 36);
        float4 yh0 = *(device const float4*)(y4 + 128);
        float4 yh1 = *(device const float4*)(y4 + 132);
        float4 yh2 = *(device const float4*)(y4 + 160);
        float4 yh3 = *(device const float4*)(y4 + 164);
        if (SWIGLU_X) {
            yl0 = sw4(yl0, *(device const float4*)(u4 + 0));
            yl1 = sw4(yl1, *(device const float4*)(u4 + 4));
            yl2 = sw4(yl2, *(device const float4*)(u4 + 32));
            yl3 = sw4(yl3, *(device const float4*)(u4 + 36));
            yh0 = sw4(yh0, *(device const float4*)(u4 + 128));
            yh1 = sw4(yh1, *(device const float4*)(u4 + 132));
            yh2 = sw4(yh2, *(device const float4*)(u4 + 160));
            yh3 = sw4(yh3, *(device const float4*)(u4 + 164));
        }
        *((thread float4*)(yl + 0)) = yl0;
        *((thread float4*)(yl + 4)) = yl1;
        *((thread float4*)(yl + 8)) = yl2;
        *((thread float4*)(yl + 12)) = yl3;
        *((thread float4*)(yh + 0)) = yh0;
        *((thread float4*)(yh + 4)) = yh1;
        *((thread float4*)(yh + 8)) = yh2;
        *((thread float4*)(yh + 12)) = yh3;
        float4 sumy = {
            yl0[0] + yl0[1] + yl0[2] + yl0[3] + yl1[0] + yl1[1] + yl1[2] + yl1[3],
            yl2[0] + yl2[1] + yl2[2] + yl2[3] + yl3[0] + yl3[1] + yl3[2] + yl3[3],
            yh0[0] + yh0[1] + yh0[2] + yh0[3] + yh1[0] + yh1[1] + yh1[2] + yh1[3],
            yh2[0] + yh2[1] + yh2[2] + yh2[3] + yh3[0] + yh3[1] + yh3[2] + yh3[3],
        };
        device const uchar* blk = row0 + ib * 144;
        device const ushort* sc = (device const ushort*)(blk + 4) + iq;
        device const ushort* q1 = (device const ushort*)(blk + 16) + 16 * iq + 4 * ir;
        device const uchar* dh = blk;

        for (uint row = 0; row < NR0; row++) {
            const ushort sc0 = sc[0];
            const ushort sc2 = sc[2];
            const ushort sc4 = sc[4];
            sc16[0] = sc0 & kmask1;
            sc16[1] = sc2 & kmask1;
            sc16[2] = ((sc4 >> 0) & kmask2) | ((sc0 & kmask3) >> 2);
            sc16[3] = ((sc4 >> 4) & kmask2) | ((sc2 & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;
            const ushort4 q1_v = ushort4(*(device const packed_ushort4*)q1);
            const ushort4 q2_v = ushort4(*(device const packed_ushort4*)q2);

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1_v[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1_v[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1_v[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1_v[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2_v[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2_v[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2_v[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2_v[i] & 0xF000);
            }

            half2 dm = *(device const half2*)dh;
            sumf[row] +=
                (float)dm.x * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * sc8[0] +
                               (acc1[2] + (1.0f / 256.0f) * acc1[3]) * sc8[1] * (1.0f / 16.0f) +
                               (acc2[0] + (1.0f / 256.0f) * acc2[1]) * sc8[4] +
                               (acc2[2] + (1.0f / 256.0f) * acc2[3]) * sc8[5] * (1.0f / 16.0f)) -
                (float)dm.y * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                               sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            ulong step = (row + 1 < nrows) ? nb01 : 0;
            q1 = (device const ushort*)((device const uchar*)q1 + step);
            sc = (device const ushort*)((device const uchar*)sc + step);
            dh += step;
        }
        y4 += 4 * 256;
        u4 += 4 * 256;
    }

    for (uint row = 0; row < NR0; row++) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && row < nrows) {
            yout[(ROWS ? tok * idx.y_row_stride : 0u) + y_base + row] = s;
        }
    }
}


// Port of llama.cpp's kernel_mul_mv_q5_K_f32 structure (MIT): NR0 rows per
// SIMD group via LPR (default NR0=2), activations registered once (yl/yh)
// and reused across rows, with the 5th bit folded via qh masks. INDEXED
// requires n_out % NR0 == 0. Used heavily for GLM shared gate/up (Q5_K).
kernel void matvec_q5_k_mv(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    device const float* xu [[buffer(8)]],
    device const uchar* w2 [[buffer(11)]],
    constant ulong& w2_off [[buffer(12)]],
    device float* y2_out [[buffer(13)]],
    uint3 tpg [[thread_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const ushort kmask1 = 0x3f3f;
    const ushort kmask2 = 0x0f0f;
    const ushort kmask3 = 0xc0c0;

    const uint NR0 = 32 / LPR;
    const bool id_grid = INDEXED && !ROWS && !DUAL_GW && MV_ID_GRID;
    uint tok = 0;
    uint slot = 0;
    uint j0 = 0;
    uint y_base = 0;
    uint nrows = 0;
    bool any;
    device const uchar* wm = w;
    ulong woff = w_off;
    device float* yout = y;
    if (id_grid) {
        slot = tpg.z;
        j0 = (tpg.x / 32) * NR0;
        any = slot < idx.slots && j0 < n_out;
        if (!any) {
            j0 = 0;
            slot = 0;
        }
        nrows = any ? min(NR0, n_out - j0) : 0u;
        y_base = slot * n_out + j0;
    } else {
        uint gate_rows = INDEXED ? n_out * idx.slots : n_out;
        uint ycols = DUAL_GW ? gate_rows * 2 : gate_rows;
        uint flat = (tpg.x / 32) * NR0;
        any = flat < (ROWS ? gate_rows * idx.n_rows : ycols);
        if (!any) flat = 0;
        tok = ROWS ? flat / gate_rows : 0;
        if (DUAL_GW && flat >= gate_rows) {
            flat -= gate_rows;
            wm = w2;
            woff = w2_off;
            yout = y2_out;
        }
        uint fr = ROWS ? flat - tok * gate_rows : flat;
        nrows = any ? min(NR0, gate_rows - fr) : 0u;
        slot = INDEXED ? fr / n_out : 0;
        j0 = INDEXED ? fr % n_out : fr;
        y_base = fr;
    }
    if (INDEXED) {
        wm += ids[(ROWS ? tok * idx.ids_stride : 0u) + slot] * idx.stride;
        x += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
        if (SWIGLU_X) {
            xu += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
        }
    }
    uint nb = n_in / 256;
    ulong nb01 = (ulong)nb * 176;
    device const uchar* row0 = wm + woff + (ulong)j0 * nb01;

    const short tid4 = tiisg / 4;
    const short ix = tiisg % 4;
    const short iq = tid4 / 4;
    const short ir = tid4 % 4;
    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;
    const uchar hm1 = (uchar)(1u << (2 * iq));
    const uchar hm2 = (uchar)(hm1 << 1);
    const uchar hm3 = (uchar)(hm1 << 4);
    const uchar hm4 = (uchar)(hm2 << 4);

    float yl[16];
    float yh[16];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    device const float* y1 = x + (ulong)ix * 256u + y_offset;
    device const float* u1 = xu + (ulong)ix * 256u + y_offset;

    for (uint ib = ix; ib < nb; ib += 4) {
        device const float* y2 = y1 + 128;
        device const float* u2 = u1 + 128;
        float4 yl0 = *(device const float4*)(y1 + 0);
        float4 yl1 = *(device const float4*)(y1 + 4);
        float4 yl2 = *(device const float4*)(y1 + 32);
        float4 yl3 = *(device const float4*)(y1 + 36);
        float4 yh0 = *(device const float4*)(y2 + 0);
        float4 yh1 = *(device const float4*)(y2 + 4);
        float4 yh2 = *(device const float4*)(y2 + 32);
        float4 yh3 = *(device const float4*)(y2 + 36);
        if (SWIGLU_X) {
            yl0 = sw4(yl0, *(device const float4*)(u1 + 0));
            yl1 = sw4(yl1, *(device const float4*)(u1 + 4));
            yl2 = sw4(yl2, *(device const float4*)(u1 + 32));
            yl3 = sw4(yl3, *(device const float4*)(u1 + 36));
            yh0 = sw4(yh0, *(device const float4*)(u2 + 0));
            yh1 = sw4(yh1, *(device const float4*)(u2 + 4));
            yh2 = sw4(yh2, *(device const float4*)(u2 + 32));
            yh3 = sw4(yh3, *(device const float4*)(u2 + 36));
        }
        *((thread float4*)(yl + 0)) = yl0;
        *((thread float4*)(yl + 4)) = yl1;
        *((thread float4*)(yl + 8)) = yl2;
        *((thread float4*)(yl + 12)) = yl3;
        *((thread float4*)(yh + 0)) = yh0;
        *((thread float4*)(yh + 4)) = yh1;
        *((thread float4*)(yh + 8)) = yh2;
        *((thread float4*)(yh + 12)) = yh3;
        float4 sumy = {
            yl0[0] + yl0[1] + yl0[2] + yl0[3] + yl1[0] + yl1[1] + yl1[2] + yl1[3],
            yl2[0] + yl2[1] + yl2[2] + yl2[3] + yl3[0] + yl3[1] + yl3[2] + yl3[3],
            yh0[0] + yh0[1] + yh0[2] + yh0[3] + yh1[0] + yh1[1] + yh1[2] + yh1[3],
            yh2[0] + yh2[1] + yh2[2] + yh2[3] + yh3[0] + yh3[1] + yh3[2] + yh3[3],
        };

        device const uchar* blk = row0 + (ulong)ib * 176u;
        device const ushort* a = (device const ushort*)(blk + 4) + iq;
        // block_q5_K: d(2) dmin(2) scales(12) qh(32) qs(128) = 176
        device const uchar* q1 = blk + 48 + q_offset;
        device const uchar* qh = blk + 16 + l0;
        device const uchar* dh = blk;

        for (uint row = 0; row < NR0; ++row) {
            device const uchar* q2 = q1 + 64;
            const ushort a0 = a[0];
            const ushort a2 = a[2];
            const ushort a4 = a[4];
            sc16[0] = a0 & kmask1;
            sc16[1] = a2 & kmask1;
            sc16[2] = ((a4 >> 0) & kmask2) | ((a0 & kmask3) >> 2);
            sc16[3] = ((a4 >> 4) & kmask2) | ((a2 & kmask3) >> 2);

            const ushort4 q1_w = *(device const packed_ushort4*)q1;
            const ushort4 q2_w = *(device const packed_ushort4*)q2;
            const ushort4 qh_w = *(device const packed_ushort4*)qh;
            thread const uchar* q1_v = (thread const uchar*)&q1_w;
            thread const uchar* q2_v = (thread const uchar*)&q2_w;
            thread const uchar* qh_v = (thread const uchar*)&qh_w;

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            _Pragma("clang loop unroll(full)")
            for (short l = 0; l < 8; ++l) {
                uchar h = qh_v[l];
                acc1[0] += yl[l + 0] * (float)(q1_v[l] & 0x0F);
                acc1[1] += yl[l + 8] * (float)(q1_v[l] & 0xF0);
                acc1[2] += yh[l + 0] * (float)(q2_v[l] & 0x0F);
                acc1[3] += yh[l + 8] * (float)(q2_v[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
                acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
                acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
                acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
            }

            half2 dm = *(device const half2*)dh;
            sumf[row] +=
                (float)dm.x * (sc8[0] * (acc1[0] + 16.0f * acc2[0]) +
                               sc8[1] * (acc1[1] / 16.0f + 16.0f * acc2[1]) +
                               sc8[4] * (acc1[2] + 16.0f * acc2[2]) +
                               sc8[5] * (acc1[3] / 16.0f + 16.0f * acc2[3])) -
                (float)dm.y * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                               sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            ulong step = (row + 1 < nrows) ? nb01 : 0;
            q1 += step;
            qh += step;
            a = (device const ushort*)((device const uchar*)a + step);
            dh += step;
        }
        y1 += 4 * 256;
        u1 += 4 * 256;
    }

    for (uint row = 0; row < NR0; row++) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && row < nrows) {
            yout[(ROWS ? tok * idx.y_row_stride : 0u) + y_base + row] = s;
        }
    }
}

kernel void matvec_q5_k(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint jt = tid / LPR;
    uint lane = tid % LPR;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    // ROWS: the grid covers every token row's slots; tok/jr split the work
    // item, per-row math below is untouched.
    uint tok = ROWS ? jt / ycols : 0;
    uint jr = ROWS ? jt - tok * ycols : jt;
    uint slot = INDEXED ? jr / n_out : 0;
    uint j = INDEXED ? jr % n_out : jr;
    // Rows past the end still run - they read row 0 and drop the result -
    // because an early return would take their lanes out of the shuffles
    // the live rows in this SIMD group depend on.
    bool active = INDEXED ? (ROWS ? jt < ycols * idx.n_rows : jr < ycols) : j < n_out;
    if (!active) j = 0;
    if (INDEXED) {
        w += ids[(ROWS ? tok * idx.ids_stride : 0u) + slot] * idx.stride;
        x += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
    }
    uint blocks = n_in / 256;
    device const uchar* row = w + w_off + (ulong)j * blocks * 176;
    float total[MAXT] = {0};
    for (uint b = lane; b < blocks; b += LPR) {
        device const uchar* blk = row + b * 176;
        float d = half_at(blk);
        float dmin = half_at(blk + 2);
        device const uchar* packed = blk + 4;
        device const uchar* qh = blk + 16;
        device const uchar* qs = blk + 48;
        float acc_d[MAXT] = {0};
        float acc_m[MAXT] = {0};
        for (uint pair = 0; pair < 4; pair++) {
            device const uchar4* q4 = (device const uchar4*)(qs + pair * 32);
            device const uchar4* h4 = (device const uchar4*)qh;
            uchar bit1 = 1 << (pair * 2);
            uchar bit2 = 2 << (pair * 2);
            uint x_lo = b * 256 + pair * 64;
            float sc1, mn1, sc2, mn2;
            scale_min_k4(pair * 2, packed, sc1, mn1);
            scale_min_k4(pair * 2 + 1, packed, sc2, mn2);
            float dot_lo[MAXT] = {0};
            float dot_hi[MAXT] = {0};
            float sum_lo[MAXT] = {0};
            float sum_hi[MAXT] = {0};
            for (uint l4 = 0; l4 < 8; l4++) {
                uchar4 qq = q4[l4];
                uchar4 hh = h4[l4];
                float4 q_lo = float4(int4(qq & (uchar4)0x0F)
                    + select(int4(0), int4(16), (hh & (uchar4)bit1) != (uchar4)0));
                float4 q_hi = float4(int4(qq >> 4)
                    + select(int4(0), int4(16), (hh & (uchar4)bit2) != (uchar4)0));
                for (uint i = 0; i < TILE; i++) {
                    device const float4* xi =
                        (device const float4*)(x + i * n_in + x_lo);
                    float4 xl = xi[l4];
                    float4 xh = xi[8 + l4];
                    dot_lo[i] += dot(q_lo, xl);
                    dot_hi[i] += dot(q_hi, xh);
                    sum_lo[i] += xl.x + xl.y + xl.z + xl.w;
                    sum_hi[i] += xh.x + xh.y + xh.z + xh.w;
                }
            }
            for (uint i = 0; i < TILE; i++) {
                acc_d[i] += sc1 * dot_lo[i] + sc2 * dot_hi[i];
                acc_m[i] += mn1 * sum_lo[i] + mn2 * sum_hi[i];
            }
        }
        for (uint i = 0; i < TILE; i++) {
            total[i] += d * acc_d[i] - dmin * acc_m[i];
        }
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[(ROWS ? tok * idx.y_row_stride : i * ycols) + (ROWS ? jr : jt)] = s;
    }
}

// ---- Whole-token decode support ----------------------------------------
//
// These four kernels close the gaps that used to force a CPU round trip
// between the attention half and the FFN half of every decode layer: the
// residual/norm seam, the (tiny, F32) router matmul, expert selection, and
// the weighted combine of expert outputs. With them, a whole decode layer -
// and therefore a whole token - encodes as one command buffer, and the CPU
// touches nothing until the logits.

// One threadgroup of 256 threads; hidden up to 16k. x += delta in place,
// then h = rmsnorm(x) * w. The norm weight is a bind (16 KB > setBytes cap).
// Batch RMSNorm over m rows for the fused prefill: row r of x (optionally
// plus delta, folding the residual add into the same pass) normalises into
// row r of h. One threadgroup per row, same reduction as residual_norm.
kernel void norm_rows(
    device float* x [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float* h [[buffer(2)]],
    device const float* w [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& use_delta [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[8];
    device float* xr = x + (ulong)row * n;
    device const float* dr = delta + (ulong)row * n;
    device float* hr = h + (ulong)row * n;
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        float v = xr[i];
        if (use_delta != 0) {
            v += dr[i];
            xr[i] = v;
        }
        acc += v * v;
    }
    float s = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; i++) {
        total += partial[i];
    }
    float scale = rsqrt(total / (float)n + eps);
    for (uint i = tid; i < n; i += 256) {
        hr[i] = xr[i] * scale * w[i];
    }
}

// The MoE combine for the fused prefill: token t's residual row gains the
// weighted sum of its expert down-rows. One threadgroup per token; the hit
// list is CSR-shaped (tok_off) so each row has exactly one writer.
kernel void combine_rows(
    device float* x [[buffer(0)]],
    device const float* flat [[buffer(1)]],
    device const uint* tok_off [[buffer(2)]],
    device const uint* hit_row [[buffer(3)]],
    device const float* hit_w [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint h0 = tok_off[token];
    uint h1 = tok_off[token + 1];
    device float* xr = x + (ulong)token * hidden;
    for (uint i = tid; i < hidden; i += 256) {
        float acc = xr[i];
        for (uint h = h0; h < h1; h++) {
            acc += hit_w[h] * flat[(ulong)hit_row[h] * hidden + i];
        }
        xr[i] = acc;
    }
}

// residual_norm plus the epoch publish for the spin-flag scheme.
kernel void residual_norm_sig(
    device float* x [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float* h [[buffer(2)]],
    device const float* w [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    device atomic_uint* flag [[buffer(6)]],
    constant uint& epoch [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[8];
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        float v = x[i] + delta[i];
        x[i] = v;
        acc += v * v;
    }
    float s = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; i++) {
        total += partial[i];
    }
    float scale = rsqrt(total / (float)n + eps);
    for (uint i = tid; i < n; i += 256) {
        h[i] = x[i] * scale * w[i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        DEVICE_FENCE();
        atomic_store_explicit(flag, epoch, memory_order_relaxed);
    }
}

kernel void residual_norm(
    device float* x [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float* h [[buffer(2)]],
    device const float* w [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[8];
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        float v = x[i] + delta[i];
        x[i] = v;
        acc += v * v;
    }
    float s = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; i++) {
        total += partial[i];
    }
    float scale = rsqrt(total / (float)n + eps);
    for (uint i = tid; i < n; i += 256) {
        h[i] = x[i] * scale * w[i];
    }
}

// Plain f32 matvec for the router: LPR lanes per output row, float4 strides.
kernel void matvec_f32(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device atomic_uint* wait_flag [[buffer(9)]],
    constant uint& wait_epoch [[buffer(10)]],
    uint tid [[thread_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (WAIT_X) {
        if (ltid == 0) {
            while (atomic_load_explicit(wait_flag, memory_order_relaxed) < wait_epoch) {
            }
            DEVICE_FENCE();
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    uint j = tid / LPR;
    uint lane = tid % LPR;
    bool active = j < n_out;
    if (!active) j = 0;
    device const float4* row =
        (device const float4*)(w + w_off + (ulong)j * n_in * 4);
    float total[MAXT] = {0};
    for (uint b = lane; b < n_in / 4; b += LPR) {
        for (uint i = 0; i < TILE; i++) {
            total[i] += dot(row[b], ((device const float4*)(x + i * n_in))[b]);
        }
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[i * n_out + j] = s;
    }
}

// Top-k of n expert logits plus renormalised softmax weights, one SIMD
// group. Repeated masked argmax: k iterations of a simd max+owner pick.
// The weights renormalise over the selected k, which equals the CPU path's
// softmax-then-renormalise exactly (the common partition function cancels).
// Router matvec + softmax + top-k in ONE kernel: the barrier-and-tiny-
// dispatch pattern between them was measured at ~8 ms/token across the
// whole decode chain, and this removes one drain per layer. One 256-thread
// threadgroup: thread e computes expert e's logit (an f32 dot over hidden),
// then the first simdgroup runs the same top-k as softmax_topk over
// threadgroup memory.
// GPU-side MoE routing for the fused prefill: top-k selection, per-expert
// counting, prefix sums and the CSR scatter, replacing the per-layer CPU
// readback. Semantics mirror route_gated (bias shifts ONLY the selection,
// optional winner renormalisation, stable low-index ties) for both gating
// rules: sigmoid scores (GLM) or softmax over all experts (Qwen3). The
// softmax case selects by the raw logit (softmax is monotonic per row, and
// the host only routes softmax here when no selection bias is present) and
// weights the picks by exp(l - max) renormalised over them, which equals
// the CPU softmax-then-renormalise exactly (the partition function cancels).
struct MmGroup {
    uint expert;
    uint row0;
    uint rows;
};
struct RouteParams {
    uint m;
    uint n_expert;
    uint n_used;
    uint norm;
    uint has_bias;
    uint sigmoid;
    uint shared;
    float scale;
    uint total_rows;
};

kernel void route_pick(
    device const float* logits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],
    device uint* picks [[buffer(3)]],
    device float* pickw [[buffer(4)]],
    constant RouteParams& p [[buffer(5)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint token = tgp * 256 + tid;
    if (token >= p.m) {
        return;
    }
    device const float* row = logits + (ulong)token * p.n_expert;
    float bs[8];
    uint be[8];
    for (uint i = 0; i < 8; i++) {
        bs[i] = -INFINITY;
        be[i] = 0xFFFFFFFF;
    }
    float full_mx = -INFINITY;
    float full_denom = 1.0f;
    if (!p.sigmoid && (!p.norm || p.has_bias)) {
        for (uint e = 0; e < p.n_expert; e++) {
            full_mx = max(full_mx, row[e]);
        }
        float sum_exp = 0.0f;
        for (uint e = 0; e < p.n_expert; e++) {
            sum_exp += exp(row[e] - full_mx);
        }
        full_denom = sum_exp > 0.0f ? sum_exp : 1.0f;
    }
    for (uint e = 0; e < p.n_expert; e++) {
        // Sigmoid gating scores each expert independently; softmax selects
        // by the raw logit when unbiased, or by softmax probability + bias
        // when selection bias is present.
        float s = p.sigmoid
            ? 1.0f / (1.0f + exp(-row[e]))
            : (p.has_bias ? exp(row[e] - full_mx) / full_denom : row[e]);
        float sel = p.has_bias ? s + bias[e] : s;
        if (sel < bs[7] || (sel == bs[7] && e >= be[7])) {
            continue;
        }
        uint j = 7;
        while (j > 0 && (sel > bs[j - 1] || (sel == bs[j - 1] && e < be[j - 1]))) {
            bs[j] = bs[j - 1];
            be[j] = be[j - 1];
            j--;
        }
        bs[j] = sel;
        be[j] = e;
    }
    float tot = 0.0f;
    float sc[8];
    float mx = -INFINITY;
    if (!p.sigmoid && p.norm) {
        for (uint i = 0; i < p.n_used; i++) {
            mx = max(mx, row[be[i]]);
        }
    }
    for (uint i = 0; i < p.n_used; i++) {
        sc[i] = p.sigmoid
            ? 1.0f / (1.0f + exp(-row[be[i]]))
            : (p.norm ? exp(row[be[i]] - mx) : exp(row[be[i]] - full_mx) / full_denom);
        tot += sc[i];
    }
    for (uint i = 0; i < p.n_used; i++) {
        float w = p.norm ? (tot > 0.0f ? sc[i] / tot : 1.0f / p.n_used) : sc[i];
        picks[token * p.n_used + i] = be[i];
        pickw[token * p.n_used + i] = w * p.scale;
        atomic_fetch_add_explicit(&counts[be[i]], 1, memory_order_relaxed);
    }
}

kernel void route_scan(
    device uint* counts [[buffer(0)]],
    device MmGroup* table [[buffer(1)]],
    device uint* counters [[buffer(2)]],
    constant uint& n_expert [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid == 0) {
        uint acc = 0;
        for (uint e = 0; e < n_expert; e++) {
            table[e].expert = e;
            table[e].row0 = acc;
            table[e].rows = counts[e];
            acc += counts[e];
            counts[e] = 0;
            counters[e] = 0;
        }
    }
}

kernel void route_scatter(
    device const uint* picks [[buffer(0)]],
    device const float* pickw [[buffer(1)]],
    device const MmGroup* table [[buffer(2)]],
    device atomic_uint* counters [[buffer(3)]],
    device uint* tok [[buffer(4)]],
    device uint* hit_row [[buffer(5)]],
    device float* hit_w [[buffer(6)]],
    device uint* tok_off [[buffer(7)]],
    constant RouteParams& p [[buffer(8)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint token = tgp * 256 + tid;
    if (token >= p.m) {
        return;
    }
    uint stride = p.n_used + p.shared;
    for (uint i = 0; i < p.n_used; i++) {
        uint e = picks[token * p.n_used + i];
        uint pos = atomic_fetch_add_explicit(&counters[e], 1, memory_order_relaxed);
        uint row = table[e].row0 + pos;
        tok[row] = token;
        hit_row[token * stride + i] = row;
        hit_w[token * stride + i] = pickw[token * p.n_used + i];
    }
    if (p.shared) {
        hit_row[token * stride + p.n_used] = p.total_rows + token;
        hit_w[token * stride + p.n_used] = 1.0f;
    }
    tok_off[token] = token * stride;
    if (token == 0) {
        tok_off[p.m] = p.m * stride;
    }
}

kernel void router_topk(
    device const float* w [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device uint* ids [[buffer(2)]],
    device float* wts [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant ulong& w_off [[buffer(7)]],
    device const float* rbias [[buffer(8)]],
    constant uint& sigmoid [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float logits[256];
    device const float* wr = (device const float*)((device const uchar*)w + w_off);
    // 8 simdgroups x 32 lanes: each simdgroup owns experts sg, sg+8, ... and
    // its 32 lanes split the dot; n <= 256 by the host-side router check.
    for (uint e = sg; e < n; e += 8) {
        device const float* row = wr + (ulong)e * hidden;
        float acc = 0.0f;
        for (uint i = lane * 4; i + 3 < hidden; i += 128) {
            float4 a = *(device const float4*)(row + i);
            float4 b = *(device const float4*)(h + i);
            acc += dot(a, b);
        }
        float s = simd_sum(acc);
        if (lane == 0) {
            logits[e] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) {
        return;
    }
    float v[8];
    uint vid[8];
    for (uint t = 0; t < 8; t++) {
        uint idx = lane + t * 32;
        if (idx < n) {
            // Sigmoid gating selects by sigmoid(l) + rbias (llama.cpp
            // `selection_probs`); softmax selects by the raw logit.
            v[t] = sigmoid != 0
                ? 1.0f / (1.0f + exp(-logits[idx])) +
                      (has_bias != 0 ? rbias[idx] : 0.0f)
                : logits[idx];
        } else {
            v[t] = -INFINITY;
        }
        vid[t] = idx;
    }
    for (uint kk = 0; kk < k; kk++) {
        float lm = -INFINITY;
        uint li = 0xFFFFFFFF;
        for (uint t = 0; t < 8; t++) {
            if (v[t] > lm) {
                lm = v[t];
                li = vid[t];
            }
        }
        float m = simd_max(lm);
        uint who = simd_min(lm == m ? li : 0xFFFFFFFF);
        if (lane == 0) {
            ids[kk] = who;
        }
        for (uint t = 0; t < 8; t++) {
            if (vid[t] == who) {
                v[t] = -INFINITY;
            }
        }
    }
    if (lane == 0) {
        if (sigmoid != 0) {
            // Weights are the UNBIASED sigmoid renormalised over the picks.
            float total = 0.0f;
            for (uint kk = 0; kk < k; kk++) {
                float s = 1.0f / (1.0f + exp(-logits[ids[kk]]));
                wts[kk] = s;
                total += s;
            }
            for (uint kk = 0; kk < k; kk++) {
                wts[kk] /= total;
            }
        } else {
            float mx = -INFINITY;
            for (uint kk = 0; kk < k; kk++) {
                mx = max(mx, logits[ids[kk]]);
            }
            float total = 0.0f;
            for (uint kk = 0; kk < k; kk++) {
                float e = exp(logits[ids[kk]] - mx);
                wts[kk] = e;
                total += e;
            }
            for (uint kk = 0; kk < k; kk++) {
                wts[kk] /= total;
            }
        }
    }
}

// residual_norm + router matvec + gating top-k in ONE dispatch: both halves
// are single-threadgroup kernels and the router reads the h the norm just
// wrote, so the boundary between them was pure launch and drain latency.
// Stage 1 is the residual_norm body; stage 2 the router_topk body.
kernel void resnorm_router(
    device float* x [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float* h [[buffer(2)]],
    device const float* normw [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    device const uchar* rw [[buffer(6)]],
    constant ulong& rw_off [[buffer(7)]],
    device uint* ids [[buffer(8)]],
    device float* wts [[buffer(9)]],
    constant uint& n [[buffer(10)]],
    constant uint& k [[buffer(11)]],
    device const float* rbias [[buffer(12)]],
    constant uint& sigmoid [[buffer(13)]],
    constant uint& has_bias [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[8];
    float acc = 0.0f;
    for (uint i = tid; i < hidden; i += 256) {
        float v = x[i] + delta[i];
        x[i] = v;
        acc += v * v;
    }
    float s = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; i++) {
        total += partial[i];
    }
    float scale = rsqrt(total / (float)hidden + eps);
    for (uint i = tid; i < hidden; i += 256) {
        h[i] = x[i] * scale * normw[i];
    }
    // h is device memory written by this same threadgroup.
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup float logits[256];
    device const float* wr = (device const float*)(rw + rw_off);
    // 8 simdgroups x 32 lanes: each simdgroup owns experts sg, sg+8, ... and
    // its 32 lanes split the dot; n <= 256 by the host-side router check.
    for (uint e = sg; e < n; e += 8) {
        device const float* row = wr + (ulong)e * hidden;
        float acc = 0.0f;
        for (uint i = lane * 4; i + 3 < hidden; i += 128) {
            float4 a = *(device const float4*)(row + i);
            float4 b = *(device const float4*)(h + i);
            acc += dot(a, b);
        }
        float s = simd_sum(acc);
        if (lane == 0) {
            logits[e] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) {
        return;
    }
    float v[8];
    uint vid[8];
    for (uint t = 0; t < 8; t++) {
        uint idx = lane + t * 32;
        if (idx < n) {
            // Sigmoid gating selects by sigmoid(l) + rbias (llama.cpp
            // `selection_probs`); softmax selects by the raw logit.
            v[t] = sigmoid != 0
                ? 1.0f / (1.0f + exp(-logits[idx])) +
                      (has_bias != 0 ? rbias[idx] : 0.0f)
                : logits[idx];
        } else {
            v[t] = -INFINITY;
        }
        vid[t] = idx;
    }
    for (uint kk = 0; kk < k; kk++) {
        float lm = -INFINITY;
        uint li = 0xFFFFFFFF;
        for (uint t = 0; t < 8; t++) {
            if (v[t] > lm) {
                lm = v[t];
                li = vid[t];
            }
        }
        float m = simd_max(lm);
        uint who = simd_min(lm == m ? li : 0xFFFFFFFF);
        if (lane == 0) {
            ids[kk] = who;
        }
        for (uint t = 0; t < 8; t++) {
            if (vid[t] == who) {
                v[t] = -INFINITY;
            }
        }
    }
    if (lane == 0) {
        if (sigmoid != 0) {
            // Weights are the UNBIASED sigmoid renormalised over the picks.
            float total = 0.0f;
            for (uint kk = 0; kk < k; kk++) {
                float s = 1.0f / (1.0f + exp(-logits[ids[kk]]));
                wts[kk] = s;
                total += s;
            }
            for (uint kk = 0; kk < k; kk++) {
                wts[kk] /= total;
            }
        } else {
            float mx = -INFINITY;
            for (uint kk = 0; kk < k; kk++) {
                mx = max(mx, logits[ids[kk]]);
            }
            float total = 0.0f;
            for (uint kk = 0; kk < k; kk++) {
                float e = exp(logits[ids[kk]] - mx);
                wts[kk] = e;
                total += e;
            }
            for (uint kk = 0; kk < k; kk++) {
                wts[kk] /= total;
            }
        }
    }
}

// resnorm_router over m token rows in ONE dispatch: threadgroup y owns the
// row, the per-row math (reduction order included) is resnorm_router's
// verbatim, so every row is bit-identical to the single-row kernel. Used by
// the batched MTP verify; decode keeps the single-row original.
kernel void resnorm_router_rows(
    device float* x [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float* h [[buffer(2)]],
    device const float* normw [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    device const uchar* rw [[buffer(6)]],
    constant ulong& rw_off [[buffer(7)]],
    device uint* ids [[buffer(8)]],
    device float* wts [[buffer(9)]],
    constant uint& n [[buffer(10)]],
    constant uint& k [[buffer(11)]],
    device const float* rbias [[buffer(12)]],
    constant uint& sigmoid [[buffer(13)]],
    constant uint& has_bias [[buffer(14)]],
    constant uint& wstride [[buffer(15)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    x += (ulong)tgp.y * hidden;
    delta += (ulong)tgp.y * hidden;
    h += (ulong)tgp.y * hidden;
    ids += (ulong)tgp.y * wstride;
    wts += (ulong)tgp.y * wstride;
    threadgroup float partial[8];
    float acc = 0.0f;
    for (uint i = tid; i < hidden; i += 256) {
        float v = x[i] + delta[i];
        x[i] = v;
        acc += v * v;
    }
    float s = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; i++) {
        total += partial[i];
    }
    float scale = rsqrt(total / (float)hidden + eps);
    for (uint i = tid; i < hidden; i += 256) {
        h[i] = x[i] * scale * normw[i];
    }
    // h is device memory written by this same threadgroup.
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup float logits[256];
    device const float* wr = (device const float*)(rw + rw_off);
    // 8 simdgroups x 32 lanes: each simdgroup owns experts sg, sg+8, ... and
    // its 32 lanes split the dot; n <= 256 by the host-side router check.
    for (uint e = sg; e < n; e += 8) {
        device const float* row = wr + (ulong)e * hidden;
        float acc = 0.0f;
        for (uint i = lane * 4; i + 3 < hidden; i += 128) {
            float4 a = *(device const float4*)(row + i);
            float4 b = *(device const float4*)(h + i);
            acc += dot(a, b);
        }
        float s = simd_sum(acc);
        if (lane == 0) {
            logits[e] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) {
        return;
    }
    float v[8];
    uint vid[8];
    for (uint t = 0; t < 8; t++) {
        uint idx = lane + t * 32;
        if (idx < n) {
            // Sigmoid gating selects by sigmoid(l) + rbias (llama.cpp
            // `selection_probs`); softmax selects by the raw logit.
            v[t] = sigmoid != 0
                ? 1.0f / (1.0f + exp(-logits[idx])) +
                      (has_bias != 0 ? rbias[idx] : 0.0f)
                : logits[idx];
        } else {
            v[t] = -INFINITY;
        }
        vid[t] = idx;
    }
    for (uint kk = 0; kk < k; kk++) {
        float lm = -INFINITY;
        uint li = 0xFFFFFFFF;
        for (uint t = 0; t < 8; t++) {
            if (v[t] > lm) {
                lm = v[t];
                li = vid[t];
            }
        }
        float m = simd_max(lm);
        uint who = simd_min(lm == m ? li : 0xFFFFFFFF);
        if (lane == 0) {
            ids[kk] = who;
        }
        for (uint t = 0; t < 8; t++) {
            if (vid[t] == who) {
                v[t] = -INFINITY;
            }
        }
    }
    if (lane == 0) {
        if (sigmoid != 0) {
            // Weights are the UNBIASED sigmoid renormalised over the picks.
            float total = 0.0f;
            for (uint kk = 0; kk < k; kk++) {
                float s = 1.0f / (1.0f + exp(-logits[ids[kk]]));
                wts[kk] = s;
                total += s;
            }
            for (uint kk = 0; kk < k; kk++) {
                wts[kk] /= total;
            }
        } else {
            float mx = -INFINITY;
            for (uint kk = 0; kk < k; kk++) {
                mx = max(mx, logits[ids[kk]]);
            }
            float total = 0.0f;
            for (uint kk = 0; kk < k; kk++) {
                float e = exp(logits[ids[kk]] - mx);
                wts[kk] = e;
                total += e;
            }
            for (uint kk = 0; kk < k; kk++) {
                wts[kk] /= total;
            }
        }
    }
}

// moe_combine over m token rows in ONE dispatch: threadgroup y owns the row,
// the per-row math is moe_combine's verbatim (experts in slot order, the
// shared expert LAST). The expert down block is per-row [slots][n] and wts
// per-row [wstride]; the shared expert rides in its own contiguous regions
// (TILE matvecs wrote all rows in one pass): sgate [m] raw logits,
// sdown [m][n] outputs. sig_last: 0 = no shared, 1 = sigmoid(sgate),
// 2 = weight 1.0 (shared expert without a gate).
kernel void moe_combine_rows(
    device const float* down [[buffer(0)]],
    device const float* wts [[buffer(1)]],
    device float* delta [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& slots [[buffer(4)]],
    constant uint& sig_last [[buffer(5)]],
    constant uint& wstride [[buffer(6)]],
    device const float* sgate [[buffer(7)]],
    device const float* sdown [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint i = tgp.x * 256 + tid;
    if (i >= n) return;
    const uint row = tgp.y;
    device const float* wr = wts + (ulong)row * wstride;
    // The per-row down blocks are wstride slots wide (the shared expert's
    // slot stays unused - it lives in the sdown region), of which the first
    // `slots` hold the routed experts.
    device const float* dr = down + (ulong)row * wstride * n;
    float acc = 0.0f;
    for (uint s = 0; s < slots; s++) {
        acc += wr[s] * dr[s * n + i];
    }
    if (sig_last == 1) {
        const float g = sgate[row];
        acc += (1.0f / (1.0f + exp(-g))) * sdown[(ulong)row * n + i];
    } else if (sig_last == 2) {
        acc += sdown[(ulong)row * n + i];
    }
    delta[(ulong)row * n + i] = acc;
}

// ---- The FFN megakernel -------------------------------------------------
//
// The whole routed-FFN half of a decode layer as ONE dispatch: resnorm ->
// router -> top-k -> gate/up -> swiglu -> down -> combine. Between stages a
// device atomic counter plus the PROVEN two-sided seq_cst fences replace
// the encoder barriers, cutting each ~7 us stage boundary to ~1-2 us. The
// stage-boundary latency was measured (barriers, spin-flags and a serial
// encoder all cost the same ~8 ms/token), so only this fusion reclaims it.
//
// Grid: args.n_tg threadgroups x 256 threads, persistent across stages;
// each stage self-partitions over global simdgroups or threads. Formats
// come in via function constants (dead branches eliminated): 0 = q2_k
// word-load core, 1 = q3_k llama-structure core.
constant uint GATE_FMT [[function_constant(10)]];
constant uint DOWN_FMT [[function_constant(11)]];
constant uint SH_GATE_FMT [[function_constant(12)]];
constant uint SH_DOWN_FMT [[function_constant(13)]];

// Format ids: 0 = q2_k, 1 = q3_k, 2 = q4_k, 3 = q5_0, 4 = q5_k, 5 = q8_0.

struct MegaArgs {
    uint hidden;
    uint ffn;
    uint n_expert;
    uint n_used;
    uint x_at;
    uint delta_at;
    uint h_at;
    uint logits_at;
    uint ids_at;
    uint wts_at;
    uint gate_at;
    uint up_at;
    uint downo_at;
    uint ctr_at;
    uint n_tg;
    uint ctr_base;
    float eps;
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong router_off;
    ulong gate_stride;
    ulong up_stride;
    ulong down_stride;
    // GLM: sigmoid gating with a selection bias (buffer 7), and an
    // always-on shared expert (buffers 8-10) riding slot n_used.
    uint sigmoid;
    // 0 = none, 1 = always-on weight 1, 2 = gated via buffer 11 (F32[hidden]
    // projection -> raw logit in wts[n_used], sigmoid applied at combine).
    uint has_shared;
    ulong sh_gate_off;
    ulong sh_up_off;
    ulong sh_down_off;
    ulong sh_gout_off;
};

// Cross-stage sync for the megakernel. Device atomic busy-wait across
// threadgroups soft-locks the Metal scheduler and can freeze the Mac; the
// spin path is deleted. Host hard-caps mega_tg to 1, so a local barrier is
// enough. Multi-TG MEGA needs a redesign (encoder barriers / no spin).
inline void mega_sync(device atomic_uint* ctr, uint target, uint ltid, uint n_tg) {
    (void)ctr;
    (void)target;
    (void)ltid;
    (void)n_tg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float lane_sum_dyn(float v, uint lpr) {
    for (uint off = lpr / 2; off > 0; off >>= 1) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

// One lane's share of a q2_k row dot (word loads, the format's winning
// structure); the caller folds over its lpr lane group.
inline float q2k_row_partial(
    device const uchar* row, device const float* x, uint blocks, uint lane, uint lpr)
{
    float total = 0.0f;
    for (uint b = lane; b < blocks; b += lpr) {
        device const uchar* blk = row + b * 84;
        float d = half_at(blk + 80);
        float dmin = half_at(blk + 82);
        device const uint* sw = (device const uint*)blk;
        device const uint* qw = (device const uint*)(blk + 16);
        uint sc_w[4] = { sw[0], sw[1], sw[2], sw[3] };
        for (uint half_i = 0; half_i < 2; half_i++) {
            for (uint group = 0; group < 2; group++) {
                uint q0 = qw[half_i * 8 + group * 4 + 0];
                uint q1 = qw[half_i * 8 + group * 4 + 1];
                uint q2 = qw[half_i * 8 + group * 4 + 2];
                uint q3 = qw[half_i * 8 + group * 4 + 3];
                for (uint sh2 = 0; sh2 < 4; sh2++) {
                    uint shift = sh2 * 2;
                    uint is = half_i * 8 + sh2 * 2 + group;
                    uint sc = (sc_w[is / 4] >> ((is % 4) * 8)) & 0xFF;
                    float dl = d * (float)(sc & 0xF);
                    float ml = dmin * (float)(sc >> 4);
                    float4 qv0 = float4(as_type<uchar4>((q0 >> shift) & 0x03030303u));
                    float4 qv1 = float4(as_type<uchar4>((q1 >> shift) & 0x03030303u));
                    float4 qv2 = float4(as_type<uchar4>((q2 >> shift) & 0x03030303u));
                    float4 qv3 = float4(as_type<uchar4>((q3 >> shift) & 0x03030303u));
                    uint x0 = b * 256 + half_i * 128 + sh2 * 32 + group * 16;
                    device const float4* xi = (device const float4*)(x + x0);
                    float4 x0v = xi[0];
                    float4 x1v = xi[1];
                    float4 x2v = xi[2];
                    float4 x3v = xi[3];
                    float dotq = dot(qv0, x0v) + dot(qv1, x1v)
                               + dot(qv2, x2v) + dot(qv3, x3v);
                    float4 xs = x0v + x1v + x2v + x3v;
                    total += dl * dotq - ml * (xs.x + xs.y + xs.z + xs.w);
                }
            }
        }
    }
    return total;
}

// Two q3_k rows per full simdgroup (the llama-structure core); writes the
// finished sums for `nrows` rows via lane 0.
inline void q3k_two_rows(
    device const uchar* row0,
    device const float* x,
    uint nb,
    ulong nb01,
    uint nrows,
    ushort tiisg,
    device float* out,
    uint out_stride)
{
    const short t = tiisg / 4;
    const short ix = tiisg % 4;
    const short ip = t / 4;
    const short il = 2 * ((t % 4) / 2);
    const short ir = t % 2;
    const short l0 = 8 * ir;
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00},
                        {0x0030, 0x3000, 0x00c0, 0xc000}};
    const ushort4 hm = mm[2 * ip + il / 2];
    const short shift = 2 * il;
    const float v1 = il == 0 ? 4.0f : 64.0f;
    const float v2 = 4.0f * v1;
    const ushort s_shift1 = 4 * ip;
    const ushort s_shift2 = s_shift1 + il;
    const short q_offset = 32 * ip + l0;
    const short y_offset = 128 * ip + 32 * il + l0;
    device const float* y1 = x + ix * 256 + y_offset;
    float yl[32];
    uint scales32, aux32;
    thread ushort* scales16 = (thread ushort*)&scales32;
    thread const char* scales = (thread const char*)&scales32;
    float sumf1[2] = {0.0f, 0.0f};
    float sumf2[2] = {0.0f, 0.0f};
    for (uint ib = ix; ib < nb; ib += 4) {
        #pragma clang loop unroll(full)
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];
            yl[l + 8] = y1[l + 16];
            yl[l + 16] = y1[l + 32];
            yl[l + 24] = y1[l + 48];
        }
        device const uchar* blk = row0 + ib * 110;
        device const ushort* q = (device const ushort*)(blk + 32 + q_offset);
        device const ushort* h = (device const ushort*)(blk + l0);
        device const ushort* a = (device const ushort*)(blk + 96);
        device const uchar* dh = blk + 108;
        #pragma clang loop unroll(full)
        for (short row = 0; row < 2; ++row) {
            const float d_all = half_at(dh);
            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il + 0];
            scales16[1] = a[il + 1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;
            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            const ushort4 qv0 = *(device const packed_ushort4*)q;
            const ushort4 hv0 = *(device const packed_ushort4*)h;
            #pragma clang loop unroll(full)
            for (short l = 0; l < 8; l += 2) {
                const int qsv = qv0[l / 2];
                const ushort hsv = hv0[l / 2];
                s1 += yl[l + 0] * (qsv & qm[il / 2][0]);
                s2 += yl[l + 1] * (qsv & qm[il / 2][1]);
                s3 += ((hsv & hm[0]) ? 0.0f : yl[l + 0]) + ((hsv & hm[1]) ? 0.0f : yl[l + 1]);
                s4 += yl[l + 16] * (qsv & qm[il / 2][2]);
                s5 += yl[l + 17] * (qsv & qm[il / 2][3]);
                s6 += ((hsv & hm[2]) ? 0.0f : yl[l + 16]) + ((hsv & hm[3]) ? 0.0f : yl[l + 17]);
            }
            float d1 = d_all * (s1 + (1.0f / 256.0f) * s2 - s3 * v1);
            float d2 = d_all * (s4 + (1.0f / 256.0f) * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);
            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            const ushort4 qv1 = *(device const packed_ushort4*)(q + 8);
            const ushort4 hv1 = *(device const packed_ushort4*)(h + 8);
            #pragma clang loop unroll(full)
            for (short l = 0; l < 8; l += 2) {
                const int qsv = qv1[l / 2];
                const ushort hsv = hv1[l / 2];
                s1 += yl[l + 8] * (qsv & qm[il / 2][0]);
                s2 += yl[l + 9] * (qsv & qm[il / 2][1]);
                s3 += ((hsv & hm[0]) ? 0.0f : yl[l + 8]) + ((hsv & hm[1]) ? 0.0f : yl[l + 9]);
                s4 += yl[l + 24] * (qsv & qm[il / 2][2]);
                s5 += yl[l + 25] * (qsv & qm[il / 2][3]);
                s6 += ((hsv & hm[2]) ? 0.0f : yl[l + 24]) + ((hsv & hm[3]) ? 0.0f : yl[l + 25]);
            }
            d1 = d_all * (s1 + (1.0f / 256.0f) * s2 - s3 * v1);
            d2 = d_all * (s4 + (1.0f / 256.0f) * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);
            ulong step = (row + 1 < nrows) ? nb01 : 0;
            q = (device const ushort*)((device const uchar*)q + step);
            h = (device const ushort*)((device const uchar*)h + step);
            a = (device const ushort*)((device const uchar*)a + step);
            dh += step;
        }
        y1 += 4 * 256;
    }
    for (uint row = 0; row < 2; row++) {
        float s = simd_sum((sumf1[row] + 0.25f * sumf2[row]) / (1 << shift));
        if (tiisg == 0 && row < nrows) {
            out[row * out_stride] = s;
        }
    }
}

// llama.cpp's get_scale_min_k4 for one 6-bit scale/min pair, i in 0..8.
inline uchar2 q4k_sc_min(uint i, device const uchar* q) {
    if (i < 4) {
        return uchar2{uchar(q[i] & 63), uchar(q[i + 4] & 63)};
    }
    return uchar2{uchar((q[i + 4] & 0xF) | ((q[i - 4] & 0xc0) >> 2)),
                  uchar((q[i + 4] >> 4) | ((q[i] & 0xc0) >> 2))};
}

// Q4_K: 144-byte blocks of 256; a lane takes whole 64-element pairs so the
// nibble unpack and the activation sum stay in the same registers.
inline float q4k_row_partial(
    device const uchar* row, device const float* x, uint nb, uint lane, uint lpr)
{
    float total = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* blk = row + b * 144;
        const float d = half_at(blk);
        const float dmin = half_at(blk + 2);
        device const uchar* packed = blk + 4;
        device const uchar* qs = blk + 16;
        device const float* xb = x + b * 256;
        for (uint pair = lane; pair < 4; pair += lpr) {
            uchar2 s1 = q4k_sc_min(pair * 2, packed);
            uchar2 s2 = q4k_sc_min(pair * 2 + 1, packed);
            device const uchar* q = qs + pair * 32;
            float dot_lo = 0.0f, sum_lo = 0.0f, dot_hi = 0.0f, sum_hi = 0.0f;
            for (uint l = 0; l < 32; l++) {
                float xl = xb[pair * 64 + l];
                float xh = xb[pair * 64 + 32 + l];
                dot_lo += float(q[l] & 0xF) * xl;
                sum_lo += xl;
                dot_hi += float(q[l] >> 4) * xh;
                sum_hi += xh;
            }
            total += d * s1.x * dot_lo - dmin * s1.y * sum_lo
                   + d * s2.x * dot_hi - dmin * s2.y * sum_hi;
        }
    }
    return total;
}

// Q5_0: 22-byte blocks of 32; a lane takes whole blocks.
inline float q50_row_partial(
    device const uchar* row, device const float* x, uint nb, uint lane, uint lpr)
{
    float total = 0.0f;
    for (uint b = lane; b < nb; b += lpr) {
        device const uchar* blk = row + b * 22;
        const float d = half_at(blk);
        const uint qh = as_type<uint>(uchar4(blk[2], blk[3], blk[4], blk[5]));
        device const uchar* qs = blk + 6;
        float s = 0.0f;
        for (uint jj = 0; jj < 32; jj++) {
            uint nib = jj < 16 ? (qs[jj] & 0xF) : (qs[jj - 16] >> 4);
            int q = (int)(nib | (((qh >> jj) & 1) << 4)) - 16;
            s += float(q) * x[b * 32 + jj];
        }
        total += d * s;
    }
    return total;
}

// Q5_K: like Q4_K with a fifth bit per value from qh.
inline float q5k_row_partial(
    device const uchar* row, device const float* x, uint nb, uint lane, uint lpr)
{
    float total = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* blk = row + b * 176;
        const float d = half_at(blk);
        const float dmin = half_at(blk + 2);
        device const uchar* packed = blk + 4;
        device const uchar* qh = blk + 16;
        device const uchar* qs = blk + 48;
        device const float* xb = x + b * 256;
        for (uint pair = lane; pair < 4; pair += lpr) {
            uchar2 s1 = q4k_sc_min(pair * 2, packed);
            uchar2 s2 = q4k_sc_min(pair * 2 + 1, packed);
            device const uchar* q = qs + pair * 32;
            const uchar bit1 = 1 << (pair * 2);
            const uchar bit2 = 1 << (pair * 2 + 1);
            float dot_lo = 0.0f, sum_lo = 0.0f, dot_hi = 0.0f, sum_hi = 0.0f;
            for (uint l = 0; l < 32; l++) {
                float xl = xb[pair * 64 + l];
                float xh = xb[pair * 64 + 32 + l];
                float lo = float((q[l] & 0xF) + ((qh[l] & bit1) != 0 ? 16 : 0));
                float hi = float((q[l] >> 4) + ((qh[l] & bit2) != 0 ? 16 : 0));
                dot_lo += lo * xl;
                sum_lo += xl;
                dot_hi += hi * xh;
                sum_hi += xh;
            }
            total += d * s1.x * dot_lo - dmin * s1.y * sum_lo
                   + d * s2.x * dot_hi - dmin * s2.y * sum_hi;
        }
    }
    return total;
}

// Q8_0: 34-byte blocks of 32; a lane takes whole blocks.
inline float q80_row_partial(
    device const uchar* row, device const float* x, uint nb, uint lane, uint lpr)
{
    float total = 0.0f;
    for (uint b = lane; b < nb; b += lpr) {
        device const uchar* blk = row + b * 34;
        const float d = half_at(blk);
        device const char* qs = (device const char*)(blk + 2);
        float s = 0.0f;
        for (uint jj = 0; jj < 32; jj++) {
            s += float(qs[jj]) * x[b * 32 + jj];
        }
        total += d * s;
    }
    return total;
}

// One expert matvec stage of the megakernel: rows of `slots` experts
// partitioned over the grid's simdgroups.
inline void mega_expert_stage(
    uint fmt,
    device const uchar* w,
    ulong w_off,
    ulong stride,
    device const float* xbase,
    ulong x_stride,
    device float* out,
    uint n_in,
    uint n_out,
    uint slots,
    device const uint* ids,
    uint sgid,
    uint total_sg,
    ushort tiisg)
{
    uint nb = n_in / 256;
    if (fmt >= 2) {
        // New formats: lpr lanes per row, whole-row work items of `tpr`
        // tasks (64-element pairs for the K formats, 32-value blocks else).
        uint row_bytes;
        uint tpr;
        switch (fmt) {
            case 2: row_bytes = nb * 144; tpr = nb * 4; break;
            case 3: row_bytes = (n_in / 32) * 22; tpr = n_in / 32; break;
            case 4: row_bytes = nb * 176; tpr = nb * 4; break;
            default: row_bytes = (n_in / 32) * 34; tpr = n_in / 32; break;
        }
        uint lpr = min(32u, max(1u, (uint)1 << (31 - clz(tpr))));
        uint rpg = 32 / lpr;
        uint lane = tiisg % lpr;
        uint sub = tiisg / lpr;
        uint tasks = (n_out * slots + rpg - 1) / rpg;
        for (uint task = sgid; task < tasks; task += total_sg) {
            uint jt = task * rpg + sub;
            bool active = jt < n_out * slots;
            uint slot = active ? jt / n_out : 0;
            uint j = active ? jt % n_out : 0;
            device const uchar* row =
                w + ids[slot] * stride + w_off + (ulong)j * row_bytes;
            device const float* x = xbase + (ulong)slot * x_stride;
            float total;
            switch (fmt) {
                case 2: total = q4k_row_partial(row, x, nb, lane, lpr); break;
                case 3: total = q50_row_partial(row, x, n_in / 32, lane, lpr); break;
                case 4: total = q5k_row_partial(row, x, nb, lane, lpr); break;
                default: total = q80_row_partial(row, x, n_in / 32, lane, lpr); break;
            }
            float s = lane_sum_dyn(total, lpr);
            if (active && lane == 0) {
                out[jt] = s;
            }
        }
        return;
    }
    if (fmt == 0) {
        // q2_k: lpr lanes per row; a simdgroup covers 32/lpr rows.
        uint lpr = min(32u, max(1u, (uint)1 << (31 - clz(nb))));
        uint rpg = 32 / lpr;
        uint lane = tiisg % lpr;
        uint sub = tiisg / lpr;
        uint tasks = (n_out * slots + rpg - 1) / rpg;
        for (uint task = sgid; task < tasks; task += total_sg) {
            uint jt = task * rpg + sub;
            bool active = jt < n_out * slots;
            uint slot = active ? jt / n_out : 0;
            uint j = active ? jt % n_out : 0;
            device const uchar* row =
                w + ids[slot] * stride + w_off + (ulong)j * nb * 84;
            device const float* x = xbase + (ulong)slot * x_stride;
            float total = q2k_row_partial(row, x, nb, lane, lpr);
            float s = lane_sum_dyn(total, lpr);
            if (active && lane == 0) {
                out[jt] = s;
            }
        }
    } else {
        // q3_k: 2 rows per simdgroup.
        ulong nb01 = (ulong)nb * 110;
        uint tasks = (n_out * slots + 1) / 2;
        for (uint task = sgid; task < tasks; task += total_sg) {
            uint jt = task * 2;
            uint slot = jt / n_out;
            uint j = jt % n_out;
            // Rows never straddle experts: n_out is even for every target.
            uint nrows = min(2u, n_out * slots - jt);
            device const uchar* row0 =
                w + ids[slot] * stride + w_off + (ulong)j * nb01;
            device const float* x = xbase + (ulong)slot * x_stride;
            q3k_two_rows(row0, x, nb, nb01, nrows, tiisg, out + jt, 1);
        }
    }
}

kernel void moe_ffn_mega(
    device float* y [[buffer(0)]],
    device const uchar* gate_w [[buffer(1)]],
    device const uchar* up_w [[buffer(2)]],
    device const uchar* down_w [[buffer(3)]],
    device const uchar* router_w [[buffer(4)]],
    device const float* norm_w [[buffer(5)]],
    constant MegaArgs& a [[buffer(6)]],
    device const float* rbias [[buffer(7)]],
    device const uchar* sh_gate_w [[buffer(8)]],
    device const uchar* sh_up_w [[buffer(9)]],
    device const uchar* sh_down_w [[buffer(10)]],
    device const float* sh_gout [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    device atomic_uint* ctr = (device atomic_uint*)(y + a.ctr_at);
    // n_tg==0 would make total_sg==0 and turn every `task += total_sg` into
    // an infinite loop; clamp so a bad constant-buffer read cannot hang.
    uint n_tg = max(1u, a.n_tg);
    uint sgid = tgid * 8 + sg;
    uint total_sg = n_tg * 8;
    uint gthread = tgid * 256 + ltid;
    uint total_threads = n_tg * 256;
    uint base = a.ctr_base;

    // s0: x += delta; h = rmsnorm(x) * ffn_norm. TG0 alone; the norm is a
    // ~3 us serial step either way, and here the boundary costs ~1 us.
    if (tgid == 0) {
        threadgroup float partial[8];
        device float* x = y + a.x_at;
        device const float* delta = y + a.delta_at;
        device float* h = y + a.h_at;
        float acc = 0.0f;
        for (uint i = ltid; i < a.hidden; i += 256) {
            float v = x[i] + delta[i];
            x[i] = v;
            acc += v * v;
        }
        float s = simd_sum(acc);
        if (tiisg == 0) {
            partial[sg] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < 8; i++) {
            tot += partial[i];
        }
        float scale = rsqrt(tot / (float)a.hidden + a.eps);
        for (uint i = ltid; i < a.hidden; i += 256) {
            h[i] = x[i] * scale * norm_w[i];
        }
    }
    mega_sync(ctr, base + 1 * n_tg, ltid, n_tg);

    // s1: router logits, one expert row per simdgroup.
    {
        device const float* h = y + a.h_at;
        device const float* wr = (device const float*)(router_w + a.router_off);
        for (uint e = sgid; e < a.n_expert; e += total_sg) {
            device const float* row = wr + (ulong)e * a.hidden;
            float acc = 0.0f;
            for (uint i = tiisg * 4; i + 3 < a.hidden; i += 128) {
                acc += dot(*(device const float4*)(row + i),
                           *(device const float4*)(h + i));
            }
            float s = simd_sum(acc);
            if (tiisg == 0) {
                y[a.logits_at + e] = s;
            }
        }
    }
    mega_sync(ctr, base + 2 * n_tg, ltid, n_tg);

    // s2: top-k over the logits; first simdgroup of TG0. Softmax gating for
    // Qwen-style models; GLM selects by sigmoid(l) + rbias and weights by
    // the unbiased sigmoid, llama.cpp's selection_probs semantics.
    if (tgid == 0 && sg == 0) {
        device const float* logits = y + a.logits_at;
        device uint* ids = (device uint*)(y + a.ids_at);
        device float* wts = y + a.wts_at;
        uint lane = tiisg;
        float v[4];
        uint vid[4];
        for (uint t = 0; t < 4; t++) {
            uint idx = lane + t * 32;
            if (idx < a.n_expert) {
                v[t] = a.sigmoid != 0
                    ? 1.0f / (1.0f + exp(-logits[idx])) + rbias[idx]
                    : logits[idx];
            } else {
                v[t] = -INFINITY;
            }
            vid[t] = idx;
        }
        for (uint kk = 0; kk < a.n_used; kk++) {
            float lm = -INFINITY;
            uint li = 0xFFFFFFFF;
            for (uint t = 0; t < 4; t++) {
                if (v[t] > lm) {
                    lm = v[t];
                    li = vid[t];
                }
            }
            float m = simd_max(lm);
            uint who = simd_min(lm == m ? li : 0xFFFFFFFF);
            if (lane == 0) {
                ids[kk] = who;
            }
            for (uint t = 0; t < 4; t++) {
                if (vid[t] == who) {
                    v[t] = -INFINITY;
                }
            }
        }
        if (lane == 0) {
            if (a.has_shared != 0) {
                // The shared expert rides slot n_used with a constant id 0;
                // its weight is applied unconditionally in the combine.
                ids[a.n_used] = 0;
            }
            if (a.sigmoid != 0) {
                float total = 0.0f;
                for (uint kk = 0; kk < a.n_used; kk++) {
                    float s = 1.0f / (1.0f + exp(-logits[ids[kk]]));
                    wts[kk] = s;
                    total += s;
                }
                for (uint kk = 0; kk < a.n_used; kk++) {
                    wts[kk] /= total;
                }
            } else {
                float mx = -INFINITY;
                for (uint kk = 0; kk < a.n_used; kk++) {
                    mx = max(mx, logits[ids[kk]]);
                }
                float total = 0.0f;
                for (uint kk = 0; kk < a.n_used; kk++) {
                    float e = exp(logits[ids[kk]] - mx);
                    wts[kk] = e;
                    total += e;
                }
                for (uint kk = 0; kk < a.n_used; kk++) {
                    wts[kk] /= total;
                }
            }
        }
        // Gated shared expert (GLM / qwen35moe): raw gate logit into
        // wts[n_used]; sigmoid applied at combine. One simdgroup owns the
        // hidden-wide dot.
        if (a.has_shared == 2 && tgid == 0 && sg == 0) {
            device const float* h = y + a.h_at;
            // sh_gout_off is a BYTE offset into the F32 projection buffer.
            device const float* proj =
                (device const float*)((device const uchar*)sh_gout + a.sh_gout_off);
            float acc = 0.0f;
            for (uint i = tiisg * 4; i + 3 < a.hidden; i += 128) {
                acc += dot(*(device const float4*)(proj + i),
                           *(device const float4*)(h + i));
            }
            float s = simd_sum(acc);
            if (tiisg == 0) {
                y[a.wts_at + a.n_used] = s;
            }
        }
    }
    mega_sync(ctr, base + 3 * n_tg, ltid, n_tg);

    // s3: gate and up projections for every hit expert (+ the shared one).
    {
        device const uint* ids = (device const uint*)(y + a.ids_at);
        device const float* h = y + a.h_at;
        mega_expert_stage(GATE_FMT, gate_w, a.gate_off, a.gate_stride, h, 0,
                          y + a.gate_at, a.hidden, a.ffn, a.n_used, ids,
                          sgid, total_sg, tiisg);
        mega_expert_stage(GATE_FMT, up_w, a.up_off, a.up_stride, h, 0,
                          y + a.up_at, a.hidden, a.ffn,
                          a.n_used, ids, sgid, total_sg, tiisg);
        if (a.has_shared != 0) {
            mega_expert_stage(SH_GATE_FMT, sh_gate_w, a.sh_gate_off, 0, h, 0,
                              y + a.gate_at + a.n_used * a.ffn, a.hidden, a.ffn,
                              1, ids + a.n_used, sgid, total_sg, tiisg);
            mega_expert_stage(SH_GATE_FMT, sh_up_w, a.sh_up_off, 0, h, 0,
                              y + a.up_at + a.n_used * a.ffn, a.hidden, a.ffn,
                              1, ids + a.n_used, sgid, total_sg, tiisg);
        }
    }
    mega_sync(ctr, base + 4 * n_tg, ltid, n_tg);

    // s4: swiglu in place over the gate half.
    {
        device float* g = y + a.gate_at;
        device const float* u = y + a.up_at;
        // has_shared is 0/1/2 (tri-state); only 0/1 contributes a slot.
        uint n = (a.n_used + (a.has_shared != 0)) * a.ffn;
        for (uint i = gthread; i < n; i += total_threads) {
            float gv = g[i];
            g[i] = u[i] * (gv / (1.0f + exp(-gv)));
        }
    }
    mega_sync(ctr, base + 5 * n_tg, ltid, n_tg);

    // s5: down projections, per-slot activations (+ the shared one).
    {
        device const uint* ids = (device const uint*)(y + a.ids_at);
        mega_expert_stage(DOWN_FMT, down_w, a.down_off, a.down_stride,
                          y + a.gate_at, a.ffn, y + a.downo_at, a.ffn,
                          a.hidden, a.n_used, ids, sgid, total_sg, tiisg);
        if (a.has_shared != 0) {
            mega_expert_stage(SH_DOWN_FMT, sh_down_w, a.sh_down_off, 0,
                              y + a.gate_at + a.n_used * a.ffn, 0,
                              y + a.downo_at + a.n_used * a.hidden, a.ffn,
                              a.hidden, 1, ids + a.n_used, sgid, total_sg, tiisg);
        }
    }
    mega_sync(ctr, base + 6 * n_tg, ltid, n_tg);

    // s6: weighted combine into delta; the shared slot joins with weight 1.
    {
        device const float* wts = y + a.wts_at;
        device const float* downo = y + a.downo_at;
        device float* delta = y + a.delta_at;
        for (uint i = gthread; i < a.hidden; i += total_threads) {
            float acc = 0.0f;
            for (uint s = 0; s < a.n_used; s++) {
                acc += wts[s] * downo[s * a.hidden + i];
            }
            if (a.has_shared == 1) {
                acc += downo[a.n_used * a.hidden + i];
            } else if (a.has_shared == 2) {
                float g = y[a.wts_at + a.n_used];
                float w = 1.0f / (1.0f + exp(-g));
                acc += w * downo[a.n_used * a.hidden + i];
            }
            delta[i] = acc;
        }
    }
}

kernel void softmax_topk(
    device const float* logits [[buffer(0)]],
    device uint* ids [[buffer(1)]],
    device float* wts [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]])
{
    float v[8];
    uint vid[8];
    for (uint t = 0; t < 8; t++) {
        uint idx = lane + t * 32;
        v[t] = idx < n ? logits[idx] : -INFINITY;
        vid[t] = idx;
    }
    for (uint kk = 0; kk < k; kk++) {
        float lm = -INFINITY;
        uint li = 0xFFFFFFFF;
        for (uint t = 0; t < 8; t++) {
            if (v[t] > lm) {
                lm = v[t];
                li = vid[t];
            }
        }
        float m = simd_max(lm);
        // The owner is the smallest id among lanes holding the max, which
        // matches the CPU sort's stable tie-break on index order.
        uint who = simd_min(lm == m ? li : 0xFFFFFFFF);
        if (lane == 0) {
            ids[kk] = who;
        }
        for (uint t = 0; t < 8; t++) {
            if (vid[t] == who) {
                v[t] = -INFINITY;
            }
        }
    }
    if (lane == 0) {
        float mx = -INFINITY;
        for (uint kk = 0; kk < k; kk++) {
            mx = max(mx, logits[ids[kk]]);
        }
        float total = 0.0f;
        for (uint kk = 0; kk < k; kk++) {
            float e = exp(logits[ids[kk]] - mx);
            wts[kk] = e;
            total += e;
        }
        for (uint kk = 0; kk < k; kk++) {
            wts[kk] /= total;
        }
    }
}

// Same top-k selection as softmax_topk, but sigmoid gating (GLM-4 MoE).
// Selection key: sigmoid(l_i) + rbias_i (llama.cpp `selection_probs`);
// weights: UNBIASED sigmoid(l_i) renormalized over the picked k.
kernel void sigmoid_topk(
    device const float* logits [[buffer(0)]],
    device uint* ids [[buffer(1)]],
    device float* wts [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    device const float* rbias [[buffer(5)]],
    constant uint& has_bias [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]])
{
    float v[8];
    uint vid[8];
    for (uint t = 0; t < 8; t++) {
        uint idx = lane + t * 32;
        if (idx < n) {
            float s = 1.0f / (1.0f + exp(-logits[idx]));
            v[t] = s + (has_bias != 0 ? rbias[idx] : 0.0f);
        } else {
            v[t] = -INFINITY;
        }
        vid[t] = idx;
    }
    for (uint kk = 0; kk < k; kk++) {
        float lm = -INFINITY;
        uint li = 0xFFFFFFFF;
        for (uint t = 0; t < 8; t++) {
            if (v[t] > lm) {
                lm = v[t];
                li = vid[t];
            }
        }
        float m = simd_max(lm);
        uint who = simd_min(lm == m ? li : 0xFFFFFFFF);
        if (lane == 0) {
            ids[kk] = who;
        }
        for (uint t = 0; t < 8; t++) {
            if (vid[t] == who) {
                v[t] = -INFINITY;
            }
        }
    }
    if (lane == 0) {
        float total = 0.0f;
        for (uint kk = 0; kk < k; kk++) {
            float s = 1.0f / (1.0f + exp(-logits[ids[kk]]));
            wts[kk] = s;
            total += s;
        }
        for (uint kk = 0; kk < k; kk++) {
            wts[kk] /= total;
        }
    }
}

// delta[h] = sum over slots of wts[s] * down[s * n + h].
// sig_last: 0 = all slots use wts[s], 1 = last slot uses sigmoid(wts[slots - 1]),
// 2 = last slot uses weight 1.0 (unweighted shared expert).
kernel void moe_combine(
    device const float* down [[buffer(0)]],
    device const float* wts [[buffer(1)]],
    device float* delta [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& slots [[buffer(4)]],
    constant uint& sig_last [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float wl = 1.0f;
    if (sig_last == 1) {
        const float g = wts[slots - 1];
        wl = 1.0f / (1.0f + exp(-g));
    }
    float acc = 0.0f;
    for (uint s = 0; s < slots; s++) {
        const float w = (sig_last != 0 && s == slots - 1) ? wl : wts[s];
        acc += w * down[s * n + i];
    }
    delta[i] = acc;
}

// residual_norm with the MoE combine folded in: the effective delta is
// delta + sum over slots of wts[s] * down[s*n + i], so the standalone
// combine dispatch and the layer-end barrier behind it drop out. x absorbs
// the combined delta; h gets the rmsnorm.
kernel void combine_resnorm(
    device float* x [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device float* h [[buffer(2)]],
    device const float* w [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    device const float* down [[buffer(6)]],
    device const float* wts [[buffer(7)]],
    constant uint& slots [[buffer(8)]],
    constant uint& sig_last [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[8];
    float wl = 1.0f;
    if (sig_last == 1) {
        const float g = wts[slots - 1];
        wl = 1.0f / (1.0f + exp(-g));
    }
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        float d = delta[i];
        for (uint s = 0; s < slots; s++) {
            const float w = (sig_last != 0 && s == slots - 1) ? wl : wts[s];
            d += w * down[s * n + i];
        }
        float v = x[i] + d;
        x[i] = v;
        acc += v * v;
    }
    float s = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; i++) {
        total += partial[i];
    }
    float scale = rsqrt(total / (float)n + eps);
    for (uint i = tid; i < n; i += 256) {
        h[i] = x[i] * scale * w[i];
    }
}

// Greedy argmax over the vocab logits, two stages (no 64-bit device
// atomics on Metal): stage one reduces strided slices to one (value, index)
// pair per threadgroup, stage two folds the pairs. Every tie resolves to
// the HIGHER index (Rust max_by's last-wins). Replaces the full-vocab
// readback plus the CPU scan with a 4-byte one when the caller is greedy.
kernel void argmax_f32(
    device const float* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    device float* parts [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tgc [[threads_per_grid]])
{
    float best = -INFINITY;
    uint bidx = 0;
    for (uint i = tgid * 256 + ltid; i < n; i += tgc) {
        float v = x[i];
        if (v > best || (v == best && i > bidx)) {
            best = v;
            bidx = i;
        }
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(bidx, off);
        if (ov > best || (ov == best && oi > bidx)) {
            best = ov;
            bidx = oi;
        }
    }
    threadgroup float svals[8];
    threadgroup uint sidxs[8];
    if (lane == 0) {
        svals[sg] = best;
        sidxs[sg] = bidx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ltid == 0) {
        for (uint i = 1; i < 8; i++) {
            if (svals[i] > best || (svals[i] == best && sidxs[i] > bidx)) {
                best = svals[i];
                bidx = sidxs[i];
            }
        }
        parts[tgid * 2] = best;
        parts[tgid * 2 + 1] = as_type<float>(bidx);
    }
}

// Stage two: one 32-lane SIMD group folds the per-threadgroup pairs into the winner.
kernel void argmax_final(
    device const float* parts [[buffer(0)]],
    constant uint& nparts [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= 32) {
        return;
    }
    float best = -INFINITY;
    uint bidx = 0;
    for (uint i = lane; i < nparts; i += 32) {
        float v = parts[i * 2];
        uint idx = as_type<uint>(parts[i * 2 + 1]);
        if (v > best || (v == best && idx > bidx)) {
            best = v;
            bidx = idx;
        }
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(bidx, off);
        if (ov > best || (ov == best && oi > bidx)) {
            best = ov;
            bidx = oi;
        }
    }
    if (lane == 0) {
        *out = bidx;
    }
}

// Dequantize one token_embd row into activations based on device-side token ID pointer.
kernel void embed_row_f32_dtoken(
    device const uchar* w [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n_in [[buffer(2)]],
    constant ulong& w_base [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    constant uint& fmt [[buffer(5)]],
    device const uint* d_token [[buffer(6)]],
    constant uint& tok_idx [[buffer(7)]],
    uint b [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    ulong w_off = w_base + (ulong)d_token[tok_idx] * (ulong)row_bytes;
    device const uchar* row = w + w_off;
    if (fmt <= 2u) {
        device float* o256 = out + b * 256u;
        if (fmt == 0u) {
            device const uchar* blk = row + b * 144u;
            float d = half_at(blk);
            float dmin = half_at(blk + 2);
            device const uchar* packed = blk + 4;
            device const uchar* qs = blk + 16;
            for (uint pair = 0; pair < 4u; pair++) {
                float sc1, mn1, sc2, mn2;
                scale_min_k4(pair * 2, packed, sc1, mn1);
                scale_min_k4(pair * 2 + 1, packed, sc2, mn2);
                uchar qq = qs[pair * 32u + lane];
                o256[pair * 64u + lane] = d * sc1 * (float)(qq & 0xf) - dmin * mn1;
                o256[pair * 64u + 32u + lane] = d * sc2 * (float)(qq >> 4) - dmin * mn2;
            }
            return;
        }
        if (fmt == 1u) {
            device const uchar* blk = row + b * 176u;
            float d = half_at(blk);
            float dmin = half_at(blk + 2);
            device const uchar* packed = blk + 4;
            device const uchar* qh = blk + 16;
            device const uchar* qs = blk + 48;
            for (uint pair = 0; pair < 4u; pair++) {
                float sc1, mn1, sc2, mn2;
                scale_min_k4(pair * 2, packed, sc1, mn1);
                scale_min_k4(pair * 2 + 1, packed, sc2, mn2);
                uchar bit1 = (uchar)(1u << (pair * 2));
                uchar bit2 = (uchar)(1u << (pair * 2 + 1));
                uchar qq = qs[pair * 32u + lane];
                int hi1 = (qh[lane] & bit1) ? 16 : 0;
                int hi2 = (qh[lane] & bit2) ? 16 : 0;
                o256[pair * 64u + lane] = d * sc1 * (float)((qq & 0xf) + hi1) - dmin * mn1;
                o256[pair * 64u + 32u + lane] = d * sc2 * (float)((qq >> 4) + hi2) - dmin * mn2;
            }
            return;
        }
        {
            device const uchar* blk = row + b * 210u;
            device const uchar* ql = blk;
            device const uchar* qh = blk + 128;
            device const char* scales = (device const char*)(blk + 192);
            float d = half_at(blk + 208);
            for (uint hi = 0; hi < 2u; hi++) {
                device const uchar* ql_h = ql + hi * 64u;
                device const uchar* qh_h = qh + hi * 32u;
                device const char* sc = scales + hi * 8;
                device float* oh = o256 + hi * 128u;
                uint is = lane / 16u;
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
    if (fmt == 3u) {
        for (uint ib = 0; ib < 8u; ib++) {
            uint blk_idx = b * 8u + ib;
            if (blk_idx * 32u >= n_in) break;
            device const uchar* blk = row + blk_idx * 34u;
            float d = half_at(blk);
            device const char* qs = (device const char*)(blk + 2);
            out[blk_idx * 32u + lane] = d * (float)qs[lane];
        }
        return;
    }
    if (fmt == 4u) {
        if (b * 256u >= n_in) return;
        device float* o256 = out + b * 256u;
        device const uchar* blk = row + b * 84u;
        device const uchar* scales = blk;
        device const uchar* qs = blk + 16;
        float d = half_at(blk + 80);
        float dmin = half_at(blk + 82);
        uint group = lane / 16u;
        for (uint half_i = 0; half_i < 2u; half_i++) {
            uchar q = qs[half_i * 32u + lane];
            for (uint si = 0; si < 4u; si++) {
                uint sub = half_i * 8u + si * 2u + group;
                uchar sc = scales[sub];
                float dl = d * (float)(sc & 0xFu);
                float ml = dmin * (float)(sc >> 4u);
                o256[half_i * 128u + si * 32u + lane] =
                    dl * (float)((q >> (si * 2u)) & 3u) - ml;
            }
        }
        return;
    }
    if (fmt == 5u) {
        if (b * 256u >= n_in) return;
        device float* o256 = out + b * 256u;
        device const uchar* blk = row + b * 110u;
        device const uchar* hmask = blk;
        device const uchar* qs = blk + 32;
        device const uchar* packed = blk + 96;
        float d_all = half_at(blk + 108);
        uint group = lane / 16u;
        uchar hm = hmask[lane];
        for (uint half_i = 0; half_i < 2u; half_i++) {
            uchar q = qs[half_i * 32u + lane];
            for (uint si = 0; si < 4u; si++) {
                uint is = half_i * 8u + si * 2u + group;
                uint lo = (is < 8u) ? (packed[is] & 0xFu) : (packed[is - 8u] >> 4u);
                uint hi = (packed[8u + (is % 4u)] >> (2u * (is / 4u))) & 3u;
                float dl = d_all * (float)((int)(lo | (hi << 4u)) - 32);
                uchar bit = (uchar)(1u << (half_i * 4u + si));
                int qv = (int)((q >> (2u * si)) & 3u) - ((hm & bit) ? 0 : 4);
                o256[half_i * 128u + si * 32u + lane] = dl * (float)qv;
            }
        }
        return;
    }
    if (fmt == 6u) {
        device const ushort* src = (device const ushort*)row;
        uint idx = b * 32u + lane;
        if (idx < n_in) out[idx] = as_type<float>((uint)src[idx] << 16u);
        return;
    }
    if (fmt == 7u) {
        device const half* src = (device const half*)row;
        uint idx = b * 32u + lane;
        if (idx < n_in) out[idx] = (float)src[idx];
        return;
    }
    if (fmt == 8u) {
        device const float* src = (device const float*)row;
        uint idx = b * 32u + lane;
        if (idx < n_in) out[idx] = src[idx];
        return;
    }
}

kernel void zero_f32(
    device float* buf [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) buf[tid] = 0.0f;
}

kernel void store_u32_at(
    device uint* buf [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    constant uint& dst_idx [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid == 0) buf[dst_idx] = *src;
}

// ---- The prefill tile-matmul path -------------------------------------
//
// The matvec kernels above read every weight byte once per 8 activation rows;
// at prefill batch sizes that is the whole cost. These kernels read a weight
// tile once per 32 rows AND run the arithmetic on the simdgroup matrix units,
// which the scalar kernels cannot touch.
//
// Shape: one threadgroup computes a [32 output rows x 32 batch rows] tile of
// y. Weights are dequantised 16 elements at a time into a threadgroup tile of
// halves, activations are loaded (and narrowed) next to them, and 8x8
// simdgroup matrices do the multiply with f32 accumulators. Precision-wise
// this matches llama.cpp's mul_mm: half inputs, float accumulation.
//
// The dequant helpers mirror the matvec kernels' bit manipulation exactly;
// every 16-element chunk stays inside one sub-scale group in every format,
// which is why 16 is the granule.

inline void dq16_q8_0(device const uchar* row, uint k0, threadgroup half* out) {
    device const uchar* blk = row + (k0 / 32) * 34;
    float d = half_at(blk);
    uint off = k0 % 32;
    for (uint i = 0; i < 16; i++) {
        out[i] = (half)(d * (float)((char)blk[2 + off + i]));
    }
}

inline void dq16_q2_k(device const uchar* row, uint k0, threadgroup half* out) {
    device const uchar* blk = row + (k0 / 256) * 84;
    uint r = k0 % 256;
    float d = half_at(blk + 80);
    float dmin = half_at(blk + 82);
    uchar sc = blk[r / 16];
    float dl = d * (float)(sc & 0xF);
    float ml = dmin * (float)(sc >> 4);
    uint half_i = r / 128, rem = r % 128;
    uint shift = (rem / 32) * 2, group = (rem % 32) / 16;
    device const uchar* q = blk + 16 + half_i * 32 + group * 16;
    for (uint i = 0; i < 16; i++) {
        out[i] = (half)(dl * (float)((q[i] >> shift) & 3) - ml);
    }
}

inline void dq16_q3_k(device const uchar* row, uint k0, threadgroup half* out) {
    device const uchar* blk = row + (k0 / 256) * 110;
    uint r = k0 % 256;
    device const uchar* hmask = blk;
    device const uchar* qs = blk + 32;
    device const uchar* packed = blk + 96;
    float d_all = half_at(blk + 108);
    uint is = r / 16;
    uint lo = (is < 8) ? (packed[is] & 0xF) : (packed[is - 8] >> 4);
    uint hi = (packed[8 + (is % 4)] >> (2 * (is / 4))) & 3;
    float dl = d_all * (float)((int)(lo | (hi << 4)) - 32);
    uint half_i = r / 128, rem = r % 128;
    uint si = rem / 32, group = (rem % 32) / 16;
    uchar mbit = 1 << (half_i * 4 + si);
    device const uchar* q = qs + half_i * 32 + group * 16;
    device const uchar* hm = hmask + group * 16;
    for (uint i = 0; i < 16; i++) {
        int qi = (int)((q[i] >> (2 * si)) & 3) - ((hm[i] & mbit) ? 0 : 4);
        out[i] = (half)(dl * (float)qi);
    }
}

inline void dq16_q4_k(device const uchar* row, uint k0, threadgroup half* out) {
    device const uchar* blk = row + (k0 / 256) * 144;
    uint r = k0 % 256;
    float d = half_at(blk);
    float dmin = half_at(blk + 2);
    device const uchar* packed = blk + 4;
    device const uchar* qs = blk + 16;
    uint pair = r / 64, off = r % 64;
    float sc, mn;
    if (off < 32) {
        scale_min_k4(pair * 2, packed, sc, mn);
        device const uchar* q = qs + pair * 32 + off;
        for (uint i = 0; i < 16; i++) {
            out[i] = (half)(d * sc * (float)(q[i] & 0xF) - dmin * mn);
        }
    } else {
        scale_min_k4(pair * 2 + 1, packed, sc, mn);
        device const uchar* q = qs + pair * 32 + off - 32;
        for (uint i = 0; i < 16; i++) {
            out[i] = (half)(d * sc * (float)(q[i] >> 4) - dmin * mn);
        }
    }
}

inline void dq16_q5_k(device const uchar* row, uint k0, threadgroup half* out) {
    device const uchar* blk = row + (k0 / 256) * 176;
    uint r = k0 % 256;
    float d = half_at(blk);
    float dmin = half_at(blk + 2);
    device const uchar* packed = blk + 4;
    device const uchar* qh = blk + 16;
    device const uchar* qs = blk + 48;
    uint pair = r / 64, off = r % 64;
    float sc, mn;
    if (off < 32) {
        scale_min_k4(pair * 2, packed, sc, mn);
        uchar bit = 1 << (pair * 2);
        device const uchar* q = qs + pair * 32 + off;
        device const uchar* h = qh + off;
        for (uint i = 0; i < 16; i++) {
            int qi = (int)(q[i] & 0xF) + ((h[i] & bit) ? 16 : 0);
            out[i] = (half)(d * sc * (float)qi - dmin * mn);
        }
    } else {
        scale_min_k4(pair * 2 + 1, packed, sc, mn);
        uchar bit = 2 << (pair * 2);
        device const uchar* q = qs + pair * 32 + off - 32;
        device const uchar* h = qh + off - 32;
        for (uint i = 0; i < 16; i++) {
            int qi = (int)(q[i] >> 4) + ((h[i] & bit) ? 16 : 0);
            out[i] = (half)(d * sc * (float)qi - dmin * mn);
        }
    }
}

inline void dq16_q6_k(device const uchar* row, uint k0, threadgroup half* out) {
    device const uchar* blk = row + (k0 / 256) * 210;
    uint r = k0 % 256;
    float d = half_at(blk + 208);
    uint half_i = r / 128, rem = r % 128;
    uint quadrant = rem / 32, is = (rem % 32) / 16;
    uint l = rem % 32;
    device const uchar* ql = blk + half_i * 64;
    device const uchar* qh = blk + 128 + half_i * 32;
    float sc = d * (float)((char)blk[192 + half_i * 8 + quadrant * 2 + is]);
    // `l` is already the position inside the 32-wide quadrant, so the same
    // ql/qh bytes serve all four quadrants at different bit positions.
    for (uint i = 0; i < 16; i++) {
        uint li = l + i;
        int qi;
        switch (quadrant) {
            case 0: qi = (int)((ql[li] & 0xF) | ((qh[li] & 3) << 4)); break;
            case 1: qi = (int)((ql[li + 32] & 0xF) | (((qh[li] >> 2) & 3) << 4)); break;
            case 2: qi = (int)((ql[li] >> 4) | (((qh[li] >> 4) & 3) << 4)); break;
            default: qi = (int)((ql[li + 32] >> 4) | (((qh[li] >> 6) & 3) << 4)); break;
        }
        out[i] = (half)(sc * (float)(qi - 32));
    }
}

inline void dq16_f32(device const uchar* row, uint k0, threadgroup half* out) {
    device const float4* f4 = (device const float4*)((device const float*)row + k0);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; i++) {
        *(threadgroup half4*)(out + i * 4) = half4(f4[i]);
    }
}

// One threadgroup per [BM x 32] output tile, 128 threads in 4 SIMD groups.
// SIMD group `sg` owns M-strips sg, sg+4, ... (BM/8 strips of 8 rows total),
// each against all 32 batch columns. BM is a macro parameter so the 32- and
// 64-row variants share every line; the sweep between them is a pipeline
// name, not an ifdef.
#define DEFINE_MM(NAME, DQ16, ROW_BYTES, BM_, BN_)                             \
kernel void NAME(                                                              \
    device const uchar* w [[buffer(0)]],                                       \
    device const float* x [[buffer(1)]],                                       \
    device float* y [[buffer(2)]],                                             \
    constant uint& n_in [[buffer(3)]],                                         \
    constant uint& n_out [[buffer(4)]],                                        \
    constant ulong& w_off [[buffer(5)]],                                       \
    constant uint& m [[buffer(6)]],                                            \
    uint2 tg [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]],                                  \
    uint sg [[simdgroup_index_in_threadgroup]])                                \
{                                                                              \
    const uint BM = BM_, BN = BN_, BK = 32;                                    \
    const uint STRIPS = BM / 32;                                               \
    const uint NT = BN / 8;                                                    \
    uint i0 = tg.x * BM;                                                       \
    uint j0 = tg.y * BN;                                                       \
    threadgroup half As[BM_ * 32];                                             \
    threadgroup half Bs[32 * BN_];                                             \
    threadgroup float Cs[BM_ * BN_];                                           \
    simdgroup_float8x8 c[STRIPS * NT];                                         \
    for (uint t = 0; t < STRIPS * NT; t++) {                                   \
        c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);                \
    }                                                                          \
    for (uint k0 = 0; k0 < n_in; k0 += BK) {                                   \
        for (uint idx = tid; idx < BM * BK / 16; idx += 128) {                 \
            uint row = idx / (BK / 16);                                        \
            uint part = idx % (BK / 16);                                       \
            uint ri = min(i0 + row, n_out - 1);                                \
            device const uchar* rp = w + w_off + (ulong)ri * (ROW_BYTES);      \
            DQ16(rp, k0 + part * 16, As + row * BK + part * 16);               \
        }                                                                      \
        for (uint idx = tid; idx < BK * BN; idx += 128) {                      \
            uint k = idx / BN;                                                 \
            uint j = j0 + idx % BN;                                            \
            Bs[idx] = (j < m) ? (half)x[(ulong)j * n_in + k0 + k] : 0.0h;      \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (uint kk = 0; kk < BK; kk += 8) {                                  \
            for (uint st = 0; st < STRIPS; st++) {                             \
                simdgroup_half8x8 a;                                           \
                simdgroup_load(a, As + (st * 4 + sg) * 8 * BK + kk, BK);       \
                for (uint t = 0; t < NT; t++) {                                \
                    simdgroup_half8x8 b;                                       \
                    simdgroup_load(b, Bs + kk * BN + t * 8, BN);               \
                    simdgroup_multiply_accumulate(                             \
                        c[st * NT + t], a, b, c[st * NT + t]);                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    for (uint st = 0; st < STRIPS; st++) {                                     \
        for (uint t = 0; t < NT; t++) {                                        \
            simdgroup_store(c[st * NT + t],                                    \
                Cs + (st * 4 + sg) * 8 * BN + t * 8, BN);                      \
        }                                                                      \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint idx = tid; idx < BM * BN; idx += 128) {                          \
        uint i = i0 + idx / BN;                                                \
        uint j = j0 + idx % BN;                                                \
        if (i < n_out && j < m) {                                              \
            y[(ulong)j * n_out + i] = Cs[idx];                                 \
        }                                                                      \
    }                                                                          \
}

DEFINE_MM(matmul_q8_0, dq16_q8_0, (n_in / 32) * 34, 32, 32)
DEFINE_MM(matmul_q2_k, dq16_q2_k, (n_in / 256) * 84, 32, 32)
DEFINE_MM(matmul_q3_k, dq16_q3_k, (n_in / 256) * 110, 32, 32)
DEFINE_MM(matmul_q4_k, dq16_q4_k, (n_in / 256) * 144, 32, 32)
DEFINE_MM(matmul_q5_k, dq16_q5_k, (n_in / 256) * 176, 32, 32)
DEFINE_MM(matmul_q6_k, dq16_q6_k, (n_in / 256) * 210, 32, 32)
DEFINE_MM(matmul_f32, dq16_f32, (ulong)n_in * 4, 32, 32)
DEFINE_MM(matmul64_q8_0, dq16_q8_0, (n_in / 32) * 34, 64, 32)
DEFINE_MM(matmul64_q2_k, dq16_q2_k, (n_in / 256) * 84, 64, 32)
DEFINE_MM(matmul64_q3_k, dq16_q3_k, (n_in / 256) * 110, 64, 32)
DEFINE_MM(matmul64_q4_k, dq16_q4_k, (n_in / 256) * 144, 64, 32)
DEFINE_MM(matmul64_q5_k, dq16_q5_k, (n_in / 256) * 176, 64, 32)
DEFINE_MM(matmul64_q6_k, dq16_q6_k, (n_in / 256) * 210, 64, 32)
DEFINE_MM(matmul64_f32, dq16_f32, (ulong)n_in * 4, 64, 32)
DEFINE_MM(matmulw_q8_0, dq16_q8_0, (n_in / 32) * 34, 32, 64)
DEFINE_MM(matmulw_q2_k, dq16_q2_k, (n_in / 256) * 84, 32, 64)
DEFINE_MM(matmulw_q3_k, dq16_q3_k, (n_in / 256) * 110, 32, 64)
DEFINE_MM(matmulw_q4_k, dq16_q4_k, (n_in / 256) * 144, 32, 64)
DEFINE_MM(matmulw_q5_k, dq16_q5_k, (n_in / 256) * 176, 32, 64)
DEFINE_MM(matmulw_q6_k, dq16_q6_k, (n_in / 256) * 210, 32, 64)
DEFINE_MM(matmulw_f32, dq16_f32, (ulong)n_in * 4, 32, 64)

// ---- Port of llama.cpp's kernel_mul_mm (MIT) ----------------------------
//
// Structural differences to DEFINE_MM above: the output tile is 64 weight
// rows x 32 batch columns per 128-thread group (2x2 simdgroups of 32x16
// each), the K step is 32, and BOTH operands stage through threadgroup
// memory as HALF in an 8x8-tile-major swizzle that simdgroup_load consumes
// directly. Dequantisation happens in registers (16 values per thread per
// K step) on the way in, so the multiply is a pure half x half simdgroup
// MMA with float accumulators.
//
// Block layouts as GGUF stores them; every size is even so the half fields
// stay 2-aligned at any block index.
struct BlkQ8_0 { half d; char qs[32]; };
// Q5_0: f16 scale, 32 high bits, 16 nibble bytes; 22 bytes per 32 values.
struct BlkQ5_0 { half d; uchar qh[4]; uchar qs[16]; };
// F32 rows viewed as 16-float tiles for the LL staging path (router only).
struct BlkF32 { float4 v[4]; };
struct BlkQ2K { uchar scales[16]; uchar qs[64]; half d; half dmin; };
struct BlkQ3K { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; };
struct BlkQ4K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct BlkQ5K { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; };
struct BlkQ6K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };

// llama.cpp's per-16-element dequantisers, verbatim except for float4x4 ->
// explicit half4x4 output.
inline uchar2 sc_min_k4(int j, int k, device const uchar* q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

inline void dqll_f32(device const BlkF32* xb, short il, thread half4x4& reg) {
    (void)il;
    for (int i = 0; i < 4; i++) {
        float4 v = xb->v[i];
        reg[i][0] = (half)v.x;
        reg[i][1] = (half)v.y;
        reg[i][2] = (half)v.z;
        reg[i][3] = (half)v.w;
    }
}

inline void dqll_q8_0(device const BlkQ8_0* xb, short il, thread half4x4& reg) {
    device const char* qs = xb->qs;
    const float d = xb->d;
    for (int i = 0; i < 16; i++) {
        reg[i / 4][i % 4] = (half)(qs[i + 16 * il] * d);
    }
}

// il = 0 takes the low-nibble half (elements 0..15), il = 1 the high-nibble
// half (elements 16..31); the high bit comes from qh bit (i + 16*il).
inline void dqll_q5_0(device const BlkQ5_0* xb, short il, thread half4x4& reg) {
    const float d = xb->d;
    const uint qh = as_type<uint>(uchar4(xb->qh[0], xb->qh[1], xb->qh[2], xb->qh[3]));
    for (int i = 0; i < 16; i++) {
        const int j = i + 16 * il;
        const uint nib = il == 0 ? (xb->qs[i] & 0xF) : (xb->qs[i] >> 4);
        const int q = (int)(nib | (((qh >> j) & 1) << 4)) - 16;
        reg[i / 4][i % 4] = (half)(q * d);
    }
}

inline void dqll_q2_k(device const BlkQ2K* xb, short il, thread half4x4& reg) {
    const float d = xb->d;
    const float mn = xb->dmin;
    device const uchar* q = xb->qs;
    uchar sc = xb->scales[il];
    q = q + 32 * (il / 8) + 16 * (il & 1);
    il = (il / 2) % 4;
    half coef = il > 1 ? (il > 2 ? 1 / 64.h : 1 / 16.h) : (il > 0 ? 1 / 4.h : 1.h);
    uchar mask = il > 1 ? (il > 2 ? 192 : 48) : (il > 0 ? 12 : 3);
    float dl = d * (sc & 0xF) * coef;
    float ml = mn * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (q[i] & mask) - ml);
    }
}

inline void dqll_q3_k(device const BlkQ3K* xb, short il, thread half4x4& reg) {
    const half d_all = xb->d;
    device const uchar* q = xb->qs;
    device const uchar* h = xb->hmask;
    device const char* scales = (device const char*)xb->scales;
    q = q + 32 * (il / 8) + 16 * (il & 1);
    h = h + 16 * (il & 1);
    uchar m = 1 << (il / 2);
    ushort kmask1 = (il / 4) > 1 ? ((il / 4) > 2 ? 192 : 48) : ((il / 4) > 0 ? 12 : 3);
    ushort kmask2 = il / 8 ? 0xF0 : 0x0F;
    short scale_2 = scales[il % 8], scale_1 = scales[8 + il % 4];
    short dl_int = (il / 4) & 1 ? (scale_2 & kmask2) | ((scale_1 & kmask1) << 2)
                                : (scale_2 & kmask2) | ((scale_1 & kmask1) << 4);
    float dl = il < 8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;
    il = (il / 2) & 3;
    const half coef = il > 1 ? (il > 2 ? 1 / 64.h : 1 / 16.h) : (il > 0 ? 1 / 4.h : 1.h);
    const uchar mask = il > 1 ? (il > 2 ? 192 : 48) : (il > 0 ? 12 : 3);
    dl *= coef;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (q[i] & mask) - (h[i] & m ? 0 : ml));
    }
}

inline void dqll_q4_k(device const BlkQ4K* xb, short il, thread half4x4& reg) {
    // qs sits 16 bytes into the block and the row stride is 144, so every
    // 16-byte slice here is aligned - one vector load instead of 16 scalar
    // ones (cold-weight latency was the bottleneck at MoE shapes).
    short is = (il / 4) * 2;
    device const uchar* q = xb->qs + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = sc_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float mn = xb->dmin;
    const float dl = d * sc[0];
    const float ml = mn * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    const uint4 qw = *(device const uint4*)q;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 16; ++i) {
        const uint qb = (qw[i >> 2] >> ((i & 3) * 8)) & 0xFFu;
        reg[i / 4][i % 4] = (half)(dl * (qb & mask) - ml);
    }
}

inline void dqll_q5_k(device const BlkQ5K* xb, short il, thread half4x4& reg) {
    device const uchar* q = xb->qs;
    device const uchar* qh = xb->qh;
    short is = (il / 4) * 2;
    q = q + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    uchar ul = 1 << (il / 2);
    il = il & 3;
    const uchar2 sc = sc_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float mn = xb->dmin;
    const float dl = d * sc[0];
    const float ml = mn * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    const float qh_val = il < 2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml);
    }
}

inline void dqll_q6_k(device const BlkQ6K* xb, short il, thread half4x4& reg) {
    const half d_all = xb->d;
    device const ushort* ql = (device const ushort*)xb->ql;
    device const ushort* qh = (device const ushort*)xb->qh;
    device const char* scales = (device const char*)xb->scales;
    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    float sc = scales[(il % 2) + 2 * (il / 2)];
    il = (il / 2) & 3;
    const uint kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0 : 0x30303030)
                               : (il > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint kmask2 = il > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uchar shr_h = il > 2 ? 2 : 0;
    const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const uchar shr_l = il > 1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint low = (ql[2 * i] | (uint)(ql[2 * i + 1] << 16)) & kmask2;
        const uint high = (qh[2 * i] | (uint)(qh[2 * i + 1] << 16)) & kmask1;
        const uint qv = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (half)(dl0 * ((half)(qv & 0xFF)) - ml);
        reg[i][1] = (half)(dl1 * ((float)(qv & 0xFF00)) - ml);
        reg[i][2] = (half)(dl2 * ((float)(qv & 0xFF0000)) - ml);
        reg[i][3] = (half)(dl3 * ((float)(qv & 0xFF000000)) - ml);
    }
}

// The kernel body. NAME(w, x, y): y[c * n_out + r] += row r of w dot x row c.
// Grid: x over ceil(n_out / 64), y over ceil(m / 32); 128 threads.
// BLOCK_T: the block struct; NL: 16-element chunks per block; ROW_BYTES: the
// row stride expression over n_in.
// NK: the K step per staged tile pass (32 = llama's shape, 64 doubles the
// work per barrier). Generic over NK via whole-chunk indexing: a thread's
// c-th 16-element chunk within the window is il0*(NK/32)+c, and the block
// pointer/sub-index derive from the global chunk counter by division, which
// the compiler folds for constant NL.
#define DEFINE_MM_LL_NK(NAME, BLOCK_T, NL, DQ, ROW_BYTES, NK)                  \
kernel void NAME(                                                              \
    device const uchar* w [[buffer(0)]],                                       \
    device const float* xin [[buffer(1)]],                                     \
    device float* y [[buffer(2)]],                                             \
    constant uint& n_in [[buffer(3)]],                                         \
    constant uint& n_out [[buffer(4)]],                                        \
    constant ulong& w_off [[buffer(5)]],                                       \
    constant uint& m [[buffer(6)]],                                            \
    uint2 tgpig [[threadgroup_position_in_grid]],                              \
    ushort tiitg [[thread_index_in_threadgroup]],                              \
    ushort sgitg [[simdgroup_index_in_threadgroup]])                           \
{                                                                              \
    threadgroup float shmem_f[(48 * NK) < 1536 ? 1536 : (48 * NK)];            \
    threadgroup half* sa = (threadgroup half*)shmem_f;                         \
    threadgroup half* sb = (threadgroup half*)shmem_f + 64 * NK;               \
    const uint r0 = tgpig.x * 64;                                              \
    const uint r1 = tgpig.y * 32;                                              \
    const short nr0 = (n_out - r0 < 64) ? (short)(n_out - r0) : 64;            \
    const short nr1 = (m - r1 < 32) ? (short)(m - r1) : 32;                    \
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1; \
    const short lr1 = ((short)(tiitg / 4) < nr1) ? (short)(tiitg / 4) : nr1 - 1; \
    const short il0 = tiitg % 2;                                               \
    const ulong nb01 = (ulong)(ROW_BYTES);                                     \
    device const BLOCK_T* bx0 =                                                \
        (device const BLOCK_T*)(w + w_off + nb01 * (r0 + lr0));                \
    const short iy = 8 * (tiitg % 4);                                          \
    device const float* by = xin + (ulong)(r1 + lr1) * n_in + iy;              \
    simdgroup_half8x8 ma[4];                                                   \
    simdgroup_half8x8 mb[2];                                                   \
    simdgroup_float8x8 mc[8];                                                  \
    for (short i = 0; i < 8; i++) {                                            \
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);                   \
    }                                                                          \
    uint chunk0 = 0;                                                           \
    for (uint loop_k = 0; loop_k < n_in; loop_k += NK) {                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (short c = 0; c < NK / 32; c++) {                                  \
            half4x4 temp_a;                                                    \
            uint cg = chunk0 + il0 * (NK / 32) + c;                            \
            DQ(bx0 + cg / NL, (short)(cg % NL), temp_a);                       \
            const short kc = il0 * (NK / 32) + c;                              \
            _Pragma("clang loop unroll(full)")                                 \
            for (short i = 0; i < 16; i++) {                                   \
                const short sx = 2 * kc + i / 8;                               \
                const short sy = (tiitg / 2) / 8;                              \
                const short lx = (tiitg / 2) % 8;                              \
                const short ly = i % 8;                                        \
                const short ib = 8 * sx + sy;                                  \
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];          \
            }                                                                  \
        }                                                                      \
        for (short c = 0; c < NK / 32; c++) {                                  \
            const short sx = tiitg % 4 + 4 * c;                                \
            const short sy = (tiitg / 4) / 8;                                  \
            const short ly = (tiitg / 4) % 8;                                  \
            const short ib = 4 * sx + sy;                                      \
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =                   \
                half2x4(*(device const float2x4*)(by + 32 * c));               \
        }                                                                      \
        chunk0 += NK / 16;                                                     \
        by += NK;                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);              \
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);              \
        _Pragma("clang loop unroll(full)")                                     \
        for (short ik = 0; ik < NK / 8; ik++) {                                \
            simdgroup_barrier(mem_flags::mem_none);                            \
            for (short i = 0; i < 4; i++) {                                    \
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);             \
            }                                                                  \
            simdgroup_barrier(mem_flags::mem_none);                            \
            for (short i = 0; i < 2; i++) {                                    \
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);             \
            }                                                                  \
            simdgroup_barrier(mem_flags::mem_none);                            \
            _Pragma("clang loop unroll(full)")                                 \
            for (short i = 0; i < 8; i++) {                                    \
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]); \
            }                                                                  \
            lsma += 8 * 64;                                                    \
            lsmb += 4 * 64;                                                    \
        }                                                                      \
    }                                                                          \
    if (r0 + 64 <= n_out && r1 + 32 <= m) {                                    \
        device float* C = y + (r0 + 32 * (sgitg & 1))                          \
            + (ulong)(r1 + 16 * (sgitg >> 1)) * n_out;                         \
        for (short i = 0; i < 8; i++) {                                        \
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)n_out * (i / 4), \
                            n_out, 0, false);                                  \
        }                                                                      \
    } else {                                                                   \
        threadgroup float* temp_str = shmem_f + sgitg * 64;                    \
        for (short i = 0; i < 8; i++) {                                        \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
            simdgroup_store(mc[i], temp_str, 8, 0, false);                     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
            const short cc0 = 8 * (i % 4) + 32 * (sgitg & 1);                  \
            const short rr0 = 8 * (i / 4) + 16 * (sgitg >> 1);                 \
            for (short e = tiitg % 32; e < 64; e += 32) {                      \
                const short cc = cc0 + (e % 8);                                \
                const short rr = rr0 + (e / 8);                                \
                if (cc < nr0 && rr < nr1) {                                    \
                    y[(ulong)(r1 + rr) * n_out + r0 + cc] = temp_str[e];       \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

#define STAGE_BLOCK_1(b_arg, blk_arg) do {                            \
    const uint stage_b_ = (b_arg);                                    \
    const uint stage_blk_ = (blk_arg);                                \
    const uint b = stage_b_;                                          \
    const uint blk = stage_blk_;                                      \
        threadgroup half* sa = sab[b];                                \
        threadgroup half* sb = sa + 64 * NK;                          \
        uint chunk0 = blk * (NK / 16);                                \
        for (short c = 0; c < NK / 32; c++) {                         \
            half4x4 temp_a;                                           \
            uint cg = chunk0 + il0 * (NK / 32) + c;                   \
            dqll_q4_k(bx0 + cg / 16, (short)(cg % 16), temp_a);       \
            const short kc = il0 * (NK / 32) + c;                     \
            _Pragma("clang loop unroll(full)")                        \
            for (short i = 0; i < 16; i++) {                          \
                const short sx = 2 * kc + i / 8;                      \
                const short sy = (tiitg / 2) / 8;                     \
                const short lx = (tiitg / 2) % 8;                     \
                const short ly = i % 8;                               \
                const short ib = 8 * sx + sy;                         \
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4]; \
            }                                                         \
        }                                                             \
        device const float* byk = by + (ulong)blk * NK;               \
        for (short c = 0; c < NK / 32; c++) {                         \
            const short sx = tiitg % 4 + 4 * c;                       \
            const short sy = (tiitg / 4) / 8;                         \
            const short ly = (tiitg / 4) % 8;                         \
            const short ib = 4 * sx + sy;                             \
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =          \
                half2x4(*(device const float2x4*)(byk + 32 * c));     \
        }                                                             \
} while (0)

// Software-pipelined variant of mmll_q4_k: double-buffered A/B staging.
// Per K-block the loop dequants block i+1 into the shadow buffer while the
// MMA chews block i, so the dequant's device loads hide behind tensor work
// and the loop pays ONE threadgroup barrier per block instead of two (the
// single barrier orders the i+1 writes against the i+1 reads and the i
// reads against the i+2 writes to the same buffer).
kernel void mmllp_q4_k(
    device const uchar* w [[buffer(0)]],
    device const float* xin [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    constant uint& m [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NK = 32;
    threadgroup float shmem_f[3072];
    threadgroup half* sab[2] = {
        (threadgroup half*)shmem_f,
        (threadgroup half*)shmem_f + 96 * NK,
    };
    const uint r0 = tgpig.x * 64;
    const uint r1 = tgpig.y * 32;
    const short nr0 = (n_out - r0 < 64) ? (short)(n_out - r0) : 64;
    const short nr1 = (m - r1 < 32) ? (short)(m - r1) : 32;
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)(tiitg / 4) < nr1) ? (short)(tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const ulong nb01 = (ulong)((n_in / 256) * 144);
    device const BlkQ4K* bx0 =
        (device const BlkQ4K*)(w + w_off + nb01 * (r0 + lr0));
    const short iy = 8 * (tiitg % 4);
    device const float* by = xin + (ulong)(r1 + lr1) * n_in + iy;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    const uint nblk = n_in / NK;
    // Dequant + stage one K-block into buffer set b.
    STAGE_BLOCK_1(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint blk = 0; blk < nblk; blk++) {
        const uint cur = blk & 1;
        if (blk + 1 < nblk) {
            // Shadow-stage the next block; its device loads overlap the MMA.
            STAGE_BLOCK_1(1 - cur, blk + 1);
        }
        threadgroup const half* sa = sab[cur];
        threadgroup const half* sb = sa + 64 * NK;
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        _Pragma("clang loop unroll(full)")
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + 64 <= n_out && r1 + 32 <= m) {
        device float* C = y + (r0 + 32 * (sgitg & 1))
            + (ulong)(r1 + 16 * (sgitg >> 1)) * n_out;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)n_out * (i / 4),
                            n_out, 0, false);
        }
    } else {
        threadgroup float* temp_str = shmem_f + sgitg * 64;
        for (short i = 0; i < 8; i++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(mc[i], temp_str, 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const short cc0 = 8 * (i % 4) + 32 * (sgitg & 1);
            const short rr0 = 8 * (i / 4) + 16 * (sgitg >> 1);
            for (short e = tiitg % 32; e < 64; e += 32) {
                const short cc = cc0 + (e % 8);
                const short rr = rr0 + (e / 8);
                if (cc < nr0 && rr < nr1) {
                    y[(ulong)(r1 + rr) * n_out + r0 + cc] = temp_str[e];
                }
            }
        }
    }
}

#define STAGE_BLOCK_2(b_arg, blk_arg) do {                            \
    const uint stage_b_ = (b_arg);                                    \
    const uint stage_blk_ = (blk_arg);                                \
    const uint b = stage_b_;                                          \
    const uint blk = stage_blk_;                                      \
        threadgroup half* sa = sab[b];                                \
        threadgroup half* sb = sa + 64 * NK;                          \
        uint chunk0 = blk * (NK / 16);                                \
        for (short c = 0; c < NK / 32; c++) {                         \
            half4x4 temp_a;                                           \
            uint cg = chunk0 + il0 * (NK / 32) + c;                   \
            dqll_q8_0(bx0 + cg / 2, (short)(cg % 2), temp_a);         \
            const short kc = il0 * (NK / 32) + c;                     \
            _Pragma("clang loop unroll(full)")                        \
            for (short i = 0; i < 16; i++) {                          \
                const short sx = 2 * kc + i / 8;                      \
                const short sy = (tiitg / 2) / 8;                     \
                const short lx = (tiitg / 2) % 8;                     \
                const short ly = i % 8;                               \
                const short ib = 8 * sx + sy;                         \
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4]; \
            }                                                         \
        }                                                             \
        device const float* byk = by + (ulong)blk * NK;               \
        for (short c = 0; c < NK / 32; c++) {                         \
            const short sx = tiitg % 4 + 4 * c;                       \
            const short sy = (tiitg / 4) / 8;                         \
            const short ly = (tiitg / 4) % 8;                         \
            const short ib = 4 * sx + sy;                             \
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =          \
                half2x4(*(device const float2x4*)(byk + 32 * c));     \
        }                                                             \
} while (0)

// Software-pipelined variant of mmll_q8_0: the same double-buffered staging
// as mmllp_q4_k, format-swapped (32-value blocks, 34 bytes each, NL=2
// 16-element chunks per block). One threadgroup barrier per K-block; the
// i+1 dequant hides behind the block-i MMA.
kernel void mmllp_q8_0(
    device const uchar* w [[buffer(0)]],
    device const float* xin [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    constant uint& m [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NK = 32;
    threadgroup float shmem_f[3072];
    threadgroup half* sab[2] = {
        (threadgroup half*)shmem_f,
        (threadgroup half*)shmem_f + 96 * NK,
    };
    const uint r0 = tgpig.x * 64;
    const uint r1 = tgpig.y * 32;
    const short nr0 = (n_out - r0 < 64) ? (short)(n_out - r0) : 64;
    const short nr1 = (m - r1 < 32) ? (short)(m - r1) : 32;
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)(tiitg / 4) < nr1) ? (short)(tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const ulong nb01 = (ulong)((n_in / 32) * 34);
    device const BlkQ8_0* bx0 =
        (device const BlkQ8_0*)(w + w_off + nb01 * (r0 + lr0));
    const short iy = 8 * (tiitg % 4);
    device const float* by = xin + (ulong)(r1 + lr1) * n_in + iy;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    const uint nblk = n_in / NK;
    // Dequant + stage one K-block into buffer set b.
    STAGE_BLOCK_2(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint blk = 0; blk < nblk; blk++) {
        const uint cur = blk & 1;
        if (blk + 1 < nblk) {
            // Shadow-stage the next block; its device loads overlap the MMA.
            STAGE_BLOCK_2(1 - cur, blk + 1);
        }
        threadgroup const half* sa = sab[cur];
        threadgroup const half* sb = sa + 64 * NK;
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        _Pragma("clang loop unroll(full)")
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + 64 <= n_out && r1 + 32 <= m) {
        device float* C = y + (r0 + 32 * (sgitg & 1))
            + (ulong)(r1 + 16 * (sgitg >> 1)) * n_out;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)n_out * (i / 4),
                            n_out, 0, false);
        }
    } else {
        threadgroup float* temp_str = shmem_f + sgitg * 64;
        for (short i = 0; i < 8; i++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(mc[i], temp_str, 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const short cc0 = 8 * (i % 4) + 32 * (sgitg & 1);
            const short rr0 = 8 * (i / 4) + 16 * (sgitg >> 1);
            for (short e = tiitg % 32; e < 64; e += 32) {
                const short cc = cc0 + (e % 8);
                const short rr = rr0 + (e / 8);
                if (cc < nr0 && rr < nr1) {
                    y[(ulong)(r1 + rr) * n_out + r0 + cc] = temp_str[e];
                }
            }
        }
    }
}

#define STAGE_BLOCK_3(b_arg, blk_arg) do {                            \
    const uint stage_b_ = (b_arg);                                    \
    const uint stage_blk_ = (blk_arg);                                \
    const uint b = stage_b_;                                          \
    const uint blk = stage_blk_;                                      \
        threadgroup half* sa = sab[b];                                \
        threadgroup half* sb = sa + 64 * NK;                          \
        uint chunk0 = blk * (NK / 16);                                \
        for (short c = 0; c < NK / 32; c++) {                         \
            half4x4 temp_a;                                           \
            uint cg = chunk0 + il0 * (NK / 32) + c;                   \
            dqll_q5_k(bx0 + cg / 16, (short)(cg % 16), temp_a);       \
            const short kc = il0 * (NK / 32) + c;                     \
            _Pragma("clang loop unroll(full)")                        \
            for (short i = 0; i < 16; i++) {                          \
                const short sx = 2 * kc + i / 8;                      \
                const short sy = (tiitg / 2) / 8;                     \
                const short lx = (tiitg / 2) % 8;                     \
                const short ly = i % 8;                               \
                const short ib = 8 * sx + sy;                         \
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4]; \
            }                                                         \
        }                                                             \
        device const float* byk = by + (ulong)blk * NK;               \
        for (short c = 0; c < NK / 32; c++) {                         \
            const short sx = tiitg % 4 + 4 * c;                       \
            const short sy = (tiitg / 4) / 8;                         \
            const short ly = (tiitg / 4) % 8;                         \
            const short ib = 4 * sx + sy;                             \
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =          \
                half2x4(*(device const float2x4*)(byk + 32 * c));     \
        }                                                             \
} while (0)

// Software-pipelined variant of mmll_q5_k: same double-buffered staging as
// mmllp_q4_k, Q5_K format (256-value superblocks, 176 bytes, NL=16 chunks).
// Estimates the pipe gain for the 35B's Q5_K down experts.
kernel void mmllp_q5_k(
    device const uchar* w [[buffer(0)]],
    device const float* xin [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    constant uint& m [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NK = 32;
    threadgroup float shmem_f[3072];
    threadgroup half* sab[2] = {
        (threadgroup half*)shmem_f,
        (threadgroup half*)shmem_f + 96 * NK,
    };
    const uint r0 = tgpig.x * 64;
    const uint r1 = tgpig.y * 32;
    const short nr0 = (n_out - r0 < 64) ? (short)(n_out - r0) : 64;
    const short nr1 = (m - r1 < 32) ? (short)(m - r1) : 32;
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)(tiitg / 4) < nr1) ? (short)(tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const ulong nb01 = (ulong)((n_in / 256) * 176);
    device const BlkQ5K* bx0 =
        (device const BlkQ5K*)(w + w_off + nb01 * (r0 + lr0));
    const short iy = 8 * (tiitg % 4);
    device const float* by = xin + (ulong)(r1 + lr1) * n_in + iy;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    const uint nblk = n_in / NK;
    STAGE_BLOCK_3(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint blk = 0; blk < nblk; blk++) {
        const uint cur = blk & 1;
        if (blk + 1 < nblk) {
            STAGE_BLOCK_3(1 - cur, blk + 1);
        }
        threadgroup const half* sa = sab[cur];
        threadgroup const half* sb = sa + 64 * NK;
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        _Pragma("clang loop unroll(full)")
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + 64 <= n_out && r1 + 32 <= m) {
        device float* C = y + (r0 + 32 * (sgitg & 1))
            + (ulong)(r1 + 16 * (sgitg >> 1)) * n_out;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)n_out * (i / 4),
                            n_out, 0, false);
        }
    } else {
        threadgroup float* temp_str = shmem_f + sgitg * 64;
        for (short i = 0; i < 8; i++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(mc[i], temp_str, 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const short cc0 = 8 * (i % 4) + 32 * (sgitg & 1);
            const short rr0 = 8 * (i / 4) + 16 * (sgitg >> 1);
            for (short e = tiitg % 32; e < 64; e += 32) {
                const short cc = cc0 + (e % 8);
                const short rr = rr0 + (e / 8);
                if (cc < nr0 && rr < nr1) {
                    y[(ulong)(r1 + rr) * n_out + r0 + cc] = temp_str[e];
                }
            }
        }
    }
}

// 64-row tile variant of the LL mm: each threadgroup computes 64 out x 64
// rows, so the dequant cost and the two per-iteration barriers amortise
// over twice the FLOPs (A staging is per out-row, barriers per iteration).
// Same K-loop structure otherwise; each simdgroup owns a 32 out x 32 row
// strip (mc[16]). q4_k only - the format the 30B experts/projections use.
// Measured SLOWER than mmll_q4_k on the 30B forms (10.8 vs 11.3-11.9
// TFLOPS): fewer threadgroups loses more than the amortisation wins.
kernel void mmllr64_q4_k(
    device const uchar* w [[buffer(0)]],
    device const float* xin [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    constant uint& m [[buffer(6)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NK = 32;
    threadgroup float shmem_f[2048];
    threadgroup half* sa = (threadgroup half*)shmem_f;              // 64 x NK
    threadgroup half* sb = (threadgroup half*)shmem_f + 64 * NK;    // 64 x NK
    const uint r0 = tgpig.x * 64;
    const uint r1 = tgpig.y * 64;
    const short nr0 = (n_out - r0 < 64) ? (short)(n_out - r0) : 64;
    const short nr1 = (m - r1 < 64) ? (short)(m - r1) : 64;
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1;
    const short il0 = tiitg % 2;
    const ulong nb01 = (ulong)((n_in / 256) * 144);
    device const BlkQ4K* bx0 =
        (device const BlkQ4K*)(w + w_off + nb01 * (r0 + lr0));
    const short iy = 8 * (tiitg % 4);
    device const float* by = xin + (ulong)r1 * n_in + iy;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    uint chunk0 = 0;
    for (uint loop_k = 0; loop_k < n_in; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A staging: identical to the 32-row kernel (64 out rows x NK).
        for (short c = 0; c < NK / 32; c++) {
            half4x4 temp_a;
            uint cg = chunk0 + il0 * (NK / 32) + c;
            dqll_q4_k(bx0 + cg / 16, (short)(cg % 16), temp_a);
            const short kc = il0 * (NK / 32) + c;
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * kc + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }
        // B staging: 64 rows x NK in two 32-row passes.
        for (short rr = 0; rr < 2; rr++) {
            const short lr1 = tiitg / 4 + 32 * rr;
            if (lr1 < nr1) {
                device const float* byr = by + (ulong)lr1 * n_in;
                for (short c = 0; c < NK / 32; c++) {
                    const short sx = tiitg % 4 + 4 * c;
                    const short sy = (tiitg / 4) / 8 + 4 * rr;
                    const short ly = (tiitg / 4) % 8;
                    const short ib = 4 * sx + sy;
                    *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
                        half2x4(*(device const float2x4*)(byr + 32 * c));
                }
            }
        }
        chunk0 += NK / 16;
        by += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Each simdgroup: 32 out strip x 32 row strip.
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 4 * 64 * (sgitg / 2);
        _Pragma("clang loop unroll(full)")
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            for (short i = 0; i < 4; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 16; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 16 * 64;
        }
    }
    if (r0 + 64 <= n_out && r1 + 64 <= m) {
        device float* C = y + (r0 + 32 * (sgitg & 1))
            + (ulong)(r1 + 32 * (sgitg >> 1)) * n_out;
        for (short i = 0; i < 16; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)n_out * (i / 4),
                            n_out, 0, false);
        }
    } else {
        threadgroup float* temp_str = shmem_f + sgitg * 64;
        for (short i = 0; i < 16; i++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(mc[i], temp_str, 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const short cc0 = 8 * (i % 4) + 32 * (sgitg & 1);
            const short rr0 = 8 * (i / 4) + 32 * (sgitg >> 1);
            for (short e = tiitg % 32; e < 64; e += 32) {
                const short cc = cc0 + (e % 8);
                const short rr = rr0 + (e / 8);
                if (cc < nr0 && rr < nr1) {
                    y[(ulong)(r1 + rr) * n_out + r0 + cc] = temp_str[e];
                }
            }
        }
    }
}

DEFINE_MM_LL_NK(mmll_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 32)
DEFINE_MM_LL_NK(mmll_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 32)
DEFINE_MM_LL_NK(mmll_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 32)
DEFINE_MM_LL_NK(mmll_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 32)
DEFINE_MM_LL_NK(mmll_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 32)
DEFINE_MM_LL_NK(mmll_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 32)
DEFINE_MM_LL_NK(mmll_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 32)
DEFINE_MM_LL_NK(mmll_f32, BlkF32, 1, dqll_f32, (ulong)n_in * 4, 32)
DEFINE_MM_LL_NK(mm64ll_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 64)
DEFINE_MM_LL_NK(mm64ll_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 64)
DEFINE_MM_LL_NK(mm64ll_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 64)
DEFINE_MM_LL_NK(mm64ll_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 64)
DEFINE_MM_LL_NK(mm64ll_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 64)
DEFINE_MM_LL_NK(mm64ll_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 64)
DEFINE_MM_LL_NK(mm64ll_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 64)
DEFINE_MM_LL_NK(mm64ll_f32, BlkF32, 1, dqll_f32, (ulong)n_in * 4, 64)

// The grouped (expert-indexed) form of the llama-structure mm: same inner
// tile, plus the MmGroup indirection of DEFINE_MM_ID. Grid z picks the
// group; a threadgroup whose token tile lies past its group's rows exits
// whole before the first barrier (uniform), like the old kernel.
#define DEFINE_MM_ID_LL_NK(NAME, BLOCK_T, NL, DQ, ROW_BYTES, GATHER, NK, SWIGLU, DUAL) \
kernel void NAME(                                                              \
    device const uchar* w [[buffer(0)]],                                       \
    device const float* x [[buffer(1)]],                                       \
    device float* y [[buffer(2)]],                                             \
    constant uint& n_in [[buffer(3)]],                                         \
    constant uint& n_out [[buffer(4)]],                                        \
    constant ulong& w_off [[buffer(5)]],                                       \
    device const MmGroup* table [[buffer(6)]],                                 \
    constant ulong& estride [[buffer(7)]],                                     \
    device const uint* tok [[buffer(8)]],                                      \
    device const float* up_in [[buffer(9)]],                                   \
    device const uchar* w2 [[buffer(10)]],                                     \
    constant ulong& w2_off [[buffer(11)]],                                     \
    constant uint& b_stride [[buffer(12)]],                                    \
    uint3 tg [[threadgroup_position_in_grid]],                                 \
    uint3 tgd [[threadgroups_per_grid]],                                       \
    ushort tiitg [[thread_index_in_threadgroup]],                              \
    ushort sgitg [[simdgroup_index_in_threadgroup]])                           \
{                                                                              \
    MmGroup grp = table[tg.z];                                                 \
    for (uint r1 = tg.y * 32; r1 < grp.rows; r1 += tgd.y * 32) {               \
    threadgroup float shmem_f[(48 * NK) < 1536 ? 1536 : (48 * NK)];            \
    threadgroup half* sa = (threadgroup half*)shmem_f;                         \
    threadgroup half* sb = (threadgroup half*)shmem_f + 64 * NK;               \
    const uint r0 = tg.x * 64;                                                 \
    const uint m = grp.rows;                                                   \
    const uint bstr = SWIGLU ? b_stride : n_in;                                \
    device const float* xin = GATHER ? x : x + (ulong)grp.row0 * bstr;         \
    device float* yg = y + (ulong)grp.row0 * n_out;                            \
    const short nr0 = (n_out - r0 < 64) ? (short)(n_out - r0) : 64;            \
    const short nr1 = (m - r1 < 32) ? (short)(m - r1) : 32;                    \
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1; \
    const short lr1 = ((short)(tiitg / 4) < nr1) ? (short)(tiitg / 4) : nr1 - 1; \
    const short il0 = tiitg % 2;                                               \
    const ulong nb01 = (ulong)(ROW_BYTES);                                     \
    device const uchar* wt = w;                                                \
    ulong wof = w_off;                                                         \
    uint r0l = r0;                                                             \
    if (DUAL && r0 >= n_out / 2) {                                             \
        wt = w2;                                                               \
        wof = w2_off;                                                          \
        r0l = r0 - n_out / 2;                                                  \
    }                                                                          \
    device const BLOCK_T* bx0 = (device const BLOCK_T*)                        \
        (wt + wof + (ulong)grp.expert * estride + nb01 * (r0l + lr0));         \
    const short iy = 8 * (tiitg % 4);                                          \
    const ulong brow = GATHER ? (ulong)tok[grp.row0 + r1 + lr1]                \
                              : (ulong)(r1 + lr1);                             \
    device const float* by = xin + brow * bstr + iy;                           \
    device const float* by2 = up_in;                                           \
    if (SWIGLU) {                                                              \
        by2 = up_in + ((ulong)grp.row0 + brow) * bstr + iy;                    \
    }                                                                          \
    simdgroup_half8x8 ma[4];                                                   \
    simdgroup_half8x8 mb[2];                                                   \
    simdgroup_float8x8 mc[8];                                                  \
    for (short i = 0; i < 8; i++) {                                            \
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);                   \
    }                                                                          \
    uint chunk0 = 0;                                                           \
    for (uint loop_k = 0; loop_k < n_in; loop_k += NK) {                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (short c = 0; c < NK / 32; c++) {                                  \
            half4x4 temp_a;                                                    \
            uint cg = chunk0 + il0 * (NK / 32) + c;                            \
            DQ(bx0 + cg / NL, (short)(cg % NL), temp_a);                       \
            const short kc = il0 * (NK / 32) + c;                              \
            _Pragma("clang loop unroll(full)")                                 \
            for (short i = 0; i < 16; i++) {                                   \
                const short sx = 2 * kc + i / 8;                               \
                const short sy = (tiitg / 2) / 8;                              \
                const short lx = (tiitg / 2) % 8;                              \
                const short ly = i % 8;                                        \
                const short ib = 8 * sx + sy;                                  \
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];          \
            }                                                                  \
        }                                                                      \
        for (short c = 0; c < NK / 32; c++) {                                  \
            const short sx = tiitg % 4 + 4 * c;                                \
            const short sy = (tiitg / 4) / 8;                                  \
            const short ly = (tiitg / 4) % 8;                                  \
            const short ib = 4 * sx + sy;                                      \
            if (SWIGLU) {                                                      \
                float2x4 gv = *(device const float2x4*)(by + 32 * c);          \
                float2x4 uv = *(device const float2x4*)(by2 + 32 * c);         \
                half2x4 hv;                                                    \
                _Pragma("clang loop unroll(full)")                             \
                for (short q = 0; q < 2; q++) {                                \
                    float4 g = gv[q];                                          \
                    float4 u = uv[q];                                          \
                    hv[q] = half4(g / (1.0f + exp(-g)) * u);                   \
                }                                                              \
                *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = hv;           \
            } else {                                                           \
                *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =               \
                    half2x4(*(device const float2x4*)(by + 32 * c));           \
            }                                                                  \
        }                                                                      \
        chunk0 += NK / 16;                                                     \
        by += NK;                                                              \
        if (SWIGLU) {                                                          \
            by2 += NK;                                                         \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);              \
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);              \
        _Pragma("clang loop unroll(full)")                                     \
        for (short ik = 0; ik < NK / 8; ik++) {                                \
            simdgroup_barrier(mem_flags::mem_none);                            \
            for (short i = 0; i < 4; i++) {                                    \
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);             \
            }                                                                  \
            simdgroup_barrier(mem_flags::mem_none);                            \
            for (short i = 0; i < 2; i++) {                                    \
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);             \
            }                                                                  \
            simdgroup_barrier(mem_flags::mem_none);                            \
            _Pragma("clang loop unroll(full)")                                 \
            for (short i = 0; i < 8; i++) {                                    \
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]); \
            }                                                                  \
            lsma += 8 * 64;                                                    \
            lsmb += 4 * 64;                                                    \
        }                                                                      \
    }                                                                          \
    if (r0 + 64 <= n_out && r1 + 32 <= m) {                                    \
        device float* C = yg + (r0 + 32 * (sgitg & 1))                         \
            + (ulong)(r1 + 16 * (sgitg >> 1)) * n_out;                         \
        for (short i = 0; i < 8; i++) {                                        \
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)n_out * (i / 4), \
                            n_out, 0, false);                                  \
        }                                                                      \
    } else {                                                                   \
        threadgroup float* temp_str = shmem_f + sgitg * 64;                    \
        for (short i = 0; i < 8; i++) {                                        \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
            simdgroup_store(mc[i], temp_str, 8, 0, false);                     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
            const short cc0 = 8 * (i % 4) + 32 * (sgitg & 1);                  \
            const short rr0 = 8 * (i / 4) + 16 * (sgitg >> 1);                 \
            for (short e = tiitg % 32; e < 64; e += 32) {                      \
                const short cc = cc0 + (e % 8);                                \
                const short rr = rr0 + (e / 8);                                \
                if (cc < nr0 && rr < nr1) {                                    \
                    yg[(ulong)(r1 + rr) * n_out + r0 + cc] = temp_str[e];      \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    }                                                                          \
}

DEFINE_MM_ID_LL_NK(mmll_id_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 0, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmllg_id_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 1, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmll_id_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 0, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmllg_id_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 1, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmll_id_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 0, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmllg_id_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 1, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmll_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 0, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmllg_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 1, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmll_id_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 0, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmllg_id_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 1, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmll_id_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 0, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmllg_id_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 1, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mm64llg_id_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 1, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64llg_id_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 1, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64llg_id_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 1, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64llg_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 1, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64llg_id_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 1, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64llg_id_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 1, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64ll_id_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 0, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64ll_id_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 0, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64ll_id_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 0, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64ll_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 0, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64ll_id_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 0, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64ll_id_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 0, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mmll_id_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 0, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mmllg_id_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 1, 32, 0, 0)
DEFINE_MM_ID_LL_NK(mm64ll_id_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 0, 64, 0, 0)
DEFINE_MM_ID_LL_NK(mm64llg_id_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 1, 64, 0, 0)

#define STAGE_BLOCK_4(b_arg, blk_arg, swiglu) do {                              \
    const uint stage_b_ = (b_arg);                                      \
    const uint stage_blk_ = (blk_arg);                                  \
    const uint b = stage_b_;                                            \
    const uint blk = stage_blk_;                                        \
        threadgroup half* sa = sab[b];                                  \
        threadgroup half* sb = sa + 64 * NK;                            \
        uint chunk0 = blk * (NK / 16);                                  \
        for (short c = 0; c < NK / 32; c++) {                           \
            half4x4 temp_a;                                             \
            uint cg = chunk0 + il0 * (NK / 32) + c;                     \
            dqll_q4_k(bx0 + cg / 16, (short)(cg % 16), temp_a);         \
            const short kc = il0 * (NK / 32) + c;                       \
            _Pragma("clang loop unroll(full)")                          \
            for (short i = 0; i < 16; i++) {                            \
                const short sx = 2 * kc + i / 8;                        \
                const short sy = (tiitg / 2) / 8;                       \
                const short lx = (tiitg / 2) % 8;                       \
                const short ly = i % 8;                                 \
                const short ib = 8 * sx + sy;                           \
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];   \
            }                                                           \
        }                                                               \
        device const float* byk = by + (ulong)blk * NK;                 \
        device const float* by2k = by2 + (ulong)blk * NK;               \
        for (short c = 0; c < NK / 32; c++) {                           \
            const short sx = tiitg % 4 + 4 * c;                         \
            const short sy = (tiitg / 4) / 8;                           \
            const short ly = (tiitg / 4) % 8;                           \
            const short ib = 4 * sx + sy;                               \
            if (swiglu) {                                                 \
                float2x4 gv = *(device const float2x4*)(byk + 32 * c);  \
                float2x4 uv = *(device const float2x4*)(by2k + 32 * c); \
                half2x4 hv;                                             \
                _Pragma("clang loop unroll(full)")                      \
                for (short q = 0; q < 2; q++) {                         \
                    float4 g = gv[q];                                   \
                    float4 u = uv[q];                                   \
                    hv[q] = half4(g / (1.0f + exp(-g)) * u);            \
                }                                                       \
                *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) = hv;    \
            } else {                                                    \
                *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =        \
                    half2x4(*(device const float2x4*)(byk + 32 * c));   \
            }                                                           \
        }                                                               \
} while (0)

// Software-pipelined grouped mm (q4_k): same double-buffered K-loop as
// mmllp_q4_k - dequant/stage block i+1 into the shadow set while the MMA
// chews block i, one threadgroup barrier per block instead of two.
#define DEFINE_MM_ID_PIPE_Q4K(NAME, GATHER, SWIGLU)                            \
kernel void NAME(                                                              \
    device const uchar* w [[buffer(0)]],                                       \
    device const float* x [[buffer(1)]],                                       \
    device float* y [[buffer(2)]],                                             \
    constant uint& n_in [[buffer(3)]],                                         \
    constant uint& n_out [[buffer(4)]],                                        \
    constant ulong& w_off [[buffer(5)]],                                       \
    device const MmGroup* table [[buffer(6)]],                                 \
    constant ulong& estride [[buffer(7)]],                                     \
    device const uint* tok [[buffer(8)]],                                      \
    device const float* up_in [[buffer(9)]],                                   \
    device const uchar* w2 [[buffer(10)]],                                     \
    constant ulong& w2_off [[buffer(11)]],                                     \
    constant uint& b_stride [[buffer(12)]],                                    \
    uint3 tg [[threadgroup_position_in_grid]],                                 \
    uint3 tgd [[threadgroups_per_grid]],                                       \
    ushort tiitg [[thread_index_in_threadgroup]],                              \
    ushort sgitg [[simdgroup_index_in_threadgroup]])                           \
{                                                                              \
    MmGroup grp = table[tg.z];                                                 \
    for (uint r1 = tg.y * 32; r1 < grp.rows; r1 += tgd.y * 32) {               \
    constexpr short NK = 32;                                                   \
    threadgroup float shmem_f[3072];                                           \
    threadgroup half* sab[2] = {                                               \
        (threadgroup half*)shmem_f,                                            \
        (threadgroup half*)shmem_f + 96 * NK,                                  \
    };                                                                         \
    const uint r0 = tg.x * 64;                                                 \
    const uint m = grp.rows;                                                   \
    const uint bstr = SWIGLU ? b_stride : n_in;                                \
    device const float* xin = GATHER ? x : x + (ulong)grp.row0 * bstr;         \
    device float* yg = y + (ulong)grp.row0 * n_out;                            \
    const short nr0 = (n_out - r0 < 64) ? (short)(n_out - r0) : 64;            \
    const short nr1 = (m - r1 < 32) ? (short)(m - r1) : 32;                    \
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1; \
    const short lr1 = ((short)(tiitg / 4) < nr1) ? (short)(tiitg / 4) : nr1 - 1; \
    const short il0 = tiitg % 2;                                               \
    const ulong nb01 = (ulong)((n_in / 256) * 144);                            \
    device const BlkQ4K* bx0 = (device const BlkQ4K*)                          \
        (w + w_off + (ulong)grp.expert * estride + nb01 * (r0 + lr0));         \
    const short iy = 8 * (tiitg % 4);                                          \
    const ulong brow = GATHER ? (ulong)tok[grp.row0 + r1 + lr1]                \
                              : (ulong)(r1 + lr1);                             \
    device const float* by = xin + brow * bstr + iy;                           \
    device const float* by2 = up_in;                                           \
    if (SWIGLU) {                                                              \
        by2 = up_in + ((ulong)grp.row0 + brow) * bstr + iy;                    \
    }                                                                          \
    simdgroup_half8x8 ma[4];                                                   \
    simdgroup_half8x8 mb[2];                                                   \
    simdgroup_float8x8 mc[8];                                                  \
    for (short i = 0; i < 8; i++) {                                            \
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);                   \
    }                                                                          \
    const uint nblk = n_in / NK;                                               \
    STAGE_BLOCK_4(0, 0, SWIGLU);                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint blk = 0; blk < nblk; blk++) {                                    \
        const uint cur = blk & 1;                                              \
        if (blk + 1 < nblk) {                                                  \
            STAGE_BLOCK_4(1 - cur, blk + 1, SWIGLU);                           \
        }                                                                      \
        threadgroup const half* sa = sab[cur];                                 \
        threadgroup const half* sb = sa + 64 * NK;                             \
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);              \
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);              \
        _Pragma("clang loop unroll(full)")                                     \
        for (short ik = 0; ik < NK / 8; ik++) {                                \
            simdgroup_barrier(mem_flags::mem_none);                            \
            for (short i = 0; i < 4; i++) {                                    \
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);             \
            }                                                                  \
            simdgroup_barrier(mem_flags::mem_none);                            \
            for (short i = 0; i < 2; i++) {                                    \
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);             \
            }                                                                  \
            simdgroup_barrier(mem_flags::mem_none);                            \
            _Pragma("clang loop unroll(full)")                                 \
            for (short i = 0; i < 8; i++) {                                    \
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]); \
            }                                                                  \
            lsma += 8 * 64;                                                    \
            lsmb += 4 * 64;                                                    \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    if (r0 + 64 <= n_out && r1 + 32 <= m) {                                    \
        device float* C = yg + (r0 + 32 * (sgitg & 1))                         \
            + (ulong)(r1 + 16 * (sgitg >> 1)) * n_out;                         \
        for (short i = 0; i < 8; i++) {                                        \
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)n_out * (i / 4), \
                            n_out, 0, false);                                  \
        }                                                                      \
    } else {                                                                   \
        threadgroup float* temp_str = shmem_f + sgitg * 64;                    \
        for (short i = 0; i < 8; i++) {                                        \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
            simdgroup_store(mc[i], temp_str, 8, 0, false);                     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
            const short cc0 = 8 * (i % 4) + 32 * (sgitg & 1);                  \
            const short rr0 = 8 * (i / 4) + 16 * (sgitg >> 1);                 \
            for (short e = tiitg % 32; e < 64; e += 32) {                      \
                const short cc = cc0 + (e % 8);                                \
                const short rr = rr0 + (e / 8);                                \
                if (cc < nr0 && rr < nr1) {                                    \
                    yg[(ulong)(r1 + rr) * n_out + r0 + cc] = temp_str[e];      \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    }                                                                          \
}

DEFINE_MM_ID_PIPE_Q4K(mmllpg_id_q4_k, 1, 0)
DEFINE_MM_ID_PIPE_Q4K(mmllps_id_q4_k, 0, 1)
// Down-projection with swiglu folded into the B staging: gate and up stay
// raw in y, the mm applies silu(g)*u while loading activations. Removes the
// standalone swiglu pass and one full-device barrier per MoE layer.
DEFINE_MM_ID_LL_NK(mmlls_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 0, 32, 1, 0)
// Fused gate+up in ONE dispatch: out cols [0, n_out/2) read expert weights
// from w (gate), the upper half from w2 (up) - one B staging per K step
// instead of two dispatches each re-reading the activations.
DEFINE_MM_ID_LL_NK(mmllgd_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 1, 32, 0, 1)

// Per-head RMSNorm + NEOX RoPE for freshly projected q/k, and the f16 store
// of k and v into the cache - everything decode needs between the qkv matmul
// and attention, so all of it can sit in the same command buffer.
//
// One SIMD group per head. A lane owns two NEOX pairs (p, p+32), which
// covers the 128-wide head exactly: elements p, p+64, p+32, p+96. The mean
// square folds with simd_sum. Heads are q first, then k (normed, roped,
// stored to cache), then v (stored only).
struct QkPrepArgs {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint kv_dim;
    float eps;
    uint pos;
    uint has_qk_norm;
    uint rot_dim;   // rotary width; == head_dim for full rope
    ulong k_base;
    ulong v_base;
    uint has_bias;
    uint pad2;
};

// Rotary for one head already loaded as e[0..3] = elements lane, lane+32,
// lane+64, lane+96. Full rotary (rot_h == 64): pairs (lane, lane+64) and
// (lane+32, lane+96). Partial (rot_h == 32): pair (lane, lane+32) only,
// the upper half of the head passes through. The Rust side rejects other
// rotary widths.
#define QK_PREP_ROPE(ROT_H)                                                      \
    if (ROT_H == 64) {                                                           \
        float2 r0 = rope[lane];                                                  \
        float2 r1 = rope[lane + 32];                                             \
        float t0 = e0 * r0.y - e2 * r0.x;                                        \
        float t2 = e0 * r0.x + e2 * r0.y;                                        \
        float t1 = e1 * r1.y - e3 * r1.x;                                        \
        float t3 = e1 * r1.x + e3 * r1.y;                                        \
        e0 = t0; e2 = t2; e1 = t1; e3 = t3;                                      \
    } else {                                                                     \
        float2 r0 = rope[lane];                                                  \
        float t0 = e0 * r0.y - e1 * r0.x;                                        \
        float t1 = e0 * r0.x + e1 * r0.y;                                        \
        e0 = t0; e1 = t1;                                                        \
    }

kernel void qk_prep(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device half* cache [[buffer(3)]],
    constant QkPrepArgs& a [[buffer(4)]],
    constant float* qw [[buffer(5)]],
    constant float* kw [[buffer(6)]],
    constant float2* rope [[buffer(7)]],
    constant float* qb [[buffer(8)]],
    constant float* kb [[buffer(9)]],
    constant float* vb [[buffer(10)]],
    uint hid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint hd = a.head_dim;
    uint rot_h = a.rot_dim / 2;
    if (hid >= a.n_heads + a.n_kv_heads) {
        // v head: optional bias, narrow to half and store.
        uint kvh = hid - a.n_heads - a.n_kv_heads;
        device const float* src = v + kvh * hd;
        device half* dst = cache + a.v_base + (ulong)a.pos * a.kv_dim + kvh * hd;
        for (uint e = lane; e < hd; e += 32) {
            float val = src[e];
            if (a.has_bias != 0) val += vb[kvh * hd + e];
            dst[e] = (half)val;
        }
        return;
    }
    bool is_q = hid < a.n_heads;
    device float* xh = is_q ? (q + hid * hd) : (k + (hid - a.n_heads) * hd);
    constant float* w = is_q ? qw : kw;
    constant float* bias = is_q ? qb : kb;
    uint boff = is_q ? hid * hd : (hid - a.n_heads) * hd;
    float e0 = xh[lane];
    float e1 = xh[lane + 32];
    float e2 = xh[lane + 64];
    float e3 = xh[lane + 96];
    if (a.has_bias != 0) {
        e0 += bias[boff + lane];
        e1 += bias[boff + lane + 32];
        e2 += bias[boff + lane + 64];
        e3 += bias[boff + lane + 96];
    }
    if (a.has_qk_norm != 0) {
        float ms = simd_sum(e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3) / (float)hd;
        float scale = rsqrt(ms + a.eps);
        e0 *= scale * w[lane];
        e1 *= scale * w[lane + 32];
        e2 *= scale * w[lane + 64];
        e3 *= scale * w[lane + 96];
    }
    QK_PREP_ROPE(rot_h)
    if (is_q) {
        xh[lane] = e0;
        xh[lane + 32] = e1;
        xh[lane + 64] = e2;
        xh[lane + 96] = e3;
    } else {
        uint kvh = hid - a.n_heads;
        device half* dst = cache + a.k_base + (ulong)a.pos * a.kv_dim + kvh * hd;
        dst[lane] = (half)e0;
        dst[lane + 32] = (half)e1;
        dst[lane + 64] = (half)e2;
        dst[lane + 96] = (half)e3;
    }
}

// The batched form of qk_prep for prefill: one threadgroup per (head, row),
// each row at its own position with its own rope table. Same math as the
// single-token kernel; the batch dimension only moves base pointers.
kernel void qk_prep_batch(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device half* cache [[buffer(3)]],
    constant QkPrepArgs& a [[buffer(4)]],
    constant float* qw [[buffer(5)]],
    constant float* kw [[buffer(6)]],
    device const float2* ropes [[buffer(7)]],
    constant float* qb [[buffer(8)]],
    constant float* kb [[buffer(9)]],
    constant float* vb [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint hid = tg.x;
    uint row = tg.y;
    uint hd = a.head_dim;
    uint rot_h = a.rot_dim / 2;
    uint q_dim = a.n_heads * hd;
    uint pos = a.pos + row; // a.pos carries the chunk base
    device const float2* rope = ropes + row * rot_h;
    if (hid >= a.n_heads + a.n_kv_heads) {
        uint kvh = hid - a.n_heads - a.n_kv_heads;
        device const float* src = v + row * a.kv_dim + kvh * hd;
        device half* dst = cache + a.v_base + (ulong)pos * a.kv_dim + kvh * hd;
        for (uint e = lane; e < hd; e += 32) {
            float val = src[e];
            if (a.has_bias != 0) val += vb[kvh * hd + e];
            dst[e] = (half)val;
        }
        return;
    }
    bool is_q = hid < a.n_heads;
    device float* xh = is_q
        ? (q + row * q_dim + hid * hd)
        : (k + row * a.kv_dim + (hid - a.n_heads) * hd);
    constant float* w = is_q ? qw : kw;
    constant float* bias = is_q ? qb : kb;
    uint boff = is_q ? hid * hd : (hid - a.n_heads) * hd;
    float e0 = xh[lane];
    float e1 = xh[lane + 32];
    float e2 = xh[lane + 64];
    float e3 = xh[lane + 96];
    if (a.has_bias != 0) {
        e0 += bias[boff + lane];
        e1 += bias[boff + lane + 32];
        e2 += bias[boff + lane + 64];
        e3 += bias[boff + lane + 96];
    }
    if (a.has_qk_norm != 0) {
        float ms = simd_sum(e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3) / (float)hd;
        float scale = rsqrt(ms + a.eps);
        e0 *= scale * w[lane];
        e1 *= scale * w[lane + 32];
        e2 *= scale * w[lane + 64];
        e3 *= scale * w[lane + 96];
    }
    QK_PREP_ROPE(rot_h)
    if (is_q) {
        xh[lane] = e0;
        xh[lane + 32] = e1;
        xh[lane + 64] = e2;
        xh[lane + 96] = e3;
    } else {
        uint kvh = hid - a.n_heads;
        device half* dst = cache + a.k_base + (ulong)pos * a.kv_dim + kvh * hd;
        dst[lane] = (half)e0;
        dst[lane + 32] = (half)e1;
        dst[lane + 64] = (half)e2;
        dst[lane + 96] = (half)e3;
    }
}

// The grouped tile-matmul: one dispatch runs EVERY expert group of a MoE
// prefill stage. Grid: (col tiles, row tiles up to the largest group,
// groups). A group table maps each z-slice to its expert's weights and its
// contiguous rows in the gathered activation buffer; threadgroups whose row
// tile lies past their group's rows exit whole (the guard is uniform per
// threadgroup, so the barriers inside never diverge).
//
// This replaces ~400 dispatches per layer-chunk with 3 - the driver's
// per-dispatch scheduling of those was measured at 3.4 s of an 8.8 s 235B
// prefill.
#define DEFINE_MM_ID(NAME, DQ16, ROW_BYTES)                                    \
kernel void NAME(                                                              \
    device const uchar* w [[buffer(0)]],                                       \
    device const float* x [[buffer(1)]],                                       \
    device float* y [[buffer(2)]],                                             \
    constant uint& n_in [[buffer(3)]],                                         \
    constant uint& n_out [[buffer(4)]],                                        \
    constant ulong& w_off [[buffer(5)]],                                       \
    device const MmGroup* table [[buffer(6)]],                                 \
    constant ulong& estride [[buffer(7)]],                                     \
    uint3 tg [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]],                                  \
    uint sg [[simdgroup_index_in_threadgroup]])                                \
{                                                                              \
    const uint BM = 32, BN = 32, BK = 32;                                      \
    MmGroup grp = table[tg.z];                                                 \
    uint j0 = tg.y * BN;                                                       \
    if (j0 >= grp.rows) {                                                      \
        return;                                                                \
    }                                                                          \
    uint i0 = tg.x * BM;                                                       \
    device const uchar* we = w + w_off + (ulong)grp.expert * estride;          \
    device const float* xg = x + (ulong)grp.row0 * n_in;                       \
    device float* yg = y + (ulong)grp.row0 * n_out;                            \
    threadgroup half As[BM * BK];                                              \
    threadgroup half Bs[BK * BN];                                              \
    threadgroup float Cs[BM * BN];                                             \
    simdgroup_float8x8 c[4];                                                   \
    for (uint t = 0; t < 4; t++) {                                             \
        c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);                \
    }                                                                          \
    for (uint k0 = 0; k0 < n_in; k0 += BK) {                                   \
        for (uint idx = tid; idx < BM * BK / 16; idx += 128) {                 \
            uint row = idx / 2;                                                \
            uint part = idx % 2;                                               \
            uint ri = min(i0 + row, n_out - 1);                                \
            device const uchar* rp = we + (ulong)ri * (ROW_BYTES);             \
            DQ16(rp, k0 + part * 16, As + row * BK + part * 16);               \
        }                                                                      \
        for (uint idx = tid; idx < BN * (BK / 4); idx += 128) {                \
            uint j = idx % BN;                                                 \
            uint k4 = idx / BN;                                                \
            uint jj = j0 + j;                                                  \
            float4 xv = (jj < grp.rows)                                        \
                ? *(device const float4*)(xg + (ulong)jj * n_in + k0 + k4 * 4) \
                : float4(0.0f);                                                \
            Bs[(k4 * 4 + 0) * BN + j] = (half)xv.x;                            \
            Bs[(k4 * 4 + 1) * BN + j] = (half)xv.y;                            \
            Bs[(k4 * 4 + 2) * BN + j] = (half)xv.z;                            \
            Bs[(k4 * 4 + 3) * BN + j] = (half)xv.w;                            \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (uint kk = 0; kk < BK; kk += 8) {                                  \
            simdgroup_half8x8 a;                                               \
            simdgroup_load(a, As + sg * 8 * BK + kk, BK);                      \
            for (uint t = 0; t < 4; t++) {                                     \
                simdgroup_half8x8 b;                                           \
                simdgroup_load(b, Bs + kk * BN + t * 8, BN);                   \
                simdgroup_multiply_accumulate(c[t], a, b, c[t]);               \
            }                                                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    for (uint t = 0; t < 4; t++) {                                             \
        simdgroup_store(c[t], Cs + sg * 8 * BN + t * 8, BN);                   \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint idx = tid; idx < BM * BN; idx += 128) {                          \
        uint i = i0 + idx / BN;                                                \
        uint j = j0 + idx % BN;                                                \
        if (i < n_out && j < grp.rows) {                                       \
            yg[(ulong)j * n_out + i] = Cs[idx];                                \
        }                                                                      \
    }                                                                          \
}

DEFINE_MM_ID(mmid_q8_0, dq16_q8_0, (n_in / 32) * 34)
DEFINE_MM_ID(mmid_q2_k, dq16_q2_k, (n_in / 256) * 84)
DEFINE_MM_ID(mmid_q3_k, dq16_q3_k, (n_in / 256) * 110)
DEFINE_MM_ID(mmid_q4_k, dq16_q4_k, (n_in / 256) * 144)
DEFINE_MM_ID(mmid_q5_k, dq16_q5_k, (n_in / 256) * 176)
DEFINE_MM_ID(mmid_q6_k, dq16_q6_k, (n_in / 256) * 210)

// Decode attention for one query head, flash style: never materialise the
// score vector, keep a running max and denominator and rescale the
// accumulator when a new maximum arrives.
//
// The cache is half precision and the offsets into it are element indices,
// not buffer offsets: binding at an offset would drag in Metal's alignment
// rules for a number that depends on the session's context capacity.
//
// One threadgroup per query head, four SIMD groups inside it, and each SIMD
// group walks every fourth cached position. head_dim is 128, so a lane owns
// exactly four output dimensions and the whole accumulator lives in
// registers: no threadgroup traffic until the four groups merge at the end.
//
// The merge is the standard one: with per-group maxima m_s and sums d_s, the
// combined result is sum_s(acc_s * exp(m_s - M)) / sum_s(d_s * exp(m_s - M)).
constant constexpr uint ATTN_SIMDS = 4;
constant constexpr uint ATTN_LANE_DIMS = 4;

kernel void attend(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    uint qh [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[ATTN_SIMDS * 128];
    threadgroup float t_max[ATTN_SIMDS];
    threadgroup float t_den[ATTN_SIMDS];

    uint kv_head = qh / group;
    device const float4* qh4 =
        (device const float4*)(q + qh * head_dim);
    float4 qv = qh4[lane];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc = float4(0.0f);

    for (uint t = sg; t < n_pos; t += ATTN_SIMDS) {
        ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        float4 kv4 = float4(((device const half4*)(k + k_base + at))[lane]);
        float s = simd_sum(dot(qv, kv4)) * scale;
        if (s > run_max) {
            float r = exp(run_max - s);
            // The first position starts from -inf, where the rescale is 0/0;
            // the denominator being zero is the marker for that.
            if (run_den > 0.0f) {
                acc *= r;
                run_den *= r;
            } else {
                acc = float4(0.0f);
            }
            run_max = s;
        }
        float p = exp(s - run_max);
        run_den += p;
        float4 vv4 = float4(((device const half4*)(v + v_base + at))[lane]);
        acc += p * vv4;
    }

    ((threadgroup float4*)(t_acc + sg * 128))[lane] = acc;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < ATTN_SIMDS; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < ATTN_SIMDS; s++) {
            float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 128 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        out[qh * head_dim + tid] = num / max(den, FLT_MIN);
    }
}

kernel void attend_s8(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    uint qh [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[8 * 128];
    threadgroup float t_max[8];
    threadgroup float t_den[8];

    uint kv_head = qh / group;
    device const float4* qh4 =
        (device const float4*)(q + qh * head_dim);
    float4 qv = qh4[lane];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc = float4(0.0f);

    for (uint t = sg; t < n_pos; t += 8u) {
        ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        float4 kv4 = float4(((device const half4*)(k + k_base + at))[lane]);
        float s = simd_sum(dot(qv, kv4)) * scale;
        if (s > run_max) {
            float r = exp(run_max - s);
            // The first position starts from -inf, where the rescale is 0/0;
            // the denominator being zero is the marker for that.
            if (run_den > 0.0f) {
                acc *= r;
                run_den *= r;
            } else {
                acc = float4(0.0f);
            }
            run_max = s;
        }
        float p = exp(s - run_max);
        run_den += p;
        float4 vv4 = float4(((device const half4*)(v + v_base + at))[lane]);
        acc += p * vv4;
    }

    ((threadgroup float4*)(t_acc + sg * 128))[lane] = acc;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < 8u; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < 8u; s++) {
            float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 128 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        out[qh * head_dim + tid] = num / max(den, FLT_MIN);
    }
}

kernel void attend_s16(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    uint qh [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[16 * 128];
    threadgroup float t_max[16];
    threadgroup float t_den[16];

    uint kv_head = qh / group;
    device const float4* qh4 =
        (device const float4*)(q + qh * head_dim);
    float4 qv = qh4[lane];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc = float4(0.0f);

    for (uint t = sg; t < n_pos; t += 16u) {
        ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        float4 kv4 = float4(((device const half4*)(k + k_base + at))[lane]);
        float s = simd_sum(dot(qv, kv4)) * scale;
        if (s > run_max) {
            float r = exp(run_max - s);
            // The first position starts from -inf, where the rescale is 0/0;
            // the denominator being zero is the marker for that.
            if (run_den > 0.0f) {
                acc *= r;
                run_den *= r;
            } else {
                acc = float4(0.0f);
            }
            run_max = s;
        }
        float p = exp(s - run_max);
        run_den += p;
        float4 vv4 = float4(((device const half4*)(v + v_base + at))[lane]);
        acc += p * vv4;
    }

    ((threadgroup float4*)(t_acc + sg * 128))[lane] = acc;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < 16u; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < 16u; s++) {
            float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 128 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        out[qh * head_dim + tid] = num / max(den, FLT_MIN);
    }
}

kernel void attend_s32(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    uint qh [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[32 * 128];
    threadgroup float t_max[32];
    threadgroup float t_den[32];

    uint kv_head = qh / group;
    device const float4* qh4 =
        (device const float4*)(q + qh * head_dim);
    float4 qv = qh4[lane];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc = float4(0.0f);

    for (uint t = sg; t < n_pos; t += 32u) {
        ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        float4 kv4 = float4(((device const half4*)(k + k_base + at))[lane]);
        float s = simd_sum(dot(qv, kv4)) * scale;
        if (s > run_max) {
            float r = exp(run_max - s);
            // The first position starts from -inf, where the rescale is 0/0;
            // the denominator being zero is the marker for that.
            if (run_den > 0.0f) {
                acc *= r;
                run_den *= r;
            } else {
                acc = float4(0.0f);
            }
            run_max = s;
        }
        float p = exp(s - run_max);
        run_den += p;
        float4 vv4 = float4(((device const half4*)(v + v_base + at))[lane]);
        acc += p * vv4;
    }

    ((threadgroup float4*)(t_acc + sg * 128))[lane] = acc;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < 32u; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < 32u; s++) {
            float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 128 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        out[qh * head_dim + tid] = num / max(den, FLT_MIN);
    }
}

// Decode attention in llama.cpp's flash_attn_ext_vec structure (MIT): each
// SIMD group keeps FOUR positions in flight (ty picks the position, the 8
// tx lanes split the 128-wide head), the Q*K dot reduces over 8 lanes in 3
// shuffles instead of a full 32-lane simd_sum per position, and the online
// softmax rescales the accumulator once per 4-position block instead of on
// every new maximum. Bindings and grid identical to `attend` above; the
// cross-SIMD-group merge at the end is the same code.
kernel void attend_mv(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    uint qh [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[ATTN_SIMDS * 128];
    threadgroup float t_max[ATTN_SIMDS];
    threadgroup float t_den[ATTN_SIMDS];

    const uint tx = lane % 8;
    const uint ty = lane / 8;

    uint kv_head = qh / group;
    device const float4* qh4 = (device const float4*)(q + qh * head_dim);
    float4 qs[4];
    for (uint ii = 0; ii < 4; ii++) {
        qs[ii] = qh4[ii * 8 + tx];
    }

    float M = -3.0e38f / 2;
    float S = 0.0f;
    float4 lo[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};

    for (uint t0 = sg * 4; t0 < n_pos; t0 += ATTN_SIMDS * 4) {
        uint t = t0 + ty;
        bool valid = t < n_pos;
        uint tc = valid ? t : n_pos - 1;
        ulong at = (ulong)tc * kv_dim + (ulong)kv_head * head_dim;
        device const half4* kr = (device const half4*)(k + k_base + at);
        float partial = 0.0f;
        for (uint ii = 0; ii < 4; ii++) {
            partial += dot(qs[ii], float4(kr[ii * 8 + tx]));
        }
        partial += simd_shuffle_down(partial, 4);
        partial += simd_shuffle_down(partial, 2);
        partial += simd_shuffle_down(partial, 1);
        // Scores of the 4 in-flight positions, visible to every lane.
        float s0 = simd_shuffle(partial, 0) * scale;
        float s1 = simd_shuffle(partial, 8) * scale;
        float s2 = simd_shuffle(partial, 16) * scale;
        float s3 = simd_shuffle(partial, 24) * scale;
        if (t0 + 0 >= n_pos) s0 = -INFINITY;
        if (t0 + 1 >= n_pos) s1 = -INFINITY;
        if (t0 + 2 >= n_pos) s2 = -INFINITY;
        if (t0 + 3 >= n_pos) s3 = -INFINITY;

        float Mnew = max(M, max(max(s0, s1), max(s2, s3)));
        float ms = exp(M - Mnew);
        float vs0 = exp(s0 - Mnew);
        float vs1 = exp(s1 - Mnew);
        float vs2 = exp(s2 - Mnew);
        float vs3 = exp(s3 - Mnew);
        M = Mnew;
        S = S * ms + vs0 + vs1 + vs2 + vs3;

        float vsy = ty == 0 ? vs0 : (ty == 1 ? vs1 : (ty == 2 ? vs2 : vs3));
        device const half4* vr = (device const half4*)(v + v_base + at);
        for (uint ii = 0; ii < 4; ii++) {
            lo[ii] = lo[ii] * ms + vsy * float4(vr[ii * 8 + tx]);
        }
    }

    // Fold the four position streams: same tx, ty 1..3 into ty 0.
    for (uint ii = 0; ii < 4; ii++) {
        lo[ii].x += simd_shuffle_down(lo[ii].x, 16);
        lo[ii].y += simd_shuffle_down(lo[ii].y, 16);
        lo[ii].z += simd_shuffle_down(lo[ii].z, 16);
        lo[ii].w += simd_shuffle_down(lo[ii].w, 16);
        lo[ii].x += simd_shuffle_down(lo[ii].x, 8);
        lo[ii].y += simd_shuffle_down(lo[ii].y, 8);
        lo[ii].z += simd_shuffle_down(lo[ii].z, 8);
        lo[ii].w += simd_shuffle_down(lo[ii].w, 8);
    }
    if (ty == 0) {
        for (uint ii = 0; ii < 4; ii++) {
            ((threadgroup float4*)(t_acc + sg * 128))[ii * 8 + tx] = lo[ii];
        }
    }
    if (lane == 0) {
        t_max[sg] = M;
        t_den[sg] = S;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < ATTN_SIMDS; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < ATTN_SIMDS; s++) {
            float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 128 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        out[qh * head_dim + tid] = num / max(den, FLT_MIN);
    }
}

// Flash-decoding split: grid (q head x position slice), each threadgroup
// runs the attend_s16 loop over its contiguous slice of positions and writes
// an UNNORMALISED partial (accumulator, running max, denominator) to scratch.
// One attention over 544 positions from 64 threadgroups is latency-bound -
// each SIMD group walks 34 positions serially - while the KV bytes themselves
// stream in ~2 us. Splitting the positions across nsplit threadgroups
// multiplies the memory-level parallelism; attend_merge below folds the
// partials. Empty slices (possible when nsplit overshoots n_pos) publish
// den = 0 and the merge skips them.
kernel void attend_split(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* part_acc [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    device float* part_md [[buffer(11)]],
    constant uint& nsplit [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint qh = gid / nsplit;
    const uint sp = gid % nsplit;
    const uint per = (n_pos + nsplit - 1) / nsplit;
    const uint t_begin = sp * per;
    const uint t_end = min(t_begin + per, n_pos);

    const uint kv_head = qh / group;
    device const float4* qh4 = (device const float4*)(q + qh * head_dim);
    float4 qv = qh4[lane];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc = float4(0.0f);

    for (uint t = t_begin + sg; t < t_end; t += 16u) {
        ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        float4 kv4 = float4(((device const half4*)(k + k_base + at))[lane]);
        float s = simd_sum(dot(qv, kv4)) * scale;
        if (s > run_max) {
            float r = exp(run_max - s);
            if (run_den > 0.0f) {
                acc *= r;
                run_den *= r;
            } else {
                acc = float4(0.0f);
            }
            run_max = s;
        }
        float p = exp(s - run_max);
        run_den += p;
        float4 vv4 = float4(((device const half4*)(v + v_base + at))[lane]);
        acc += p * vv4;
    }

    // 16 simdgroups of the threadgroup fold through shared memory, lane 0 of
    // each group owns 128/16 = 8 consecutive output floats.
    threadgroup float t_acc[16 * 128];
    threadgroup float t_max[16];
    threadgroup float t_den[16];
    ((threadgroup float4*)(t_acc + sg * 128))[lane] = acc;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint tid = sg * 32 + lane;
    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < 16u; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < 16u; s++) {
            float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 128 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        const uint slot = qh * nsplit + sp;
        part_acc[slot * head_dim + tid] = num;
        if (tid == 0) {
            // The denominator travels unscaled; the merge re-weights every
            // slice by exp(slice_max - global_max), so the partial stays in
            // its own frame and den = 0 marks the empty slice.
            part_md[slot * 2 + 0] = m;
            part_md[slot * 2 + 1] = den;
        }
    }
}

// Fold nsplit flash-decoding partials into the attention output. One
// threadgroup per q head, one thread per head_dim element.
kernel void attend_merge(
    device const float* part_acc [[buffer(0)]],
    device const float* part_md [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& nsplit [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint qh [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid >= head_dim) return;
    const uint base = qh * nsplit;
    float m = -INFINITY;
    for (uint sp = 0; sp < nsplit; sp++) {
        if (part_md[(base + sp) * 2 + 1] > 0.0f) m = max(m, part_md[(base + sp) * 2 + 0]);
    }
    float num = 0.0f;
    float den = 0.0f;
    for (uint sp = 0; sp < nsplit; sp++) {
        float sden = part_md[(base + sp) * 2 + 1];
        if (sden > 0.0f) {
            float wgt = exp(part_md[(base + sp) * 2 + 0] - m);
            num += part_acc[(base + sp) * head_dim + tid] * wgt;
            den += sden * wgt;
        }
    }
    out[qh * head_dim + tid] = num / max(den, FLT_MIN);
}

// Prefill attention, llama.cpp kernel_flash_attn_ext style (f16 KV path,
// causal, no mask): 8 query rows per threadgroup, K/V streamed in
// 64-position tiles read DIRECTLY from device memory by the MMAs (no
// threadgroup staging), scores in threadgroup f32, online softmax per row
// in the owning simdgroup's registers, O accumulated in threadgroup with a
// per-tile rescale. attend_mm v1 (staged K/V, two passes, 32 rows/tg) only
// tied attend_rows_t8 standalone (0.78 vs 0.81 ms at m=480) and lost
// in-model: the staging and the K re-read cost what the MMAs saved.
// head_dim is fixed at 128 (the ATTN_HEAD_DIM the prefill path assumes).
kernel void attend_mm(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& base [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    constant uint& n_q_heads [[buffer(11)]],
    constant uint& n_rows [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half sq[8 * 128];
    threadgroup float so[8 * 128];
    threadgroup float ss[8 * 64];    // raw scores, then P in place (float)
    const uint qh = tg.x;
    const uint row0 = tg.y * 8;
    const uint kv_head = qh / group;
    const uint q_dim = n_q_heads * head_dim;
    const uint max_row = min(row0 + 7, n_rows - 1);
    const uint max_pos = base + max_row + 1;
    // Fully-masked 8-position blocks read this in-span block instead (the
    // chunk's own K/V is already written by qk_prep). Blocks that intersect
    // the valid range are never clamped - clamping would alias stale data
    // into valid positions.
    const uint span8 = (base + n_rows) > 8 ? base + n_rows - 8 : 0;

    // Stage Q as half (clamped row reads), zero O.
    for (uint i = tiitg; i < 8 * 128 / 4; i += 128) {
        uint r = i / 32;
        uint d4 = i % 32;
        uint rr = min(row0 + r, n_rows - 1);
        float4 f = ((device const float4*)(q + (ulong)rr * q_dim + qh * head_dim))[d4];
        ((threadgroup half4*)sq)[i] = half4(f);
    }
    for (uint i = tiitg; i < 8 * 128; i += 128) {
        so[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This simdgroup owns rows sgitg and sgitg + 4 for the softmax.
    float M[2] = {-INFINITY, -INFINITY};
    float S[2] = {0.0f, 0.0f};

    for (uint ic = 0; ic < max_pos; ic += 64) {
        // Q . K^T: each simdgroup scores ALL 8 rows against its two
        // 8-position strips of the tile, K straight from device memory.
        for (ushort cc = 0; cc < 2; cc++) {
            uint pb = sgitg * 2 + cc;
            uint pstart = ic + pb * 8;
            ulong p0 = pstart < max_pos ? pstart : min(pstart, span8);
            device const half* pk =
                k + k_base + p0 * kv_dim + (ulong)kv_head * head_dim;
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (ushort i = 0; i < 16; i++) {
                simdgroup_half8x8 mq;
                simdgroup_half8x8 mk;
                simdgroup_load(mq, sq + 8 * i, 128, 0, false);
                simdgroup_load(mk, pk + 8 * i, kv_dim, 0, true); // dims x pos
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            simdgroup_store(mqk, ss + pb * 8, 64, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax per owned row; P to sp; rescale the row of O.
        for (ushort jj = 0; jj < 2; jj++) {
            uint j = jj * 4 + sgitg;
            uint valid = base + min(row0 + j, max_row) + 1;
            float2 s2 = ((threadgroup float2*)(ss + j * 64))[lane];
            uint t = ic + lane * 2;
            s2 = float2(t < valid ? s2.x : -INFINITY,
                        t + 1 < valid ? s2.y : -INFINITY);
            float m = simd_max(max(s2.x, s2.y));
            float mold = M[jj];
            M[jj] = max(M[jj], m);
            float ms = mold == -INFINITY ? 1.0f : exp((mold - M[jj]) * scale);
            float2 p2 = float2(t < valid ? exp((s2.x - M[jj]) * scale) : 0.0f,
                               t + 1 < valid ? exp((s2.y - M[jj]) * scale) : 0.0f);
            S[jj] = S[jj] * ms + simd_sum(p2.x + p2.y);
            // P stays float for the PV MMA (mixed float x half is legal);
            // in-place over the scores, each lane rewriting its own pair.
            ((threadgroup float2*)(ss + j * 64))[lane] = p2;
            if (ms != 1.0f) {
                ((threadgroup float4*)(so + j * 128))[lane] *= ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P . V: each simdgroup accumulates its 32-dim strip for all 8
        // rows; V straight from device memory, O round-trips once per tile.
        simdgroup_float8x8 lo[4];
        for (ushort ii = 0; ii < 4; ii++) {
            simdgroup_load(lo[ii], so + sgitg * 32 + ii * 8, 128, 0, false);
        }
        for (ushort cc = 0; cc < 8; cc++) {
            uint pstart = ic + cc * 8;
            ulong p0 = pstart < max_pos ? pstart : min(pstart, span8);
            device const half* pv = v + v_base + p0 * kv_dim
                + (ulong)kv_head * head_dim + sgitg * 32;
            simdgroup_float8x8 pa;
            simdgroup_load(pa, ss + cc * 8, 64, 0, false);
            for (ushort ii = 0; ii < 4; ii++) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, pv + ii * 8, kv_dim, 0, false);
                simdgroup_multiply_accumulate(lo[ii], pa, mv, lo[ii]);
            }
        }
        for (ushort ii = 0; ii < 4; ii++) {
            simdgroup_store(lo[ii], so + sgitg * 32 + ii * 8, 128, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: divide by the denominator, store with row bounds. Each
    // simdgroup writes its two owned rows (one float4 per lane).
    for (ushort jj = 0; jj < 2; jj++) {
        uint j = jj * 4 + sgitg;
        uint row = row0 + j;
        if (row < n_rows && S[jj] > 0.0f) {
            float inv = 1.0f / S[jj];
            device float4* o4 =
                (device float4*)(out + (ulong)row * q_dim + qh * head_dim);
            threadgroup float4* s4 = (threadgroup float4*)(so + j * 128);
            o4[lane] = s4[lane] * inv;
        }
    }
}

// attend_mm for head_dim 256 (qwen35moe): same structure - 8 query rows per
// threadgroup, K/V in 64-position tiles read directly from device memory,
// scores in threadgroup f32, online softmax per row, O in threadgroup with
// a per-tile rescale. The 256-wide head doubles the QK tile count and the
// V strips (4 simdgroups own 64 dims each); threadgroup memory is 14 KB.
kernel void attend_mm256(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& base [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    constant uint& n_q_heads [[buffer(11)]],
    constant uint& n_rows [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half sq[8 * 256];
    threadgroup float so[8 * 256];
    threadgroup float ss[8 * 64];    // raw scores, then P in place (float)
    const uint qh = tg.x;
    const uint row0 = tg.y * 8;
    const uint kv_head = qh / group;
    const uint q_dim = n_q_heads * head_dim;
    const uint max_row = min(row0 + 7, n_rows - 1);
    const uint max_pos = base + max_row + 1;
    // Fully-masked 8-position blocks read this in-span block instead (the
    // chunk's own K/V is already written by qk_prep). Blocks that intersect
    // the valid range are never clamped - clamping would alias stale data
    // into valid positions.
    const uint span8 = (base + n_rows) > 8 ? base + n_rows - 8 : 0;

    // Stage Q as half (clamped row reads), zero O.
    for (uint i = tiitg; i < 8 * 256 / 4; i += 128) {
        uint r = i / 64;
        uint d4 = i % 64;
        uint rr = min(row0 + r, n_rows - 1);
        float4 f = ((device const float4*)(q + (ulong)rr * q_dim + qh * head_dim))[d4];
        ((threadgroup half4*)sq)[i] = half4(f);
    }
    for (uint i = tiitg; i < 8 * 256; i += 128) {
        so[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This simdgroup owns rows sgitg and sgitg + 4 for the softmax.
    float M[2] = {-INFINITY, -INFINITY};
    float S[2] = {0.0f, 0.0f};

    for (uint ic = 0; ic < max_pos; ic += 64) {
        // Q . K^T: each simdgroup scores ALL 8 rows against its two
        // 8-position strips of the tile, K straight from device memory.
        for (ushort cc = 0; cc < 2; cc++) {
            uint pb = sgitg * 2 + cc;
            uint pstart = ic + pb * 8;
            ulong p0 = pstart < max_pos ? pstart : min(pstart, span8);
            device const half* pk =
                k + k_base + p0 * kv_dim + (ulong)kv_head * head_dim;
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (ushort i = 0; i < 32; i++) {
                simdgroup_half8x8 mq;
                simdgroup_half8x8 mk;
                simdgroup_load(mq, sq + 8 * i, 256, 0, false);
                simdgroup_load(mk, pk + 8 * i, kv_dim, 0, true); // dims x pos
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            simdgroup_store(mqk, ss + pb * 8, 64, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax per owned row; P to sp; rescale the row of O.
        for (ushort jj = 0; jj < 2; jj++) {
            uint j = jj * 4 + sgitg;
            uint valid = base + min(row0 + j, max_row) + 1;
            float2 s2 = ((threadgroup float2*)(ss + j * 64))[lane];
            uint t = ic + lane * 2;
            s2 = float2(t < valid ? s2.x : -INFINITY,
                        t + 1 < valid ? s2.y : -INFINITY);
            float m = simd_max(max(s2.x, s2.y));
            float mold = M[jj];
            M[jj] = max(M[jj], m);
            float ms = mold == -INFINITY ? 1.0f : exp((mold - M[jj]) * scale);
            float2 p2 = float2(t < valid ? exp((s2.x - M[jj]) * scale) : 0.0f,
                               t + 1 < valid ? exp((s2.y - M[jj]) * scale) : 0.0f);
            S[jj] = S[jj] * ms + simd_sum(p2.x + p2.y);
            // P stays float for the PV MMA (mixed float x half is legal);
            // in-place over the scores, each lane rewriting its own pair.
            ((threadgroup float2*)(ss + j * 64))[lane] = p2;
            if (ms != 1.0f) {
                ((threadgroup float4*)(so + j * 256))[lane] *= ms;
                ((threadgroup float4*)(so + j * 256))[lane + 32] *= ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P . V: each simdgroup accumulates its 64-dim strip for all 8
        // rows; V straight from device memory, O round-trips once per tile.
        simdgroup_float8x8 lo[8];
        for (ushort ii = 0; ii < 8; ii++) {
            simdgroup_load(lo[ii], so + sgitg * 64 + ii * 8, 256, 0, false);
        }
        for (ushort cc = 0; cc < 8; cc++) {
            uint pstart = ic + cc * 8;
            ulong p0 = pstart < max_pos ? pstart : min(pstart, span8);
            device const half* pv = v + v_base + p0 * kv_dim
                + (ulong)kv_head * head_dim + sgitg * 64;
            simdgroup_float8x8 pa;
            simdgroup_load(pa, ss + cc * 8, 64, 0, false);
            for (ushort ii = 0; ii < 8; ii++) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, pv + ii * 8, kv_dim, 0, false);
                simdgroup_multiply_accumulate(lo[ii], pa, mv, lo[ii]);
            }
        }
        for (ushort ii = 0; ii < 8; ii++) {
            simdgroup_store(lo[ii], so + sgitg * 64 + ii * 8, 256, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: divide by the denominator, store with row bounds. Each
    // simdgroup writes its two owned rows (two float4 per lane).
    for (ushort jj = 0; jj < 2; jj++) {
        uint j = jj * 4 + sgitg;
        uint row = row0 + j;
        if (row < n_rows && S[jj] > 0.0f) {
            float inv = 1.0f / S[jj];
            device float4* o4 =
                (device float4*)(out + (ulong)row * q_dim + qh * head_dim);
            threadgroup float4* s4 = (threadgroup float4*)(so + j * 256);
            o4[lane] = s4[lane] * inv;
            o4[lane + 32] = s4[lane + 32] * inv;
        }
    }
}

// The batched attend: grid (query head, row), each row causal over its own
// prefix (base + row + 1 positions). Replaces one dispatch per prefill row -
// the driver's ~50 us per dispatch over ~240 rows per layer-chunk was the
// last big prefill cost.
// Prefill attention with a tile of FOUR query rows per threadgroup: the
// K/V prefix streams through once per tile instead of once per row, cutting
// cache traffic 4x. The four per-row score reductions are independent
// simd_sums, so their shuffle chains overlap. Causality per row via its own
// n_pos; rows past n_rows only skip their write.
kernel void attend_rows_t8(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& base [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    constant uint& n_q_heads [[buffer(11)]],
    constant uint& n_rows [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[ATTN_SIMDS * 8 * 128];
    threadgroup float t_max[ATTN_SIMDS * 8];
    threadgroup float t_den[ATTN_SIMDS * 8];

    uint qh = tg.x;
    uint row0 = tg.y * 8;
    uint kv_head = qh / group;
    uint q_dim = n_q_heads * head_dim;
    uint last = min(row0 + 7, n_rows - 1);
    uint max_pos = base + last + 1;

    float4 qv[8];
    for (uint r = 0; r < 8; r++) {
        uint rr = min(row0 + r, n_rows - 1);
        qv[r] = ((device const float4*)(q + (ulong)rr * q_dim + qh * head_dim))[lane];
    }

    float run_max[8] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float run_den[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc[8] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};

    for (uint t = sg; t < max_pos; t += ATTN_SIMDS) {
        ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        float4 kv4 = float4(((device const half4*)(k + k_base + at))[lane]);
        float4 vv4 = float4(((device const half4*)(v + v_base + at))[lane]);
        float s[8];
        for (uint r = 0; r < 8; r++) {
            s[r] = simd_sum(dot(qv[r], kv4)) * scale;
        }
        for (uint r = 0; r < 8; r++) {
            if (t >= base + row0 + r + 1) {
                continue;
            }
            if (s[r] > run_max[r]) {
                float rs = exp(run_max[r] - s[r]);
                if (run_den[r] > 0.0f) {
                    acc[r] *= rs;
                    run_den[r] *= rs;
                } else {
                    acc[r] = float4(0.0f);
                }
                run_max[r] = s[r];
            }
            float p = exp(s[r] - run_max[r]);
            run_den[r] += p;
            acc[r] += p * vv4;
        }
    }

    for (uint r = 0; r < 8; r++) {
        ((threadgroup float4*)(t_acc + (sg * 8 + r) * 128))[lane] = acc[r];
    }
    if (lane == 0) {
        for (uint r = 0; r < 8; r++) {
            t_max[sg * 8 + r] = run_max[r];
            t_den[sg * 8 + r] = run_den[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        for (uint r = 0; r < 8; r++) {
            if (row0 + r >= n_rows) {
                break;
            }
            float mx = -INFINITY;
            for (uint s2 = 0; s2 < ATTN_SIMDS; s2++) {
                mx = max(mx, t_max[s2 * 8 + r]);
            }
            float num = 0.0f;
            float den = 0.0f;
            for (uint s2 = 0; s2 < ATTN_SIMDS; s2++) {
                float wgt = t_den[s2 * 8 + r] > 0.0f ? exp(t_max[s2 * 8 + r] - mx) : 0.0f;
                num += t_acc[(s2 * 8 + r) * 128 + tid] * wgt;
                den += t_den[s2 * 8 + r] * wgt;
            }
            out[(ulong)(row0 + r) * q_dim + qh * head_dim + tid] = num / max(den, FLT_MIN);
        }
    }
}

kernel void attend_rows_t4(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& base [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    constant uint& n_q_heads [[buffer(11)]],
    constant uint& n_rows [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[ATTN_SIMDS * 4 * 128];
    threadgroup float t_max[ATTN_SIMDS * 4];
    threadgroup float t_den[ATTN_SIMDS * 4];

    uint qh = tg.x;
    uint row0 = tg.y * 4;
    uint kv_head = qh / group;
    uint q_dim = n_q_heads * head_dim;
    uint last = min(row0 + 3, n_rows - 1);
    uint max_pos = base + last + 1;

    float4 qv[4];
    for (uint r = 0; r < 4; r++) {
        uint rr = min(row0 + r, n_rows - 1);
        qv[r] = ((device const float4*)(q + (ulong)rr * q_dim + qh * head_dim))[lane];
    }

    float run_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float run_den[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};

    for (uint t = sg; t < max_pos; t += ATTN_SIMDS) {
        ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        float4 kv4 = float4(((device const half4*)(k + k_base + at))[lane]);
        float4 vv4 = float4(((device const half4*)(v + v_base + at))[lane]);
        float s[4];
        for (uint r = 0; r < 4; r++) {
            s[r] = simd_sum(dot(qv[r], kv4)) * scale;
        }
        for (uint r = 0; r < 4; r++) {
            if (t >= base + row0 + r + 1) {
                continue;
            }
            if (s[r] > run_max[r]) {
                float rs = exp(run_max[r] - s[r]);
                if (run_den[r] > 0.0f) {
                    acc[r] *= rs;
                    run_den[r] *= rs;
                } else {
                    acc[r] = float4(0.0f);
                }
                run_max[r] = s[r];
            }
            float p = exp(s[r] - run_max[r]);
            run_den[r] += p;
            acc[r] += p * vv4;
        }
    }

    for (uint r = 0; r < 4; r++) {
        ((threadgroup float4*)(t_acc + (sg * 4 + r) * 128))[lane] = acc[r];
    }
    if (lane == 0) {
        for (uint r = 0; r < 4; r++) {
            t_max[sg * 4 + r] = run_max[r];
            t_den[sg * 4 + r] = run_den[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        for (uint r = 0; r < 4; r++) {
            if (row0 + r >= n_rows) {
                break;
            }
            float mx = -INFINITY;
            for (uint s2 = 0; s2 < ATTN_SIMDS; s2++) {
                mx = max(mx, t_max[s2 * 4 + r]);
            }
            float num = 0.0f;
            float den = 0.0f;
            for (uint s2 = 0; s2 < ATTN_SIMDS; s2++) {
                float wgt = t_den[s2 * 4 + r] > 0.0f ? exp(t_max[s2 * 4 + r] - mx) : 0.0f;
                num += t_acc[(s2 * 4 + r) * 128 + tid] * wgt;
                den += t_den[s2 * 4 + r] * wgt;
            }
            out[(ulong)(row0 + r) * q_dim + qh * head_dim + tid] = num / max(den, FLT_MIN);
        }
    }
}

kernel void attend_rows(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& base [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    constant uint& n_q_heads [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[ATTN_SIMDS * 128];
    threadgroup float t_max[ATTN_SIMDS];
    threadgroup float t_den[ATTN_SIMDS];

    uint qh = tg.x;
    uint row = tg.y;
    uint n_pos = base + row + 1;
    uint kv_head = qh / group;
    uint q_dim = n_q_heads * head_dim;
    device const float4* qh4 =
        (device const float4*)(q + (ulong)row * q_dim + qh * head_dim);
    float4 qv = qh4[lane];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc = float4(0.0f);
    for (uint t = sg; t < n_pos; t += ATTN_SIMDS) {
        ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        float4 kv4 = float4(((device const half4*)(k + k_base + at))[lane]);
        float s = simd_sum(dot(qv, kv4)) * scale;
        if (s > run_max) {
            float r = exp(run_max - s);
            if (run_den > 0.0f) {
                acc *= r;
                run_den *= r;
            } else {
                acc = float4(0.0f);
            }
            run_max = s;
        }
        float p = exp(s - run_max);
        run_den += p;
        float4 vv4 = float4(((device const half4*)(v + v_base + at))[lane]);
        acc += p * vv4;
    }
    ((threadgroup float4*)(t_acc + sg * 128))[lane] = acc;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < ATTN_SIMDS; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < ATTN_SIMDS; s++) {
            float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 128 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        out[(ulong)row * q_dim + qh * head_dim + tid] = num / max(den, FLT_MIN);
    }
}


// SwiGLU in place over the gate buffer: the bridge that lets gate/up/down
// matmuls of one FFN encode into a single command buffer with no CPU stop.
kernel void swiglu(
    device float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float g = gate[i];
    gate[i] = (g / (1.0f + exp(-g))) * up[i];
}

// Plain device-to-device f32 copy: the one-buffer prefill rescues the
// router logits out of y_arena with this before y_arena can grow - a CPU
// readback is impossible there because nothing has committed yet.
kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    dst[i] = src[i];
}

kernel void matvec_q6_k(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n_in [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],
    constant ulong& w_off [[buffer(5)]],
    device const uint* ids [[buffer(6)]],
    constant IdxArgs& idx [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint jt = tid / LPR;
    uint lane = tid % LPR;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    // ROWS: the grid covers every token row's slots; tok/jr split the work
    // item, per-row math below is untouched.
    uint tok = ROWS ? jt / ycols : 0;
    uint jr = ROWS ? jt - tok * ycols : jt;
    uint slot = INDEXED ? jr / n_out : 0;
    uint j = INDEXED ? jr % n_out : jr;
    bool active = INDEXED ? (ROWS ? jt < ycols * idx.n_rows : jr < ycols) : j < n_out;
    if (!active) j = 0;
    if (INDEXED) {
        w += ids[(ROWS ? tok * idx.ids_stride : 0u) + slot] * idx.stride;
        x += (ulong)tok * idx.x_row_stride + (ulong)slot * idx.x_stride;
    }
    uint blocks = n_in / 256;
    device const uchar* row = w + w_off + (ulong)j * blocks * 210;
    float total[MAXT] = {0};
    for (uint b = lane; b < blocks; b += LPR) {
        device const uchar* blk = row + b * 210;
        float d = half_at(blk + 208);
        for (uint half_i = 0; half_i < 2; half_i++) {
            // 210-byte blocks are 2-aligned, so the quant bytes pair up from
            // ushort loads into registers - 12 word loads per 16-element
            // group where the old form did 48 dependent byte loads.
            device const ushort* ql16 = (device const ushort*)(blk + half_i * 64);
            device const ushort* qh16 = (device const ushort*)(blk + 128 + half_i * 32);
            device const uchar* sc = blk + 192 + half_i * 8;
            uint xh0 = b * 256 + half_i * 128;
            for (uint is = 0; is < 2; is++) {
                uint la[4];
                uint lb[4];
                uint hh[4];
                for (uint k = 0; k < 4; k++) {
                    uint base = is * 8 + k * 2;
                    la[k] = (uint)ql16[base] | ((uint)ql16[base + 1] << 16);
                    lb[k] = (uint)ql16[16 + base] | ((uint)ql16[16 + base + 1] << 16);
                    hh[k] = (uint)qh16[base] | ((uint)qh16[base + 1] << 16);
                }
                float a1[MAXT] = {0};
                float a2[MAXT] = {0};
                float a3[MAXT] = {0};
                float a4[MAXT] = {0};
                for (uint k = 0; k < 4; k++) {
                    uint l = is * 16 + k * 4;
                    float4 q1 = float4(int4(as_type<uchar4>(
                        (la[k] & 0x0F0F0F0Fu) | ((hh[k] & 0x03030303u) << 4))) - 32);
                    float4 q2 = float4(int4(as_type<uchar4>(
                        (lb[k] & 0x0F0F0F0Fu) | (((hh[k] >> 2) & 0x03030303u) << 4))) - 32);
                    float4 q3 = float4(int4(as_type<uchar4>(
                        ((la[k] >> 4) & 0x0F0F0F0Fu) | (((hh[k] >> 4) & 0x03030303u) << 4))) - 32);
                    float4 q4v = float4(int4(as_type<uchar4>(
                        ((lb[k] >> 4) & 0x0F0F0F0Fu) | (((hh[k] >> 6) & 0x03030303u) << 4))) - 32);
                    for (uint i = 0; i < TILE; i++) {
                        device const float* xi = x + i * n_in + xh0 + l;
                        a1[i] += dot(q1, *(device const float4*)(xi));
                        a2[i] += dot(q2, *(device const float4*)(xi + 32));
                        a3[i] += dot(q3, *(device const float4*)(xi + 64));
                        a4[i] += dot(q4v, *(device const float4*)(xi + 96));
                    }
                }
                for (uint i = 0; i < TILE; i++) {
                    total[i] += d * ((float)((char)sc[is]) * a1[i]
                        + (float)((char)sc[2 + is]) * a2[i]
                        + (float)((char)sc[4 + is]) * a3[i]
                        + (float)((char)sc[6 + is]) * a4[i]);
                }
            }
        }
    }
    for (uint i = 0; i < TILE; i++) {
        float s = lane_sum(total[i]);
        if (active && lane == 0) y[(ROWS ? tok * idx.y_row_stride : i * ycols) + (ROWS ? jr : jt)] = s;
    }
}
// ---------------------------------------------------------------------------
// qwen35moe: gated-delta-net decode (one token) and head_dim-256 attention.
// CPU reference: ops::gated_deltanet_step / Model::gdn_forward.
// ---------------------------------------------------------------------------

struct GdnConvArgs {
    uint channels;      // qkv row width (key_dim*2 + value_dim)
    uint d_conv;        // kernel taps (4)
    uint pad0;
    uint pad1;
    ulong conv_off;     // element offset of this layer's window in the ssm region
};

// One depthwise causal-conv step over the token's qkv row: out[c] =
// silu(sum_j w[c][j] * window[j][c] + w[c][last] * cur[c]); the window then
// shifts one row left and keeps the RAW cur row (llama.cpp kernel_ssm_conv).
kernel void gdn_conv(
    device float* qkv [[buffer(0)]],
    device float* ssm [[buffer(1)]],
    device const float* w [[buffer(2)]],
    constant GdnConvArgs& a [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= a.channels) return;
    device float* win = ssm + a.conv_off;
    const uint C = a.channels;
    float acc = 0.0f;
    for (uint j = 0; j + 1 < a.d_conv; j++) {
        acc += win[j * C + i] * w[i * a.d_conv + j];
    }
    const float cur = qkv[i];
    acc += cur * w[i * a.d_conv + a.d_conv - 1];
    for (uint j = 0; j + 2 < a.d_conv; j++) {
        win[j * C + i] = win[(j + 1) * C + i];
    }
    win[(a.d_conv - 2) * C + i] = cur;
    qkv[i] = acc / (1.0f + exp(-acc));
}

struct GdnStepArgs {
    uint heads_k;       // 16
    uint heads_v;       // 32
    uint d;             // head dim, 128
    uint key_dim;       // heads_k * d
    float eps;
    uint pad0;
    ulong state_off;    // element offset of this layer's S in the ssm region
};

// The gated deltanet recurrence for one token (a port of ggml's
// kernel_gated_delta_net_impl for n_tokens = 1, with the l2 norms folded
// in). One simdgroup per state row: 32 lanes x 4 columns each.
//   S *= exp(g); s_k = S.k; d = (v - s_k)*beta; S += k.d; y = (S.q)/sqrt(d)
// State layout matches the CPU reference: [head][out_row][key_col].
kernel void gdn_step(
    device float* ssm [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device const float* ab [[buffer(2)]],   // [alpha(heads_v) | beta_raw(heads_v)]
    constant GdnStepArgs& args [[buffer(3)]],
    constant float* a_log [[buffer(4)]],    // [heads_v]
    constant float* dt_bias [[buffer(5)]],  // [heads_v]
    device float* out [[buffer(6)]],        // [heads_v * d]
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;
    const uint h = tgpig.y;
    const uint i20 = tgpig.x * 4 + ty;
    const uint d = args.d;
    const uint kh = h % args.heads_k;

    device const float* q_ptr = qkv + kh * d;
    device const float* k_ptr = qkv + args.key_dim + kh * d;
    device const float* v_ptr = qkv + 2 * args.key_dim + h * d;

    // l2-normalized q/k, each simdgroup computing the head sum redundantly.
    float q0 = q_ptr[tx*4+0], q1 = q_ptr[tx*4+1], q2 = q_ptr[tx*4+2], q3 = q_ptr[tx*4+3];
    float k0 = k_ptr[tx*4+0], k1 = k_ptr[tx*4+1], k2 = k_ptr[tx*4+2], k3 = k_ptr[tx*4+3];
    const float qi = rsqrt(simd_sum(q0*q0 + q1*q1 + q2*q2 + q3*q3) + args.eps);
    const float ki = rsqrt(simd_sum(k0*k0 + k1*k1 + k2*k2 + k3*k3) + args.eps);
    q0 *= qi; q1 *= qi; q2 *= qi; q3 *= qi;
    k0 *= ki; k1 *= ki; k2 *= ki; k3 *= ki;

    const float sp = ab[h] + dt_bias[h];
    // softplus(alpha + dt) * a_log; the GGUF's ssm_a already holds -exp(A_log).
    const float g = (sp > 20.0f ? sp : log(1.0f + exp(sp))) * a_log[h];
    const float g_exp = exp(g);
    const float beta = 1.0f / (1.0f + exp(-ab[args.heads_v + h]));

    device float* s_ptr = ssm + args.state_off + (ulong)h * d * d + (ulong)i20 * d;
    float ls0 = s_ptr[tx*4+0] * g_exp;
    float ls1 = s_ptr[tx*4+1] * g_exp;
    float ls2 = s_ptr[tx*4+2] * g_exp;
    float ls3 = s_ptr[tx*4+3] * g_exp;
    const float s_k = simd_sum(ls0*k0 + ls1*k1 + ls2*k2 + ls3*k3);
    const float dv = (v_ptr[i20] - s_k) * beta;
    ls0 += k0 * dv; ls1 += k1 * dv; ls2 += k2 * dv; ls3 += k3 * dv;
    const float y = simd_sum(ls0*q0 + ls1*q1 + ls2*q2 + ls3*q3) * rsqrt((float)d);
    s_ptr[tx*4+0] = ls0; s_ptr[tx*4+1] = ls1; s_ptr[tx*4+2] = ls2; s_ptr[tx*4+3] = ls3;
    if (tx == 0) {
        out[h * d + i20] = y;
    }
}

// The gated per-head rmsnorm over the recurrence output, times silu(z)
// (llama.cpp build_norm_gated): out[h] = rmsnorm(y[h], w) * silu(z[h]).
// One threadgroup per head, 128 threads.
kernel void gdn_out_norm(
    device float* y [[buffer(0)]],
    device const float* z [[buffer(1)]],
    constant float* w [[buffer(2)]],
    constant uint& heads_v [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[8];
    const float yv = y[h * d + tid];
    const float ps = simd_sum(yv * yv);
    if (lane == 0) {
        partial[sg] = ps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < d / 32; i++) {
        total += partial[i];
    }
    const float scale = rsqrt(total / (float)d + eps);
    const float zv = z[h * d + tid];
    y[h * d + tid] = yv * scale * w[tid] * (zv / (1.0f + exp(-zv)));
}

// qk_prep for head_dim 256 (qwen35moe full-attention layers): 8 registers
// per lane instead of 4, and the fused output gate riding inside the q
// projection ([q(256) | gate(256)] per head) split out to its own buffer.
// Partial rotary only (rot_h == 32 pairs): the model's rope covers 64 of 256.
struct QkPrep256Args {
    uint n_heads;
    uint n_kv_heads;
    uint kv_dim;
    float eps;
    uint pos;
    uint has_qk_norm;
    uint rot_dim;
    uint gate_in_q;
    ulong k_base;
    ulong v_base;
};

kernel void qk_prep256(
    device float* q [[buffer(0)]],        // in: interleaved [q|gate] rows when gate_in_q
    device float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device half* cache [[buffer(3)]],
    constant QkPrep256Args& a [[buffer(4)]],
    constant float* qw [[buffer(5)]],
    constant float* kw [[buffer(6)]],
    constant float2* rope [[buffer(7)]],
    device float* q_out [[buffer(8)]],    // deinterleaved normed q
    device float* gate_out [[buffer(9)]], // raw gate rows (sigmoid applied post-attention)
    uint hid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint hd = 256;
    if (hid >= a.n_heads + a.n_kv_heads) {
        // v head: narrow to half and store.
        const uint kvh = hid - a.n_heads - a.n_kv_heads;
        device const float* src = v + kvh * hd;
        device half* dst = cache + a.v_base + (ulong)a.pos * a.kv_dim + kvh * hd;
        for (uint e = lane; e < hd; e += 32) {
            dst[e] = (half)src[e];
        }
        return;
    }
    const bool is_q = hid < a.n_heads;
    const uint src_stride = (is_q && a.gate_in_q != 0) ? 2 * hd : hd;
    device float* xh = is_q ? (q + hid * src_stride) : (k + (hid - a.n_heads) * hd);
    constant float* w = is_q ? qw : kw;
    float e[8];
    for (uint j = 0; j < 8; j++) {
        e[j] = xh[lane + j * 32];
    }
    if (a.gate_in_q != 0 && is_q) {
        device float* gs = q + hid * 2 * hd + hd;
        for (uint j = 0; j < 8; j++) {
            gate_out[hid * hd + lane + j * 32] = gs[lane + j * 32];
        }
    }
    if (a.has_qk_norm != 0) {
        float ss = 0.0f;
        for (uint j = 0; j < 8; j++) {
            ss += e[j] * e[j];
        }
        const float scale = rsqrt(simd_sum(ss) / (float)hd + a.eps);
        for (uint j = 0; j < 8; j++) {
            e[j] *= scale * w[lane + j * 32];
        }
    }
    // Partial rotary: rot_h == 32 pairs elements (lane, lane + 32) of the
    // first 64; the rest of the 256-wide head passes through. The host
    // rejects other widths.
    const uint rot_h = a.rot_dim / 2;
    if (rot_h == 32 && lane < 32) {
        const float2 r0 = rope[lane];
        const float t0 = e[0] * r0.y - e[1] * r0.x;
        const float t1 = e[0] * r0.x + e[1] * r0.y;
        e[0] = t0; e[1] = t1;
    }
    if (is_q) {
        for (uint j = 0; j < 8; j++) {
            q_out[hid * hd + lane + j * 32] = e[j];
        }
    } else {
        const uint kvh = hid - a.n_heads;
        device half* dst = cache + a.k_base + (ulong)a.pos * a.kv_dim + kvh * hd;
        for (uint j = 0; j < 8; j++) {
            dst[lane + j * 32] = (half)e[j];
        }
    }
}

// Decode attention for head_dim 256: attend_s32 with two float4 of q per
// lane. 16 simdgroups, not 32: t_acc[32*256] would exceed the 32 KB
// threadgroup-memory limit by 256 bytes.
kernel void attend_s32_256(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    uint qh [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[16 * 256];
    threadgroup float t_max[16];
    threadgroup float t_den[16];

    const uint kv_head = qh / group;
    device const float4* qh4 = (device const float4*)(q + qh * head_dim);
    const float4 qv0 = qh4[lane];
    const float4 qv1 = qh4[lane + 32];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint t = sg; t < n_pos; t += 16u) {
        const ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        const float4 ka = float4(((device const half4*)(k + k_base + at))[lane]);
        const float4 kb = float4(((device const half4*)(k + k_base + at))[lane + 32]);
        const float s = simd_sum(dot(qv0, ka) + dot(qv1, kb)) * scale;
        if (s > run_max) {
            const float r = exp(run_max - s);
            if (run_den > 0.0f) {
                acc0 *= r; acc1 *= r;
                run_den *= r;
            } else {
                acc0 = float4(0.0f); acc1 = float4(0.0f);
            }
            run_max = s;
        }
        const float p = exp(s - run_max);
        run_den += p;
        const float4 va = float4(((device const half4*)(v + v_base + at))[lane]);
        const float4 vb = float4(((device const half4*)(v + v_base + at))[lane + 32]);
        acc0 += p * va;
        acc1 += p * vb;
    }

    ((threadgroup float4*)(t_acc + sg * 256))[lane] = acc0;
    ((threadgroup float4*)(t_acc + sg * 256))[lane + 32] = acc1;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < 16u; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < 16u; s++) {
            const float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 256 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        out[qh * head_dim + tid] = num / max(den, FLT_MIN);
    }
}

// Flash-decoding split for head_dim 256, attend_split with two float4 of q.
kernel void attend_split_256(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* part_acc [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& n_pos [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    device float* part_md [[buffer(11)]],
    constant uint& nsplit [[buffer(12)]],
    uint gid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint qh = gid / nsplit;
    const uint sp = gid % nsplit;
    const uint per = (n_pos + nsplit - 1) / nsplit;
    const uint t_begin = sp * per;
    const uint t_end = min(t_begin + per, n_pos);

    const uint kv_head = qh / group;
    device const float4* qh4 = (device const float4*)(q + qh * head_dim);
    const float4 qv0 = qh4[lane];
    const float4 qv1 = qh4[lane + 32];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint t = t_begin + sg; t < t_end; t += 16u) {
        const ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        const float4 ka = float4(((device const half4*)(k + k_base + at))[lane]);
        const float4 kb = float4(((device const half4*)(k + k_base + at))[lane + 32]);
        const float s = simd_sum(dot(qv0, ka) + dot(qv1, kb)) * scale;
        if (s > run_max) {
            const float r = exp(run_max - s);
            if (run_den > 0.0f) {
                acc0 *= r; acc1 *= r;
                run_den *= r;
            } else {
                acc0 = float4(0.0f); acc1 = float4(0.0f);
            }
            run_max = s;
        }
        const float p = exp(s - run_max);
        run_den += p;
        const float4 va = float4(((device const half4*)(v + v_base + at))[lane]);
        const float4 vb = float4(((device const half4*)(v + v_base + at))[lane + 32]);
        acc0 += p * va;
        acc1 += p * vb;
    }

    threadgroup float t_acc[16 * 256];
    threadgroup float t_max[16];
    threadgroup float t_den[16];
    ((threadgroup float4*)(t_acc + sg * 256))[lane] = acc0;
    ((threadgroup float4*)(t_acc + sg * 256))[lane + 32] = acc1;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint tid = sg * 32 + lane;
    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < 16u; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < 16u; s++) {
            const float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 256 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        const uint slot = qh * nsplit + sp;
        part_acc[slot * head_dim + tid] = num;
        if (tid == 0) {
            part_md[slot * 2 + 0] = m;
            part_md[slot * 2 + 1] = den;
        }
    }
}

// ---------------------------------------------------------------------------
// qwen35moe prefill (batched over a chunk of m tokens).
// ---------------------------------------------------------------------------

// qwen35moe: the sigmoid output gate scales the attention output before wo.
kernel void attn_gate_mul(
    device float* attn [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float g = gate[i];
    attn[i] *= 1.0f / (1.0f + exp(-g));
}

struct GdnConvBatchArgs {
    uint channels;
    uint d_conv;
    uint m;
    uint pad0;
    ulong conv_off;     // this layer's window in the ssm region (d_conv-1 rows)
};

// The depthwise causal conv over a whole chunk. Row i mixes the d_conv-1
// preceding RAW qkv rows: the window from the session for i < d_conv-1,
// the chunk's own rows after that (llama.cpp kernel_ssm_conv over
// [state | chunk]). Input and output are different buffers - row i+1..i+3
// still need row i's raw value. The session window is only READ here; the
// tail rows land in it later, after a barrier (a concurrent rewrite would
// race the early rows' reads).
kernel void gdn_conv_batch(
    device const float* qkv [[buffer(0)]],   // [m][channels] raw rows
    device float* qkc [[buffer(1)]],         // [m][channels] silu(conv) out
    device const float* ssm [[buffer(2)]],
    device const float* w [[buffer(3)]],
    constant GdnConvBatchArgs& a [[buffer(4)]],
    device float* slots [[buffer(5)]],
    constant uint& slot_total [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]])
{
    const uint c = tgp.x * 256 + i;
    if (c >= a.channels) return;
    const uint row = tgp.y;
    const uint C = a.channels;
    device const float* win = ssm + a.conv_off;
    const uint K = a.d_conv - 1;
    float acc = 0.0f;
    for (uint j = 0; j < K; j++) {
        // virtual index of the source row: negative reads the window.
        const int src = (int)row - (int)(K - j);
        const float v = src >= 0 ? qkv[(uint)src * C + c] : win[(uint)(src + (int)K) * C + c];
        acc += v * w[c * a.d_conv + j];
    }
    const float cur = qkv[row * C + c];
    acc += cur * w[c * a.d_conv + K];
    qkc[row * C + c] = acc / (1.0f + exp(-acc));
    // Rollback slots (MTP verification): this raw row is one of the last K
    // window rows for slots row..row+K-1; the session window rows (read by
    // the first K chunk rows) seed slots 0..K-1 the same way. The conv
    // section sits at a.conv_off within each slot, mirroring the region.
    if (slot_total != 0) {
        for (uint r = row; r < min(row + K, a.m); r++) {
            const uint wj = K - 1 - (r - row);
            slots[(ulong)r * slot_total + a.conv_off + wj * C + c] = cur;
        }
        if (row == 0) {
            for (uint j = 0; j < K; j++) {
                const float v = win[j * C + c];
                // Session window row j (virtual raw row j-K) sits in slot
                // r's window at position j-1-r: slot r's window position p
                // holds raw row r-K+1+p, and j-K = r-K+1+p gives p = j-1-r.
                // Positions beyond that (p >= K-1-r, i.e. raw rows >= 0)
                // belong to the per-row threads - the r < j bound keeps the
                // two halves disjoint, no race.
                for (uint r = 0; r < j; r++) {
                    slots[(ulong)r * slot_total + a.conv_off + (j - 1 - r) * C + c] = v;
                }
            }
        }
    }
}

struct GdnStepBatchArgs {
    uint heads_k;
    uint heads_v;
    uint d;
    uint key_dim;
    uint m;
    float eps;
    ulong state_off;
};

// The gated deltanet recurrence over a whole chunk in ONE dispatch: each
// simdgroup owns one state row and walks the tokens sequentially with the
// row kept in registers (llama.cpp kernel_gated_delta_net_impl's t loop;
// the l2 norms ride along per token). Grid (d/4, heads_v), threads (32, 4).
// When slot_total != 0, the per-row states ALSO land in the rollback slots
// (MTP verification): after token t every thread dumps its register row
// into slot t, whose layout mirrors the whole SSM region, so a partial
// round rolls back with one contiguous copy instead of a replay.
kernel void gdn_step_batch(
    device float* ssm [[buffer(0)]],
    device const float* qkc [[buffer(1)]],   // [m][channels] post-conv
    device const float* alpha [[buffer(2)]], // [m][heads_v]
    device const float* beta_raw [[buffer(3)]],
    constant GdnStepBatchArgs& args [[buffer(4)]],
    constant float* a_log [[buffer(5)]],
    constant float* dt_bias [[buffer(6)]],
    device float* out [[buffer(7)]],         // [m][heads_v * d]
    device float* slots [[buffer(8)]],
    constant uint& slot_total [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;
    const uint h = tgpig.y;
    const uint i20 = tgpig.x * 4 + ty;
    const uint d = args.d;
    const uint kh = h % args.heads_k;
    const uint channels = 2 * args.key_dim + args.heads_v * d;

    device float* s_ptr = ssm + args.state_off + (ulong)h * d * d + (ulong)i20 * d;
    float ls0 = s_ptr[tx*4+0], ls1 = s_ptr[tx*4+1], ls2 = s_ptr[tx*4+2], ls3 = s_ptr[tx*4+3];

    for (uint t = 0; t < args.m; t++) {
        device const float* q_ptr = qkc + (ulong)t * channels + kh * d;
        device const float* k_ptr = qkc + (ulong)t * channels + args.key_dim + kh * d;
        device const float* v_ptr = qkc + (ulong)t * channels + 2 * args.key_dim + h * d;

        float q0 = q_ptr[tx*4+0], q1 = q_ptr[tx*4+1], q2 = q_ptr[tx*4+2], q3 = q_ptr[tx*4+3];
        float k0 = k_ptr[tx*4+0], k1 = k_ptr[tx*4+1], k2 = k_ptr[tx*4+2], k3 = k_ptr[tx*4+3];
        const float qi = rsqrt(simd_sum(q0*q0 + q1*q1 + q2*q2 + q3*q3) + args.eps);
        const float ki = rsqrt(simd_sum(k0*k0 + k1*k1 + k2*k2 + k3*k3) + args.eps);
        q0 *= qi; q1 *= qi; q2 *= qi; q3 *= qi;
        k0 *= ki; k1 *= ki; k2 *= ki; k3 *= ki;

        const float sp = alpha[(ulong)t * args.heads_v + h] + dt_bias[h];
        const float g = (sp > 20.0f ? sp : log(1.0f + exp(sp))) * a_log[h];
        const float g_exp = exp(g);
        const float beta = 1.0f / (1.0f + exp(-beta_raw[(ulong)t * args.heads_v + h]));

        ls0 *= g_exp; ls1 *= g_exp; ls2 *= g_exp; ls3 *= g_exp;
        const float s_k = simd_sum(ls0*k0 + ls1*k1 + ls2*k2 + ls3*k3);
        const float dv = (v_ptr[i20] - s_k) * beta;
        ls0 += k0 * dv; ls1 += k1 * dv; ls2 += k2 * dv; ls3 += k3 * dv;
        const float y = simd_sum(ls0*q0 + ls1*q1 + ls2*q2 + ls3*q3) * rsqrt((float)d);
        if (tx == 0) {
            out[(ulong)t * args.heads_v * d + h * d + i20] = y;
        }
        if (slot_total != 0) {
            device float* slot = slots + (ulong)t * slot_total + args.state_off
                + (ulong)h * d * d + (ulong)i20 * d;
            slot[tx*4+0] = ls0; slot[tx*4+1] = ls1; slot[tx*4+2] = ls2; slot[tx*4+3] = ls3;
        }
    }
    s_ptr[tx*4+0] = ls0; s_ptr[tx*4+1] = ls1; s_ptr[tx*4+2] = ls2; s_ptr[tx*4+3] = ls3;
}

// The gated per-head rmsnorm over the recurrence output, times silu(z),
// batched: one threadgroup per (head, token), 128 threads.
kernel void gdn_out_norm_batch(
    device float* y [[buffer(0)]],         // [m][heads_v * d]
    device const float* z [[buffer(1)]],   // [m][heads_v * d]
    constant float* w [[buffer(2)]],
    constant uint& heads_v [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[8];
    const uint h = tgp.x;
    const uint t = tgp.y;
    const uint row = t * heads_v * d + h * d;
    const float yv = y[row + tid];
    const float ps = simd_sum(yv * yv);
    if (lane == 0) {
        partial[sg] = ps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < d / 32; i++) {
        total += partial[i];
    }
    const float scale = rsqrt(total / (float)d + eps);
    const float zv = z[row + tid];
    y[row + tid] = yv * scale * w[tid] * (zv / (1.0f + exp(-zv)));
}

// Batched qk_prep for head_dim 256 (qwen35moe): one threadgroup per
// (head, row); the fused output gate splits out of the q projection rows.
struct QkPrepBatch256Args {
    uint n_heads;
    uint n_kv_heads;
    uint kv_dim;
    float eps;
    uint base;          // first row's position
    uint has_qk_norm;
    uint rot_dim;
    uint gate_in_q;
    ulong k_base;
    ulong v_base;
};

kernel void qk_prep_batch256(
    device float* q [[buffer(0)]],        // [m][2*256] interleaved when gate_in_q
    device float* k [[buffer(1)]],        // [m][kv_dim]
    device const float* v [[buffer(2)]],
    device half* cache [[buffer(3)]],
    constant QkPrepBatch256Args& a [[buffer(4)]],
    constant float* qw [[buffer(5)]],
    constant float* kw [[buffer(6)]],
    device const float2* rope [[buffer(7)]],  // [m][rot_dim/2]
    device float* q_out [[buffer(8)]],    // [m][n_heads*256] deinterleaved
    device float* gate_out [[buffer(9)]], // [m][n_heads*256] raw gate rows
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint hd = 256;
    const uint hid = tgp.x;
    const uint row = tgp.y;
    const uint pos = a.base + row;
    if (hid >= a.n_heads + a.n_kv_heads) {
        const uint kvh = hid - a.n_heads - a.n_kv_heads;
        device const float* src = v + (ulong)row * a.kv_dim + kvh * hd;
        device half* dst = cache + a.v_base + (ulong)pos * a.kv_dim + kvh * hd;
        for (uint e = lane; e < hd; e += 32) {
            dst[e] = (half)src[e];
        }
        return;
    }
    const bool is_q = hid < a.n_heads;
    const uint src_stride = (is_q && a.gate_in_q != 0) ? 2 * hd : hd;
    device float* xh = is_q
        ? (q + (ulong)row * a.n_heads * src_stride + hid * src_stride)
        : (k + (ulong)row * a.kv_dim + (hid - a.n_heads) * hd);
    constant float* w = is_q ? qw : kw;
    float e[8];
    for (uint j = 0; j < 8; j++) {
        e[j] = xh[lane + j * 32];
    }
    if (a.gate_in_q != 0 && is_q) {
        device float* gs = q + (ulong)row * a.n_heads * 2 * hd + hid * 2 * hd + hd;
        device float* gd = gate_out + (ulong)row * a.n_heads * hd + hid * hd;
        for (uint j = 0; j < 8; j++) {
            gd[lane + j * 32] = gs[lane + j * 32];
        }
    }
    if (a.has_qk_norm != 0) {
        float ss = 0.0f;
        for (uint j = 0; j < 8; j++) {
            ss += e[j] * e[j];
        }
        const float scale = rsqrt(simd_sum(ss) / (float)hd + a.eps);
        for (uint j = 0; j < 8; j++) {
            e[j] *= scale * w[lane + j * 32];
        }
    }
    // Partial rotary: rot_h == 32 pairs (lane, lane + 32) of the first 64.
    const uint rot_h = a.rot_dim / 2;
    if (rot_h == 32 && lane < 32) {
        const float2 r0 = rope[(ulong)row * rot_h + lane];
        const float t0 = e[0] * r0.y - e[1] * r0.x;
        const float t1 = e[0] * r0.x + e[1] * r0.y;
        e[0] = t0; e[1] = t1;
    }
    if (is_q) {
        device float* dst = q_out + (ulong)row * a.n_heads * hd + hid * hd;
        for (uint j = 0; j < 8; j++) {
            dst[lane + j * 32] = e[j];
        }
    } else {
        const uint kvh = hid - a.n_heads;
        device half* dst = cache + a.k_base + (ulong)pos * a.kv_dim + kvh * hd;
        for (uint j = 0; j < 8; j++) {
            dst[lane + j * 32] = (half)e[j];
        }
    }
}

// Causal attention over every chunk row for head_dim 256: one threadgroup
// per (q head, row), row i covering base + i + 1 positions (attend_s32_256's
// body with the row from grid.y).
kernel void attend_rows256(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& base [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant ulong& k_base [[buffer(9)]],
    constant ulong& v_base [[buffer(10)]],
    constant uint& n_heads [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t_acc[16 * 256];
    threadgroup float t_max[16];
    threadgroup float t_den[16];

    const uint qh = tgp.x;
    const uint row = tgp.y;
    const uint n_pos = base + row + 1;
    const uint kv_head = qh / group;
    device const float4* qh4 =
        (device const float4*)(q + ((ulong)row * n_heads + qh) * head_dim);
    const float4 qv0 = qh4[lane];
    const float4 qv1 = qh4[lane + 32];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint t = sg; t < n_pos; t += 16u) {
        const ulong at = (ulong)t * kv_dim + (ulong)kv_head * head_dim;
        const float4 ka = float4(((device const half4*)(k + k_base + at))[lane]);
        const float4 kb = float4(((device const half4*)(k + k_base + at))[lane + 32]);
        const float s = simd_sum(dot(qv0, ka) + dot(qv1, kb)) * scale;
        if (s > run_max) {
            const float r = exp(run_max - s);
            if (run_den > 0.0f) {
                acc0 *= r; acc1 *= r;
                run_den *= r;
            } else {
                acc0 = float4(0.0f); acc1 = float4(0.0f);
            }
            run_max = s;
        }
        const float p = exp(s - run_max);
        run_den += p;
        const float4 va = float4(((device const half4*)(v + v_base + at))[lane]);
        const float4 vb = float4(((device const half4*)(v + v_base + at))[lane + 32]);
        acc0 += p * va;
        acc1 += p * vb;
    }

    ((threadgroup float4*)(t_acc + sg * 256))[lane] = acc0;
    ((threadgroup float4*)(t_acc + sg * 256))[lane + 32] = acc1;
    if (lane == 0) {
        t_max[sg] = run_max;
        t_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float m = -INFINITY;
        for (uint s = 0; s < 16u; s++) m = max(m, t_max[s]);
        float num = 0.0f;
        float den = 0.0f;
        for (uint s = 0; s < 16u; s++) {
            const float wgt = t_den[s] > 0.0f ? exp(t_max[s] - m) : 0.0f;
            num += t_acc[s * 256 + tid] * wgt;
            den += t_den[s] * wgt;
        }
        out[((ulong)row * n_heads + qh) * head_dim + tid] = num / max(den, FLT_MIN);
    }
}

// qwen35moe prefill: overwrite the shared expert's CSR weight (a constant 1
// from route_scatter) with sigmoid(gate_logit[token]).
kernel void route_patch_shared_w(
    device float* hit_w [[buffer(0)]],
    device const float* gate_logits [[buffer(1)]],
    constant RouteParams& p [[buffer(2)]],
    uint token [[thread_position_in_grid]])
{
    if (token >= p.m || p.shared == 0) return;
    const uint stride = p.n_used + p.shared;
    const float g = gate_logits[token];
    hit_w[token * stride + p.n_used] = 1.0f / (1.0f + exp(-g));
}
"#;
