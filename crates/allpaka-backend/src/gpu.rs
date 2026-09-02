//! The Metal path: fused dequant-matvec kernels on Apple GPUs.
//!
//! Only matrix-vector products go to the GPU - they are where every byte and
//! every FLOP of decode lives. Norms, RoPE and softmax stay on the CPU: they
//! touch kilobytes, and a dispatch costs more than they do.
//!
//! Weights are never copied. The GGUF mmap is wrapped in `bytesNoCopy` shared
//! buffers (Apple Silicon memory is unified; the GPU reads the same pages the
//! page cache holds), and kernels address tensors by byte offset. A mapping
//! larger than the device's `maxBufferLength` is covered by several
//! overlapping chunk buffers - one giant buffer would silently come back nil
//! and every kernel would read garbage, which is exactly how an 88 GiB model
//! produced NaN logits while a 17 GiB one verified clean.
//!
//! The kernels mirror `quantmat.rs` line for line. The CPU implementation is
//! the reference; `verify` against llama.cpp arbitrates both.

#![cfg(target_os = "macos")]

use metal::foreign_types::ForeignType;
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use allpaka_gguf::GgmlType;

/// Where GPU wall time goes, accumulated across every call. The split that
/// matters: `wait_ns` is dead time the CPU spends blocked on the GPU, and its
/// ratio to `calls` is the per-round-trip cost every extra command buffer pays.
static CALLS: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// y_arena f32 offset of the router logits the last fused attention block
/// produced, when GPU-side routing skipped the CPU readback.
static PF_ROUTER_AT: AtomicU64 = AtomicU64::new(u64::MAX);
static ENCODE_NS: AtomicU64 = AtomicU64::new(0);
static WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// GPU-side execution time (GPUEndTime - GPUStartTime summed over buffers)
/// and the scheduling delay before it (GPUStartTime - kernelStartTime).
/// The difference between WAIT_NS and their sum is time the CPU spent
/// blocked while the GPU had nothing of ours to run - the number that says
/// whether decode is bound by kernels or by getting work to them.
static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);
static SCHED_NS: AtomicU64 = AtomicU64::new(0);

/// `(gpu_busy_ns, sched_ns)` since process start.
pub fn gpu_time_stats() -> (u64, u64) {
    (
        GPU_BUSY_NS.load(Ordering::Relaxed),
        SCHED_NS.load(Ordering::Relaxed),
    )
}

/// Read the completed buffer's own timestamps. CFTimeIntervals in seconds on
/// a shared clock, so differences are valid; zeros (no GPU work) are skipped.
fn note_gpu_times(cmd: &metal::CommandBufferRef) {
    use objc::{msg_send, sel, sel_impl};
    let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
    let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
    let ks: f64 = unsafe { msg_send![cmd, kernelStartTime] };
    if ge > gs && gs > 0.0 {
        GPU_BUSY_NS.fetch_add(((ge - gs) * 1e9) as u64, Ordering::Relaxed);
        if gs > ks && ks > 0.0 {
            SCHED_NS.fetch_add(((gs - ks) * 1e9) as u64, Ordering::Relaxed);
        }
    }
}

/// `(calls, dispatches, encode_ns, wait_ns)` since process start.
pub fn stats() -> (u64, u64, u64, u64) {
    (
        CALLS.load(Ordering::Relaxed),
        DISPATCHES.load(Ordering::Relaxed),
        ENCODE_NS.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed),
    )
}

pub const KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

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
    for (uint off = LPR / 2; off > 0; off >>= 1) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

inline float half_at(device const uchar* p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_type<half>(u);
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
        device const uchar* qs = blk + 6;
        float s[MAXT] = {0};
        for (uint jj = 0; jj < 32; jj++) {
            uint nib = jj < 16 ? (qs[jj] & 0xF) : (qs[jj - 16] >> 4);
            int q = (int)(nib | (((qh >> jj) & 1) << 4)) - 16;
            for (uint i = 0; i < TILE; i++) {
                s[i] += float(q) * x[i * n_in + b * 32 + jj];
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
                total[i] += d * sc1 * dot_lo[i] - dmin * mn1 * sum_lo[i];
                total[i] += d * sc2 * dot_hi[i] - dmin * mn2 * sum_hi[i];
            }
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
                        total[i] += dl * dotq - ml * (xs.x + xs.y + xs.z + xs.w);
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
    uint tid [[thread_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // Rows per SIMD group rides on the LPR function constant (NR0 = 32/LPR)
    // so the host can sweep the occupancy-vs-reuse tradeoff: more rows reuse
    // the registered activations more but launch fewer SIMD groups.
    const uint NR0 = 32 / LPR;
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    uint flat = (tid / 32) * NR0;
    // Whole-group overshoot clamps to row 0 and drops the write; a partial
    // tail keeps its live rows via the per-row write guard below.
    bool any = flat < ycols;
    if (!any) flat = 0;
    // Live rows in this SIMD group. The row loop must stay fully unrolled
    // (a runtime break would demote sumf[] to memory), so tail rows still
    // compute - but their pointer step is zeroed so they re-read the last
    // live row instead of walking past the tensor, and the write guard
    // drops their result.
    uint nrows = any ? min(NR0, ycols - flat) : 0u;
    uint slot = INDEXED ? flat / n_out : 0;
    uint j0 = INDEXED ? flat % n_out : flat;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
    }
    uint nb = n_in / 256;
    ulong nb01 = (ulong)nb * 84;
    device const uchar* row0 = w + w_off + (ulong)j0 * nb01;

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

        for (uint row = 0; row < NR0; row++) {
            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            for (short i = 0; i < 8; i += 2) {
                acc1[0] += yl[i + 0] * (qs[i / 2] & 0x0003);
                acc2[0] += yl[i + 1] * (qs[i / 2] & 0x0300);
                acc1[1] += yl[i + 8] * (qs[i / 2] & 0x000c);
                acc2[1] += yl[i + 9] * (qs[i / 2] & 0x0c00);
                acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
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
            y[flat + row] = s;
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
                    float dl = d_all * (float)((int)(lo | (hi << 4)) - 32);
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
                        total[i] += dl * dotq;
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
    uint tid [[thread_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    uint flat = (tid / 32) * 2;
    bool any = flat < ycols;
    if (!any) flat = 0;
    uint nrows = any ? min(2u, ycols - flat) : 0u;
    uint slot = INDEXED ? flat / n_out : 0;
    uint j0 = INDEXED ? flat % n_out : flat;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
        if (SWIGLU_X) {
            xu += (ulong)slot * idx.x_stride;
        }
    }
    uint nb = n_in / 256;
    ulong nb01 = (ulong)nb * 110;
    device const uchar* row0 = w + w_off + (ulong)j0 * nb01;

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
        for (short l = 0; l < 8; ++l) {
            if (SWIGLU_X) {
                yl[l + 0] = sw1(y1[l + 0], u1[l + 0]);
                yl[l + 8] = sw1(y1[l + 16], u1[l + 16]);
                yl[l + 16] = sw1(y1[l + 32], u1[l + 32]);
                yl[l + 24] = sw1(y1[l + 48], u1[l + 48]);
            } else {
                yl[l + 0] = y1[l + 0];
                yl[l + 8] = y1[l + 16];
                yl[l + 16] = y1[l + 32];
                yl[l + 24] = y1[l + 48];
            }
        }
        device const uchar* blk = row0 + ib * 110;
        device const ushort* q = (device const ushort*)(blk + 32 + q_offset);
        device const ushort* h = (device const ushort*)(blk + l0);
        device const ushort* a = (device const ushort*)(blk + 96);
        device const uchar* dh = blk + 108;

        for (short row = 0; row < 2; ++row) {
            const float d_all = half_at(dh);

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il + 0];
            scales16[1] = a[il + 1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int qsv = q[l / 2];
                s1 += yl[l + 0] * (qsv & qm[il / 2][0]);
                s2 += yl[l + 1] * (qsv & qm[il / 2][1]);
                s3 += ((h[l / 2] & hm[0]) ? 0.0f : yl[l + 0]) + ((h[l / 2] & hm[1]) ? 0.0f : yl[l + 1]);
                s4 += yl[l + 16] * (qsv & qm[il / 2][2]);
                s5 += yl[l + 17] * (qsv & qm[il / 2][3]);
                s6 += ((h[l / 2] & hm[2]) ? 0.0f : yl[l + 16]) + ((h[l / 2] & hm[3]) ? 0.0f : yl[l + 17]);
            }
            float d1 = d_all * (s1 + (1.0f / 256.0f) * s2 - s3 * v1);
            float d2 = d_all * (s4 + (1.0f / 256.0f) * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int qsv = q[l / 2 + 8];
                s1 += yl[l + 8] * (qsv & qm[il / 2][0]);
                s2 += yl[l + 9] * (qsv & qm[il / 2][1]);
                s3 += ((h[l / 2 + 8] & hm[0]) ? 0.0f : yl[l + 8]) + ((h[l / 2 + 8] & hm[1]) ? 0.0f : yl[l + 9]);
                s4 += yl[l + 24] * (qsv & qm[il / 2][2]);
                s5 += yl[l + 25] * (qsv & qm[il / 2][3]);
                s6 += ((h[l / 2 + 8] & hm[2]) ? 0.0f : yl[l + 24]) + ((h[l / 2 + 8] & hm[3]) ? 0.0f : yl[l + 25]);
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
            y[flat + row] = s;
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
    uint tid [[thread_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;

    uint ycols = INDEXED ? n_out * idx.slots : n_out;
    uint flat = (tid / 32) * 2;
    bool any = flat < ycols;
    if (!any) flat = 0;
    uint nrows = any ? min(2u, ycols - flat) : 0u;
    uint slot = INDEXED ? flat / n_out : 0;
    uint j0 = INDEXED ? flat % n_out : flat;
    if (INDEXED) {
        w += ids[slot] * idx.stride;
        x += (ulong)slot * idx.x_stride;
    }
    uint nb = n_in / 256;
    ulong nb01 = (ulong)nb * 210;
    device const uchar* row0 = w + w_off + (ulong)j0 * nb01;

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
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = yv[l + 0];
            yl[4 * l + 1] = yv[l + 32];
            yl[4 * l + 2] = yv[l + 64];
            yl[4 * l + 3] = yv[l + 96];
        }

        for (uint row = 0; row < 2; ++row) {
            float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((char)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((char)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
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
            y[flat + row] = s;
        }
    }
}

// Port of llama.cpp's kernel_mul_mv_q4_K_f32 structure (MIT), same skeleton
// as matvec_q3_k_mv: 2 rows per SIMD group, activations registered once
// (yl low half, yh high half), nibble masks in place with 1/16 and 1/256
// folded into the scale multipliers. 16 threads per output row; INDEXED
// requires n_out % 2 == 0.
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
    uint tid [[thread_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    if (WAIT_X) {
        if (ltid == 0) {
            while (atomic_load_explicit(wait_flag, memory_order_relaxed) < wait_epoch) {
            }
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    const ushort kmask1 = 0x3f3f;
    const ushort kmask2 = 0x0f0f;
    const ushort kmask3 = 0xc0c0;

    uint gate_rows = INDEXED ? n_out * idx.slots : n_out;
    uint ycols = DUAL_GW ? gate_rows * 2 : gate_rows;
    uint flat = (tid / 32) * 2;
    bool any = flat < (ROWS ? gate_rows * idx.n_rows : ycols);
    if (!any) flat = 0;
    uint tok = ROWS ? flat / gate_rows : 0;
    // The second half of the flat rows belongs to the up matrix.
    device const uchar* wm = w;
    ulong woff = w_off;
    device float* yout = y;
    if (DUAL_GW && flat >= gate_rows) {
        flat -= gate_rows;
        wm = w2;
        woff = w2_off;
        yout = y2;
    }
    uint fr = ROWS ? flat - tok * gate_rows : flat;
    uint nrows = any ? min(2u, gate_rows - fr) : 0u;
    uint slot = INDEXED ? fr / n_out : 0;
    uint j0 = INDEXED ? fr % n_out : fr;
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
    float sumf[2] = {0.0f, 0.0f};

    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;
    device const float* u4 = xu + ix * 256 + 64 * iq + 8 * ir;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = SWIGLU_X ? sw1(y4[i + 0], u4[i + 0]) : y4[i + 0];
            sumy[0] += yl[i + 0];
            yl[i + 8] = SWIGLU_X ? sw1(y4[i + 32], u4[i + 32]) : y4[i + 32];
            sumy[1] += yl[i + 8];
            yh[i + 0] = SWIGLU_X ? sw1(y4[i + 128], u4[i + 128]) : y4[i + 128];
            sumy[2] += yh[i + 0];
            yh[i + 8] = SWIGLU_X ? sw1(y4[i + 160], u4[i + 160]) : y4[i + 160];
            sumy[3] += yh[i + 8];
        }
        device const uchar* blk = row0 + ib * 144;
        device const ushort* sc = (device const ushort*)(blk + 4) + iq;
        device const ushort* q1 = (device const ushort*)(blk + 16) + 16 * iq + 4 * ir;
        device const uchar* dh = blk;

        for (uint row = 0; row < 2; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
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

    for (uint row = 0; row < 2; row++) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && row < nrows) {
            yout[(ROWS ? tok * idx.y_row_stride : 0u) + fr + row] = s;
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
                total[i] += d * sc1 * dot_lo[i] - dmin * mn1 * sum_lo[i];
                total[i] += d * sc2 * dot_hi[i] - dmin * mn2 * sum_hi[i];
            }
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
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
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
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
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
    for (uint e = 0; e < p.n_expert; e++) {
        // Sigmoid gating scores each expert independently; softmax selects
        // by the raw logit (monotonic with the softmax score).
        float s = p.sigmoid ? 1.0f / (1.0f + exp(-row[e])) : row[e];
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
    if (!p.sigmoid) {
        for (uint i = 0; i < p.n_used; i++) {
            mx = max(mx, row[be[i]]);
        }
    }
    for (uint i = 0; i < p.n_used; i++) {
        sc[i] = p.sigmoid ? 1.0f / (1.0f + exp(-row[be[i]]))
                          : exp(row[be[i]] - mx);
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
    uint has_shared;
    ulong sh_gate_off;
    ulong sh_up_off;
    ulong sh_down_off;
};

inline void mega_sync(device atomic_uint* ctr, uint target, uint ltid) {
    threadgroup_barrier(mem_flags::mem_device);
    if (ltid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        while (atomic_load_explicit(ctr, memory_order_relaxed) < target) {
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device);
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
        for (short row = 0; row < 2; ++row) {
            const float d_all = half_at(dh);
            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il + 0];
            scales16[1] = a[il + 1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;
            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int qsv = q[l / 2];
                s1 += yl[l + 0] * (qsv & qm[il / 2][0]);
                s2 += yl[l + 1] * (qsv & qm[il / 2][1]);
                s3 += ((h[l / 2] & hm[0]) ? 0.0f : yl[l + 0]) + ((h[l / 2] & hm[1]) ? 0.0f : yl[l + 1]);
                s4 += yl[l + 16] * (qsv & qm[il / 2][2]);
                s5 += yl[l + 17] * (qsv & qm[il / 2][3]);
                s6 += ((h[l / 2] & hm[2]) ? 0.0f : yl[l + 16]) + ((h[l / 2] & hm[3]) ? 0.0f : yl[l + 17]);
            }
            float d1 = d_all * (s1 + (1.0f / 256.0f) * s2 - s3 * v1);
            float d2 = d_all * (s4 + (1.0f / 256.0f) * s5 - s6 * v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);
            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int qsv = q[l / 2 + 8];
                s1 += yl[l + 8] * (qsv & qm[il / 2][0]);
                s2 += yl[l + 9] * (qsv & qm[il / 2][1]);
                s3 += ((h[l / 2 + 8] & hm[0]) ? 0.0f : yl[l + 8]) + ((h[l / 2 + 8] & hm[1]) ? 0.0f : yl[l + 9]);
                s4 += yl[l + 24] * (qsv & qm[il / 2][2]);
                s5 += yl[l + 25] * (qsv & qm[il / 2][3]);
                s6 += ((h[l / 2 + 8] & hm[2]) ? 0.0f : yl[l + 24]) + ((h[l / 2 + 8] & hm[3]) ? 0.0f : yl[l + 25]);
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
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    device atomic_uint* ctr = (device atomic_uint*)(y + a.ctr_at);
    uint sgid = tgid * 8 + sg;
    uint total_sg = a.n_tg * 8;
    uint gthread = tgid * 256 + ltid;
    uint total_threads = a.n_tg * 256;
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
    mega_sync(ctr, base + 1 * a.n_tg, ltid);

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
    mega_sync(ctr, base + 2 * a.n_tg, ltid);

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
    }
    mega_sync(ctr, base + 3 * a.n_tg, ltid);

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
    mega_sync(ctr, base + 4 * a.n_tg, ltid);

    // s4: swiglu in place over the gate half.
    {
        device float* g = y + a.gate_at;
        device const float* u = y + a.up_at;
        uint n = (a.n_used + a.has_shared) * a.ffn;
        for (uint i = gthread; i < n; i += total_threads) {
            float gv = g[i];
            g[i] = u[i] * (gv / (1.0f + exp(-gv)));
        }
    }
    mega_sync(ctr, base + 5 * a.n_tg, ltid);

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
    mega_sync(ctr, base + 6 * a.n_tg, ltid);

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
            if (a.has_shared != 0) {
                acc += downo[a.n_used * a.hidden + i];
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

// delta[h] = sum over slots of wts[s] * down[s * n + h]. When sig_last is
// set the last slot's weight is the RAW logit of qwen35moe's shared-expert
// gate and goes through a sigmoid first (every other slot is a finished
// router weight).
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
    float wl = 0.0f;
    if (sig_last != 0) {
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
    float wl = 0.0f;
    if (sig_last != 0) {
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

// Stage two: one thread folds the per-threadgroup pairs into the winner.
kernel void argmax_final(
    device const float* parts [[buffer(0)]],
    constant uint& nparts [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) {
        return;
    }
    float best = parts[0];
    uint bidx = as_type<uint>(parts[1]);
    for (uint i = 1; i < nparts; i++) {
        float v = parts[i * 2];
        uint idx = as_type<uint>(parts[i * 2 + 1]);
        if (v > best || (v == best && idx > bidx)) {
            best = v;
            bidx = idx;
        }
    }
    *out = bidx;
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
    device const float* f = (device const float*)row + k0;
    for (uint i = 0; i < 16; i++) {
        out[i] = (half)f[i];
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
    auto stage_block = [&](uint b, uint blk) {
        threadgroup half* sa = sab[b];
        threadgroup half* sb = sa + 64 * NK;
        uint chunk0 = blk * (NK / 16);
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
        device const float* byk = by + (ulong)blk * NK;
        for (short c = 0; c < NK / 32; c++) {
            const short sx = tiitg % 4 + 4 * c;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
                half2x4(*(device const float2x4*)(byk + 32 * c));
        }
    };
    stage_block(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint blk = 0; blk < nblk; blk++) {
        const uint cur = blk & 1;
        if (blk + 1 < nblk) {
            // Shadow-stage the next block; its device loads overlap the MMA.
            stage_block(1 - cur, blk + 1);
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
    auto stage_block = [&](uint b, uint blk) {
        threadgroup half* sa = sab[b];
        threadgroup half* sb = sa + 64 * NK;
        uint chunk0 = blk * (NK / 16);
        for (short c = 0; c < NK / 32; c++) {
            half4x4 temp_a;
            uint cg = chunk0 + il0 * (NK / 32) + c;
            dqll_q8_0(bx0 + cg / 2, (short)(cg % 2), temp_a);
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
        device const float* byk = by + (ulong)blk * NK;
        for (short c = 0; c < NK / 32; c++) {
            const short sx = tiitg % 4 + 4 * c;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
                half2x4(*(device const float2x4*)(byk + 32 * c));
        }
    };
    stage_block(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint blk = 0; blk < nblk; blk++) {
        const uint cur = blk & 1;
        if (blk + 1 < nblk) {
            // Shadow-stage the next block; its device loads overlap the MMA.
            stage_block(1 - cur, blk + 1);
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
    auto stage_block = [&](uint b, uint blk) {
        threadgroup half* sa = sab[b];
        threadgroup half* sb = sa + 64 * NK;
        uint chunk0 = blk * (NK / 16);
        for (short c = 0; c < NK / 32; c++) {
            half4x4 temp_a;
            uint cg = chunk0 + il0 * (NK / 32) + c;
            dqll_q5_k(bx0 + cg / 16, (short)(cg % 16), temp_a);
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
        device const float* byk = by + (ulong)blk * NK;
        for (short c = 0; c < NK / 32; c++) {
            const short sx = tiitg % 4 + 4 * c;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4*)(sb + 64 * ib + 8 * ly) =
                half2x4(*(device const float2x4*)(byk + 32 * c));
        }
    };
    stage_block(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint blk = 0; blk < nblk; blk++) {
        const uint cur = blk & 1;
        if (blk + 1 < nblk) {
            stage_block(1 - cur, blk + 1);
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
DEFINE_MM_LL_NK(mmll_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 32)
DEFINE_MM_LL_NK(mmll_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 32)
DEFINE_MM_LL_NK(mmll_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 32)
DEFINE_MM_LL_NK(mmll_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 32)
DEFINE_MM_LL_NK(mmll_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 32)
DEFINE_MM_LL_NK(mmll_f32, BlkF32, 1, dqll_f32, (ulong)n_in * 4, 32)
DEFINE_MM_LL_NK(mm64ll_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 64)
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
    auto stage_block = [&](uint b, uint blk) {                                 \
        threadgroup half* sa = sab[b];                                         \
        threadgroup half* sb = sa + 64 * NK;                                   \
        uint chunk0 = blk * (NK / 16);                                         \
        for (short c = 0; c < NK / 32; c++) {                                  \
            half4x4 temp_a;                                                    \
            uint cg = chunk0 + il0 * (NK / 32) + c;                            \
            dqll_q4_k(bx0 + cg / 16, (short)(cg % 16), temp_a);                \
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
        device const float* byk = by + (ulong)blk * NK;                        \
        device const float* by2k = by2 + (ulong)blk * NK;                      \
        for (short c = 0; c < NK / 32; c++) {                                  \
            const short sx = tiitg % 4 + 4 * c;                                \
            const short sy = (tiitg / 4) / 8;                                  \
            const short ly = (tiitg / 4) % 8;                                  \
            const short ib = 4 * sx + sy;                                      \
            if (SWIGLU) {                                                      \
                float2x4 gv = *(device const float2x4*)(byk + 32 * c);         \
                float2x4 uv = *(device const float2x4*)(by2k + 32 * c);        \
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
                    half2x4(*(device const float2x4*)(byk + 32 * c));          \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    stage_block(0, 0);                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint blk = 0; blk < nblk; blk++) {                                    \
        const uint cur = blk & 1;                                              \
        if (blk + 1 < nblk) {                                                  \
            stage_block(1 - cur, blk + 1);                                     \
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
            uint pb = sgitg + 4 * cc;
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
            uint pb = sgitg + 4 * cc;
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

/// One `bytesNoCopy` window over the weight mapping. Chunks overlap by more
/// than any single tensor, so every request fits wholly inside some chunk.
struct WeightChunk {
    buf: Buffer,
    /// Absolute address of this chunk's first byte. Chunks from different
    /// attached mappings coexist in one list; an absolute range is what lets
    /// a weight slice find its window no matter which model it belongs to.
    start: usize,
    len: usize,
}

pub struct Gpu {
    device: Device,
    queue: CommandQueue,
    /// One pipeline per (kernel, tile, lanes-per-row): both are function
    /// constants, so each shape is its own specialisation. The full cross
    /// product is 4 tiles x 6 lane counts x 6 formats, and a given model uses
    /// a handful of those, so they are compiled on first use rather than at
    /// attach. `swiglu` ignores both and is stored under (1, 1).
    pipelines: HashMap<(&'static str, usize, usize), ComputePipelineState>,
    library: metal::Library,
    chunks: Vec<WeightChunk>,
    /// The queue's MTLResidencySet, retained as a raw pointer (usize keeps
    /// the struct Send; access is serialised by the surrounding Mutex).
    residency: Option<usize>,
    /// Reusable staging for activations in and outputs back: bump-allocated
    /// slices bound by buffer offset, so a dispatch creates no buffer objects.
    x_arena: Buffer,
    x_cap: usize,
    y_arena: Buffer,
    y_cap: usize,
    /// Fused-prefill residual stream and normed activations: xs lives here
    /// for the whole chunk, so norms, residual adds and the MoE combine
    /// never round-trip through the CPU.
    pf_x: Buffer,
    pf_x_cap: usize,
    pf_hs: Buffer,
    pf_hs_cap: usize,
    /// GPU-side MoE routing scratch: counters, the group table, picks,
    /// gathered token ids and the combine CSR, all produced on-device.
    route_buf: Buffer,
    route_cap: usize,
    /// The bias pointer (0 = none) route_buf's counters were last zeroed
    /// and staged for; route_scan self-cleans them on the GPU between
    /// layers, so CPU re-staging is needed only on realloc or a bias change
    /// (per-layer CPU writes would race the deferred buffers).
    route_staged: Option<usize>,
    /// Fused-prefill deferred commits: layer buffers chain through this
    /// event instead of one CPU wait per layer. `pf_ev_val` is the last
    /// signalled value; `pf_pending` holds committed buffers until their
    /// GPU clock stats are reaped (bounded run-ahead).
    pf_ev: metal::SharedEvent,
    pf_ev_val: u64,
    pf_pending: std::collections::VecDeque<metal::CommandBuffer>,
    /// ALLPAKA_PF_ONEBUF: one-buffer prefill SEGMENTS. The shared command
    /// buffer is armed lazily by the first eligible layer of the chunk
    /// (chained behind the last committed buffer through `pf_ev`) and
    /// sealed - committed chained - by the first layer that cannot join
    /// it, so e.g. GLM's leading Dense layer splits the chunk into
    /// segments instead of disabling the mode. A still-open segment is
    /// committed once from prefill_end. Each stage still gets its own
    /// sequential encoder.
    pf_obuf_cmd: Option<metal::CommandBuffer>,
    /// ALLPAKA_GPU_COUNTERS=1: per-dispatch GPUTimestamp sampling through
    /// the fused prefill (the only counter set this device exposes). Index
    /// and labels are Cell/RefCell so encode closures borrowing `gpu`
    /// immutably can still stamp.
    cstamps: Option<metal::CounterSampleBuffer>,
    cstamps_dst: Option<metal::Buffer>,
    cstamp_idx: std::cell::Cell<usize>,
    cstamp_labels: std::cell::RefCell<Vec<&'static str>>,
}

/// route_buf layout in f32/u32 units (4-byte words).
struct RouteLayout {
    counts: usize,
    counters: usize,
    table: usize,
    picks: usize,
    pickw: usize,
    tok: usize,
    hit_row: usize,
    hit_w: usize,
    tok_off: usize,
    logits: usize,
    bias: usize,
    total: usize,
}

fn route_layout(m: usize, n_expert: usize, n_used: usize, shared: bool) -> RouteLayout {
    let align = |n: usize| (n + 63) & !63;
    let hits = n_used + shared as usize;
    let counts = 0;
    let counters = counts + align(n_expert);
    let table = counters + align(n_expert);
    let picks = table + align(n_expert * 3);
    let pickw = picks + align(m * n_used);
    let tok = pickw + align(m * n_used);
    let hit_row = tok + align(m * n_used);
    let hit_w = hit_row + align(m * hits);
    let tok_off = hit_w + align(m * hits);
    let logits = tok_off + align(m + 1);
    let bias = logits + align(m * n_expert);
    let total = bias + align(n_expert);
    RouteLayout {
        counts,
        counters,
        table,
        picks,
        pickw,
        tok,
        hit_row,
        hit_w,
        tok_off,
        logits,
        bias,
        total,
    }
}

// Metal objects are reference-counted ObjC objects; the global below is
// guarded by a Mutex, so cross-thread use is serialised.
unsafe impl Send for Gpu {}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

/// Wrap a model's mmap for the GPU. Call once per opened file; every later
/// `QuantMat::matmul` whose bytes live inside an attached mapping runs on
/// Metal automatically. Several mappings can be attached - speculative
/// decode runs the draft and the target on the same device - and each call
/// adds windows for its mapping. Returns false when no Metal device exists.
pub fn attach(mapping: &[u8]) -> bool {
    // Diagnostic hook: force the pure CPU path to bisect GPU-vs-engine bugs.
    if std::env::var_os("ALLPAKA_NO_GPU").is_some() {
        return false;
    }
    let Some(cell) = GPU.get_or_init(|| init_device().map(Mutex::new)) else {
        return false;
    };
    let mut gpu = match cell.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if gpu.covers(mapping.as_ptr() as usize, mapping.len()) {
        return true; // already attached (tests attach the same region twice)
    }
    gpu.add_mapping(mapping)
}

pub fn is_attached() -> bool {
    GPU.get().map_or(false, Option::is_some)
}

/// Runtime state that determines whether large mapped models avoid Metal's
/// per-submit residency scan. The values are reported by benchmarks so a
/// missing residency set cannot masquerade as slow kernels.
pub fn residency_status() -> (usize, bool) {
    let Some(Some(cell)) = GPU.get() else {
        return (0, false);
    };
    cell.lock()
        .map(|gpu| (gpu.chunks.len(), gpu.residency.is_some()))
        .unwrap_or((0, false))
}

/// Largest request a chunked mapping can serve; chunks overlap by this much.
/// Tensors in real GGUFs top out in the hundreds of megabytes (the 235B's
/// biggest is ~0.5 GiB), so 2 GiB of overlap leaves a wide margin.
const CHUNK_OVERLAP: usize = 2 << 30;

fn init_device() -> Option<Gpu> {
    let device = Device::system_default()?;
    let library = device
        .new_library_with_source(KERNELS, &metal::CompileOptions::new())
        .map_err(|e| eprintln!("metal: kernel compile failed, staying on CPU: {e}"))
        .ok()?;
    let mut pipelines = HashMap::new();
    // These two take no specialisation; they are stored under the unit shape.
    for name in [
        "swiglu",
        "attend",
        "attend_s8",
        "attend_s16",
        "attend_s32",
        "attend_mv",
        "attend_split",
        "attend_merge",
        "qk_prep",
        "qk_prep_batch",
        "attend_rows",
        "attend_rows_t4",
        "attend_rows_t8",
        "attend_mm",
        "residual_norm",
        "residual_norm_sig",
        "norm_rows",
        "combine_rows",
        "softmax_topk",
        "sigmoid_topk",
        "router_topk",
        "moe_combine",
        "moe_combine_rows",
        "combine_resnorm",
        "resnorm_router",
        "resnorm_router_rows",
        "mmllr64_q4_k",
        "mmllp_q4_k",
        "mmllp_q8_0",
        "mmllpg_id_q4_k",
        "mmllps_id_q4_k",
        "route_pick",
        "route_scan",
        "route_scatter",
        "argmax_f32",
        "argmax_final",
        "copy_f32",
        "gdn_conv",
        "gdn_step",
        "gdn_out_norm",
        "qk_prep256",
        "attend_s32_256",
        "attend_split_256",
        "attn_gate_mul",
        "gdn_conv_batch",
        "gdn_step_batch",
        "gdn_out_norm_batch",
        "qk_prep_batch256",
        "attend_rows256",
        "attend_mm256",
        "route_patch_shared_w",
    ] {
        let f = library
            .get_function(name, None)
            .map_err(|e| eprintln!("metal: get_function {name} failed: {e}"))
            .ok()?;
        let p = device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| eprintln!("metal: pipeline {name} failed: {e}"))
            .ok()?;
        pipelines.insert((name, 1, 1), p);
    }

    // Arenas start big enough for a decode layer and grow on demand.
    let x_arena = device.new_buffer(1 << 22, MTLResourceOptions::StorageModeShared);
    let y_arena = device.new_buffer(1 << 22, MTLResourceOptions::StorageModeShared);
    let pf_x = device.new_buffer(1 << 12, MTLResourceOptions::StorageModeShared);
    let pf_hs = device.new_buffer(1 << 12, MTLResourceOptions::StorageModeShared);
    let route_buf = device.new_buffer(1 << 12, MTLResourceOptions::StorageModeShared);
    let pf_ev = device.new_shared_event();

    Some(Gpu {
        queue: device.new_command_queue(),
        device,
        pipelines,
        library,
        chunks: Vec::new(),
        residency: None,
        x_arena,
        x_cap: 1 << 22,
        y_arena,
        y_cap: 1 << 22,
        pf_x,
        pf_x_cap: 1 << 12,
        pf_hs,
        pf_hs_cap: 1 << 12,
        route_buf,
        route_cap: 1 << 12,
        route_staged: None,
        pf_ev,
        pf_ev_val: 0,
        pf_pending: std::collections::VecDeque::new(),
        pf_obuf_cmd: None,
        cstamps: None,
        cstamps_dst: None,
        cstamp_idx: std::cell::Cell::new(0),
        cstamp_labels: std::cell::RefCell::new(Vec::new()),
    })
}

impl Gpu {
    /// Whether `[addr, addr + len)` is wholly inside some attached window.
    fn covers(&self, addr: usize, len: usize) -> bool {
        self.chunk_for(addr, len).is_some()
    }

    /// Cover one model's mmap with overlapping `bytesNoCopy` windows.
    ///
    /// The mmap base is page-aligned by definition; lengths must be whole
    /// pages for bytesNoCopy. The kernel maps whole pages anyway, so a
    /// rounded tail is readable zeroes, never touched by valid offsets.
    ///
    /// One buffer cannot exceed the device's maxBufferLength (36 GiB class
    /// on Apple Silicon). newBufferWithBytesNoCopy does NOT error past it -
    /// it returns nil, and kernels then read garbage. Large mappings
    /// therefore get several overlapping chunk windows, and every dispatch
    /// picks the window that wholly contains its tensor.
    fn add_mapping(&mut self, mapping: &[u8]) -> bool {
        let page = 16384usize;
        let base = mapping.as_ptr() as usize;
        let aligned_len = mapping.len().div_ceil(page) * page;
        // Stay well under the device's maxBufferLength: a buffer AT the
        // limit (80.6 GiB here) comes back non-nil yet reads wrong - the
        // 235B produced garbage logits until its mapping was split into
        // windows this size. Extra windows cost nothing; they alias pages.
        const WINDOW_CAP: usize = 32 << 30;
        let mut max_buf = (self.device.max_buffer_length() as usize / page) * page;
        max_buf = max_buf.min(WINDOW_CAP);
        // Test hook: cap the window size to exercise multi-window mapping on
        // small regions (the real trigger needs an 80+ GiB file).
        if let Ok(cap) = std::env::var("ALLPAKA_GPU_WINDOW_GIB") {
            if let Ok(gib) = cap.parse::<usize>() {
                max_buf = max_buf.min(gib << 30);
            }
        }
        if max_buf == 0 {
            return false;
        }
        let step = if max_buf > CHUNK_OVERLAP {
            max_buf - CHUNK_OVERLAP
        } else {
            max_buf / 2
        };
        let step = (step / page).max(1) * page;

        let mut added = Vec::new();
        let mut start = 0usize;
        loop {
            let len = max_buf.min(aligned_len - start);
            let buf = self.device.new_buffer_with_bytes_no_copy(
                (base + start) as *const _,
                len as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            if buf.as_ptr().is_null() {
                eprintln!(
                    "metal: bytesNoCopy declined a {:.1} GiB window, staying on CPU",
                    len as f64 / (1u64 << 30) as f64
                );
                return false;
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
            "metal: {} attached, {:.1} GiB of weights shared with the GPU{}",
            self.device.name(),
            mapping.len() as f64 / (1u64 << 30) as f64,
            if added.len() > 1 {
                format!(
                    " in {} windows (maxBufferLength {:.1} GiB)",
                    added.len(),
                    max_buf as f64 / (1u64 << 30) as f64
                )
            } else {
                String::new()
            }
        );
        // Pin the new windows in a residency set attached to the queue.
        // Without one, the driver re-evaluates residency for the 83 GiB of
        // bytesNoCopy windows around submissions - llama.cpp pins its
        // buffers the same way (its init log: "use residency sets = true").
        self.ensure_residency(&added);
        self.chunks.extend(added);
        true
    }

    /// Add buffers to the queue's residency set, creating it on first use.
    /// Best effort: on failure the buffers still work, just unpinned.
    fn ensure_residency(&mut self, bufs: &[WeightChunk]) {
        use objc::runtime::Object;
        use objc::{class, msg_send, sel, sel_impl};
        unsafe {
            if self.residency.is_none() {
                let desc: *mut Object = msg_send![class!(MTLResidencySetDescriptor), new];
                if desc.is_null() {
                    return;
                }
                let mut err: *mut Object = std::ptr::null_mut();
                let set: *mut Object = msg_send![
                    self.device.as_ptr() as *mut Object,
                    newResidencySetWithDescriptor: desc
                    error: &mut err
                ];
                let _: () = msg_send![desc, release];
                if set.is_null() {
                    return;
                }
                let queue = self.queue.as_ptr() as *mut Object;
                let _: () = msg_send![queue, addResidencySet: set];
                self.residency = Some(set as usize);
            }
            if let Some(set) = self.residency {
                let set = set as *mut Object;
                for c in bufs {
                    let _: () = msg_send![set, addAllocation: c.buf.as_ptr() as *mut Object];
                }
                let _: () = msg_send![set, commit];
                let _: () = msg_send![set, requestResidency];
            }
        }
    }

    /// The pipeline for one (kernel, tile, lanes-per-row) shape, specialised
    /// and cached on first use. Returns None if the shape will not compile,
    /// which sends the whole batch to the CPU rather than half of it.
    fn pipeline(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
    ) -> Option<&ComputePipelineState> {
        self.pipeline_full(name, tile, lpr, false, false)
    }

    fn pipeline_ex(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
        indexed: bool,
    ) -> Option<&ComputePipelineState> {
        self.pipeline_full(name, tile, lpr, indexed, false)
    }

    /// The full form: `indexed` compiles the expert-indexed variant,
    /// `swiglu` the down-projection variant that applies silu(gate)*up on
    /// its activation loads.
    fn pipeline_full(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
        indexed: bool,
        swiglu: bool,
    ) -> Option<&ComputePipelineState> {
        self.pipeline_wait(name, tile, lpr, indexed, swiglu, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline_wait(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
        indexed: bool,
        swiglu: bool,
        wait: bool,
    ) -> Option<&ComputePipelineState> {
        // The llama-structure kernel is single-column; multi-row tiles fall
        // back to the block-per-lane kernel, which is correct at any LPR.
        let name = if tile != 1 {
            match name {
                "matvec_q2_k_mv" => "matvec_q2_k",
                "matvec_q3_k_mv" => "matvec_q3_k",
                "matvec_q4_k_mv" => "matvec_q4_k",
                "matvec_q6_k_mv" => "matvec_q6_k",
                other => other,
            }
        } else {
            name
        };
        let key = (
            name,
            tile,
            lpr + if indexed { 1000 } else { 0 }
                + if swiglu { 2000 } else { 0 }
                + if wait { 4000 } else { 0 },
        );
        if !self.pipelines.contains_key(&key) {
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [(0u64, tile as u32), (1, lpr as u32)] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            let ind = indexed;
            consts.set_constant_value_at_index(
                &ind as *const bool as *const _,
                metal::MTLDataType::Bool,
                2,
            );
            let sw = swiglu;
            consts.set_constant_value_at_index(
                &sw as *const bool as *const _,
                metal::MTLDataType::Bool,
                3,
            );
            let wt = wait;
            consts.set_constant_value_at_index(
                &wt as *const bool as *const _,
                metal::MTLDataType::Bool,
                4,
            );
            // NB: an UNSET bool function constant reads as TRUE on this
            // Metal (measured, macOS 26 / M4 Max) - DUAL_GW (5) and ROWS (6)
            // must be pinned to false explicitly or the non-dual/non-rows
            // specialisations silently take the variant path.
            for index in [5u64, 6] {
                let f = false;
                consts.set_constant_value_at_index(
                    &f as *const bool as *const _,
                    metal::MTLDataType::Bool,
                    index,
                );
            }
            let f = self
                .library
                .get_function(name, Some(consts))
                .map_err(|e| eprintln!("metal: {name}<tile {tile}, lanes {lpr}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// The dual gate/up variant of an INDEXED matvec: function constant 5
    /// (DUAL_GW) on top of the plain indexed form.
    fn pipeline_dual(
        &mut self,
        name: &'static str,
        tile: usize,
        lpr: usize,
    ) -> Option<&ComputePipelineState> {
        let key = (name, tile, lpr + 8000);
        if !self.pipelines.contains_key(&key) {
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [(0u64, tile as u32), (1, lpr as u32)] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            for (index, value) in [(2u64, true), (3, false), (4, false), (5, true), (6, false)] {
                consts.set_constant_value_at_index(
                    &value as *const bool as *const _,
                    metal::MTLDataType::Bool,
                    index,
                );
            }
            let f = self
                .library
                .get_function(name, Some(consts))
                .map_err(|e| eprintln!("metal: {name}<dual, lanes {lpr}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// The ROWS variant of an INDEXED matvec (MTP verify): function constant
    /// 6, indexed always on, tile always 1. Only the kernels whose source
    /// carries the ROWS mapping may be requested - the host checks against
    /// ROWS_KERNELS.
    fn pipeline_rows(
        &mut self,
        name: &'static str,
        lpr: usize,
        swiglu: bool,
    ) -> Option<&ComputePipelineState> {
        let key = (name, 1, lpr + 16000 + if swiglu { 2000 } else { 0 });
        if !self.pipelines.contains_key(&key) {
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [(0u64, 1u32), (1, lpr as u32)] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            for (index, value) in [(2u64, true), (3, swiglu), (4, false), (5, false), (6, true)] {
                consts.set_constant_value_at_index(
                    &value as *const bool as *const _,
                    metal::MTLDataType::Bool,
                    index,
                );
            }
            let f = self
                .library
                .get_function(name, Some(consts))
                .map_err(|e| eprintln!("metal: {name}<rows, lanes {lpr}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// cached under the kernel name with the formats folded into the key.
    fn mega_pipeline(
        &mut self,
        gate_fmt: u32,
        down_fmt: u32,
        sh_gate_fmt: u32,
        sh_down_fmt: u32,
    ) -> Option<&ComputePipelineState> {
        // Four format nibbles packed into the two key slots the pipelines
        // map allows.
        let key = (
            "moe_ffn_mega",
            (gate_fmt | (sh_gate_fmt << 4)) as usize,
            (down_fmt | (sh_down_fmt << 4)) as usize,
        );
        if !self.pipelines.contains_key(&key) {
            let consts = metal::FunctionConstantValues::new();
            for (index, value) in [
                (10u64, gate_fmt),
                (11, down_fmt),
                (12, sh_gate_fmt),
                (13, sh_down_fmt),
            ] {
                consts.set_constant_value_at_index(
                    &value as *const u32 as *const _,
                    metal::MTLDataType::UInt,
                    index,
                );
            }
            let f = self
                .library
                .get_function("moe_ffn_mega", Some(consts))
                .map_err(|e| eprintln!("metal: moe_ffn_mega<{gate_fmt},{down_fmt},{sh_gate_fmt},{sh_down_fmt}> failed: {e}"))
                .ok()?;
            let p = self
                .device
                .new_compute_pipeline_state_with_function(&f)
                .ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// Index of the chunk window wholly containing the absolute range
    /// `[addr, addr + len)`, across every attached mapping.
    fn chunk_for(&self, addr: usize, len: usize) -> Option<usize> {
        self.chunks
            .iter()
            .position(|c| addr >= c.start && addr + len <= c.start + c.len)
    }

    fn ensure_prefill(&mut self, bytes: usize) {
        if bytes > self.pf_x_cap {
            let cap = bytes.next_power_of_two();
            self.pf_x = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pf_x_cap = cap;
        }
        if bytes > self.pf_hs_cap {
            let cap = bytes.next_power_of_two();
            self.pf_hs = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pf_hs_cap = cap;
        }
    }

    fn ensure_arenas(&mut self, x_need: usize, y_need: usize) {
        if x_need > self.x_cap {
            let cap = x_need.next_power_of_two();
            self.x_arena = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.x_cap = cap;
        }
        if y_need > self.y_cap {
            let cap = y_need.next_power_of_two();
            self.y_arena = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.y_cap = cap;
        }
    }

    fn ensure_route(&mut self, words: usize) {
        let bytes = words * 4;
        if bytes > self.route_cap {
            let cap = bytes.next_power_of_two();
            self.route_buf = self
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.route_cap = cap;
            self.route_staged = None;
        }
    }

    /// Deferred commit for the fused prefill: signal the chain event and
    /// reap the oldest pending buffer past the run-ahead window, collecting
    /// its GPU clock stats.
    fn commit_chained(&mut self, cmd: &metal::CommandBufferRef) {
        self.pf_ev_val += 1;
        cmd.encode_signal_event(&self.pf_ev, self.pf_ev_val);
        cmd.commit();
        self.pf_pending.push_back(cmd.to_owned());
        while self.pf_pending.len() > 4 {
            let old = self.pf_pending.pop_front().expect("len checked");
            old.wait_until_completed();
            note_gpu_times(&old);
        }
    }

    /// Seal the open one-buffer prefill segment: commit it chained, so a
    /// layer that cannot join the shared buffer (or a split-timing probe)
    /// runs after it in event order, and the next eligible layer lazily
    /// re-arms a fresh segment.
    fn pf_obuf_seal(&mut self) {
        if let Some(cmd) = self.pf_obuf_cmd.take() {
            self.commit_chained(&cmd);
        }
    }

    /// Lazily arm the chunk's shared one-buffer command buffer, chained
    /// behind the last committed buffer (the wait is already satisfied
    /// right after `prefill_begin`'s drain). Returns without arming when
    /// a segment is already open.
    fn pf_obuf_arm(&mut self) {
        if self.pf_obuf_cmd.is_none() {
            // Retained: the per-stage encoders are autoreleased per call
            // site, the segment's command buffer must outlive them all.
            let cmd = self.queue.new_command_buffer().to_owned();
            cmd.encode_wait_for_event(&self.pf_ev, self.pf_ev_val);
            self.pf_obuf_cmd = Some(cmd);
        }
    }

    /// Wait out every deferred prefill buffer: prefill_end, a mid-chunk
    /// fallback, or any CPU access to arenas the GPU may still be writing.
    fn prefill_drain(&mut self) {
        while let Some(old) = self.pf_pending.pop_front() {
            old.wait_until_completed();
            note_gpu_times(&old);
        }
    }

    /// ALLPAKA_GPU_COUNTERS=1: (re)arm the per-dispatch timestamp sampling
    /// for a fused prefill chunk. Lazily creates the sample buffer on the
    /// only counter set this device has.
    fn cstamps_begin(&mut self) {
        if !gpu_counters() {
            return;
        }
        if self.cstamps.is_none() {
            if !gpu_counters_supported(&self.device) {
                eprintln!(
                    "gpu counters: atDispatchBoundary sampling unsupported on this device, off"
                );
                return;
            }
            let Some(set) = self
                .device
                .counter_sets()
                .into_iter()
                .find(|s| s.name() == "timestamp")
            else {
                eprintln!("gpu counters: no timestamp counter set");
                return;
            };
            let desc = metal::CounterSampleBufferDescriptor::new();
            desc.set_counter_set(&set);
            desc.set_sample_count(CSTAMP_CAP as u64);
            desc.set_storage_mode(metal::MTLStorageMode::Shared);
            match self.device.new_counter_sample_buffer_with_descriptor(&desc) {
                Ok(buf) => {
                    self.cstamps_dst = Some(self.device.new_buffer(
                        (CSTAMP_CAP * 8) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ));
                    self.cstamps = Some(buf);
                }
                Err(e) => eprintln!("gpu counters: sample buffer failed: {e}"),
            }
        }
        self.cstamp_idx.set(0);
        self.cstamp_labels.borrow_mut().clear();
    }

    /// Resolve the chunk's stamps and print the per-label timing summary.
    /// The interval before stamp i is attributed to the dispatch stamp i-1
    /// names; ns/tick is calibrated by the chunk buffer's GPU start/end when
    /// available (else a 1 GHz assumption).
    fn cstamps_report(&mut self, cal: Option<(f64, f64)>) {
        let Some(buf) = &self.cstamps else {
            return;
        };
        let n = self.cstamp_idx.get().min(CSTAMP_CAP);
        if n < 2 {
            return;
        }
        let dst = self.cstamps_dst.as_ref().expect("dst with buf").clone();
        let cmd = self.queue.new_command_buffer().to_owned();
        let be = cmd.new_blit_command_encoder();
        be.resolve_counters(buf, metal::NSRange::new(0, n as u64), &dst, 0);
        be.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ts = unsafe { std::slice::from_raw_parts(dst.contents() as *const u64, n).to_vec() };
        let labels = self.cstamp_labels.borrow();
        let scale = match cal {
            Some((gs, ge)) if ts[n - 1] > ts[0] => (ge - gs) * 1e9 / (ts[n - 1] - ts[0]) as f64,
            _ => 1.0,
        };
        let mut total: std::collections::HashMap<&str, (f64, u32, f64)> =
            std::collections::HashMap::new();
        for i in 1..n {
            let dt = (ts[i] - ts[i - 1]) as f64 * scale / 1000.0;
            let l = labels.get(i - 1).copied().unwrap_or("?");
            let e = total.entry(l).or_default();
            e.0 += dt;
            e.1 += 1;
            e.2 = e.2.max(dt);
        }
        let sum: f64 = total.values().map(|v| v.0).sum();
        eprintln!("gpu counters: {n} stamps, {sum:.2} ms attributed");
        let mut v: Vec<_> = total.into_iter().collect();
        v.sort_by(|a, b| (b.1).0.partial_cmp(&(a.1).0).unwrap());
        for (l, (s, c, mx)) in v {
            eprintln!(
                "  {l:<6} {s:8.3} ms  n={c:<3} mean {:.3} max {:.3}  ({:.1}%)",
                s / c as f64,
                mx,
                s / sum * 100.0
            );
        }
    }
}

/// One matmul request inside a batch: `x` holds `m` activation rows.
pub struct MatvecReq<'a> {
    pub ty: GgmlType,
    pub w: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
    pub x: &'a [f32],
    pub m: usize,
}

/// How many lanes cooperate on one output row, given how many quant blocks
/// that row holds. The lanes each take whole blocks, so more lanes than
/// blocks is pure waste; fewer means a lane walks several blocks, which is
/// fine. Rounded up to a power of two so the shuffle reduction stays inside
/// its lane group, and capped at the SIMD width.
fn lanes_per_row(ty: GgmlType, n_in: usize) -> usize {
    // The llama-structure q2_k kernel has a fixed geometry: 8 threads per
    // output row (one 32-lane SIMD group per 4 rows), independent of the
    // block count. The old kernel also runs correctly at LPR 8, so the tile
    // fallbacks in pipeline_ex need no geometry special-casing.
    if ty == GgmlType::Q2K && q2_mv() && std::env::var_os("ALLPAKA_Q2_ILV").is_none() {
        // 32 / NR0 threads per row; NR0 sweepable (occupancy vs activation
        // reuse). ffnbench picked 2 rows per SIMD group on M4 Max.
        static NR0: OnceLock<usize> = OnceLock::new();
        let nr0 = *NR0.get_or_init(|| {
            std::env::var("ALLPAKA_Q2_NR0")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n: &usize| [1, 2, 4].contains(n))
                .unwrap_or(2)
        });
        return 32 / nr0;
    }
    // The q3/q4 counterparts carry 2 rows per SIMD group: 16 threads per row.
    if ty == GgmlType::Q3K && q3_mv() {
        return 16;
    }
    if ty == GgmlType::Q4K && q4_mv() {
        return 16;
    }
    if ty == GgmlType::Q6K && q6_mv() {
        return 16;
    }
    let be = ty.block_elements().unwrap_or(32) as usize;
    let blocks = (n_in / be.max(1)).max(1);
    let mut base = blocks.next_power_of_two().min(32);
    // The interleaved q2_k variant owns TWO blocks per lane at half the
    // lanes; the pipeline swap happens in the kernel-name maps below.
    static ILV: OnceLock<bool> = OnceLock::new();
    if ty == GgmlType::Q2K
        && *ILV.get_or_init(|| std::env::var_os("ALLPAKA_Q2_ILV").is_some())
        && base >= 2
    {
        base /= 2;
    }
    // Sweep knob: fewer lanes per row means more rows per SIMD group and
    // more quant blocks in flight per lane - the occupancy-vs-ILP tradeoff
    // the ffn microbench measures. Power of two, so the shuffle reduction
    // stays intact.
    static DIV: OnceLock<usize> = OnceLock::new();
    let div = *DIV.get_or_init(|| {
        std::env::var("ALLPAKA_LPR_DIV")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|d: &usize| d.is_power_of_two())
            .unwrap_or(1)
    });
    (base / div).max(1)
}

/// Batch size from which the tile-matmul kernels beat the tiled matvecs.
/// Below this the [32 x 32] output tile is mostly padding and the matvec
/// path's exact tiles waste nothing.
const MM_MIN_M: usize = 16;

/// The tile-matmul kernel for a format, and its output-tile height.
///
/// Two compiled heights exist, 32 and 64 rows; 64 reads each staged
/// activation tile against twice the weight rows. `ALLPAKA_MM64` switches at
/// startup so the two can be swept as whole-bench A/B runs.
/// `(kernel name, BM, BN)`. Three compiled tile shapes: 32x32, 64x32
/// (`ALLPAKA_MM64`, measured worse) and 32x64 (`ALLPAKA_MM_BN64`), the
/// wide-batch tile that halves how often the weights are re-read per pass.
fn mm_kernel_for(ty: GgmlType, m: usize) -> Option<(&'static str, usize, usize)> {
    #[derive(Clone, Copy, PartialEq)]
    enum Shape {
        Base,
        Tall,
        Wide,
    }
    static SHAPE: OnceLock<Option<Shape>> = OnceLock::new();
    let forced = *SHAPE.get_or_init(|| {
        if std::env::var_os("ALLPAKA_MM64").is_some() {
            Some(Shape::Tall)
        } else if std::env::var_os("ALLPAKA_MM_BN64").is_some() {
            Some(Shape::Wide)
        } else if std::env::var_os("ALLPAKA_MM_BN32").is_some() {
            Some(Shape::Base)
        } else {
            None
        }
    });
    // 32x32 unconditionally: both alternatives MEASURED WORSE on this GPU,
    // even where their theory was strongest. Tall (64x32) lost 9% outright;
    // wide (64 columns) lost 6% forced everywhere AND ~3% when selected
    // per-m for the dense m=256 projections only - with concurrent encoders
    // the device is occupancy-bound, and fewer threadgroups hurt more than
    // halved weight re-reads help. The env pins stay for re-sweeping on
    // other hardware.
    let _ = m;
    // The llama.cpp-structure mm port (simdgroup MMA over half-staged
    // operands, 64x32 tile): `ALLPAKA_MM_LL=1` routes every quant format to
    // it; F32 (router only) keeps the old kernel.
    if mm_ll() {
        let k64 = mm_k64();
        let name = match (ty, k64) {
            (GgmlType::Q8_0, false) => {
                // The pipelined K-loop measured +9-11% over mmll_q8_0 on the
                // 35B forms (wqkv/zgate/ssm_out); ALLPAKA_MM_PIPE=0 reverts.
                if mm_pipe() {
                    "mmllp_q8_0"
                } else {
                    "mmll_q8_0"
                }
            }
            (GgmlType::Q2K, false) => "mmll_q2_k",
            (GgmlType::Q3K, false) => "mmll_q3_k",
            (GgmlType::Q4K, false) => {
                // Software-pipelined double-buffer K-loop: mmbench on the
                // 30B forms measured 12.4 vs 11.4 TFLOPS loop and 8.9 vs
                // 8.0 one-shot against mmll_q4_k. `ALLPAKA_MM_PIPE=0`
                // reverts.
                if mm_pipe() {
                    "mmllp_q4_k"
                } else {
                    "mmll_q4_k"
                }
            }
            (GgmlType::Q5K, false) => "mmll_q5_k",
            (GgmlType::Q6K, false) => "mmll_q6_k",
            (GgmlType::Q8_0, true) => "mm64ll_q8_0",
            (GgmlType::Q2K, true) => "mm64ll_q2_k",
            (GgmlType::Q3K, true) => "mm64ll_q3_k",
            (GgmlType::Q4K, true) => "mm64ll_q4_k",
            (GgmlType::Q5K, true) => "mm64ll_q5_k",
            (GgmlType::Q6K, true) => "mm64ll_q6_k",
            // The router: f32 weights staged to half like everything else.
            // Top-8 expert selection tolerates the rounding; the f32_router
            // parity test and verify are the arbiters. Own kill switch for
            // clean A/B: ALLPAKA_MM_LL_F32=0.
            (GgmlType::F32, _) => {
                static F32LL: OnceLock<bool> = OnceLock::new();
                if *F32LL.get_or_init(|| std::env::var("ALLPAKA_MM_LL_F32").is_ok_and(|v| v == "1"))
                {
                    "mmll_f32"
                } else {
                    return Some(("matmul_f32", 32, 32));
                }
            }
            _ => return None,
        };
        return Some((name, 64, 32));
    }
    let shape = forced.unwrap_or(Shape::Base);
    let name = match (ty, shape) {
        (GgmlType::Q8_0, Shape::Base) => "matmul_q8_0",
        (GgmlType::Q2K, Shape::Base) => "matmul_q2_k",
        (GgmlType::Q3K, Shape::Base) => "matmul_q3_k",
        (GgmlType::Q4K, Shape::Base) => "matmul_q4_k",
        (GgmlType::Q5K, Shape::Base) => "matmul_q5_k",
        (GgmlType::Q6K, Shape::Base) => "matmul_q6_k",
        (GgmlType::F32, Shape::Base) => "matmul_f32",
        (GgmlType::Q8_0, Shape::Tall) => "matmul64_q8_0",
        (GgmlType::Q2K, Shape::Tall) => "matmul64_q2_k",
        (GgmlType::Q3K, Shape::Tall) => "matmul64_q3_k",
        (GgmlType::Q4K, Shape::Tall) => "matmul64_q4_k",
        (GgmlType::Q5K, Shape::Tall) => "matmul64_q5_k",
        (GgmlType::Q6K, Shape::Tall) => "matmul64_q6_k",
        (GgmlType::F32, Shape::Tall) => "matmul64_f32",
        (GgmlType::Q8_0, Shape::Wide) => "matmulw_q8_0",
        (GgmlType::Q2K, Shape::Wide) => "matmulw_q2_k",
        (GgmlType::Q3K, Shape::Wide) => "matmulw_q3_k",
        (GgmlType::Q4K, Shape::Wide) => "matmulw_q4_k",
        (GgmlType::Q5K, Shape::Wide) => "matmulw_q5_k",
        (GgmlType::Q6K, Shape::Wide) => "matmulw_q6_k",
        (GgmlType::F32, Shape::Wide) => "matmulw_f32",
        _ => return None,
    };
    let (bm, bn) = match shape {
        Shape::Base => (32, 32),
        Shape::Tall => (64, 32),
        Shape::Wide => (32, 64),
    };
    Some((name, bm, bn))
}

/// The compiled TILE specialisations, largest last. A batch is split into
/// tiles from this set, so every dispatch covers exactly the rows it claims.
const TILES: &[usize] = &[1, 2, 4, 8];

/// Split `m` rows into `(start, tile)` dispatches, taking the largest
/// available tile first. Exact by construction: the tiles are powers of two
/// down to 1, so the remainder is always coverable and no dispatch ever
/// touches a row past `m`.
fn tiles_for(m: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < m {
        let left = m - start;
        let tile = TILES
            .iter()
            .rev()
            .copied()
            .find(|&t| t <= left)
            .expect("TILES contains 1");
        out.push((start, tile));
        start += tile;
    }
    out
}

/// Run one matvec on the GPU. Returns None to mean "do it on the CPU".
pub fn matvec(ty: GgmlType, w: &[u8], n_in: usize, n_out: usize, x: &[f32]) -> Option<Vec<f32>> {
    matvec_batch(&[MatvecReq {
        ty,
        w,
        n_in,
        n_out,
        x,
        m: 1,
    }])
    .map(|mut v| v.pop().unwrap())
}

/// Run several independent matvecs as ONE command buffer with ONE wait.
///
/// This is where the GPU path earns its keep. Decode issues over a thousand
/// small matvecs per token, and a synchronous wait per dispatch costs more
/// than the arithmetic; batching the independent ones (q/k/v together, all
/// experts' gate+up together) collapses those waits to a handful per layer.
///
/// All-or-nothing: if any request has no kernel or foreign bytes, the whole
/// batch declines and the caller runs it on the CPU.
pub fn matvec_batch(reqs: &[MatvecReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    // Validate everything up front: all-or-nothing keeps CPU/GPU results
    // from interleaving within one logical operation.
    let mut kernels = Vec::with_capacity(reqs.len());
    for r in reqs {
        // A format needs a matvec kernel below the mm threshold, an mm
        // kernel above it. F32 has only the latter: it exists for the MoE
        // router, whose prefill batches are large and whose decode matvec is
        // a rounding error on the CPU.
        let kernel = match r.ty {
            GgmlType::Q5_0 => Some("matvec_q5_0"),
            GgmlType::Q8_0 => Some("matvec_q8_0"),
            GgmlType::Q2K => Some(q2_kernel()),
            GgmlType::Q3K => Some(q3_kernel()),
            GgmlType::Q4K => Some(q4_kernel()),
            GgmlType::Q5K => Some("matvec_q5_k"),
            GgmlType::Q6K => Some(q6_kernel()),
            _ => None,
        };
        let routable_mm = r.m >= MM_MIN_M && mm_kernel_for(r.ty, r.m).is_some();
        if kernel.is_none() && !routable_mm {
            return None;
        }
        if r.x.len() != r.m * r.n_in {
            return None;
        }
        let addr = r.w.as_ptr() as usize;
        // Not inside any attached mapping (e.g. a test matrix on the heap)
        // sends the whole batch to the CPU.
        let chunk = gpu.chunk_for(addr, r.w.len())?;
        kernels.push((kernel, chunk, (addr - gpu.chunks[chunk].start) as u64));
    }

    // Bump-plan every dispatch's arena slices up front, then grow the arenas
    // once. Offsets are 256-aligned, which satisfies setBuffer:offset:.
    struct Slot {
        ri: usize,
        start_row: usize,
        rows: usize,
        x_off: usize,
        y_off: usize,
        /// Route this slot to the tile-matmul kernel instead of the matvec.
        mm: bool,
    }
    let align = |n: usize| (n + 255) & !255;
    let mut x_need = 0usize;
    let mut y_need = 0usize;
    let mut slots = Vec::new();
    for (ri, r) in reqs.iter().enumerate() {
        // Big batches go whole to the tile-matmul kernel; small ones become
        // matvec dispatches, each running the tile specialisation that
        // matches its row count exactly.
        if r.m >= MM_MIN_M && mm_kernel_for(r.ty, r.m).is_some() {
            slots.push(Slot {
                ri,
                start_row: 0,
                rows: r.m,
                x_off: x_need,
                y_off: y_need,
                mm: true,
            });
            x_need += align(r.m * r.n_in * 4);
            y_need += align(r.m * r.n_out * 4);
            continue;
        }
        for (start_row, rows) in tiles_for(r.m) {
            slots.push(Slot {
                ri,
                start_row,
                rows,
                x_off: x_need,
                y_off: y_need,
                mm: false,
            });
            x_need += align(rows * r.n_in * 4);
            y_need += align(rows * r.n_out * 4);
        }
    }
    gpu.ensure_arenas(x_need, y_need);

    // Specialise every dispatch's pipeline before encoding: the lookup may
    // compile a new variant and needs &mut, while the encoder below holds the
    // rest of `gpu` immutably.
    let mut states = Vec::with_capacity(slots.len());
    for s in &slots {
        if s.mm {
            states.push(
                gpu.pipeline(mm_kernel_for(reqs[s.ri].ty, s.rows)?.0, 1, 1)?
                    .to_owned(),
            );
        } else {
            let lpr = lanes_per_row(reqs[s.ri].ty, reqs[s.ri].n_in);
            states.push(gpu.pipeline(kernels[s.ri].0?, s.rows, lpr)?.to_owned());
        }
    }

    // Stage the activations into the shared arena: the only per-call copy.
    unsafe {
        let xp = gpu.x_arena.contents() as *mut u8;
        for s in &slots {
            let r = &reqs[s.ri];
            let xs = &r.x[s.start_row * r.n_in..(s.start_row + s.rows) * r.n_in];
            std::ptr::copy_nonoverlapping(xs.as_ptr() as *const u8, xp.add(s.x_off), xs.len() * 4);
        }
    }

    // The autorelease pool matters: command buffers and encoders are
    // autoreleased ObjC objects, and without a pool on this thread they pile
    // up until some outer drain - slow and leaky at 200 calls per token.
    let out = objc::rc::autoreleasepool(|| {
        // Encode + wait; see the identical structure in ffn_batch.
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Concurrent: the slots read shared weights and disjoint x slices
        // and write disjoint y slices, so they can fill the GPU together. A
        // single expert-sized matvec is ~25k threads - a fraction of the
        // device - and the serial encoder was running them ONE AT A TIME:
        // that, not arithmetic, was the measured ceiling of the decode ffn.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        for (si, s) in slots.iter().enumerate() {
            let r = &reqs[s.ri];
            let (_, chunk, w_off) = &kernels[s.ri];
            enc.set_compute_pipeline_state(&states[si]);
            enc.set_buffer(0, Some(&gpu.chunks[*chunk].buf), 0);
            enc.set_buffer(1, Some(&gpu.x_arena), s.x_off as u64);
            enc.set_buffer(2, Some(&gpu.y_arena), s.y_off as u64);
            let n_in32 = r.n_in as u32;
            let n_out32 = r.n_out as u32;
            enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
            enc.set_bytes(5, 8, w_off as *const u64 as *const _);

            if s.mm {
                // One threadgroup per [BM x BN] output tile.
                let (bm, bn) = mm_kernel_for(r.ty, s.rows).map_or((32, 32), |(_, bm, bn)| (bm, bn));
                let m32 = s.rows as u32;
                enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        (r.n_out as u64).div_ceil(bm as u64),
                        (s.rows as u64).div_ceil(bn as u64),
                        1,
                    ),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                // One lane group per output row, threadgroups of four SIMD
                // groups.
                let per_group = 128u64;
                let total_threads = r.n_out as u64 * lanes_per_row(r.ty, r.n_in) as u64;
                enc.dispatch_thread_groups(
                    MTLSize::new(total_threads.div_ceil(per_group), 1, 1),
                    MTLSize::new(per_group, 1, 1),
                );
            }
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(slots.len() as u64, Ordering::Relaxed);

        let mut out: Vec<Vec<f32>> = reqs
            .iter()
            .map(|r| Vec::with_capacity(r.m * r.n_out))
            .collect();
        let yp = gpu.y_arena.contents() as *const u8;
        for s in &slots {
            let n = s.rows * reqs[s.ri].n_out;
            let at = out[s.ri].len();
            out[s.ri].resize(at + n, 0.0);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    yp.add(s.y_off),
                    out[s.ri][at..].as_mut_ptr() as *mut u8,
                    n * 4,
                );
            }
        }
        out
    });
    Some(out)
}

/// A CPU allocation the GPU reads in place.
///
/// The KV cache is written by the CPU every step and read by the attention
/// kernel; on unified memory neither needs a copy, so the cache is allocated
/// page-aligned and wrapped here exactly like the weight mapping is. Holding
/// the `Buffer` keeps the wrapper alive: dropping this releases it, and the
/// caller still owns the memory.
pub struct SharedRegion {
    buf: Buffer,
    len: usize,
}

// The buffer is an ObjC refcounted handle; the region it wraps is owned by
// the caller, who is responsible for not writing it while a kernel reads it
// (the forward pass stores the step, then attends, in that order).
unsafe impl Send for SharedRegion {}
unsafe impl Sync for SharedRegion {}

/// Wrap a page-aligned allocation for the GPU. `region` must start on a page
/// boundary and span whole pages; anything else is declined rather than
/// silently mapped short.
pub fn wrap_region(region: &[u8]) -> Option<SharedRegion> {
    const PAGE: usize = 16384;
    if region.is_empty() || region.as_ptr() as usize % PAGE != 0 || region.len() % PAGE != 0 {
        return None;
    }
    let gpu = GPU.get()?.as_ref()?.lock().ok()?;
    let buf = gpu.device.new_buffer_with_bytes_no_copy(
        region.as_ptr() as *const _,
        region.len() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    if buf.as_ptr().is_null() {
        return None;
    }
    Some(SharedRegion {
        buf,
        len: region.len(),
    })
}

/// One decode step's attention over the cache, for every query head.
///
/// `k_off` and `v_off` are *element* offsets of this layer's K and V inside
/// the wrapped region of halves; from there the cache is `[pos][kv_dim]`.
pub struct AttnReq<'a> {
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub q: &'a [f32],
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub group: usize,
    pub n_pos: usize,
    pub scale: f32,
}

/// The kernel assumes a lane owns four dimensions of a 128-wide head.
const ATTN_HEAD_DIM: usize = 128;

/// Attention for one token followed by the output projection, as ONE command
/// buffer with ONE wait.
///
/// The two are adjacent and dependent - `wo` consumes exactly what attention
/// produces - so fusing them costs nothing and saves a round trip. Encoded
/// separately, attention's own submission ate most of what the kernel won:
/// 48 extra command buffers per token on a 48-layer model, at the ~83 us
/// each one costs, against a phase that took 7 ms in total.
///
/// The intermediate never leaves the GPU arena, so the attention output is
/// not copied to the CPU at all.
pub fn attend_project(req: &AttnReq, wo_ty: GgmlType, wo: &[u8], n_out: usize) -> Option<Vec<f32>> {
    if !attn_shape_ok(req) {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    let mat = resolve(&gpu, wo_ty, wo)?;
    let n_in = req.n_q_heads * req.head_dim;
    if n_in % 4 != 0 || n_out % 4 != 0 {
        return None;
    }
    let lpr = lanes_per_row(wo_ty, n_in);
    let state = gpu.pipeline(mat.kernel, 1, lpr)?.to_owned();

    let align = |n: usize| (n + 255) & !255;
    let attn_bytes = align(n_in * 4);
    gpu.ensure_arenas(req.q.len() * 4, attn_bytes + align(n_out * 4));
    unsafe {
        std::ptr::copy_nonoverlapping(
            req.q.as_ptr() as *const u8,
            gpu.x_arena.contents() as *mut u8,
            req.q.len() * 4,
        );
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_attend(enc, &gpu, req, 0);
        // The compute encoder is serial, so the projection sees attention's
        // writes without any barrier of ours.
        enc.set_compute_pipeline_state(&state);
        enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
        enc.set_buffer(1, Some(&gpu.y_arena), 0);
        enc.set_buffer(2, Some(&gpu.y_arena), attn_bytes as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
        let per_group = 128u64;
        let total_threads = n_out as u64 * lpr as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(total_threads.div_ceil(per_group), 1, 1),
            MTLSize::new(per_group, 1, 1),
        );
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(2, Ordering::Relaxed);

        let mut out = vec![0f32; n_out];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const u8).add(attn_bytes),
                out.as_mut_ptr() as *mut u8,
                n_out * 4,
            );
        }
        out
    }))
}

/// Shapes the attention kernel is written for.
fn attn_shape_ok(req: &AttnReq) -> bool {
    req.head_dim == ATTN_HEAD_DIM
        && req.n_pos > 0
        && req.group > 0
        && req.q.len() == req.n_q_heads * req.head_dim
        && (req.k_off + req.n_pos * req.kv_dim) * 2 <= req.cache.len
        && (req.v_off + req.n_pos * req.kv_dim) * 2 <= req.cache.len
}

/// Encode attention into an open encoder, q already staged in the x arena and
/// the result landing at `y_off` in the y arena.
fn encode_attend(enc: &metal::ComputeCommandEncoderRef, gpu: &Gpu, req: &AttnReq, y_off: usize) {
    encode_attend_at(enc, gpu, req, 0, y_off)
}

fn encode_attend_at(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    req: &AttnReq,
    x_off: usize,
    y_off: usize,
) {
    encode_attend_from(enc, gpu, req, &gpu.x_arena, x_off, y_off)
}

/// The general form: q can live in either arena - the fused attention block
/// reads it from the y arena, where the qkv matmul just wrote it.
fn encode_attend_from(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    req: &AttnReq,
    q_buf: &Buffer,
    q_off: usize,
    y_off: usize,
) {
    enc.set_compute_pipeline_state(&gpu.pipelines[&(attend_kernel(), 1, 1)]);
    enc.set_buffer(0, Some(&req.cache.buf), 0);
    enc.set_buffer(1, Some(&req.cache.buf), 0);
    enc.set_buffer(2, Some(q_buf), q_off as u64);
    enc.set_buffer(3, Some(&gpu.y_arena), y_off as u64);
    for (index, value) in [
        (4u64, req.kv_dim as u32),
        (5, req.head_dim as u32),
        (6, req.n_pos as u32),
        (7, req.group as u32),
    ] {
        enc.set_bytes(index, 4, &value as *const u32 as *const _);
    }
    enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
    for (index, value) in [(9u64, req.k_off as u64), (10, req.v_off as u64)] {
        enc.set_bytes(index, 8, &value as *const u64 as *const _);
    }
    enc.dispatch_thread_groups(
        MTLSize::new(req.n_q_heads as u64, 1, 1),
        MTLSize::new(attend_tg(), 1, 1),
    );
}

/// The gated-delta-net branch of a qwen35moe layer (linear attention):
/// replaces the whole attention half of the token (no KV cache, the
/// recurrence state lives in the shared SSM region instead).
pub struct TokenGdn<'a> {
    /// hidden -> conv channels (key_dim*2 + value_dim), Q8_0.
    pub wqkv: (GgmlType, &'a [u8], usize),
    /// hidden -> value_dim (the z gate), Q8_0.
    pub zgate: (GgmlType, &'a [u8], usize),
    /// F32 [hidden -> heads_v] projections, raw bytes in the mmap.
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    /// F32 conv weights [channels * d_conv], raw bytes in the mmap.
    pub conv1d: &'a [u8],
    /// Small F32 vectors handed to the kernel by value.
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    /// value_dim -> hidden, Q8_0.
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    /// Element offsets of this layer's conv window and deltanet state
    /// inside the SSM region.
    pub conv_off: usize,
    pub state_off: usize,
}

/// One decode layer's description for the whole-token encoder.
pub struct TokenLayer<'a> {
    /// Raw F32 norm weights, borrowed from the mmap (16 KB exceeds setBytes).
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// qwen35moe: the sigmoid output gate is fused into the q projection
    /// ([q | gate] per head); split and applied post-attention on the GPU.
    pub gate_in_q: bool,
    /// Raw F32 Q/K/V biases in the mmap (GLM-4); None for bias-free models.
    pub q_bias: Option<&'a [u8]>,
    pub k_bias: Option<&'a [u8]>,
    pub v_bias: Option<&'a [u8]>,
    /// Element offsets of this layer's K and V in the cache region.
    pub k_off: usize,
    pub v_off: usize,
    /// Present on linear-attention layers: the whole attention half is the
    /// gated-delta-net branch, and the attention fields above go unused.
    pub gdn: Option<TokenGdn<'a>>,
    pub ffn: TokenFfn<'a>,
}

pub enum TokenFfn<'a> {
    Dense {
        gate: (GgmlType, &'a [u8], usize),
        up: (GgmlType, &'a [u8], usize),
        down: (GgmlType, &'a [u8], usize),
    },
    Moe {
        /// F32 router `[n_expert, hidden]`.
        router: (GgmlType, &'a [u8], usize),
        /// Raw F32 router bias (GLM `exp_probs_b`); sigmoid gating only.
        router_bias: Option<&'a [u8]>,
        /// Stacked expert tensors; per-expert geometry alongside.
        gate: (GgmlType, &'a [u8]),
        up: (GgmlType, &'a [u8]),
        down: (GgmlType, &'a [u8]),
        expert_ffn: usize,
        n_used: usize,
        /// Sigmoid gating (GLM-4) instead of softmax; weights already
        /// renormalized over the top-k by the caller's contract.
        sigmoid: bool,
        /// Optional shared expert, same ffn width as a routed expert; runs
        /// as an extra combine slot with weight 1.
        shared: Option<[(GgmlType, &'a [u8], usize); 3]>,
        /// qwen35moe: the shared expert's combine weight is not a constant
        /// 1 but sigmoid(gate . h), an F32 [hidden] vector in the mmap. The
        /// raw logit lands in the slot's weight and the combine kernel
        /// applies the sigmoid.
        shared_gate: Option<&'a [u8]>,
    },
}

pub struct TokenReq<'a> {
    /// The embedded input rows, `m * hidden` floats (`m` = 1 for decode).
    pub x: &'a [f32],
    /// Tokens to run through the layers in one buffer. m > 1 is the
    /// speculative-verify path: batched projections (TILE = m), per-token
    /// attention and MoE, the GDN batch kernels.
    pub m: usize,
    pub layers: &'a [TokenLayer<'a>],
    pub cache: &'a SharedRegion,
    /// qwen35moe: the SSM region holding every GDN layer's conv window and
    /// deltanet state (f32 elements), wrapped like the KV cache.
    pub ssm: Option<&'a SharedRegion>,
    /// MTP verification: per-row rollback slots written by the GDN batch
    /// kernels (region, slot stride in elements); None for plain decode.
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// First token's position; token i lands at pos + i.
    pub pos: usize,
    pub scale: f32,
    /// `(sin, cos)` pairs per rotary pair per token: `m * rot_dim / 2`.
    pub rope: &'a [[f32; 2]],
    /// Rotary width; equals head_dim for full rope, head_dim/2 for GLM-4,
    /// head_dim/4 for qwen35moe's 64-of-256 partial rope.
    pub rot_dim: usize,
    pub eps: f32,
    pub output_norm: &'a [u8],
    pub output: (GgmlType, &'a [u8], usize),
    /// Greedy caller: run the argmax on the GPU after the output projection
    /// and read back 4 bytes instead of the whole vocabulary.
    pub argmax: bool,
}

/// What decode_token produced: the full vocabulary logits, or just the
/// greedy winner when the request asked for the on-GPU argmax. For a
/// multi-token request: the per-row argmax and the output-normed hidden
/// rows (the speculative verifier's arbitration inputs and the next
/// round's draft seed).
pub enum TokenOut {
    Logits(Vec<f32>),
    Argmax(u32),
    Rows { argmax: Vec<u32>, hidden: Vec<f32> },
}

/// A resolved norm bind: which chunk window holds the F32 weights.
struct NormRef {
    chunk: usize,
    off: u64,
}

/// F32 weights for the router: resolve() deliberately excludes F32 so the
/// general matmul paths never route tiny test tensors to a kernel that
/// assumes float4-divisible rows; the whole-token path opts in here, with
/// that assumption checked.
fn resolve_f32(gpu: &Gpu, w: &[u8], n_in: usize) -> Option<MatRef> {
    if n_in % 4 != 0 || w.as_ptr() as usize % 4 != 0 {
        return None;
    }
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some(MatRef {
        kernel: "matvec_f32",
        ty: GgmlType::F32,
        chunk,
        w_off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

fn resolve_norm(gpu: &Gpu, raw: &[u8], hidden: usize) -> Option<NormRef> {
    if raw.len() != hidden * 4 {
        return None;
    }
    let addr = raw.as_ptr() as usize;
    if addr % 4 != 0 {
        return None;
    }
    let chunk = gpu.chunk_for(addr, raw.len())?;
    Some(NormRef {
        chunk,
        off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

#[repr(C)]
#[repr(C)]
struct GpuMegaArgs {
    hidden: u32,
    ffn: u32,
    n_expert: u32,
    n_used: u32,
    x_at: u32,
    delta_at: u32,
    h_at: u32,
    logits_at: u32,
    ids_at: u32,
    wts_at: u32,
    gate_at: u32,
    up_at: u32,
    downo_at: u32,
    ctr_at: u32,
    n_tg: u32,
    ctr_base: u32,
    eps: f32,
    _pad: u32,
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    router_off: u64,
    gate_stride: u64,
    up_stride: u64,
    down_stride: u64,
    sigmoid: u32,
    has_shared: u32,
    sh_gate_off: u64,
    sh_up_off: u64,
    sh_down_off: u64,
}

/// How many threadgroups the megakernel launches. Every TG must be RESIDENT
/// simultaneously or the in-kernel sync deadlocks; 48 x 256 threads sits
/// comfortably inside a 40-core M4 Max. `ALLPAKA_MEGA_TG` to sweep.
fn mega_tg() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ALLPAKA_MEGA_TG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(48)
    })
}

fn mega_enabled() -> bool {
    static M: OnceLock<bool> = OnceLock::new();
    *M.get_or_init(|| std::env::var("ALLPAKA_MEGA").is_ok_and(|v| v == "1"))
}

/// Format code for the megakernel's expert cores; None = unsupported.
fn mega_fmt(ty: GgmlType) -> Option<u32> {
    match ty {
        GgmlType::Q2K => Some(0),
        GgmlType::Q3K => Some(1),
        GgmlType::Q4K => Some(2),
        GgmlType::Q5_0 => Some(3),
        GgmlType::Q5K => Some(4),
        GgmlType::Q8_0 => Some(5),
        _ => None,
    }
}

struct GpuIdxArgs {
    stride: u64,
    slots: u32,
    x_stride: u32,
    // ROWS (MTP verify) only: per-token-row strides of the ids, x and y
    // blocks, plus the row count. Zero otherwise.
    ids_stride: u32,
    x_row_stride: u32,
    y_row_stride: u32,
    n_rows: u32,
}

/// A whole decode token - every layer's attention AND FFN, ending with the
/// output projection - as ONE command buffer with ONE wait.
///
/// The CPU's remaining role per token is: stage the embedding, encode, wait,
/// argmax. Routing runs on the GPU (softmax_topk), and the expert matmuls
/// dispatch INDEXED - the grid covers `n_used` expert slots whose ids only
/// ever exist in GPU memory. By the GPU's own clock, the per-buffer driver
/// scheduling plus round-trip idle this replaces was ~24 ms of a 74 ms
/// 235B token.
/// A resolved GDN branch's weights for the whole-token encoders
/// (decode_token and the speculative multi-token verify).
struct GdnRefs {
    mats: [MatRef; 5], // wqkv, zgate, alpha, beta, ssm_out
    states: [ComputePipelineState; 5],
    conv1d: NormRef,
    heads_k: u32,
    heads_v: u32,
    d: u32,
    d_conv: u32,
    conv_off: u64,
    state_off: u64,
    channels: usize,
    value_dim: usize,
}

/// A resolved decode layer for the whole-token encoders.
struct LayerRefs {
    attn_norm: NormRef,
    ffn_norm: NormRef,
    /// None on GDN layers (no attention mats).
    mats: Option<[MatRef; 4]>,
    mat_states: Option<[ComputePipelineState; 4]>,
    q_bias: Option<NormRef>,
    k_bias: Option<NormRef>,
    v_bias: Option<NormRef>,
    gdn: Option<GdnRefs>,
    ffn: FfnRefs,
    normflag: bool,
}

enum FfnRefs {
    Dense {
        mats: [MatRef; 3],
        states: [ComputePipelineState; 3],
        ffn_dim: usize,
    },
    Moe {
        router: MatRef,
        router_state: ComputePipelineState,
        router_bias: Option<NormRef>,
        n_expert: usize,
        mats: [MatRef; 3],
        states: [ComputePipelineState; 3],
        strides: [u64; 3],
        expert_ffn: usize,
        n_used: usize,
        sw_fused: bool,
        mega: Option<ComputePipelineState>,
        sigmoid: bool,
        shared: Option<([MatRef; 3], [ComputePipelineState; 3])>,
        /// qwen35moe's shared-expert gate projection (F32 [hidden]);
        /// its raw logit becomes the shared slot's weight.
        shared_gate: Option<(MatRef, ComputePipelineState)>,
        /// gate + up as one dual-output indexed dispatch
        /// (ALLPAKA_DECODE_GUFUSE).
        gu_dual: Option<ComputePipelineState>,
    },
}

/// A whole-token GPU request was rejected before submission. The checked API
/// keeps this distinct from a successful GPU result so callers cannot silently
/// mistake a CPU fallback for GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeDecline {
    pub stage: &'static str,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodePathStats {
    pub attempts: u64,
    pub successes: u64,
    pub declines: u64,
}

static DECODE_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_SUCCESSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_DECLINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn decode_path_stats() -> DecodePathStats {
    use std::sync::atomic::Ordering::Relaxed;
    DecodePathStats {
        attempts: DECODE_ATTEMPTS.load(Relaxed),
        successes: DECODE_SUCCESSES.load(Relaxed),
        declines: DECODE_DECLINES.load(Relaxed),
    }
}

impl std::fmt::Display for DecodeDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stage={} reason={}", self.stage, self.reason)
    }
}

impl std::error::Error for DecodeDecline {}

/// Diagnostic-preserving entry point for production fast paths.
pub fn decode_token_checked(req: &TokenReq) -> Result<TokenOut, DecodeDecline> {
    use std::sync::atomic::Ordering::Relaxed;
    DECODE_ATTEMPTS.fetch_add(1, Relaxed);
    match decode_token(req) {
        Some(out) => {
            DECODE_SUCCESSES.fetch_add(1, Relaxed);
            Ok(out)
        }
        None => {
            DECODE_DECLINES.fetch_add(1, Relaxed);
            Err(DecodeDecline {
                stage: "backend-decode",
                reason: "request shape, tensor format, or Metal pipeline is unsupported",
            })
        }
    }
}

/// Backend-neutral checked decode contract. New callers should use this
/// rather than interpreting `None` from the compatibility API below.
pub fn decode_token_outcome(req: &TokenReq) -> crate::accel::AccelOutcome<TokenOut> {
    match decode_token_checked(req) {
        Ok(out) => crate::accel::AccelOutcome::Executed(out),
        Err(reason) => crate::accel::AccelOutcome::Declined(crate::accel::DeclineReason::Backend {
            operation: "decode-token",
            detail: reason.to_string(),
        }),
    }
}

pub fn decode_token(req: &TokenReq) -> Option<TokenOut> {
    let dbg = std::env::var_os("ALLPAKA_TOKENBUF_DEBUG").is_some();
    macro_rules! why {
        ($cond:expr, $msg:expr) => {
            if $cond {
                if dbg {
                    eprintln!("tokenbuf declined: {}", $msg);
                }
                return None;
            }
        };
    }
    let m = req.m;
    why!(m == 0 || m > 8 || req.x.len() % m != 0, "m");
    let hidden = req.x.len() / m;
    let hd = req.head_dim;
    // 128 everywhere; 256 on qwen35moe's full-attention layers (own kernels).
    why!(hd != 128 && hd != 256 || hidden % 4 != 0, "shape");
    // Full rotary, GLM-style half rotary, qwen35moe's quarter rotary at 256.
    why!(req.rope.len() != m * req.rot_dim / 2, "rope table");
    why!(
        req.rot_dim != hd && req.rot_dim * 2 != hd && !(hd == 256 && req.rot_dim == 64),
        "rot dim"
    );
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let span = (req.pos + m) * req.kv_dim;
    let vocab = req.output.2;
    // GDN layers need the shared SSM region; attention layers the KV cache.
    let has_gdn = req.layers.iter().any(|l| l.gdn.is_some());
    why!(has_gdn && req.ssm.is_none(), "no ssm region");

    let Some(Some(cell)) = GPU.get().map(|c| c.as_ref()) else {
        why!(true, "no gpu");
        return None;
    };
    let Ok(mut gpu) = cell.lock() else {
        why!(true, "lock");
        return None;
    };
    let resolve_or_decline =
        |label: &str, ty: GgmlType, w: &[u8], dbg: bool, gpu: &mut _| -> Option<MatRef> {
            let Some(mat) = resolve(gpu, ty, w) else {
                if dbg {
                    eprintln!("tokenbuf declined: {label} resolve");
                }
                return None;
            };
            Some(mat)
        };
    let resolve_f32_or_decline =
        |label: &str, b: &[u8], n: usize, dbg: bool, gpu: &mut _| -> Option<MatRef> {
            let Some(norm) = resolve_f32(gpu, b, n) else {
                if dbg {
                    eprintln!("tokenbuf declined: {label} norm resolve");
                }
                return None;
            };
            Some(norm)
        };

    let mut layers = Vec::with_capacity(req.layers.len());
    let mut max_ffn = 0usize;
    let mut max_slots = 1usize;
    let mut max_expert = 0usize;
    for l in req.layers {
        why!(
            (l.k_off + span) * 2 > req.cache.len || (l.v_off + span) * 2 > req.cache.len,
            "cache span"
        );
        // GDN layer: no attention mats at all; resolve the deltanet branch.
        let gdn = match &l.gdn {
            None => None,
            Some(g) => {
                let key_dim = g.heads_k * g.d;
                let value_dim = g.heads_v * g.d;
                let channels = key_dim * 2 + value_dim;
                why!(
                    g.wqkv.2 != channels || g.zgate.2 != value_dim || g.ssm_out.2 != hidden,
                    "gdn dims"
                );
                why!(
                    g.alpha.len() != hidden * g.heads_v * 4
                        || g.beta.len() != hidden * g.heads_v * 4,
                    "gdn ab dims"
                );
                why!(g.conv1d.len() != channels * g.d_conv * 4, "gdn conv dims");
                why!(
                    g.a.len() != g.heads_v || g.dt.len() != g.heads_v || g.ssm_norm.len() != g.d,
                    "gdn vec dims"
                );
                let mats = [
                    resolve_or_decline("gdn.wqkv", g.wqkv.0, g.wqkv.1, dbg, &mut gpu)?,
                    resolve_or_decline("gdn.zgate", g.zgate.0, g.zgate.1, dbg, &mut gpu)?,
                    resolve_f32_or_decline("gdn.alpha", g.alpha, hidden, dbg, &mut gpu)?,
                    resolve_f32_or_decline("gdn.beta", g.beta, hidden, dbg, &mut gpu)?,
                    resolve_or_decline("gdn.ssm_out", g.ssm_out.0, g.ssm_out.1, dbg, &mut gpu)?,
                ];
                let mut states = Vec::with_capacity(5);
                for (mat, n_in) in mats.iter().zip([hidden, hidden, hidden, hidden, value_dim]) {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    states.push(gpu.pipeline(mat.kernel, 1, lpr)?.to_owned());
                }
                let conv1d = {
                    let addr = g.conv1d.as_ptr() as usize;
                    if addr % 4 != 0 {
                        return None;
                    }
                    let chunk = gpu.chunk_for(addr, g.conv1d.len())?;
                    NormRef {
                        chunk,
                        off: (addr - gpu.chunks[chunk].start) as u64,
                    }
                };
                Some(GdnRefs {
                    mats,
                    states: states.try_into().ok()?,
                    conv1d,
                    heads_k: g.heads_k as u32,
                    heads_v: g.heads_v as u32,
                    d: g.d as u32,
                    d_conv: g.d_conv as u32,
                    conv_off: g.conv_off as u64,
                    state_off: g.state_off as u64,
                    channels,
                    value_dim,
                })
            }
        };
        let wq_out = if l.gate_in_q { 2 * q_dim } else { q_dim };
        let mats_pre = if gdn.is_none() {
            let dims = [
                (hidden, wq_out),
                (hidden, l.wk.2),
                (hidden, l.wv.2),
                (q_dim, l.wo.2),
            ];
            why!(
                l.wq.2 != wq_out || l.wk.2 != kv || l.wv.2 != kv || l.wo.2 != hidden,
                "attn dims"
            );
            let mats = [
                resolve_or_decline("attn.wq", l.wq.0, l.wq.1, dbg, &mut gpu)?,
                resolve_or_decline("attn.wk", l.wk.0, l.wk.1, dbg, &mut gpu)?,
                resolve_or_decline("attn.wv", l.wv.0, l.wv.1, dbg, &mut gpu)?,
                resolve_or_decline("attn.wo", l.wo.0, l.wo.1, dbg, &mut gpu)?,
            ];
            Some((mats, dims))
        } else {
            None
        };
        // The spin-flag scheme replaces the two norm barriers when every
        // consumer kernel carries the WAIT variant: qkv (q4_k_mv) and the
        // router (f32). `ALLPAKA_NORMFLAG=0` reverts to barriers.
        let normflag = crate::runtime::get().normflag
            && mats_pre.as_ref().is_some_and(|(m, _)| {
                m.iter().take(3).all(|m| m.kernel == "matvec_q4_k_mv")
            })
            // Dense gate/up read h with no intervening barrier; only the MoE
            // chain (router waits on the flag, the rest orders behind its
            // barriers) is safe without the norm barrier.
            && matches!(l.ffn, TokenFfn::Moe { .. });
        let (mats, mat_states) = match mats_pre {
            Some((mats, dims)) => {
                let mut mat_states = Vec::with_capacity(4);
                for (i, (mat, &(n_in, _))) in mats.iter().zip(&dims).enumerate() {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    let wait = normflag && i < 3;
                    mat_states.push(
                        gpu.pipeline_wait(mat.kernel, 1, lpr, false, false, wait)?
                            .to_owned(),
                    );
                }
                (Some(mats), Some(mat_states.try_into().ok()?))
            }
            None => (None, None),
        };
        let ffn = match &l.ffn {
            TokenFfn::Dense { gate, up, down } => {
                if gate.2 != up.2 || down.2 != hidden {
                    return None;
                }
                let mats = [
                    resolve_or_decline("ffn.gate", gate.0, gate.1, dbg, &mut gpu)?,
                    resolve_or_decline("ffn.up", up.0, up.1, dbg, &mut gpu)?,
                    resolve_or_decline("ffn.down", down.0, down.1, dbg, &mut gpu)?,
                ];
                let mut states = Vec::with_capacity(3);
                for (mat, n_in) in mats.iter().zip([hidden, hidden, gate.2]) {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    states.push(gpu.pipeline(mat.kernel, 1, lpr)?.to_owned());
                }
                max_ffn = max_ffn.max(gate.2);
                FfnRefs::Dense {
                    mats,
                    states: states.try_into().ok()?,
                    ffn_dim: gate.2,
                }
            }
            TokenFfn::Moe {
                router,
                router_bias,
                gate,
                up,
                down,
                expert_ffn,
                n_used,
                sigmoid,
                shared,
                shared_gate,
            } => {
                why!(
                    router.0 != GgmlType::F32 || router.2 > 256 || *n_used > 16,
                    "router shape"
                );
                why!(
                    shared_gate.is_some() && shared.is_none(),
                    "shared gate without shared expert"
                );
                let router_ref = resolve_f32_or_decline("router", router.1, hidden, dbg, &mut gpu)?;
                let router_state = gpu
                    .pipeline_wait(
                        "matvec_f32",
                        1,
                        lanes_per_row(GgmlType::F32, hidden),
                        false,
                        false,
                        normflag,
                    )?
                    .to_owned();
                let mats = [
                    resolve_or_decline("ffn.gate", gate.0, gate.1, dbg, &mut gpu)?,
                    resolve_or_decline("ffn.up", up.0, up.1, dbg, &mut gpu)?,
                    resolve_or_decline("ffn.down", down.0, down.1, dbg, &mut gpu)?,
                ];
                let n_expert = router.2;
                let strides = [
                    (gate.1.len() / n_expert) as u64,
                    (up.1.len() / n_expert) as u64,
                    (down.1.len() / n_expert) as u64,
                ];
                let mut states = Vec::with_capacity(3);
                let n_outs = [*expert_ffn, *expert_ffn, hidden];
                let mut sw_fused = false;
                let mut gu_kernels = [""; 2];
                // Kernels carrying the SWIGLU_X down-projection variant.
                let sw_capable = |k: &str| {
                    matches!(
                        k,
                        "matvec_q2_k" | "matvec_q3_k_mv" | "matvec_q4_k_mv" | "matvec_q8_0"
                    )
                };
                // With a shared expert, fusion is only safe when the shared
                // down carries the variant too - otherwise its slot would
                // read raw gate values once the standalone swiglu is gone.
                let shared_down_sw = match shared {
                    Some(sh) => {
                        resolve_or_decline("ffn.shared.down", sh[2].0, sh[2].1, dbg, &mut gpu)
                            .is_some_and(|m| sw_capable(m.kernel))
                    }
                    None => true,
                };
                for (i, ((mat, n_in), n_out)) in mats
                    .iter()
                    .zip([hidden, hidden, *expert_ffn])
                    .zip(n_outs)
                    .enumerate()
                {
                    let lpr = lanes_per_row(mat.ty, n_in);
                    // The mv kernel's INDEXED form maps 4 consecutive flat
                    // rows to one SIMD group and one expert; n_out % 4 != 0
                    // would straddle experts, so fall back per matrix.
                    let kernel = match mat.kernel {
                        "matvec_q2_k_mv" if n_out % 4 != 0 => "matvec_q2_k",
                        "matvec_q3_k_mv" if n_out % 2 != 0 => "matvec_q3_k",
                        "matvec_q4_k_mv" if n_out % 2 != 0 => "matvec_q4_k",
                        "matvec_q6_k_mv" if n_out % 2 != 0 => "matvec_q6_k",
                        k => k,
                    };
                    if i < 2 {
                        gu_kernels[i] = kernel;
                    }
                    // The down projection folds swiglu into its loads when
                    // its kernel carries the variant; the standalone swiglu
                    // dispatch and one barrier then drop out of the layer.
                    let swiglu = i == 2
                        && sw_capable(kernel)
                        && shared_down_sw
                        && std::env::var("ALLPAKA_SWFUSE").is_ok_and(|v| v == "1");
                    if swiglu {
                        sw_fused = true;
                    }
                    states.push(gpu.pipeline_full(kernel, 1, lpr, true, swiglu)?.to_owned());
                }
                max_ffn = max_ffn.max(*expert_ffn);
                let shared_refs = match shared {
                    Some(sh) => {
                        // The shared expert rides the combine as an extra
                        // slot, so its ffn width must match a routed one.
                        why!(
                            sh[0].2 != *expert_ffn || sh[1].2 != *expert_ffn || sh[2].2 != hidden,
                            "shared dims"
                        );
                        let smats = [
                            resolve_or_decline("ffn.shared.gate", sh[0].0, sh[0].1, dbg, &mut gpu)?,
                            resolve_or_decline("ffn.shared.up", sh[1].0, sh[1].1, dbg, &mut gpu)?,
                            resolve_or_decline("ffn.shared.down", sh[2].0, sh[2].1, dbg, &mut gpu)?,
                        ];
                        let mut sstates = Vec::with_capacity(3);
                        for (i, (mat, n_in)) in
                            smats.iter().zip([hidden, hidden, *expert_ffn]).enumerate()
                        {
                            let lpr = lanes_per_row(mat.ty, n_in);
                            let swiglu = i == 2 && sw_fused && sw_capable(mat.kernel);
                            sstates.push(
                                gpu.pipeline_full(mat.kernel, 1, lpr, false, swiglu)?
                                    .to_owned(),
                            );
                        }
                        max_slots = max_slots.max(*n_used + 1);
                        Some((smats, sstates.try_into().ok()?))
                    }
                    None => {
                        max_slots = max_slots.max(*n_used);
                        None
                    }
                };
                max_expert = max_expert.max(n_expert);
                let mega = if mega_enabled() && n_expert <= 128 && shared_gate.is_none() {
                    let gf = mega_fmt(mats[0].ty);
                    let uf = mega_fmt(mats[1].ty);
                    let df = mega_fmt(mats[2].ty);
                    // GLM's shared expert needs its own formats in the
                    // kernel; without one they are 0 and unused.
                    let (sgf, suf, sdf) = match &shared_refs {
                        Some((smats, _)) => (
                            mega_fmt(smats[0].ty),
                            mega_fmt(smats[1].ty),
                            mega_fmt(smats[2].ty),
                        ),
                        None => (Some(0), Some(0), Some(0)),
                    };
                    match (gf, uf, df, sgf, suf, sdf) {
                        (Some(g), Some(u), Some(d), Some(sg), Some(su), Some(sd))
                            if g == u && sg == su =>
                        {
                            gpu.mega_pipeline(g, d, sg, sd).map(|p| p.to_owned())
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                // gate + up as ONE dual-output indexed dispatch: same h,
                // same expert ids, one launch less per layer
                // (`ALLPAKA_DECODE_GUFUSE=1`). Requires the q4_k_mv kernel
                // pair, equal expert strides and an even row space so the
                // 2-row simdgroup pairs never straddle the gate/up boundary.
                let gu_dual = if gufuse()
                    && gu_kernels == ["matvec_q4_k_mv"; 2]
                    && strides[0] == strides[1]
                    && (*expert_ffn * *n_used) % 2 == 0
                {
                    gpu.pipeline_dual("matvec_q4_k_mv", 1, lanes_per_row(mats[0].ty, hidden))
                        .map(|p| p.to_owned())
                } else {
                    None
                };
                FfnRefs::Moe {
                    router: router_ref,
                    router_state,
                    router_bias: match router_bias {
                        Some(b) => Some(resolve_norm(&gpu, b, n_expert)?),
                        None => None,
                    },
                    n_expert,
                    mats,
                    states: states.try_into().ok()?,
                    strides,
                    expert_ffn: *expert_ffn,
                    n_used: *n_used,
                    sw_fused,
                    mega,
                    sigmoid: *sigmoid,
                    shared: shared_refs,
                    shared_gate: match shared_gate {
                        Some(sg) => {
                            why!(sg.len() != hidden * 4, "shared gate dims");
                            let mat = resolve_f32_or_decline(
                                "ffn.shared.gate_out",
                                sg,
                                hidden,
                                dbg,
                                &mut gpu,
                            )?;
                            let state = gpu
                                .pipeline("matvec_f32", 1, lanes_per_row(GgmlType::F32, hidden))?
                                .to_owned();
                            Some((mat, state))
                        }
                        None => None,
                    },
                    gu_dual,
                }
            }
        };
        let Some(attn_norm_ref) = resolve_norm(&gpu, l.attn_norm, hidden) else {
            why!(true, "attn norm resolve");
            return None;
        };
        let Some(ffn_norm_ref) = resolve_norm(&gpu, l.ffn_norm, hidden) else {
            why!(true, "ffn norm resolve");
            return None;
        };
        let bias_ref = |raw: Option<&[u8]>, n: usize| -> Option<Option<NormRef>> {
            match raw {
                Some(b) => Some(Some(resolve_norm(&gpu, b, n)?)),
                None => Some(None),
            }
        };
        layers.push(LayerRefs {
            attn_norm: attn_norm_ref,
            ffn_norm: ffn_norm_ref,
            mats,
            mat_states,
            q_bias: bias_ref(l.q_bias, q_dim)?,
            k_bias: bias_ref(l.k_bias, kv)?,
            v_bias: bias_ref(l.v_bias, kv)?,
            gdn,
            ffn,
            normflag,
        });
    }
    let out_norm = resolve_norm(&gpu, req.output_norm, hidden)?;
    let out_mat = resolve(&gpu, req.output.0, req.output.1)?;
    let out_state = gpu
        .pipeline(out_mat.kernel, 1, lanes_per_row(out_mat.ty, hidden))?
        .to_owned();
    let attend_state = gpu.pipelines[&(attend_kernel(), 1, 1)].to_owned();
    let attend_split_state = gpu.pipelines[&("attend_split", 1, 1)].to_owned();
    let attend_merge_state = gpu.pipelines[&("attend_merge", 1, 1)].to_owned();
    let qk_prep_state = gpu.pipelines[&("qk_prep", 1, 1)].to_owned();
    let resnorm_state = gpu.pipelines[&("residual_norm", 1, 1)].to_owned();
    let resnorm_sig_state = gpu.pipelines[&("residual_norm_sig", 1, 1)].to_owned();
    // head_dim-256 attention (qwen35moe) and the GDN branch.
    let qk_prep256_state = gpu.pipelines[&("qk_prep256", 1, 1)].to_owned();
    let attend256_state = gpu.pipelines[&("attend_s32_256", 1, 1)].to_owned();
    let attend_split256_state = gpu.pipelines[&("attend_split_256", 1, 1)].to_owned();
    let gate_mul_state = gpu.pipelines[&("attn_gate_mul", 1, 1)].to_owned();
    let gdn_conv_state = gpu.pipelines[&("gdn_conv", 1, 1)].to_owned();
    let gdn_step_state = gpu.pipelines[&("gdn_step", 1, 1)].to_owned();
    let gdn_norm_state = gpu.pipelines[&("gdn_out_norm", 1, 1)].to_owned();
    let sigmoid_gating = req
        .layers
        .iter()
        .any(|l| matches!(l.ffn, TokenFfn::Moe { sigmoid: true, .. }));
    let topk_state = gpu.pipelines[&(
        if sigmoid_gating {
            "sigmoid_topk"
        } else {
            "softmax_topk"
        },
        1,
        1,
    )]
        .to_owned();
    let rtopk_state = gpu.pipelines[&("router_topk", 1, 1)].to_owned();
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();
    let combine_state = gpu.pipelines[&("moe_combine", 1, 1)].to_owned();
    let combine_resnorm_state = gpu.pipelines[&("combine_resnorm", 1, 1)].to_owned();
    let resnorm_router_state = gpu.pipelines[&("resnorm_router", 1, 1)].to_owned();
    let argmax_state = gpu.pipelines[&("argmax_f32", 1, 1)].to_owned();
    let argmax_final_state = gpu.pipelines[&("argmax_final", 1, 1)].to_owned();

    // The speculative multi-token verify: same resolved layers, one command
    // buffer, batched projections (TILE = m) and per-token attention/MoE.
    if m > 1 {
        return encode_verify_tokens(req, &mut gpu, &layers, sigmoid_gating);
    }

    // Arena layout, element offsets over f32 (ids are u32-in-f32 slots).
    let align = |n: usize| (n + 63) & !63;
    // Raw wq output can be 2x q_dim when the gate rides inside it; the GDN
    // scratch pads size off the largest deltanet branch.
    let wq_max = req
        .layers
        .iter()
        .filter(|l| l.gdn.is_none())
        .map(|l| l.wq.2)
        .max()
        .unwrap_or(q_dim)
        .max(q_dim);
    let (mut max_channels, mut max_value_dim, mut max_heads_v) = (0usize, 0usize, 1usize);
    for l in req.layers {
        if let Some(g) = &l.gdn {
            max_channels = max_channels.max(g.heads_k * g.d * 2 + g.heads_v * g.d);
            max_value_dim = max_value_dim.max(g.heads_v * g.d);
            max_heads_v = max_heads_v.max(g.heads_v);
        }
    }
    let x_at = 0usize;
    let delta_at = x_at + align(hidden);
    let h_at = delta_at + align(hidden);
    let q_at = h_at + align(hidden);
    // Deinterleaved normed q and the raw gate rows (head_dim-256 path).
    let qn_at = q_at + align(wq_max);
    let qgate_at = qn_at + align(q_dim);
    let k_at = qgate_at + align(q_dim);
    let v_at = k_at + align(kv);
    let attn_at = v_at + align(kv);
    // GDN scratch: the conv-mixed qkv row, z gate, alpha/beta, deltanet out.
    let gqkv_at = attn_at + align(q_dim);
    let gz_at = gqkv_at + align(max_channels);
    let gab_at = gz_at + align(max_value_dim);
    let gy_at = gab_at + align(2 * max_heads_v);
    let logits_at = gy_at + align(max_value_dim);
    let ids_at = logits_at + align(max_expert.max(1));
    let wts_at = ids_at + align(max_slots);
    let gate_at = wts_at + align(max_slots);
    let up_at = gate_at + align(max_slots * max_ffn);
    let downo_at = up_at + align(max_slots * max_ffn);
    let out_logits_at = downo_at + align(max_slots * hidden);
    let flag_at = out_logits_at + align(vocab);
    let ctr_at = flag_at + align(1);
    // Flash-decoding split fan-out: past a few hundred cached positions one
    // threadgroup per q head walks the KV cache too serially (measured 86 us
    // per layer at 544 tokens against ~2 us of cache bytes). Split the
    // positions across nsplit threadgroups and merge the partials.
    // ALLPAKA_ATTN_SPLIT forces the fan-out, 1 disables the split path.
    let n_pos = req.pos + 1;
    let forced_split = crate::runtime::get().attention_split;
    // On M4 Max, twelve slices are consistently faster than sixteen once the
    // cache is long enough to saturate memory-level parallelism. Keep the
    // larger range available to explicit overrides for other GPUs.
    let nsplit = forced_split
        .unwrap_or_else(|| (n_pos / 128).clamp(1, 12))
        .clamp(1, 16)
        .min(n_pos);
    let sp_acc_at = ctr_at + align(1);
    let sp_md_at = sp_acc_at + align(req.n_heads * nsplit * hd);
    // The greedy argmax: 64 (value, index) partial pairs plus the winner.
    let amax_at = sp_md_at + align(req.n_heads * nsplit * 2);
    let total = amax_at + align(64 * 2 + 1);
    gpu.ensure_arenas(4096, total * 4);

    // Stage x and zero the first delta.
    unsafe {
        let yp = gpu.y_arena.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(req.x.as_ptr(), yp.add(x_at), hidden);
        std::ptr::write_bytes(yp.add(delta_at), 0, hidden);
        std::ptr::write_bytes(yp.add(flag_at), 0, 1);
        std::ptr::write_bytes(yp.add(ctr_at), 0, 1);
        // The shared expert's combine weight is a constant 1 in the slot past
        // the router-written ones - unless the layer carries qwen35moe's
        // gate projection, whose matvec overwrites it during the token.
        for l in &layers {
            if let FfnRefs::Moe {
                n_used,
                shared: Some(_),
                shared_gate: None,
                ..
            } = &l.ffn
            {
                *yp.add(wts_at + *n_used) = 1.0;
            }
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // ALLPAKA_DECODE_SERIAL=1: serial encoder, implicit ordering, all
        // explicit barriers skipped - probes whether the driver's own
        // hazard tracking beats our barrier drains per stage boundary.
        let serial = crate::runtime::get().decode_serial;
        // ALLPAKA_DECODE_SPLIT=1: sample the GPU timestamp counter at existing
        // stage boundaries. Profile mode uses multiple compute encoders inside
        // the same command buffer; production mode remains one concurrent
        // encoder. Both modes keep one submit and one wait per token.
        let split = std::env::var_os("ALLPAKA_DECODE_SPLIT").is_some();
        const MAX_SPLIT_SAMPLES: u64 = 2048;
        let split_counter = if split
            && gpu
                .device
                .supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
        {
            let desc = metal::CounterSampleBufferDescriptor::new();
            desc.set_storage_mode(metal::MTLStorageMode::Shared);
            desc.set_sample_count(MAX_SPLIT_SAMPLES);
            gpu.device
                .counter_sets()
                .iter()
                .find(|set| set.name() == "timestamp")
                .and_then(|set| {
                    desc.set_counter_set(set);
                    gpu.device
                        .new_counter_sample_buffer_with_descriptor(&desc)
                        .ok()
                })
        } else {
            None
        };
        let split_resolved = split_counter.as_ref().map(|_| {
            gpu.device.new_buffer(
                MAX_SPLIT_SAMPLES * std::mem::size_of::<u64>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        });
        let mut split_labels: Vec<&'static str> = Vec::new();
        let mut split_index = 0u64;
        let mut split_cpu_start = 0u64;
        let mut split_gpu_start = 0u64;
        let profile_encoder = |counter: &metal::CounterSampleBufferRef, start: u64, end: u64| {
            let desc = metal::ComputePassDescriptor::new();
            let attachment = desc
                .sample_buffer_attachments()
                .object_at(0)
                .expect("Metal compute counter attachment 0");
            attachment.set_sample_buffer(counter);
            attachment.set_start_of_encoder_sample_index(start);
            attachment.set_end_of_encoder_sample_index(end);
            cmd.compute_command_encoder_with_descriptor(desc)
        };
        let mut enc = if let Some(counter) = split_counter.as_ref() {
            gpu.device
                .sample_timestamps(&mut split_cpu_start, &mut split_gpu_start);
            split_index = 2;
            profile_encoder(counter, 0, 1)
        } else {
            cmd.compute_command_encoder_with_dispatch_type(if serial {
                metal::MTLDispatchType::Serial
            } else {
                metal::MTLDispatchType::Concurrent
            })
        };
        macro_rules! split_here {
            ($label:expr) => {
                if let Some(counter) = split_counter.as_ref() {
                    if split_index + 1 < MAX_SPLIT_SAMPLES {
                        enc.end_encoding();
                        split_labels.push($label);
                        enc = profile_encoder(counter, split_index, split_index + 1);
                        split_index += 2;
                    }
                }
            };
        }
        let y = &gpu.y_arena;
        // Probe: ALLPAKA_SKIP=bar drops every barrier; barn/bara/barf drop
        // the norm / attention-chain / ffn-chain subsets (values race,
        // timing shows what each sync class costs).
        let no_bar = probe_skip("bar");
        let no_barn = probe_skip("barn");
        let no_bara = probe_skip("bara");
        let no_barf = probe_skip("barf");
        let bar_c = |enc: &metal::ComputeCommandEncoderRef, class: u8| {
            let skip = serial
                || no_bar
                || (class == b'n' && no_barn)
                || (class == b'a' && no_bara)
                || (class == b'f' && no_barf);
            if !skip {
                enc.memory_barrier_with_resources(&[y]);
            }
        };
        let bar = |enc: &metal::ComputeCommandEncoderRef| bar_c(enc, b'x');
        let e = |off: usize| (off * 4) as u64;

        // One plain matvec: in and out are arena element offsets.
        let matvec = |enc: &metal::ComputeCommandEncoderRef,
                      state: &ComputePipelineState,
                      mat: &MatRef,
                      n_in: usize,
                      n_out: usize,
                      x_off: usize,
                      y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            let lpr = lanes_per_row(mat.ty, n_in) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64 * lpr).div_ceil(128), 1, 1),
                MTLSize::new(128, 1, 1),
            );
        };
        // The indexed form: n_out is per expert, the grid covers all slots.
        let matvec_idx = |enc: &metal::ComputeCommandEncoderRef,
                          state: &ComputePipelineState,
                          mat: &MatRef,
                          n_in: usize,
                          n_out: usize,
                          x_off: usize,
                          y_off: usize,
                          stride: u64,
                          slots: usize,
                          x_stride: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(y), e(ids_at));
            let idx = GpuIdxArgs {
                stride,
                slots: slots as u32,
                x_stride: x_stride as u32,
                ids_stride: 0,
                x_row_stride: 0,
                y_row_stride: 0,
                n_rows: 0,
            };
            enc.set_bytes(
                7,
                std::mem::size_of::<GpuIdxArgs>() as u64,
                &idx as *const GpuIdxArgs as *const _,
            );
            let lpr = lanes_per_row(mat.ty, n_in) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new(((n_out * slots) as u64 * lpr).div_ceil(128), 1, 1),
                MTLSize::new(128, 1, 1),
            );
        };
        let resnorm = |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef| {
            enc.set_compute_pipeline_state(&resnorm_state);
            enc.set_buffer(0, Some(y), e(x_at));
            enc.set_buffer(1, Some(y), e(delta_at));
            enc.set_buffer(2, Some(y), e(h_at));
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        };
        // The MoE combine: delta = sum over slots of wts[s] * down[s].
        // sig_last: the last slot's weight is a raw gate logit (qwen35moe).
        let combine = |enc: &metal::ComputeCommandEncoderRef, slots: u32, sig_last: u32| {
            enc.set_compute_pipeline_state(&combine_state);
            enc.set_buffer(0, Some(y), e(downo_at));
            enc.set_buffer(1, Some(y), e(wts_at));
            enc.set_buffer(2, Some(y), e(delta_at));
            let n = hidden as u32;
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &slots as *const u32 as *const _);
            enc.set_bytes(5, 4, &sig_last as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((hidden as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        };
        // residual_norm with the combine folded in: one dispatch and one
        // barrier less per MoE layer boundary.
        let combine_resnorm =
            |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef, slots: u32, sig_last: u32| {
                enc.set_compute_pipeline_state(&combine_resnorm_state);
                enc.set_buffer(0, Some(y), e(x_at));
                enc.set_buffer(1, Some(y), e(delta_at));
                enc.set_buffer(2, Some(y), e(h_at));
                enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
                let n = hidden as u32;
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                enc.set_buffer(6, Some(y), e(downo_at));
                enc.set_buffer(7, Some(y), e(wts_at));
                enc.set_bytes(8, 4, &slots as *const u32 as *const _);
                enc.set_bytes(9, 4, &sig_last as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            };
        // The signalling form: publishes `epoch` to the flag word instead of
        // relying on the barrier the caller then skips.
        let resnorm_sig = |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef, epoch: u32| {
            enc.set_compute_pipeline_state(&resnorm_sig_state);
            enc.set_buffer(0, Some(y), e(x_at));
            enc.set_buffer(1, Some(y), e(delta_at));
            enc.set_buffer(2, Some(y), e(h_at));
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.set_buffer(6, Some(y), e(flag_at));
            enc.set_bytes(7, 4, &epoch as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        };

        let mut dispatched = 0u64;
        // MoE combine work deferred into the next norm: Some((slot count,
        // sig_last)) between the down stage of one MoE layer and the
        // residual norm that absorbs it (the next layer's attention norm, or
        // the final norm).
        let mut pending_combine: Option<(u32, u32)> = None;
        for (li, (l, refs)) in req.layers.iter().zip(&layers).enumerate() {
            let ep_attn = (li as u32) * 2 + 1;
            let ep_ffn = (li as u32) * 2 + 2;
            // h = rmsnorm(x + delta) * attn_norm; x absorbs delta. A pending
            // combine folds into the norm kernel; the normflag path keeps
            // the separate combine (its norm has no fused sig variant).
            let fused = match pending_combine.take() {
                Some((slots, sig_last)) if !refs.normflag => {
                    combine_resnorm(enc, &refs.attn_norm, slots, sig_last);
                    bar_c(enc, b'n');
                    true
                }
                Some((slots, sig_last)) => {
                    combine(enc, slots, sig_last);
                    bar(enc);
                    false
                }
                None => false,
            };
            if !fused {
                if refs.normflag {
                    resnorm_sig(enc, &refs.attn_norm, ep_attn);
                    enc.set_buffer(9, Some(y), e(flag_at));
                    enc.set_bytes(10, 4, &ep_attn as *const u32 as *const _);
                    if probe_skip("keepbar") {
                        bar_c(enc, b'n');
                    }
                } else {
                    resnorm(enc, &refs.attn_norm);
                    bar_c(enc, b'n');
                }
            }
            split_here!("resnorm");

            // The gated-delta-net branch replaces the whole attention half.
            if let Some(g) = &refs.gdn {
                let tg = &l.gdn.as_ref().expect("gdn refs without gdn layer");
                let ssm_buf = &req.ssm.expect("gdn layer without ssm region").buf;
                if !probe_skip("qkv") {
                    matvec(
                        enc,
                        &g.states[0],
                        &g.mats[0],
                        hidden,
                        g.channels,
                        h_at,
                        gqkv_at,
                    );
                    matvec(
                        enc,
                        &g.states[1],
                        &g.mats[1],
                        hidden,
                        g.value_dim,
                        h_at,
                        gz_at,
                    );
                    matvec(
                        enc,
                        &g.states[2],
                        &g.mats[2],
                        hidden,
                        g.heads_v as usize,
                        h_at,
                        gab_at,
                    );
                    matvec(
                        enc,
                        &g.states[3],
                        &g.mats[3],
                        hidden,
                        g.heads_v as usize,
                        h_at,
                        gab_at + g.heads_v as usize,
                    );
                }
                bar_c(enc, b'a');
                split_here!("qkv");
                if !probe_skip("attend") {
                    // Depthwise conv (+silu), updating the window in place.
                    enc.set_compute_pipeline_state(&gdn_conv_state);
                    enc.set_buffer(0, Some(y), e(gqkv_at));
                    enc.set_buffer(1, Some(ssm_buf), 0);
                    enc.set_buffer(2, Some(&gpu.chunks[g.conv1d.chunk].buf), g.conv1d.off);
                    let cargs = GdnConvArgs {
                        channels: g.channels as u32,
                        d_conv: g.d_conv,
                        pad0: 0,
                        pad1: 0,
                        conv_off: g.conv_off,
                    };
                    enc.set_bytes(
                        3,
                        std::mem::size_of::<GdnConvArgs>() as u64,
                        &cargs as *const GdnConvArgs as *const _,
                    );
                    enc.dispatch_thread_groups(
                        MTLSize::new((g.channels as u64).div_ceil(256), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // The recurrence, l2 norms folded in, state in place.
                    enc.set_compute_pipeline_state(&gdn_step_state);
                    enc.set_buffer(0, Some(ssm_buf), 0);
                    enc.set_buffer(1, Some(y), e(gqkv_at));
                    enc.set_buffer(2, Some(y), e(gab_at));
                    let sargs = GdnStepArgs {
                        heads_k: g.heads_k,
                        heads_v: g.heads_v,
                        d: g.d,
                        key_dim: g.heads_k * g.d,
                        eps: req.eps,
                        pad0: 0,
                        state_off: g.state_off,
                    };
                    enc.set_bytes(
                        3,
                        std::mem::size_of::<GdnStepArgs>() as u64,
                        &sargs as *const GdnStepArgs as *const _,
                    );
                    enc.set_bytes(4, (tg.a.len() * 4) as u64, tg.a.as_ptr() as *const _);
                    enc.set_bytes(5, (tg.dt.len() * 4) as u64, tg.dt.as_ptr() as *const _);
                    enc.set_buffer(6, Some(y), e(gy_at));
                    enc.dispatch_thread_groups(
                        MTLSize::new((g.d / 4) as u64, g.heads_v as u64, 1),
                        MTLSize::new(32, 4, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // Gated rmsnorm over the head outputs, times silu(z).
                    enc.set_compute_pipeline_state(&gdn_norm_state);
                    enc.set_buffer(0, Some(y), e(gy_at));
                    enc.set_buffer(1, Some(y), e(gz_at));
                    enc.set_bytes(
                        2,
                        (tg.ssm_norm.len() * 4) as u64,
                        tg.ssm_norm.as_ptr() as *const _,
                    );
                    let hv = g.heads_v;
                    let dd = g.d;
                    enc.set_bytes(3, 4, &hv as *const u32 as *const _);
                    enc.set_bytes(4, 4, &dd as *const u32 as *const _);
                    enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(g.heads_v as u64, 1, 1),
                        MTLSize::new(g.d as u64, 1, 1),
                    );
                }
                split_here!("attend");
                bar_c(enc, b'a');
                // ssm_out straight into delta.
                if !probe_skip("qkv") {
                    matvec(
                        enc,
                        &g.states[4],
                        &g.mats[4],
                        g.value_dim,
                        hidden,
                        gy_at,
                        delta_at,
                    );
                }
                bar_c(enc, b'a');
                split_here!("wo");
            } else {
                // qkv from h.
                let amats = refs.mats.as_ref().expect("attention layer without mats");
                let astates = refs
                    .mat_states
                    .as_ref()
                    .expect("attention layer without mat states");
                if !probe_skip("qkv") {
                    matvec(enc, &astates[0], &amats[0], hidden, l.wq.2, h_at, q_at);
                    matvec(enc, &astates[1], &amats[1], hidden, kv, h_at, k_at);
                    matvec(enc, &astates[2], &amats[2], hidden, kv, h_at, v_at);
                }
                bar_c(enc, b'a');
                split_here!("qkv");
                // Norm + rope + cache store.
                if !probe_skip("attend") {
                    if hd == 256 {
                        // head_dim-256 path: qk_prep256 deinterleaves the fused gate and
                        // writes normed q to qn_at; attend_s32_256/attend_split_256 run
                        // the scores; the sigmoid gate lands before wo.
                        enc.set_compute_pipeline_state(&qk_prep256_state);
                        enc.set_buffer(0, Some(y), e(q_at));
                        enc.set_buffer(1, Some(y), e(k_at));
                        enc.set_buffer(2, Some(y), e(v_at));
                        enc.set_buffer(3, Some(&req.cache.buf), 0);
                        let args = QkPrep256Args {
                            n_heads: req.n_heads as u32,
                            n_kv_heads: req.n_kv_heads as u32,
                            kv_dim: req.kv_dim as u32,
                            eps: req.eps,
                            pos: req.pos as u32,
                            has_qk_norm: l.q_norm.is_some() as u32,
                            rot_dim: req.rot_dim as u32,
                            gate_in_q: l.gate_in_q as u32,
                            k_base: l.k_off as u64,
                            v_base: l.v_off as u64,
                        };
                        enc.set_bytes(
                            4,
                            std::mem::size_of::<QkPrep256Args>() as u64,
                            &args as *const QkPrep256Args as *const _,
                        );
                        let ones = [1.0f32; 256];
                        let qw = l.q_norm.unwrap_or(&ones);
                        let kw = l.k_norm.unwrap_or(&ones);
                        enc.set_bytes(5, (256 * 4) as u64, qw.as_ptr() as *const _);
                        enc.set_bytes(6, (256 * 4) as u64, kw.as_ptr() as *const _);
                        enc.set_bytes(
                            7,
                            (req.rope.len() * 8) as u64,
                            req.rope.as_ptr() as *const _,
                        );
                        enc.set_buffer(8, Some(y), e(qn_at));
                        enc.set_buffer(9, Some(y), e(qgate_at));
                        enc.dispatch_thread_groups(
                            MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                            MTLSize::new(32, 1, 1),
                        );
                        enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                        let group = (req.n_heads / req.n_kv_heads.max(1)) as u32;
                        enc.set_buffer(0, Some(&req.cache.buf), 0);
                        enc.set_buffer(1, Some(&req.cache.buf), 0);
                        enc.set_buffer(2, Some(y), e(qn_at));
                        for (index, value) in [
                            (4u64, req.kv_dim as u32),
                            (5, 256u32),
                            (6, (req.pos + 1) as u32),
                            (7, group),
                        ] {
                            enc.set_bytes(index, 4, &value as *const u32 as *const _);
                        }
                        enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                        for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                            enc.set_bytes(index, 8, &value as *const u64 as *const _);
                        }
                        if nsplit > 1 {
                            enc.set_compute_pipeline_state(&attend_split256_state);
                            enc.set_buffer(3, Some(y), e(sp_acc_at));
                            enc.set_buffer(11, Some(y), e(sp_md_at));
                            enc.set_bytes(12, 4, &(nsplit as u32) as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((req.n_heads * nsplit) as u64, 1, 1),
                                MTLSize::new(512, 1, 1),
                            );
                            enc.memory_barrier_with_resources(&[y]);
                            enc.set_compute_pipeline_state(&attend_merge_state);
                            enc.set_buffer(0, Some(y), e(sp_acc_at));
                            enc.set_buffer(1, Some(y), e(sp_md_at));
                            enc.set_buffer(2, Some(y), e(attn_at));
                            enc.set_bytes(3, 4, &(nsplit as u32) as *const u32 as *const _);
                            enc.set_bytes(4, 4, &(hd as u32) as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        } else {
                            enc.set_compute_pipeline_state(&attend256_state);
                            enc.set_buffer(3, Some(y), e(attn_at));
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(512, 1, 1),
                            );
                        }
                        if l.gate_in_q {
                            enc.memory_barrier_with_resources(&[y]);
                            enc.set_compute_pipeline_state(&gate_mul_state);
                            enc.set_buffer(0, Some(y), e(attn_at));
                            enc.set_buffer(1, Some(y), e(qgate_at));
                            let n = (q_dim) as u32;
                            enc.set_bytes(2, 4, &n as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((q_dim as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                    } else {
                        enc.set_compute_pipeline_state(&qk_prep_state);
                        enc.set_buffer(0, Some(y), e(q_at));
                        enc.set_buffer(1, Some(y), e(k_at));
                        enc.set_buffer(2, Some(y), e(v_at));
                        enc.set_buffer(3, Some(&req.cache.buf), 0);
                        let args = QkPrepArgs {
                            n_heads: req.n_heads as u32,
                            n_kv_heads: req.n_kv_heads as u32,
                            head_dim: hd as u32,
                            kv_dim: req.kv_dim as u32,
                            eps: req.eps,
                            pos: req.pos as u32,
                            has_qk_norm: l.q_norm.is_some() as u32,
                            rot_dim: req.rot_dim as u32,
                            k_base: l.k_off as u64,
                            v_base: l.v_off as u64,
                            has_bias: refs.q_bias.is_some() as u32,
                            pad2: 0,
                        };
                        enc.set_bytes(
                            4,
                            std::mem::size_of::<QkPrepArgs>() as u64,
                            &args as *const QkPrepArgs as *const _,
                        );
                        let ones = [1.0f32; ATTN_HEAD_DIM];
                        let qw = l.q_norm.unwrap_or(&ones);
                        let kw = l.k_norm.unwrap_or(&ones);
                        enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
                        enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
                        enc.set_bytes(
                            7,
                            (req.rope.len() * 8) as u64,
                            req.rope.as_ptr() as *const _,
                        );
                        // Bias buffers: real F32 regions when the model has them, any
                        // valid buffer otherwise (the kernel skips them on has_bias == 0).
                        let dummy = &gpu.chunks[refs.attn_norm.chunk].buf;
                        for (index, b) in
                            [(8u64, &refs.q_bias), (9, &refs.k_bias), (10, &refs.v_bias)]
                        {
                            match b {
                                Some(r) => {
                                    enc.set_buffer(index, Some(&gpu.chunks[r.chunk].buf), r.off)
                                }
                                None => enc.set_buffer(index, Some(dummy), 0),
                            }
                        }
                        enc.dispatch_thread_groups(
                            MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                            MTLSize::new(32, 1, 1),
                        );
                        enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                        // Attention.
                        let attn_req = AttnReq {
                            cache: req.cache,
                            k_off: l.k_off,
                            v_off: l.v_off,
                            q: &[],
                            kv_dim: req.kv_dim,
                            head_dim: hd,
                            n_q_heads: req.n_heads,
                            group: req.n_heads / req.n_kv_heads.max(1),
                            n_pos: req.pos + 1,
                            scale: req.scale,
                        };
                        enc.set_buffer(0, Some(&req.cache.buf), 0);
                        enc.set_buffer(1, Some(&req.cache.buf), 0);
                        enc.set_buffer(2, Some(y), e(q_at));
                        for (index, value) in [
                            (4u64, attn_req.kv_dim as u32),
                            (5, attn_req.head_dim as u32),
                            (6, attn_req.n_pos as u32),
                            (7, attn_req.group as u32),
                        ] {
                            enc.set_bytes(index, 4, &value as *const u32 as *const _);
                        }
                        enc.set_bytes(8, 4, &attn_req.scale as *const f32 as *const _);
                        for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                            enc.set_bytes(index, 8, &value as *const u64 as *const _);
                        }
                        if nsplit > 1 {
                            // Flash-decoding: partial attention per position slice, then
                            // a merge. The scalar bindings 4..=10 set above are identical;
                            // only the outputs change.
                            enc.set_compute_pipeline_state(&attend_split_state);
                            enc.set_buffer(3, Some(y), e(sp_acc_at));
                            enc.set_buffer(11, Some(y), e(sp_md_at));
                            enc.set_bytes(12, 4, &(nsplit as u32) as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((req.n_heads * nsplit) as u64, 1, 1),
                                MTLSize::new(512, 1, 1),
                            );
                            enc.memory_barrier_with_resources(&[y]);
                            enc.set_compute_pipeline_state(&attend_merge_state);
                            enc.set_buffer(0, Some(y), e(sp_acc_at));
                            enc.set_buffer(1, Some(y), e(sp_md_at));
                            enc.set_buffer(2, Some(y), e(attn_at));
                            enc.set_bytes(3, 4, &(nsplit as u32) as *const u32 as *const _);
                            enc.set_bytes(4, 4, &(hd as u32) as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        } else {
                            enc.set_compute_pipeline_state(&attend_state);
                            enc.set_buffer(3, Some(y), e(attn_at));
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(attend_tg(), 1, 1),
                            );
                        }
                    }
                }
                split_here!("attend");
                bar_c(enc, b'a');
                // Output projection straight into delta.
                if !probe_skip("qkv") {
                    matvec(
                        enc,
                        &astates[3],
                        &amats[3],
                        q_dim,
                        hidden,
                        attn_at,
                        delta_at,
                    );
                }
                bar_c(enc, b'a');
                split_here!("wo");
            }
            // The megakernel absorbs the whole FFN half, resnorm included.
            if let FfnRefs::Moe {
                mega: Some(mega_state),
                router,
                router_bias,
                mats,
                strides,
                n_expert,
                expert_ffn,
                n_used,
                sigmoid,
                shared,
                ..
            } = &refs.ffn
            {
                // The megakernel covers softmax (Qwen) and sigmoid+bias
                // gating with a shared-expert slot (GLM) when the quant
                // formats have cores in the kernel.
                {
                    enc.set_compute_pipeline_state(mega_state);
                    enc.set_buffer(0, Some(y), 0);
                    enc.set_buffer(1, Some(&gpu.chunks[mats[0].chunk].buf), 0);
                    enc.set_buffer(2, Some(&gpu.chunks[mats[1].chunk].buf), 0);
                    enc.set_buffer(3, Some(&gpu.chunks[mats[2].chunk].buf), 0);
                    enc.set_buffer(4, Some(&gpu.chunks[router.chunk].buf), 0);
                    enc.set_buffer(
                        5,
                        Some(&gpu.chunks[refs.ffn_norm.chunk].buf),
                        refs.ffn_norm.off,
                    );
                    let ntg = mega_tg();
                    let margs = GpuMegaArgs {
                        hidden: hidden as u32,
                        ffn: *expert_ffn as u32,
                        n_expert: *n_expert as u32,
                        n_used: *n_used as u32,
                        x_at: x_at as u32,
                        delta_at: delta_at as u32,
                        h_at: h_at as u32,
                        logits_at: logits_at as u32,
                        ids_at: ids_at as u32,
                        wts_at: wts_at as u32,
                        gate_at: gate_at as u32,
                        up_at: up_at as u32,
                        downo_at: downo_at as u32,
                        ctr_at: ctr_at as u32,
                        n_tg: ntg,
                        ctr_base: (li as u32) * 6 * ntg,
                        eps: req.eps,
                        _pad: 0,
                        gate_off: mats[0].w_off,
                        up_off: mats[1].w_off,
                        down_off: mats[2].w_off,
                        router_off: router.w_off,
                        gate_stride: strides[0],
                        up_stride: strides[1],
                        down_stride: strides[2],
                        sigmoid: *sigmoid as u32,
                        has_shared: shared.is_some() as u32,
                        sh_gate_off: shared.as_ref().map_or(0, |(m, _)| m[0].w_off),
                        sh_up_off: shared.as_ref().map_or(0, |(m, _)| m[1].w_off),
                        sh_down_off: shared.as_ref().map_or(0, |(m, _)| m[2].w_off),
                    };
                    enc.set_bytes(
                        6,
                        std::mem::size_of::<GpuMegaArgs>() as u64,
                        &margs as *const GpuMegaArgs as *const _,
                    );
                    // Buffers 7-10: router bias and the shared expert's weights;
                    // dummies when unused (the kernel skips them on the flags).
                    match router_bias {
                        Some(rb) => enc.set_buffer(7, Some(&gpu.chunks[rb.chunk].buf), rb.off),
                        None => enc.set_buffer(7, Some(y), 0),
                    }
                    match shared {
                        Some((smats, _)) => {
                            enc.set_buffer(8, Some(&gpu.chunks[smats[0].chunk].buf), 0);
                            enc.set_buffer(9, Some(&gpu.chunks[smats[1].chunk].buf), 0);
                            enc.set_buffer(10, Some(&gpu.chunks[smats[2].chunk].buf), 0);
                        }
                        None => {
                            enc.set_buffer(8, Some(y), 0);
                            enc.set_buffer(9, Some(y), 0);
                            enc.set_buffer(10, Some(y), 0);
                        }
                    }
                    enc.dispatch_thread_groups(
                        MTLSize::new(ntg as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    bar_c(enc, b'f');
                    dispatched += 1;
                    continue;
                }
            }
            // h = rmsnorm(x + delta) * ffn_norm. On the plain MoE path the
            // norm fuses with the router into ONE dispatch below
            // (resnorm_router), so it runs there instead.
            // `ALLPAKA_RFUSE=0` reverts to the separate norm dispatch.
            let moe_plain = !refs.normflag && rfuse() && matches!(&refs.ffn, FfnRefs::Moe { .. });
            if !moe_plain {
                if refs.normflag {
                    resnorm_sig(enc, &refs.ffn_norm, ep_ffn);
                    enc.set_buffer(9, Some(y), e(flag_at));
                    enc.set_bytes(10, 4, &ep_ffn as *const u32 as *const _);
                    if probe_skip("keepbar") {
                        bar_c(enc, b'n');
                    }
                } else {
                    resnorm(enc, &refs.ffn_norm);
                    bar_c(enc, b'n');
                }
            }
            split_here!("resnorm");

            match &refs.ffn {
                FfnRefs::Dense {
                    mats,
                    states,
                    ffn_dim,
                } => {
                    matvec(enc, &states[0], &mats[0], hidden, *ffn_dim, h_at, gate_at);
                    matvec(enc, &states[1], &mats[1], hidden, *ffn_dim, h_at, up_at);
                    split_here!("gate_up");
                    bar(enc);
                    enc.set_compute_pipeline_state(&swiglu_state);
                    enc.set_buffer(0, Some(y), e(gate_at));
                    enc.set_buffer(1, Some(y), e(up_at));
                    let n = *ffn_dim as u32;
                    enc.set_bytes(2, 4, &n as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((n as u64).div_ceil(256), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    split_here!("swiglu");
                    bar(enc);
                    matvec(
                        enc, &states[2], &mats[2], *ffn_dim, hidden, gate_at, delta_at,
                    );
                    split_here!("down");
                    dispatched += 4;
                }
                FfnRefs::Moe {
                    router,
                    router_state,
                    router_bias,
                    n_expert,
                    mats,
                    states,
                    strides,
                    expert_ffn,
                    n_used,
                    sw_fused,
                    mega: _,
                    sigmoid,
                    shared,
                    shared_gate,
                    gu_dual,
                } => {
                    let n_slots = *n_used + shared.is_some() as usize;
                    let sig_last = shared_gate.is_some() as u32;
                    let mut ffn_dispatches = 0u64;
                    if !probe_skip("router") {
                        if moe_plain {
                            // FFN residual norm + router matvec + gating + top-k
                            // in ONE dispatch: both halves are single-threadgroup
                            // kernels back to back, so the boundary between them
                            // was pure launch and drain latency. Not under
                            // normflag: that path orders the norm by spin-flag.
                            enc.set_compute_pipeline_state(&resnorm_router_state);
                            enc.set_buffer(0, Some(y), e(x_at));
                            enc.set_buffer(1, Some(y), e(delta_at));
                            enc.set_buffer(2, Some(y), e(h_at));
                            enc.set_buffer(
                                3,
                                Some(&gpu.chunks[refs.ffn_norm.chunk].buf),
                                refs.ffn_norm.off,
                            );
                            let hd = hidden as u32;
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                            enc.set_buffer(6, Some(&gpu.chunks[router.chunk].buf), 0);
                            enc.set_bytes(7, 8, &router.w_off as *const u64 as *const _);
                            enc.set_buffer(8, Some(y), e(ids_at));
                            enc.set_buffer(9, Some(y), e(wts_at));
                            let n = *n_expert as u32;
                            let k = *n_used as u32;
                            enc.set_bytes(10, 4, &n as *const u32 as *const _);
                            enc.set_bytes(11, 4, &k as *const u32 as *const _);
                            match router_bias {
                                Some(rb) => {
                                    enc.set_buffer(12, Some(&gpu.chunks[rb.chunk].buf), rb.off)
                                }
                                None => enc.set_buffer(12, Some(&gpu.chunks[router.chunk].buf), 0),
                            }
                            let sg = *sigmoid as u32;
                            let hb = router_bias.is_some() as u32;
                            enc.set_bytes(13, 4, &sg as *const u32 as *const _);
                            enc.set_bytes(14, 4, &hb as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            ffn_dispatches += 1;
                        } else if !refs.normflag && rtopk_fused() {
                            // Router matvec + gating + top-k in ONE dispatch: the
                            // two tiny dispatches and the drain between them were
                            // pure stage-boundary latency. Not under normflag:
                            // that path orders the norm by spin-flag, which only
                            // the router matvec's WAIT variant can wait on.
                            enc.set_compute_pipeline_state(&rtopk_state);
                            enc.set_buffer(0, Some(&gpu.chunks[router.chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(h_at));
                            enc.set_buffer(2, Some(y), e(ids_at));
                            enc.set_buffer(3, Some(y), e(wts_at));
                            let hd = hidden as u32;
                            let n = *n_expert as u32;
                            let k = *n_used as u32;
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &n as *const u32 as *const _);
                            enc.set_bytes(6, 4, &k as *const u32 as *const _);
                            enc.set_bytes(7, 8, &router.w_off as *const u64 as *const _);
                            match router_bias {
                                Some(rb) => {
                                    enc.set_buffer(8, Some(&gpu.chunks[rb.chunk].buf), rb.off)
                                }
                                None => enc.set_buffer(8, Some(&gpu.chunks[router.chunk].buf), 0),
                            }
                            let sg = *sigmoid as u32;
                            let hb = router_bias.is_some() as u32;
                            enc.set_bytes(9, 4, &sg as *const u32 as *const _);
                            enc.set_bytes(10, 4, &hb as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            ffn_dispatches += 1;
                        } else {
                            matvec(
                                enc,
                                router_state,
                                router,
                                hidden,
                                *n_expert,
                                h_at,
                                logits_at,
                            );
                            bar(enc);
                            enc.set_compute_pipeline_state(&topk_state);
                            enc.set_buffer(0, Some(y), e(logits_at));
                            enc.set_buffer(1, Some(y), e(ids_at));
                            enc.set_buffer(2, Some(y), e(wts_at));
                            let n = *n_expert as u32;
                            let k = *n_used as u32;
                            enc.set_bytes(3, 4, &n as *const u32 as *const _);
                            enc.set_bytes(4, 4, &k as *const u32 as *const _);
                            // Sigmoid gating may carry a router bias (GLM exp_probs_b);
                            // the softmax kernel never reads past buffer 4.
                            match router_bias {
                                Some(rb) => {
                                    enc.set_buffer(5, Some(&gpu.chunks[rb.chunk].buf), rb.off);
                                    let one = 1u32;
                                    enc.set_bytes(6, 4, &one as *const u32 as *const _);
                                }
                                None => {
                                    enc.set_buffer(5, Some(&gpu.chunks[router.chunk].buf), 0);
                                    let zero = 0u32;
                                    enc.set_bytes(6, 4, &zero as *const u32 as *const _);
                                }
                            }
                            enc.dispatch_thread_groups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(32, 1, 1),
                            );
                            ffn_dispatches += 2;
                        }
                    }
                    split_here!("router");
                    // qwen35moe: the shared expert's gate projection writes
                    // the raw logit into its combine slot; the combine
                    // kernel applies the sigmoid.
                    if let Some((sg_mat, sg_state)) = shared_gate {
                        matvec(enc, sg_state, sg_mat, hidden, 1, h_at, wts_at + *n_used);
                        ffn_dispatches += 1;
                    }
                    bar_c(enc, b'f');
                    if !probe_skip("experts") {
                        match gu_dual {
                            // gate + up in ONE dual-output indexed dispatch.
                            Some(st) => {
                                enc.set_compute_pipeline_state(st);
                                enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
                                enc.set_buffer(1, Some(y), e(h_at));
                                enc.set_buffer(2, Some(y), e(gate_at));
                                let a = hidden as u32;
                                let b = *expert_ffn as u32;
                                enc.set_bytes(3, 4, &a as *const u32 as *const _);
                                enc.set_bytes(4, 4, &b as *const u32 as *const _);
                                enc.set_bytes(5, 8, &mats[0].w_off as *const u64 as *const _);
                                enc.set_buffer(6, Some(y), e(ids_at));
                                let idx = GpuIdxArgs {
                                    stride: strides[0],
                                    slots: *n_used as u32,
                                    x_stride: 0,
                                    ids_stride: 0,
                                    x_row_stride: 0,
                                    y_row_stride: 0,
                                    n_rows: 0,
                                };
                                enc.set_bytes(
                                    7,
                                    std::mem::size_of::<GpuIdxArgs>() as u64,
                                    &idx as *const GpuIdxArgs as *const _,
                                );
                                enc.set_buffer(11, Some(&gpu.chunks[mats[1].chunk].buf), 0);
                                enc.set_bytes(12, 8, &mats[1].w_off as *const u64 as *const _);
                                enc.set_buffer(13, Some(y), e(up_at));
                                let lpr = lanes_per_row(mats[0].ty, hidden) as u64;
                                enc.dispatch_thread_groups(
                                    MTLSize::new(
                                        ((*expert_ffn * *n_used * 2) as u64 * lpr).div_ceil(128),
                                        1,
                                        1,
                                    ),
                                    MTLSize::new(128, 1, 1),
                                );
                                ffn_dispatches += 1;
                            }
                            None => {
                                matvec_idx(
                                    enc,
                                    &states[0],
                                    &mats[0],
                                    hidden,
                                    *expert_ffn,
                                    h_at,
                                    gate_at,
                                    strides[0],
                                    *n_used,
                                    0,
                                );
                                matvec_idx(
                                    enc,
                                    &states[1],
                                    &mats[1],
                                    hidden,
                                    *expert_ffn,
                                    h_at,
                                    up_at,
                                    strides[1],
                                    *n_used,
                                    0,
                                );
                                ffn_dispatches += 2;
                            }
                        }
                        // The shared expert is a plain matvec into the extra slot.
                        if let Some((smats, sstates)) = shared {
                            matvec(
                                enc,
                                &sstates[0],
                                &smats[0],
                                hidden,
                                *expert_ffn,
                                h_at,
                                gate_at + *n_used * *expert_ffn,
                            );
                            matvec(
                                enc,
                                &sstates[1],
                                &smats[1],
                                hidden,
                                *expert_ffn,
                                h_at,
                                up_at + *n_used * *expert_ffn,
                            );
                            ffn_dispatches += 2;
                        }
                        split_here!("gate_up");
                        bar_c(enc, b'f');
                        if !*sw_fused {
                            enc.set_compute_pipeline_state(&swiglu_state);
                            enc.set_buffer(0, Some(y), e(gate_at));
                            enc.set_buffer(1, Some(y), e(up_at));
                            let n32 = (n_slots * *expert_ffn) as u32;
                            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            ffn_dispatches += 1;
                            split_here!("swiglu");
                            bar_c(enc, b'f');
                        } else {
                            // The down kernel reads raw gate (x) and raw up
                            // (buffer 8) and applies swiglu on load.
                            enc.set_buffer(8, Some(y), e(up_at));
                        }
                        matvec_idx(
                            enc,
                            &states[2],
                            &mats[2],
                            *expert_ffn,
                            hidden,
                            gate_at,
                            downo_at,
                            strides[2],
                            *n_used,
                            *expert_ffn,
                        );
                        ffn_dispatches += 1;
                        if let Some((smats, sstates)) = shared {
                            // The shared down reads its own raw up slot.
                            if *sw_fused {
                                enc.set_buffer(8, Some(y), e(up_at + *n_used * *expert_ffn));
                            }
                            matvec(
                                enc,
                                &sstates[2],
                                &smats[2],
                                *expert_ffn,
                                hidden,
                                gate_at + *n_used * *expert_ffn,
                                downo_at + *n_used * hidden,
                            );
                            ffn_dispatches += 1;
                        }
                    }
                    split_here!("down");
                    bar_c(enc, b'f');
                    if cfuse() {
                        // No standalone combine: it folds into the next
                        // residual norm (combine_resnorm), which the barrier
                        // above already orders against the down writes.
                        pending_combine = Some((n_slots as u32, sig_last));
                    } else {
                        combine(enc, n_slots as u32, sig_last);
                        ffn_dispatches += 1;
                    }
                    split_here!("combine");
                    dispatched += ffn_dispatches;
                }
            }
            // The fused combine+norm is ordered by the down stage's barrier;
            // only the plain paths still need the layer-end drain.
            if pending_combine.is_none() {
                bar(enc);
            }
            dispatched += 8;
        }
        // Final norm and the output projection, still in the same buffer; a
        // pending combine from the last MoE layer folds into the final norm.
        match pending_combine.take() {
            Some((slots, sig_last)) => combine_resnorm(enc, &out_norm, slots, sig_last),
            None => resnorm(enc, &out_norm),
        }
        split_here!("resnorm");
        bar(enc);
        if !probe_skip("head") {
            matvec(
                enc,
                &out_state,
                &out_mat,
                hidden,
                vocab,
                h_at,
                out_logits_at,
            );
        }
        split_here!("head");
        if req.argmax {
            // Greedy: the winner reduces on the GPU; the readback below is
            // one 32-bit word instead of the whole vocabulary.
            bar(enc);
            enc.set_compute_pipeline_state(&argmax_state);
            enc.set_buffer(0, Some(y), e(out_logits_at));
            let v32 = vocab as u32;
            enc.set_bytes(1, 4, &v32 as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(amax_at));
            enc.dispatch_thread_groups(MTLSize::new(64, 1, 1), MTLSize::new(256, 1, 1));
            bar(enc);
            enc.set_compute_pipeline_state(&argmax_final_state);
            enc.set_buffer(0, Some(y), e(amax_at));
            let np = 64u32;
            enc.set_bytes(1, 4, &np as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(amax_at + 64 * 2));
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            split_here!("argmax");
        }
        enc.end_encoding();
        if let (Some(counter), Some(resolved)) = (split_counter.as_ref(), split_resolved.as_ref()) {
            let blit = cmd.new_blit_command_encoder();
            blit.resolve_counters(counter, metal::NSRange::new(0, split_index), resolved, 0);
            blit.end_encoding();
        }
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        if let Some(resolved) = split_resolved.as_ref() {
            let mut cpu_end = 0u64;
            let mut gpu_end = 0u64;
            gpu.device.sample_timestamps(&mut cpu_end, &mut gpu_end);
            let samples = unsafe {
                std::slice::from_raw_parts(resolved.contents() as *const u64, split_index as usize)
            };
            let scale = if gpu_end > split_gpu_start {
                (cpu_end - split_cpu_start) as f64 / (gpu_end - split_gpu_start) as f64
            } else {
                1.0
            };
            let mut split_acc: Vec<(&'static str, f64)> = Vec::new();
            for (stage, label) in split_labels.iter().enumerate() {
                let begin = samples[stage * 2];
                let end = samples[stage * 2 + 1];
                let ms = end.saturating_sub(begin) as f64 * scale / 1e6;
                match split_acc.iter_mut().find(|(known, _)| known == label) {
                    Some(entry) => entry.1 += ms,
                    None => split_acc.push((*label, ms)),
                }
            }
            let total: f64 = split_acc.iter().map(|entry| entry.1).sum();
            static PRINTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !PRINTED.swap(true, Ordering::Relaxed) {
                eprintln!("decode profile, ms/token (GPU counters, one submit):");
                for (label, ms) in split_acc {
                    eprintln!("  {label:<10} {ms:6.3} ({:4.1}%)", ms / total * 100.0);
                }
            }
        }
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(dispatched + 2, Ordering::Relaxed);

        if req.argmax {
            let next = unsafe {
                ((gpu.y_arena.contents() as *const f32).add(amax_at + 64 * 2) as *const u32).read()
            };
            TokenOut::Argmax(next)
        } else {
            let mut out = vec![0f32; vocab];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (gpu.y_arena.contents() as *const f32).add(out_logits_at),
                    out.as_mut_ptr(),
                    vocab,
                );
            }
            TokenOut::Logits(out)
        }
    });
    Some(out)
}

/// The speculative multi-token verify: `m` tokens through the already
/// resolved layers in ONE command buffer with ONE wait. Projections batch
/// all m rows (TILE = m matvec pipelines, one weight pass); attention and
/// MoE run per token; the GDN recurrence takes the batch kernels, with
/// rollback slots written when armed. The matvec f32 dot per output row is
/// the decode path's numerics, so the arbitration rows stay bit-compatible
/// with plain greedy decode - unlike the mm (half-staged) prefill path,
/// whose drift flips near-tie argmaxes (measured: stream divergence).
fn encode_verify_tokens(
    req: &TokenReq,
    gpu: &mut Gpu,
    layers: &[LayerRefs],
    _sigmoid_gating: bool,
) -> Option<TokenOut> {
    let m = req.m;
    let hidden = req.x.len() / m;
    let hd = req.head_dim;
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let vocab = req.output.2;
    let rp = req.rot_dim / 2;

    let out_norm = resolve_norm(gpu, req.output_norm, hidden)?;
    let out_mat = resolve(gpu, req.output.0, req.output.1)?;

    let resnorm_state = gpu.pipelines[&("residual_norm", 1, 1)].to_owned();
    let norm_rows_state = gpu.pipelines[&("norm_rows", 1, 1)].to_owned();
    let combine_rows_state = gpu.pipelines[&("moe_combine_rows", 1, 1)].to_owned();
    let resnorm_router_rows_state = gpu.pipelines[&("resnorm_router_rows", 1, 1)].to_owned();
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();
    let attend_state = gpu.pipelines[&(attend_kernel(), 1, 1)].to_owned();
    let qk_prep_state = gpu.pipelines[&("qk_prep", 1, 1)].to_owned();
    let qk_prep256_state = gpu.pipelines[&("qk_prep256", 1, 1)].to_owned();
    let attend256_state = gpu.pipelines[&("attend_s32_256", 1, 1)].to_owned();
    let qk_prep_batch256_state = gpu.pipelines[&("qk_prep_batch256", 1, 1)].to_owned();
    let attend_rows256_state = gpu.pipelines[&("attend_rows256", 1, 1)].to_owned();
    let gate_mul_state = gpu.pipelines[&("attn_gate_mul", 1, 1)].to_owned();
    let conv_state = gpu.pipelines[&("gdn_conv_batch", 1, 1)].to_owned();
    let step_state = gpu.pipelines[&("gdn_step_batch", 1, 1)].to_owned();
    let norm_out_state = gpu.pipelines[&("gdn_out_norm_batch", 1, 1)].to_owned();
    let argmax_state = gpu.pipelines[&("argmax_f32", 1, 1)].to_owned();
    let argmax_final_state = gpu.pipelines[&("argmax_final", 1, 1)].to_owned();

    // TILE=m pipelines for the batched projections (created up front: the
    // registry needs &mut Gpu).
    let tstate = |gpu: &mut Gpu, mat: &MatRef, n_in: usize| -> Option<ComputePipelineState> {
        let lpr = lanes_per_row(mat.ty, n_in);
        Some(gpu.pipeline(mat.kernel, m, lpr)?.to_owned())
    };
    let mut t_attn: Vec<Option<[ComputePipelineState; 4]>> = Vec::with_capacity(layers.len());
    let mut t_gdn: Vec<Option<[ComputePipelineState; 5]>> = Vec::with_capacity(layers.len());
    // TILE=m pipelines for the shared expert / its gate logit (MoE layers).
    let mut t_sgate: Vec<Option<ComputePipelineState>> = Vec::with_capacity(layers.len());
    let mut t_shared: Vec<Option<[ComputePipelineState; 3]>> = Vec::with_capacity(layers.len());
    for l in layers {
        match &l.ffn {
            FfnRefs::Moe {
                shared,
                shared_gate,
                sw_fused,
                expert_ffn,
                ..
            } => {
                t_sgate.push(match shared_gate {
                    Some((m0, _)) => Some(tstate(gpu, m0, hidden)?),
                    None => None,
                });
                t_shared.push(match shared {
                    Some((smats, _)) => Some([
                        tstate(gpu, &smats[0], hidden)?,
                        tstate(gpu, &smats[1], hidden)?,
                        {
                            let lpr = lanes_per_row(smats[2].ty, *expert_ffn);
                            gpu.pipeline_full(smats[2].kernel, m, lpr, false, *sw_fused)?
                                .to_owned()
                        },
                    ]),
                    None => None,
                });
            }
            _ => {
                t_sgate.push(None);
                t_shared.push(None);
            }
        }
    }
    // ROWS expert pipelines: one dispatch per matrix covering every row's
    // expert slots. Only the kernels whose source carries the ROWS mapping
    // qualify (the per-output math there is the single-row kernel's
    // verbatim); anything else falls back to the per-row dispatches below.
    let mut t_moe_rows: Vec<Option<[ComputePipelineState; 3]>> = Vec::with_capacity(layers.len());
    for l in layers {
        t_moe_rows.push(match &l.ffn {
            FfnRefs::Moe {
                mats,
                expert_ffn,
                sw_fused,
                ..
            } => {
                let n_outs = [*expert_ffn, *expert_ffn, hidden];
                let mut v: Vec<ComputePipelineState> = Vec::with_capacity(3);
                for (i, (mat, n_in)) in mats.iter().zip([hidden, hidden, *expert_ffn]).enumerate() {
                    // The same _mv fallback decode applies per matrix.
                    let kernel = match mat.kernel {
                        "matvec_q2_k_mv" if n_outs[i] % 4 != 0 => "matvec_q2_k",
                        "matvec_q3_k_mv" if n_outs[i] % 2 != 0 => "matvec_q3_k",
                        "matvec_q4_k_mv" if n_outs[i] % 2 != 0 => "matvec_q4_k",
                        "matvec_q6_k_mv" if n_outs[i] % 2 != 0 => "matvec_q6_k",
                        k => k,
                    };
                    if !matches!(kernel, "matvec_q4_k_mv" | "matvec_q5_k" | "matvec_q6_k") {
                        v.clear();
                        break;
                    }
                    let lpr = lanes_per_row(mat.ty, n_in);
                    match gpu.pipeline_rows(kernel, lpr, i == 2 && *sw_fused) {
                        Some(p) => v.push(p.to_owned()),
                        None => {
                            v.clear();
                            break;
                        }
                    }
                }
                if v.len() == 3 {
                    Some([v[0].clone(), v[1].clone(), v[2].clone()])
                } else {
                    None
                }
            }
            _ => None,
        });
    }
    for l in layers {
        t_attn.push(match &l.mats {
            Some(mats) => Some([
                tstate(gpu, &mats[0], hidden)?,
                tstate(gpu, &mats[1], hidden)?,
                tstate(gpu, &mats[2], hidden)?,
                tstate(gpu, &mats[3], q_dim)?,
            ]),
            None => None,
        });
        t_gdn.push(match &l.gdn {
            Some(g) => Some([
                tstate(gpu, &g.mats[0], hidden)?,
                tstate(gpu, &g.mats[1], hidden)?,
                tstate(gpu, &g.mats[2], hidden)?,
                tstate(gpu, &g.mats[3], hidden)?,
                tstate(gpu, &g.mats[4], g.value_dim)?,
            ]),
            None => None,
        });
    }
    let t_out = tstate(gpu, &out_mat, hidden)?;

    // Arena layout, element offsets over f32 (ids are u32-in-f32 slots).
    let align = |n: usize| (n + 63) & !63;
    let wq_max = req
        .layers
        .iter()
        .filter(|l| l.gdn.is_none())
        .map(|l| l.wq.2)
        .max()
        .unwrap_or(q_dim)
        .max(q_dim);
    let mut max_ffn = 1usize;
    let mut max_slots = 1usize;
    let mut max_expert = 1usize;
    let (mut max_channels, mut max_value_dim, mut max_heads_v) = (0usize, 0usize, 1usize);
    for l in layers {
        if let Some(g) = &l.gdn {
            max_channels = max_channels.max(g.channels);
            max_value_dim = max_value_dim.max(g.value_dim);
            max_heads_v = max_heads_v.max(g.heads_v as usize);
        }
        if let FfnRefs::Moe {
            expert_ffn,
            n_used,
            shared,
            n_expert,
            ..
        } = &l.ffn
        {
            max_ffn = max_ffn.max(*expert_ffn);
            max_slots = max_slots.max(*n_used + shared.is_some() as usize);
            max_expert = max_expert.max(*n_expert);
        }
    }
    // The batched MoE below packs the per-row expert scratch contiguously
    // and the row-batched kernels index it by the global maxima, so every
    // MoE layer must share one expert shape; anything else declines (the
    // caller falls back to the batch path). The router kernels are
    // single-threadgroup per row: n_expert <= 256.
    for l in layers {
        if let FfnRefs::Moe {
            expert_ffn,
            n_used,
            shared,
            n_expert,
            ..
        } = &l.ffn
        {
            if *n_used + shared.is_some() as usize != max_slots
                || *expert_ffn != max_ffn
                || *n_expert > 256
            {
                return None;
            }
        }
    }
    let x_at = 0usize;
    let delta_at = x_at + align(m * hidden);
    let h_at = delta_at + align(m * hidden);
    let q_at = h_at + align(m * hidden);
    let qn_at = q_at + align(m * wq_max);
    let qgate_at = qn_at + align(m * q_dim);
    let k_at = qgate_at + align(m * q_dim);
    let v_at = k_at + align(m * kv);
    let attn_at = v_at + align(m * kv);
    let gqkv_at = attn_at + align(m * q_dim);
    let gqkc_at = gqkv_at + align(m * max_channels);
    let gz_at = gqkc_at + align(m * max_channels);
    let gab_at = gz_at + align(m * max_value_dim);
    let gy_at = gab_at + align(2 * m * max_heads_v);
    let logits_at = gy_at + align(m * max_value_dim);
    let ids_at = logits_at + align(m * max_expert);
    let wts_at = ids_at + align(m * max_slots);
    // The expert FFN scratch is per-row: every token's gate/up/down blocks
    // live side by side, so the whole MoE half dispatches stage-by-stage
    // (one barrier per stage) instead of token-by-token.
    let gate_at = wts_at + align(m * max_slots);
    let up_at = gate_at + align(m * max_slots * max_ffn);
    let downo_at = up_at + align(m * max_slots * max_ffn);
    // The shared expert runs as plain TILE=m matvecs over ALL rows (one
    // weight pass), so it gets its own contiguous regions instead of a slot
    // in the per-row blocks.
    let sgate_at = downo_at + align(m * max_slots * hidden);
    let sg_gate_at = sgate_at + align(m);
    let sg_up_at = sg_gate_at + align(m * max_ffn);
    let sg_down_at = sg_up_at + align(m * max_ffn);
    let out_logits_at = sg_down_at + align(m * hidden);
    let flag_at = out_logits_at + align(m * vocab);
    let ctr_at = flag_at + align(1);
    let amax_at = ctr_at + align(1);
    let total = amax_at + align(m * (64 * 2 + 1));
    gpu.ensure_arenas(4096, total * 4);

    unsafe {
        let yp = gpu.y_arena.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(req.x.as_ptr(), yp.add(x_at), req.x.len());
        std::ptr::write_bytes(yp.add(delta_at), 0, m * hidden);
        std::ptr::write_bytes(yp.add(flag_at), 0, 1);
        std::ptr::write_bytes(yp.add(ctr_at), 0, 1);
        for l in layers {
            if let FfnRefs::Moe {
                n_used,
                shared: Some(_),
                shared_gate: None,
                ..
            } = &l.ffn
            {
                for i in 0..m {
                    *yp.add(wts_at + i * max_slots + *n_used) = 1.0;
                }
            }
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let mut cmd = gpu.queue.new_command_buffer();
        let mut enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let y = &gpu.y_arena;
        let bar = |enc: &metal::ComputeCommandEncoderRef| enc.memory_barrier_with_resources(&[y]);
        let e = |off: usize| (off * 4) as u64;

        // One plain TILE=m matvec over all m rows: one weight pass.
        let matvec = |enc: &metal::ComputeCommandEncoderRef,
                      state: &ComputePipelineState,
                      mat: &MatRef,
                      n_in: usize,
                      n_out: usize,
                      x_off: usize,
                      y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            let lpr = lanes_per_row(mat.ty, n_in) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64 * lpr).div_ceil(128), 1, 1),
                MTLSize::new(128, 1, 1),
            );
        };
        // Single-row matvec for the per-token expert/shared projections.
        let resnorm = |enc: &metal::ComputeCommandEncoderRef, norm: &NormRef, i: usize| {
            enc.set_compute_pipeline_state(&resnorm_state);
            enc.set_buffer(0, Some(y), e(x_at + i * hidden));
            enc.set_buffer(1, Some(y), e(delta_at + i * hidden));
            enc.set_buffer(2, Some(y), e(h_at + i * hidden));
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        };
        let matvec_row = |enc: &metal::ComputeCommandEncoderRef,
                          state: &ComputePipelineState,
                          mat: &MatRef,
                          n_in: usize,
                          n_out: usize,
                          x_off: usize,
                          y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(y), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            let lpr = lanes_per_row(mat.ty, n_in) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64 * lpr).div_ceil(128), 1, 1),
                MTLSize::new(128, 1, 1),
            );
        };

        let mut dispatched = 0u64;
        let vdbg = std::env::var_os("ALLPAKA_VERIFY_DEBUG").is_some();
        for (li, (l, refs)) in req.layers.iter().zip(layers).enumerate() {
            if vdbg {
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let yp = gpu.y_arena.contents() as *const f32;
                unsafe {
                    let x0 = std::slice::from_raw_parts(yp.add(x_at), hidden);
                    let d0 = std::slice::from_raw_parts(yp.add(delta_at), hidden);
                    let sx: f64 = x0.iter().map(|&v| v as f64).sum();
                    let sd: f64 = d0.iter().map(|&v| v as f64).sum();
                    eprintln!("verify layer {li} in: sum_x={sx:.6} sum_delta={sd:.6}");
                    if li == 1 {
                        let wts = std::slice::from_raw_parts(yp.add(wts_at), 16);
                        eprintln!("  wts[..]: {:?}", &wts[..9.min(16)]);
                        let do0 = std::slice::from_raw_parts(yp.add(downo_at), 9 * hidden);
                        for srow in 0..9 {
                            let r = &do0[srow * hidden..(srow + 1) * hidden];
                            let s: f64 = r.iter().map(|&v| v as f64).sum();
                            eprintln!("  downo[{srow}] sum={s:.6} head={:.6}", r[0]);
                        }
                        let h0 = std::slice::from_raw_parts(yp.add(h_at), hidden);
                        let hs: f64 = h0.iter().map(|&v| v as f64).sum();
                        eprintln!("  h_ffn_in sum={hs:.6}");
                        std::fs::write("/tmp/h0.bin", unsafe {
                            std::slice::from_raw_parts(h0.as_ptr() as *const u8, hidden * 4)
                        })
                        .ok();
                        eprintln!(
                            "  h_ffn_in head=[{:.6} {:.6} {:.6}] tail=[{:.6} {:.6} {:.6}]",
                            h0[0], h0[1], h0[2], h0[2045], h0[2046], h0[2047]
                        );
                        let lg = std::slice::from_raw_parts(yp.add(logits_at), 256);
                        let ls: f64 = lg.iter().map(|&v| v as f64).sum();
                        eprintln!("  router logits sum={ls:.6}");
                        eprintln!(
                            "  router logits head=[{:.6} {:.6} {:.6} {:.6}]",
                            lg[0], lg[1], lg[2], lg[3]
                        );
                    }
                }
                cmd = gpu.queue.new_command_buffer();
                enc = cmd
                    .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
            }
            // h = rmsnorm(x + delta) * attn_norm over ALL rows in ONE
            // dispatch (norm_rows is residual_norm's body with the row from
            // the grid, bit-identical per row).
            enc.set_compute_pipeline_state(&norm_rows_state);
            enc.set_buffer(0, Some(y), e(x_at));
            enc.set_buffer(1, Some(y), e(delta_at));
            enc.set_buffer(2, Some(y), e(h_at));
            enc.set_buffer(
                3,
                Some(&gpu.chunks[refs.attn_norm.chunk].buf),
                refs.attn_norm.off,
            );
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            let one = 1u32;
            enc.set_bytes(6, 4, &one as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
            dispatched += 1;
            bar(enc);

            if let Some(g) = &refs.gdn {
                // Debug kill switch, mirrors the CPU path's gdn_forward:
                // zero the branch output (into delta) to isolate bugs.
                if std::env::var_os("ALLPAKA_GDN_ZERO").is_some() {
                    unsafe {
                        std::ptr::write_bytes(
                            (gpu.y_arena.contents() as *mut f32).add(delta_at),
                            0,
                            m * hidden,
                        );
                    }
                } else {
                    let tg = &l.gdn.as_ref().expect("gdn refs without gdn layer");
                    let ts = t_gdn[li].as_ref().expect("gdn tile states");
                    let ssm_buf = &req.ssm.expect("gdn layer without ssm region").buf;
                    // The four projections over all m rows, two weight passes.
                    matvec(enc, &ts[0], &g.mats[0], hidden, g.channels, h_at, gqkv_at);
                    matvec(enc, &ts[1], &g.mats[1], hidden, g.value_dim, h_at, gz_at);
                    matvec(
                        enc,
                        &ts[2],
                        &g.mats[2],
                        hidden,
                        g.heads_v as usize,
                        h_at,
                        gab_at,
                    );
                    matvec(
                        enc,
                        &ts[3],
                        &g.mats[3],
                        hidden,
                        g.heads_v as usize,
                        h_at,
                        gab_at + m * g.heads_v as usize,
                    );
                    bar(enc);
                    // Depthwise conv over the chunk (with rollback slots), then
                    // the session window commit.
                    enc.set_compute_pipeline_state(&conv_state);
                    enc.set_buffer(0, Some(y), e(gqkv_at));
                    enc.set_buffer(1, Some(y), e(gqkc_at));
                    enc.set_buffer(2, Some(ssm_buf), 0);
                    enc.set_buffer(3, Some(&gpu.chunks[g.conv1d.chunk].buf), g.conv1d.off);
                    let cargs = GdnConvBatchArgs {
                        channels: g.channels as u32,
                        d_conv: g.d_conv,
                        m: m as u32,
                        pad0: 0,
                        conv_off: g.conv_off,
                    };
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<GdnConvBatchArgs>() as u64,
                        &cargs as *const GdnConvBatchArgs as *const _,
                    );
                    let (slots_buf, slot_total) = match req.ssm_slots {
                        Some((r, total)) => (r, total as u32),
                        None => (req.ssm.expect("ssm region"), 0u32),
                    };
                    enc.set_buffer(5, Some(&slots_buf.buf), 0);
                    enc.set_bytes(6, 4, &slot_total as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((g.channels as u64).div_ceil(256), m as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // The last d_conv-1 raw rows become the next chunk's window.
                    enc.set_compute_pipeline_state(&gpu.pipelines[&("copy_f32", 1, 1)]);
                    let window_rows = g.d_conv as usize - 1;
                    let window_src = if m >= window_rows {
                        gqkv_at + (m - window_rows) * g.channels
                    } else {
                        let old = (window_rows - m) * g.channels;
                        enc.set_buffer(
                            0,
                            Some(ssm_buf),
                            ((g.conv_off + m as u64 * g.channels as u64) * 4) as u64,
                        );
                        enc.set_buffer(1, Some(y), e(gate_at));
                        let old32 = old as u32;
                        enc.set_bytes(2, 4, &old32 as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((old as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        enc.set_buffer(0, Some(y), e(gqkv_at));
                        enc.set_buffer(1, Some(y), e(gate_at + old));
                        let new32 = (m * g.channels) as u32;
                        enc.set_bytes(2, 4, &new32 as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((new32 as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        enc.memory_barrier_with_resources(&[y, ssm_buf]);
                        gate_at
                    };
                    enc.set_buffer(0, Some(y), e(window_src));
                    enc.set_buffer(1, Some(ssm_buf), (g.conv_off * 4) as u64);
                    let nwin = (window_rows * g.channels) as u32;
                    enc.set_bytes(2, 4, &nwin as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((nwin as u64).div_ceil(256), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // The recurrence over the chunk, slots per row when armed.
                    enc.set_compute_pipeline_state(&step_state);
                    enc.set_buffer(0, Some(ssm_buf), 0);
                    enc.set_buffer(1, Some(y), e(gqkc_at));
                    enc.set_buffer(2, Some(y), e(gab_at));
                    enc.set_buffer(3, Some(y), e(gab_at + m * g.heads_v as usize));
                    let sargs = GdnStepBatchArgs {
                        heads_k: g.heads_k,
                        heads_v: g.heads_v,
                        d: g.d,
                        key_dim: g.heads_k * g.d,
                        m: m as u32,
                        eps: req.eps,
                        state_off: g.state_off,
                    };
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<GdnStepBatchArgs>() as u64,
                        &sargs as *const GdnStepBatchArgs as *const _,
                    );
                    enc.set_bytes(5, (tg.a.len() * 4) as u64, tg.a.as_ptr() as *const _);
                    enc.set_bytes(6, (tg.dt.len() * 4) as u64, tg.dt.as_ptr() as *const _);
                    enc.set_buffer(7, Some(y), e(gy_at));
                    enc.set_buffer(8, Some(&slots_buf.buf), 0);
                    enc.set_bytes(9, 4, &slot_total as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((g.d / 4) as u64, g.heads_v as u64, 1),
                        MTLSize::new(32, 4, 1),
                    );
                    enc.memory_barrier_with_resources(&[y, ssm_buf]);
                    // Gated rmsnorm over the head outputs, times silu(z).
                    enc.set_compute_pipeline_state(&norm_out_state);
                    enc.set_buffer(0, Some(y), e(gy_at));
                    enc.set_buffer(1, Some(y), e(gz_at));
                    enc.set_bytes(
                        2,
                        (tg.ssm_norm.len() * 4) as u64,
                        tg.ssm_norm.as_ptr() as *const _,
                    );
                    let hv = g.heads_v;
                    let dd = g.d;
                    enc.set_bytes(3, 4, &hv as *const u32 as *const _);
                    enc.set_bytes(4, 4, &dd as *const u32 as *const _);
                    enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(g.heads_v as u64, m as u64, 1),
                        MTLSize::new(g.d as u64, 1, 1),
                    );
                    bar(enc);
                    matvec(
                        enc,
                        &ts[4],
                        &g.mats[4],
                        g.value_dim,
                        hidden,
                        gy_at,
                        delta_at,
                    );
                    bar(enc);
                    dispatched += 11;
                }
            } else {
                let amats = refs.mats.as_ref().expect("attention layer without mats");
                let ts = t_attn[li].as_ref().expect("attention tile states");
                let wq_out = if l.gate_in_q { 2 * q_dim } else { q_dim };
                matvec(enc, &ts[0], &amats[0], hidden, wq_out, h_at, q_at);
                matvec(enc, &ts[1], &amats[1], hidden, kv, h_at, k_at);
                matvec(enc, &ts[2], &amats[2], hidden, kv, h_at, v_at);
                bar(enc);
                if hd == 256 && wq_max == wq_out {
                    // Batched qk_prep: one threadgroup per (head, row); the
                    // per-row math is qk_prep256's verbatim (prefill's
                    // kernel), so every row is bit-identical to decode.
                    enc.set_compute_pipeline_state(&qk_prep_batch256_state);
                    enc.set_buffer(0, Some(y), e(q_at));
                    enc.set_buffer(1, Some(y), e(k_at));
                    enc.set_buffer(2, Some(y), e(v_at));
                    enc.set_buffer(3, Some(&req.cache.buf), 0);
                    let args = QkPrepBatch256Args {
                        n_heads: req.n_heads as u32,
                        n_kv_heads: req.n_kv_heads as u32,
                        kv_dim: req.kv_dim as u32,
                        eps: req.eps,
                        base: req.pos as u32,
                        has_qk_norm: l.q_norm.is_some() as u32,
                        rot_dim: req.rot_dim as u32,
                        gate_in_q: l.gate_in_q as u32,
                        k_base: l.k_off as u64,
                        v_base: l.v_off as u64,
                    };
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<QkPrepBatch256Args>() as u64,
                        &args as *const QkPrepBatch256Args as *const _,
                    );
                    let ones = [1.0f32; 256];
                    let qw = l.q_norm.unwrap_or(&ones);
                    let kw = l.k_norm.unwrap_or(&ones);
                    enc.set_bytes(5, (256 * 4) as u64, qw.as_ptr() as *const _);
                    enc.set_bytes(6, (256 * 4) as u64, kw.as_ptr() as *const _);
                    enc.set_bytes(7, (rp * 8) as u64, req.rope.as_ptr() as *const _);
                    enc.set_buffer(8, Some(y), e(qn_at));
                    enc.set_buffer(9, Some(y), e(qgate_at));
                    enc.dispatch_thread_groups(
                        MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, m as u64, 1),
                        MTLSize::new(32, 1, 1),
                    );
                } else {
                    for i in 0..m {
                        let pos = req.pos + i;
                        if hd == 256 {
                            enc.set_compute_pipeline_state(&qk_prep256_state);
                            enc.set_buffer(0, Some(y), e(q_at + i * wq_max));
                            enc.set_buffer(1, Some(y), e(k_at + i * kv));
                            enc.set_buffer(2, Some(y), e(v_at + i * kv));
                            enc.set_buffer(3, Some(&req.cache.buf), 0);
                            let args = QkPrep256Args {
                                n_heads: req.n_heads as u32,
                                n_kv_heads: req.n_kv_heads as u32,
                                kv_dim: req.kv_dim as u32,
                                eps: req.eps,
                                pos: pos as u32,
                                has_qk_norm: l.q_norm.is_some() as u32,
                                rot_dim: req.rot_dim as u32,
                                gate_in_q: l.gate_in_q as u32,
                                k_base: l.k_off as u64,
                                v_base: l.v_off as u64,
                            };
                            enc.set_bytes(
                                4,
                                std::mem::size_of::<QkPrep256Args>() as u64,
                                &args as *const QkPrep256Args as *const _,
                            );
                            let ones = [1.0f32; 256];
                            let qw = l.q_norm.unwrap_or(&ones);
                            let kw = l.k_norm.unwrap_or(&ones);
                            enc.set_bytes(5, (256 * 4) as u64, qw.as_ptr() as *const _);
                            enc.set_bytes(6, (256 * 4) as u64, kw.as_ptr() as *const _);
                            enc.set_bytes(
                                7,
                                (rp * 8) as u64,
                                req.rope[i * rp..].as_ptr() as *const _,
                            );
                            enc.set_buffer(8, Some(y), e(qn_at + i * q_dim));
                            enc.set_buffer(9, Some(y), e(qgate_at + i * q_dim));
                            enc.dispatch_thread_groups(
                                MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                                MTLSize::new(32, 1, 1),
                            );
                        } else {
                            enc.set_compute_pipeline_state(&qk_prep_state);
                            enc.set_buffer(0, Some(y), e(q_at + i * wq_max));
                            enc.set_buffer(1, Some(y), e(k_at + i * kv));
                            enc.set_buffer(2, Some(y), e(v_at + i * kv));
                            enc.set_buffer(3, Some(&req.cache.buf), 0);
                            let args = QkPrepArgs {
                                n_heads: req.n_heads as u32,
                                n_kv_heads: req.n_kv_heads as u32,
                                head_dim: hd as u32,
                                kv_dim: req.kv_dim as u32,
                                eps: req.eps,
                                pos: pos as u32,
                                has_qk_norm: l.q_norm.is_some() as u32,
                                rot_dim: req.rot_dim as u32,
                                k_base: l.k_off as u64,
                                v_base: l.v_off as u64,
                                has_bias: refs.q_bias.is_some() as u32,
                                pad2: 0,
                            };
                            enc.set_bytes(
                                4,
                                std::mem::size_of::<QkPrepArgs>() as u64,
                                &args as *const QkPrepArgs as *const _,
                            );
                            let ones = [1.0f32; ATTN_HEAD_DIM];
                            let qw = l.q_norm.unwrap_or(&ones);
                            let kw = l.k_norm.unwrap_or(&ones);
                            enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
                            enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
                            enc.set_bytes(
                                7,
                                (rp * 8) as u64,
                                req.rope[i * rp..].as_ptr() as *const _,
                            );
                            let dummy = &gpu.chunks[refs.attn_norm.chunk].buf;
                            for (index, b) in
                                [(8u64, &refs.q_bias), (9, &refs.k_bias), (10, &refs.v_bias)]
                            {
                                match b {
                                    Some(r) => {
                                        enc.set_buffer(index, Some(&gpu.chunks[r.chunk].buf), r.off)
                                    }
                                    None => enc.set_buffer(index, Some(dummy), 0),
                                }
                            }
                            enc.dispatch_thread_groups(
                                MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
                                MTLSize::new(32, 1, 1),
                            );
                        }
                    }
                }
                enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
                if hd == 256 && wq_max == wq_out {
                    // Batched attend: one threadgroup per (head, row), row r
                    // covers base + r + 1 positions; attend_rows256 is
                    // attend_s32_256's body with the row from grid.y.
                    enc.set_compute_pipeline_state(&attend_rows256_state);
                    enc.set_buffer(0, Some(&req.cache.buf), 0);
                    enc.set_buffer(1, Some(&req.cache.buf), 0);
                    enc.set_buffer(2, Some(y), e(qn_at));
                    enc.set_buffer(3, Some(y), e(attn_at));
                    for (index, value) in [
                        (4u64, req.kv_dim as u32),
                        (5, hd as u32),
                        (6, req.pos as u32),
                        (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
                    ] {
                        enc.set_bytes(index, 4, &value as *const u32 as *const _);
                    }
                    enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                    for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                        enc.set_bytes(index, 8, &value as *const u64 as *const _);
                    }
                    let heads32 = req.n_heads as u32;
                    enc.set_bytes(11, 4, &heads32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(req.n_heads as u64, m as u64, 1),
                        MTLSize::new(512, 1, 1),
                    );
                } else {
                    for i in 0..m {
                        let n_pos = (req.pos + i + 1) as u32;
                        enc.set_buffer(0, Some(&req.cache.buf), 0);
                        enc.set_buffer(1, Some(&req.cache.buf), 0);
                        let q_off = if hd == 256 {
                            qn_at + i * q_dim
                        } else {
                            q_at + i * wq_max
                        };
                        enc.set_buffer(2, Some(y), e(q_off));
                        enc.set_buffer(3, Some(y), e(attn_at + i * q_dim));
                        for (index, value) in [
                            (4u64, req.kv_dim as u32),
                            (5, hd as u32),
                            (6, n_pos),
                            (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
                        ] {
                            enc.set_bytes(index, 4, &value as *const u32 as *const _);
                        }
                        enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
                        for (index, value) in [(9u64, l.k_off as u64), (10, l.v_off as u64)] {
                            enc.set_bytes(index, 8, &value as *const u64 as *const _);
                        }
                        if hd == 256 {
                            enc.set_compute_pipeline_state(&attend256_state);
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(512, 1, 1),
                            );
                        } else {
                            enc.set_compute_pipeline_state(&attend_state);
                            enc.dispatch_thread_groups(
                                MTLSize::new(req.n_heads as u64, 1, 1),
                                MTLSize::new(attend_tg(), 1, 1),
                            );
                        }
                    }
                }
                bar(enc);
                if l.gate_in_q && hd == 256 {
                    enc.set_compute_pipeline_state(&gate_mul_state);
                    enc.set_buffer(0, Some(y), e(attn_at));
                    enc.set_buffer(1, Some(y), e(qgate_at));
                    let n32 = (m * q_dim) as u32;
                    enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    bar(enc);
                }
                // Debug kill switch, mirrors the CPU path: zero the
                // attention branch output (into delta).
                if std::env::var_os("ALLPAKA_ATTN_ZERO").is_some() {
                    unsafe {
                        std::ptr::write_bytes(
                            (gpu.y_arena.contents() as *mut f32).add(delta_at),
                            0,
                            m * hidden,
                        );
                    }
                } else {
                    matvec(enc, &ts[3], &amats[3], q_dim, hidden, attn_at, delta_at);
                }
                bar(enc);
                dispatched += 6 + 2 * m as u64;
            }

            // The FFN half: resnorm+router+top-k fused per token
            // (resnorm_router - decode_token's default moe_plain path, so
            // the router logits are bit-identical to plain decode; a
            // matvec_f32+topk pair rounds differently and flips near-tie
            // expert picks). The barrier is load-bearing. Dense layers take
            // the plain resnorm first (resnorm_router is MoE-only).
            if matches!(&refs.ffn, FfnRefs::Dense { .. }) {
                for i in 0..m {
                    resnorm(enc, &refs.ffn_norm, i);
                }
                bar(enc);
            }
            // Debug kill switch, mirrors the CPU path's MOE_ZERO: zero the
            // FFN output (into delta), skipping the whole stage.
            if std::env::var_os("ALLPAKA_MOE_ZERO").is_some() {
                unsafe {
                    std::ptr::write_bytes(
                        (gpu.y_arena.contents() as *mut f32).add(delta_at),
                        0,
                        m * hidden,
                    );
                }
                continue;
            }
            match &refs.ffn {
                FfnRefs::Dense {
                    mats,
                    states,
                    ffn_dim,
                } => {
                    for i in 0..m {
                        matvec_row(
                            enc,
                            &states[0],
                            &mats[0],
                            hidden,
                            *ffn_dim,
                            h_at + i * hidden,
                            gate_at,
                        );
                        matvec_row(
                            enc,
                            &states[1],
                            &mats[1],
                            hidden,
                            *ffn_dim,
                            h_at + i * hidden,
                            up_at,
                        );
                        bar(enc);
                        enc.set_compute_pipeline_state(&swiglu_state);
                        enc.set_buffer(0, Some(y), e(gate_at));
                        enc.set_buffer(1, Some(y), e(up_at));
                        let n = *ffn_dim as u32;
                        enc.set_bytes(2, 4, &n as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((n as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        bar(enc);
                        matvec_row(
                            enc,
                            &states[2],
                            &mats[2],
                            *ffn_dim,
                            hidden,
                            gate_at,
                            delta_at + i * hidden,
                        );
                        bar(enc);
                        dispatched += 4;
                    }
                }
                FfnRefs::Moe {
                    router,
                    router_state: _,
                    router_bias,
                    n_expert,
                    mats,
                    states,
                    strides,
                    expert_ffn,
                    n_used,
                    sw_fused,
                    mega: _,
                    sigmoid,
                    shared,
                    shared_gate,
                    gu_dual: _,
                } => {
                    let n_slots = *n_used + shared.is_some() as usize;
                    // Stage 1: FFN resnorm + router matvec + gating top-k over
                    // ALL rows in ONE dispatch (resnorm_router_rows is
                    // resnorm_router's body with the row from grid.y, so every
                    // row is bit-identical to decode's fused router).
                    enc.set_compute_pipeline_state(&resnorm_router_rows_state);
                    enc.set_buffer(0, Some(y), e(x_at));
                    enc.set_buffer(1, Some(y), e(delta_at));
                    enc.set_buffer(2, Some(y), e(h_at));
                    enc.set_buffer(
                        3,
                        Some(&gpu.chunks[refs.ffn_norm.chunk].buf),
                        refs.ffn_norm.off,
                    );
                    let hd = hidden as u32;
                    enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                    enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
                    enc.set_buffer(6, Some(&gpu.chunks[router.chunk].buf), 0);
                    enc.set_bytes(7, 8, &router.w_off as *const u64 as *const _);
                    enc.set_buffer(8, Some(y), e(ids_at));
                    enc.set_buffer(9, Some(y), e(wts_at));
                    let n = *n_expert as u32;
                    let k = *n_used as u32;
                    enc.set_bytes(10, 4, &n as *const u32 as *const _);
                    enc.set_bytes(11, 4, &k as *const u32 as *const _);
                    match router_bias {
                        Some(rb) => enc.set_buffer(12, Some(&gpu.chunks[rb.chunk].buf), rb.off),
                        None => enc.set_buffer(12, Some(&gpu.chunks[router.chunk].buf), 0),
                    }
                    let sg = *sigmoid as u32;
                    let hb = router_bias.is_some() as u32;
                    enc.set_bytes(13, 4, &sg as *const u32 as *const _);
                    enc.set_bytes(14, 4, &hb as *const u32 as *const _);
                    enc.set_bytes(15, 4, &(max_slots as u32) as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, m as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    bar(enc);
                    // Stage 2: gate/up. Every row is independent (per-row
                    // scratch blocks), so the rows dispatch back to back with
                    // ONE barrier after the whole stage. The shared expert
                    // and its gate logit run as TILE matvecs over ALL rows
                    // (one weight pass each).
                    if let (Some((sg_mat, _)), Some(tsg)) = (shared_gate, t_sgate[li].as_ref()) {
                        matvec(enc, tsg, sg_mat, hidden, 1, h_at, sgate_at);
                    }
                    if let (Some((smats, _)), Some(tsh)) = (shared, t_shared[li].as_ref()) {
                        matvec(
                            enc,
                            &tsh[0],
                            &smats[0],
                            hidden,
                            *expert_ffn,
                            h_at,
                            sg_gate_at,
                        );
                        matvec(enc, &tsh[1], &smats[1], hidden, *expert_ffn, h_at, sg_up_at);
                    }
                    let a = hidden as u32;
                    let b = *expert_ffn as u32;
                    if std::env::var_os("ALLPAKA_ROWS_DEBUG").is_some() && li == 0 {
                        eprintln!("verify rows pipelines: {}", t_moe_rows[li].is_some());
                    }
                    if let Some(rs) = &t_moe_rows[li] {
                        // ROWS: ONE dispatch per matrix covering every row's
                        // expert slots (per-output math identical to the
                        // per-row indexed dispatches).
                        for (mi, y_off) in [gate_at, up_at].into_iter().enumerate() {
                            enc.set_compute_pipeline_state(&rs[mi]);
                            enc.set_buffer(0, Some(&gpu.chunks[mats[mi].chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(h_at));
                            enc.set_buffer(2, Some(y), e(y_off));
                            enc.set_bytes(3, 4, &a as *const u32 as *const _);
                            enc.set_bytes(4, 4, &b as *const u32 as *const _);
                            enc.set_bytes(5, 8, &mats[mi].w_off as *const u64 as *const _);
                            enc.set_buffer(6, Some(y), e(ids_at));
                            let idx = GpuIdxArgs {
                                stride: strides[mi],
                                slots: *n_used as u32,
                                x_stride: 0,
                                ids_stride: max_slots as u32,
                                x_row_stride: hidden as u32,
                                y_row_stride: (max_slots * max_ffn) as u32,
                                n_rows: m as u32,
                            };
                            enc.set_bytes(
                                7,
                                std::mem::size_of::<GpuIdxArgs>() as u64,
                                &idx as *const GpuIdxArgs as *const _,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(
                                    ((*expert_ffn * *n_used * m) as u64
                                        * lanes_per_row(mats[mi].ty, hidden) as u64)
                                        .div_ceil(128),
                                    1,
                                    1,
                                ),
                                MTLSize::new(128, 1, 1),
                            );
                        }
                    } else {
                        for i in 0..m {
                            let ids_off = ids_at + i * max_slots;
                            let gate_row = gate_at + i * max_slots * max_ffn;
                            let up_row = up_at + i * max_slots * max_ffn;
                            let x_off = h_at + i * hidden;
                            enc.set_compute_pipeline_state(&states[0]);
                            enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(x_off));
                            enc.set_buffer(2, Some(y), e(gate_row));
                            enc.set_bytes(3, 4, &a as *const u32 as *const _);
                            enc.set_bytes(4, 4, &b as *const u32 as *const _);
                            enc.set_bytes(5, 8, &mats[0].w_off as *const u64 as *const _);
                            enc.set_buffer(6, Some(y), e(ids_off));
                            let idx = GpuIdxArgs {
                                stride: strides[0],
                                slots: *n_used as u32,
                                x_stride: 0,
                                ids_stride: 0,
                                x_row_stride: 0,
                                y_row_stride: 0,
                                n_rows: 0,
                            };
                            enc.set_bytes(
                                7,
                                std::mem::size_of::<GpuIdxArgs>() as u64,
                                &idx as *const GpuIdxArgs as *const _,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(
                                    ((*expert_ffn * *n_used) as u64
                                        * lanes_per_row(mats[0].ty, hidden) as u64)
                                        .div_ceil(128),
                                    1,
                                    1,
                                ),
                                MTLSize::new(128, 1, 1),
                            );
                            enc.set_compute_pipeline_state(&states[1]);
                            enc.set_buffer(0, Some(&gpu.chunks[mats[1].chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(x_off));
                            enc.set_buffer(2, Some(y), e(up_row));
                            enc.set_bytes(3, 4, &a as *const u32 as *const _);
                            enc.set_bytes(4, 4, &b as *const u32 as *const _);
                            enc.set_bytes(5, 8, &mats[1].w_off as *const u64 as *const _);
                            enc.set_buffer(6, Some(y), e(ids_off));
                            let idx = GpuIdxArgs {
                                stride: strides[1],
                                slots: *n_used as u32,
                                x_stride: 0,
                                ids_stride: 0,
                                x_row_stride: 0,
                                y_row_stride: 0,
                                n_rows: 0,
                            };
                            enc.set_bytes(
                                7,
                                std::mem::size_of::<GpuIdxArgs>() as u64,
                                &idx as *const GpuIdxArgs as *const _,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(
                                    ((*expert_ffn * *n_used) as u64
                                        * lanes_per_row(mats[1].ty, hidden) as u64)
                                        .div_ceil(128),
                                    1,
                                    1,
                                ),
                                MTLSize::new(128, 1, 1),
                            );
                        }
                    }
                    bar(enc);
                    if !*sw_fused {
                        // One swiglu over ALL expert rows (contiguous uniform
                        // blocks), and one over the shared expert's rows.
                        enc.set_compute_pipeline_state(&swiglu_state);
                        enc.set_buffer(0, Some(y), e(gate_at));
                        enc.set_buffer(1, Some(y), e(up_at));
                        let n32 = (m * n_slots * *expert_ffn) as u32;
                        enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        if shared.is_some() {
                            enc.set_buffer(0, Some(y), e(sg_gate_at));
                            enc.set_buffer(1, Some(y), e(sg_up_at));
                            let nsg = (m * *expert_ffn) as u32;
                            enc.set_bytes(2, 4, &nsg as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new((nsg as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                        bar(enc);
                    }
                    // Stage 3: down projections (the fused-swiglu variants
                    // read the up block through buffer 8).
                    let a2 = *expert_ffn as u32;
                    let b2 = hidden as u32;
                    if let Some(rs) = &t_moe_rows[li] {
                        if *sw_fused {
                            enc.set_buffer(8, Some(y), e(up_at));
                        }
                        enc.set_compute_pipeline_state(&rs[2]);
                        enc.set_buffer(0, Some(&gpu.chunks[mats[2].chunk].buf), 0);
                        enc.set_buffer(1, Some(y), e(gate_at));
                        enc.set_buffer(2, Some(y), e(downo_at));
                        enc.set_bytes(3, 4, &a2 as *const u32 as *const _);
                        enc.set_bytes(4, 4, &b2 as *const u32 as *const _);
                        enc.set_bytes(5, 8, &mats[2].w_off as *const u64 as *const _);
                        enc.set_buffer(6, Some(y), e(ids_at));
                        let idx = GpuIdxArgs {
                            stride: strides[2],
                            slots: *n_used as u32,
                            x_stride: *expert_ffn as u32,
                            ids_stride: max_slots as u32,
                            x_row_stride: (max_slots * max_ffn) as u32,
                            y_row_stride: (max_slots * hidden) as u32,
                            n_rows: m as u32,
                        };
                        enc.set_bytes(
                            7,
                            std::mem::size_of::<GpuIdxArgs>() as u64,
                            &idx as *const GpuIdxArgs as *const _,
                        );
                        enc.dispatch_thread_groups(
                            MTLSize::new(
                                ((hidden * *n_used * m) as u64
                                    * lanes_per_row(mats[2].ty, *expert_ffn) as u64)
                                    .div_ceil(128),
                                1,
                                1,
                            ),
                            MTLSize::new(128, 1, 1),
                        );
                    } else {
                        for i in 0..m {
                            let ids_off = ids_at + i * max_slots;
                            let gate_row = gate_at + i * max_slots * max_ffn;
                            let up_row = up_at + i * max_slots * max_ffn;
                            let downo_row = downo_at + i * max_slots * hidden;
                            if *sw_fused {
                                enc.set_buffer(8, Some(y), e(up_row));
                            }
                            enc.set_compute_pipeline_state(&states[2]);
                            enc.set_buffer(0, Some(&gpu.chunks[mats[2].chunk].buf), 0);
                            enc.set_buffer(1, Some(y), e(gate_row));
                            enc.set_buffer(2, Some(y), e(downo_row));
                            enc.set_bytes(3, 4, &a2 as *const u32 as *const _);
                            enc.set_bytes(4, 4, &b2 as *const u32 as *const _);
                            enc.set_bytes(5, 8, &mats[2].w_off as *const u64 as *const _);
                            enc.set_buffer(6, Some(y), e(ids_off));
                            let idx = GpuIdxArgs {
                                stride: strides[2],
                                slots: *n_used as u32,
                                x_stride: *expert_ffn as u32,
                                ids_stride: 0,
                                x_row_stride: 0,
                                y_row_stride: 0,
                                n_rows: 0,
                            };
                            enc.set_bytes(
                                7,
                                std::mem::size_of::<GpuIdxArgs>() as u64,
                                &idx as *const GpuIdxArgs as *const _,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(
                                    ((hidden * *n_used) as u64
                                        * lanes_per_row(mats[2].ty, *expert_ffn) as u64)
                                        .div_ceil(128),
                                    1,
                                    1,
                                ),
                                MTLSize::new(128, 1, 1),
                            );
                        }
                    }
                    // The shared expert's down projection as one TILE matvec
                    // over all rows (the fused-swiglu variant reads the up
                    // rows through buffer 8).
                    if let (Some((smats, _)), Some(tsh)) = (shared, t_shared[li].as_ref()) {
                        if *sw_fused {
                            enc.set_buffer(8, Some(y), e(sg_up_at));
                        }
                        matvec(
                            enc,
                            &tsh[2],
                            &smats[2],
                            *expert_ffn,
                            hidden,
                            sg_gate_at,
                            sg_down_at,
                        );
                    }
                    bar(enc);
                    // Stage 4: the combine over ALL rows in ONE dispatch
                    // (moe_combine_rows; it OVERWRITES delta with the FFN
                    // output - the wo in delta is already absorbed into x,
                    // folding it again double-counts). The shared expert rides
                    // in its own regions: sig_last 0 = none, 1 = sigmoid gate,
                    // 2 = weight 1.
                    let comb_shared = if shared.is_none() {
                        0u32
                    } else if shared_gate.is_some() {
                        1
                    } else {
                        2
                    };
                    enc.set_compute_pipeline_state(&combine_rows_state);
                    enc.set_buffer(0, Some(y), e(downo_at));
                    enc.set_buffer(1, Some(y), e(wts_at));
                    enc.set_buffer(2, Some(y), e(delta_at));
                    let n = hidden as u32;
                    enc.set_bytes(3, 4, &n as *const u32 as *const _);
                    enc.set_bytes(4, 4, &(*n_used as u32) as *const u32 as *const _);
                    enc.set_bytes(5, 4, &comb_shared as *const u32 as *const _);
                    enc.set_bytes(6, 4, &(max_slots as u32) as *const u32 as *const _);
                    enc.set_buffer(7, Some(y), e(sgate_at));
                    enc.set_buffer(8, Some(y), e(sg_down_at));
                    enc.dispatch_thread_groups(
                        MTLSize::new((hidden as u64).div_ceil(256), m as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    bar(enc);
                    dispatched += 4 + 3 * m as u64;
                }
            }
        }
        // Final norm over ALL rows in one dispatch (norm_rows), then the
        // head over all rows and a per-row argmax.
        enc.set_compute_pipeline_state(&norm_rows_state);
        enc.set_buffer(0, Some(y), e(x_at));
        enc.set_buffer(1, Some(y), e(delta_at));
        enc.set_buffer(2, Some(y), e(h_at));
        enc.set_buffer(3, Some(&gpu.chunks[out_norm.chunk].buf), out_norm.off);
        let n = hidden as u32;
        enc.set_bytes(4, 4, &n as *const u32 as *const _);
        enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
        let one = 1u32;
        enc.set_bytes(6, 4, &one as *const u32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
        dispatched += 1;
        bar(enc);
        matvec(enc, &t_out, &out_mat, hidden, vocab, h_at, out_logits_at);
        bar(enc);
        for i in 0..m {
            let slot = amax_at + i * (64 * 2 + 1);
            enc.set_compute_pipeline_state(&argmax_state);
            enc.set_buffer(0, Some(y), e(out_logits_at + i * vocab));
            let v32 = vocab as u32;
            enc.set_bytes(1, 4, &v32 as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(slot));
            enc.dispatch_thread_groups(MTLSize::new(64, 1, 1), MTLSize::new(256, 1, 1));
            bar(enc);
            enc.set_compute_pipeline_state(&argmax_final_state);
            enc.set_buffer(0, Some(y), e(slot));
            let np = 64u32;
            enc.set_bytes(1, 4, &np as *const u32 as *const _);
            enc.set_buffer(2, Some(y), e(slot + 64 * 2));
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            bar(enc);
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(dispatched + 2, Ordering::Relaxed);

        let mut argmax = Vec::with_capacity(m);
        let mut hidden_rows = vec![0f32; m * hidden];
        unsafe {
            let yp = gpu.y_arena.contents() as *const f32;
            for i in 0..m {
                argmax.push((yp.add(amax_at + i * (64 * 2 + 1) + 64 * 2) as *const u32).read());
            }
            std::ptr::copy_nonoverlapping(yp.add(h_at), hidden_rows.as_mut_ptr(), m * hidden);
        }
        TokenOut::Rows {
            argmax,
            hidden: hidden_rows,
        }
    });
    Some(out)
}

/// One decode layer's whole attention half as ONE command buffer: the three
/// qkv projections, per-head norm + RoPE, the f16 store of k/v into the
/// cache, attention over it, and the output projection. Five waits become
/// one, and by the GPU's own clock the driver's per-buffer scheduling was as
/// expensive as the round trips themselves.
///
/// NEOX rope only (what every current target model uses); anything else
/// declines to the CPU path.
pub struct AttnBlockReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    /// The normed layer input, `hidden` floats.
    pub x: &'a [f32],
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// `(sin, cos)` per NEOX pair, `head_dim / 2` entries.
    pub rope: &'a [[f32; 2]],
    pub eps: f32,
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub pos: usize,
    pub scale: f32,
}

#[repr(C)]
struct QkPrepArgs {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    eps: f32,
    pos: u32,
    has_qk_norm: u32,
    rot_dim: u32,
    k_base: u64,
    v_base: u64,
    has_bias: u32,
    pad2: u32,
}

#[repr(C)]
struct QkPrep256Args {
    n_heads: u32,
    n_kv_heads: u32,
    kv_dim: u32,
    eps: f32,
    pos: u32,
    has_qk_norm: u32,
    rot_dim: u32,
    gate_in_q: u32,
    k_base: u64,
    v_base: u64,
}

#[repr(C)]
struct GdnConvArgs {
    channels: u32,
    d_conv: u32,
    pad0: u32,
    pad1: u32,
    conv_off: u64,
}

#[repr(C)]
struct GdnStepArgs {
    heads_k: u32,
    heads_v: u32,
    d: u32,
    key_dim: u32,
    eps: f32,
    pad0: u32,
    state_off: u64,
}

#[repr(C)]
struct QkPrepBatch256Args {
    n_heads: u32,
    n_kv_heads: u32,
    kv_dim: u32,
    eps: f32,
    base: u32,
    has_qk_norm: u32,
    rot_dim: u32,
    gate_in_q: u32,
    k_base: u64,
    v_base: u64,
}

#[repr(C)]
struct GdnConvBatchArgs {
    channels: u32,
    d_conv: u32,
    m: u32,
    pad0: u32,
    conv_off: u64,
}

#[repr(C)]
struct GdnStepBatchArgs {
    heads_k: u32,
    heads_v: u32,
    d: u32,
    key_dim: u32,
    m: u32,
    eps: f32,
    state_off: u64,
}

pub fn attn_block(req: &AttnBlockReq) -> Option<Vec<f32>> {
    let hd = req.head_dim;
    if hd != ATTN_HEAD_DIM || req.rope.len() != hd / 2 {
        return None;
    }
    let hidden = req.x.len();
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    if req.wq.2 != q_dim || req.wk.2 != kv || req.wv.2 != kv || req.wo.2 != hidden {
        return None;
    }
    let span = (req.pos + 1) * req.kv_dim;
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    if (req.k_off + span) * 2 > req.cache.len || (req.v_off + span) * 2 > req.cache.len {
        return None;
    }

    let mats = [
        resolve(&gpu, req.wq.0, req.wq.1)?,
        resolve(&gpu, req.wk.0, req.wk.1)?,
        resolve(&gpu, req.wv.0, req.wv.1)?,
        resolve(&gpu, req.wo.0, req.wo.1)?,
    ];
    let dims = [(hidden, q_dim), (hidden, kv), (hidden, kv), (q_dim, hidden)];
    let mut states = Vec::with_capacity(4);
    for (mat, &(n_in, _)) in mats.iter().zip(&dims) {
        let lpr = lanes_per_row(mat.ty, n_in);
        states.push(gpu.pipeline(mat.kernel, 1, lpr)?.to_owned());
    }

    // Arena plan: q, k, v, attention out and the projection, all in y.
    let align = |n: usize| (n + 255) & !255;
    let q_off = 0usize;
    let k_arena = align(q_dim * 4);
    let v_arena = k_arena + align(kv * 4);
    let attn_off = v_arena + align(kv * 4);
    let out_off = attn_off + align(q_dim * 4);
    gpu.ensure_arenas(align(hidden * 4), out_off + align(hidden * 4));
    unsafe {
        std::ptr::copy_nonoverlapping(
            req.x.as_ptr() as *const u8,
            gpu.x_arena.contents() as *mut u8,
            hidden * 4,
        );
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Concurrent, with a barrier at each stage boundary: q, k and v are
        // independent and small, and the serial encoder ran them one by one.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);

        // qkv projections out of the x arena.
        for (i, y_at) in [(0usize, q_off), (1, k_arena), (2, v_arena)] {
            let (n_in, n_out) = dims[i];
            enc.set_compute_pipeline_state(&states[i]);
            enc.set_buffer(0, Some(&gpu.chunks[mats[i].chunk].buf), 0);
            enc.set_buffer(1, Some(&gpu.x_arena), 0);
            enc.set_buffer(2, Some(&gpu.y_arena), y_at as u64);
            let n_in32 = n_in as u32;
            let n_out32 = n_out as u32;
            enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
            enc.set_bytes(5, 8, &mats[i].w_off as *const u64 as *const _);
            let lpr = lanes_per_row(mats[i].ty, n_in) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64 * lpr).div_ceil(128), 1, 1),
                MTLSize::new(128, 1, 1),
            );
        }

        enc.memory_barrier_with_resources(&[&gpu.y_arena]);

        // Norm + rope + cache store.
        let args = QkPrepArgs {
            n_heads: req.n_heads as u32,
            n_kv_heads: req.n_kv_heads as u32,
            head_dim: hd as u32,
            kv_dim: req.kv_dim as u32,
            eps: req.eps,
            pos: req.pos as u32,
            has_qk_norm: req.q_norm.is_some() as u32,
            rot_dim: hd as u32,
            k_base: req.k_off as u64,
            v_base: req.v_off as u64,
            has_bias: 0,
            pad2: 0,
        };
        enc.set_compute_pipeline_state(&gpu.pipelines[&("qk_prep", 1, 1)]);
        enc.set_buffer(0, Some(&gpu.y_arena), q_off as u64);
        enc.set_buffer(1, Some(&gpu.y_arena), k_arena as u64);
        enc.set_buffer(2, Some(&gpu.y_arena), v_arena as u64);
        enc.set_buffer(3, Some(&req.cache.buf), 0);
        enc.set_bytes(
            4,
            std::mem::size_of::<QkPrepArgs>() as u64,
            &args as *const QkPrepArgs as *const _,
        );
        let ones = [1.0f32; ATTN_HEAD_DIM];
        let qw = req.q_norm.unwrap_or(&ones);
        let kw = req.k_norm.unwrap_or(&ones);
        enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
        enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
        enc.set_bytes(
            7,
            (req.rope.len() * 8) as u64,
            req.rope.as_ptr() as *const _,
        );
        for index in 8u64..=10 {
            enc.set_buffer(index, Some(&req.cache.buf), 0);
        }
        enc.dispatch_thread_groups(
            MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );

        enc.memory_barrier_with_resources(&[&gpu.y_arena, &req.cache.buf]);

        // Attention over the cache the previous dispatch just extended.
        let attn_req = AttnReq {
            cache: req.cache,
            k_off: req.k_off,
            v_off: req.v_off,
            q: &[],
            kv_dim: req.kv_dim,
            head_dim: hd,
            n_q_heads: req.n_heads,
            group: req.n_heads / req.n_kv_heads.max(1),
            n_pos: req.pos + 1,
            scale: req.scale,
        };
        encode_attend_from(enc, &gpu, &attn_req, &gpu.y_arena, q_off, attn_off);
        enc.memory_barrier_with_resources(&[&gpu.y_arena]);

        // Output projection.
        let (n_in, n_out) = dims[3];
        enc.set_compute_pipeline_state(&states[3]);
        enc.set_buffer(0, Some(&gpu.chunks[mats[3].chunk].buf), 0);
        enc.set_buffer(1, Some(&gpu.y_arena), attn_off as u64);
        enc.set_buffer(2, Some(&gpu.y_arena), out_off as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mats[3].w_off as *const u64 as *const _);
        let lpr = lanes_per_row(mats[3].ty, n_in) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new((n_out as u64 * lpr).div_ceil(128), 1, 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(6, Ordering::Relaxed);

        let mut out = vec![0f32; hidden];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const u8).add(out_off),
                out.as_mut_ptr() as *mut u8,
                hidden * 4,
            );
        }
        out
    }))
}

/// A prefill chunk's whole attention half as ONE command buffer: the three
/// qkv projections over all `m` rows (tile-matmul kernels), per-row norm +
/// rope + f16 cache store (qk_prep_batch), causal attention per row, and
/// the output projection. Mirrors the decode token buffer; the FFN half
/// stays separate because routing groups tokens on the CPU.
/// Extra weights for the fused prefill: with this present (and
/// `prefill_begin` called for the chunk), the block reads the residual
/// stream from GPU memory, norms on the GPU, folds the attention output
/// back in, re-norms for the FFN and finishes with the router matmul -
/// returning router logits instead of the projected rows.
pub struct PrefillFusion<'a> {
    /// F32 raw norm weights, `hidden` floats each.
    pub attn_norm: &'a [u8],
    pub ffn_norm: &'a [u8],
    /// F32 raw router weights, `n_expert * hidden` floats.
    pub router: &'a [u8],
    pub n_expert: usize,
}

/// Upload the chunk's residual stream for the fused prefill path.
/// Stamp a GPUTimestamp after a dispatch. No-op unless ALLPAKA_GPU_COUNTERS.
macro_rules! cstamp {
    ($gpu:expr, $enc:expr, $label:expr) => {
        if let Some(buf) = &$gpu.cstamps {
            let i = $gpu.cstamp_idx.get();
            if i < CSTAMP_CAP {
                $enc.sample_counters_in_buffer(buf, i as u64, false);
                $gpu.cstamp_labels.borrow_mut().push($label);
                $gpu.cstamp_idx.set(i + 1);
            }
        }
    };
}

pub fn prefill_begin(xs: &[f32]) -> Option<()> {
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    gpu.prefill_drain();
    gpu.cstamps_begin();
    // One-buffer segments arm lazily: the first eligible layer of the
    // chunk opens the shared command buffer (see prefill_attn_block), an
    // ineligible layer seals it again (pf_obuf_seal). Nothing to arm here.
    gpu.pf_obuf_cmd = None;
    gpu.ensure_prefill(xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), gpu.pf_x.contents() as *mut f32, xs.len());
    }
    Some(())
}

/// Download the residual stream after the last fused layer.
pub fn prefill_end(xs: &mut [f32]) -> Option<()> {
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    let mut cal = None;
    if let Some(cmd) = gpu.pf_obuf_cmd.take() {
        // The one-buffer chunk commits here: the only wait of the prefill.
        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        use objc::{msg_send, sel, sel_impl};
        let (gs, ge): (f64, f64) =
            unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
        cal = Some((gs, ge));
        note_gpu_times(&cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
    }
    gpu.prefill_drain();
    gpu.cstamps_report(cal);
    if xs.len() * 4 > gpu.pf_x_cap {
        return None;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(gpu.pf_x.contents() as *const f32, xs.as_mut_ptr(), xs.len());
    }
    Some(())
}

/// Wait out deferred prefill buffers after a mid-chunk fallback: the CPU
/// path that takes over would otherwise race buffers still in flight.
pub fn prefill_abort() {
    if let Some(Some(cell)) = GPU.get().map(|c| c.as_ref()) {
        if let Ok(mut gpu) = cell.lock() {
            // A partial one-buffer chunk was never committed; dropping the
            // retained handle discards it.
            gpu.pf_obuf_cmd = None;
            gpu.prefill_drain();
        }
    }
}

pub struct PrefillAttnReq<'a> {
    pub wq: (GgmlType, &'a [u8], usize),
    pub wk: (GgmlType, &'a [u8], usize),
    pub wv: (GgmlType, &'a [u8], usize),
    pub wo: (GgmlType, &'a [u8], usize),
    /// Normed layer input, `m * hidden` floats.
    pub hs: &'a [f32],
    pub m: usize,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// `(sin, cos)` pairs for every row: `m * rot_dim / 2` entries.
    pub ropes: &'a [[f32; 2]],
    /// Rotary width; == head_dim for full rope, smaller for GLM's partial,
    /// 64 for qwen35moe's 256-wide heads.
    pub rot_dim: usize,
    /// qwen35moe: the sigmoid output gate rides inside the q projection
    /// ([q | gate] per head, wq.n_out == 2 * q_dim); split and applied
    /// post-attention.
    pub gate_in_q: bool,
    /// GLM's per-head q/k/v additive biases as raw F32 GGUF bytes.
    pub attn_bias: Option<(&'a [u8], &'a [u8], &'a [u8])>,
    pub eps: f32,
    pub cache: &'a SharedRegion,
    pub k_off: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// First row's position; row i attends over base + i + 1 positions.
    pub base: usize,
    pub scale: f32,
    pub fusion: Option<PrefillFusion<'a>>,
}

pub fn prefill_attn_block(req: &PrefillAttnReq) -> Option<Vec<f32>> {
    let hidden = req.hs.len() / req.m.max(1);
    let hd = req.head_dim;
    // 128 everywhere; 256 on qwen35moe's full-attention layers. Below the
    // tile-matmul threshold the projections would need per-row matvec tiling
    // this encoder does not carry; short tail chunks decline to the
    // step-by-step path instead. (Caught by decode_paths_agree: a TILE=1
    // pipeline silently computes only the first row.)
    let rot_ok = (hd == ATTN_HEAD_DIM && (req.rot_dim == hd || req.rot_dim * 2 == hd))
        || (hd == 256 && req.rot_dim == 64);
    if !(hd == ATTN_HEAD_DIM || hd == 256)
        || !rot_ok
        || req.ropes.len() != req.m * req.rot_dim / 2
        || req.hs.len() != req.m * hidden
        || req.m < MM_MIN_M
    {
        return None;
    }
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let wq_out = if req.gate_in_q { 2 * q_dim } else { q_dim };
    if req.wq.2 != wq_out || req.wk.2 != kv || req.wv.2 != kv || req.wo.2 != hidden {
        return None;
    }
    let span = (req.base + req.m) * req.kv_dim;
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    if (req.k_off + span) * 2 > req.cache.len || (req.v_off + span) * 2 > req.cache.len {
        return None;
    }

    let mats = [
        resolve(&gpu, req.wq.0, req.wq.1)?,
        resolve(&gpu, req.wk.0, req.wk.1)?,
        resolve(&gpu, req.wv.0, req.wv.1)?,
        resolve(&gpu, req.wo.0, req.wo.1)?,
    ];
    #[allow(unused_variables)]
    let mut states = Vec::with_capacity(4);
    for mat in &mats {
        states.push(
            gpu.pipeline(mm_kernel_for(mat.ty, req.m)?.0, 1, 1)?
                .to_owned(),
        );
    }
    let prep_state = gpu.pipelines[&("qk_prep_batch", 1, 1)].to_owned();
    let attend_state = gpu.pipelines[&(attend_rows_kernel(), 1, 1)].to_owned();

    // Fusion: norms and the router resolved up front; the fused path reads
    // xs from pf_x, so the hs upload disappears. A leading Dense layer
    // (GLM's layer 0) fuses the norms but has no router: n_expert == 0.
    struct FusionRefs {
        attn_norm: NormRef,
        ffn_norm: NormRef,
        router: Option<(MatRef, ComputePipelineState)>,
        n_expert: usize,
    }
    let fusion = match &req.fusion {
        Some(f) => {
            if gpu.pf_x_cap < req.m * hidden * 4 || gpu.pf_hs_cap < req.m * hidden * 4 {
                return None;
            }
            let attn_norm = resolve_norm(&gpu, f.attn_norm, hidden)?;
            let ffn_norm = resolve_norm(&gpu, f.ffn_norm, hidden)?;
            let router = if f.n_expert > 0 {
                let router = resolve_f32(&gpu, f.router, hidden)?;
                let router_state = gpu
                    .pipeline(mm_kernel_for(GgmlType::F32, req.m)?.0, 1, 1)?
                    .to_owned();
                Some((router, router_state))
            } else {
                None
            };
            Some(FusionRefs {
                attn_norm,
                ffn_norm,
                router,
                n_expert: f.n_expert,
            })
        }
        None => None,
    };
    // One-buffer note: a layer that cannot join the chunk's shared
    // command buffer no longer declines here. Resolve failures above
    // already returned None (the chunk aborts and the fallback recomputes
    // it); a merely non-onebuf layer (CPU routing, PF_SPLIT) seals the
    // open segment at the encode site below and runs on its own buffer.
    let norm_state = gpu.pipelines[&("norm_rows", 1, 1)].to_owned();
    let prep256_state = gpu.pipelines[&("qk_prep_batch256", 1, 1)].to_owned();
    let attend256_state = gpu.pipelines[&("attend_rows256", 1, 1)].to_owned();
    let attend_mm256_state = gpu.pipelines[&("attend_mm256", 1, 1)].to_owned();
    let gate_mul_state = gpu.pipelines[&("attn_gate_mul", 1, 1)].to_owned();
    // GLM's q/k/v biases resolved outside the encoder closure (`?` needs
    // the function's Option, not the closure's Vec).
    let bias_refs = match req.attn_bias {
        Some((qb, kb, vb)) => Some((
            resolve_f32(&gpu, qb, 0)?,
            resolve_f32(&gpu, kb, 0)?,
            resolve_f32(&gpu, vb, 0)?,
        )),
        None => None,
    };

    // Arenas: x holds hs then the rope tables; y holds q (raw, possibly
    // double-wide with the gate), k, v, the deinterleaved q and gate rows,
    // attention output and the projection.
    let align = |n: usize| (n + 63) & !63;
    let hs_at = 0usize;
    let ropes_at = hs_at + align(req.m * hidden);
    let x_total = ropes_at + align(req.ropes.len() * 2);
    let q_at = 0usize;
    let k_at = q_at + align(req.m * wq_out);
    let v_at = k_at + align(req.m * kv);
    let qn_at = v_at + align(req.m * kv);
    let qgate_at = qn_at + align(req.m * q_dim);
    let attn_at = qgate_at + align(req.m * q_dim);
    let out_at = attn_at + align(req.m * q_dim);
    let router_at = out_at + align(req.m * hidden);
    let n_expert = fusion.as_ref().map_or(0, |f| f.n_expert);
    let y_total = router_at + align(req.m * n_expert);
    gpu.ensure_arenas(x_total * 4, y_total * 4);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut f32;
        if fusion.is_none() {
            std::ptr::copy_nonoverlapping(req.hs.as_ptr(), xp.add(hs_at), req.hs.len());
        }
        std::ptr::copy_nonoverlapping(
            req.ropes.as_ptr() as *const f32,
            xp.add(ropes_at),
            req.ropes.len() * 2,
        );
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        // Cloned queue handle: command buffers borrow it, leaving `gpu`
        // free for the deferred commit bookkeeping.
        let queue = gpu.queue.clone();
        let mut cmd = queue.new_command_buffer();
        // ALLPAKA_PF_SPLIT=1: commit after every stage and print its GPU
        // time. Debug-only; serialises the stages, so wall time degrades.
        let split = std::env::var_os("ALLPAKA_PF_SPLIT").is_some();
        // One-buffer chunk (ALLPAKA_PF_ONEBUF=1, default): encode into the
        // chunk's shared command buffer - a sequential encoder per stage,
        // the commit happens once in prefill_end. The segment arms lazily
        // on the first eligible layer; a layer that stays off it seals the
        // open segment first, so its own buffer stays ordered after the
        // work already encoded there.
        let onebuf = fusion.is_some() && gpu_route() && pf_onebuf() && !split;
        if !onebuf {
            gpu.pf_obuf_seal();
        } else {
            gpu.pf_obuf_arm();
        }
        // Deferred commit: chain behind the previous layer's buffer through
        // the shared event instead of a CPU wait. Only the fused GPU-routing
        // path qualifies - it reads nothing back. PF_SPLIT keeps the old
        // waits (its per-stage commits are a different buffer shape anyway).
        let defer = !onebuf && fusion.is_some() && gpu_route() && pf_defer() && !split;
        if defer {
            cmd.encode_wait_for_event(&gpu.pf_ev, gpu.pf_ev_val);
        }
        let mut enc = if onebuf {
            gpu.pf_obuf_cmd
                .as_deref()
                .expect("onebuf armed")
                .compute_command_encoder_with_dispatch_type(serial_dispatch())
        } else {
            cmd.compute_command_encoder_with_dispatch_type(serial_dispatch())
        };
        macro_rules! split_here {
            ($label:expr) => {
                if split {
                    use objc::{msg_send, sel, sel_impl};
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
                    let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
                    eprintln!("attn {}: {:.3} ms", $label, (ge - gs) * 1e3);
                    cmd = queue.new_command_buffer();
                    enc = cmd.compute_command_encoder_with_dispatch_type(
                        metal::MTLDispatchType::Concurrent,
                    );
                }
            };
        }
        let y = &gpu.y_arena;
        let e = |off: usize| (off * 4) as u64;

        let matmul = |enc: &metal::ComputeCommandEncoderRef,
                      state: &ComputePipelineState,
                      mat: &MatRef,
                      n_in: usize,
                      n_out: usize,
                      x_buf: &Buffer,
                      x_off: usize,
                      y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            let m32 = req.m as u32;
            enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
            let (bm, bn) = mm_kernel_for(mat.ty, req.m).map_or((32, 32), |(_, bm, bn)| (bm, bn));
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (n_out as u64).div_ceil(bm as u64),
                    (req.m as u64).div_ceil(bn as u64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        };

        let norm_rows = |enc: &metal::ComputeCommandEncoderRef,
                         norm: &NormRef,
                         use_delta: u32,
                         delta_off: usize| {
            enc.set_compute_pipeline_state(&norm_state);
            enc.set_buffer(0, Some(&gpu.pf_x), 0);
            enc.set_buffer(1, Some(y), e(delta_off));
            enc.set_buffer(2, Some(&gpu.pf_hs), 0);
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.set_bytes(6, 4, &use_delta as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(req.m as u64, 1, 1), MTLSize::new(256, 1, 1));
        };
        let (hs_buf, hs_off): (&Buffer, usize) = if let Some(f) = &fusion {
            if onebuf {
                // The previous layer's FFN stages (pf_x combine, y writes)
                // live on the same command buffer now - order them.
                enc.memory_barrier_with_resources(&[y, &gpu.pf_x, &gpu.pf_hs, &gpu.route_buf]);
            }
            norm_rows(enc, &f.attn_norm, 0, out_at);
            cstamp!(gpu, enc, "anorm");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
            }
            (&gpu.pf_hs, 0)
        } else {
            (&gpu.x_arena, hs_at)
        };
        matmul(
            enc, &states[0], &mats[0], hidden, wq_out, hs_buf, hs_off, q_at,
        );
        cstamp!(gpu, enc, "wq");
        matmul(enc, &states[1], &mats[1], hidden, kv, hs_buf, hs_off, k_at);
        cstamp!(gpu, enc, "wk");
        matmul(enc, &states[2], &mats[2], hidden, kv, hs_buf, hs_off, v_at);
        cstamp!(gpu, enc, "wv");
        split_here!("qkv");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }

        // head_dim-256 path: batched qk_prep256 splits the fused gate out of
        // the q rows and writes normed q to qn_at; attend_rows256 runs one
        // threadgroup per (head, row); the sigmoid gate lands before wo.
        if hd == 256 {
            enc.set_compute_pipeline_state(&prep256_state);
            enc.set_buffer(0, Some(y), e(q_at));
            enc.set_buffer(1, Some(y), e(k_at));
            enc.set_buffer(2, Some(y), e(v_at));
            enc.set_buffer(3, Some(&req.cache.buf), 0);
            let args = QkPrepBatch256Args {
                n_heads: req.n_heads as u32,
                n_kv_heads: req.n_kv_heads as u32,
                kv_dim: req.kv_dim as u32,
                eps: req.eps,
                base: req.base as u32,
                has_qk_norm: req.q_norm.is_some() as u32,
                rot_dim: req.rot_dim as u32,
                gate_in_q: req.gate_in_q as u32,
                k_base: req.k_off as u64,
                v_base: req.v_off as u64,
            };
            enc.set_bytes(
                4,
                std::mem::size_of::<QkPrepBatch256Args>() as u64,
                &args as *const QkPrepBatch256Args as *const _,
            );
            let ones = [1.0f32; 256];
            let qw = req.q_norm.unwrap_or(&ones);
            let kw = req.k_norm.unwrap_or(&ones);
            enc.set_bytes(5, (256 * 4) as u64, qw.as_ptr() as *const _);
            enc.set_bytes(6, (256 * 4) as u64, kw.as_ptr() as *const _);
            enc.set_buffer(7, Some(&gpu.x_arena), e(ropes_at));
            enc.set_buffer(8, Some(y), e(qn_at));
            enc.set_buffer(9, Some(y), e(qgate_at));
            enc.dispatch_thread_groups(
                MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, req.m as u64, 1),
                MTLSize::new(32, 1, 1),
            );
            cstamp!(gpu, enc, "prep");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
            }
            split_here!("prep");

            enc.set_compute_pipeline_state(&attend256_state);
            enc.set_buffer(0, Some(&req.cache.buf), 0);
            enc.set_buffer(1, Some(&req.cache.buf), 0);
            enc.set_buffer(2, Some(y), e(qn_at));
            enc.set_buffer(3, Some(y), e(attn_at));
            for (index, value) in [
                (4u64, req.kv_dim as u32),
                (5, 256u32),
                (6, req.base as u32),
                (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
            ] {
                enc.set_bytes(index, 4, &value as *const u32 as *const _);
            }
            enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
            for (index, value) in [(9u64, req.k_off as u64), (10, req.v_off as u64)] {
                enc.set_bytes(index, 8, &value as *const u64 as *const _);
            }
            let heads32 = req.n_heads as u32;
            enc.set_bytes(11, 4, &heads32 as *const u32 as *const _);
            if attend_mm() {
                // MMA tiles: same bindings, 8 rows per threadgroup.
                enc.set_compute_pipeline_state(&attend_mm256_state);
                let rows32 = req.m as u32;
                enc.set_bytes(12, 4, &rows32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(req.n_heads as u64, (req.m as u64).div_ceil(8), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                enc.dispatch_thread_groups(
                    MTLSize::new(req.n_heads as u64, req.m as u64, 1),
                    MTLSize::new(512, 1, 1),
                );
            }
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
            if req.gate_in_q {
                enc.set_compute_pipeline_state(&gate_mul_state);
                enc.set_buffer(0, Some(y), e(attn_at));
                enc.set_buffer(1, Some(y), e(qgate_at));
                let n32 = (req.m * q_dim) as u32;
                enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[y]);
                }
            }
            cstamp!(gpu, enc, "attend");
            split_here!("attend");

            matmul(
                enc,
                &states[3],
                &mats[3],
                q_dim,
                hidden,
                &gpu.y_arena,
                attn_at,
                out_at,
            );
            cstamp!(gpu, enc, "wo");
            split_here!("wo");
            if let Some(f) = &fusion {
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[y]);
                }
                // xs += wo output, then the FFN norm - one fused pass - and the
                // router matmul, all before the single wait.
                norm_rows(enc, &f.ffn_norm, 1, out_at);
                cstamp!(gpu, enc, "fnorm");
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
                }
                if let Some((router, router_state)) = &f.router {
                    matmul(
                        enc,
                        router_state,
                        router,
                        hidden,
                        f.n_expert,
                        &gpu.pf_hs,
                        0,
                        router_at,
                    );
                    cstamp!(gpu, enc, "rmm");
                }
                split_here!("ffnnorm_router");
            }
            enc.end_encoding();
            ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let t_wait = std::time::Instant::now();
            if onebuf {
                // Nothing: the chunk commits once, in prefill_end.
            } else if defer {
                gpu.commit_chained(cmd);
            } else {
                cmd.commit();
                cmd.wait_until_completed();
                note_gpu_times(cmd);
            }
            WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if !defer && !onebuf && std::env::var_os("ALLPAKA_FFN_TIME").is_some() {
                use objc::{msg_send, sel, sel_impl};
                let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
                let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
                eprintln!("attbuf: {:.3} ms", (ge - gs) * 1e3);
            }
            if !onebuf {
                CALLS.fetch_add(1, Ordering::Relaxed);
            }
            DISPATCHES.fetch_add(6, Ordering::Relaxed);

            if fusion.is_some() && gpu_route() {
                PF_ROUTER_AT.store(router_at as u64, Ordering::Relaxed);
                return Vec::new();
            }
            let (src_at, out_len) = if fusion.is_some() {
                (router_at, req.m * n_expert)
            } else {
                (out_at, req.m * hidden)
            };
            let mut out = vec![0f32; out_len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (gpu.y_arena.contents() as *const f32).add(src_at),
                    out.as_mut_ptr(),
                    out.len(),
                );
            }
            return out;
        }

        enc.set_compute_pipeline_state(&prep_state);
        enc.set_buffer(0, Some(y), e(q_at));
        enc.set_buffer(1, Some(y), e(k_at));
        enc.set_buffer(2, Some(y), e(v_at));
        enc.set_buffer(3, Some(&req.cache.buf), 0);
        let args = QkPrepArgs {
            n_heads: req.n_heads as u32,
            n_kv_heads: req.n_kv_heads as u32,
            head_dim: hd as u32,
            kv_dim: req.kv_dim as u32,
            eps: req.eps,
            pos: req.base as u32,
            has_qk_norm: req.q_norm.is_some() as u32,
            rot_dim: req.rot_dim as u32,
            k_base: req.k_off as u64,
            v_base: req.v_off as u64,
            has_bias: req.attn_bias.is_some() as u32,
            pad2: 0,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<QkPrepArgs>() as u64,
            &args as *const QkPrepArgs as *const _,
        );
        let ones = [1.0f32; ATTN_HEAD_DIM];
        let qw = req.q_norm.unwrap_or(&ones);
        let kw = req.k_norm.unwrap_or(&ones);
        enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
        enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
        enc.set_buffer(7, Some(&gpu.x_arena), e(ropes_at));
        // Bias buffers: resolved F32 regions for GLM, any valid buffer
        // otherwise (the kernel skips them on has_bias == 0).
        match &bias_refs {
            Some((qb, kb, vb)) => {
                enc.set_buffer(8, Some(&gpu.chunks[qb.chunk].buf), qb.w_off);
                enc.set_buffer(9, Some(&gpu.chunks[kb.chunk].buf), kb.w_off);
                enc.set_buffer(10, Some(&gpu.chunks[vb.chunk].buf), vb.w_off);
            }
            None => {
                for index in 8u64..=10 {
                    enc.set_buffer(index, Some(&req.cache.buf), 0);
                }
            }
        }
        enc.dispatch_thread_groups(
            MTLSize::new((req.n_heads + 2 * req.n_kv_heads) as u64, req.m as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        cstamp!(gpu, enc, "prep");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y, &req.cache.buf]);
        }
        split_here!("prep");

        // Causal attention over every row in ONE dispatch: rows in grid.y,
        // each over its own prefix.
        enc.set_compute_pipeline_state(&attend_state);
        enc.set_buffer(0, Some(&req.cache.buf), 0);
        enc.set_buffer(1, Some(&req.cache.buf), 0);
        enc.set_buffer(2, Some(y), e(q_at));
        enc.set_buffer(3, Some(y), e(attn_at));
        for (index, value) in [
            (4u64, req.kv_dim as u32),
            (5, hd as u32),
            (6, req.base as u32),
            (7, (req.n_heads / req.n_kv_heads.max(1)) as u32),
        ] {
            enc.set_bytes(index, 4, &value as *const u32 as *const _);
        }
        enc.set_bytes(8, 4, &req.scale as *const f32 as *const _);
        for (index, value) in [(9u64, req.k_off as u64), (10, req.v_off as u64)] {
            enc.set_bytes(index, 8, &value as *const u64 as *const _);
        }
        let heads32 = req.n_heads as u32;
        enc.set_bytes(11, 4, &heads32 as *const u32 as *const _);
        if attend_mm() {
            // MMA tiles: same bindings as attend_rows, 8 rows per
            // threadgroup.
            enc.set_compute_pipeline_state(&gpu.pipelines[&("attend_mm", 1, 1)]);
            let rows32 = req.m as u32;
            enc.set_bytes(12, 4, &rows32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(req.n_heads as u64, (req.m as u64).div_ceil(8), 1),
                MTLSize::new(128, 1, 1),
            );
        } else {
            let tile = match attend_rows_kernel() {
                "attend_rows_t8" => 8u64,
                "attend_rows_t4" => 4,
                _ => 1,
            };
            if tile > 1 {
                let rows32 = req.m as u32;
                enc.set_bytes(12, 4, &rows32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(req.n_heads as u64, (req.m as u64).div_ceil(tile), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                enc.dispatch_thread_groups(
                    MTLSize::new(req.n_heads as u64, req.m as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
        }
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        cstamp!(gpu, enc, "attend");
        split_here!("attend");

        matmul(
            enc,
            &states[3],
            &mats[3],
            q_dim,
            hidden,
            &gpu.y_arena,
            attn_at,
            out_at,
        );
        cstamp!(gpu, enc, "wo");
        split_here!("wo");
        if let Some(f) = &fusion {
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
            // xs += wo output, then the FFN norm - one fused pass - and the
            // router matmul, all before the single wait.
            norm_rows(enc, &f.ffn_norm, 1, out_at);
            cstamp!(gpu, enc, "fnorm");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
            }
            if let Some((router, router_state)) = &f.router {
                matmul(
                    enc,
                    router_state,
                    router,
                    hidden,
                    f.n_expert,
                    &gpu.pf_hs,
                    0,
                    router_at,
                );
                cstamp!(gpu, enc, "rmm");
            }
            split_here!("ffnnorm_router");
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        if onebuf {
            // Nothing: the chunk commits once, in prefill_end.
        } else if defer {
            gpu.commit_chained(cmd);
        } else {
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
        }
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !defer && !onebuf && std::env::var_os("ALLPAKA_FFN_TIME").is_some() {
            use objc::{msg_send, sel, sel_impl};
            let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
            let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
            eprintln!("attbuf: {:.3} ms", (ge - gs) * 1e3);
        }
        if !onebuf {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        DISPATCHES.fetch_add(6, Ordering::Relaxed);

        // Fused: the caller needs only the router logits; the projected
        // rows already live in pf_x. Plain: the projected rows come back.
        // GPU routing leaves the logits in y_arena and reports the offset.
        if fusion.is_some() && gpu_route() {
            PF_ROUTER_AT.store(router_at as u64, Ordering::Relaxed);
            return Vec::new();
        }
        let (src_at, out_len) = if fusion.is_some() {
            (router_at, req.m * n_expert)
        } else {
            (out_at, req.m * hidden)
        };
        let mut out = vec![0f32; out_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const f32).add(src_at),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    });
    Some(out)
}

/// A prefill chunk of one gated-delta-net layer (qwen35moe linear
/// attention), mirroring [`prefill_attn_block`]'s fusion contract: reads the
/// residual stream from pf_x, leaves it updated there (branch output folded
/// into the FFN norm), the router logits in y_arena for GPU routing.
pub struct PrefillGdnReq<'a> {
    /// hidden -> conv channels (key_dim*2 + value_dim), Q8_0.
    pub wqkv: (GgmlType, &'a [u8], usize),
    /// hidden -> value_dim (the z gate), Q8_0.
    pub zgate: (GgmlType, &'a [u8], usize),
    /// F32 [hidden -> heads_v] projections, raw bytes in the mmap.
    pub alpha: &'a [u8],
    pub beta: &'a [u8],
    /// F32 conv weights [channels * d_conv], raw bytes.
    pub conv1d: &'a [u8],
    /// Small F32 vectors handed to the kernel by value.
    pub a: &'a [f32],
    pub dt: &'a [f32],
    pub ssm_norm: &'a [f32],
    /// value_dim -> hidden, Q8_0.
    pub ssm_out: (GgmlType, &'a [u8], usize),
    pub heads_k: usize,
    pub heads_v: usize,
    pub d: usize,
    pub d_conv: usize,
    pub hidden: usize,
    pub m: usize,
    pub eps: f32,
    /// The shared SSM region: conv window and deltanet state, updated in
    /// place (element offsets per layer).
    pub ssm: &'a SharedRegion,
    /// MTP verification: per-row rollback slots (`SsmCache::arm_slots`),
    /// (region, slot stride in elements). The batch kernels then also write
    /// every row's state/window into the slots.
    pub ssm_slots: Option<(&'a SharedRegion, usize)>,
    pub conv_off: usize,
    pub state_off: usize,
    pub fusion: Option<PrefillFusion<'a>>,
}

pub fn prefill_gdn_block(req: &PrefillGdnReq) -> Option<Vec<f32>> {
    let hidden = req.hidden;
    let key_dim = req.heads_k * req.d;
    let value_dim = req.heads_v * req.d;
    let channels = key_dim * 2 + value_dim;
    // The recurrence kernel keeps one state row per simdgroup: 32 lanes x 4
    // columns, so d must be 128. Short chunks take the CPU path (the mm
    // pipeline refuses them anyway).
    if req.d != 128
        || req.wqkv.2 != channels
        || req.zgate.2 != value_dim
        || req.ssm_out.2 != hidden
        || req.m < MM_MIN_M
        || req.alpha.len() != hidden * req.heads_v * 4
        || req.beta.len() != hidden * req.heads_v * 4
        || req.conv1d.len() != channels * req.d_conv * 4
        || req.a.len() != req.heads_v
        || req.dt.len() != req.heads_v
        || req.ssm_norm.len() != req.d
    {
        return None;
    }
    let conv_span = req.conv_off + (req.d_conv - 1) * channels;
    let state_span = req.state_off + req.heads_v * req.d * req.d;
    if conv_span * 4 > req.ssm.len || state_span * 4 > req.ssm.len {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    let mats = [
        resolve(&gpu, req.wqkv.0, req.wqkv.1)?,
        resolve(&gpu, req.zgate.0, req.zgate.1)?,
        resolve_f32(&gpu, req.alpha, hidden)?,
        resolve_f32(&gpu, req.beta, hidden)?,
        resolve(&gpu, req.ssm_out.0, req.ssm_out.1)?,
    ];
    let mut states = Vec::with_capacity(5);
    for mat in &mats {
        states.push(
            gpu.pipeline(mm_kernel_for(mat.ty, req.m)?.0, 1, 1)?
                .to_owned(),
        );
    }
    let conv1d = {
        let addr = req.conv1d.as_ptr() as usize;
        if addr % 4 != 0 {
            return None;
        }
        let chunk = gpu.chunk_for(addr, req.conv1d.len())?;
        NormRef {
            chunk,
            off: (addr - gpu.chunks[chunk].start) as u64,
        }
    };

    struct FusionRefs {
        attn_norm: NormRef,
        ffn_norm: NormRef,
        router: Option<(MatRef, ComputePipelineState)>,
        n_expert: usize,
    }
    let fusion = match &req.fusion {
        Some(f) => {
            if gpu.pf_x_cap < req.m * hidden * 4 || gpu.pf_hs_cap < req.m * hidden * 4 {
                return None;
            }
            let attn_norm = resolve_norm(&gpu, f.attn_norm, hidden)?;
            let ffn_norm = resolve_norm(&gpu, f.ffn_norm, hidden)?;
            let router = if f.n_expert > 0 {
                let router = resolve_f32(&gpu, f.router, hidden)?;
                let router_state = gpu
                    .pipeline(mm_kernel_for(GgmlType::F32, req.m)?.0, 1, 1)?
                    .to_owned();
                Some((router, router_state))
            } else {
                None
            };
            Some(FusionRefs {
                attn_norm,
                ffn_norm,
                router,
                n_expert: f.n_expert,
            })
        }
        None => None,
    };
    // One-buffer note: same as prefill_attn_block - no decline here;
    // a non-onebuf layer seals the open segment at the encode site below.
    let norm_state = gpu.pipelines[&("norm_rows", 1, 1)].to_owned();
    let conv_state = gpu.pipelines[&("gdn_conv_batch", 1, 1)].to_owned();
    let step_state = gpu.pipelines[&("gdn_step_batch", 1, 1)].to_owned();
    let norm_out_state = gpu.pipelines[&("gdn_out_norm_batch", 1, 1)].to_owned();

    // y layout: raw qkv rows, conv output, z gate, alpha|beta, deltanet out,
    // the ssm_out projection, the router logits.
    let align = |n: usize| (n + 63) & !63;
    let qkv_at = 0usize;
    let qkc_at = qkv_at + align(req.m * channels);
    let z_at = qkc_at + align(req.m * channels);
    let ab_at = z_at + align(req.m * value_dim);
    let gy_at = ab_at + align(2 * req.m * req.heads_v);
    let out_at = gy_at + align(req.m * value_dim);
    let router_at = out_at + align(req.m * hidden);
    let n_expert = fusion.as_ref().map_or(0, |f| f.n_expert);
    let window_at = router_at + align(req.m * n_expert);
    gpu.ensure_arenas(4096, (window_at + align((req.d_conv - 1) * channels)) * 4);

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let queue = gpu.queue.clone();
        let mut cmd = queue.new_command_buffer();
        let split = std::env::var_os("ALLPAKA_PF_SPLIT").is_some();
        // One-buffer segment: lazily armed on the first eligible layer,
        // sealed by a layer that stays off it (see prefill_attn_block).
        let onebuf = fusion.is_some() && gpu_route() && pf_onebuf() && !split;
        if !onebuf {
            gpu.pf_obuf_seal();
        } else {
            gpu.pf_obuf_arm();
        }
        let defer = !onebuf && fusion.is_some() && gpu_route() && pf_defer() && !split;
        if defer {
            cmd.encode_wait_for_event(&gpu.pf_ev, gpu.pf_ev_val);
        }
        let mut enc = if onebuf {
            gpu.pf_obuf_cmd
                .as_deref()
                .expect("onebuf armed")
                .compute_command_encoder_with_dispatch_type(serial_dispatch())
        } else {
            cmd.compute_command_encoder_with_dispatch_type(serial_dispatch())
        };
        macro_rules! split_here {
            ($label:expr) => {
                if split {
                    use objc::{msg_send, sel, sel_impl};
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
                    let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
                    eprintln!("gdn {}: {:.3} ms", $label, (ge - gs) * 1e3);
                    cmd = queue.new_command_buffer();
                    enc = cmd.compute_command_encoder_with_dispatch_type(
                        metal::MTLDispatchType::Concurrent,
                    );
                }
            };
        }
        let y = &gpu.y_arena;
        let e = |off: usize| (off * 4) as u64;

        let matmul = |enc: &metal::ComputeCommandEncoderRef,
                      state: &ComputePipelineState,
                      mat: &MatRef,
                      n_in: usize,
                      n_out: usize,
                      x_buf: &Buffer,
                      x_off: usize,
                      y_off: usize| {
            enc.set_compute_pipeline_state(state);
            enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
            let m32 = req.m as u32;
            enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
            let (bm, bn) = mm_kernel_for(mat.ty, req.m).map_or((32, 32), |(_, bm, bn)| (bm, bn));
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (n_out as u64).div_ceil(bm as u64),
                    (req.m as u64).div_ceil(bn as u64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        };
        let norm_rows = |enc: &metal::ComputeCommandEncoderRef,
                         norm: &NormRef,
                         use_delta: u32,
                         delta_off: usize| {
            enc.set_compute_pipeline_state(&norm_state);
            enc.set_buffer(0, Some(&gpu.pf_x), 0);
            enc.set_buffer(1, Some(y), e(delta_off));
            enc.set_buffer(2, Some(&gpu.pf_hs), 0);
            enc.set_buffer(3, Some(&gpu.chunks[norm.chunk].buf), norm.off);
            let n = hidden as u32;
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
            enc.set_bytes(6, 4, &use_delta as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(req.m as u64, 1, 1), MTLSize::new(256, 1, 1));
        };

        let f = fusion.as_ref().expect("gdn prefill without fusion");
        if onebuf {
            // The previous layer's stages live on the same command buffer.
            enc.memory_barrier_with_resources(&[
                y,
                &gpu.pf_x,
                &gpu.pf_hs,
                &gpu.route_buf,
                &req.ssm.buf,
            ]);
        }
        // h = rmsnorm(xs) * attn_norm, then the four projections off it.
        norm_rows(enc, &f.attn_norm, 0, out_at);
        cstamp!(gpu, enc, "anorm");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
        }
        matmul(
            enc, &states[0], &mats[0], hidden, channels, &gpu.pf_hs, 0, qkv_at,
        );
        cstamp!(gpu, enc, "wqkv");
        matmul(
            enc, &states[1], &mats[1], hidden, value_dim, &gpu.pf_hs, 0, z_at,
        );
        matmul(
            enc,
            &states[2],
            &mats[2],
            hidden,
            req.heads_v,
            &gpu.pf_hs,
            0,
            ab_at,
        );
        matmul(
            enc,
            &states[3],
            &mats[3],
            hidden,
            req.heads_v,
            &gpu.pf_hs,
            0,
            ab_at + req.m * req.heads_v,
        );
        split_here!("qkv");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }

        // Depthwise conv over the chunk, updating the session window.
        enc.set_compute_pipeline_state(&conv_state);
        enc.set_buffer(0, Some(y), e(qkv_at));
        enc.set_buffer(1, Some(y), e(qkc_at));
        enc.set_buffer(2, Some(&req.ssm.buf), 0);
        enc.set_buffer(3, Some(&gpu.chunks[conv1d.chunk].buf), conv1d.off);
        let cargs = GdnConvBatchArgs {
            channels: channels as u32,
            d_conv: req.d_conv as u32,
            m: req.m as u32,
            pad0: 0,
            conv_off: req.conv_off as u64,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GdnConvBatchArgs>() as u64,
            &cargs as *const GdnConvBatchArgs as *const _,
        );
        let (slots_buf, slot_total) = match req.ssm_slots {
            Some((r, total)) => (r, total as u32),
            None => (req.ssm, 0u32),
        };
        enc.set_buffer(5, Some(&slots_buf.buf), 0);
        enc.set_bytes(6, 4, &slot_total as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((channels as u64).div_ceil(256), req.m as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        cstamp!(gpu, enc, "conv");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y, &req.ssm.buf]);
        }
        // The session window update, now that every row's read is done: the
        // last d_conv-1 RAW qkv rows become the next chunk's window.
        enc.set_compute_pipeline_state(&gpu.pipelines[&("copy_f32", 1, 1)]);
        let window_rows = req.d_conv - 1;
        let window_src = if req.m >= window_rows {
            qkv_at + (req.m - window_rows) * channels
        } else {
            let old = (window_rows - req.m) * channels;
            enc.set_buffer(
                0,
                Some(&req.ssm.buf),
                ((req.conv_off + req.m * channels) * 4) as u64,
            );
            enc.set_buffer(1, Some(y), e(window_at));
            let old32 = old as u32;
            enc.set_bytes(2, 4, &old32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((old as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.set_buffer(0, Some(y), e(qkv_at));
            enc.set_buffer(1, Some(y), e(window_at + old));
            let new32 = (req.m * channels) as u32;
            enc.set_bytes(2, 4, &new32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((new32 as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.memory_barrier_with_resources(&[y, &req.ssm.buf]);
            window_at
        };
        enc.set_buffer(0, Some(y), e(window_src));
        enc.set_buffer(1, Some(&req.ssm.buf), (req.conv_off * 4) as u64);
        let nwin = (window_rows * channels) as u32;
        enc.set_bytes(2, 4, &nwin as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((nwin as u64).div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y, &req.ssm.buf]);
        }
        split_here!("conv");

        // The recurrence over the chunk, state updated in place.
        enc.set_compute_pipeline_state(&step_state);
        enc.set_buffer(0, Some(&req.ssm.buf), 0);
        enc.set_buffer(1, Some(y), e(qkc_at));
        enc.set_buffer(2, Some(y), e(ab_at));
        enc.set_buffer(3, Some(y), e(ab_at + req.m * req.heads_v));
        let sargs = GdnStepBatchArgs {
            heads_k: req.heads_k as u32,
            heads_v: req.heads_v as u32,
            d: req.d as u32,
            key_dim: key_dim as u32,
            m: req.m as u32,
            eps: req.eps,
            state_off: req.state_off as u64,
        };
        enc.set_bytes(
            4,
            std::mem::size_of::<GdnStepBatchArgs>() as u64,
            &sargs as *const GdnStepBatchArgs as *const _,
        );
        enc.set_bytes(5, (req.a.len() * 4) as u64, req.a.as_ptr() as *const _);
        enc.set_bytes(6, (req.dt.len() * 4) as u64, req.dt.as_ptr() as *const _);
        enc.set_buffer(7, Some(y), e(gy_at));
        enc.set_buffer(8, Some(&slots_buf.buf), 0);
        enc.set_bytes(9, 4, &slot_total as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((req.d / 4) as u64, req.heads_v as u64, 1),
            MTLSize::new(32, 4, 1),
        );
        cstamp!(gpu, enc, "step");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y, &req.ssm.buf]);
        }
        split_here!("step");

        // Gated rmsnorm over the head outputs, times silu(z).
        enc.set_compute_pipeline_state(&norm_out_state);
        enc.set_buffer(0, Some(y), e(gy_at));
        enc.set_buffer(1, Some(y), e(z_at));
        enc.set_bytes(
            2,
            (req.ssm_norm.len() * 4) as u64,
            req.ssm_norm.as_ptr() as *const _,
        );
        let hv = req.heads_v as u32;
        let dd = req.d as u32;
        enc.set_bytes(3, 4, &hv as *const u32 as *const _);
        enc.set_bytes(4, 4, &dd as *const u32 as *const _);
        enc.set_bytes(5, 4, &req.eps as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(req.heads_v as u64, req.m as u64, 1),
            MTLSize::new(req.d as u64, 1, 1),
        );
        cstamp!(gpu, enc, "onorm");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        split_here!("onorm");

        matmul(
            enc, &states[4], &mats[4], value_dim, hidden, y, gy_at, out_at,
        );
        cstamp!(gpu, enc, "wo");
        split_here!("wo");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        // xs += ssm_out output, then the FFN norm and the router matmul.
        norm_rows(enc, &f.ffn_norm, 1, out_at);
        cstamp!(gpu, enc, "fnorm");
        if !no_barrier() {
            enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
        }
        if let Some((router, router_state)) = &f.router {
            matmul(
                enc,
                router_state,
                router,
                hidden,
                f.n_expert,
                &gpu.pf_hs,
                0,
                router_at,
            );
            cstamp!(gpu, enc, "rmm");
        }
        split_here!("ffnnorm_router");
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        if onebuf {
            // Nothing: the chunk commits once, in prefill_end.
        } else if defer {
            gpu.commit_chained(cmd);
        } else {
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
        }
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !onebuf {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        DISPATCHES.fetch_add(8, Ordering::Relaxed);

        // Same contract as the attention block: GPU routing leaves the
        // logits in y_arena and reports the offset; a CPU-routing caller
        // gets them downloaded.
        if fusion.is_some() && gpu_route() {
            PF_ROUTER_AT.store(router_at as u64, Ordering::Relaxed);
            return Vec::new();
        }
        let mut out = vec![0f32; req.m * n_expert];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const f32).add(router_at),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    });
    Some(out)
}

/// A whole prefill chunk's attention as ONE command buffer with ONE wait.

///
/// The requests are one per chunk position, each attending over the cache up
/// to its own position - causality is in `n_pos`, not in any mask. They
/// reuse the decode kernel unchanged: a dispatch per position costs ~1 us to
/// encode, against a CPU pass that was nearly half of prefill wall time.
///
/// All-or-nothing like the matmul batches: one undispatchable request sends
/// the whole chunk to the CPU path.
pub fn attend_batch(reqs: &[AttnReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    if reqs.iter().any(|r| !attn_shape_ok(r)) {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    let align = |n: usize| (n + 255) & !255;
    let mut x_need = 0usize;
    let mut y_need = 0usize;
    let mut offs = Vec::with_capacity(reqs.len());
    for r in reqs {
        offs.push((x_need, y_need));
        x_need += align(r.q.len() * 4);
        y_need += align(r.n_q_heads * r.head_dim * 4);
    }
    gpu.ensure_arenas(x_need, y_need);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut u8;
        for (r, (x_off, _)) in reqs.iter().zip(&offs) {
            std::ptr::copy_nonoverlapping(r.q.as_ptr() as *const u8, xp.add(*x_off), r.q.len() * 4);
        }
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Every position attends independently over an already-written
        // cache: no hazards, full concurrency.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        for (r, (x_off, y_off)) in reqs.iter().zip(&offs) {
            encode_attend_at(enc, &gpu, r, *x_off, *y_off);
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(reqs.len() as u64, Ordering::Relaxed);

        let yp = gpu.y_arena.contents() as *const u8;
        reqs.iter()
            .zip(&offs)
            .map(|(r, (_, y_off))| {
                let n = r.n_q_heads * r.head_dim;
                let mut v = vec![0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(yp.add(*y_off), v.as_mut_ptr() as *mut u8, n * 4);
                }
                v
            })
            .collect()
    }))
}

/// Attention for one token on the GPU. None means "do it on the CPU": no
/// device, a head width the kernel is not written for, or an empty cache.
pub fn attend(req: &AttnReq) -> Option<Vec<f32>> {
    if !attn_shape_ok(req) {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    let out_len = req.n_q_heads * req.head_dim;
    gpu.ensure_arenas(req.q.len() * 4, out_len * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(
            req.q.as_ptr() as *const u8,
            gpu.x_arena.contents() as *mut u8,
            req.q.len() * 4,
        );
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        // The same encoding the fused path uses; two copies of these
        // bindings is exactly how the half switch broke one of them.
        encode_attend(enc, &gpu, req, 0);
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(1, Ordering::Relaxed);

        let mut out = vec![0f32; out_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                gpu.y_arena.contents() as *const u8,
                out.as_mut_ptr() as *mut u8,
                out_len * 4,
            );
        }
        out
    }))
}

/// A prefill chunk's whole routed FFN - every expert group's gate, up,
/// swiglu and down - as ONE command buffer with SIX dispatches, however many
/// experts were hit. `x` holds the gathered rows (each group's tokens
/// contiguous), `groups` maps each to its expert and row range. Returns the
/// down outputs in the same row layout.
pub struct GroupedFfnReq<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    pub n_expert: usize,
    pub hidden: usize,
    pub ffn: usize,
    /// `[expert, row0, rows]` per hit group; row0 indexes into the flat row
    /// space. An array, not a tuple: this is copied verbatim into GPU memory
    /// and a tuple's layout is not guaranteed.
    pub groups: &'a [[u32; 3]],
    /// The UNGATHERED activations (`m` rows); `tok[row]` names each flat
    /// row's source token. The gather happens inside the gate/up kernels.
    pub x: &'a [f32],
    pub tok: &'a [u32],
    pub total_rows: usize,
    /// Fused prefill: read activations from the pf_hs buffer written by the
    /// preceding fused attention block, and scatter the weighted expert
    /// outputs into pf_x on the GPU (CSR hits per token) instead of
    /// returning them.
    pub fused: Option<GroupedCombine<'a>>,
    /// GLM's shared expert (fused path only): dense gate/up/down applied to
    /// every token. Its down rows land in the down region right after the
    /// `total_rows` expert rows, so the caller can reference them in the CSR
    /// hits with weight 1.0.
    pub shared: Option<GroupedShared<'a>>,
    /// GPU-side routing (fused path only): route_pick/scan/scatter build the
    /// group table, token gather and CSR hits on-device at the head of the
    /// FFN command buffer, from the router logits the attention block left
    /// in y_arena. `groups`/`tok`/CSR slices are ignored in this mode.
    pub route: Option<GroupedRoute<'a>>,
}

pub struct GroupedRoute<'a> {
    pub n_used: usize,
    /// Renormalise the winners to sum to 1 (`norm_topk_prob`).
    pub norm: bool,
    pub scale: f32,
    /// Selection bias (`+bias` shifts only the top-k choice).
    pub bias: Option<&'a [f32]>,
    /// Sigmoid gating (GLM) when true, softmax over all experts (Qwen3)
    /// when false. The softmax case is only exact without a selection bias
    /// and with `norm` set; the caller falls back to the CPU router
    /// otherwise.
    pub sigmoid: bool,
}

pub struct GroupedShared<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    /// Shared-expert FFN width (n_ff_exp * n_expert_shared).
    pub ffn: usize,
    /// qwen35moe: the shared expert's combine weight is sigmoid(gate . h)
    /// per token, an F32 [hidden] projection in the mmap. Its raw logits
    /// land in y; route_patch_shared_w folds them into the CSR weights.
    pub gate_out: Option<(GgmlType, &'a [u8])>,
}

pub struct GroupedCombine<'a> {
    /// CSR offsets, `m + 1` entries.
    pub tok_off: &'a [u32],
    /// Flat-row index and routing weight per hit.
    pub hit_row: &'a [u32],
    pub hit_w: &'a [f32],
    pub m: usize,
}

/// Deferred fused-prefill commits: layer buffers chain through an
/// MTLSharedEvent instead of one CPU wait per layer, so the CPU encodes
/// layer N+1 while the GPU still runs layer N. `ALLPAKA_PF_DEFER=0` reverts
/// to per-layer waits.
fn pf_defer() -> bool {
    crate::runtime::get().prefill_defer
}

/// The whole fused prefill chunk as ONE command buffer (sequential
/// encoders per stage, one commit in prefill_end). Default ON: measured
/// qwen3-30b pp480 1222-1230 vs 1196-1198 tok/s with the event chain
/// (pp1200: 1124 vs 1095) - pure buffer-start overhead removal, GPU
/// executing unchanged. `ALLPAKA_PF_ONEBUF=0` reverts to PF_DEFER chaining.
fn pf_onebuf() -> bool {
    crate::runtime::get().prefill_one_buffer
}

/// ALLPAKA_GPU_COUNTERS=1: per-dispatch GPUTimestamp sampling in the fused
/// prefill. This device exposes only the "timestamp" counter set, and only
/// atCommandBoundary sampling - per-dispatch compute-encoder samples assert
/// ("not supported on this device"), so the mode is a silent no-op here.
/// Kept as the probe harness for a device that supports it.
fn gpu_counters() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| std::env::var("ALLPAKA_GPU_COUNTERS").is_ok_and(|v| v == "1"))
}

/// Whether sampleCountersInBuffer works on compute encoders here
/// (MTLCounterSamplingPoint::atDispatchBoundary == 2).
fn gpu_counters_supported(device: &metal::DeviceRef) -> bool {
    use objc::{msg_send, sel, sel_impl};
    unsafe { msg_send![device, supportsCounterSampling: 2u64] }
}

/// Stamp capacity per prefill chunk (~16 dispatches x 48 layers x chunks).
const CSTAMP_CAP: usize = 1024;

/// The llama.cpp-structure q2_k matvec, `ALLPAKA_Q2_MV=1` to enable. OFF by
/// default: measured 95 GB/s against the word-load kernel's ~170 in
/// ffn-shaped decode (the word-load rewrite already beat llama's structure
/// for this format; the port only won for q3_k, whose old kernel still
/// assembled ushort pairs).
fn q2_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q2_MV").is_ok_and(|v| v == "1"))
}

/// Same switch for the q3_k llama-structure kernel: `ALLPAKA_Q3_MV=0` off.
fn q3_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q3_MV").map_or(true, |v| v != "0"))
}

fn q3_kernel() -> &'static str {
    if q3_mv() {
        "matvec_q3_k_mv"
    } else {
        "matvec_q3_k"
    }
}

/// The q4_k llama-structure matvec. Default ON: decode 30B 90 -> 102 tok/s,
/// 235B 22.2 -> 24.3 (q4_k attention weights). `ALLPAKA_Q4_MV=0` reverts.
fn q4_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q4_MV").map_or(true, |v| v != "0"))
}

fn q4_kernel() -> &'static str {
    if q4_mv() {
        "matvec_q4_k_mv"
    } else {
        "matvec_q4_k"
    }
}

/// The q6_k llama-structure matvec probe: `ALLPAKA_Q6_MV=1` to enable.
fn q6_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q6_MV").is_ok_and(|v| v == "1"))
}

fn q6_kernel() -> &'static str {
    if q6_mv() {
        "matvec_q6_k_mv"
    } else {
        "matvec_q6_k"
    }
}

/// Prefill attention over simdgroup MMA tiles (attend_mm, llama-style: K/V
/// read directly by the MMAs, online softmax). Default ON: standalone
/// 0.278 vs 0.742 ms/dispatch for attend_rows_t8 at m=480 (qwen3-30b
/// shape), exact to 1e-5. `ALLPAKA_ATTN_MM=0` reverts.
fn attend_mm() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| std::env::var("ALLPAKA_ATTN_MM").map_or(true, |v| v != "0"))
}

/// Prefill attention row tiling: `ALLPAKA_ATTN_T4=0` reverts to the
/// row-per-threadgroup kernel.
fn attend_rows_kernel() -> &'static str {
    static T4: OnceLock<usize> = OnceLock::new();
    match *T4.get_or_init(|| {
        std::env::var("ALLPAKA_ATTN_T4")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    }) {
        8 => "attend_rows_t8",
        0 => "attend_rows",
        _ => "attend_rows_t4",
    }
}

/// The llama-structure decode attention: `ALLPAKA_ATTN_MV=0` reverts to the
/// position-per-step kernel.
fn attend_kernel() -> &'static str {
    static MV: OnceLock<bool> = OnceLock::new();
    if *MV.get_or_init(|| std::env::var("ALLPAKA_ATTN_MV").is_ok_and(|v| v == "1")) {
        return "attend_mv";
    }
    static SG: OnceLock<usize> = OnceLock::new();
    match *SG.get_or_init(|| {
        std::env::var("ALLPAKA_ATTN_S8")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32)
    }) {
        32 => "attend_s32",
        16 => "attend_s16",
        4 | 0 => "attend",
        _ => "attend_s8",
    }
}

/// Threads per attend threadgroup, matching the kernel's simdgroup count.
fn attend_tg() -> u64 {
    match attend_kernel() {
        "attend_s32" => 1024,
        "attend_s16" => 512,
        "attend_s8" => 256,
        _ => 128,
    }
}

/// Probe-before-surgery for decode: `ALLPAKA_SKIP=qkv,attend,experts,router,head`
/// drops those dispatch groups from the whole-token buffer. Values become
/// garbage; the timing is the honest cost of what remains.
fn probe_skip(what: &str) -> bool {
    static SKIP: OnceLock<Vec<String>> = OnceLock::new();
    SKIP.get_or_init(|| {
        std::env::var("ALLPAKA_SKIP")
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default()
    })
    .iter()
    .any(|s| s == what)
}

/// The decode q2_k kernel: the interleaved dual-block variant under its env.
fn q2_kernel() -> &'static str {
    static ILV: OnceLock<bool> = OnceLock::new();
    if *ILV.get_or_init(|| std::env::var_os("ALLPAKA_Q2_ILV").is_some()) {
        "matvec_q2_k_ilv"
    } else if q2_mv() {
        "matvec_q2_k_mv"
    } else {
        "matvec_q2_k"
    }
}

/// Whether the llama-structure mm kernels are routed to (shared switch for
/// mm_kernel_for and mmid_kernel_for). Default ON: measured 235B prefill
/// 111 -> 136 tok/s, verify prefill PASS with a smaller diff than the old
/// kernels. `ALLPAKA_MM_LL=0` reverts.
/// GPU-side MoE routing for the fused prefill: route_pick/scan/scatter run
/// at the head of the FFN command buffer instead of a CPU readback + top-k.
/// `ALLPAKA_GPU_ROUTE=0` reverts to the CPU router.
fn gpu_route() -> bool {
    crate::runtime::get().gpu_route
}

/// The decode-side combine fold into the next residual norm
/// (combine_resnorm). OFF by default: measured SLOWER on M4 Max
/// (qwen3-30b decode 78.5 -> 75.0 tok/s, +20 ms GPU executing over 32
/// tokens) - the single-threadgroup fused kernel walks all expert slots'
/// down rows with one thread's latency chain, which costs more than the
/// dispatch and barrier it removes. `ALLPAKA_CFUSE=1` re-enables.
fn cfuse() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| std::env::var("ALLPAKA_CFUSE").is_ok_and(|v| v == "1"))
}

/// The decode gate+up dual dispatch: one INDEXED launch covers both
/// matrices (DUAL_GW function constant). OFF by default: qwen3-30b measured
/// 105.0 tok/s versus 113.7 for two concurrent dispatches on M4 Max.
/// `ALLPAKA_DECODE_GUFUSE=1` enables the diagnostic variant.
fn gufuse() -> bool {
    static G: OnceLock<bool> = OnceLock::new();
    *G.get_or_init(|| std::env::var("ALLPAKA_DECODE_GUFUSE").is_ok_and(|v| v == "1"))
}

/// The fused single-threadgroup router_topk in decode. OFF by default: the
/// fused kernel walks the experts serially inside one threadgroup and is
/// latency-bound - on qwen3-30b the parallel matvec_f32 + tiny top-k pair
/// (one extra dispatch and barrier, same buffer) measured 113.8 vs 79.2
/// tok/s decode (265 vs 388 ms GPU executing over 32 tokens).
/// `ALLPAKA_RTOPK=1` re-enables the fused kernel.
fn rtopk_fused() -> bool {
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| std::env::var("ALLPAKA_RTOPK").is_ok_and(|v| v == "1"))
}

/// The decode-side fold of the FFN residual norm into the router top-k
/// (resnorm_router). OFF by default: effect in the noise on M4 Max
/// (qwen3-30b decode, 385 vs 382-387 ms GPU executing over 32 tokens).
/// `ALLPAKA_RFUSE=1` enables.
fn rfuse() -> bool {
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| std::env::var("ALLPAKA_RFUSE").is_ok_and(|v| v == "1"))
}

/// K-step 64 for the LL mm kernels: `ALLPAKA_MM_K64=1` to enable.
fn mm_k64() -> bool {
    static K: OnceLock<bool> = OnceLock::new();
    *K.get_or_init(|| std::env::var("ALLPAKA_MM_K64").map_or(false, |v| v != "0"))
}

/// The software-pipelined double-buffer K-loop in the q4_k mm kernels
/// (mmllp_q4_k, mmllpg/ps_id_q4_k). Default ON. `ALLPAKA_MM_PIPE=0` reverts
/// to the two-barrier llama-structure loop.
fn mm_pipe() -> bool {
    crate::runtime::get().mm_pipeline
}

/// K-step 64 for the mmid (MoE) kernels only: `ALLPAKA_MMID_K64=1`. The FFN
/// K-loop runs 128 iterations at NK=32 with two threadgroup barriers each;
/// halving them costs shared memory but the MoE stages are latency-bound.
fn mmid_k64() -> bool {
    static K: OnceLock<bool> = OnceLock::new();
    *K.get_or_init(|| std::env::var("ALLPAKA_MMID_K64").map_or(false, |v| v != "0"))
}

fn mm_ll() -> bool {
    static LL: OnceLock<bool> = OnceLock::new();
    *LL.get_or_init(|| std::env::var("ALLPAKA_MM_LL").map_or(true, |v| v != "0"))
}

/// ALLPAKA_NO_BARRIER=1 skips in-buffer memory barriers (FFN path only) to
/// measure their cost. Timing probe only - results are racy garbage.
/// ALLPAKA_NO_STAGE_BARRIER=1 drops only the FFN stage-to-stage barriers
/// (route and combine barriers stay, so the mm stages still do full work
/// on valid tables; the down stage may read torn gate data - timing only).
fn no_stage_barrier() -> bool {
    static NB: OnceLock<bool> = OnceLock::new();
    *NB.get_or_init(|| std::env::var_os("ALLPAKA_NO_STAGE_BARRIER").is_some())
}

fn no_barrier() -> bool {
    static NB: OnceLock<bool> = OnceLock::new();
    *NB.get_or_init(|| {
        std::env::var_os("ALLPAKA_NO_BARRIER").is_some()
            || std::env::var_os("ALLPAKA_SERIAL").is_some()
    })
}

/// ALLPAKA_SERIAL=1: encode the prefill command buffers with
/// MTLDispatchType::Serial and drop the explicit barriers - serial dispatch
/// gives in-order execution (ordering without the full pipeline flush that
/// memory_barrier_with_resources pays). llama.cpp encodes this way.
fn serial_dispatch() -> metal::MTLDispatchType {
    static S: OnceLock<bool> = OnceLock::new();
    if *S.get_or_init(|| std::env::var_os("ALLPAKA_SERIAL").is_some()) {
        metal::MTLDispatchType::Serial
    } else {
        metal::MTLDispatchType::Concurrent
    }
}

/// Down-projection kernel with swiglu folded into B staging: gate and up
/// stay raw in y, the mm applies silu(g)*u while loading. q4_k only;
/// anything else keeps the standalone swiglu pass.
fn mmid_swiglu_kernel_for(ty: GgmlType) -> Option<(&'static str, usize)> {
    if !mm_ll() || mm_k64() {
        return None;
    }
    match ty {
        GgmlType::Q4K => Some((
            if mm_pipe() {
                "mmllps_id_q4_k"
            } else {
                "mmlls_id_q4_k"
            },
            64,
        )),
        _ => None,
    }
}

fn mmid_kernel_for(ty: GgmlType, gather: bool) -> Option<(&'static str, usize)> {
    if mm_ll() {
        let name = match (ty, gather, mm_k64() || mmid_k64()) {
            (GgmlType::Q8_0, false, false) => "mmll_id_q8_0",
            (GgmlType::Q2K, false, false) => "mmll_id_q2_k",
            (GgmlType::Q3K, false, false) => "mmll_id_q3_k",
            (GgmlType::Q4K, false, false) => "mmll_id_q4_k",
            (GgmlType::Q5K, false, false) => "mmll_id_q5_k",
            (GgmlType::Q6K, false, false) => "mmll_id_q6_k",
            (GgmlType::Q5_0, false, false) => "mmll_id_q5_0",
            (GgmlType::Q8_0, true, false) => "mmllg_id_q8_0",
            (GgmlType::Q2K, true, false) => "mmllg_id_q2_k",
            (GgmlType::Q3K, true, false) => "mmllg_id_q3_k",
            (GgmlType::Q4K, true, false) => {
                if mm_pipe() {
                    "mmllpg_id_q4_k"
                } else {
                    "mmllg_id_q4_k"
                }
            }
            (GgmlType::Q5K, true, false) => "mmllg_id_q5_k",
            (GgmlType::Q6K, true, false) => "mmllg_id_q6_k",
            (GgmlType::Q5_0, true, false) => "mmllg_id_q5_0",
            (GgmlType::Q8_0, false, true) => "mm64ll_id_q8_0",
            (GgmlType::Q2K, false, true) => "mm64ll_id_q2_k",
            (GgmlType::Q3K, false, true) => "mm64ll_id_q3_k",
            (GgmlType::Q4K, false, true) => "mm64ll_id_q4_k",
            (GgmlType::Q5K, false, true) => "mm64ll_id_q5_k",
            (GgmlType::Q6K, false, true) => "mm64ll_id_q6_k",
            (GgmlType::Q5_0, false, true) => "mm64ll_id_q5_0",
            (GgmlType::Q8_0, true, true) => "mm64llg_id_q8_0",
            (GgmlType::Q2K, true, true) => "mm64llg_id_q2_k",
            (GgmlType::Q3K, true, true) => "mm64llg_id_q3_k",
            (GgmlType::Q4K, true, true) => "mm64llg_id_q4_k",
            (GgmlType::Q5K, true, true) => "mm64llg_id_q5_k",
            (GgmlType::Q6K, true, true) => "mm64llg_id_q6_k",
            (GgmlType::Q5_0, true, true) => "mm64llg_id_q5_0",
            _ => return None,
        };
        return Some((name, 64));
    }
    if gather {
        // The pre-LL kernels have no gather form; the caller falls back to
        // the CPU-gathered path.
        return None;
    }
    Some((
        match ty {
            GgmlType::Q8_0 => "mmid_q8_0",
            GgmlType::Q2K => "mmid_q2_k",
            GgmlType::Q3K => "mmid_q3_k",
            GgmlType::Q4K => "mmid_q4_k",
            GgmlType::Q5K => "mmid_q5_k",
            GgmlType::Q6K => "mmid_q6_k",
            _ => return None,
        },
        32,
    ))
}

pub fn ffn_batch_grouped(req: &GroupedFfnReq) -> Option<Vec<f32>> {
    let route_mode = req.route.is_some();
    if route_mode {
        let r = req.route.as_ref()?;
        // route_pick holds 8 candidates per token and the fused buffers are
        // the only ones it can read logits from; anything else keeps CPU
        // routing. total_rows is exactly m * n_used in this mode.
        if r.n_used > 8 || req.fused.as_ref().map_or(0, |c| c.m) * r.n_used != req.total_rows {
            return None;
        }
    } else if req.groups.is_empty() || req.tok.len() != req.total_rows {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    // One-buffer note: a non-routing FFN (a leading Dense layer, GLM) no
    // longer declines the chunk - it seals the open one-buffer segment
    // at the encode site below and runs fused on an event-chained buffer
    // of its own; the next MoE layer re-arms.
    if req.shared.is_some() && req.fused.is_none() {
        // The shared expert reads pf_hs and lands in the combine region;
        // both exist only on the fused path. Callers on the CPU path run
        // the shared expert themselves.
        return None;
    }
    let mats = [
        resolve(&gpu, req.gate.0, req.gate.1)?,
        resolve(&gpu, req.up.0, req.up.1)?,
        resolve(&gpu, req.down.0, req.down.1)?,
    ];
    let m_fused = req.fused.as_ref().map_or(0, |c| c.m);
    let sh_mats = match &req.shared {
        Some(sh) => Some([
            resolve(&gpu, sh.gate.0, sh.gate.1)?,
            resolve(&gpu, sh.up.0, sh.up.1)?,
            resolve(&gpu, sh.down.0, sh.down.1)?,
        ]),
        None => None,
    };
    // qwen35moe's shared-expert gate projection (F32 [hidden]); the patch
    // kernel after route_scatter folds its sigmoid into the CSR weights.
    let sh_gate_out = match &req.shared {
        Some(sh) => match &sh.gate_out {
            Some((ty, bytes)) => {
                if *ty != GgmlType::F32 || bytes.len() != req.hidden * 4 {
                    return None;
                }
                let mat = resolve_f32(&gpu, bytes, req.hidden)?;
                let state = gpu
                    .pipeline(mm_kernel_for(GgmlType::F32, m_fused.max(MM_MIN_M))?.0, 1, 1)?
                    .to_owned();
                Some((mat, state))
            }
            None => None,
        },
        None => None,
    };
    let strides = [
        (req.gate.1.len() / req.n_expert) as u64,
        (req.up.1.len() / req.n_expert) as u64,
        (req.down.1.len() / req.n_expert) as u64,
    ];
    let mut states = Vec::with_capacity(3);
    let mut bms = [32u64; 3];
    let mut fused_swiglu = false;
    for (i, mat) in mats.iter().enumerate() {
        let (name, bm) = if i == 2 {
            match mmid_swiglu_kernel_for(mat.ty) {
                Some(nb) => {
                    fused_swiglu = true;
                    nb
                }
                None => mmid_kernel_for(mat.ty, false)?,
            }
        } else {
            mmid_kernel_for(mat.ty, true)?
        };
        bms[i] = bm as u64;
        states.push(gpu.pipeline(name, 1, 1)?.to_owned());
    }
    // Plain tile-matmul pipelines for the shared expert (no gather).
    let mut sh_states = Vec::with_capacity(3);
    if let Some(sh_mats) = &sh_mats {
        for mat in sh_mats {
            sh_states.push(
                gpu.pipeline(mm_kernel_for(mat.ty, m_fused)?.0, 1, 1)?
                    .to_owned(),
            );
        }
    }
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();

    // Fused gate+up as ONE dual-weight dispatch: same expert stride, 64-wide
    // tiles never straddle the gate/up boundary (ffn % 64 == 0), q4_k only.
    // Measured SLOWER than the two-dispatch path on M4 Max (317 vs 331 tok/s
    // pp480, GLM-4.5-Air) - one kernel doing 2x work per threadgroup loses to
    // two back-to-back dispatches. Kept behind ALLPAKA_DUAL=1 for reference.
    let dual_state = if fused_swiglu
        && strides[0] == strides[1]
        && req.ffn % 64 == 0
        && std::env::var_os("ALLPAKA_DUAL").is_some()
    {
        gpu.pipeline("mmllgd_id_q4_k", 1, 1).map(|p| p.to_owned())
    } else {
        None
    };
    let dual = dual_state.is_some();

    let (row_tiles, n_groups) = if route_mode {
        // Group sizes exist only on the GPU. The mmid kernels stride over
        // row tiles (r1 += grid.y * 32), so any grid covers every row;
        // launch ~2x the average to skip a storm of empty threadgroups
        // without looping in the common case.
        let avg = (req.total_rows / req.n_expert.max(1)) as u64;
        let _ = avg;
        // Worst-case grid: measured FASTER than a tight one - empty
        // threadgroups retire for free, while a small grid serialises the
        // hot experts through the striding loop. ALLPAKA_MMID_RTILES pins
        // the grid for A/B tests.
        let auto = (m_fused as u64).div_ceil(32).max(1);
        let rt = std::env::var("ALLPAKA_MMID_RTILES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(auto, |n| n.max(1));
        (rt, req.n_expert as u64)
    } else {
        let max_rows = req.groups.iter().map(|g| g[2]).max()? as usize;
        (max_rows.div_ceil(32) as u64, req.groups.len() as u64)
    };

    // Arenas: x rows then the group table in x; gate, up, down out in y.
    // GPU routing keeps the table, gather and CSR in route_buf instead.
    let align = |n: usize| (n + 63) & !63;
    let x_at = 0usize;
    let x_len = if req.fused.is_some() { 0 } else { req.x.len() };
    let table_at = x_at + align(x_len);
    let tok_at = table_at + align(if route_mode { 0 } else { req.groups.len() * 3 });
    let off_at = tok_at + align(if route_mode { 0 } else { req.total_rows });
    let (n_off, n_hits) = if route_mode {
        (0, 0)
    } else {
        req.fused
            .as_ref()
            .map_or((0, 0), |c| (c.tok_off.len(), c.hit_row.len()))
    };
    let hrow_at = off_at + align(n_off);
    let hw_at = hrow_at + align(n_hits);
    let x_total = hw_at + align(n_hits);
    let gate_at = 0usize;
    let up_at = gate_at + align(req.total_rows * req.ffn);
    // Dual dispatch writes gate and up interleaved per row (stride 2*ffn);
    // the separate up region exists only in the two-dispatch fallback.
    let out_at = if dual {
        gate_at + align(req.total_rows * 2 * req.ffn)
    } else {
        up_at + align(req.total_rows * req.ffn)
    };
    // The shared expert's down rows live right after the expert down rows so
    // the combine can reference them as ordinary hit rows.
    let sh_rows = if req.shared.is_some() { m_fused } else { 0 };
    let sh_ffn = req.shared.as_ref().map_or(0, |s| s.ffn);
    let shg_at = out_at + align((req.total_rows + sh_rows) * req.hidden);
    let shu_at = shg_at + align(m_fused * sh_ffn);
    // qwen35moe's shared-expert gate logits, one float per token.
    let shw_at = shu_at + align(m_fused * sh_ffn);
    let y_total = shw_at + align(m_fused);
    // GPU routing: size route_buf, then rescue the logits out of y_arena
    // BEFORE ensure_arenas can reallocate it (steady state binds y_arena
    // directly - zero copies). Counts/counters must start at zero.
    let rl = req
        .route
        .as_ref()
        .map(|r| route_layout(m_fused, req.n_expert, r.n_used, req.shared.is_some()));
    let mut logits_at: Option<(&'static str, usize)> = None;
    // One-buffer rescue: (old y_arena, logits offset, logits target, count).
    let mut obuf_rescue: Option<(Buffer, usize, usize, usize)> = None;
    if let Some(rl) = &rl {
        gpu.ensure_route(rl.total);
        let r_at = PF_ROUTER_AT.load(Ordering::Relaxed) as usize;
        if r_at == usize::MAX {
            return None; // the attention block did not leave logits behind
        }
        let n_logits = m_fused * req.n_expert;
        // Counts/counters self-clean on the GPU (route_scan zeroes them
        // after every layer), so staging once per allocation/bias is enough;
        // per-layer CPU writes would race the deferred buffers.
        let bptr = req
            .route
            .as_ref()
            .and_then(|r| r.bias)
            .map_or(0, |b| b.as_ptr() as usize);
        if gpu.route_staged != Some(bptr) {
            unsafe {
                let rb = gpu.route_buf.contents() as *mut f32;
                std::ptr::write_bytes(rb, 0, rl.table); // counts + counters
                if let Some(b) = req.route.as_ref().and_then(|r| r.bias) {
                    std::ptr::copy_nonoverlapping(b.as_ptr(), rb.add(rl.bias), req.n_expert);
                }
            }
            gpu.route_staged = Some(bptr);
        }
        if y_total * 4 > gpu.y_cap {
            if gpu.pf_obuf_cmd.is_some() && route_mode {
                // One-buffer mode: nothing has committed yet, a CPU
                // readback would see stale data. Rescue on the GPU instead,
                // encoded in order at the head of the FFN stages.
                obuf_rescue = Some((gpu.y_arena.clone(), r_at, rl.logits, n_logits));
                logits_at = Some(("route", rl.logits));
            } else {
                // The logits leave y_arena before it can grow; with deferred
                // commits the buffer that wrote them may still be in flight.
                gpu.prefill_drain();
                unsafe {
                    let rb = gpu.route_buf.contents() as *mut f32;
                    let src = (gpu.y_arena.contents() as *const f32).add(r_at);
                    std::ptr::copy_nonoverlapping(src, rb.add(rl.logits), n_logits);
                }
                logits_at = Some(("route", rl.logits));
            }
        }
        if logits_at.is_none() {
            logits_at = Some(("y", r_at));
        }
    }
    gpu.ensure_arenas(x_total * 4, y_total * 4);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut f32;
        if req.fused.is_none() {
            std::ptr::copy_nonoverlapping(req.x.as_ptr(), xp.add(x_at), req.x.len());
        }
        if !route_mode {
            std::ptr::copy_nonoverlapping(
                req.groups.as_ptr() as *const f32,
                xp.add(table_at),
                req.groups.len() * 3,
            ); // [u32;3] is layout-guaranteed; the arena type is just bytes.
            std::ptr::copy_nonoverlapping(
                req.tok.as_ptr() as *const f32,
                xp.add(tok_at),
                req.tok.len(),
            );
            if let Some(c) = &req.fused {
                std::ptr::copy_nonoverlapping(
                    c.tok_off.as_ptr() as *const f32,
                    xp.add(off_at),
                    c.tok_off.len(),
                );
                std::ptr::copy_nonoverlapping(
                    c.hit_row.as_ptr() as *const f32,
                    xp.add(hrow_at),
                    c.hit_row.len(),
                );
                std::ptr::copy_nonoverlapping(c.hit_w.as_ptr(), xp.add(hw_at), c.hit_w.len());
            }
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        // ALLPAKA_FFN_SPLIT=1: commit after every stage and print its GPU
        // time. Debug-only; serialises the stages, so wall time degrades.
        let split = std::env::var_os("ALLPAKA_FFN_SPLIT").is_some();
        // One-buffer chunk (ALLPAKA_PF_ONEBUF=1, default): encode into the
        // chunk's shared command buffer, lazily armed by the first
        // eligible layer; the commit happens once in prefill_end. A layer
        // that stays off the shared buffer (CPU routing, FFN_SPLIT) seals
        // the open segment first, so its own buffer stays ordered after
        // the work already encoded there.
        let onebuf = route_mode && pf_onebuf() && !split;
        if !onebuf {
            gpu.pf_obuf_seal();
        } else {
            gpu.pf_obuf_arm();
        }
        // Deferred commit: chain behind the previous layer's buffer through
        // the shared event instead of a CPU wait. Any fused-path buffer
        // qualifies - it reads nothing back (the combine lands in pf_x).
        // FFN_SPLIT keeps the old waits.
        let defer = !onebuf && req.fused.is_some() && pf_defer() && !split;
        // Cloned queue handle: command buffers borrow it, leaving `gpu`
        // free for the deferred commit bookkeeping.
        let queue = gpu.queue.clone();
        let cmd0 = queue.new_command_buffer();
        let mut cmd = cmd0;
        if defer {
            cmd.encode_wait_for_event(&gpu.pf_ev, gpu.pf_ev_val);
        }
        let mut enc = if onebuf {
            gpu.pf_obuf_cmd
                .as_deref()
                .expect("onebuf armed")
                .compute_command_encoder_with_dispatch_type(serial_dispatch())
        } else {
            cmd.compute_command_encoder_with_dispatch_type(serial_dispatch())
        };
        macro_rules! split_here {
            ($label:expr) => {
                if split {
                    use objc::{msg_send, sel, sel_impl};
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
                    let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
                    eprintln!("ffn {}: {:.3} ms", $label, (ge - gs) * 1e3);
                    cmd = queue.new_command_buffer();
                    enc = cmd.compute_command_encoder_with_dispatch_type(
                        metal::MTLDispatchType::Concurrent,
                    );
                }
            };
        }
        let y = &gpu.y_arena;
        let e = |off: usize| (off * 4) as u64;
        if onebuf {
            // The attention stages of this layer live on the same command
            // buffer: order the router logits (y), the normed activations
            // (pf_hs) and the residual stream (pf_x) against the FFN stages.
            match &obuf_rescue {
                Some((oy, ..)) => enc.memory_barrier_with_resources(&[
                    y,
                    &gpu.pf_x,
                    &gpu.pf_hs,
                    &gpu.route_buf,
                    oy,
                ]),
                None => {
                    enc.memory_barrier_with_resources(&[y, &gpu.pf_x, &gpu.pf_hs, &gpu.route_buf])
                }
            }
            if let Some((oy, r_at, lg_off, n_logits)) = &obuf_rescue {
                // GPU-side logits rescue out of the pre-growth y_arena.
                enc.set_compute_pipeline_state(&gpu.pipelines[&("copy_f32", 1, 1)]);
                enc.set_buffer(0, Some(oy), (*r_at * 4) as u64);
                enc.set_buffer(1, Some(&gpu.route_buf), (*lg_off * 4) as u64);
                let n32 = *n_logits as u32;
                enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new((*n_logits as u64).div_ceil(256), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.memory_barrier_with_resources(&[&gpu.route_buf]);
            }
        }
        // Group table, gather and CSR: route_buf on the GPU-routing path,
        // x_arena otherwise.
        let (tab_buf, tab_off, tk_buf, tk_off) = match &rl {
            Some(rl) => (&gpu.route_buf, rl.table, &gpu.route_buf, rl.tok),
            None => (&gpu.x_arena, table_at, &gpu.x_arena, tok_at),
        };
        let (off_buf, off_off, hr_buf, hr_off, hw_buf, hw_off) = match &rl {
            Some(rl) => (
                &gpu.route_buf,
                rl.tok_off,
                &gpu.route_buf,
                rl.hit_row,
                &gpu.route_buf,
                rl.hit_w,
            ),
            None => (
                &gpu.x_arena,
                off_at,
                &gpu.x_arena,
                hrow_at,
                &gpu.x_arena,
                hw_at,
            ),
        };
        let stage = |enc: &metal::ComputeCommandEncoderRef,
                     si: usize,
                     n_in: usize,
                     n_out: usize,
                     x_buf: &Buffer,
                     x_off: usize,
                     y_off: usize,
                     sw_up: Option<(usize, usize)>| {
            enc.set_compute_pipeline_state(&states[si]);
            enc.set_buffer(0, Some(&gpu.chunks[mats[si].chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mats[si].w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(tab_buf), e(tab_off));
            enc.set_bytes(7, 8, &strides[si] as *const u64 as *const _);
            enc.set_buffer(8, Some(tk_buf), e(tk_off));
            if let Some((uo, bstr)) = sw_up {
                enc.set_buffer(9, Some(y), e(uo));
                let bs = bstr as u32;
                enc.set_bytes(12, 4, &bs as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64).div_ceil(bms[si]), row_tiles, n_groups),
                MTLSize::new(128, 1, 1),
            );
        };

        // GPU-side routing: pick top-k per token, prefix-sum the per-expert
        // counts into the group table, scatter token ids and CSR hits.
        if let (Some(r), Some(rl), Some((lg_tag, lg_off))) = (req.route.as_ref(), &rl, logits_at) {
            #[repr(C)]
            struct RouteParams {
                m: u32,
                n_expert: u32,
                n_used: u32,
                norm: u32,
                has_bias: u32,
                sigmoid: u32,
                shared: u32,
                scale: f32,
                total_rows: u32,
            }
            let params = RouteParams {
                m: m_fused as u32,
                n_expert: req.n_expert as u32,
                n_used: r.n_used as u32,
                norm: r.norm as u32,
                has_bias: r.bias.is_some() as u32,
                sigmoid: r.sigmoid as u32,
                shared: req.shared.is_some() as u32,
                scale: r.scale,
                total_rows: req.total_rows as u32,
            };
            let pp = &params as *const RouteParams as *const _;
            let psz = std::mem::size_of::<RouteParams>() as u64;
            let rb = &gpu.route_buf;
            let tgs = (m_fused as u64).div_ceil(256);
            let lg_buf = if lg_tag == "route" { rb } else { y };
            enc.set_compute_pipeline_state(&gpu.pipelines[&("route_pick", 1, 1)]);
            enc.set_buffer(0, Some(lg_buf), e(lg_off));
            enc.set_buffer(1, Some(rb), e(rl.bias));
            enc.set_buffer(2, Some(rb), e(rl.counts));
            enc.set_buffer(3, Some(rb), e(rl.picks));
            enc.set_buffer(4, Some(rb), e(rl.pickw));
            enc.set_bytes(5, psz, pp);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
            cstamp!(gpu, enc, "rp");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[rb]);
            }
            enc.set_compute_pipeline_state(&gpu.pipelines[&("route_scan", 1, 1)]);
            enc.set_buffer(0, Some(rb), e(rl.counts));
            enc.set_buffer(1, Some(rb), e(rl.table));
            enc.set_buffer(2, Some(rb), e(rl.counters));
            let ne32 = req.n_expert as u32;
            enc.set_bytes(3, 4, &ne32 as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
            cstamp!(gpu, enc, "rs");
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[rb]);
            }
            enc.set_compute_pipeline_state(&gpu.pipelines[&("route_scatter", 1, 1)]);
            enc.set_buffer(0, Some(rb), e(rl.picks));
            enc.set_buffer(1, Some(rb), e(rl.pickw));
            enc.set_buffer(2, Some(rb), e(rl.table));
            enc.set_buffer(3, Some(rb), e(rl.counters));
            enc.set_buffer(4, Some(rb), e(rl.tok));
            enc.set_buffer(5, Some(rb), e(rl.hit_row));
            enc.set_buffer(6, Some(rb), e(rl.hit_w));
            enc.set_buffer(7, Some(rb), e(rl.tok_off));
            enc.set_bytes(8, psz, pp);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
            cstamp!(gpu, enc, "rc");
            // The gate stage reads the table/tok from route_buf and route_pick
            // read the logits from y - order both against what follows.
            if !no_barrier() {
                enc.memory_barrier_with_resources(&[rb, y]);
            }
        }

        let (xb, xo): (&Buffer, usize) = if req.fused.is_some() {
            (&gpu.pf_hs, 0)
        } else {
            (&gpu.x_arena, x_at)
        };
        split_here!("route");
        // The shared expert's plain tile matmuls (borrowed by the phases
        // below): they depend only on pf_hs, so they fill the same
        // barrier-free windows as the expert gate/up and down stages.
        let mm = |enc: &metal::ComputeCommandEncoderRef,
                  si: usize,
                  n_in: usize,
                  n_out: usize,
                  x_buf: &Buffer,
                  x_off: usize,
                  y_off: usize| {
            let sh_mats = sh_mats.as_ref().expect("shared mm without mats");
            enc.set_compute_pipeline_state(&sh_states[si]);
            enc.set_buffer(0, Some(&gpu.chunks[sh_mats[si].chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &sh_mats[si].w_off as *const u64 as *const _);
            let m32 = m_fused as u32;
            enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
            let (bm, bn) =
                mm_kernel_for(sh_mats[si].ty, m_fused).map_or((32, 32), |(_, bm, bn)| (bm, bn));
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (n_out as u64).div_ceil(bm as u64),
                    (m_fused as u64).div_ceil(bn as u64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        };
        // Phase 1: everything reading pf_hs - expert gate/up plus the shared
        // expert's gate/up. No barriers between them, they run concurrently.
        if dual {
            let ds = dual_state.as_ref().unwrap();
            enc.set_compute_pipeline_state(ds);
            enc.set_buffer(0, Some(&gpu.chunks[mats[0].chunk].buf), 0);
            enc.set_buffer(1, Some(xb), e(xo));
            enc.set_buffer(2, Some(y), e(gate_at));
            let a = req.hidden as u32;
            let b = (req.ffn * 2) as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mats[0].w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(tab_buf), e(tab_off));
            enc.set_bytes(7, 8, &strides[0] as *const u64 as *const _);
            enc.set_buffer(8, Some(tk_buf), e(tk_off));
            enc.set_buffer(10, Some(&gpu.chunks[mats[1].chunk].buf), 0);
            enc.set_bytes(11, 8, &mats[1].w_off as *const u64 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(((req.ffn * 2) as u64).div_ceil(64), row_tiles, n_groups),
                MTLSize::new(128, 1, 1),
            );
        } else {
            stage(enc, 0, req.hidden, req.ffn, xb, xo, gate_at, None);
            cstamp!(gpu, enc, "g");
            stage(enc, 1, req.hidden, req.ffn, xb, xo, up_at, None);
            cstamp!(gpu, enc, "u");
        }
        split_here!("gate");
        if let Some(sh) = &req.shared {
            mm(enc, 0, req.hidden, sh.ffn, &gpu.pf_hs, 0, shg_at);
            mm(enc, 1, req.hidden, sh.ffn, &gpu.pf_hs, 0, shu_at);
            // qwen35moe: the shared expert's gate projection, [m] logits.
            if let Some((sgmat, sgstate)) = &sh_gate_out {
                enc.set_compute_pipeline_state(sgstate);
                enc.set_buffer(0, Some(&gpu.chunks[sgmat.chunk].buf), 0);
                enc.set_buffer(1, Some(&gpu.pf_hs), 0);
                enc.set_buffer(2, Some(y), e(shw_at));
                let a = req.hidden as u32;
                let b = 1u32;
                enc.set_bytes(3, 4, &a as *const u32 as *const _);
                enc.set_bytes(4, 4, &b as *const u32 as *const _);
                enc.set_bytes(5, 8, &sgmat.w_off as *const u64 as *const _);
                let m32 = m_fused as u32;
                enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
                let (_bm, bn) =
                    mm_kernel_for(GgmlType::F32, m_fused).map_or((32, 32), |(_, bm, bn)| (bm, bn));
                enc.dispatch_thread_groups(
                    MTLSize::new(1, (m_fused as u64).div_ceil(bn as u64), 1),
                    MTLSize::new(128, 1, 1),
                );
            }
        }
        if !no_barrier() && !no_stage_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        split_here!("up");
        // Phase 2: the down stage (fused swiglu in its B staging, or the
        // standalone pass first) plus the shared expert's swiglu.
        if !fused_swiglu {
            enc.set_compute_pipeline_state(&swiglu_state);
            enc.set_buffer(0, Some(y), e(gate_at));
            enc.set_buffer(1, Some(y), e(up_at));
            let n32 = (req.total_rows * req.ffn) as u32;
            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            cstamp!(gpu, enc, "sw");
            if !no_barrier() && !no_stage_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
        }
        split_here!("swiglu");
        // The fused down kernel applies swiglu while staging B, so it reads
        // the raw gate and up regions; the plain kernel reads post-swiglu.
        stage(
            enc,
            2,
            req.ffn,
            req.hidden,
            &gpu.y_arena,
            gate_at,
            out_at,
            if dual {
                Some((gate_at + req.ffn, 2 * req.ffn))
            } else if fused_swiglu {
                Some((up_at, req.ffn))
            } else {
                None
            },
        );
        cstamp!(gpu, enc, "dn");
        if let Some(sh) = &req.shared {
            enc.set_compute_pipeline_state(&swiglu_state);
            enc.set_buffer(0, Some(y), e(shg_at));
            enc.set_buffer(1, Some(y), e(shu_at));
            let n32 = (m_fused * sh.ffn) as u32;
            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
        if !no_barrier() && !no_stage_barrier() {
            enc.memory_barrier_with_resources(&[y]);
        }
        split_here!("down");
        // Phase 3: the shared expert's down, into the rows right after the
        // expert rows.
        if let Some(sh) = &req.shared {
            mm(
                enc,
                2,
                sh.ffn,
                req.hidden,
                &gpu.y_arena,
                shg_at,
                out_at + req.total_rows * req.hidden,
            );
        }
        split_here!("shared");
        if let Some(c) = &req.fused {
            if !no_barrier() && !no_stage_barrier() {
                enc.memory_barrier_with_resources(&[y]);
            }
            // qwen35moe: fold sigmoid(shared gate logit) into the shared
            // slot's CSR weight (route_scatter wrote a constant 1). Runs
            // after the phase-1 gate projection; the barrier above covers it.
            if sh_gate_out.is_some() && route_mode {
                #[repr(C)]
                struct RouteParams {
                    m: u32,
                    n_expert: u32,
                    n_used: u32,
                    norm: u32,
                    has_bias: u32,
                    sigmoid: u32,
                    shared: u32,
                    scale: f32,
                    total_rows: u32,
                }
                let r = req.route.as_ref().expect("route mode");
                let params = RouteParams {
                    m: m_fused as u32,
                    n_expert: req.n_expert as u32,
                    n_used: r.n_used as u32,
                    norm: r.norm as u32,
                    has_bias: r.bias.is_some() as u32,
                    sigmoid: r.sigmoid as u32,
                    shared: req.shared.is_some() as u32,
                    scale: r.scale,
                    total_rows: req.total_rows as u32,
                };
                let rl = rl.as_ref().expect("route mode layout");
                enc.set_compute_pipeline_state(&gpu.pipelines[&("route_patch_shared_w", 1, 1)]);
                enc.set_buffer(0, Some(&gpu.route_buf), e(rl.hit_w));
                enc.set_buffer(1, Some(y), e(shw_at));
                enc.set_bytes(
                    2,
                    std::mem::size_of::<RouteParams>() as u64,
                    &params as *const RouteParams as *const _,
                );
                enc.dispatch_thread_groups(
                    MTLSize::new((m_fused as u64).div_ceil(32), 1, 1),
                    MTLSize::new(32, 1, 1),
                );
                if !no_barrier() {
                    enc.memory_barrier_with_resources(&[&gpu.route_buf]);
                }
            }
            enc.set_compute_pipeline_state(&gpu.pipelines[&("combine_rows", 1, 1)]);
            enc.set_buffer(0, Some(&gpu.pf_x), 0);
            enc.set_buffer(1, Some(y), e(out_at));
            enc.set_buffer(2, Some(off_buf), e(off_off));
            enc.set_buffer(3, Some(hr_buf), e(hr_off));
            enc.set_buffer(4, Some(hw_buf), e(hw_off));
            let h32 = req.hidden as u32;
            enc.set_bytes(5, 4, &h32 as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(c.m as u64, 1, 1), MTLSize::new(256, 1, 1));
            cstamp!(gpu, enc, "cb");
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        if onebuf {
            // Nothing: the chunk commits once, in prefill_end.
        } else if defer {
            gpu.commit_chained(cmd);
        } else {
            cmd.commit();
            cmd.wait_until_completed();
            note_gpu_times(cmd);
        }
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !defer && !onebuf && std::env::var_os("ALLPAKA_FFN_TIME").is_some() {
            use objc::{msg_send, sel, sel_impl};
            let gs: f64 = unsafe { msg_send![cmd, GPUStartTime] };
            let ge: f64 = unsafe { msg_send![cmd, GPUEndTime] };
            eprintln!("ffnbuf: {:.3} ms", (ge - gs) * 1e3);
        }
        if !onebuf {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        DISPATCHES.fetch_add(4, Ordering::Relaxed);

        // Fused: the combine already landed in pf_x; skip the download.
        let out_rows = if req.fused.is_some() {
            0
        } else {
            req.total_rows
        };
        let mut out = vec![0f32; out_rows * req.hidden];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const f32).add(out_at),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    });
    Some(out)
}

/// One FFN inside a fused batch: `y = swiglu(x·Gᵀ, x·Uᵀ) · Dᵀ` for `m`
/// activation rows, without the CPU touching the intermediates.
pub struct FfnReq<'a> {
    pub gate_ty: GgmlType,
    pub gate_w: &'a [u8],
    pub up_ty: GgmlType,
    pub up_w: &'a [u8],
    pub down_ty: GgmlType,
    pub down_w: &'a [u8],
    pub hidden: usize,
    pub ffn: usize,
    pub x: &'a [f32],
    pub m: usize,
}

/// A validated weight reference: kernel name plus window-relative offset.
struct MatRef {
    kernel: &'static str,
    ty: GgmlType,
    chunk: usize,
    w_off: u64,
}

fn resolve(gpu: &Gpu, ty: GgmlType, w: &[u8]) -> Option<MatRef> {
    let kernel = match ty {
        GgmlType::Q5_0 => "matvec_q5_0",
        GgmlType::Q8_0 => "matvec_q8_0",
        GgmlType::Q2K => q2_kernel(),
        GgmlType::Q3K => q3_kernel(),
        GgmlType::Q4K => q4_kernel(),
        GgmlType::Q5K => "matvec_q5_k",
        GgmlType::Q6K => q6_kernel(),
        _ => return None,
    };
    let addr = w.as_ptr() as usize;
    let chunk = gpu.chunk_for(addr, w.len())?;
    Some(MatRef {
        kernel,
        ty,
        chunk,
        w_off: (addr - gpu.chunks[chunk].start) as u64,
    })
}

/// Encode one logical matmul as tiled sub-dispatches. Offsets into the
/// x/y buffers are per sub-batch; row strides keep them 16-byte aligned for
/// the kernels' float4 loads (n_in and n_out are whole quant blocks).
///
/// `states` holds one pipeline per tile, in `tiles_for(m)` order; they are
/// specialised ahead of encoding because that needs `&mut Gpu`.
#[allow(clippy::too_many_arguments)]
fn encode_matvec(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    mat: &MatRef,
    states: &[ComputePipelineState],
    n_in: usize,
    n_out: usize,
    x_buf: &Buffer,
    x_off: usize,
    y_buf: &Buffer,
    y_off: usize,
    m: usize,
) -> u64 {
    let mut dispatched = 0u64;
    if m >= MM_MIN_M {
        // One tile-matmul dispatch for the whole batch; `states` holds the
        // single mm pipeline the caller resolved under the same condition.
        enc.set_compute_pipeline_state(&states[0]);
        enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
        enc.set_buffer(1, Some(x_buf), x_off as u64);
        enc.set_buffer(2, Some(y_buf), y_off as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        let m32 = m as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
        enc.set_bytes(6, 4, &m32 as *const u32 as *const _);
        let (bm, bn) = mm_kernel_for(mat.ty, m).map_or((32, 32), |(_, bm, bn)| (bm, bn));
        enc.dispatch_thread_groups(
            MTLSize::new(
                (n_out as u64).div_ceil(bm as u64),
                (m as u64).div_ceil(bn as u64),
                1,
            ),
            MTLSize::new(128, 1, 1),
        );
        return 1;
    }
    for ((start, _), state) in tiles_for(m).into_iter().zip(states) {
        enc.set_compute_pipeline_state(state);
        enc.set_buffer(0, Some(&gpu.chunks[mat.chunk].buf), 0);
        enc.set_buffer(1, Some(x_buf), (x_off + start * n_in * 4) as u64);
        enc.set_buffer(2, Some(y_buf), (y_off + start * n_out * 4) as u64);
        let n_in32 = n_in as u32;
        let n_out32 = n_out as u32;
        enc.set_bytes(3, 4, &n_in32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &n_out32 as *const u32 as *const _);
        enc.set_bytes(5, 8, &mat.w_off as *const u64 as *const _);
        let per_group = 128u64;
        let total_threads = n_out as u64 * lanes_per_row(mat.ty, n_in) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(total_threads.div_ceil(per_group), 1, 1),
            MTLSize::new(per_group, 1, 1),
        );
        dispatched += 1;
    }
    dispatched
}

/// Run whole FFNs (gate, up, swiglu, down) as ONE command buffer with ONE
/// wait. The compute encoder is serial, so each dispatch sees the previous
/// one's writes; the intermediates never leave the GPU arena.
///
/// Per decoded token on a MoE this replaces two waits (gate+up batch, down
/// batch) with one, and removes the CPU swiglu pass between them.
///
/// All-or-nothing, like matvec_batch: any request without kernels for all
/// three weights declines the whole batch to the CPU path.
pub fn ffn_batch(reqs: &[FfnReq]) -> Option<Vec<Vec<f32>>> {
    if reqs.is_empty() {
        return Some(Vec::new());
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;

    let mut mats = Vec::with_capacity(reqs.len());
    for r in reqs {
        if r.x.len() != r.m * r.hidden || r.hidden % 4 != 0 || r.ffn % 4 != 0 {
            return None;
        }
        mats.push((
            resolve(&gpu, r.gate_ty, r.gate_w)?,
            resolve(&gpu, r.up_ty, r.up_w)?,
            resolve(&gpu, r.down_ty, r.down_w)?,
        ));
    }

    // Arena plan per request: x in the x arena; gate, up and the final out
    // rows in the y arena, each 256-aligned at the request level.
    struct Plan {
        x_off: usize,
        gate_off: usize,
        up_off: usize,
        out_off: usize,
    }
    let align = |n: usize| (n + 255) & !255;
    let mut x_need = 0usize;
    let mut y_need = 0usize;
    let mut plans = Vec::with_capacity(reqs.len());
    for r in reqs {
        let p = Plan {
            x_off: x_need,
            gate_off: y_need,
            up_off: y_need + align(r.m * r.ffn * 4),
            out_off: y_need + 2 * align(r.m * r.ffn * 4),
        };
        x_need += align(r.m * r.hidden * 4);
        y_need = p.out_off + align(r.m * r.hidden * 4);
        plans.push(p);
    }
    gpu.ensure_arenas(x_need, y_need);

    // One pipeline per (matrix, tile), specialised before encoding starts;
    // see encode_matvec. Gate and up share a shape, down has its own.
    let mut states = Vec::with_capacity(reqs.len());
    for (r, (gate, up, down)) in reqs.iter().zip(&mats) {
        let mut per_mat = Vec::with_capacity(3);
        for (mat, n_in) in [(gate, r.hidden), (up, r.hidden), (down, r.ffn)] {
            // Same routing as matvec_batch: a big batch is one tile-matmul
            // dispatch, a small one is exact matvec tiles.
            if r.m >= MM_MIN_M {
                per_mat.push(vec![gpu
                    .pipeline(mm_kernel_for(mat.ty, r.m)?.0, 1, 1)?
                    .to_owned()]);
            } else {
                let lpr = lanes_per_row(mat.ty, n_in);
                let mut tiles = Vec::new();
                for (_, rows) in tiles_for(r.m) {
                    tiles.push(gpu.pipeline(mat.kernel, rows, lpr)?.to_owned());
                }
                per_mat.push(tiles);
            }
        }
        states.push(per_mat);
    }

    unsafe {
        let xp = gpu.x_arena.contents() as *mut u8;
        for (r, p) in reqs.iter().zip(&plans) {
            std::ptr::copy_nonoverlapping(
                r.x.as_ptr() as *const u8,
                xp.add(p.x_off),
                r.x.len() * 4,
            );
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Concurrent encoder in three barriered waves: every expert's gate
        // AND up together, every swiglu, every down. One expert's matvec is
        // far too small to fill the GPU; eight experts' worth at once is
        // what the serial encoder was quietly forbidding.
        let enc =
            cmd.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let mut dispatched = 0u64;
        for (((r, p), (gate, up, _)), st) in reqs.iter().zip(&plans).zip(&mats).zip(&states) {
            dispatched += encode_matvec(
                enc,
                &gpu,
                gate,
                &st[0],
                r.hidden,
                r.ffn,
                &gpu.x_arena,
                p.x_off,
                &gpu.y_arena,
                p.gate_off,
                r.m,
            );
            dispatched += encode_matvec(
                enc,
                &gpu,
                up,
                &st[1],
                r.hidden,
                r.ffn,
                &gpu.x_arena,
                p.x_off,
                &gpu.y_arena,
                p.up_off,
                r.m,
            );
        }
        enc.memory_barrier_with_resources(&[&gpu.y_arena]);
        for (r, p) in reqs.iter().zip(&plans) {
            // swiglu(gate, up) in place over m*ffn elements.
            enc.set_compute_pipeline_state(&gpu.pipelines[&("swiglu", 1, 1)]);
            enc.set_buffer(0, Some(&gpu.y_arena), p.gate_off as u64);
            enc.set_buffer(1, Some(&gpu.y_arena), p.up_off as u64);
            let n32 = (r.m * r.ffn) as u32;
            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
            let per_group = 256u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n32 as u64).div_ceil(per_group), 1, 1),
                MTLSize::new(per_group, 1, 1),
            );
            dispatched += 1;
        }
        enc.memory_barrier_with_resources(&[&gpu.y_arena]);
        for (((r, p), (_, _, down)), st) in reqs.iter().zip(&plans).zip(&mats).zip(&states) {
            dispatched += encode_matvec(
                enc,
                &gpu,
                down,
                &st[2],
                r.ffn,
                r.hidden,
                &gpu.y_arena,
                p.gate_off,
                &gpu.y_arena,
                p.out_off,
                r.m,
            );
        }
        enc.end_encoding();
        ENCODE_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_wait = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        note_gpu_times(cmd);
        WAIT_NS.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        DISPATCHES.fetch_add(dispatched, Ordering::Relaxed);

        let yp = gpu.y_arena.contents() as *const u8;
        reqs.iter()
            .zip(&plans)
            .map(|(r, p)| {
                let mut v = vec![0f32; r.m * r.hidden];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        yp.add(p.out_off),
                        v.as_mut_ptr() as *mut u8,
                        v.len() * 4,
                    );
                }
                v
            })
            .collect()
    });
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{tiles_for, TILES};

    #[test]
    fn tiles_cover_every_row_exactly_once() {
        for m in 0..40 {
            let tiles = tiles_for(m);
            let mut at = 0;
            for &(start, rows) in &tiles {
                assert_eq!(start, at, "tiles for {m} leave a gap or overlap");
                assert!(TILES.contains(&rows), "{rows} has no compiled kernel");
                at += rows;
            }
            assert_eq!(at, m, "tiles for {m} do not reach the last row");
        }
    }

    #[test]
    fn tiles_prefer_the_widest_kernel() {
        // A dispatch per 8 rows plus an exact tail - never 8 one-row passes
        // over the same weights.
        let widest = *TILES.last().unwrap();
        assert_eq!(tiles_for(widest), vec![(0, widest)]);
        assert_eq!(tiles_for(11), vec![(0, 8), (8, 2), (10, 1)]);
        assert_eq!(tiles_for(16), vec![(0, 8), (8, 8)]);
    }
}
