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
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use allpaka_gguf::GgmlType;

/// Where GPU wall time goes, accumulated across every call. The split that
/// matters: `wait_ns` is dead time the CPU spends blocked on the GPU, and its
/// ratio to `calls` is the per-round-trip cost every extra command buffer pays.
static CALLS: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
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
    (GPU_BUSY_NS.load(Ordering::Relaxed), SCHED_NS.load(Ordering::Relaxed))
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

const KERNELS: &str = r#"
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
    uint blocks = n_in / 32;
    device const uchar* row = w + w_off + (ulong)j * blocks * 34;
    float total[MAXT] = {0};
    for (uint b = lane; b < blocks; b += LPR) {
        device const uchar* blk = row + b * 34;
        float d = half_at(blk);
        float s[MAXT] = {0};
        for (uint l4 = 0; l4 < 8; l4++) {
            uchar4 raw;
            raw.x = blk[2 + l4 * 4];
            raw.y = blk[3 + l4 * 4];
            raw.z = blk[4 + l4 * 4];
            raw.w = blk[5 + l4 * 4];
            float4 qv = float4(as_type<char4>(raw));
            for (uint i = 0; i < TILE; i++) {
                float4 xv = ((device const float4*)(x + i * n_in + b * 32))[l4];
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
    ulong nb01 = (ulong)nb * 144;
    device const uchar* row0 = w + w_off + (ulong)j0 * nb01;

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
            y[flat + row] = s;
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
        if (active && lane == 0) y[i * ycols + jt] = s;
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
    device const float4* x4 = (device const float4*)x;
    float acc = 0.0f;
    for (uint b = lane; b < n_in / 4; b += LPR) {
        acc += dot(row[b], x4[b]);
    }
    float s = lane_sum(acc);
    if (active && lane == 0) y[j] = s;
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
kernel void router_topk(
    device const float* w [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device uint* ids [[buffer(2)]],
    device float* wts [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant ulong& w_off [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float logits[128];
    device const float* wr = (device const float*)((device const uchar*)w + w_off);
    // 8 simdgroups x 32 lanes: each simdgroup owns experts sg, sg+8, ... and
    // its 32 lanes split the dot; n <= 128 by the host-side router check.
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
    float v[4];
    uint vid[4];
    for (uint t = 0; t < 4; t++) {
        uint idx = lane + t * 32;
        v[t] = idx < n ? logits[idx] : -INFINITY;
        vid[t] = idx;
    }
    for (uint kk = 0; kk < k; kk++) {
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
    float v[4];
    uint vid[4];
    for (uint t = 0; t < 4; t++) {
        uint idx = lane + t * 32;
        v[t] = idx < n ? logits[idx] : -INFINITY;
        vid[t] = idx;
    }
    for (uint kk = 0; kk < k; kk++) {
        float lm = -INFINITY;
        uint li = 0xFFFFFFFF;
        for (uint t = 0; t < 4; t++) {
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
        for (uint t = 0; t < 4; t++) {
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
    float v[4];
    uint vid[4];
    for (uint t = 0; t < 4; t++) {
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
kernel void moe_combine(
    device const float* down [[buffer(0)]],
    device const float* wts [[buffer(1)]],
    device float* delta [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& slots [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float acc = 0.0f;
    for (uint s = 0; s < slots; s++) {
        acc += wts[s] * down[s * n + i];
    }
    delta[i] = acc;
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
struct MmGroup {
    uint expert;
    uint row0;
    uint rows;
};
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
    device const uchar* q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = sc_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float mn = xb->dmin;
    const float dl = d * sc[0];
    const float ml = mn * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (q[i] & mask) - ml);
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
    threadgroup float shmem_f[(48 * NK) < 2048 ? 2048 : (48 * NK)];            \
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
            float4 v0 = *(device const float4*)(by + 32 * c);                  \
            float4 v1 = *(device const float4*)(by + 32 * c + 4);              \
            threadgroup half* dstb = sb + 64 * ib + 8 * ly;                    \
            dstb[0] = (half)v0.x; dstb[1] = (half)v0.y;                        \
            dstb[2] = (half)v0.z; dstb[3] = (half)v0.w;                        \
            dstb[4] = (half)v1.x; dstb[5] = (half)v1.y;                        \
            dstb[6] = (half)v1.z; dstb[7] = (half)v1.w;                        \
        }                                                                      \
        chunk0 += NK / 16;                                                     \
        by += NK;                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);              \
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);              \
        for (short ik = 0; ik < NK / 8; ik++) {                                \
            for (short i = 0; i < 4; i++) {                                    \
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);             \
            }                                                                  \
            for (short i = 0; i < 2; i++) {                                    \
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);             \
            }                                                                  \
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
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        threadgroup float* temp_str =                                          \
            shmem_f + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;             \
        for (short i = 0; i < 8; i++) {                                        \
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),  \
                            64, 0, false);                                     \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        if (sgitg == 0) {                                                      \
            for (short j = tiitg; j < nr1; j += 32) {                          \
                device float* D = y + r0 + (ulong)(r1 + j) * n_out;            \
                threadgroup const float* C = temp_str + (ulong)j * 64;         \
                for (short i = 0; i < nr0; i++) {                              \
                    D[i] = C[i];                                               \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
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
#define DEFINE_MM_ID_LL_NK(NAME, BLOCK_T, NL, DQ, ROW_BYTES, GATHER, NK)        \
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
    uint3 tg [[threadgroup_position_in_grid]],                                 \
    ushort tiitg [[thread_index_in_threadgroup]],                              \
    ushort sgitg [[simdgroup_index_in_threadgroup]])                           \
{                                                                              \
    MmGroup grp = table[tg.z];                                                 \
    const uint r1 = tg.y * 32;                                                 \
    if (r1 >= grp.rows) {                                                      \
        return;                                                                \
    }                                                                          \
    threadgroup float shmem_f[(48 * NK) < 2048 ? 2048 : (48 * NK)];            \
    threadgroup half* sa = (threadgroup half*)shmem_f;                         \
    threadgroup half* sb = (threadgroup half*)shmem_f + 64 * NK;               \
    const uint r0 = tg.x * 64;                                                 \
    const uint m = grp.rows;                                                   \
    device const float* xin = GATHER ? x : x + (ulong)grp.row0 * n_in;         \
    device float* yg = y + (ulong)grp.row0 * n_out;                            \
    const short nr0 = (n_out - r0 < 64) ? (short)(n_out - r0) : 64;            \
    const short nr1 = (m - r1 < 32) ? (short)(m - r1) : 32;                    \
    const short lr0 = ((short)(tiitg / 2) < nr0) ? (short)(tiitg / 2) : nr0 - 1; \
    const short lr1 = ((short)(tiitg / 4) < nr1) ? (short)(tiitg / 4) : nr1 - 1; \
    const short il0 = tiitg % 2;                                               \
    const ulong nb01 = (ulong)(ROW_BYTES);                                     \
    device const BLOCK_T* bx0 = (device const BLOCK_T*)                        \
        (w + w_off + (ulong)grp.expert * estride + nb01 * (r0 + lr0));         \
    const short iy = 8 * (tiitg % 4);                                          \
    const ulong brow = GATHER ? (ulong)tok[grp.row0 + r1 + lr1]                \
                              : (ulong)(r1 + lr1);                             \
    device const float* by = xin + brow * n_in + iy;                           \
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
            float4 v0 = *(device const float4*)(by + 32 * c);                  \
            float4 v1 = *(device const float4*)(by + 32 * c + 4);              \
            threadgroup half* dstb = sb + 64 * ib + 8 * ly;                    \
            dstb[0] = (half)v0.x; dstb[1] = (half)v0.y;                        \
            dstb[2] = (half)v0.z; dstb[3] = (half)v0.w;                        \
            dstb[4] = (half)v1.x; dstb[5] = (half)v1.y;                        \
            dstb[6] = (half)v1.z; dstb[7] = (half)v1.w;                        \
        }                                                                      \
        chunk0 += NK / 16;                                                     \
        by += NK;                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);              \
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);              \
        for (short ik = 0; ik < NK / 8; ik++) {                                \
            for (short i = 0; i < 4; i++) {                                    \
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);             \
            }                                                                  \
            for (short i = 0; i < 2; i++) {                                    \
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);             \
            }                                                                  \
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
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        threadgroup float* temp_str =                                          \
            shmem_f + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;             \
        for (short i = 0; i < 8; i++) {                                        \
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),  \
                            64, 0, false);                                     \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        if (sgitg == 0) {                                                      \
            for (short j = tiitg; j < nr1; j += 32) {                          \
                device float* D = yg + r0 + (ulong)(r1 + j) * n_out;           \
                threadgroup const float* C = temp_str + (ulong)j * 64;         \
                for (short i = 0; i < nr0; i++) {                              \
                    D[i] = C[i];                                               \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_MM_ID_LL_NK(mmll_id_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 0, 32)
DEFINE_MM_ID_LL_NK(mmllg_id_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 1, 32)
DEFINE_MM_ID_LL_NK(mmll_id_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 0, 32)
DEFINE_MM_ID_LL_NK(mmllg_id_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 1, 32)
DEFINE_MM_ID_LL_NK(mmll_id_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 0, 32)
DEFINE_MM_ID_LL_NK(mmllg_id_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 1, 32)
DEFINE_MM_ID_LL_NK(mmll_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 0, 32)
DEFINE_MM_ID_LL_NK(mmllg_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 1, 32)
DEFINE_MM_ID_LL_NK(mmll_id_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 0, 32)
DEFINE_MM_ID_LL_NK(mmllg_id_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 1, 32)
DEFINE_MM_ID_LL_NK(mmll_id_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 0, 32)
DEFINE_MM_ID_LL_NK(mmllg_id_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 1, 32)
DEFINE_MM_ID_LL_NK(mm64llg_id_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 1, 64)
DEFINE_MM_ID_LL_NK(mm64llg_id_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 1, 64)
DEFINE_MM_ID_LL_NK(mm64llg_id_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 1, 64)
DEFINE_MM_ID_LL_NK(mm64llg_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 1, 64)
DEFINE_MM_ID_LL_NK(mm64llg_id_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 1, 64)
DEFINE_MM_ID_LL_NK(mm64llg_id_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 1, 64)
DEFINE_MM_ID_LL_NK(mm64ll_id_q8_0, BlkQ8_0, 2, dqll_q8_0, (n_in / 32) * 34, 0, 64)
DEFINE_MM_ID_LL_NK(mm64ll_id_q2_k, BlkQ2K, 16, dqll_q2_k, (n_in / 256) * 84, 0, 64)
DEFINE_MM_ID_LL_NK(mm64ll_id_q3_k, BlkQ3K, 16, dqll_q3_k, (n_in / 256) * 110, 0, 64)
DEFINE_MM_ID_LL_NK(mm64ll_id_q4_k, BlkQ4K, 16, dqll_q4_k, (n_in / 256) * 144, 0, 64)
DEFINE_MM_ID_LL_NK(mm64ll_id_q5_k, BlkQ5K, 16, dqll_q5_k, (n_in / 256) * 176, 0, 64)
DEFINE_MM_ID_LL_NK(mm64ll_id_q6_k, BlkQ6K, 16, dqll_q6_k, (n_in / 256) * 210, 0, 64)
DEFINE_MM_ID_LL_NK(mmll_id_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 0, 32)
DEFINE_MM_ID_LL_NK(mmllg_id_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 1, 32)
DEFINE_MM_ID_LL_NK(mm64ll_id_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 0, 64)
DEFINE_MM_ID_LL_NK(mm64llg_id_q5_0, BlkQ5_0, 2, dqll_q5_0, (n_in / 32) * 22, 1, 64)

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
        if (active && lane == 0) y[i * ycols + jt] = s;
    }
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
        "residual_norm",
        "residual_norm_sig",
        "norm_rows",
        "combine_rows",
        "softmax_topk",
        "sigmoid_topk",
        "router_topk",
        "moe_combine",
    ] {
        let f = library.get_function(name, None).ok()?;
        let p = device.new_compute_pipeline_state_with_function(&f).ok()?;
        pipelines.insert((name, 1, 1), p);
    }

    // Arenas start big enough for a decode layer and grow on demand.
    let x_arena = device.new_buffer(1 << 22, MTLResourceOptions::StorageModeShared);
    let y_arena = device.new_buffer(1 << 22, MTLResourceOptions::StorageModeShared);
    let pf_x = device.new_buffer(1 << 12, MTLResourceOptions::StorageModeShared);
    let pf_hs = device.new_buffer(1 << 12, MTLResourceOptions::StorageModeShared);

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
        let step = if max_buf > CHUNK_OVERLAP { max_buf - CHUNK_OVERLAP } else { max_buf / 2 };
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
            added.push(WeightChunk { buf, start: base + start, len });
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
                format!(" in {} windows (maxBufferLength {:.1} GiB)",
                    added.len(),
                    max_buf as f64 / (1u64 << 30) as f64)
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
    fn pipeline(&mut self, name: &'static str, tile: usize, lpr: usize) -> Option<&ComputePipelineState> {
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
            let f = self
                .library
                .get_function(name, Some(consts))
                .map_err(|e| eprintln!("metal: {name}<tile {tile}, lanes {lpr}> failed: {e}"))
                .ok()?;
            let p = self.device.new_compute_pipeline_state_with_function(&f).ok()?;
            self.pipelines.insert(key, p);
        }
        self.pipelines.get(&key)
    }

    /// The megakernel pipeline for one (gate format, down format) pair,
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
            let p = self.device.new_compute_pipeline_state_with_function(&f).ok()?;
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
            self.pf_x = self.device.new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pf_x_cap = cap;
        }
        if bytes > self.pf_hs_cap {
            let cap = bytes.next_power_of_two();
            self.pf_hs = self.device.new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.pf_hs_cap = cap;
        }
    }

    fn ensure_arenas(&mut self, x_need: usize, y_need: usize) {
        if x_need > self.x_cap {
            let cap = x_need.next_power_of_two();
            self.x_arena = self.device.new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.x_cap = cap;
        }
        if y_need > self.y_cap {
            let cap = y_need.next_power_of_two();
            self.y_arena = self.device.new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
            self.y_cap = cap;
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
            (GgmlType::Q8_0, false) => "mmll_q8_0",
            (GgmlType::Q2K, false) => "mmll_q2_k",
            (GgmlType::Q3K, false) => "mmll_q3_k",
            (GgmlType::Q4K, false) => "mmll_q4_k",
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
                if *F32LL.get_or_init(|| {
                    std::env::var("ALLPAKA_MM_LL_F32").is_ok_and(|v| v == "1")
                }) {
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
    matvec_batch(&[MatvecReq { ty, w, n_in, n_out, x, m: 1 }]).map(|mut v| v.pop().unwrap())
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
            slots.push(Slot { ri, start_row: 0, rows: r.m, x_off: x_need, y_off: y_need, mm: true });
            x_need += align(r.m * r.n_in * 4);
            y_need += align(r.m * r.n_out * 4);
            continue;
        }
        for (start_row, rows) in tiles_for(r.m) {
            slots.push(Slot { ri, start_row, rows, x_off: x_need, y_off: y_need, mm: false });
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
                gpu.pipeline(mm_kernel_for(reqs[s.ri].ty, s.rows)?.0, 1, 1)?.to_owned(),
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
            std::ptr::copy_nonoverlapping(
                xs.as_ptr() as *const u8,
                xp.add(s.x_off),
                xs.len() * 4,
            );
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
        let enc = cmd.compute_command_encoder_with_dispatch_type(
            metal::MTLDispatchType::Concurrent,
        );
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
                let (bm, bn) =
                    mm_kernel_for(r.ty, s.rows).map_or((32, 32), |(_, bm, bn)| (bm, bn));
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

        let mut out: Vec<Vec<f32>> =
            reqs.iter().map(|r| Vec::with_capacity(r.m * r.n_out)).collect();
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
    Some(SharedRegion { buf, len: region.len() })
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
pub fn attend_project(
    req: &AttnReq,
    wo_ty: GgmlType,
    wo: &[u8],
    n_out: usize,
) -> Option<Vec<f32>> {
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
fn encode_attend(
    enc: &metal::ComputeCommandEncoderRef,
    gpu: &Gpu,
    req: &AttnReq,
    y_off: usize,
) {
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
    /// Raw F32 Q/K/V biases in the mmap (GLM-4); None for bias-free models.
    pub q_bias: Option<&'a [u8]>,
    pub k_bias: Option<&'a [u8]>,
    pub v_bias: Option<&'a [u8]>,
    /// Element offsets of this layer's K and V in the cache region.
    pub k_off: usize,
    pub v_off: usize,
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
    },
}

pub struct TokenReq<'a> {
    /// The embedded input row, `hidden` floats.
    pub x: &'a [f32],
    pub layers: &'a [TokenLayer<'a>],
    pub cache: &'a SharedRegion,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub pos: usize,
    pub scale: f32,
    pub rope: &'a [[f32; 2]],
    /// Rotary width; equals head_dim for full rope, head_dim/2 for GLM-4.
    pub rot_dim: usize,
    pub eps: f32,
    pub output_norm: &'a [u8],
    pub output: (GgmlType, &'a [u8], usize),
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
    Some(NormRef { chunk, off: (addr - gpu.chunks[chunk].start) as u64 })
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
        std::env::var("ALLPAKA_MEGA_TG").ok().and_then(|v| v.parse().ok()).unwrap_or(48)
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
pub fn decode_token(req: &TokenReq) -> Option<Vec<f32>> {
    let dbg = std::env::var_os("ALLPAKA_TOKENBUF_DEBUG").is_some();
    macro_rules! why {
        ($cond:expr, $msg:expr) => {
            if $cond {
                if dbg { eprintln!("tokenbuf declined: {}", $msg); }
                return None;
            }
        };
    }
    let hidden = req.x.len();
    let hd = req.head_dim;
    why!(hd != ATTN_HEAD_DIM || hidden % 4 != 0, "shape");
    // Full rotary or GLM-style half rotary; the prep kernels know only these.
    why!(req.rope.len() != req.rot_dim / 2, "rope table");
    why!(req.rot_dim != hd && req.rot_dim * 2 != hd, "rot dim");
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    let span = (req.pos + 1) * req.kv_dim;
    let vocab = req.output.2;

    let Some(Some(cell)) = GPU.get().map(|c| c.as_ref()) else {
        why!(true, "no gpu");
        return None;
    };
    let Ok(mut gpu) = cell.lock() else {
        why!(true, "lock");
        return None;
    };

    // Resolve every weight and norm up front; any miss declines the token.
    struct LayerRefs {
        attn_norm: NormRef,
        ffn_norm: NormRef,
        mats: [MatRef; 4],
        mat_states: [ComputePipelineState; 4],
        q_bias: Option<NormRef>,
        k_bias: Option<NormRef>,
        v_bias: Option<NormRef>,
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
        },
    }

    let mut layers = Vec::with_capacity(req.layers.len());
    let mut max_ffn = 0usize;
    let mut max_slots = 1usize;
    let mut max_expert = 0usize;
    for l in req.layers {
        why!((l.k_off + span) * 2 > req.cache.len || (l.v_off + span) * 2 > req.cache.len, "cache span");
        let dims = [(hidden, l.wq.2), (hidden, l.wk.2), (hidden, l.wv.2), (q_dim, l.wo.2)];
        why!(l.wq.2 != q_dim || l.wk.2 != kv || l.wv.2 != kv || l.wo.2 != hidden, "attn dims");
        let mats = [
            resolve(&gpu, l.wq.0, l.wq.1)?,
            resolve(&gpu, l.wk.0, l.wk.1)?,
            resolve(&gpu, l.wv.0, l.wv.1)?,
            resolve(&gpu, l.wo.0, l.wo.1)?,
        ];
        // The spin-flag scheme replaces the two norm barriers when every
        // consumer kernel carries the WAIT variant: qkv (q4_k_mv) and the
        // router (f32). `ALLPAKA_NORMFLAG=0` reverts to barriers.
        let normflag = std::env::var("ALLPAKA_NORMFLAG").is_ok_and(|v| v == "1")
            && mats.iter().take(3).all(|m| m.kernel == "matvec_q4_k_mv")
            // Dense gate/up read h with no intervening barrier; only the MoE
            // chain (router waits on the flag, the rest orders behind its
            // barriers) is safe without the norm barrier.
            && matches!(l.ffn, TokenFfn::Moe { .. });
        let mut mat_states = Vec::with_capacity(4);
        for (i, (mat, &(n_in, _))) in mats.iter().zip(&dims).enumerate() {
            let lpr = lanes_per_row(mat.ty, n_in);
            let wait = normflag && i < 3;
            mat_states.push(
                gpu.pipeline_wait(mat.kernel, 1, lpr, false, false, wait)?.to_owned(),
            );
        }
        let ffn = match &l.ffn {
            TokenFfn::Dense { gate, up, down } => {
                if gate.2 != up.2 || down.2 != hidden {
                    return None;
                }
                let mats = [
                    resolve(&gpu, gate.0, gate.1)?,
                    resolve(&gpu, up.0, up.1)?,
                    resolve(&gpu, down.0, down.1)?,
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
            TokenFfn::Moe { router, router_bias, gate, up, down, expert_ffn, n_used, sigmoid, shared } => {
                why!(router.0 != GgmlType::F32 || router.2 > 128 || *n_used > 16, "router shape");
                let router_ref = resolve_f32(&gpu, router.1, hidden)?;
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
                    resolve(&gpu, gate.0, gate.1)?,
                    resolve(&gpu, up.0, up.1)?,
                    resolve(&gpu, down.0, down.1)?,
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
                for (i, ((mat, n_in), n_out)) in
                    mats.iter().zip([hidden, hidden, *expert_ffn]).zip(n_outs).enumerate()
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
                    // The down projection folds swiglu into its loads when
                    // its kernel carries the variant; the standalone swiglu
                    // dispatch and one barrier then drop out of the layer.
                    let swiglu = i == 2
                        && shared.is_none()
                        && matches!(
                            kernel,
                            "matvec_q2_k" | "matvec_q3_k_mv" | "matvec_q4_k_mv"
                        )
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
                        why!(sh[0].2 != *expert_ffn || sh[1].2 != *expert_ffn
                            || sh[2].2 != hidden, "shared dims");
                        let smats = [
                            resolve(&gpu, sh[0].0, sh[0].1)?,
                            resolve(&gpu, sh[1].0, sh[1].1)?,
                            resolve(&gpu, sh[2].0, sh[2].1)?,
                        ];
                        let mut sstates = Vec::with_capacity(3);
                        for (mat, n_in) in
                            smats.iter().zip([hidden, hidden, *expert_ffn])
                        {
                            let lpr = lanes_per_row(mat.ty, n_in);
                            sstates.push(gpu.pipeline(mat.kernel, 1, lpr)?.to_owned());
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
                let mega = if mega_enabled() {
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
            mat_states: mat_states.try_into().ok()?,
            q_bias: bias_ref(l.q_bias, q_dim)?,
            k_bias: bias_ref(l.k_bias, kv)?,
            v_bias: bias_ref(l.v_bias, kv)?,
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
    let sigmoid_gating = req.layers.iter().any(|l| {
        matches!(l.ffn, TokenFfn::Moe { sigmoid: true, .. })
    });
    let topk_state = gpu.pipelines[&(
        if sigmoid_gating { "sigmoid_topk" } else { "softmax_topk" },
        1,
        1,
    )]
    .to_owned();
    let rtopk_state = gpu.pipelines[&("router_topk", 1, 1)].to_owned();
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();
    let combine_state = gpu.pipelines[&("moe_combine", 1, 1)].to_owned();

    // Arena layout, element offsets over f32 (ids are u32-in-f32 slots).
    let align = |n: usize| (n + 63) & !63;
    let x_at = 0usize;
    let delta_at = x_at + align(hidden);
    let h_at = delta_at + align(hidden);
    let q_at = h_at + align(hidden);
    let k_at = q_at + align(q_dim);
    let v_at = k_at + align(kv);
    let attn_at = v_at + align(kv);
    let logits_at = attn_at + align(q_dim);
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
    static SPLIT_FORCE: OnceLock<Option<usize>> = OnceLock::new();
    let nsplit = match *SPLIT_FORCE.get_or_init(|| {
        std::env::var("ALLPAKA_ATTN_SPLIT").ok().and_then(|v| v.parse().ok())
    }) {
        Some(n) => n,
        None => n_pos / 128,
    }
    .clamp(1, 16)
    .min(n_pos);
    let sp_acc_at = ctr_at + align(1);
    let sp_md_at = sp_acc_at + align(req.n_heads * nsplit * hd);
    let total = sp_md_at + align(req.n_heads * nsplit * 2);
    gpu.ensure_arenas(4096, total * 4);

    // Stage x and zero the first delta.
    unsafe {
        let yp = gpu.y_arena.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(req.x.as_ptr(), yp.add(x_at), hidden);
        std::ptr::write_bytes(yp.add(delta_at), 0, hidden);
        std::ptr::write_bytes(yp.add(flag_at), 0, 1);
        std::ptr::write_bytes(yp.add(ctr_at), 0, 1);
        // The shared expert's combine weight is a constant 1 in the slot past
        // the router-written ones.
        for l in &layers {
            if let FfnRefs::Moe { n_used, shared: Some(_), .. } = &l.ffn {
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
        static SERIAL: OnceLock<bool> = OnceLock::new();
        let serial = *SERIAL
            .get_or_init(|| std::env::var("ALLPAKA_DECODE_SERIAL").is_ok_and(|v| v == "1"));
        let enc = cmd.compute_command_encoder_with_dispatch_type(if serial {
            metal::MTLDispatchType::Serial
        } else {
            metal::MTLDispatchType::Concurrent
        });
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
            let idx = GpuIdxArgs { stride, slots: slots as u32, x_stride: x_stride as u32 };
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
        for (li, (l, refs)) in req.layers.iter().zip(&layers).enumerate() {
            let ep_attn = (li as u32) * 2 + 1;
            let ep_ffn = (li as u32) * 2 + 2;
            // h = rmsnorm(x + delta) * attn_norm; x absorbs delta.
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
            // qkv from h.
            if !probe_skip("qkv") {
                matvec(enc, &refs.mat_states[0], &refs.mats[0], hidden, q_dim, h_at, q_at);
                matvec(enc, &refs.mat_states[1], &refs.mats[1], hidden, kv, h_at, k_at);
                matvec(enc, &refs.mat_states[2], &refs.mats[2], hidden, kv, h_at, v_at);
            }
            bar_c(enc, b'a');
            // Norm + rope + cache store.
            if !probe_skip("attend") {
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
            enc.set_bytes(4, std::mem::size_of::<QkPrepArgs>() as u64,
                &args as *const QkPrepArgs as *const _);
            let ones = [1.0f32; ATTN_HEAD_DIM];
            let qw = l.q_norm.unwrap_or(&ones);
            let kw = l.k_norm.unwrap_or(&ones);
            enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
            enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
            enc.set_bytes(7, (req.rope.len() * 8) as u64, req.rope.as_ptr() as *const _);
            // Bias buffers: real F32 regions when the model has them, any
            // valid buffer otherwise (the kernel skips them on has_bias == 0).
            let dummy = &gpu.chunks[refs.attn_norm.chunk].buf;
            for (index, b) in [(8u64, &refs.q_bias), (9, &refs.k_bias), (10, &refs.v_bias)] {
                match b {
                    Some(r) => enc.set_buffer(index, Some(&gpu.chunks[r.chunk].buf), r.off),
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
            bar_c(enc, b'a');
            // Output projection straight into delta.
            if !probe_skip("qkv") {
                matvec(enc, &refs.mat_states[3], &refs.mats[3], q_dim, hidden, attn_at, delta_at);
            }
            bar_c(enc, b'a');
            // The megakernel absorbs the whole FFN half, resnorm included.
            if let FfnRefs::Moe { mega: Some(mega_state), router, router_bias, mats, strides, n_expert, expert_ffn, n_used, sigmoid, shared, .. } = &refs.ffn {
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
                enc.set_buffer(5, Some(&gpu.chunks[refs.ffn_norm.chunk].buf), refs.ffn_norm.off);
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
            // h = rmsnorm(x + delta) * ffn_norm.
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

            match &refs.ffn {
                FfnRefs::Dense { mats, states, ffn_dim } => {
                    matvec(enc, &states[0], &mats[0], hidden, *ffn_dim, h_at, gate_at);
                    matvec(enc, &states[1], &mats[1], hidden, *ffn_dim, h_at, up_at);
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
                    matvec(enc, &states[2], &mats[2], *ffn_dim, hidden, gate_at, delta_at);
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
                    sigmoid: _,
                    shared,
                } => {
                    let n_slots = *n_used + shared.is_some() as usize;
                    if !probe_skip("router") {
                    let _ = &rtopk_state;
                    matvec(enc, router_state, router, hidden, *n_expert, h_at, logits_at);
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
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
                    }
                    bar_c(enc, b'f');
                    if !probe_skip("experts") {
                    matvec_idx(enc, &states[0], &mats[0], hidden, *expert_ffn,
                        h_at, gate_at, strides[0], *n_used, 0);
                    matvec_idx(enc, &states[1], &mats[1], hidden, *expert_ffn,
                        h_at, up_at, strides[1], *n_used, 0);
                    // The shared expert is a plain matvec into the extra slot.
                    if let Some((smats, sstates)) = shared {
                        matvec(enc, &sstates[0], &smats[0], hidden, *expert_ffn,
                            h_at, gate_at + *n_used * *expert_ffn);
                        matvec(enc, &sstates[1], &smats[1], hidden, *expert_ffn,
                            h_at, up_at + *n_used * *expert_ffn);
                    }
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
                        bar_c(enc, b'f');
                    } else {
                        // The down kernel reads raw gate (x) and raw up
                        // (buffer 8) and applies swiglu on load.
                        enc.set_buffer(8, Some(y), e(up_at));
                    }
                    matvec_idx(enc, &states[2], &mats[2], *expert_ffn, hidden,
                        gate_at, downo_at, strides[2], *n_used, *expert_ffn);
                    if let Some((smats, sstates)) = shared {
                        matvec(enc, &sstates[2], &smats[2], *expert_ffn, hidden,
                            gate_at + *n_used * *expert_ffn,
                            downo_at + *n_used * hidden);
                    }
                    }
                    bar_c(enc, b'f');
                    enc.set_compute_pipeline_state(&combine_state);
                    enc.set_buffer(0, Some(y), e(downo_at));
                    enc.set_buffer(1, Some(y), e(wts_at));
                    enc.set_buffer(2, Some(y), e(delta_at));
                    let n = hidden as u32;
                    let slots = n_slots as u32;
                    enc.set_bytes(3, 4, &n as *const u32 as *const _);
                    enc.set_bytes(4, 4, &slots as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new((hidden as u64).div_ceil(256), 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    dispatched += 8;
                }
            }
            bar(enc);
            dispatched += 8;
        }
        // Final norm and the output projection, still in the same buffer.
        resnorm(enc, &out_norm);
        bar(enc);
        if !probe_skip("head") {
            matvec(enc, &out_state, &out_mat, hidden, vocab, h_at, out_logits_at);
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

        let mut out = vec![0f32; vocab];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (gpu.y_arena.contents() as *const f32).add(out_logits_at),
                out.as_mut_ptr(),
                vocab,
            );
        }
        out
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
        let enc = cmd.compute_command_encoder_with_dispatch_type(
            metal::MTLDispatchType::Concurrent,
        );

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
        enc.set_bytes(4, std::mem::size_of::<QkPrepArgs>() as u64,
            &args as *const QkPrepArgs as *const _);
        let ones = [1.0f32; ATTN_HEAD_DIM];
        let qw = req.q_norm.unwrap_or(&ones);
        let kw = req.k_norm.unwrap_or(&ones);
        enc.set_bytes(5, (hd * 4) as u64, qw.as_ptr() as *const _);
        enc.set_bytes(6, (hd * 4) as u64, kw.as_ptr() as *const _);
        enc.set_bytes(7, (req.rope.len() * 8) as u64, req.rope.as_ptr() as *const _);
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
pub fn prefill_begin(xs: &[f32]) -> Option<()> {
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
    gpu.ensure_prefill(xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), gpu.pf_x.contents() as *mut f32, xs.len());
    }
    Some(())
}

/// Download the residual stream after the last fused layer.
pub fn prefill_end(xs: &mut [f32]) -> Option<()> {
    let gpu = GPU.get()?.as_ref()?.lock().ok()?;
    if xs.len() * 4 > gpu.pf_x_cap {
        return None;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(gpu.pf_x.contents() as *const f32, xs.as_mut_ptr(), xs.len());
    }
    Some(())
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
    /// Rotary width; == head_dim for full rope, smaller for GLM's partial.
    pub rot_dim: usize,
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
    // Below the tile-matmul threshold the projections would need per-row
    // matvec tiling this encoder does not carry; short tail chunks decline
    // to the step-by-step path instead. (Caught by decode_paths_agree: a
    // TILE=1 pipeline silently computes only the first row.)
    if hd != ATTN_HEAD_DIM
        || req.ropes.len() != req.m * req.rot_dim / 2
        || req.hs.len() != req.m * hidden
        || req.m < MM_MIN_M
    {
        return None;
    }
    let q_dim = req.n_heads * hd;
    let kv = req.n_kv_heads * hd;
    if req.wq.2 != q_dim || req.wk.2 != kv || req.wv.2 != kv || req.wo.2 != hidden {
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
        states.push(gpu.pipeline(mm_kernel_for(mat.ty, req.m)?.0, 1, 1)?.to_owned());
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
                let router_state =
                    gpu.pipeline(mm_kernel_for(GgmlType::F32, req.m)?.0, 1, 1)?.to_owned();
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
    let norm_state = gpu.pipelines[&("norm_rows", 1, 1)].to_owned();
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

    // Arenas: x holds hs then the rope tables; y holds q, k, v, attention
    // output and the projection.
    let align = |n: usize| (n + 63) & !63;
    let hs_at = 0usize;
    let ropes_at = hs_at + align(req.m * hidden);
    let x_total = ropes_at + align(req.ropes.len() * 2);
    let q_at = 0usize;
    let k_at = q_at + align(req.m * q_dim);
    let v_at = k_at + align(req.m * kv);
    let attn_at = v_at + align(req.m * kv);
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
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(
            metal::MTLDispatchType::Concurrent,
        );
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
            enc.dispatch_thread_groups(
                MTLSize::new(req.m as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        };
        let (hs_buf, hs_off): (&Buffer, usize) = if let Some(f) = &fusion {
            norm_rows(enc, &f.attn_norm, 0, out_at);
            enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
            (&gpu.pf_hs, 0)
        } else {
            (&gpu.x_arena, hs_at)
        };
        matmul(enc, &states[0], &mats[0], hidden, q_dim, hs_buf, hs_off, q_at);
        matmul(enc, &states[1], &mats[1], hidden, kv, hs_buf, hs_off, k_at);
        matmul(enc, &states[2], &mats[2], hidden, kv, hs_buf, hs_off, v_at);
        enc.memory_barrier_with_resources(&[y]);

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
        enc.set_bytes(4, std::mem::size_of::<QkPrepArgs>() as u64,
            &args as *const QkPrepArgs as *const _);
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
        enc.memory_barrier_with_resources(&[y, &req.cache.buf]);

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
        enc.memory_barrier_with_resources(&[y]);

        matmul(enc, &states[3], &mats[3], q_dim, hidden, &gpu.y_arena, attn_at, out_at);
        if let Some(f) = &fusion {
            enc.memory_barrier_with_resources(&[y]);
            // xs += wo output, then the FFN norm - one fused pass - and the
            // router matmul, all before the single wait.
            norm_rows(enc, &f.ffn_norm, 1, out_at);
            enc.memory_barrier_with_resources(&[&gpu.pf_hs]);
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
        DISPATCHES.fetch_add(6, Ordering::Relaxed);

        // Fused: the caller needs only the router logits; the projected
        // rows already live in pf_x. Plain: the projected rows come back.
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
            std::ptr::copy_nonoverlapping(
                r.q.as_ptr() as *const u8,
                xp.add(*x_off),
                r.q.len() * 4,
            );
        }
    }

    Some(objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        // Every position attends independently over an already-written
        // cache: no hazards, full concurrency.
        let enc = cmd.compute_command_encoder_with_dispatch_type(
            metal::MTLDispatchType::Concurrent,
        );
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
                    std::ptr::copy_nonoverlapping(
                        yp.add(*y_off),
                        v.as_mut_ptr() as *mut u8,
                        n * 4,
                    );
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
}

pub struct GroupedShared<'a> {
    pub gate: (GgmlType, &'a [u8]),
    pub up: (GgmlType, &'a [u8]),
    pub down: (GgmlType, &'a [u8]),
    /// Shared-expert FFN width (n_ff_exp * n_expert_shared).
    pub ffn: usize,
}

pub struct GroupedCombine<'a> {
    /// CSR offsets, `m + 1` entries.
    pub tok_off: &'a [u32],
    /// Flat-row index and routing weight per hit.
    pub hit_row: &'a [u32],
    pub hit_w: &'a [f32],
    pub m: usize,
}

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
    if q3_mv() { "matvec_q3_k_mv" } else { "matvec_q3_k" }
}

/// The q4_k llama-structure matvec. Default ON: decode 30B 90 -> 102 tok/s,
/// 235B 22.2 -> 24.3 (q4_k attention weights). `ALLPAKA_Q4_MV=0` reverts.
fn q4_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q4_MV").map_or(true, |v| v != "0"))
}

fn q4_kernel() -> &'static str {
    if q4_mv() { "matvec_q4_k_mv" } else { "matvec_q4_k" }
}

/// The q6_k llama-structure matvec probe: `ALLPAKA_Q6_MV=1` to enable.
fn q6_mv() -> bool {
    static MV: OnceLock<bool> = OnceLock::new();
    *MV.get_or_init(|| std::env::var("ALLPAKA_Q6_MV").is_ok_and(|v| v == "1"))
}

fn q6_kernel() -> &'static str {
    if q6_mv() { "matvec_q6_k_mv" } else { "matvec_q6_k" }
}

/// Prefill attention row tiling: `ALLPAKA_ATTN_T4=0` reverts to the
/// row-per-threadgroup kernel.
fn attend_rows_kernel() -> &'static str {
    static T4: OnceLock<usize> = OnceLock::new();
    match *T4.get_or_init(|| {
        std::env::var("ALLPAKA_ATTN_T4").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
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
        std::env::var("ALLPAKA_ATTN_S8").ok().and_then(|v| v.parse().ok()).unwrap_or(32)
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
/// K-step 64 for the LL mm kernels: `ALLPAKA_MM_K64=1` to enable.
fn mm_k64() -> bool {
    static K: OnceLock<bool> = OnceLock::new();
    *K.get_or_init(|| std::env::var("ALLPAKA_MM_K64").map_or(true, |v| v != "0"))
}

fn mm_ll() -> bool {
    static LL: OnceLock<bool> = OnceLock::new();
    *LL.get_or_init(|| std::env::var("ALLPAKA_MM_LL").map_or(true, |v| v != "0"))
}

fn mmid_kernel_for(ty: GgmlType, gather: bool) -> Option<(&'static str, usize)> {
    if mm_ll() {
        let name = match (ty, gather, mm_k64()) {
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
            (GgmlType::Q4K, true, false) => "mmllg_id_q4_k",
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
    if req.groups.is_empty() || req.tok.len() != req.total_rows {
        return None;
    }
    let mut gpu = GPU.get()?.as_ref()?.lock().ok()?;
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
    let strides = [
        (req.gate.1.len() / req.n_expert) as u64,
        (req.up.1.len() / req.n_expert) as u64,
        (req.down.1.len() / req.n_expert) as u64,
    ];
    let mut states = Vec::with_capacity(3);
    let mut bms = [32u64; 3];
    for (i, mat) in mats.iter().enumerate() {
        let (name, bm) = mmid_kernel_for(mat.ty, i < 2)?;
        bms[i] = bm as u64;
        states.push(gpu.pipeline(name, 1, 1)?.to_owned());
    }
    // Plain tile-matmul pipelines for the shared expert (no gather).
    let mut sh_states = Vec::with_capacity(3);
    if let Some(sh_mats) = &sh_mats {
        for mat in sh_mats {
            sh_states.push(gpu.pipeline(mm_kernel_for(mat.ty, m_fused)?.0, 1, 1)?.to_owned());
        }
    }
    let swiglu_state = gpu.pipelines[&("swiglu", 1, 1)].to_owned();

    let max_rows = req.groups.iter().map(|g| g[2]).max()? as usize;
    let row_tiles = max_rows.div_ceil(32) as u64;
    let n_groups = req.groups.len() as u64;

    // Arenas: x rows then the group table in x; gate, up, down out in y.
    let align = |n: usize| (n + 63) & !63;
    let x_at = 0usize;
    let x_len = if req.fused.is_some() { 0 } else { req.x.len() };
    let table_at = x_at + align(x_len);
    let tok_at = table_at + align(req.groups.len() * 3);
    let off_at = tok_at + align(req.total_rows);
    let (n_off, n_hits) = req.fused.as_ref().map_or((0, 0), |c| {
        (c.tok_off.len(), c.hit_row.len())
    });
    let hrow_at = off_at + align(n_off);
    let hw_at = hrow_at + align(n_hits);
    let x_total = hw_at + align(n_hits);
    let gate_at = 0usize;
    let up_at = gate_at + align(req.total_rows * req.ffn);
    let out_at = up_at + align(req.total_rows * req.ffn);
    // The shared expert's down rows live right after the expert down rows so
    // the combine can reference them as ordinary hit rows.
    let sh_rows = if req.shared.is_some() { m_fused } else { 0 };
    let sh_ffn = req.shared.as_ref().map_or(0, |s| s.ffn);
    let shg_at = out_at + align((req.total_rows + sh_rows) * req.hidden);
    let shu_at = shg_at + align(m_fused * sh_ffn);
    let y_total = shu_at + align(m_fused * sh_ffn);
    gpu.ensure_arenas(x_total * 4, y_total * 4);
    unsafe {
        let xp = gpu.x_arena.contents() as *mut f32;
        if req.fused.is_none() {
            std::ptr::copy_nonoverlapping(req.x.as_ptr(), xp.add(x_at), req.x.len());
        }
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
                c.tok_off.as_ptr() as *const f32, xp.add(off_at), c.tok_off.len());
            std::ptr::copy_nonoverlapping(
                c.hit_row.as_ptr() as *const f32, xp.add(hrow_at), c.hit_row.len());
            std::ptr::copy_nonoverlapping(c.hit_w.as_ptr(), xp.add(hw_at), c.hit_w.len());
        }
    }

    let out = objc::rc::autoreleasepool(|| {
        let t_encode = std::time::Instant::now();
        let cmd = gpu.queue.new_command_buffer();
        let enc = cmd.compute_command_encoder_with_dispatch_type(
            metal::MTLDispatchType::Concurrent,
        );
        let y = &gpu.y_arena;
        let e = |off: usize| (off * 4) as u64;
        let stage = |enc: &metal::ComputeCommandEncoderRef,
                     si: usize,
                     n_in: usize,
                     n_out: usize,
                     x_buf: &Buffer,
                     x_off: usize,
                     y_off: usize| {
            enc.set_compute_pipeline_state(&states[si]);
            enc.set_buffer(0, Some(&gpu.chunks[mats[si].chunk].buf), 0);
            enc.set_buffer(1, Some(x_buf), e(x_off));
            enc.set_buffer(2, Some(y), e(y_off));
            let a = n_in as u32;
            let b = n_out as u32;
            enc.set_bytes(3, 4, &a as *const u32 as *const _);
            enc.set_bytes(4, 4, &b as *const u32 as *const _);
            enc.set_bytes(5, 8, &mats[si].w_off as *const u64 as *const _);
            enc.set_buffer(6, Some(&gpu.x_arena), e(table_at));
            enc.set_bytes(7, 8, &strides[si] as *const u64 as *const _);
            enc.set_buffer(8, Some(&gpu.x_arena), e(tok_at));
            enc.dispatch_thread_groups(
                MTLSize::new((n_out as u64).div_ceil(bms[si]), row_tiles, n_groups),
                MTLSize::new(128, 1, 1),
            );
        };

        let (xb, xo): (&Buffer, usize) =
            if req.fused.is_some() { (&gpu.pf_hs, 0) } else { (&gpu.x_arena, x_at) };
        stage(enc, 0, req.hidden, req.ffn, xb, xo, gate_at);
        stage(enc, 1, req.hidden, req.ffn, xb, xo, up_at);
        enc.memory_barrier_with_resources(&[y]);
        enc.set_compute_pipeline_state(&swiglu_state);
        enc.set_buffer(0, Some(y), e(gate_at));
        enc.set_buffer(1, Some(y), e(up_at));
        let n32 = (req.total_rows * req.ffn) as u32;
        enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.memory_barrier_with_resources(&[y]);
        stage(enc, 2, req.ffn, req.hidden, &gpu.y_arena, gate_at, out_at);
        if let (Some(sh), Some(sh_mats)) = (&req.shared, &sh_mats) {
            // The shared expert: plain (ungathered) tile matmuls over the
            // m pf_hs rows, swiglu, then down into the row region right
            // after the expert rows.
            let mm = |enc: &metal::ComputeCommandEncoderRef,
                      si: usize,
                      n_in: usize,
                      n_out: usize,
                      x_buf: &Buffer,
                      x_off: usize,
                      y_off: usize| {
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
            enc.memory_barrier_with_resources(&[y]);
            mm(enc, 0, req.hidden, sh.ffn, &gpu.pf_hs, 0, shg_at);
            mm(enc, 1, req.hidden, sh.ffn, &gpu.pf_hs, 0, shu_at);
            enc.memory_barrier_with_resources(&[y]);
            enc.set_compute_pipeline_state(&swiglu_state);
            enc.set_buffer(0, Some(y), e(shg_at));
            enc.set_buffer(1, Some(y), e(shu_at));
            let n32 = (m_fused * sh.ffn) as u32;
            enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((n32 as u64).div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.memory_barrier_with_resources(&[y]);
            mm(enc, 2, sh.ffn, req.hidden, &gpu.y_arena, shg_at,
                out_at + req.total_rows * req.hidden);
        }
        if let Some(c) = &req.fused {
            enc.memory_barrier_with_resources(&[y]);
            enc.set_compute_pipeline_state(&gpu.pipelines[&("combine_rows", 1, 1)]);
            enc.set_buffer(0, Some(&gpu.pf_x), 0);
            enc.set_buffer(1, Some(y), e(out_at));
            enc.set_buffer(2, Some(&gpu.x_arena), e(off_at));
            enc.set_buffer(3, Some(&gpu.x_arena), e(hrow_at));
            enc.set_buffer(4, Some(&gpu.x_arena), e(hw_at));
            let h32 = req.hidden as u32;
            enc.set_bytes(5, 4, &h32 as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(c.m as u64, 1, 1),
                MTLSize::new(256, 1, 1),
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
        DISPATCHES.fetch_add(4, Ordering::Relaxed);

        // Fused: the combine already landed in pf_x; skip the download.
        let out_rows = if req.fused.is_some() { 0 } else { req.total_rows };
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
    Some(MatRef { kernel, ty, chunk, w_off: (addr - gpu.chunks[chunk].start) as u64 })
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
                per_mat.push(vec![gpu.pipeline(mm_kernel_for(mat.ty, r.m)?.0, 1, 1)?.to_owned()]);
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
        let enc = cmd.compute_command_encoder_with_dispatch_type(
            metal::MTLDispatchType::Concurrent,
        );
        let mut dispatched = 0u64;
        for (((r, p), (gate, up, _)), st) in
            reqs.iter().zip(&plans).zip(&mats).zip(&states)
        {
            dispatched += encode_matvec(
                enc, &gpu, gate, &st[0], r.hidden, r.ffn,
                &gpu.x_arena, p.x_off, &gpu.y_arena, p.gate_off, r.m,
            );
            dispatched += encode_matvec(
                enc, &gpu, up, &st[1], r.hidden, r.ffn,
                &gpu.x_arena, p.x_off, &gpu.y_arena, p.up_off, r.m,
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
        for (((r, p), (_, _, down)), st) in
            reqs.iter().zip(&plans).zip(&mats).zip(&states)
        {
            dispatched += encode_matvec(
                enc, &gpu, down, &st[2], r.ffn, r.hidden,
                &gpu.y_arena, p.gate_off, &gpu.y_arena, p.out_off, r.m,
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
