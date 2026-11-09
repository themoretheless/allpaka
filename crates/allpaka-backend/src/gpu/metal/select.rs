//! Kernel and policy selection: which pipeline variant a tensor format and
//! shape gets, and the runtime switches that A/B the alternatives.

use super::*;

/// Deferred fused-prefill commits: layer buffers chain through an
/// MTLSharedEvent instead of one CPU wait per layer, so the CPU encodes
/// layer N+1 while the GPU still runs layer N. `ALLPAKA_PF_DEFER=0` reverts
/// to per-layer waits.
pub(super) fn pf_defer() -> bool {
    crate::runtime::get().prefill_defer
}

/// The whole fused prefill chunk as ONE command buffer (sequential
/// encoders per stage, one commit in prefill_end). Default ON: measured
/// qwen3-30b pp480 1222-1230 vs 1196-1198 tok/s with the event chain
/// (pp1200: 1124 vs 1095) - pure buffer-start overhead removal, GPU
/// executing unchanged. `ALLPAKA_PF_ONEBUF=0` reverts to PF_DEFER chaining.
pub(super) fn pf_onebuf() -> bool {
    crate::runtime::get().prefill_one_buffer
}

/// Join the shared prefill buffer without GPU-side MoE routing, for a grouped
/// FFN that has no route table and whose combine already lands in `pf_x`
/// (`req.fused.is_some()`: the download is skipped, so nothing is read back).
/// Dense models reach this. The `route_mode` gate guards the router-logits
/// rescue, which a dense layer has no logits for, so it excluded dense from the
/// segment over a hazard that cannot occur there. A CPU-routed MoE keeps the
/// old path: it is not what this was measured on.
/// `ALLPAKA_PF_OBUF_DENSE=0` reverts.
pub(super) fn dense_obuf() -> bool {
    crate::runtime::get().dense_obuf
}

/// ALLPAKA_GPU_COUNTERS=1: per-dispatch GPUTimestamp sampling in the fused
/// prefill. This device exposes only the "timestamp" counter set, and only
/// atCommandBoundary sampling - per-dispatch compute-encoder samples assert
/// ("not supported on this device"), so the mode is a silent no-op here.
/// Kept as the probe harness for a device that supports it.
pub(super) fn gpu_counters() -> bool {
    crate::runtime::get().gpu_counters
}

/// Whether sampleCountersInBuffer works on compute encoders here
/// (MTLCounterSamplingPoint::atDispatchBoundary == 2).
pub(super) fn gpu_counters_supported(device: &metal::DeviceRef) -> bool {
    use objc::{msg_send, sel, sel_impl};
    unsafe { msg_send![device, supportsCounterSampling: 2u64] }
}

/// Stamp capacity per prefill chunk (~16 dispatches x 48 layers x chunks).
pub(super) const CSTAMP_CAP: usize = 1024;

/// The llama.cpp-structure q2_k matvec, `ALLPAKA_Q2_MV=1` to enable. OFF by
/// default: measured 95 GB/s against the word-load kernel's ~170 in
/// ffn-shaped decode (the word-load rewrite already beat llama's structure
/// for this format; the port only won for q3_k, whose old kernel still
/// assembled ushort pairs).
pub(super) fn q2_mv() -> bool {
    crate::runtime::get().q2_mv
}

/// Same switch for the q3_k llama-structure kernel: `ALLPAKA_Q3_MV=0` off.
pub(super) fn q3_mv() -> bool {
    crate::runtime::get().q3_mv
}

pub(super) fn q3_kernel() -> &'static str {
    if q3_mv() {
        "matvec_q3_k_mv"
    } else {
        "matvec_q3_k"
    }
}

/// The q4_k llama-structure matvec. Default ON: decode 30B 90 -> 102 tok/s,
/// 235B 22.2 -> 24.3 (q4_k attention weights). `ALLPAKA_Q4_MV=0` reverts.
pub(super) fn q4_mv() -> bool {
    crate::runtime::get().q4_mv
}

pub(super) fn q4_kernel() -> &'static str {
    if q4_mv() {
        "matvec_q4_k_mv"
    } else {
        "matvec_q4_k"
    }
}

/// The q6_k llama-structure matvec. Default ON after alternating 30B A/B on
/// M4 Max measured 133.4 versus 131.6 tok/s with decode parity preserved.
/// `ALLPAKA_Q6_MV=0` restores the reference kernel.
pub(super) fn q6_mv() -> bool {
    crate::runtime::get().q6_mv
}

pub(super) fn q6_kernel() -> &'static str {
    if q6_mv() {
        "matvec_q6_k_mv"
    } else {
        "matvec_q6_k"
    }
}

/// The q5_k llama-structure matvec. Default ON for GLM shared gate/up and
/// qwen35moe expert down (Q5_K). Carries SWIGLU_X so decode SWFUSE can fuse
/// the down projection. `ALLPAKA_Q5_MV=0` restores the reference
/// block-per-lane kernel. Needs Mac A/B confirmation on glm/qwen35 shapes.
pub(super) fn q5_mv() -> bool {
    crate::runtime::get().q5_mv
}

pub(super) fn q5_kernel() -> &'static str {
    if q5_mv() {
        "matvec_q5_k_mv"
    } else {
        "matvec_q5_k"
    }
}

/// The q8_0 llama-structure matvec. Default ON: GLM expert/shared down and
/// qwen35moe GDN/attn projections. Carries SWIGLU_X for decode fusion.
/// `ALLPAKA_Q8_MV=0` restores the packed-load reference kernel.
pub(super) fn q8_mv() -> bool {
    crate::runtime::get().q8_mv
}

pub(super) fn q8_kernel() -> &'static str {
    if q8_mv() {
        "matvec_q8_0_mv"
    } else {
        "matvec_q8_0"
    }
}

/// Prefill attention over simdgroup MMA tiles (attend_mm, llama-style: K/V
/// read directly by the MMAs, online softmax). Default ON: standalone
/// 0.278 vs 0.742 ms/dispatch for attend_rows_t8 at m=480 (qwen3-30b
/// shape), exact to 1e-5. `ALLPAKA_ATTN_MM=0` reverts.
pub(super) fn attend_mm() -> bool {
    crate::runtime::get().attend_mm
}

/// Prefill attention row tiling: `ALLPAKA_ATTN_T4=0` reverts to the
/// row-per-threadgroup kernel.
pub(super) fn attend_rows_kernel() -> &'static str {
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

/// The llama-structure decode attention. Default ON: four-position-in-flight
/// flash-attn vec path. `ALLPAKA_ATTN_MV=0` reverts to the position-per-step
/// kernel.
pub(super) fn attend_kernel() -> &'static str {
    static MV: OnceLock<bool> = OnceLock::new();
    if *MV.get_or_init(|| std::env::var("ALLPAKA_ATTN_MV").map_or(true, |v| v != "0")) {
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
pub(super) fn attend_tg() -> u64 {
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
pub(super) fn probe_skip(what: &str) -> bool {
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
pub(super) fn q2_kernel() -> &'static str {
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
pub(super) fn gpu_route() -> bool {
    crate::runtime::get().gpu_route
}

/// The decode-side combine fold into the next residual norm
/// (combine_resnorm). OFF by default: measured SLOWER on M4 Max
/// (qwen3-30b decode 78.5 -> 75.0 tok/s, +20 ms GPU executing over 32
/// tokens) - the single-threadgroup fused kernel walks all expert slots'
/// down rows with one thread's latency chain, which costs more than the
/// dispatch and barrier it removes. `ALLPAKA_CFUSE=1` re-enables.
pub(super) fn cfuse() -> bool {
    crate::runtime::get().cfuse
}

/// The decode gate+up dual dispatch: one INDEXED launch covers both
/// matrices (DUAL_GW function constant). OFF by default: qwen3-30b measured
/// 105.0 tok/s versus 113.7 for two concurrent dispatches on M4 Max.
/// `ALLPAKA_DECODE_GUFUSE=1` enables the diagnostic variant.
pub(super) fn gufuse() -> bool {
    crate::runtime::get().gufuse
}

/// The fused single-threadgroup router_topk in decode. OFF by default: the
/// fused kernel walks the experts serially inside one threadgroup and is
/// latency-bound - on qwen3-30b the parallel matvec_f32 + tiny top-k pair
/// (one extra dispatch and barrier, same buffer) measured 113.8 vs 79.2
/// tok/s decode (265 vs 388 ms GPU executing over 32 tokens).
/// `ALLPAKA_RTOPK=1` re-enables the fused kernel.
pub(super) fn rtopk_fused() -> bool {
    crate::runtime::get().rtopk_fused
}

/// The decode-side fold of the FFN residual norm into the router top-k
/// (resnorm_router). OFF by default: under max-performance `normflag`,
/// enabling costs ~20–27% decode tok/s on GLM-Air / qwen3-30b (2026-09-12)
/// despite fewer dispatches. `ALLPAKA_RFUSE=1` enables only when
/// `normflag` is also off (`moe_plain`).
pub(super) fn rfuse() -> bool {
    crate::runtime::get().rfuse
}

/// Fold the GLM shared expert's Q8_0 down into the indexed expert-down
/// launch (+1 parallel slot). OFF by default until cool A/B confirms a win;
/// `ALLPAKA_SHARED_TAIL=1` enables.
pub(super) fn shared_tail() -> bool {
    crate::runtime::get().shared_tail
}

/// Stagger GLM shared Q5 gate/up out of the expert Q4 window: expert
/// gate/up → barrier → expert Q8 down ∥ shared Q5 gate/up → barrier →
/// shared Q8 down. OFF by default until cool Terminal A/B
/// (`scripts/glm-ab-stagger.sh`) confirms a win; `ALLPAKA_SHARED_STAGGER=1`
/// enables. Disabled automatically when `SHARED_TAIL` folds shared down
/// into the indexed expert-down launch (shared gate/up must finish first).
pub(super) fn shared_stagger() -> bool {
    crate::runtime::get().shared_stagger
}

/// Rows per SIMD group for `matvec_q8_0_mv` (NR0 = 32/LPR). Default 2;
/// sweep with `ALLPAKA_Q8_NR0` ∈ {1,2,4} (GLM expert down is the hot shape).
pub(super) fn q8_nr0() -> usize {
    crate::runtime::get().q8_nr0
}

/// K-step 64 for the LL mm kernels: `ALLPAKA_MM_K64=1` to enable.
pub(super) fn mm_k64() -> bool {
    crate::runtime::get().mm_k64
}

/// The software-pipelined double-buffer K-loop in the q4_k mm kernels
/// (mmllp_q4_k, mmllpg/ps_id_q4_k). Default ON. `ALLPAKA_MM_PIPE=0` reverts
/// to the two-barrier llama-structure loop.
pub(super) fn mm_pipe() -> bool {
    crate::runtime::get().mm_pipeline
}

/// K-step 64 for the mmid (MoE) kernels only: `ALLPAKA_MMID_K64=1`. The FFN
/// K-loop runs 128 iterations at NK=32 with two threadgroup barriers each;
/// halving them costs shared memory but the MoE stages are latency-bound.
pub(super) fn mmid_k64() -> bool {
    crate::runtime::get().mmid_k64
}

pub(super) fn mm_ll() -> bool {
    crate::runtime::get().mm_ll
}

/// ALLPAKA_NO_BARRIER=1 skips in-buffer memory barriers (FFN path only) to
/// measure their cost. Timing probe only - results are racy garbage.
/// ALLPAKA_NO_STAGE_BARRIER=1 drops only the FFN stage-to-stage barriers
/// (route and combine barriers stay, so the mm stages still do full work
/// on valid tables; the down stage may read torn gate data - timing only).
pub(super) fn no_stage_barrier() -> bool {
    crate::runtime::get().no_stage_barrier
}

pub(super) fn no_barrier() -> bool {
    let p = crate::runtime::get();
    p.no_barrier_probe || p.serial
}

/// ALLPAKA_PF_SKIP="qkv,attend,wo,fnorm,gate_up,down,combine,norms": drop the
/// named prefill dispatches (same command-buffer structure, stage just never
/// encodes). Wall(baseline) - wall(skip) prices that stage in real onebuf
/// conditions - the only per-stage measurement this device supports (counter
/// sampling is dispatch-boundary-only, which this GPU lacks). Results are
/// numerically wrong by construction; timings are the point.
pub(super) fn pf_skip(stage: &str) -> bool {
    static SKIP: OnceLock<Vec<String>> = OnceLock::new();
    SKIP.get_or_init(|| {
        std::env::var("ALLPAKA_PF_SKIP")
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_owned())
            .collect()
    })
    .iter()
    .any(|s| s == stage)
}

/// ALLPAKA_SERIAL=1: encode the prefill command buffers with
/// MTLDispatchType::Serial and drop the explicit barriers - serial dispatch
/// gives in-order execution (ordering without the full pipeline flush that
/// memory_barrier_with_resources pays). llama.cpp encodes this way.
pub(super) fn serial_dispatch() -> metal::MTLDispatchType {
    if crate::runtime::get().serial {
        metal::MTLDispatchType::Serial
    } else {
        metal::MTLDispatchType::Concurrent
    }
}

/// Use specialized BN=16 grouped MoE tile when average rows per active expert
/// <= 24 (or forced via `ALLPAKA_MMID_BN16=1`, disabled via `=0`).
pub(super) fn mmid_use_bn16(avg_rows: usize) -> bool {
    static FORCE: OnceLock<Option<bool>> = OnceLock::new();
    match *FORCE.get_or_init(|| std::env::var("ALLPAKA_MMID_BN16").ok().map(|v| v != "0")) {
        Some(b) => b,
        None => mm_ll() && !(mm_k64() || mmid_k64()) && avg_rows <= 24,
    }
}

/// Down-projection kernel with swiglu folded into B staging: gate and up
/// stay raw in y, the mm applies silu(g)*u while loading. q4_k only;
/// anything else keeps the standalone swiglu pass.
pub(super) fn mmid_swiglu_kernel_for(ty: GgmlType, bn16: bool) -> Option<(&'static str, usize)> {
    if !mm_ll() || mm_k64() {
        return None;
    }
    match (ty, bn16) {
        (GgmlType::Q4K, false) => Some((
            if mm_pipe() {
                "mmllps_id_q4_k"
            } else {
                "mmlls_id_q4_k"
            },
            64,
        )),
        (GgmlType::Q4K, true) => Some((
            if mm_pipe() {
                "mmllps_id_q4_k_bn16"
            } else {
                "mmlls_id_q4_k_bn16"
            },
            64,
        )),
        _ => None,
    }
}

pub(super) fn mmid_kernel_for(
    ty: GgmlType,
    gather: bool,
    bn16: bool,
) -> Option<(&'static str, usize)> {
    if mm_ll() {
        let k64 = mm_k64() || mmid_k64();
        if bn16 && !k64 {
            let name = match (ty, gather) {
                (GgmlType::Q8_0, false) => "mmll_id_q8_0_bn16",
                (GgmlType::Q2K, false) => "mmll_id_q2_k_bn16",
                (GgmlType::Q3K, false) => "mmll_id_q3_k_bn16",
                (GgmlType::Q4K, false) => "mmll_id_q4_k_bn16",
                (GgmlType::Q5K, false) => "mmll_id_q5_k_bn16",
                (GgmlType::Q6K, false) => "mmll_id_q6_k_bn16",
                (GgmlType::Q5_0, false) => "mmll_id_q5_0_bn16",
                (GgmlType::Q8_0, true) => "mmllg_id_q8_0_bn16",
                (GgmlType::Q2K, true) => "mmllg_id_q2_k_bn16",
                (GgmlType::Q3K, true) => "mmllg_id_q3_k_bn16",
                (GgmlType::Q4K, true) => {
                    if mm_pipe() {
                        "mmllpg_id_q4_k_bn16"
                    } else {
                        "mmllg_id_q4_k_bn16"
                    }
                }
                (GgmlType::Q5K, true) => "mmllg_id_q5_k_bn16",
                (GgmlType::Q6K, true) => "mmllg_id_q6_k_bn16",
                (GgmlType::Q5_0, true) => "mmllg_id_q5_0_bn16",
                _ => return None,
            };
            return Some((name, 64));
        }
        let name = match (ty, gather, k64) {
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
