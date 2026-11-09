//! Optional L2 cache persistence and streaming hints for weight and
//! activation ranges.

use super::*;

pub(super) fn l2_persist_wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ALLPAKA_L2_PERSIST")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Pin the Q8 activation buffer in the persisting L2 slice so thousands of
/// MMVQ CTAs re-hit X instead of thrashing it with weight traffic.
pub(super) fn apply_q8_l2_persist(gpu: &mut CudaGpu, stream: &CudaStream) {
    if !l2_persist_wanted() {
        return;
    }
    let nbytes = gpu.q8_q_cap;
    if nbytes == 0 || gpu.q8_l2_bytes == nbytes {
        return;
    }
    use cudarc::driver::sys::{
        cuCtxSetLimit, cuStreamSetAttribute, CUaccessPolicyWindow, CUaccessProperty, CUlimit,
        CUstreamAttrID, CUstreamAttrValue,
    };
    static LIMIT_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LIMIT_LOGGED.swap(true, Ordering::Relaxed) {
        // Keep a small persisting carve-out (Q8 X is tens of KiB).
        let rc = unsafe { cuCtxSetLimit(CUlimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 2 << 20) };
        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("cuda: L2 persist for Q8 X (ALLPAKA_L2_PERSIST=1)");
        } else {
            eprintln!("cuda: L2 persist limit failed ({rc:?}); continuing without");
            return;
        }
    }
    let (base, g) = DevicePtr::device_ptr(&gpu.q8_q, stream);
    drop(g);
    let win = CUaccessPolicyWindow {
        base_ptr: base as usize as *mut std::ffi::c_void,
        num_bytes: nbytes,
        hitRatio: 1.0,
        hitProp: CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING,
        missProp: CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
    };
    let mut val = unsafe { std::mem::zeroed::<CUstreamAttrValue>() };
    unsafe {
        val.accessPolicyWindow = win;
        let rc = cuStreamSetAttribute(
            stream.cu_stream(),
            CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &val,
        );
        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            gpu.q8_l2_bytes = nbytes;
        }
    }
}

pub(super) fn l2_stream_w_wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ALLPAKA_L2_STREAM_W")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Hint weight traffic as STREAMING so it prefers not to pin L2 (leaves more
/// for the reused Q8 activation). Mutually exclusive with Q8 persist window.
pub(super) fn apply_w_l2_stream(gpu: &CudaGpu, stream: &CudaStream, chunk: usize) {
    if !l2_stream_w_wanted() || l2_persist_wanted() {
        return;
    }
    use cudarc::driver::sys::{
        cuStreamSetAttribute, CUaccessPolicyWindow, CUaccessProperty, CUstreamAttrID,
        CUstreamAttrValue,
    };
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("cuda: L2 STREAMING hint on weights (ALLPAKA_L2_STREAM_W=1)");
    }
    let (base, g) = DevicePtr::device_ptr(&gpu.chunks[chunk].buf, stream);
    drop(g);
    let nbytes = gpu.chunks[chunk].len;
    if nbytes == 0 {
        return;
    }
    let win = CUaccessPolicyWindow {
        base_ptr: base as usize as *mut std::ffi::c_void,
        num_bytes: nbytes,
        hitRatio: 1.0,
        hitProp: CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
        missProp: CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
    };
    let mut val = unsafe { std::mem::zeroed::<CUstreamAttrValue>() };
    unsafe {
        val.accessPolicyWindow = win;
        let _ = cuStreamSetAttribute(
            stream.cu_stream(),
            CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &val,
        );
    }
}
