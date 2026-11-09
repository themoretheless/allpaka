//! Caller-owned host memory mirrored in device memory (the KV cache and SSM
//! state), with explicit ranged upload and download across the PCIe boundary.

use super::*;

pub struct SharedRegion {
    pub(super) buf: CudaSlice<u8>,
    pub(crate) len: usize,
    #[allow(dead_code)]
    pub(super) host: usize,
}

impl SharedRegion {
    /// Memory architecture backing this region (`dUMA` discrete VRAM buffer).
    #[inline]
    pub const fn architecture(&self) -> crate::memory::MemoryArchitecture {
        crate::memory::MemoryArchitecture::Duma
    }

    /// False on dUMA: `buf` is a separate `CudaSlice<u8>` in device VRAM, not
    /// a zero-copy alias of the host allocation.
    #[inline]
    pub const fn is_unified(&self) -> bool {
        false
    }

    /// True on dUMA: host and device buffers are physically distinct and must
    /// be synchronized via `upload_region_range` (H2D) / `download_region_range` (D2H).
    #[inline]
    pub const fn is_discrete(&self) -> bool {
        true
    }
}

unsafe impl Send for SharedRegion {}
unsafe impl Sync for SharedRegion {}

pub fn wrap_region(region: &[u8]) -> Option<SharedRegion> {
    const PAGE: usize = 16384;
    if region.is_empty() || region.as_ptr() as usize % PAGE != 0 || region.len() % PAGE != 0 {
        return None;
    }
    with_gpu(|gpu| {
        let mut buf = unsafe { gpu.stream.alloc::<u8>(region.len()) }.ok()?;
        gpu.stream.memcpy_htod(region, &mut buf).ok()?;
        Some(SharedRegion {
            buf,
            len: region.len(),
            host: region.as_ptr() as usize,
        })
    })
}

pub fn upload_region_range(region: &mut SharedRegion, offset: usize, data: &[u8]) -> bool {
    let Some(end) = offset.checked_add(data.len()) else {
        return false;
    };
    if end > region.len {
        return false;
    }
    if data.is_empty() {
        return true;
    }
    with_gpu(|gpu| {
        gpu.stream
            .memcpy_htod(data, &mut region.buf.slice_mut(offset..end))
            .ok()?;
        Some(())
    })
    .is_some()
}

pub fn download_region_range(region: &SharedRegion, offset: usize, dst: &mut [u8]) -> bool {
    let Some(end) = offset.checked_add(dst.len()) else {
        return false;
    };
    if end > region.len {
        return false;
    }
    if dst.is_empty() {
        return true;
    }
    with_gpu(|gpu| {
        gpu.stream
            .memcpy_dtoh(&region.buf.slice(offset..end), dst)
            .ok()?;
        Some(())
    })
    .is_some()
}

pub fn decode_attention_capacity_safe(capacity: usize) -> bool {
    let native = std::env::var("ALLPAKA_NATIVE_ATTEND")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    !crate::gpu::cuda::ggml::enabled()
        || native
        || capacity >= crate::gpu::cuda::ggml::fa_kv_floor()
}

pub fn minimum_kv_capacity() -> usize {
    if crate::gpu::cuda::ggml::enabled() {
        crate::gpu::cuda::ggml::fa_kv_floor()
    } else {
        1
    }
}
