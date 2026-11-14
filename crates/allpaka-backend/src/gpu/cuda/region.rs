//! Caller-owned host memory mirrored in device memory (the KV cache and SSM
//! state), with explicit ranged upload and download across the PCIe boundary.

use super::*;

pub struct SharedRegion {
    pub(super) buf: CudaSlice<u8>,
    pub(crate) len: usize,
    /// Distinct for every device buffer this process ever wrapped. State
    /// derived from a region (a captured decode graph, cached KV offsets) is
    /// keyed by this, never by where the struct lives: a session that is
    /// replaced in place gets a new buffer at the old address.
    pub(crate) id: u64,
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

    /// Device address mixed with the length: what a captured prefill graph
    /// bakes in for this region (0 if the device is unavailable).
    pub fn graph_id(&self) -> u64 {
        with_gpu(|gpu| {
            let (p, g) = DevicePtr::device_ptr(&self.buf, &gpu.stream);
            drop(g);
            Some(p ^ ((self.len as u64).rotate_left(40)))
        })
        .unwrap_or(0)
    }
}

// SAFETY: the fields are a `CudaSlice<u8>` (cudarc already declares it Send + Sync) and
// two `usize`s; `host` is only an address recorded for diagnostics and is never turned
// back into a pointer, so no host memory is reachable through a `SharedRegion`. The
// device buffer is only touched through `with_gpu`, i.e. under the global `GPU` mutex,
// and `&SharedRegion` methods only read `len` or pass `&buf` to stream-ordered copies.
// These impls are therefore what the auto traits would already derive.
unsafe impl Send for SharedRegion {}
// SAFETY: see the `Send` impl above; shared access never mutates any field.
unsafe impl Sync for SharedRegion {}

pub fn wrap_region(region: &[u8]) -> Option<SharedRegion> {
    const PAGE: usize = 16384;
    if region.is_empty() || region.as_ptr() as usize % PAGE != 0 || region.len() % PAGE != 0 {
        return None;
    }
    with_gpu(|gpu| {
        let mut buf = unsafe { gpu.stream.alloc::<u8>(region.len()) }.ok()?;
        gpu.stream.memcpy_htod(region, &mut buf).ok()?;
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Some(SharedRegion {
            buf,
            len: region.len(),
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
    let native = crate::runtime::get().cuda_native_attend;
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
