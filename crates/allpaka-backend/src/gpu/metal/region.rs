//! Caller-owned memory wrapped as a shared Metal buffer (the KV cache and
//! SSM state), with ranged upload and download for discrete-memory parity.

use super::*;

/// A CPU allocation the GPU reads in place.
///
/// The KV cache is written by the CPU every step and read by the attention
/// kernel; on unified memory neither needs a copy, so the cache is allocated
/// page-aligned and wrapped here exactly like the weight mapping is. Holding
/// the `Buffer` keeps the wrapper alive: dropping this releases it, and the
/// caller still owns the memory.
pub struct SharedRegion {
    pub(super) buf: Buffer,
    pub(super) len: usize,
}

impl SharedRegion {
    /// Memory architecture backing this region (`UMA` zero-copy shared memory).
    #[inline]
    pub const fn architecture(&self) -> crate::memory::MemoryArchitecture {
        crate::memory::MemoryArchitecture::Uma
    }

    /// True because Metal `StorageModeShared` aliases the host allocation in
    /// unified physical memory without separate device VRAM storage.
    #[inline]
    pub const fn is_unified(&self) -> bool {
        true
    }

    /// False on UMA: host and GPU share the same physical bytes.
    #[inline]
    pub const fn is_discrete(&self) -> bool {
        false
    }
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

pub fn upload_region_range(region: &mut SharedRegion, offset: usize, data: &[u8]) -> bool {
    if offset.saturating_add(data.len()) > region.len {
        return false;
    }
    if data.is_empty() {
        return true;
    }
    unsafe {
        let dst = (region.buf.contents() as *mut u8).add(offset);
        if !std::ptr::eq(dst, data.as_ptr()) {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }
    true
}

pub fn download_region_range(region: &SharedRegion, offset: usize, dst: &mut [u8]) -> bool {
    if offset.saturating_add(dst.len()) > region.len {
        return false;
    }
    if dst.is_empty() {
        return true;
    }
    unsafe {
        let src = (region.buf.contents() as *const u8).add(offset);
        if !std::ptr::eq(src, dst.as_ptr()) {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
    }
    true
}

pub fn decode_attention_capacity_safe(_capacity: usize) -> bool {
    true
}

pub fn minimum_kv_capacity() -> usize {
    1
}
