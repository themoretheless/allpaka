//! The per-layer attention cache.
//!
//! Deliberately behind a narrow interface: the forward pass asks a layer's
//! cache to *store* what attention needs and to *view* what has been stored,
//! and never assumes the storage is literally K and V per head. That is the
//! seam DeepSeek's MLA slots into later - MLA caches a compressed latent
//! instead of full K/V, and only this module will know.

use allpaka_backend::ops;

/// Page-aligned f16 storage, so the GPU can read it with no copy at all.
///
/// A `Vec` would do for the CPU, but `newBufferWithBytesNoCopy` insists on a
/// page-aligned pointer over whole pages, and the attention kernel reads the
/// cache the CPU just wrote. One allocation covers every layer's K and V.
struct Region {
    ptr: *mut u16,
    elems: usize,
    layout: std::alloc::Layout,
}

// The region is plain memory owned by this struct; the raw pointer is an
// implementation detail of the alignment, not shared mutable state.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    const PAGE: usize = 16384;

    fn zeroed(elems: usize) -> Self {
        let bytes = (elems * 2).div_ceil(Self::PAGE) * Self::PAGE;
        let layout = std::alloc::Layout::from_size_align(bytes, Self::PAGE)
            .expect("kv cache layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut u16;
        assert!(!ptr.is_null(), "out of memory for a {bytes}-byte kv cache");
        Self { ptr, elems: bytes / 2, layout }
    }

    fn as_slice(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.elems) }
    }

    fn as_mut_slice(&mut self) -> &mut [u16] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.elems) }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.elems * 2) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout) }
    }
}

/// Plain K/V storage for one model: `[layer][pos][kv_dim]`, f16, with K for
/// every layer followed by V for every layer in one allocation.
///
/// Half precision costs nothing measurable in the logits - attention scores
/// are exponentiated and normalised, and llama.cpp caches f16 by default -
/// and halves the one tensor that is both written and fully re-read every
/// single step.
pub struct KvCache {
    store: Region,
    /// Element offset of the V half.
    v_base: usize,
    layer_stride: usize,
    kv_dim: usize,
    len: usize,
    capacity: usize,
    /// The same memory, wrapped for the attention kernel. None without a GPU,
    /// which leaves attention on its CPU path.
    shared: Option<allpaka_backend::gpu::SharedRegion>,
}

impl KvCache {
    pub fn new(n_layers: usize, kv_dim: usize, capacity: usize) -> Self {
        let layer_stride = kv_dim * capacity;
        let half = n_layers * layer_stride;
        let store = Region::zeroed(half * 2);
        let shared = allpaka_backend::gpu::wrap_region(store.as_bytes());
        Self {
            store,
            v_base: half,
            layer_stride,
            kv_dim,
            len: 0,
            capacity,
            shared,
        }
    }

    /// The wrapped region and this layer's K and V *element* offsets inside
    /// it, when the GPU can read the cache in place. Elements rather than
    /// bytes because the kernel indexes them, which keeps the buffer bound at
    /// offset zero and out of Metal's alignment rules.
    pub fn gpu_view(
        &mut self,
        layer: usize,
    ) -> Option<(&allpaka_backend::gpu::SharedRegion, usize, usize)> {
        if self.shared.is_none() {
            self.shared = allpaka_backend::gpu::wrap_region(self.store.as_bytes());
        }
        let shared = self.shared.as_ref()?;
        Some((
            shared,
            layer * self.layer_stride,
            self.v_base + layer * self.layer_stride,
        ))
    }

    /// [`gpu_view`] for callers that only hold `&self`: the offsets are pure
    /// arithmetic, and the region is already wrapped whenever a device
    /// existed at construction. Returns None only if it is not - the same
    /// "no device" answer, without a mutable borrow. CPU attention runs
    /// under `thread::scope` with `&self` shared across threads, and the
    /// whole-token path needs the region and the per-layer offsets at once.
    pub fn gpu_view_ref(
        &self,
        layer: usize,
    ) -> Option<(&allpaka_backend::gpu::SharedRegion, usize, usize)> {
        Some((
            self.shared.as_ref()?,
            layer * self.layer_stride,
            self.v_base + layer * self.layer_stride,
        ))
    }

    fn k_range(&self, layer: usize) -> std::ops::Range<usize> {
        let at = layer * self.layer_stride;
        at..at + self.layer_stride
    }

    fn v_range(&self, layer: usize) -> std::ops::Range<usize> {
        let at = self.v_base + layer * self.layer_stride;
        at..at + self.layer_stride
    }

    /// Positions stored so far. Every layer advances together; a partially
    /// appended step is a bug, not a state.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Store one position's k and v for one layer. The position is implicit:
    /// always appended at the end, `advance` seals the step.
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert!(self.len < self.capacity, "kv cache full at {} positions", self.capacity);
        self.store_at(layer, self.len, k, v);
    }

    /// Store at an explicit position at or past the sealed length.
    ///
    /// Batched prefill writes a whole chunk of positions per layer before any
    /// of them is sealed; `commit` then seals the chunk in one step.
    pub fn store_at(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert_eq!(v.len(), self.kv_dim);
        assert!(pos >= self.len, "position {pos} is already sealed");
        assert!(pos < self.capacity, "position {pos} past capacity {}", self.capacity);
        let at = pos * self.kv_dim;
        let (kr, vr) = (self.k_range(layer), self.v_range(layer));
        let kv_dim = self.kv_dim;
        let store = self.store.as_mut_slice();
        ops::f16::from_f32(k, &mut store[kr.start + at..kr.start + at + kv_dim]);
        ops::f16::from_f32(v, &mut store[vr.start + at..vr.start + at + kv_dim]);
    }

    /// Seal the current position after every layer has stored into it.
    pub fn advance(&mut self) {
        self.len += 1;
    }

    /// Seal everything up to `len` after a batch of `store_at` writes.
    pub fn commit(&mut self, len: usize) {
        assert!(len >= self.len && len <= self.capacity);
        self.len = len;
    }

    /// Forget everything after the first `keep` positions.
    ///
    /// This is what makes prefix reuse work for a chat: histories diverge in
    /// the middle (a think block the client stripped, an edited message), and
    /// rolling back to the common prefix keeps every token before the fork.
    /// Storage is not cleared - later stores overwrite it.
    pub fn truncate(&mut self, keep: usize) {
        assert!(keep <= self.len, "cannot truncate {} up to {keep}", self.len);
        self.len = keep;
    }

    /// K of one kv head at one stored position, still in half.
    pub fn k_at(&self, layer: usize, pos: usize, kv_head: usize, head_dim: usize) -> &[u16] {
        let start = self.k_range(layer).start + pos * self.kv_dim + kv_head * head_dim;
        &self.store.as_slice()[start..start + head_dim]
    }

    /// V of one kv head at one stored position, still in half.
    pub fn v_at(&self, layer: usize, pos: usize, kv_head: usize, head_dim: usize) -> &[u16] {
        let start = self.v_range(layer).start + pos * self.kv_dim + kv_head * head_dim;
        &self.store.as_slice()[start..start + head_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_vectors_come_back_from_the_right_layer_position_and_head() {
        let mut c = KvCache::new(2, 4, 8); // 2 kv heads of dim 2
        c.store(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        c.store(1, &[9.0, 9.0, 9.0, 9.0], &[0.0, 0.0, 0.0, 0.0]);
        c.advance();
        c.store(0, &[10.0, 20.0, 30.0, 40.0], &[0.5, 0.5, 0.5, 0.5]);
        c.store(1, &[0.0; 4], &[0.0; 4]);
        c.advance();

        // Read back through the half conversion; the values above are all
        // exactly representable, so this stays an equality test.
        let read = |v: &[u16]| -> Vec<f32> { v.iter().map(|&h| ops::f16::to_f32(h)).collect() };
        assert_eq!(c.len(), 2);
        assert_eq!(read(c.k_at(0, 0, 0, 2)), vec![1.0, 2.0]);
        assert_eq!(read(c.k_at(0, 0, 1, 2)), vec![3.0, 4.0]);
        assert_eq!(read(c.v_at(0, 0, 1, 2)), vec![7.0, 8.0]);
        assert_eq!(read(c.k_at(0, 1, 0, 2)), vec![10.0, 20.0]);
        assert_eq!(read(c.k_at(1, 0, 0, 2)), vec![9.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "kv cache full")]
    fn overflowing_the_capacity_panics_rather_than_corrupting() {
        let mut c = KvCache::new(1, 2, 1);
        c.store(0, &[1.0, 2.0], &[3.0, 4.0]);
        c.advance();
        c.store(0, &[5.0, 6.0], &[7.0, 8.0]);
    }
}
