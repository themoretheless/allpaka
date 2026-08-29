//! The per-layer attention cache.
//!
//! Deliberately behind a narrow interface: the forward pass asks a layer's
//! cache to *store* what attention needs and to *view* what has been stored,
//! and never assumes the storage is literally K and V per head. That is the
//! seam DeepSeek's MLA slots into later - MLA caches a compressed latent
//! instead of full K/V, and only this module will know.

use allpaka_backend::ops;
use std::fmt;

/// Concrete reason why cache storage cannot participate in a GPU fast path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuViewError {
    pub cache: &'static str,
    pub layer: Option<usize>,
    pub reason: &'static str,
}

impl fmt::Display for GpuViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layer {
            Some(layer) => write!(f, "{} layer {}: {}", self.cache, layer, self.reason),
            None => write!(f, "{}: {}", self.cache, self.reason),
        }
    }
}

impl std::error::Error for GpuViewError {}

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
/// Gated-delta-net per-layer state (qwen35moe): the depthwise-conv window
/// (d_conv-1 previous qkv rows per layer) and the deltanet recurrence state
/// (dt_rank matrices of d_state x d_state). Only present when the model has
/// linear-attention layers. Plain f32 in one page-aligned allocation, so the
/// GPU decode kernels read and update it in place (same trick as the KV
/// cache): CPU prefill and GPU decode share the memory, no copies at the
/// boundary.
pub struct SsmCache {
    store: Region,
    /// The conv half's element count; the state half follows.
    conv_elems: usize,
    state_elems: usize,
    /// The same memory, wrapped for the GDN kernels; None without a GPU.
    shared: Option<allpaka_backend::gpu::SharedRegion>,
    /// Per-row rollback slots for MTP verification: `n_slots` copies of the
    /// whole [conv | state] region, written per chunk row by the GPU batch
    /// kernels (and by the CPU fallback) while armed. Restoring slot i rolls
    /// the recurrence back to "after row i" - the speculative round's
    /// accepted prefix - without a replay.
    slot_store: Option<Region>,
    slot_shared: Option<allpaka_backend::gpu::SharedRegion>,
    slots_armed: bool,
    pub conv_channels: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub d_state: usize,
}

impl SsmCache {
    pub fn new(n_layers: usize, d_conv: usize, conv_channels: usize, dt_rank: usize, d_state: usize) -> Self {
        let conv_elems = n_layers * (d_conv - 1) * conv_channels;
        let state_elems = n_layers * dt_rank * d_state * d_state;
        let store = Region::zeroed((conv_elems + state_elems) * 2);
        let shared = allpaka_backend::gpu::wrap_region(store.as_bytes());
        Self {
            store,
            conv_elems,
            state_elems,
            shared,
            slot_store: None,
            slot_shared: None,
            slots_armed: false,
            conv_channels,
            d_conv,
            dt_rank,
            d_state,
        }
    }

    /// Elements of the whole region (conv + state); one slot's stride.
    pub fn total_elems(&self) -> usize {
        self.conv_elems + self.state_elems
    }

    /// Allocate (once) and arm the rollback slots for a speculative verify.
    /// The buffer is `slots` whole-region copies; re-arming a smaller count
    /// reuses it.
    pub fn arm_slots(&mut self, slots: usize) {
        if self.slot_store.is_none() {
            let store = Region::zeroed(slots * self.total_elems() * 2);
            self.slot_shared = allpaka_backend::gpu::wrap_region(store.as_bytes());
            self.slot_store = Some(store);
        }
        self.slots_armed = true;
    }

    /// Stop writing slots (the next plain prefill leaves them stale).
    pub fn disarm_slots(&mut self) {
        self.slots_armed = false;
    }

    /// Both GPU regions in one borrow for the fused prefill: the state
    /// region (wrapped if needed) and the armed rollback slots.
    pub fn regions_for_gpu(
        &mut self,
    ) -> (
        Option<&allpaka_backend::gpu::SharedRegion>,
        Option<(&allpaka_backend::gpu::SharedRegion, usize)>,
    ) {
        if self.shared.is_none() {
            self.shared = allpaka_backend::gpu::wrap_region(self.store.as_bytes());
        }
        let slots = self.slots_region();
        (self.shared.as_ref(), slots)
    }

    /// The wrapped slot buffer and the slot stride in elements, when armed.
    pub fn slots_region(&self) -> Option<(&allpaka_backend::gpu::SharedRegion, usize)> {
        if !self.slots_armed {
            return None;
        }
        Some((self.slot_shared.as_ref()?, self.total_elems()))
    }

    /// Slot `i` (conv + state after chunk row i).
    pub fn slot(&self, i: usize) -> &[f32] {
        let store = self.slot_store.as_ref().expect("slots not allocated");
        let s = store.as_slice();
        let floats = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len() / 2) };
        let n = self.total_elems();
        &floats[i * n..(i + 1) * n]
    }

    /// Roll the recurrence back to "after chunk row `i`": one contiguous
    /// copy, the slot layout mirrors the region.
    pub fn restore_slot(&mut self, i: usize) {
        let n = self.total_elems();
        unsafe {
            let sp = self.slot(i).as_ptr();
            let dp = self.store.as_mut_slice().as_mut_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(sp, dp, n);
        }
        self.slots_armed = false;
    }

    fn floats(&self) -> &[f32] {
        let s = self.store.as_slice();
        unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len() / 2) }
    }

    fn floats_mut(&mut self) -> &mut [f32] {
        let s = self.store.as_mut_slice();
        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut f32, s.len() / 2) }
    }

    /// Every layer's conv windows, `n_layers * (d_conv-1) * conv_channels`.
    pub fn conv_all_mut(&mut self) -> &mut [f32] {
        let n = self.conv_elems;
        &mut self.floats_mut()[..n]
    }

    /// Every layer's deltanet states, after the conv half.
    pub fn state_all_mut(&mut self) -> &mut [f32] {
        let (c, s) = (self.conv_elems, self.state_elems);
        &mut self.floats_mut()[c..c + s]
    }

    /// Both halves at once, disjoint (the recurrence touches window and
    /// state in the same loop).
    pub fn both_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        let c = self.conv_elems;
        let all = self.floats_mut();
        let (conv, state) = all.split_at_mut(c);
        (conv, state)
    }

    /// A copy of the whole recurrent state (conv windows + deltanet
    /// states), for the serve prompt cache: unlike the KV cache, the
    /// recurrence cannot be rolled back by truncating, so a prefix hit has
    /// to restore a snapshot taken at that prefix's end.
    pub fn snapshot(&self) -> Vec<f32> {
        self.floats()[..self.conv_elems + self.state_elems].to_vec()
    }

    /// Raw pointers for the CPU path's per-row slot write: this layer's
    /// conv/state source pointers (already offset), the slot buffer base,
    /// and the geometry (conv_off, state_off, n_conv, n_state, slot
    /// stride). Addresses are stable for the region's lifetime; the caller
    /// must not outlive the cache.
    #[allow(clippy::type_complexity)]
    pub fn slot_write_ptrs(
        &mut self,
        li: usize,
    ) -> Option<(*const f32, *const f32, *mut f32, usize, usize, usize, usize, usize)> {
        if !self.slots_armed {
            return None;
        }
        let n_conv = (self.d_conv - 1) * self.conv_channels;
        let n_state = self.dt_rank * self.d_state * self.d_state;
        let conv_off = li * n_conv;
        let state_off = self.conv_elems + li * n_state;
        let total = self.total_elems();
        unsafe {
            let sp = self.store.as_slice().as_ptr() as *const f32;
            let dp = self
                .slot_store
                .as_mut()
                .expect("slots armed without buffer")
                .as_mut_slice()
                .as_mut_ptr() as *mut f32;
            Some((
                sp.add(conv_off),
                sp.add(state_off),
                dp,
                conv_off,
                state_off,
                n_conv,
                n_state,
                total,
            ))
        }
    }

    /// Restore a [`snapshot`]; the slice must come from the same-shaped
    /// cache (same layer count and dims).
    pub fn restore(&mut self, snap: &[f32]) {
        assert_eq!(
            snap.len(),
            self.conv_elems + self.state_elems,
            "ssm snapshot shape mismatch"
        );
        self.floats_mut()[..snap.len()].copy_from_slice(snap);
    }

    /// This layer's conv window rows, `(d_conv - 1) * conv_channels`.
    pub fn conv_layer(&mut self, layer: usize) -> &mut [f32] {
        let n = (self.d_conv - 1) * self.conv_channels;
        let at = layer * n;
        &mut self.conv_all_mut()[at..at + n]
    }

    /// This layer's deltanet state, `dt_rank * d_state * d_state`.
    pub fn state_layer(&mut self, layer: usize) -> &mut [f32] {
        let n = self.dt_rank * self.d_state * self.d_state;
        let at = layer * n;
        &mut self.state_all_mut()[at..at + n]
    }

    /// The wrapped region for the GDN decode kernels. Conv window of layer
    /// `i` sits at element offset `i * (d_conv-1) * conv_channels`, its
    /// deltanet state at `conv_elems + i * dt_rank * d_state * d_state`.
    pub fn gpu_view(&mut self) -> Option<&allpaka_backend::gpu::SharedRegion> {
        if self.shared.is_none() {
            self.shared = allpaka_backend::gpu::wrap_region(self.store.as_bytes());
        }
        self.shared.as_ref()
    }

    /// Element offset of a layer's deltanet state inside the region.
    pub fn state_off(&self, layer: usize) -> usize {
        self.conv_elems + layer * self.dt_rank * self.d_state * self.d_state
    }

    /// Element offset of a layer's conv window inside the region.
    pub fn conv_off(&self, layer: usize) -> usize {
        layer * (self.d_conv - 1) * self.conv_channels
    }
}

/// Per-position K and V for one transformer layer: `kv_dim` floats each.
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
        self.gpu_view_checked(layer).ok()
    }

    /// Fast-path view that never erases the reason Metal wrapping failed.
    pub fn gpu_view_checked(
        &mut self,
        layer: usize,
    ) -> Result<(&allpaka_backend::gpu::SharedRegion, usize, usize), GpuViewError> {
        if self.shared.is_none() {
            self.shared = allpaka_backend::gpu::wrap_region(self.store.as_bytes());
        }
        let shared = self.shared.as_ref().ok_or(GpuViewError {
            cache: "kv-cache",
            layer: Some(layer),
            reason: if allpaka_backend::gpu::is_attached() {
                "Metal rejected the page-aligned shared region"
            } else {
                "no Metal device or model mapping is attached"
            },
        })?;
        Ok((
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

    /// Store at an explicit position that may already be sealed. Only for
    /// the MTP draft layer's slot: the trunk's verify batch seals positions
    /// in the trunk layers, and the MTP block appends its own layer at the
    /// same positions afterwards. Truncated positions are rewritten, never
    /// read past `len`, so no clearing is needed.
    pub fn store_at_rewrite(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert_eq!(v.len(), self.kv_dim);
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
