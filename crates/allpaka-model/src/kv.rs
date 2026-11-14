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

// SAFETY: `Region` uniquely owns its allocation (freed only in Drop) and has
// no interior mutability, so moving it to another thread moves that ownership
// like a `Box<[u16]>` would; the raw pointer is only there for page alignment.
unsafe impl Send for Region {}
// SAFETY: shared access only yields `&[u16]` / `&[u8]` (the `_mut` accessors
// take `&mut self`), so `&Region` across threads is like `&[u16]`.
unsafe impl Sync for Region {}

impl Region {
    const PAGE: usize = 16384;

    /// Always zeroed: the accessors hand out `&[u16]` / `&[u8]` over the
    /// whole region, and a reference to uninitialised integers is undefined
    /// behaviour. At least one page, because a zero-sized `alloc` is too
    /// (an empty cache, e.g. `capacity == 0`, still gets a valid region).
    fn zeroed(elems: usize) -> Self {
        let bytes = (elems * 2).div_ceil(Self::PAGE).max(1) * Self::PAGE;
        let layout =
            std::alloc::Layout::from_size_align(bytes, Self::PAGE).expect("kv cache layout");
        // SAFETY: `layout` has a non-zero size (at least one page) and a valid
        // power-of-two alignment, as `alloc_zeroed` requires; null is checked below.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut u16;
        assert!(!ptr.is_null(), "out of memory for a {bytes}-byte kv cache");
        Self {
            ptr,
            elems: bytes / 2,
            layout,
        }
    }

    fn as_slice(&self) -> &[u16] {
        // SAFETY: `ptr` is a live, zero-initialised allocation of `elems * 2` bytes
        // aligned to PAGE (so to u16) and owned by `self`, which the borrow keeps alive.
        // The GPU may also write this memory through a wrapped SharedRegion; callers
        // must not hold a CPU view across a backend call that writes it asynchronously.
        unsafe { std::slice::from_raw_parts(self.ptr, self.elems) }
    }

    fn as_mut_slice(&mut self) -> &mut [u16] {
        // SAFETY: as `as_slice`, and `&mut self` makes this the only live view.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.elems) }
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: the same `elems * 2` initialised bytes viewed as u8 (align 1).
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.elems * 2) }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `as_bytes`, unique through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.elems * 2) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed` with exactly this `layout` and is
        // freed once, here; no view can outlive `self`.
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
    /// Maps model layer index -> compact SSM slot index (`usize::MAX` for non-GDN layers).
    slot_map: Vec<usize>,
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
    pub fn new(
        n_layers: usize,
        d_conv: usize,
        conv_channels: usize,
        dt_rank: usize,
        d_state: usize,
    ) -> Self {
        let active = vec![true; n_layers];
        Self::with_active_layers(&active, d_conv, conv_channels, dt_rank, d_state)
    }

    pub fn with_active_layers(
        active: &[bool],
        d_conv: usize,
        conv_channels: usize,
        dt_rank: usize,
        d_state: usize,
    ) -> Self {
        let mut slot_map = Vec::with_capacity(active.len());
        let mut n_active = 0usize;
        for &is_active in active {
            if is_active {
                slot_map.push(n_active);
                n_active += 1;
            } else {
                slot_map.push(usize::MAX);
            }
        }
        let n_slots = n_active.max(1);
        let conv_elems = n_slots * (d_conv - 1) * conv_channels;
        let state_elems = n_slots * dt_rank * d_state * d_state;
        let store = Region::zeroed((conv_elems + state_elems) * 2);
        let shared = allpaka_backend::gpu::wrap_region(store.as_bytes());
        Self {
            store,
            slot_map,
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

    #[inline]
    pub fn slot_of(&self, layer: usize) -> usize {
        let s = self.slot_map[layer];
        debug_assert_ne!(s, usize::MAX, "inactive SSM layer {layer}");
        s
    }

    /// Elements of the whole region (conv + state); one slot's stride.
    pub fn total_elems(&self) -> usize {
        self.conv_elems + self.state_elems
    }

    /// Allocate (or grow) and arm the rollback slots for a speculative verify.
    /// The buffer is `slots` whole-region copies; re-arming a smaller or equal
    /// count reuses the existing allocation.
    pub fn arm_slots(&mut self, slots: usize) {
        let needed_elems = slots * self.total_elems() * 2;
        let needs_alloc = self
            .slot_store
            .as_ref()
            .is_none_or(|store| store.as_slice().len() < needed_elems);
        if needs_alloc {
            let store = Region::zeroed(needed_elems);
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
        // SAFETY: the slot region is PAGE-aligned, so its u16 storage is f32-aligned,
        // and `s.len() / 2` f32s cover exactly its `2 * s.len()` initialised bytes
        // (zeroed at allocation, any bit pattern is a valid f32). The borrow of `self`
        // keeps the region alive.
        let floats = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len() / 2) };
        let n = self.total_elems();
        &floats[i * n..(i + 1) * n]
    }

    /// Roll the recurrence back to "after chunk row `i`": one contiguous
    /// copy, the slot layout mirrors the region.
    ///
    /// * On **UMA** (Metal Unified Memory): `slot_shared` and `shared` alias
    ///   `slot_store` and `store` directly, so copying `slot(i)` into `store`
    ///   in host RAM immediately updates the GPU-visible buffer in place.
    /// * On **dUMA** (discrete CUDA VRAM): GPU batch kernels wrote slot `i`
    ///   into `slot_shared` in VRAM, so we first download slot `i` from VRAM
    ///   (when discrete) and then upload the restored state into `shared`.
    pub fn restore_slot(&mut self, i: usize) {
        let n = self.total_elems();
        let byte_len = n * 4;
        let slot_byte_off = i * byte_len;
        if let (Some(slot_sh), Some(slot_st)) =
            (self.slot_shared.as_ref(), self.slot_store.as_mut())
        {
            if slot_sh.is_discrete() {
                let dst = &mut slot_st.as_bytes_mut()[slot_byte_off..slot_byte_off + byte_len];
                allpaka_backend::gpu::download_region_range(slot_sh, slot_byte_off, dst);
            }
        }
        // SAFETY: `slot(i)` is `n` floats inside `slot_store` and the destination is
        // the first `n` floats of `store` (which holds `total_elems() == n` floats);
        // the two are separate allocations, so they cannot overlap. `self.slot(i)` ends
        // its shared borrow before `as_mut_slice` takes the unique one.
        unsafe {
            let sp = self.slot(i).as_ptr();
            let dp = self.store.as_mut_slice().as_mut_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(sp, dp, n);
        }
        if let Some(sh) = self.shared.as_mut() {
            if sh.is_discrete() {
                let bytes = &self.store.as_bytes()[..byte_len];
                allpaka_backend::gpu::upload_region_range(sh, 0, bytes);
            }
        }
        self.slots_armed = false;
    }

    fn floats(&self) -> &[f32] {
        let s = self.store.as_slice();
        // SAFETY: PAGE-aligned region, so f32-aligned; `s.len() / 2` f32s span its
        // initialised bytes, and any bits are a valid f32.
        unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len() / 2) }
    }

    fn floats_mut(&mut self) -> &mut [f32] {
        let s = self.store.as_mut_slice();
        // SAFETY: as `floats`, unique through `&mut self`.
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
    ///
    /// On **UMA**, host memory (`self.store`) is already coherent with GPU
    /// prefill/decode writes. On **dUMA**, if a discrete VRAM buffer is
    /// attached, we pull the latest recurrent state from VRAM (`D2H`).
    pub fn snapshot(&self) -> Vec<f32> {
        let n = self.conv_elems + self.state_elems;
        if let Some(sh) = self.shared.as_ref() {
            if sh.is_discrete() {
                let mut out = vec![0.0f32; n];
                // SAFETY: `out` is `n` initialised f32s, i.e. `n * 4` bytes, viewed as u8
                // (align 1) for the download; `dst` is the only live view of `out`.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
                if allpaka_backend::gpu::download_region_range(sh, 0, dst) {
                    return out;
                }
            }
        }
        self.floats()[..n].to_vec()
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
    ) -> Option<(
        *const f32,
        *const f32,
        *mut f32,
        usize,
        usize,
        usize,
        usize,
        usize,
    )> {
        if !self.slots_armed {
            return None;
        }
        let slot = self.slot_of(li);
        let n_conv = (self.d_conv - 1) * self.conv_channels;
        let n_state = self.dt_rank * self.d_state * self.d_state;
        let conv_off = slot * n_conv;
        let state_off = self.conv_elems + slot * n_state;
        let total = self.total_elems();
        // SAFETY: `conv_off + n_conv <= conv_elems` and `state_off + n_state <=
        // total` (the slot index is below the active-layer count the region was sized
        // for), so both `add`s stay inside `store`; `dp` points at `slot_store`, a
        // separate allocation. Callers may write through `dp` but must not read
        // through the conv/state pointers after taking `&mut` views of the store
        // (see the slot copy in `gdn.rs`).
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
        if let Some(sh) = self.shared.as_mut() {
            if sh.is_discrete() {
                let bytes = &self.store.as_bytes()[..snap.len() * 4];
                allpaka_backend::gpu::upload_region_range(sh, 0, bytes);
            }
        }
    }

    /// This layer's conv window rows, `(d_conv - 1) * conv_channels`.
    pub fn conv_layer(&mut self, layer: usize) -> &mut [f32] {
        let slot = self.slot_of(layer);
        let n = (self.d_conv - 1) * self.conv_channels;
        let at = slot * n;
        &mut self.conv_all_mut()[at..at + n]
    }

    /// This layer's deltanet state, `dt_rank * d_state * d_state`.
    pub fn state_layer(&mut self, layer: usize) -> &mut [f32] {
        let slot = self.slot_of(layer);
        let n = self.dt_rank * self.d_state * self.d_state;
        let at = slot * n;
        &mut self.state_all_mut()[at..at + n]
    }

    /// The wrapped region for the GDN decode kernels. Conv window of layer
    /// `i` sits at element offset `slot * (d_conv-1) * conv_channels`, its
    /// deltanet state at `conv_elems + slot * dt_rank * d_state * d_state`.
    pub fn gpu_view(&mut self) -> Option<&allpaka_backend::gpu::SharedRegion> {
        if self.shared.is_none() {
            self.shared = allpaka_backend::gpu::wrap_region(self.store.as_bytes());
        }
        self.shared.as_ref()
    }

    /// Element offset of a layer's deltanet state inside the region.
    pub fn state_off(&self, layer: usize) -> usize {
        let slot = self.slot_of(layer);
        self.conv_elems + slot * self.dt_rank * self.d_state * self.d_state
    }

    /// Element offset of a layer's conv window inside the region.
    pub fn conv_off(&self, layer: usize) -> usize {
        let slot = self.slot_of(layer);
        slot * (self.d_conv - 1) * self.conv_channels
    }
}

/// Per-position K and V for one transformer layer: `kv_dim` floats each.
/// Half precision costs nothing measurable in the logits - attention scores
/// are exponentiated and normalised, and llama.cpp caches f16 by default -
/// and halves the one tensor that is both written and fully re-read every
/// single step.
pub struct KvCache {
    store: Region,
    /// Maps model layer index -> compact KV slot index (`usize::MAX` for non-attention layers).
    slot_map: Vec<usize>,
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

/// The largest floor a decode attention slot may be given
/// (`ALLPAKA_FA_KV_FLOOR`); a slot is a power of two no smaller than this.
const FA_SLOT_FLOOR_MAX: usize = 8192;

/// Rows of zeroes kept after the last layer's values.
///
/// The CUDA decode attention reads K and V through a slot whose length is a
/// power of two at least as large as the context, not the context itself, and
/// masks the rows past the current position. For every layer but the last
/// those rows are the next layer's (finite) rows; for the last layer's values
/// they would lie past the end of the cache, where the memory is not ours and
/// a stray Inf or NaN turns a masked row's zero weight into NaN. The tail
/// makes every slot the cache can be asked for land inside it, on zeroes.
pub fn slot_tail_rows(capacity: usize) -> usize {
    capacity
        .next_power_of_two()
        .max(FA_SLOT_FLOOR_MAX)
        .saturating_sub(capacity)
}

impl KvCache {
    pub fn new(n_layers: usize, kv_dim: usize, capacity: usize) -> Self {
        let active = vec![true; n_layers];
        Self::with_active_layers(&active, kv_dim, capacity)
    }

    pub fn with_active_layers(active: &[bool], kv_dim: usize, capacity: usize) -> Self {
        let mut slot_map = Vec::with_capacity(active.len());
        let mut n_active = 0usize;
        for &is_active in active {
            if is_active {
                slot_map.push(n_active);
                n_active += 1;
            } else {
                slot_map.push(usize::MAX);
            }
        }
        let n_slots = n_active.max(1);
        let layer_stride = kv_dim * capacity;
        let half = n_slots * layer_stride;
        let store = Region::zeroed(half * 2 + slot_tail_rows(capacity) * kv_dim);
        let shared = allpaka_backend::gpu::wrap_region(store.as_bytes());
        Self {
            store,
            slot_map,
            v_base: half,
            layer_stride,
            kv_dim,
            len: 0,
            capacity,
            shared,
        }
    }

    #[inline]
    fn slot_of(&self, layer: usize) -> usize {
        let s = self.slot_map[layer];
        debug_assert_ne!(s, usize::MAX, "inactive KV layer {layer}");
        s
    }

    /// Returns `(k_off, v_off)` element offsets for `layer` (or `(0, 0)` if `layer` is inactive).
    #[inline]
    pub fn layer_offsets(&self, layer: usize) -> (usize, usize) {
        let s = self.slot_map[layer];
        if s == usize::MAX {
            (0, 0)
        } else {
            let k_off = s * self.layer_stride;
            (k_off, self.v_base + k_off)
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

    /// Fast-path view that never erases the reason GPU wrapping failed.
    pub fn gpu_view_checked(
        &mut self,
        layer: usize,
    ) -> Result<(&allpaka_backend::gpu::SharedRegion, usize, usize), GpuViewError> {
        if self.shared.is_none() {
            self.shared = allpaka_backend::gpu::wrap_region(self.store.as_bytes());
        }
        let shared = self.shared.as_ref().ok_or_else(|| {
            let reason = match allpaka_backend::gpu::memory_architecture() {
                allpaka_backend::memory::MemoryArchitecture::Uma => {
                    "UMA backend rejected the page-aligned shared region"
                }
                allpaka_backend::memory::MemoryArchitecture::Duma => {
                    "dUMA backend failed to allocate or upload the discrete device region"
                }
                allpaka_backend::memory::MemoryArchitecture::HostOnly => {
                    "no GPU device or model mapping is attached"
                }
            };
            GpuViewError {
                cache: "kv-cache",
                layer: Some(layer),
                reason,
            }
        })?;
        let (k_off, v_off) = self.layer_offsets(layer);
        Ok((shared, k_off, v_off))
    }

    /// [`gpu_view`] for callers that only hold `&self`: the offsets are pure
    /// arithmetic, and the region is already wrapped whenever a device
    /// existed at construction. Returns None only if it is not - the same
    /// "no device" answer, without a mutable borrow. CPU attention runs
    /// under `thread::scope` with `&self` shared across threads, and the
    /// whole-token path needs the region and the per-layer offsets at once.
    #[inline]
    pub fn gpu_view_ref(
        &self,
        layer: usize,
    ) -> Option<(&allpaka_backend::gpu::SharedRegion, usize, usize)> {
        let (k_off, v_off) = self.layer_offsets(layer);
        Some((self.shared.as_ref()?, k_off, v_off))
    }

    #[inline]
    pub fn layer_stride(&self) -> usize {
        self.layer_stride
    }

    #[inline]
    pub fn v_base(&self) -> usize {
        self.v_base
    }

    fn k_range(&self, layer: usize) -> std::ops::Range<usize> {
        let at = self.slot_of(layer) * self.layer_stride;
        at..at + self.layer_stride
    }

    fn v_range(&self, layer: usize) -> std::ops::Range<usize> {
        let at = self.v_base + self.slot_of(layer) * self.layer_stride;
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
    ///
    /// On **UMA** (Metal), the write lands directly in unified memory shared
    /// with the GPU. On **dUMA** (CUDA), if a discrete VRAM cache is attached,
    /// the written position is also uploaded (`H2D`) so subsequent GPU
    /// attention kernels (`attend_project`) observe the current step's K/V.
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert!(
            self.len < self.capacity,
            "kv cache full at {} positions",
            self.capacity
        );
        let pos = self.len;
        self.store_at(layer, pos, k, v);
        self.sync_gpu_pos_if_discrete(layer, pos);
    }

    #[inline]
    fn sync_gpu_pos_if_discrete(&mut self, layer: usize, pos: usize) {
        if self
            .shared
            .as_ref()
            .is_some_and(allpaka_backend::gpu::SharedRegion::is_discrete)
        {
            self.sync_gpu_range(layer, pos..pos + 1);
        }
    }

    /// Store at an explicit position at or past the sealed length.
    ///
    /// Batched prefill writes a whole chunk of positions per layer before any
    /// of them is sealed; `commit` then seals the chunk in one step.
    pub fn store_at(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert_eq!(v.len(), self.kv_dim);
        assert!(pos >= self.len, "position {pos} is already sealed");
        assert!(
            pos < self.capacity,
            "position {pos} past capacity {}",
            self.capacity
        );
        let at = pos * self.kv_dim;
        let (kr, vr) = (self.k_range(layer), self.v_range(layer));
        let kv_dim = self.kv_dim;
        let store = self.store.as_mut_slice();
        ops::f16::from_f32(k, &mut store[kr.start + at..kr.start + at + kv_dim]);
        ops::f16::from_f32(v, &mut store[vr.start + at..vr.start + at + kv_dim]);
    }

    /// Synchronize a range of host-written KV positions to the GPU cache.
    ///
    /// * On **UMA** (`SharedRegion::is_unified()`), host and GPU share physical
    ///   memory in place — this is a zero-copy no-op returning `true`.
    /// * On **dUMA** (`SharedRegion::is_discrete()`), uploads the K and V byte
    ///   slices for `positions` to device VRAM (`H2D`).
    pub fn sync_gpu_range(&mut self, layer: usize, positions: std::ops::Range<usize>) -> bool {
        if positions.start > positions.end || positions.end > self.capacity {
            return false;
        }
        let slot = self.slot_of(layer);
        let Some(shared) = self.shared.as_mut() else {
            return false;
        };
        if shared.is_unified() || positions.is_empty() {
            return true;
        }
        let count = (positions.end - positions.start) * self.kv_dim;
        let k_at = slot * self.layer_stride + positions.start * self.kv_dim;
        let v_at = self.v_base + k_at;
        let bytes = self.store.as_bytes();
        let k_bytes = k_at * 2;
        let v_bytes = v_at * 2;
        let byte_count = count * 2;
        allpaka_backend::gpu::upload_region_range(
            shared,
            k_bytes,
            &bytes[k_bytes..k_bytes + byte_count],
        ) && allpaka_backend::gpu::upload_region_range(
            shared,
            v_bytes,
            &bytes[v_bytes..v_bytes + byte_count],
        )
    }

    /// Store at an explicit position that may already be sealed. Only for
    /// the MTP draft layer's slot: the trunk's verify batch seals positions
    /// in the trunk layers, and the MTP block appends its own layer at the
    /// same positions afterwards. Truncated positions are rewritten, never
    /// read past `len`, so no clearing is needed.
    pub fn store_at_rewrite(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert_eq!(v.len(), self.kv_dim);
        assert!(
            pos < self.capacity,
            "position {pos} past capacity {}",
            self.capacity
        );
        let at = pos * self.kv_dim;
        let (kr, vr) = (self.k_range(layer), self.v_range(layer));
        let kv_dim = self.kv_dim;
        let store = self.store.as_mut_slice();
        ops::f16::from_f32(k, &mut store[kr.start + at..kr.start + at + kv_dim]);
        ops::f16::from_f32(v, &mut store[vr.start + at..vr.start + at + kv_dim]);
        self.sync_gpu_pos_if_discrete(layer, pos);
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
        assert!(
            keep <= self.len,
            "cannot truncate {} up to {keep}",
            self.len
        );
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
    /// Whatever slot the decode attention is given, the last layer's values
    /// still end inside the cache, on zeroes.
    #[test]
    fn every_attention_slot_ends_inside_the_cache() {
        let (layers, kv_dim) = (3usize, 8usize);
        for capacity in [1usize, 132, 300, 5000, 8192, 9000, 16472] {
            let cache = super::KvCache::new(layers, kv_dim, capacity);
            let (_, v_off) = cache.layer_offsets(layers - 1);
            // The slot for the last position, with the largest allowed floor.
            let slot = capacity.next_power_of_two().max(8192);
            let end = v_off + slot * kv_dim;
            let store = cache.store.as_slice();
            assert!(
                end <= store.len(),
                "capacity {capacity}: {end} > {}",
                store.len()
            );
            let tail = &store[v_off + capacity * kv_dim..end];
            assert!(tail.iter().all(|&bits| bits == 0), "capacity {capacity}");
            assert_eq!(super::slot_tail_rows(capacity), slot - capacity);
        }
    }

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

    #[test]
    fn hybrid_active_layers_compact_kv_and_ssm_storage() {
        let kv_mask = [false, false, false, true, false, false, false, true];
        let ssm_mask = [true, true, true, false, true, true, true, false];
        let kv_full = KvCache::new(8, 256, 64);
        let mut kv_compact = KvCache::with_active_layers(&kv_mask, 256, 64);
        // Two of eight layers hold KV: a quarter of the rows, plus the same
        // slot tail either way.
        let tail = slot_tail_rows(64) * 256;
        assert_eq!(
            (kv_compact.store.as_slice().len() - tail) * 4,
            kv_full.store.as_slice().len() - tail
        );
        assert_eq!(kv_compact.slot_of(3), 0);
        assert_eq!(kv_compact.slot_of(7), 1);
        assert_eq!(kv_compact.layer_offsets(0), (0, 0));
        assert_eq!(kv_compact.layer_offsets(3), (0, 2 * 256 * 64));
        assert_eq!(kv_compact.layer_offsets(7), (256 * 64, 3 * 256 * 64));

        kv_compact.store(3, &[1.0; 256], &[2.0; 256]);
        kv_compact.store(7, &[3.0; 256], &[4.0; 256]);
        kv_compact.advance();
        assert_eq!(ops::f16::to_f32(kv_compact.k_at(3, 0, 0, 4)[0]), 1.0);
        assert_eq!(ops::f16::to_f32(kv_compact.v_at(7, 0, 0, 4)[0]), 4.0);

        let ssm_full = SsmCache::new(8, 3, 2048, 16, 16);
        let mut ssm_compact = SsmCache::with_active_layers(&ssm_mask, 3, 2048, 16, 16);
        assert_eq!(
            ssm_compact.store.as_slice().len() * 4,
            ssm_full.store.as_slice().len() * 3
        );
        assert_eq!(ssm_compact.slot_of(0), 0);
        assert_eq!(ssm_compact.slot_of(4), 3);
        ssm_compact.conv_layer(4)[0] = 42.0;
        assert_eq!(ssm_compact.conv_layer(4)[0], 42.0);
    }

    #[test]
    fn ssm_snapshot_restore_and_slot_rollback_roundtrip() {
        let mut ssm = SsmCache::new(2, 3, 8, 2, 4);
        ssm.conv_layer(0)[0] = 1.5;
        ssm.state_layer(1)[3] = -2.25;
        let snap = ssm.snapshot();
        assert_eq!(snap.len(), ssm.total_elems());

        ssm.arm_slots(2);
        if let Some((cp, sp, dp, conv_off, state_off, n_conv, n_state, stride)) =
            ssm.slot_write_ptrs(0)
        {
            // SAFETY: `arm_slots(2)` sized `slot_store` for two whole-region copies,
            // so `stride + conv_off + n_conv` and `state_off + n_state` are within it, and
            // the sources are inside `store`; no reference to either region is live here.
            unsafe {
                std::ptr::copy_nonoverlapping(cp, dp.add(conv_off), n_conv);
                std::ptr::copy_nonoverlapping(sp, dp.add(state_off), n_state);
                // Slot 1 gets mutated values
                std::ptr::copy_nonoverlapping(cp, dp.add(stride + conv_off), n_conv);
                *dp.add(stride + conv_off) = 99.0;
            }
        }
        ssm.conv_layer(0)[0] = 777.0;
        ssm.restore_slot(0);
        assert_eq!(ssm.conv_layer(0)[0], 1.5);

        ssm.conv_layer(0)[0] = 0.0;
        ssm.state_layer(1)[3] = 0.0;
        ssm.restore(&snap);
        assert_eq!(ssm.conv_layer(0)[0], 1.5);
        assert_eq!(ssm.state_layer(1)[3], -2.25);
    }

    /// Empty caches used to request a zero-sized allocation, which is
    /// undefined behaviour for the global allocator; they now get a page.
    #[test]
    fn empty_caches_allocate_a_valid_region() {
        let kv = KvCache::new(2, 8, 0);
        assert!(!kv.store.as_bytes().is_empty());
        let kv = KvCache::new(0, 0, 16);
        assert!(!kv.store.as_bytes().is_empty());
        let mut ssm = SsmCache::new(1, 2, 4, 1, 2);
        ssm.arm_slots(0);
        assert!(ssm
            .slot_store
            .as_ref()
            .is_some_and(|s| !s.as_bytes().is_empty()));
    }

    /// Rollback slots are handed out as `&[f32]` before any row writes them,
    /// so the buffer must start initialised (it used to be `alloc`ed raw).
    #[test]
    fn armed_slots_start_zeroed() {
        let mut ssm = SsmCache::new(2, 3, 8, 2, 4);
        ssm.arm_slots(3);
        for i in 0..3 {
            assert!(ssm.slot(i).iter().all(|&v| v == 0.0));
        }
    }
}
