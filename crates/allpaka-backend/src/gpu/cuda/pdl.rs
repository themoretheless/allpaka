//! Programmatic Dependent Launch (PDL) for Hopper+/Blackwell.
//!
//! Host: `cuLaunchKernelEx` + `PROGRAMMATIC_STREAM_SERIALIZATION` on the
//! *dependent* kernel: it may start as soon as every block of the previous
//! kernel in the stream has executed `griddepcontrol.launch_dependents` (or
//! exited). Device: the dependent kernel must execute `griddepcontrol.wait`
//! before touching anything the previous kernel reads or writes.
//!
//! Nothing here depends on cudarc's private layouts: kernel modules are
//! loaded through the public driver API ([`RawModule`]), which hands out the
//! `CUfunction`, and the kernel argument vector is collected by
//! [`PdlLaunch`] itself. Every kernel launch of the backend goes through it.

use cudarc::driver::result::{self, DriverError};
use cudarc::driver::sys::{self, CUresult};
use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, LaunchConfig};
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Launches that carried the PDL attribute (process-wide; includes launches
/// recorded into CUDA graphs, which replay without passing through here).
static PDL_LAUNCHES: AtomicU64 = AtomicU64::new(0);
/// Launches through [`PdlLaunch`] without the attribute (PDL off or refused).
static PLAIN_LAUNCHES: AtomicU64 = AtomicU64::new(0);
/// Set when the driver refuses the PDL attribute; launches stay plain after.
static PDL_REFUSED: AtomicBool = AtomicBool::new(false);
static ANNOUNCED: AtomicBool = AtomicBool::new(false);

/// `ALLPAKA_CUDA_PDL_DEBUG`: per-kernel `(pdl, plain)` launch counts.
static DEBUG_KERNELS: std::sync::Mutex<std::collections::BTreeMap<&'static str, (u64, u64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// `(pdl, plain)` launch counts issued through [`PdlLaunch`] so far.
pub fn pdl_launch_counts() -> (u64, u64) {
    (
        PDL_LAUNCHES.load(Ordering::Relaxed),
        PLAIN_LAUNCHES.load(Ordering::Relaxed),
    )
}

/// True while launches carry the PDL attribute (flag on, driver accepted it).
pub fn pdl_active() -> bool {
    crate::runtime::get().cuda_pdl && !PDL_REFUSED.load(Ordering::Relaxed)
}

/// `(programmatic, total)` dependency edges of a captured graph. Programmatic
/// edges are what PDL launches turn into under stream capture.
pub fn graph_edge_counts(g: &cudarc::driver::CudaGraph) -> Option<(usize, usize)> {
    let h = g.cu_graph();
    let mut n = 0usize;
    // SAFETY: null arrays with a count pointer query the edge count.
    let r = unsafe {
        sys::cuGraphGetEdges_v2(
            h,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut n,
        )
    };
    if r != CUresult::CUDA_SUCCESS {
        return None;
    }
    if n == 0 {
        return Some((0, 0));
    }
    let mut from = vec![std::ptr::null_mut(); n];
    let mut to = vec![std::ptr::null_mut(); n];
    // SAFETY: plain-old-data out array, fully written by the driver below.
    let mut data = vec![unsafe { std::mem::zeroed::<sys::CUgraphEdgeData>() }; n];
    // SAFETY: the three arrays hold `n` elements, as reported above.
    let r = unsafe {
        sys::cuGraphGetEdges_v2(
            h,
            from.as_mut_ptr(),
            to.as_mut_ptr(),
            data.as_mut_ptr(),
            &mut n,
        )
    };
    if r != CUresult::CUDA_SUCCESS {
        return None;
    }
    let prog = sys::CUgraphDependencyType_enum::CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC as u8;
    let p = data.iter().take(n).filter(|e| e.type_ == prog).count();
    Some((p, n))
}

/// `ALLPAKA_CUDA_PDL_DEBUG`: report a captured graph's programmatic edges and
/// the running launch counters.
pub fn debug_report_graph(label: &str, g: &cudarc::driver::CudaGraph) {
    if !crate::runtime::get().cuda_pdl_debug {
        return;
    }
    let (pdl, plain) = pdl_launch_counts();
    match graph_edge_counts(g) {
        Some((p, n)) => eprintln!(
            "cuda: pdl-debug: {label}: {p}/{n} programmatic edges; launches so far pdl={pdl} plain={plain} active={}",
            pdl_active()
        ),
        None => eprintln!("cuda: pdl-debug: {label}: edge query failed"),
    }
    // Per-kernel launches since the previous report (i.e. this capture).
    let mut m = DEBUG_KERNELS.lock().unwrap_or_else(|e| e.into_inner());
    for (name, (p, q)) in m.iter() {
        eprintln!("cuda: pdl-debug:   {name}: pdl={p} plain={q}");
    }
    m.clear();
}

/// A kernel module loaded through the raw driver API. Never unloaded: it
/// lives as long as the device.
pub struct RawModule {
    module: sys::CUmodule,
    /// Entry points whose PTX body contains `griddepcontrol.wait`.
    waits: std::collections::HashSet<String>,
}

/// Names of the `.entry` kernels in `ptx` that execute `griddepcontrol.wait`.
/// Only these may be launched as programmatic dependents: a kernel without
/// the wait could start reading while the previous kernel is still writing.
fn entries_with_wait(ptx: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let mut cur: Option<&str> = None;
    for line in ptx.lines() {
        let t = line.trim_start();
        if let Some(i) = t.find(".entry ") {
            if t.starts_with(".entry") || t.starts_with(".visible") || t.starts_with(".weak") {
                let rest = t[i + 7..].trim_start();
                let end = rest
                    .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$'))
                    .unwrap_or(rest.len());
                cur = Some(&rest[..end]);
                continue;
            }
        }
        if t.starts_with(".func") || t.starts_with(".visible .func") || t.starts_with(".weak .func")
        {
            cur = None;
        } else if t.contains("griddepcontrol.wait") {
            if let Some(name) = cur {
                out.insert(name.to_string());
            }
        }
    }
    out
}

// SAFETY: a `CUmodule` is an opaque driver handle, valid from any thread that
// has the owning context current (same contract as cudarc's `CudaModule`).
unsafe impl Send for RawModule {}
// SAFETY: see `Send`; the handle is only read.
unsafe impl Sync for RawModule {}

impl RawModule {
    /// Loads PTX source text. The context must be current on this thread.
    pub fn load_ptx(src: &str) -> Result<Self, DriverError> {
        let c = CString::new(src).map_err(|_| DriverError(CUresult::CUDA_ERROR_INVALID_SOURCE))?;
        // SAFETY: `c` is a NUL-terminated PTX image that outlives the call.
        let m = unsafe { result::module::load_data(c.as_ptr() as *const c_void) }?;
        Ok(Self {
            module: m,
            waits: entries_with_wait(src),
        })
    }

    /// Kernel `name` of this module.
    pub fn function(&self, name: &'static str) -> Option<KernelFn> {
        let c = CString::new(name).ok()?;
        // SAFETY: `self.module` is a live module (never unloaded).
        let raw = unsafe { result::module::get_function(self.module, c) }.ok()?;
        Some(KernelFn {
            raw,
            waits: self.waits.contains(name),
            name,
        })
    }
}

/// A kernel of a [`RawModule`].
#[derive(Clone, Copy)]
pub struct KernelFn {
    raw: sys::CUfunction,
    /// The kernel executes `griddepcontrol.wait`, so it may be a programmatic
    /// dependent. Kernels without it always get a plain launch.
    waits: bool,
    name: &'static str,
}

// SAFETY: a `CUfunction` is an opaque driver handle, usable from any thread
// that has the owning context current (cudarc's `CudaFunction` is the same).
unsafe impl Send for KernelFn {}
// SAFETY: see `Send`; the handle is only read.
unsafe impl Sync for KernelFn {}

impl KernelFn {
    /// `cuFuncSetAttribute`; the context must be current on this thread.
    pub fn set_attribute(
        &self,
        attribute: sys::CUfunction_attribute,
        value: i32,
    ) -> Result<(), DriverError> {
        // SAFETY: `self.raw` belongs to a module that is never unloaded.
        unsafe { result::function::set_function_attribute(self.raw, attribute, value) }
    }
}

const MAX_ARGS: usize = 32;

#[derive(Clone, Copy)]
enum Slot {
    /// Pointer to a caller-owned scalar that outlives the builder.
    Ref(*mut c_void),
    /// Device pointer stored inline; its address is taken at launch.
    Dev(sys::CUdeviceptr),
}

/// Kernel launch builder that owns its argument vector.
pub struct PdlLaunch<'a> {
    stream: &'a CudaStream,
    func: sys::CUfunction,
    waits: bool,
    name: &'static str,
    slots: [Slot; MAX_ARGS],
    n: usize,
}

/// Something [`PdlLaunch::arg`] accepts.
pub trait PdlArg<'a> {
    fn push(self, launch: &mut PdlLaunch<'a>);
}

macro_rules! scalar_arg {
    ($($t:ty),*) => {$(
        impl<'a, 'b: 'a> PdlArg<'a> for &'b $t {
            #[inline(always)]
            fn push(self, launch: &mut PdlLaunch<'a>) {
                launch.push_slot(Slot::Ref(self as *const $t as *mut c_void));
            }
        }
    )*};
}
scalar_arg!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

macro_rules! device_arg {
    ($($t:ty),*) => {$(
        impl<'a, 'b: 'a, 'c: 'b, T> PdlArg<'a> for $t {
            #[inline(always)]
            fn push(self, launch: &mut PdlLaunch<'a>) {
                // Event tracking is disabled on this context
                // (`disable_event_tracking` in `init_device`), so the guard
                // records nothing; the pointer value is all we need.
                let (p, guard) = DevicePtr::device_ptr(&*self, launch.stream);
                drop(guard);
                launch.push_slot(Slot::Dev(p));
            }
        }
    )*};
}
device_arg!(
    &'b CudaSlice<T>,
    &'b mut CudaSlice<T>,
    &'b CudaView<'c, T>,
    &'b mut CudaView<'c, T>,
    &'b CudaViewMut<'c, T>,
    &'b mut CudaViewMut<'c, T>
);

impl<'a> PdlLaunch<'a> {
    #[inline(always)]
    fn push_slot(&mut self, s: Slot) {
        assert!(
            self.n < MAX_ARGS,
            "cuda: kernel has more than {MAX_ARGS} args"
        );
        self.slots[self.n] = s;
        self.n += 1;
    }

    #[inline(always)]
    pub fn arg<A: PdlArg<'a>>(&mut self, a: A) -> &mut Self {
        a.push(self);
        self
    }

    /// Submits the kernel; with PDL on it carries the
    /// programmatic-stream-serialization attribute.
    ///
    /// # Safety
    /// Same contract as cudarc's `LaunchArgs::launch`: argument types, order
    /// and the memory the kernel touches are the caller's responsibility.
    /// In addition the kernel must execute `griddepcontrol.wait` before it
    /// reads or writes anything the previous kernel in the stream touches.
    pub unsafe fn launch_pdl(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_ARGS];
        for (p, s) in ptrs.iter_mut().zip(self.slots[..self.n].iter_mut()) {
            *p = match s {
                Slot::Ref(r) => *r,
                Slot::Dev(d) => d as *mut sys::CUdeviceptr as *mut c_void,
            };
        }
        let args = &mut ptrs[..self.n];
        if crate::runtime::get().cuda_pdl_debug {
            let mut m = DEBUG_KERNELS.lock().unwrap_or_else(|e| e.into_inner());
            let e = m.entry(self.name).or_insert((0, 0));
            if self.waits && pdl_active() {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
        if self.waits && pdl_active() {
            let mut attr = std::mem::zeroed::<sys::CUlaunchAttribute>();
            attr.id =
                sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attr.value.programmaticStreamSerializationAllowed = 1;
            let config = sys::CUlaunchConfig {
                gridDimX: cfg.grid_dim.0,
                gridDimY: cfg.grid_dim.1,
                gridDimZ: cfg.grid_dim.2,
                blockDimX: cfg.block_dim.0,
                blockDimY: cfg.block_dim.1,
                blockDimZ: cfg.block_dim.2,
                sharedMemBytes: cfg.shared_mem_bytes,
                hStream: self.stream.cu_stream(),
                attrs: &mut attr,
                numAttrs: 1,
            };
            let r =
                sys::cuLaunchKernelEx(&config, self.func, args.as_mut_ptr(), std::ptr::null_mut());
            if r == CUresult::CUDA_SUCCESS {
                let n = PDL_LAUNCHES.fetch_add(1, Ordering::Relaxed);
                if n == 0 && !ANNOUNCED.swap(true, Ordering::Relaxed) {
                    eprintln!("cuda: PDL active (first programmatic launch accepted; ALLPAKA_CUDA_PDL=0 to disable)");
                }
                return Ok(());
            }
            // Not supported by this driver/device: stay plain from now on.
            if !PDL_REFUSED.swap(true, Ordering::Relaxed) {
                eprintln!("cuda: PDL launch attribute refused ({r:?}); using plain launches");
            }
        }
        PLAIN_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        result::launch_kernel(
            self.func,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream(),
            args,
        )
    }
}

pub trait StreamPdlExt {
    fn launch_builder_pdl<'a>(&'a self, func: &KernelFn) -> PdlLaunch<'a>;
}

impl StreamPdlExt for CudaStream {
    #[inline(always)]
    fn launch_builder_pdl<'a>(&'a self, func: &KernelFn) -> PdlLaunch<'a> {
        PdlLaunch {
            stream: self,
            func: func.raw,
            waits: func.waits,
            name: func.name,
            slots: [Slot::Dev(0); MAX_ARGS],
            n: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::entries_with_wait;

    #[test]
    fn finds_entries_that_wait() {
        let ptx = ".visible .entry a(
.param .u64 a_param_0
)
{
	griddepcontrol.wait;
	ret;
}
.visible .entry b_2(
)
{
	griddepcontrol.launch_dependents;
	ret;
}
.func helper(
)
{
	griddepcontrol.wait;
}
.entry c(
)
{
	griddepcontrol.wait;
}
";
        let w = entries_with_wait(ptx);
        assert!(w.contains("a") && w.contains("c"));
        assert!(!w.contains("b_2") && !w.contains("helper"));
        assert_eq!(w.len(), 2);
    }
}
